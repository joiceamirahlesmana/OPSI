# client_app.py
import os
from typing import Tuple, Union, List
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torch import nn
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Dataset (multi-view: abdomen, body, head_thorax)
# ----------------------------
class MosquitoDataset(Dataset):
    def __init__(self, dataset_path: str, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.classes = ["larva_ae_aegypti", "larva_ae_albopictus", "larva_an", "larva_cx", "nyamuk_ae_aegypti_betina", "nyamuk_ae_aegypti_jantan", "nyamuk_ae_albopictus_betina", "nyamuk_ae_albopictus_jantan", "nyamuk_an", "nyamuk_cx"]
        self.image_pairs = []  # list of (label, filename)

        for label in self.classes:
            a_path = os.path.join(dataset_path, label, "abdomen")
            b_path = os.path.join(dataset_path, label, "body")
            h_path = os.path.join(dataset_path, label, "head_thorax")
            if not (os.path.isdir(a_path) and os.path.isdir(b_path) and os.path.isdir(h_path)):
                # skip if any view missing
                continue
            common = set(os.listdir(a_path)) & set(os.listdir(b_path)) & set(os.listdir(h_path))
            for img in sorted(common):
                self.image_pairs.append((label, img))

    def __len__(self) -> int:
        return len(self.image_pairs)

    def _open(self, *parts) -> Image.Image:
        p = os.path.join(*parts)
        img = Image.open(p).convert("RGB")
        return img

    def __getitem__(self, idx: int):
        label, img_name = self.image_pairs[idx]
        a = self._open(self.dataset_path, label, "abdomen", img_name)
        b = self._open(self.dataset_path, label, "body", img_name)
        h = self._open(self.dataset_path, label, "head_thorax", img_name)

        if self.transform:
            a = self.transform(a)
            b = self.transform(b)
            h = self.transform(h)

        y = self.classes.index(label)
        return a, b, h, y


# ----------------------------
# DataLoaders
# ----------------------------
def _make_transform(img_size: Union[int, Tuple[int, int]]):
    if isinstance(img_size, int):
        size = (img_size, img_size)
    else:
        size = img_size
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_data_loaders(
    dataset_path: str,
    batch_size: int = 8,
    img_size: Union[int, Tuple[int, int]] = 224,
    num_workers: int = 2,
):
    transform = _make_transform(img_size)
    dataset = MosquitoDataset(dataset_path, transform)

    n = len(dataset)
    if n == 0:
        raise RuntimeError(f"No images found under {dataset_path}")

    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    common_loader_args = dict(
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **common_loader_args)
    val_loader   = DataLoader(val_ds, shuffle=False, **common_loader_args)
    test_loader  = DataLoader(test_ds, shuffle=False, **common_loader_args)

    return train_loader, val_loader, test_loader, dataset.classes


class MobileFusionNet(nn.Module):   
    def __init__(self, base, embed_dim, out_dim=10, margin=1.0):
        super().__init__()
        self.base = base
        self.embed_dim = embed_dim
        self.margin = margin

        # === SAN Block ===
        self.fc_abdomen = nn.Linear(embed_dim*2, embed_dim)
        self.fc_body = nn.Linear(embed_dim*2, embed_dim)
        self.fc_head = nn.Linear(embed_dim*2, embed_dim)

        # Key Network (K): CNN2 with 3 conv layers + FC
        self.key_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.key_fc = nn.Linear(embed_dim * 32, embed_dim)
        self.softmax = nn.Softmax(dim=0)  # across views (a, b, h)

        self.output_fc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )

    def pool_features(self, feat):
        max_pool = F.adaptive_max_pool1d(feat.unsqueeze(1), output_size=self.embed_dim).squeeze(1)
        avg_pool = F.adaptive_avg_pool1d(feat.unsqueeze(1), output_size=self.embed_dim).squeeze(1)
        return torch.cat((max_pool, avg_pool), dim=1)

    def get_key_weights(self, x): 
        # CNN + FC to extract key weights 
        x = x.unsqueeze(1) # shape: (batch, 1, embed_dim) 
        cnn_out = self.key_cnn(x) # shape: (batch, 32, embed_dim) 
        cnn_out_flat = cnn_out.view(cnn_out.size(0), -1) # flatten to (batch, 32*embed_dim) 
        key_vec = self.key_fc(cnn_out_flat) # (batch, embed_dim) 
        return key_vec

    def forward(self, a, b, h):
        a_feat = self.base(a)
        b_feat = self.base(b) 
        h_feat = self.base(h)
        
        a_pool = self.pool_features(a_feat) 
        b_pool = self.pool_features(b_feat) 
        h_pool = self.pool_features(h_feat)

        a_feat = self.fc_abdomen(a_pool)
        b_feat = self.fc_body(b_pool)
        h_feat = self.fc_head(h_pool)

        # === Key Network: CNN2 + FC === 
        a_key = self.get_key_weights(a_feat) 
        b_key = self.get_key_weights(b_feat) 
        h_key = self.get_key_weights(h_feat)

        # === Softmax Attention === 
        keys = torch.stack([a_key, b_key, h_key], dim=0) 
        attn_weights = self.softmax(keys) 
        values = torch.stack([a_feat, b_feat, h_feat], dim=0) 
        pooled = (attn_weights * values).sum(dim=0)

        output = self.output_fc(pooled)
        return output, pooled

    # Triplet loss with batch-wise mining
    def triplet_loss(self, features, labels, epoch=None):
        """
        Adaptive triplet loss:
        - Epoch <= 8: gunakan semi-hard saja.
        - Epoch > 8 : gunakan kombinasi semi-hard dan sebagian kecil hard samples.
        """
        device = features.device
        batch_size = features.size(0)

        # 1️⃣ Hitung semua jarak antar sampel (anchor vs lainnya)
        dist_matrix = torch.cdist(features, features, p=2)

        # 2️⃣ Mask pasangan sekelas dan beda kelas
        labels = labels.unsqueeze(1)
        label_mask = (labels == labels.t()).float().to(device)
        diag_mask = 1 - torch.eye(batch_size, device=device)

        # 3️⃣ Positive distance (hardest positive)
        pos_dist = (dist_matrix * label_mask * diag_mask).max(dim=1)[0]

        # 4️⃣ Negative mining
        neg_dist = dist_matrix + 1e5 * label_mask  # abaikan pasangan sekelas

        # --- Semi-hard (lebih jauh dari positive, tapi < margin)
        semi_mask = (neg_dist > pos_dist.unsqueeze(1)) & (neg_dist < pos_dist.unsqueeze(1) + self.margin)
        semi_hard_neg = torch.where(semi_mask, neg_dist, torch.tensor(float('inf'), device=device))
        semi_hard_neg = semi_hard_neg.min(dim=1)[0]

        # --- Hard (negatif terdekat)
        hard_neg = (dist_matrix + 1e5 * label_mask).min(dim=1)[0]

        # 5️⃣ Pilih sesuai epoch
        if epoch is None or epoch <= 8:
            # Hanya semi-hard
            final_neg = torch.where(torch.isinf(semi_hard_neg), hard_neg, semi_hard_neg)
        else:
            # Gabungkan semi-hard dengan sebagian kecil hard negatives
            hard_weight = 0.2  # gunakan 20% hard samples
            # Mask: gunakan hard_neg untuk sebagian indeks secara acak
            mix_mask = (torch.rand(batch_size, device=device) < hard_weight)
            final_neg = semi_hard_neg.clone()
            final_neg[mix_mask] = hard_neg[mix_mask]
            # Jika semi_hard tidak ada (inf), ganti dengan hard
            final_neg = torch.where(torch.isinf(final_neg), hard_neg, final_neg)

        # 6️⃣ Triplet loss = ReLU(pos - neg + margin)
        triplet_loss = F.relu(pos_dist - final_neg + self.margin)

        # 7️⃣ Buang easy samples (loss = 0)
        non_zero_loss = triplet_loss[triplet_loss > 1e-6]
        if len(non_zero_loss) == 0:
            return torch.tensor(0.0, device=device)
        return non_zero_loss.mean()
        
def load_model(pretrained: bool = True, out_dim: int = 10, freeze_backbone: bool = False) -> nn.Module:
    base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
    # MobilenetV2 outputs a pooled 1280-dim vector at classifier[1].
    base.classifier = nn.Identity()
    model = MobileFusionNet(mobilenet, embed_dim, out_dim=len(class_names), margin=MARGIN).to(DEVICE)

    if freeze_backbone:
        for p in model.base.parameters():
            p.requires_grad = False

    return model
