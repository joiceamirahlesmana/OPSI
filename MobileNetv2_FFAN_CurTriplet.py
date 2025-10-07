# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 09:28:59 2025

@author: Joice Amirah Lesmana
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc  # Added import
from sklearn.preprocessing import label_binarize
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
MARGIN = 1.0

dataset_path = "dataset_opsi"
prefix_file_output = "MobileNetv2_FFAN_Triplet_"
path_dir_output = "output_model"
os.makedirs(path_dir_output, exist_ok=True)

class_names = [
    'larva_ae_aegypti', 'larva_ae_albopictus', 'larva_an', 'larva_cx',
    'nyamuk_ae_aegypti_betina', 'nyamuk_ae_aegypti_jantan',
    'nyamuk_ae_albopictus_betina', 'nyamuk_ae_albopictus_jantan',
    'nyamuk_an', 'nyamuk_cx'
]

# ================= Dataset =================
class MosquitoDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_paths = []

        for label in class_names:
            a_path = f'{dataset_path}/{label}/abdomen'
            b_path = f'{dataset_path}/{label}/body'
            h_path = f'{dataset_path}/{label}/head_thorax'
            common = set(os.listdir(a_path)) & set(os.listdir(b_path)) & set(os.listdir(h_path))
            for img in common:
                self.image_paths.append((img, label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name, label = self.image_paths[idx]
        a = Image.open(f'{self.dataset_path}/{label}/abdomen/{img_name}')
        b = Image.open(f'{self.dataset_path}/{label}/body/{img_name}')
        h = Image.open(f'{self.dataset_path}/{label}/head_thorax/{img_name}')

        if self.transform:
            a = self.transform(a)
            b = self.transform(b)
            h = self.transform(h)

        label_idx = class_names.index(label)
        return a, b, h, label_idx

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = MosquitoDataset(dataset_path, transform)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ================= Model =================
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.classifier = nn.Identity()

total_layers = len(list(mobilenet.parameters()))
for idx, param in enumerate(mobilenet.parameters()):
    param.requires_grad = idx >= total_layers * 3 // 4

mobilenet.to(DEVICE)
embed_dim = 1280

# Fusion Feature Attention Network dan Curriculum Triplet Loss
class MobileFusionNetWithTriplet(nn.Module):   
    def __init__(self, base, embed_dim, out_dim=10, margin=1.0):
        super().__init__()
        self.base = base
        self.embed_dim = embed_dim
        self.margin = margin

        self.fc_abdomen = nn.Linear(embed_dim*2, embed_dim)
        self.fc_body = nn.Linear(embed_dim*2, embed_dim)
        self.fc_head = nn.Linear(embed_dim*2, embed_dim)

        self.key_cnn = nn.Sequential(
            nn.Conv1d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv1d(8, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU()
        )

        self.key_fc = nn.Linear(embed_dim * 32, embed_dim)
        self.softmax = nn.Softmax(dim=0)

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
        x = x.unsqueeze(1)
        cnn_out = self.key_cnn(x)
        cnn_out_flat = cnn_out.view(cnn_out.size(0), -1)
        key_vec = self.key_fc(cnn_out_flat)
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

        a_key = self.get_key_weights(a_feat)
        b_key = self.get_key_weights(b_feat)
        h_key = self.get_key_weights(h_feat)

        keys = torch.stack([a_key, b_key, h_key], dim=0)
        attn_weights = self.softmax(keys)
        values = torch.stack([a_feat, b_feat, h_feat], dim=0)
        pooled = (attn_weights * values).sum(dim=0)

        output = self.output_fc(pooled)
        return output, pooled

    def triplet_loss(self, features, labels, epoch=None):
        device = features.device
        batch_size = features.size(0)
        dist_matrix = torch.cdist(features, features, p=2)
        labels = labels.unsqueeze(1)
        label_mask = (labels == labels.t()).float().to(device)
        diag_mask = 1 - torch.eye(batch_size, device=device)
        pos_dist = (dist_matrix * label_mask * diag_mask).max(dim=1)[0]
        neg_dist = dist_matrix + 1e5 * label_mask
        semi_mask = (neg_dist > pos_dist.unsqueeze(1)) & (neg_dist < pos_dist.unsqueeze(1) + self.margin)
        semi_hard_neg = torch.where(semi_mask, neg_dist, torch.tensor(float('inf'), device=device)).min(dim=1)[0]
        hard_neg = (dist_matrix + 1e5 * label_mask).min(dim=1)[0]

        if epoch is None or epoch <= 8:
            final_neg = torch.where(torch.isinf(semi_hard_neg), hard_neg, semi_hard_neg)
        else:
            hard_weight = 0.2
            mix_mask = (torch.rand(batch_size, device=device) < hard_weight)
            final_neg = semi_hard_neg.clone()
            final_neg[mix_mask] = hard_neg[mix_mask]
            final_neg = torch.where(torch.isinf(final_neg), hard_neg, final_neg)

        triplet_loss = F.relu(pos_dist - final_neg + self.margin)
        return triplet_loss

def visualize_tsne(features, labels, losses, epoch, save_path):
    features = features.detach().cpu().numpy() if torch.is_tensor(features) else np.array(features)
    labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
    losses = losses.detach().cpu().numpy() if torch.is_tensor(losses) else np.array(losses)

    # === Pastikan panjang sama ===
    n = len(features)
    if len(losses) != n:
        print(f"[WARN] Adjusting loss length: losses={len(losses)}, features={n}")
        # Jika terlalu sedikit, ulangi atau potong supaya sama
        if len(losses) < n:
            losses = np.pad(losses, (0, n - len(losses)), mode='edge')
        else:
            losses = losses[:n]

    # === Mask untuk kategori triplet ===
    easy_mask = losses < 0.2
    semi_mask = (losses >= 0.2) & (losses < 0.6)
    hard_mask = losses >= 0.6

    # === t-SNE ===
    try:
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)

    reduced = tsne.fit_transform(features)

    # === Plot ===
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[easy_mask, 0], reduced[easy_mask, 1],
                c='green', label='Easy Triplet', alpha=0.6, s=15)
    plt.scatter(reduced[semi_mask, 0], reduced[semi_mask, 1],
                c='orange', label='Semi-Hard Triplet', alpha=0.6, s=15)
    plt.scatter(reduced[hard_mask, 0], reduced[hard_mask, 1],
                c='red', label='Hard Triplet', alpha=0.6, s=15)
    plt.title(f"Federated Deep Learning Feature Embedding (Epoch {epoch})")
    plt.legend()
    #plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def train_with_tsne(model, train_loader, val_loader, optimizer, scheduler, criterion, device, epochs=20):
    for epoch in range(epochs):
        print(f"\n[INFO] Epoch {epoch+1}")
        model.train()
        all_features, all_labels, all_losses = [], [], []

        for a, b, h, y in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            a, b, h, y = a.to(device), b.to(device), h.to(device), y.to(device)
            optimizer.zero_grad()

            out, pooled = model(a, b, h)
            class_loss = criterion(out, y)
            losses = model.triplet_loss(pooled, y, epoch=epoch+1)

            if isinstance(losses, torch.Tensor):
                triplet_loss = losses.mean()
            else:
                triplet_loss = torch.tensor(losses, device=device)

            total_loss = class_loss + triplet_loss
            total_loss.backward()
            optimizer.step()

            all_features.append(pooled)
            all_labels.append(y)
            all_losses.append(triplet_loss)

        all_features = torch.cat(all_features)
        all_labels = torch.cat(all_labels)
        all_losses = torch.stack(all_losses)

        # Visualisasi hanya pada epoch tertentu
        if epoch + 1 in [1, 5, 8, 10, 11, 13, 15, 17, epochs]:
            save_path = os.path.join(path_dir_output, f"tsne_epoch_{epoch+1}.png")
            visualize_tsne(all_features, all_labels, all_losses, epoch+1, save_path)
            print(f"✅ t-SNE visualization saved: {save_path}")

        scheduler.step()

# ================= Run Training =================
model = MobileFusionNetWithTriplet(mobilenet, embed_dim, out_dim=len(class_names), margin=MARGIN).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.4)

history = train_with_tsne(model, train_loader, val_loader, optimizer, scheduler, criterion, DEVICE, epochs=NUM_EPOCHS)

# === Plot Accuracy ===
plt.figure(figsize=(8, 6))
plt.plot(history['epoch'], history['train_acc'], label='Train Accuracy', marker='o')
plt.plot(history['epoch'], history['val_acc'], label='Validation Accuracy', marker='^')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(np.arange(1, len(history['epoch']) + 1, step=1))
plt.title('Train and Validation Accuracy vs Epoch')
plt.grid(True)
plt.legend(loc='lower right')
file_path = os.path.join(path_dir_output, prefix_file_output + "accuracy.png")
plt.savefig(file_path)
plt.show()

# === Plot All Losses ===
plt.figure(figsize=(8, 6))
plt.plot(history['epoch'], history['train_loss'], label='Train Loss', marker='o', color='blue')
plt.plot(history['epoch'], history['val_loss'], label='Validation Loss', marker='^', color='orange')
plt.plot(history['epoch'], history['triplet_loss'], label='Triplet Loss', marker='s', color='green')
plt.plot(history['epoch'], history['total_loss'], label='Total Loss', marker='d', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(np.arange(1, len(history['epoch']) + 1, step=1))
plt.title('Train, Validation, Triplet, and Total Loss vs Epoch')
plt.grid(True)
plt.legend(loc='upper right')
file_path = os.path.join(path_dir_output, prefix_file_output + "all_losses.png")
plt.savefig(file_path)
plt.show()

def evaluate(model, test_loader, class_labels, device, criterion):
    model.eval()  # Set model to evaluation mode
    all_preds, all_labels = [], []
    all_probabilities = []
    test_loss = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for a, b, h, y in test_loader:  # Inputs: abdomen, body, head
            a, b, h, y = a.to(device), b.to(device), h.to(device), y.to(device)
            out, _ = model(a, b, h)  # Model output
            loss = criterion(out, y)  # Cross-entropy loss
            test_loss += loss.item()

            preds = out.argmax(1)  # Predicted class index
            probabilities = torch.softmax(out, dim=1)  # Softmax to get probabilities

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)  # Confusion Matrix

    def calculate_metrics(cm):
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (TP + FP + FN)
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)
        specificity = TN / (TN + FP + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        return {
            'Accuracy': TP.sum() / cm.sum(),
            'Precision (Macro)': precision.mean(),
            'Recall/Sensitivity (Macro)': recall.mean(),
            'Specificity (Macro)': specificity.mean(),
            'F1-Score (Macro)': f1.mean()
        }

    metrics = calculate_metrics(cm)

    print("\n=== Test Metrics ===")
    print(f"Test Loss                      : {test_loss / len(test_loader):.4f}")
    print(f"Test Accuracy                  : {metrics['Accuracy']:.4f}")
    print(f"Precision (Macro)              : {metrics['Precision (Macro)']:.4f}")
    print(f"Sensitivity/Recall (Macro)     : {metrics['Recall/Sensitivity (Macro)']:.4f}")
    print(f"Specificity (Macro)            : {metrics['Specificity (Macro)']:.4f}")
    print(f"F1-Score (Macro)               : {metrics['F1-Score (Macro)']:.4f}")
    print(f"TP: {np.diag(cm)}")
    print(f"FP: {np.sum(cm, axis=0) - np.diag(cm)}")
    print(f"FN: {np.sum(cm, axis=1) - np.diag(cm)}")
    print(f"TN: {np.sum(cm) - (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))}")
    print(f"Number of Samples: {len(all_labels)}")

    # Confusion Matrix Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    os.makedirs(path_dir_output, exist_ok=True)
    file_path = os.path.join(path_dir_output, prefix_file_output + "confusion_matrix.png")
    plt.savefig(file_path, dpi=300)
    plt.show()

    # ROC Curve
    y_true = label_binarize(all_labels, classes=list(range(len(class_labels))))
    y_score = np.array(all_probabilities)
    n_classes = y_true.shape[1]

    fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
    for i in range(n_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes

    fpr_macro, tpr_macro = all_fpr, mean_tpr
    roc_auc_macro = auc(fpr_macro, tpr_macro)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_micro, tpr_micro, label=f'Micro-average ROC (AUC = {roc_auc_micro:.2f})')
    plt.plot(fpr_macro, tpr_macro, linestyle='--', label=f'Macro-average ROC (AUC = {roc_auc_macro:.2f})')

    for i in range(n_classes):
        plt.plot(fpr_dict[i], tpr_dict[i], label=f'{class_labels[i]} (AUC = {roc_auc_dict[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Micro, Macro, Per-Class)')
    plt.legend(loc='lower right')
    plt.tight_layout()

    file_path = os.path.join(path_dir_output, prefix_file_output + "combined_roc_curve.png")
    plt.savefig(file_path, dpi=300)
    plt.show()

    # Save Results
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    file_path = os.path.join(path_dir_output, prefix_file_output + "confusion_matrix.csv")
    cm_df.to_csv(file_path)

    if 'history' in globals():
        file_path = os.path.join(path_dir_output, prefix_file_output + "training_history.csv")
        pd.DataFrame(history).to_csv(file_path, index=False)

    print("\n✅ Results saved in '{}' folder.".format(path_dir_output))

evaluate(model, test_loader, class_labels=class_names, device=DEVICE, criterion=nn.CrossEntropyLoss())
