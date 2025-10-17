# my_flower_app/client_fn.py
from typing import List, Tuple, Optional
import os
import json
import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp  # API baru: torch.amp

from flwr.client import NumPyClient

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from my_flower_app.client_app import load_model, get_data_loaders, DEVICE


# ------------------ Helpers ------------------
def _ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def _eval_loader(model: torch.nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    """Return (avg_loss, accuracy) on given loader."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    tot_loss, tot_correct, tot = 0.0, 0, 0
    with torch.no_grad():
        for a, b, h, y in loader:
            a = a.to(device, non_blocking=True)
            b = b.to(device, non_blocking=True)
            h = h.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(a, b, h)
            loss = criterion(logits, y)
            preds = logits.argmax(1)
            bs = y.size(0)
            tot_loss += float(loss.item()) * bs
            tot_correct += int((preds == y).sum().item())
            tot += int(bs)
    return (tot_loss / max(1, tot)), (tot_correct / max(1, tot))

def _specificity_from_cm(cm: np.ndarray):
    cm = np.asarray(cm, dtype=float)
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)
    denom = (TN + FP)
    denom[denom == 0] = 1.0
    spec_per_class = TN / denom
    support = cm.sum(axis=1).astype(float)
    w = support / support.sum() if support.sum() > 0 else np.zeros_like(support)
    return spec_per_class, float(spec_per_class.mean()), float((spec_per_class * w).sum())


class MosquitoClient(NumPyClient):
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        class_names: List[str],
        client_id: str,
        device,
        lr: float = 1e-4,
        grad_accum_steps: int = 1,
        local_epochs: int = 1,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.client_id = client_id
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.grad_accum_steps = max(int(grad_accum_steps), 1)
        self.local_epochs = max(int(local_epochs), 1)

    # ----- Flower I/O -----
    def get_parameters(self) -> List:
        return [v.detach().cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_parameters(self, parameters: List):
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(state_dict, strict=True)

    # ----- Train (kirim kurva per-epoch) -----
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(config.get("lr", self.lr)),  # LR dari server (lintas round)
            weight_decay=5e-4,
        )
        local_epochs = int(config.get("local_epochs", self.local_epochs))

        use_amp = (self.device.type == "cuda")
        scaler = amp.GradScaler("cuda") if use_amp else None  # API torch.amp

        train_acc_curve, val_acc_curve = [], []
        train_loss_curve, val_loss_curve = [], []

        for _epoch in range(local_epochs):
            running_loss, correct, total = 0.0, 0, 0
            optimizer.zero_grad(set_to_none=True)
            step_in_accum = 0

            for a, b, h, y in self.train_loader:
                a = a.to(self.device, non_blocking=True)
                b = b.to(self.device, non_blocking=True)
                h = h.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if use_amp:
                    with amp.autocast(device_type="cuda"):
                        logits = self.model(a, b, h)
                        loss = self.criterion(logits, y) / self.grad_accum_steps
                    scaler.scale(loss).backward()
                else:
                    with contextlib.nullcontext():
                        logits = self.model(a, b, h)
                        loss = self.criterion(logits, y) / self.grad_accum_steps
                    loss.backward()

                step_in_accum += 1
                if step_in_accum % self.grad_accum_steps == 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    running_loss += float(loss.item()) * y.size(0) * self.grad_accum_steps
                    preds = logits.argmax(1)
                    correct += int((preds == y).sum().item())
                    total += int(y.size(0))

            avg_tr_loss = running_loss / max(total, 1)
            avg_tr_acc = correct / max(total, 1)
            avg_val_loss, avg_val_acc = _eval_loader(self.model, self.val_loader, self.device)

            train_loss_curve.append(float(avg_tr_loss))
            train_acc_curve.append(float(avg_tr_acc))
            val_loss_curve.append(float(avg_val_loss))
            val_acc_curve.append(float(avg_val_acc))

            print(
                f"[Train][{self.client_id}] tr_loss={avg_tr_loss:.4f} tr_acc={avg_tr_acc:.4f} "
                f"val_loss={avg_val_loss:.4f} val_acc={avg_val_acc:.4f}",
                flush=True,
            )

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        metrics = {
            "client_id": self.client_id,
            "train_acc_curve": json.dumps(train_acc_curve),
            "val_acc_curve": json.dumps(val_acc_curve),
            "train_loss_curve": json.dumps(train_loss_curve),
            "val_loss_curve": json.dumps(val_loss_curve),
        }
        return self.get_parameters(), len(self.train_loader.dataset), metrics

    # ----- Validate (full metrics + CM/ROC per-client) -----
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        use_amp = (self.device.type == "cuda")

        all_labels, all_preds, all_probs = [], [], []
        tot, tot_loss = 0, 0.0
        with torch.no_grad():
            for a, b, h, y in self.val_loader:
                a = a.to(self.device, non_blocking=True)
                b = b.to(self.device, non_blocking=True)
                h = h.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if use_amp:
                    with amp.autocast(device_type="cuda"):
                        logits = self.model(a, b, h)
                        loss = self.criterion(logits, y)
                else:
                    with contextlib.nullcontext():
                        logits = self.model(a, b, h)
                        loss = self.criterion(logits, y)

                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(1)

                bs = int(y.size(0))
                tot_loss += float(loss.item()) * bs
                tot += bs
                all_labels.extend(y.detach().cpu().numpy().tolist())
                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_probs.extend(probs.detach().cpu().numpy().tolist())

        # --- metrik utama (weighted untuk multikelas) ---
        avg_loss = tot_loss / max(tot, 1)
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean().item()

        precision_w = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall_w    = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1_w        = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        n_classes = len(self.class_names)
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_classes)))
        _spec_per_class, spec_macro, spec_weighted = _specificity_from_cm(cm)

        # --- ROC micro-average ---
        y_true_bin = label_binarize(all_labels, classes=list(range(n_classes))) if n_classes > 2 else np.vstack(
            [1 - np.array(all_labels), np.array(all_labels)]
        ).T
        y_score = np.asarray(all_probs, dtype=float)
        if y_true_bin.size > 0 and y_score.size > 0 and y_score.shape[1] == n_classes:
            fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
            auc_micro = auc(fpr_micro, tpr_micro)
            roc_fpr_json = json.dumps([float(x) for x in fpr_micro.tolist()])
            roc_tpr_json = json.dumps([float(x) for x in tpr_micro.tolist()])
        else:
            auc_micro = float("nan")
            roc_fpr_json = json.dumps([])
            roc_tpr_json = json.dumps([])

        # --- ROC per-kelas (one-vs-rest) ---
        per_class_roc = {}
        if y_true_bin.size > 0 and y_score.shape[1] == n_classes:
            for i, cname in enumerate(self.class_names):
                fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                auc_i = auc(fpr_i, tpr_i)
                per_class_roc[cname] = {
                    "fpr": [float(x) for x in fpr_i.tolist()],
                    "tpr": [float(x) for x in tpr_i.tolist()],
                    "auc": float(auc_i),
                }

        # === Simpan CM + ROC COMBINED (micro + macro + per-kelas) ===
        out_root = os.getenv("FL_OUTPUT_DIR", "fl_output")
        client_dir = os.path.join(out_root, "clients", self.client_id)
        _ensure_dir(client_dir)
        try:
            import matplotlib.pyplot as plt
            # Confusion matrix
            try:
                import seaborn as sns
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=self.class_names, yticklabels=self.class_names)
                plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"CM {self.client_id}")
                plt.tight_layout()
                for ext in ("png", "jpg"):
                    plt.savefig(os.path.join(client_dir, f"confusion_matrix.{ext}"), dpi=200)
                plt.close()
            except Exception:
                plt.figure(figsize=(6, 5))
                plt.imshow(cm, aspect="auto"); plt.title(f"CM {self.client_id}")
                plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
                for ext in ("png", "jpg"):
                    plt.savefig(os.path.join(client_dir, f"confusion_matrix.{ext}"), dpi=200)
                plt.close()

            # ROC Combined
            grid = np.linspace(0.0, 1.0, 201)
            curves = []  # (label, fpr, tpr, style)

            if 'fpr_micro' in locals() and len(fpr_micro) > 1:
                tpr_micro_i = np.interp(grid, np.asarray(fpr_micro), np.asarray(tpr_micro))
                curves.append((f"Micro-average ROC (area = {auc_micro:.2f})", grid, tpr_micro_i, {}))

            stack = []
            for cname, obj in per_class_roc.items():
                fpr_i = np.asarray(obj["fpr"], dtype=float)
                tpr_i = np.asarray(obj["tpr"], dtype=float)
                auc_i = float(obj["auc"])
                tpr_i_i = np.interp(grid, fpr_i, tpr_i)
                stack.append(tpr_i_i)
                curves.append((f"Class {cname} ROC (area = {auc_i:.2f})", grid, tpr_i_i, {}))

            if stack:
                macro_tpr = np.mean(np.vstack(stack), axis=0)
                auc_macro = float(np.trapezoid(macro_tpr, grid))
                curves.insert(1, (f"Macro-average ROC (area = {auc_macro:.2f})", grid, macro_tpr, {"linestyle": "--"}))

            plt.figure(figsize=(7.5, 6.0))
            for label, fpr_c, tpr_c, style in curves:
                plt.plot(fpr_c, tpr_c, label=label, **style)
            plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
            plt.title("ROC Curves (Micro, Macro, Per-Class)")
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.grid(True, linestyle="--", alpha=0.35)
            plt.tight_layout()
            for ext in ("png", "jpg"):
                plt.savefig(os.path.join(client_dir, f"roc_combined.{ext}"), dpi=200)
            plt.close()
        except Exception:
            pass

        # Metrik dikirim ke server (tipe sederhana agar lolos validasi Flower)
        metrics_to_server = {
            "client_id": self.client_id,
            "class_names": json.dumps(self.class_names),
            "confusion_matrix": json.dumps(cm.tolist()),
            "accuracy": float(accuracy),
            "precision": float(precision_w),
            "recall": float(recall_w),            # sensitivity (weighted)
            "specificity": float(spec_weighted),  # specificity (weighted)
            "f1_score": float(f1_w),
            "roc_fpr": roc_fpr_json,              # micro-average
            "roc_tpr": roc_tpr_json,
            "roc_auc": float(auc_micro) if np.isfinite(auc_micro) else float("nan"),
            "roc_per_class_json": json.dumps(per_class_roc),  # per-kelas
        }
        return float(avg_loss), tot, metrics_to_server


def client_fn(
    dataset_path: str,
    batch_size: int = 8,
    img_size=224,
    num_workers: int = 2,
    lr: float = 1e-4,
    grad_accum_steps: int = 1,
    local_epochs: int = 1,
):
    train_loader, val_loader, _test_loader, class_names = get_data_loaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
    )
    model = load_model()
    client_id = os.path.basename(os.path.normpath(dataset_path))
    return MosquitoClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names,
        client_id=client_id,
        device=DEVICE,
        lr=lr,
        grad_accum_steps=grad_accum_steps,
        local_epochs=local_epochs,
    ).to_client()
