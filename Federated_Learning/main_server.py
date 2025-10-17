import os
import json
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import start_server, ServerConfig

from my_flower_app.client_app import load_model, get_data_loaders

OUTPUT_DIR = os.getenv("FL_OUTPUT_DIR", "fl_output")

# ---------------- Utils ----------------
def _ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def _json_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None

def _save_csv(path: str, header: list, rows: list):
    _ensure_dir(os.path.dirname(path))
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        for r in rows:
            w.writerow(r)

def _mean_curves(curves: List[List[float]]) -> Optional[List[float]]:
    if not curves:
        return None
    lengths = [len(c) for c in curves if c is not None]
    if not lengths:
        return None
    k = min(lengths)
    if k == 0:
        return None
    arr = np.stack([np.asarray(c[:k], dtype=float) for c in curves], axis=0)
    return arr.mean(axis=0).tolist()

def _interp_mean_roc(fprs: List[List[float]], tprs: List[List[float]], grid_pts: int = 201):
    valid = []
    for fpr, tpr in zip(fprs, tprs):
        if not fpr or not tpr or len(fpr) != len(tpr):
            continue
        f = np.asarray(fpr, dtype=float)
        t = np.asarray(tpr, dtype=float)
        if f[0] > 0.0:
            f = np.concatenate([[0.0], f]); t = np.concatenate([[0.0], t])
        if f[-1] < 1.0:
            f = np.concatenate([f, [1.0]]); t = np.concatenate([t, [1.0]])
        o = np.argsort(f)
        valid.append((f[o], t[o]))
    if not valid:
        return None, None, None, None, None

    grid = np.linspace(0, 1, grid_pts)
    t_interp, aucs = [], []
    per_client_interp = []
    for f, t in valid:
        ti = np.interp(grid, f, t)
        t_interp.append(ti)
        per_client_interp.append((grid, ti))
        aucs.append(np.trapezoid(t, f))  # NumPy 2.x
    mean_tpr = np.mean(np.stack(t_interp, axis=0), axis=0)
    return grid.tolist(), mean_tpr.tolist(), float(np.trapezoid(mean_tpr, grid)), float(np.mean(aucs)), per_client_interp

def _all_metrics_from_cm(cm: np.ndarray):
    cm = np.asarray(cm, dtype=float)
    TP = np.diag(cm)
    support = cm.sum(axis=1)
    total = cm.sum()
    correct = TP.sum()

    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = total - (TP + FP + FN)

    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(TP + FP > 0, TP / (TP + FP), 0.0)
        rec  = np.where(TP + FN > 0, TP / (TP + FN), 0.0)
        spec = np.where(TN + FP > 0, TN / (TN + FP), 0.0)
        f1   = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)

    acc_micro = correct / total if total > 0 else float("nan")
    valid = support > 0
    macro_precision = prec[valid].mean() if np.any(valid) else float("nan")
    macro_recall    = rec[valid].mean()  if np.any(valid) else float("nan")
    macro_specific  = spec[valid].mean() if np.any(valid) else float("nan")
    macro_f1        = f1[valid].mean()   if np.any(valid) else float("nan")

    w = support / support.sum() if support.sum() > 0 else np.zeros_like(support)
    weighted_precision = float((prec * w).sum()) if support.sum() > 0 else float("nan")
    weighted_recall    = float((rec  * w).sum()) if support.sum() > 0 else float("nan")
    weighted_specific  = float((spec * w).sum()) if support.sum() > 0 else float("nan")
    weighted_f1        = float((f1   * w).sum()) if support.sum() > 0 else float("nan")

    return {
        "accuracy_micro": float(acc_micro),
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "specificity_macro": float(macro_specific),
        "f1_macro": float(macro_f1),
        "precision_weighted": float(weighted_precision),
        "recall_weighted": float(weighted_recall),
        "specificity_weighted": float(weighted_specific),
        "f1_weighted": float(weighted_f1),
        "per_class": {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "specificity": spec.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
        }
    }

def _plot_and_save_xy(x, series: dict, title: str, xlabel: str, ylabel: str, out_base: str, add_diag: bool = False):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(f"[PLOT] matplotlib not available, skip {out_base}")
        return
    _ensure_dir(os.path.dirname(out_base))
    plt.figure()
    for label, y in series.items():
        if y is None or len(y) == 0:
            continue
        plt.plot(x, y, label=label)
    if add_diag:
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    if any(series.values()):
        plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    for ext in ("png", "jpg"):
        plt.savefig(f"{out_base}.{ext}", dpi=180)
    plt.close()

def _plot_and_save_cm(cm: np.ndarray, class_names: Optional[List[str]], out_base: str, title: str):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        print(f"[PLOT] matplotlib/seaborn not available, skip {out_base}")
        return
    _ensure_dir(os.path.dirname(out_base))
    plt.figure(figsize=(7.2, 6.4))
    xt = class_names if class_names else [f"pred_{j}" for j in range(cm.shape[1])]
    yt = [f"true_{i}" for i in range(cm.shape[0])]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=xt, yticklabels=yt)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title)
    plt.tight_layout()
    for ext in ("png", "jpg"):
        plt.savefig(f"{out_base}.{ext}", dpi=180)
    plt.close()

def _fmt(v):
    if isinstance(v, (int, float)) and np.isfinite(v):
        return f"{v:.4f}"
    try:
        vv = float(v)
        return f"{vv:.4f}" if np.isfinite(vv) else "NA"
    except Exception:
        return "NA"


# ---------------- Strategy (LR turun tiap 4 round) + cache + FINAL print ----------------
class CustomFedAvg(FedAvg):
    def __init__(self, base_lr=5e-4, lr_gamma=0.85, min_lr=5e-6, decay_every=4, total_rounds: Optional[int]=None, **kwargs):
        super().__init__(**kwargs)
        self.latest_parameters = None
        self.base_lr = float(base_lr); self.lr_gamma = float(lr_gamma); self.min_lr=float(min_lr)
        self.decay_every = int(decay_every)
        self.total_rounds = int(total_rounds) if total_rounds is not None else None

        # ---- cache untuk laporan akhir paksa ----
        self._last_round: int = 0
        self._last_per_client_rows: List[List] = []
        self._last_global_cm: Optional[np.ndarray] = None
        self._last_weighted_acc: Optional[float] = None
        self._last_auc_mean_curve: Optional[float] = None
        self._last_auc_mean_of_aucs: Optional[float] = None
        self._last_class_auc: Dict[str, Tuple[float, float]] = {}  # cname -> (auc_mean_curve, mean_of_aucs)

    def configure_fit(self, server_round, parameters, client_manager):
        cfg_list = super().configure_fit(server_round, parameters, client_manager)
        # step LR tiap 4 round: 1–4 -> step 0, 5–8 -> step 1, dst.
        step_idx = max(0, (server_round - 1) // self.decay_every)
        lr = max(self.min_lr, self.base_lr * (self.lr_gamma ** step_idx))
        for _, fitins in cfg_list:
            cfg = dict(fitins.config or {})
            cfg["lr"] = float(lr)
            cfg.setdefault("local_epochs", 5)
            fitins.config = cfg
        print(f"[ROUND {server_round}] Set client LR = {lr:.3e} (step={step_idx}, decay_every={self.decay_every})")
        return cfg_list

    def aggregate_fit(self, rnd, results, failures):
        agg = super().aggregate_fit(rnd, results, failures)
        if agg is None:
            return agg
        aggregated_parameters = agg[0] if isinstance(agg, tuple) else agg
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters

        round_dir = os.path.join(OUTPUT_DIR, f"round_{rnd}")
        clients_base = os.path.join(round_dir, "clients")
        global_base  = os.path.join(round_dir, "global")
        _ensure_dir(clients_base); _ensure_dir(global_base)

        tr_accs, va_accs, tr_losses, va_losses = [], [], [], []
        per_client_rows = []
        for client_proxy, fit_res in results:
            m = getattr(fit_res, "metrics", {}) or {}
            cid = m.get("client_id") or getattr(client_proxy, "cid", "unknown")

            tr_acc = _json_list(m.get("train_acc_curve"))
            va_acc = _json_list(m.get("val_acc_curve"))
            tr_los = _json_list(m.get("train_loss_curve"))
            va_los = _json_list(m.get("val_loss_curve"))

            if any([tr_acc, va_acc, tr_los, va_los]):
                k = max(len(x) for x in [tr_acc or [], va_acc or [], tr_los or [], va_los or []])
                rows = []
                for e in range(k):
                    rows.append([
                        e + 1,
                        (tr_acc[e] if tr_acc and e < len(tr_acc) else ""),
                        (va_acc[e] if va_acc and e < len(va_acc) else ""),
                        (tr_los[e] if tr_los and e < len(tr_los) else ""),
                        (va_los[e] if va_los and e < len(va_los) else ""),
                    ])
                _ensure_dir(os.path.join(clients_base, cid))
                _save_csv(os.path.join(clients_base, cid, f"curves_round{rnd}.csv"),
                          ["epoch", "train_acc", "val_acc", "train_loss", "val_loss"], rows)

            if tr_acc: tr_accs.append(tr_acc)
            if va_acc: va_accs.append(va_acc)
            if tr_los: tr_losses.append(tr_los)
            if va_los: va_losses.append(va_los)

            per_client_rows.append([cid,
                                    (tr_acc[-1] if tr_acc else ""),
                                    (va_acc[-1] if va_acc else ""),
                                    (tr_los[-1] if tr_los else ""),
                                    (va_los[-1] if va_los else "")])

        if per_client_rows:
            _save_csv(os.path.join(round_dir, f"summary_fit_round{rnd}.csv"),
                      ["client_id", "last_train_acc", "last_val_acc", "last_train_loss", "last_val_loss"],
                      per_client_rows)

        mean_tr_acc  = _mean_curves(tr_accs)
        mean_va_acc  = _mean_curves(va_accs)
        mean_tr_loss = _mean_curves(tr_losses)
        mean_va_loss = _mean_curves(va_losses)
        if any([mean_tr_acc, mean_va_acc, mean_tr_loss, mean_va_loss]):
            k = max(len(x) if x else 0 for x in [mean_tr_acc, mean_va_acc, mean_tr_loss, mean_va_loss])
            rows, epochs = [], list(range(1, k + 1))
            for e in range(k):
                rows.append([
                    e + 1,
                    (mean_tr_acc[e]  if mean_tr_acc  and e < len(mean_tr_acc)  else ""),
                    (mean_va_acc[e]  if mean_va_acc  and e < len(mean_va_acc)  else ""),
                    (mean_tr_loss[e] if mean_tr_loss and e < len(mean_tr_loss) else ""),
                    (mean_va_loss[e] if mean_va_loss and e < len(mean_va_loss) else ""),
                ])
            _save_csv(os.path.join(global_base, f"global_curves_round{rnd}.csv"),
                      ["epoch", "mean_train_acc", "mean_val_acc", "mean_train_loss", "mean_val_loss"], rows)
            _plot_and_save_xy(epochs,
                              {"Train Acc (mean)": mean_tr_acc, "Val Acc (mean)": mean_va_acc},
                              f"Training vs Validation Accuracy (Round {rnd})", "Epoch", "Accuracy",
                              os.path.join(global_base, f"global_accuracy_round{rnd}"))
            _plot_and_save_xy(epochs,
                              {"Train Loss (mean)": mean_tr_loss, "Val Loss (mean)": mean_va_loss},
                              f"Training vs Validation Loss (Round {rnd})", "Epoch", "Loss",
                              os.path.join(global_base, f"global_loss_round{rnd}"))

        return agg

    def aggregate_evaluate(self, rnd, results, failures):
        agg_loss = super().aggregate_evaluate(rnd, results, failures)

        round_dir   = os.path.join(OUTPUT_DIR, f"round_{rnd}")
        clients_dir = os.path.join(round_dir, "clients")
        global_dir  = os.path.join(round_dir, "global")
        _ensure_dir(clients_dir); _ensure_dir(global_dir)

        fp_rs, tp_rs, cids = [], [], []
        per_class_fp_rs, per_class_tp_rs = {}, {}
        weighted_acc_num, weighted_acc_den = 0.0, 0.0
        global_cm = None
        global_class_names = None

        per_client_rows = []
        for client_proxy, eval_res in results:
            m = eval_res.metrics or {}
            n = getattr(eval_res, "num_examples", None)
            cid = getattr(client_proxy, "cid", None) or m.get("client_id", "unknown")
            cdir = os.path.join(clients_dir, cid); _ensure_dir(cdir)

            acc  = m.get("accuracy", None)
            prec = m.get("precision", None)
            rec  = m.get("recall", None)
            spec = m.get("specificity", None)
            f1   = m.get("f1_score", None)
            aucv = m.get("roc_auc", None)

            print(f"[ROUND {rnd}] [Client {cid}] "
                  f"Acc={_fmt(acc)} | Prec={_fmt(prec)} | Sens/Rec={_fmt(rec)} | "
                  f"Spec={_fmt(spec)} | F1={_fmt(f1)} | AUC={_fmt(aucv)}")

            if n is not None and isinstance(acc, (int, float)):
                weighted_acc_num += float(acc) * float(n)
                weighted_acc_den += float(n)

            per_client_rows.append([cid,
                                    float(acc) if isinstance(acc, (int, float)) else "",
                                    float(prec) if isinstance(prec, (int, float)) else "",
                                    float(rec) if isinstance(rec, (int, float)) else "",
                                    float(spec) if isinstance(spec, (int, float)) else "",
                                    float(f1) if isinstance(f1, (int, float)) else "",
                                    float(aucv) if isinstance(aucv, (int, float)) else ""])

            # ROC micro-average per client
            fpr = _json_list(m.get("roc_fpr"))
            tpr = _json_list(m.get("roc_tpr"))
            if fpr and tpr and len(fpr) == len(tpr) and len(fpr) > 0:
                rows = [[ff, tt] for ff, tt in zip(fpr, tpr)]
                _save_csv(os.path.join(cdir, f"roc_round{rnd}.csv"), ["fpr", "tpr"], rows)
                _plot_and_save_xy(fpr, {f"{cid} (AUC={_fmt(aucv)})": tpr},
                                  f"ROC {cid} (Round {rnd})", "False Positive Rate", "True Positive Rate",
                                  os.path.join(cdir, f"roc_round{rnd}"), add_diag=True)
                fp_rs.append([float(x) for x in fpr]); tp_rs.append([float(x) for x in tpr]); cids.append(cid)

            # ROC per-kelas (one-vs-rest) dari klien
            rpc = m.get("roc_per_class_json")
            if isinstance(rpc, str):
                try: rpc = json.loads(rpc)
                except Exception: rpc = None
            if isinstance(rpc, dict):
                for cname, obj in rpc.items():
                    fpr_i = obj.get("fpr", []); tpr_i = obj.get("tpr", [])
                    if fpr_i and tpr_i and len(fpr_i) == len(tpr_i):
                        per_class_fp_rs.setdefault(cname, []).append([float(x) for x in fpr_i])
                        per_class_tp_rs.setdefault(cname, []).append([float(x) for x in tpr_i])
                        # simpan csv + plot per-klien per-kelas (opsional)
                        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in cname)
                        _save_csv(os.path.join(cdir, f"roc_{slug}_round{rnd}.csv"),
                                  ["fpr","tpr"], [[ff,tt] for ff,tt in zip(fpr_i,tpr_i)])
                        _plot_and_save_xy(fpr_i, {f"{cname}": tpr_i},
                                          f"ROC {cid} - {cname} (Round {rnd})",
                                          "False Positive Rate", "True Positive Rate",
                                          os.path.join(cdir, f"roc_{slug}_round{rnd}"), add_diag=True)

            # Confusion matrix client
            cm = m.get("confusion_matrix")
            if isinstance(cm, str):
                try: cm = json.loads(cm)
                except Exception: cm = None
            if isinstance(cm, list) and cm and isinstance(cm[0], list):
                cm_arr = np.asarray(cm, dtype=int)
                rows_cm = [[f"true_{i}"] + [int(x) for x in row] for i, row in enumerate(cm_arr.tolist())]
                _save_csv(os.path.join(cdir, f"confusion_matrix_round{rnd}.csv"),
                          [""] + [f"pred_{j}" for j in range(cm_arr.shape[1])], rows_cm)
                _plot_and_save_cm(cm_arr, None, os.path.join(cdir, f"confusion_matrix_round{rnd}"),
                                  f"Confusion Matrix {cid} (Round {rnd})")
                if global_cm is None:
                    global_cm = cm_arr.copy()
                else:
                    if global_cm.shape == cm_arr.shape:
                        global_cm += cm_arr
                    else:
                        print(f"[WARN] CM shape mismatch for client {cid}: {cm_arr.shape} vs {global_cm.shape}")

            # simpan class names sekali
            cls_names = m.get("class_names")
            if isinstance(cls_names, str):
                try: cls_names = json.loads(cls_names)
                except Exception: cls_names = None
            if isinstance(cls_names, list) and not global_class_names:
                global_class_names = [str(x) for x in cls_names]

        if per_client_rows:
            _save_csv(os.path.join(round_dir, f"summary_eval_round{rnd}.csv"),
                      ["client_id", "accuracy", "precision", "recall", "specificity", "f1_score", "roc_auc"],
                      per_client_rows)

        if weighted_acc_den > 0:
            acc_global_weighted = weighted_acc_num / weighted_acc_den
            print(f"[ROUND {rnd}] Global Accuracy (weighted by #samples): {acc_global_weighted:.4f}")
            _save_csv(os.path.join(global_dir, f"global_metrics_weighted_round{rnd}.csv"),
                      ["metric", "value"], [["accuracy_weighted", acc_global_weighted]])
        else:
            acc_global_weighted = None

        # ---- Global metrics dari total CM ----
        if global_cm is not None:
            rows_cm = [[f"true_{i}"] + [int(x) for x in row] for i, row in enumerate(global_cm.tolist())]
            _save_csv(os.path.join(global_dir, f"global_confusion_matrix_round{rnd}.csv"),
                      [""] + [f"pred_{j}" for j in range(global_cm.shape[1])], rows_cm)
            _plot_and_save_cm(global_cm, global_class_names,
                              os.path.join(global_dir, f"global_confusion_matrix_round{rnd}"),
                              f"Global Confusion Matrix (Round {rnd})")

            gm = _all_metrics_from_cm(global_cm)
            _save_csv(os.path.join(global_dir, f"global_metrics_from_cm_round{rnd}.csv"),
                      ["metric", "value"],
                      [["accuracy_micro", gm["accuracy_micro"]],
                       ["precision_macro", gm["precision_macro"]],
                       ["recall_macro", gm["recall_macro"]],
                       ["specificity_macro", gm["specificity_macro"]],
                       ["f1_macro", gm["f1_macro"]],
                       ["precision_weighted", gm["precision_weighted"]],
                       ["recall_weighted", gm["recall_weighted"]],
                       ["specificity_weighted", gm["specificity_weighted"]],
                       ["f1_weighted", gm["f1_weighted"]]])
            print(f"[ROUND {rnd}] Global (from CM) -> "
                  f"Acc(micro)={gm['accuracy_micro']:.4f}, "
                  f"Prec(macro)={gm['precision_macro']:.4f}, Rec(macro)={gm['recall_macro']:.4f}, "
                  f"Spec(macro)={gm['specificity_macro']:.4f}, F1(macro)={gm['f1_macro']:.4f}; "
                  f"Prec(w)={gm['precision_weighted']:.4f}, Rec(w)={gm['recall_weighted']:.4f}, "
                  f"Spec(w)={gm['specificity_weighted']:.4f}, F1(w)={gm['f1_weighted']:.4f}")
        else:
            gm = None

        # ---- Global ROC (micro-average) ----
        auc_mean_curve = auc_mean_of_auc = None
        if fp_rs and tp_rs:
            mean_fpr, mean_tpr, auc_mean_curve, auc_mean_of_auc, _ = _interp_mean_roc(fp_rs, tp_rs, grid_pts=201)
            _save_csv(os.path.join(global_dir, f"global_roc_round{rnd}.csv"),
                      ["fpr", "mean_tpr"], [[f, t] for f, t in zip(mean_fpr, mean_tpr)])
            _save_csv(os.path.join(global_dir, f"global_auc_round{rnd}.csv"),
                      ["metric", "value"], [["auc_mean_curve", auc_mean_curve], ["auc_mean_of_AUCs", auc_mean_of_auc]])
            print(f"[ROUND {rnd}] ROC AUC (mean curve): {auc_mean_curve:.4f} | ROC AUC (mean of AUCs): {auc_mean_of_auc:.4f}")

            try:
                import matplotlib.pyplot as plt
                _ensure_dir(global_dir)
                plt.figure()
                for cid, fpr, tpr in zip(cids, fp_rs, tp_rs):
                    plt.plot(fpr, tpr, linewidth=1, alpha=0.35, label=f"{cid}")
                plt.plot(mean_fpr, mean_tpr, linewidth=2.5, label=f"Mean ROC (AUC={auc_mean_curve:.4f})")
                plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
                plt.title(f"ROC Global (Round {rnd})"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
                plt.legend(loc="lower right", fontsize=8, ncol=2)
                plt.grid(True, linestyle="--", alpha=0.4); plt.tight_layout()
                for ext in ("png", "jpg"):
                    plt.savefig(os.path.join(global_dir, f"global_roc_round{rnd}.{ext}"), dpi=180)
                plt.close()
            except Exception:
                pass

        # ---- Global ROC per-kelas (mean curve + mean-of-AUCs) + COMBINED ----
        class_mean_curves = {}  # cname -> (mfpr, mtpr, auc_mean_curve, auc_mean_of_aucs)
        for cname in sorted(per_class_fp_rs.keys()):
            fprs_list = per_class_fp_rs[cname]
            tprs_list = per_class_tp_rs[cname]
            if not fprs_list or not tprs_list:
                continue
            mfpr, mtpr, auc_mean_curve_cls, auc_mean_of_auc_cls, _ = _interp_mean_roc(fprs_list, tprs_list, grid_pts=201)
            class_mean_curves[cname] = (mfpr, mtpr, auc_mean_curve_cls, auc_mean_of_auc_cls)

            slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in cname)
            _save_csv(os.path.join(global_dir, f"global_roc_{slug}_round{rnd}.csv"),
                      ["fpr", "mean_tpr"], [[f, t] for f, t in zip(mfpr, mtpr)])
            _plot_and_save_xy(mfpr, {f"Mean ROC {cname} (AUC={auc_mean_curve_cls:.4f})": mtpr},
                              f"ROC Global - {cname} (Round {rnd})",
                              "False Positive Rate", "True Positive Rate",
                              os.path.join(global_dir, f"global_roc_{slug}_round{rnd}"),
                              add_diag=True)
            print(f"[ROUND {rnd}] {cname} ROC AUC (mean curve): {auc_mean_curve_cls:.4f} | "
                  f"{cname} ROC AUC (mean of AUCs): {auc_mean_of_auc_cls:.4f}")

        # COMBINED: micro + macro + semua kelas dalam SATU gambar
        if class_mean_curves:
            try:
                import matplotlib.pyplot as plt
                any_cname = next(iter(class_mean_curves))
                grid = np.asarray(class_mean_curves[any_cname][0], dtype=float)

                plt.figure(figsize=(7.5, 6.0))
                stack = []
                # micro-average (jika tersedia)
                if (fp_rs and tp_rs) and (auc_mean_curve is not None):
                    # mean_fpr/mean_tpr sudah dihitung di atas
                    mean_fpr, mean_tpr, *_ = _interp_mean_roc(fp_rs, tp_rs, grid_pts=201)
                    plt.plot(mean_fpr, mean_tpr, label=f"Micro-average ROC (area = {auc_mean_curve:.2f})")

                # per-kelas
                for cname, (fpr_c, tpr_c, auc_c, _) in sorted(class_mean_curves.items()):
                    plt.plot(fpr_c, tpr_c, label=f"Class {cname} ROC (area = {auc_c:.2f})")
                    stack.append(np.asarray(tpr_c, dtype=float))

                macro_tpr = np.mean(np.vstack(stack), axis=0)
                auc_macro = float(np.trapezoid(macro_tpr, grid))
                plt.plot(grid, macro_tpr, linestyle="--", label=f"Macro-average ROC (area = {auc_macro:.2f})")

                plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
                plt.title(f"ROC Curves (Micro, Macro, Per-Class) — Round {rnd}")
                plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
                plt.legend(loc="lower right")
                plt.grid(True, linestyle="--", alpha=0.35)
                plt.tight_layout()
                for ext in ("png", "jpg"):
                    plt.savefig(os.path.join(global_dir, f"global_roc_combined_round{rnd}.{ext}"), dpi=180)
                plt.close()
            except Exception:
                pass

        # -------- cache untuk keperluan "paksa cetak" setelah server selesai ----------
        self._last_round = rnd
        self._last_per_client_rows = per_client_rows
        self._last_global_cm = global_cm
        self._last_weighted_acc = acc_global_weighted
        self._last_auc_mean_curve = auc_mean_curve
        self._last_auc_mean_of_aucs = auc_mean_of_auc
        self._last_class_auc = {k: (v[2], v[3]) for k, v in class_mean_curves.items()}

        # ================= FINAL PRINT (kalau dipanggil di round terakhir) =================
        if hasattr(self, "total_rounds") and self.total_rounds is not None and rnd == self.total_rounds:
            print("\n===== FINAL PER-CLIENT METRICS =====")
            for row in per_client_rows:
                cid, acc, prec, rec, spec, f1, aucv = row
                print(f"[FINAL][Client {cid}] "
                      f"Acc={_fmt(acc)} | Prec={_fmt(prec)} | Sens/Rec={_fmt(rec)} | "
                      f"Spec={_fmt(spec)} | F1={_fmt(f1)} | AUC={_fmt(aucv)}")
            if global_cm is not None:
                gm = _all_metrics_from_cm(global_cm)
                print("\n===== FINAL GLOBAL METRICS (from confusion matrix) =====")
                print(f"Acc(micro)={gm['accuracy_micro']:.4f}")
                print(f"Precision(macro)={gm['precision_macro']:.4f}")
                print(f"Sensitivity/Recall(macro)={gm['recall_macro']:.4f}")
                print(f"Specificity(macro)={gm['specificity_macro']:.4f}")
                print(f"F1(macro)={gm['f1_macro']:.4f}")

        return agg_loss

    # ---------- fungsi util untuk memaksa cetak setelah server selesai ----------
    def _print_final_blocks(self, rnd: int, per_client_rows: List[List], global_cm: Optional[np.ndarray]):
        print("\n===== FINAL PER-CLIENT METRICS =====")
        for row in per_client_rows:
            cid, acc, prec, rec, spec, f1, aucv = row
            print(f"[FINAL][Client {cid}] "
                  f"Acc={_fmt(acc)} | Prec={_fmt(prec)} | Sens/Rec={_fmt(rec)} | "
                  f"Spec={_fmt(spec)} | F1={_fmt(f1)} | AUC={_fmt(aucv)}")
        if global_cm is not None:
            gm = _all_metrics_from_cm(global_cm)
            print("\n===== FINAL GLOBAL METRICS (from confusion matrix) =====")
            print(f"Acc(micro)={gm['accuracy_micro']:.4f}")
            print(f"Precision(macro)={gm['precision_macro']:.4f}")
            print(f"Sensitivity/Recall(macro)={gm['recall_macro']:.4f}")
            print(f"Specificity(macro)={gm['specificity_macro']:.4f}")
            print(f"F1(macro)={gm['f1_macro']:.4f}")

    def print_final_report_if_needed(self):
        """Dipanggil setelah start_server() selesai; memaksa cetak round terakhir kalau belum muncul."""
        if self.total_rounds is None:
            return
        if self._last_round < self.total_rounds:
            # Cetak header round terakhir agar mirip log Flower
            print("INFO :      ")
            print(f"INFO :      [ROUND {self.total_rounds}]")
            # LR yang di-set pada round terakhir (informasi indikatif)
            step_idx = max(0, (self.total_rounds - 1) // self.decay_every)
            lr = max(self.min_lr, self.base_lr * (self.lr_gamma ** step_idx))
            print(f"[ROUND {self.total_rounds}] Set client LR = {lr:.3e} (step={step_idx}, decay_every={self.decay_every})")

            # Cetak ulang per-client dari cache terakhir
            for row in self._last_per_client_rows:
                cid, acc, prec, rec, spec, f1, aucv = row
                print(f"[ROUND {self.total_rounds}] [Client {cid}] "
                      f"Acc={_fmt(acc)} | Prec={_fmt(prec)} | Sens/Rec={_fmt(rec)} | "
                      f"Spec={_fmt(spec)} | F1={_fmt(f1)} | AUC={_fmt(aucv)}")

            if self._last_weighted_acc is not None:
                print(f"[ROUND {self.total_rounds}] Global Accuracy (weighted by #samples): {self._last_weighted_acc:.4f}")

            if self._last_global_cm is not None:
                gm = _all_metrics_from_cm(self._last_global_cm)
                print(f"[ROUND {self.total_rounds}] Global (from CM) -> "
                      f"Acc(micro)={gm['accuracy_micro']:.4f}, "
                      f"Prec(macro)={gm['precision_macro']:.4f}, Rec(macro)={gm['recall_macro']:.4f}, "
                      f"Spec(macro)={gm['specificity_macro']:.4f}, F1(macro)={gm['f1_macro']:.4f}; "
                      f"Prec(w)={gm['precision_weighted']:.4f}, Rec(w)={gm['recall_weighted']:.4f}, "
                      f"Spec(w)={gm['specificity_weighted']:.4f}, F1(w)={gm['f1_weighted']:.4f}")

            if (self._last_auc_mean_curve is not None) and (self._last_auc_mean_of_aucs is not None):
                print(f"[ROUND {self.total_rounds}] ROC AUC (mean curve): {self._last_auc_mean_curve:.4f} | "
                      f"ROC AUC (mean of AUCs): {self._last_auc_mean_of_aucs:.4f}")

            # per-kelas
            for cname, (auc_mean_curve_cls, auc_mean_of_auc_cls) in sorted(self._last_class_auc.items()):
                print(f"[ROUND {self.total_rounds}] {cname} ROC AUC (mean curve): {auc_mean_curve_cls:.4f} | "
                      f"{cname} ROC AUC (mean of AUCs): {auc_mean_of_auc_cls:.4f}")

            # Blok FINAL
            self._print_final_blocks(self.total_rounds, self._last_per_client_rows, self._last_global_cm)


# --------- Optional: simple global eval on CPU (setelah training) ----------
def evaluate_global(parameters, dataset_path_eval: str, batch_size_eval: int = 8):
    device = torch.device("cpu")
    model = load_model(pretrained=False).to(device)
    final_weights = parameters_to_ndarrays(parameters)
    state = {k: torch.from_numpy(v) for k, v in zip(model.state_dict().keys(), final_weights)}
    model.load_state_dict(state, strict=True)

    test_partition = os.path.join(dataset_path_eval, "partition_0")
    _, _, test_loader, class_names = get_data_loaders(
        dataset_path=test_partition, batch_size=batch_size_eval, img_size=224, num_workers=0,
    )

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for a, b, h, y in test_loader:
            logits = model(a.to(device), b.to(device), h.to(device))
            preds = logits.argmax(1)
            all_preds.append(preds.cpu().numpy()); all_labels.append(y.cpu().numpy())

    if not all_preds:
        print("[WARN] Empty test set for evaluation."); return
    y_pred = np.concatenate(all_preds); y_true = np.concatenate(all_labels)
    acc = (y_pred == y_true).mean()
    print(f"[GLOBAL EVAL] Accuracy: {acc:.4f}  (N={y_true.size})")


# ---------------- Start server ----------------
def start_fl_server(num_rounds: int = 20, server_address: str = "127.0.0.1:8090", dataset_root_for_eval: str = "dataset_opsi_part"):
    init_model = load_model(pretrained=True)
    initial_parameters = ndarrays_to_parameters([v.detach().cpu().numpy() for _, v in init_model.state_dict().items()])

    strategy = CustomFedAvg(
        base_lr=5e-4, lr_gamma=0.85, min_lr=5e-6, decay_every=4,
        total_rounds=num_rounds,                     # agar FINAL print dieksekusi/di-cache
        fraction_fit=1.0, fraction_evaluate=1.0, min_available_clients=2,
        initial_parameters=initial_parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    # Jalankan FL
    start_server(server_address=server_address, config=config, strategy=strategy)

    # Paksa cetak bila round terakhir tidak tercetak lengkap di log
    strategy.print_final_report_if_needed()

    # Evaluasi dan simpan model global
    if strategy.latest_parameters is not None:
        print("[SERVER] Evaluating aggregated global model on CPU...")
        evaluate_global(strategy.latest_parameters, dataset_root_for_eval, batch_size_eval=8)
        final_weights = parameters_to_ndarrays(strategy.latest_parameters)
        save_model = load_model(pretrained=False)
        save_model.load_state_dict({k: torch.from_numpy(v) for k, v in zip(save_model.state_dict().keys(), final_weights)}, strict=True)
        os.makedirs("global_output", exist_ok=True)
        path = os.path.join("global_output", "global_model_fl.pth")
        torch.save(save_model.state_dict(), path)
        print(f"[✔] Global model saved at {path}")
    else:
        print("[WARN] No global parameters to save/evaluate.")

if __name__ == "__main__":
    start_fl_server(num_rounds=20, server_address="127.0.0.1:8090", dataset_root_for_eval="dataset_opsi_part")
