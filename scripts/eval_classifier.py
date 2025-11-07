import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

from src.inference_timm import TimmClassifier

CLASS_NAMES = ["no_tumor","glioma","meningioma","pituitary"]

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    need = {"path","multiclass_label","class_name","binary_label"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing {missing}")
    return df

def ensure_dir(d: Path): d.mkdir(parents=True, exist_ok=True)

def plot_cm(cm, labels, out_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels, yticklabels=labels,
        ylabel="True label", xlabel="Predicted label", title=title
    )
    # annotate
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def plot_roc(y_true, y_score, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0,1],[0,1],"--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC (Tumor vs No Tumor)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)
    return roc_auc

def plot_pr(y_true, y_score, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (Tumor vs No Tumor)")
    ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)
    return ap

def threshold_sweep(y_true, y_score, out_path):
    # returns threshold with best Youden-J; plots F1 and J vs t
    ts = np.linspace(0.0, 1.0, 101)
    f1s, js = [], []
    best_t, best_j = 0.5, -1.0
    for t in ts:
        y_pred = (y_score >= t).astype(int)
        tp = ((y_pred==1) & (y_true==1)).sum()
        tn = ((y_pred==0) & (y_true==0)).sum()
        fp = ((y_pred==1) & (y_true==0)).sum()
        fn = ((y_pred==0) & (y_true==1)).sum()
        prec = tp / max(tp+fp, 1)
        rec  = tp / max(tp+fn, 1)
        f1   = 2*prec*rec / max(prec+rec, 1e-9)
        tpr  = rec
        fpr  = fp / max(fp+tn, 1)
        J    = tpr - fpr
        f1s.append(f1); js.append(J)
        if J > best_j:
            best_j, best_t = J, t
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(ts, f1s, label="F1")
    ax.plot(ts, js, label="Youden J")
    ax.axvline(best_t, linestyle="--", label=f"Best J @ {best_t:.2f}")
    ax.set(xlabel="Threshold", ylabel="Score", title="Threshold sweep (F1 & Youden-J)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)
    return float(best_t), float(best_j)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV (e.g., data/folder_csv/val.csv)")
    ap.add_argument("--out", default="runs/eval_resnet18")
    ap.add_argument("--meta", default="models/metadata.json")
    ap.add_argument("--ckpt", default="models/classifier_state.pt")
    ap.add_argument("--img_limit", type=int, default=0, help="Limit #images for quick runs (0=all)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Initial tumor decision threshold")
    args = ap.parse_args()

    out_dir = Path(args.out); ensure_dir(out_dir)
    print(f"[eval] writing artifacts to {out_dir}")

    clf = TimmClassifier(meta_path=args.meta, ckpt_path=args.ckpt)

    df = load_csv(args.csv)
    if args.img_limit > 0:
        df = df.sample(n=min(args.img_limit, len(df)), random_state=42).reset_index(drop=True)

    y_true_mc = df["multiclass_label"].to_numpy().astype(int)
    y_true_bin = df["binary_label"].to_numpy().astype(int)

    probs = np.zeros((len(df), 4), dtype=np.float32)
    start = time.time()
    for i, p in enumerate(df["path"]):
        # Open as RGB -> HWC float [0..1]
        img = Image.open(p).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        probs[i] = clf.predict_proba(arr)
    dt = time.time() - start
    print(f"[eval] ran {len(df)} images in {dt:.1f}s")

    pred_mc = probs.argmax(axis=1)
    tumor_prob = 1.0 - probs[:,0]           # 1 - P(no_tumor)
    pred_bin = (tumor_prob >= args.threshold).astype(int)

    # --- Confusion matrices ---
    cm_mc = confusion_matrix(y_true_mc, pred_mc, labels=[0,1,2,3])
    plot_cm(cm_mc, CLASS_NAMES, out_dir/"cm_multiclass.png", "4-class Confusion Matrix")

    cm_bin = confusion_matrix(y_true_bin, pred_bin, labels=[0,1])
    plot_cm(cm_bin, ["no_tumor","tumor"], out_dir/"cm_binary.png", "Binary Confusion Matrix")

    # --- Classification report (multiclass) ---
    report = classification_report(y_true_mc, pred_mc, target_names=CLASS_NAMES, output_dict=True)

    # --- ROC / PR for tumor vs no_tumor ---
    roc_auc = plot_roc(y_true_bin, tumor_prob, out_dir/"roc_binary.png")
    ap = plot_pr(y_true_bin, tumor_prob, out_dir/"pr_binary.png")

    # --- Threshold sweep ---
    best_t, best_j = threshold_sweep(y_true_bin, tumor_prob, out_dir/"threshold_sweep.png")

    # --- Save predictions CSV ---
    pred_rows = []
    for i, row in df.iterrows():
        pred_rows.append({
            "path": row["path"],
            "true_class": CLASS_NAMES[y_true_mc[i]],
            "pred_class": CLASS_NAMES[int(pred_mc[i])],
            "correct": bool(pred_mc[i] == y_true_mc[i]),
            "p_no_tumor": float(probs[i,0]),
            "p_glioma": float(probs[i,1]),
            "p_meningioma": float(probs[i,2]),
            "p_pituitary": float(probs[i,3]),
            "tumor_prob": float(tumor_prob[i])
        })
    pd.DataFrame(pred_rows).to_csv(out_dir/"predictions.csv", index=False)

    # --- Save summary JSON ---
    summary = {
        "n": int(len(df)),
        "acc_multiclass": float((pred_mc == y_true_mc).mean()),
        "report_multiclass": report,
        "cm_multiclass": cm_mc.tolist(),
        "acc_binary@{:.2f}".format(args.threshold): float((pred_bin == y_true_bin).mean()),
        "cm_binary@{:.2f}".format(args.threshold): cm_bin.tolist(),
        "roc_auc_binary": float(roc_auc),
        "ap_binary": float(ap),
        "best_threshold_by_YoudenJ": float(best_t),
        "best_J": float(best_j),
        "note": "Non-diagnostic use"
    }
    with open(out_dir/"eval_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[eval] acc(4-class)={summary['acc_multiclass']:.4f} | ROC_AUC={roc_auc:.4f} | AP={ap:.4f}")
    print(f"[eval] Suggested tumor threshold (Youden-J): {best_t:.2f}")
    print(f"[eval] Artifacts: {out_dir}")
if __name__ == "__main__":
    main()
