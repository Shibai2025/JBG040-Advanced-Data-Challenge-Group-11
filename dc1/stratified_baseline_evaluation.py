"""
stratified_baseline_evaluation.py

Evaluate a trained baseline model on the official TEST set only.

Designed to work with the new training workflow where the training script:
- performs a stratified train/validation split on the original training set
- saves checkpoints into nested folders such as:
    dc1/model_weights/baseline_best/run_YYYYMMDD_HHMMSS/best_model.pt
    dc1/model_weights/baseline_final/run_YYYYMMDD_HHMMSS/final_model.pt

Behavior:
- If --model_path is provided, that checkpoint is used.
- If --model_path is NOT provided, the script automatically finds the latest
  best_model.pt under --weights_dir.

What this script does:
- loads X_test.npy / Y_test.npy
- loads a saved checkpoint
- runs inference on the official test set
- computes:
    - test loss
    - accuracy
    - macro-F1
    - per-class precision / recall / F1 / support
    - per-class TP / FP / FN / TN
    - confusion matrix
    - selective prediction curve (coverage vs accuracy / macro-F1)
- saves:
    - metrics.json
    - summary.txt
    - classification_report.txt
    - confusion_matrix.png
    - confusion_matrix_normalized.png
    - threshold_coverage_curve.png

Recommended run from project root:
    python -m dc1.stratified_baseline_evaluation

Or specify a checkpoint explicitly:
    python -m dc1.stratified_baseline_evaluation --model_path dc1/model_weights/baseline_best/run_20260315_220000/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

plt.style.use("default")

from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net


CLASS_NAMES = [
    "Atelectasis",     # class 0
    "Effusion",        # class 1
    "Infiltration",    # class 2
    "No Finding",      # class 3
    "Nodule",          # class 4
    "Pneumothorax",    # class 5
]


@dataclass
class EvalResults:
    test_loss: float
    accuracy: float
    macro_f1: float
    confusion_matrix: List[List[int]]
    class_report: str
    per_class_metrics: List[Dict[str, float]]
    per_class_stats: List[Dict[str, int]]


def pick_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_path(input_path: Optional[str], default: Path, project_root: Path) -> Path:
    if input_path is None:
        return default
    p = Path(input_path)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def find_latest_best_model(weights_dir: Path) -> Path:
    """
    Recursively find the most recently modified best_model.pt under weights_dir.
    """
    candidates = list(weights_dir.rglob("best_model.pt"))

    if not candidates:
        raise FileNotFoundError(
            f"No best_model.pt found under '{weights_dir}'.\n"
            "Run the training script first so it creates model_weights/.../best_model.pt"
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


@torch.no_grad()
def run_inference_with_batch_sampler(
    model: torch.nn.Module,
    sampler: BatchSampler,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns:
        y_true: (N,)
        y_pred: (N,)
        y_prob: (N, C)
        mean_loss: float
    """
    model.eval()

    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_prob_all: List[np.ndarray] = []

    total_loss = 0.0
    total_samples = 0

    for xb, yb in sampler:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        batch_size = yb.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        y_true_all.extend(yb.detach().cpu().numpy().tolist())
        y_pred_all.extend(pred.detach().cpu().numpy().tolist())
        y_prob_all.append(probs.detach().cpu().numpy())

    y_true = np.array(y_true_all, dtype=int)
    y_pred = np.array(y_pred_all, dtype=int)
    y_prob = np.concatenate(y_prob_all, axis=0) if len(y_prob_all) > 0 else np.zeros((0, 0), dtype=float)
    mean_loss = total_loss / total_samples if total_samples > 0 else 0.0

    return y_true, y_pred, y_prob, float(mean_loss)


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else 0.0


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def compute_per_class_metrics(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> List[Dict[str, float]]:
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    metrics: List[Dict[str, float]] = []

    for i in range(n_classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) != 0 else 0.0

        metrics.append(
            {
                "class_index": i,
                "class_name": class_names[i],
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
            }
        )

    return metrics


def compute_per_class_stats(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> List[Dict[str, int]]:
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    total = int(cm.sum())
    stats: List[Dict[str, int]] = []

    for i in range(n_classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(total - (tp + fp + fn))

        stats.append(
            {
                "class_index": i,
                "class_name": class_names[i],
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
            }
        )

    return stats


def compute_macro_f1_from_cm(cm: np.ndarray) -> float:
    per_class = compute_per_class_metrics(cm)
    f1s = [item["f1"] for item in per_class]
    return float(np.mean(f1s)) if len(f1s) else 0.0


def compute_classification_report(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> str:
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    lines: List[str] = []
    lines.append("Class\tPrecision\tRecall\tF1\tSupport")

    per_class = compute_per_class_metrics(cm, class_names)

    precisions = [item["precision"] for item in per_class]
    recalls = [item["recall"] for item in per_class]
    f1s = [item["f1"] for item in per_class]
    supports = [item["support"] for item in per_class]

    for item in per_class:
        lines.append(
            f"{item['class_name']}\t"
            f"{item['precision']:.3f}\t\t"
            f"{item['recall']:.3f}\t"
            f"{item['f1']:.3f}\t"
            f"{item['support']}"
        )

    macro_p = float(np.mean(precisions)) if len(precisions) else 0.0
    macro_r = float(np.mean(recalls)) if len(recalls) else 0.0
    macro_f1 = float(np.mean(f1s)) if len(f1s) else 0.0
    total_support = int(np.sum(supports)) if len(supports) else 0

    lines.append("")
    lines.append(f"MacroAvg\t{macro_p:.3f}\t\t{macro_r:.3f}\t{macro_f1:.3f}\t{total_support}")
    return "\n".join(lines)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_loss: float,
    n_classes: int,
    class_names: Optional[List[str]] = None,
) -> EvalResults:
    cm = compute_confusion_matrix(y_true, y_pred, n_classes)
    acc = compute_accuracy(y_true, y_pred)
    macro_f1 = compute_macro_f1_from_cm(cm)
    report = compute_classification_report(cm, class_names)
    per_class_metrics = compute_per_class_metrics(cm, class_names)
    per_class_stats = compute_per_class_stats(cm, class_names)

    return EvalResults(
        test_loss=float(test_loss),
        accuracy=float(acc),
        macro_f1=float(macro_f1),
        confusion_matrix=cm.tolist(),
        class_report=report,
        per_class_metrics=per_class_metrics,
        per_class_stats=per_class_stats,
    )


def save_confusion_matrix_png(
    cm: np.ndarray,
    out_path: Path,
    class_names: Optional[List[str]] = None,
) -> None:
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    plt.figure(figsize=(8, 6.5), facecolor="white")
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix", fontsize=16)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=10)

    ticks = np.arange(n_classes)
    plt.xticks(ticks, class_names, rotation=30, ha="right", fontsize=11)
    plt.yticks(ticks, class_names, fontsize=11)

    thresh = cm.max() * 0.6 if cm.size else 0
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=10,
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label", fontsize=13)
    plt.xlabel("Predicted label", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close()


def save_confusion_matrix_normalized_png(
    cm: np.ndarray,
    out_path: Path,
    class_names: Optional[List[str]] = None,
) -> None:
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    plt.figure(figsize=(8, 6.5), facecolor="white")
    plt.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.title("Normalized Confusion Matrix", fontsize=16)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=10)

    ticks = np.arange(n_classes)
    plt.xticks(ticks, class_names, rotation=30, ha="right", fontsize=11)
    plt.yticks(ticks, class_names, fontsize=11)

    thresh = 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if cm_norm[i, j] > thresh else "black",
            )

    plt.ylabel("True label", fontsize=13)
    plt.xlabel("Predicted label", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close()


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Selective prediction:
      keep a sample if max_prob >= threshold
    Computes coverage, accuracy, and macro-F1 on kept samples.
    """
    n = len(y_true)
    if n == 0 or y_prob.size == 0:
        zeros = np.zeros_like(thresholds, dtype=float)
        return zeros, zeros, zeros

    y_pred = np.argmax(y_prob, axis=1)
    max_prob = np.max(y_prob, axis=1)
    n_classes = y_prob.shape[1]

    coverages: List[float] = []
    accs: List[float] = []
    mf1s: List[float] = []

    for t in thresholds:
        keep = max_prob >= t
        kept = int(np.sum(keep))
        coverage = kept / n
        coverages.append(float(coverage))

        if kept == 0:
            accs.append(0.0)
            mf1s.append(0.0)
            continue

        yt = y_true[keep]
        yp = y_pred[keep]
        cm = compute_confusion_matrix(yt, yp, n_classes)

        accs.append(compute_accuracy(yt, yp))
        mf1s.append(compute_macro_f1_from_cm(cm))

    return np.array(coverages), np.array(accs), np.array(mf1s)


def save_coverage_curve_png(
    coverage: np.ndarray,
    macro_f1: np.ndarray,
    acc: np.ndarray,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 5.5), facecolor="white")
    plt.plot(
        coverage,
        macro_f1,
        label="Macro-F1 (kept)",
        color="#1b5e9a",
        linewidth=2.4,
    )
    plt.plot(
        coverage,
        acc,
        label="Accuracy (kept)",
        color="#d97706",
        linewidth=2.4,
    )

    plt.gca().invert_xaxis()
    plt.xlabel("Coverage (fraction kept)", fontsize=13)
    plt.ylabel("Score", fontsize=13)
    plt.title("Selective Prediction: Coverage vs Score", fontsize=16)
    plt.legend(frameon=True, fontsize=11)
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close()


def results_to_dict(results: EvalResults) -> Dict:
    return {
        "test_loss": results.test_loss,
        "accuracy": results.accuracy,
        "macro_f1": results.macro_f1,
        "confusion_matrix": results.confusion_matrix,
        "class_report": results.class_report,
        "per_class_metrics": results.per_class_metrics,
        "per_class_stats": results.per_class_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the latest best baseline model on the official test set.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional explicit path to best_model.pt. If omitted, the latest best_model.pt under --weights_dir is used.",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=None,
        help="Root directory containing saved model folders. Default: <project_root>/dc1/model_weights",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing X_test.npy and Y_test.npy. Default: <project_root>/dc1/data",
    )
    parser.add_argument("--batch_size", type=int, default=100, help="Test batch size")
    parser.add_argument("--n_classes", type=int, default=6, help="Number of classes")
    parser.add_argument(
        "--balanced_test",
        action="store_true",
        help="Use balanced sampling on the test set. Normally keep this OFF.",
    )
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU even if CUDA/MPS is available")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent      # .../dc1
    project_root = base_dir.parent                  # project root

    device = pick_device(force_cpu=args.force_cpu)
    print(f"Using device: {device}")

    data_dir = resolve_path(args.data_dir, base_dir / "data", project_root)
    x_test_path = data_dir / "X_test.npy"
    y_test_path = data_dir / "Y_test.npy"

    if not x_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            f"Could not find test data at:\n  {x_test_path}\n  {y_test_path}\n\n"
            "Tip: run from the project root:\n  python -m dc1.stratified_baseline_evaluation"
        )

    if args.n_classes != len(CLASS_NAMES):
        raise ValueError(
            f"n_classes={args.n_classes}, but CLASS_NAMES has {len(CLASS_NAMES)} names. "
            "Please keep them consistent."
        )

    test_dataset = ImageDataset(x_test_path, y_test_path)
    test_sampler = BatchSampler(
        batch_size=args.batch_size,
        dataset=test_dataset,
        balanced=args.balanced_test,
    )

    weights_dir = resolve_path(args.weights_dir, base_dir / "model_weights", project_root)
    if args.model_path is None:
        model_path = find_latest_best_model(weights_dir)
        print(f"--model_path not provided. Using latest BEST model: {model_path}")
    else:
        model_path = resolve_path(args.model_path, base_dir / "model_weights", project_root)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Provide a valid --model_path or let the script auto-select the latest best_model.pt."
        )

    model = Net(n_classes=args.n_classes).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded model weights from: {model_path}")

    y_true, y_pred, y_prob, test_loss = run_inference_with_batch_sampler(
        model=model,
        sampler=test_sampler,
        device=device,
    )

    results = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        test_loss=test_loss,
        n_classes=args.n_classes,
        class_names=CLASS_NAMES,
    )

    print("\n=== Stratified Baseline Evaluation on Official TEST Set ===")
    print(f"Test Loss:     {results.test_loss:.4f}")
    print(f"Test Accuracy: {results.accuracy:.4f}")
    print(f"Test Macro-F1: {results.macro_f1:.4f}\n")
    print(results.class_report)

    cm = np.array(results.confusion_matrix, dtype=int)
    print("\nConfusion Matrix:")
    print(cm)

    now = datetime.now()
    out_dir = (base_dir / "artifacts") / f"stratified_baseline_eval_{now:%m_%d_%H_%M}"
    os.makedirs(out_dir, exist_ok=True)

    metrics_dict = results_to_dict(results)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_dict, indent=2),
        encoding="utf-8",
    )

    (out_dir / "classification_report.txt").write_text(
        results.class_report,
        encoding="utf-8",
    )

    summary_text = (
        "Stratified Baseline Model Evaluation\n"
        "===================================\n\n"
        f"Model path: {model_path}\n"
        f"Device: {device}\n"
        f"Test samples: {len(y_true)}\n"
        f"Classes: {args.n_classes}\n\n"
        f"Test loss: {results.test_loss:.4f}\n"
        f"Accuracy: {results.accuracy:.4f}\n"
        f"Macro-F1: {results.macro_f1:.4f}\n"
    )
    (out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    save_confusion_matrix_png(
        cm=cm,
        out_path=out_dir / "confusion_matrix.png",
        class_names=CLASS_NAMES,
    )

    save_confusion_matrix_normalized_png(
        cm=cm,
        out_path=out_dir / "confusion_matrix_normalized.png",
        class_names=CLASS_NAMES,
    )

    thresholds = np.linspace(0.0, 0.99, 50)
    coverage, acc_t, mf1_t = threshold_sweep(
        y_true=y_true,
        y_prob=y_prob,
        thresholds=thresholds,
    )

    print("\n=== Threshold sweep (coverage / accuracy / macro-F1 on kept samples) ===")
    for i in range(0, len(thresholds), 10):
        print(
            f"t={thresholds[i]:.2f} | "
            f"coverage={coverage[i]:.3f} | "
            f"acc={acc_t[i]:.3f} | "
            f"macroF1={mf1_t[i]:.3f}"
        )

    save_coverage_curve_png(
        coverage=coverage,
        macro_f1=mf1_t,
        acc=acc_t,
        out_path=out_dir / "threshold_coverage_curve.png",
    )

    print(f"\nSaved evaluation artifacts to: {out_dir}")


if __name__ == "__main__":
    main()