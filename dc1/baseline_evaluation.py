"""
baseline_evaluation.py

Evaluate a trained model on the test set using the project's BatchSampler.

UPDATED: --model_path is OPTIONAL.
- If --model_path is not provided, the script automatically loads the latest model file
  from --weights_dir (default: <project_root>/dc1/model_weights).

Recommended run (from project root):
  python -m dc1.baseline_evaluation

Or specify a model explicitly:
  python -m dc1.baseline_evaluation --model_path dc1/model_weights/model_03_02_10_15.txt

Outputs:
- Accuracy
- Macro-F1
- Per-class precision/recall/F1 (classification report)
- Confusion matrix (printed + saved as PNG)
- Threshold sweep: coverage vs macro-F1 / accuracy (printed + saved as PNG)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Package imports (recommended: run from project root)
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net


@dataclass
class EvalResults:
    accuracy: float
    macro_f1: float
    confusion_matrix: List[List[int]]
    class_report: str


def pick_device(force_cpu: bool = False) -> str:
    """Pick device similar to main.py logic."""
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_latest_model(weights_dir: Path) -> Path:
    """
    Find the most recently modified model file in weights_dir.
    Supports .txt (your current saving) and .pt.
    """
    candidates = list(weights_dir.glob("model_*.txt")) + list(weights_dir.glob("model_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No model files found in '{weights_dir}'.\n"
            "Run main.py first so it creates weights in model_weights/."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


@torch.no_grad()
def run_inference_with_batch_sampler(
    model: torch.nn.Module,
    sampler: BatchSampler,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      y_true: (N,)
      y_pred: (N,)
      y_prob: (N, C) softmax probabilities
    """
    model.eval()

    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_prob_all: List[np.ndarray] = []

    for xb, yb in sampler:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        probs = torch.softmax(logits, dim=1)

        pred = torch.argmax(probs, dim=1)

        y_true_all.extend(yb.detach().cpu().numpy().tolist())
        y_pred_all.extend(pred.detach().cpu().numpy().tolist())
        y_prob_all.append(probs.detach().cpu().numpy())

    y_true = np.array(y_true_all, dtype=int)
    y_pred = np.array(y_pred_all, dtype=int)
    y_prob = np.concatenate(y_prob_all, axis=0) if len(y_prob_all) > 0 else np.zeros((0, 0))

    return y_true, y_pred, y_prob


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else 0.0


def compute_classification_report(cm: np.ndarray, class_names: Optional[List[str]] = None) -> str:
    """
    Simple text report (precision/recall/f1 per class + macro avg).
    """
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    lines = []
    lines.append("Class\tPrecision\tRecall\tF1\tSupport")

    precisions = []
    recalls = []
    f1s = []
    supports = []

    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) != 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(int(support))

        lines.append(f"{class_names[i]}\t{precision:.3f}\t\t{recall:.3f}\t{f1:.3f}\t{support}")

    macro_p = float(np.mean(precisions)) if len(precisions) else 0.0
    macro_r = float(np.mean(recalls)) if len(recalls) else 0.0
    macro_f1 = float(np.mean(f1s)) if len(f1s) else 0.0
    total_support = int(np.sum(supports)) if len(supports) else 0

    lines.append("")
    lines.append(f"MacroAvg\t{macro_p:.3f}\t\t{macro_r:.3f}\t{macro_f1:.3f}\t{total_support}")
    return "\n".join(lines)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def compute_macro_f1(cm: np.ndarray) -> float:
    n_classes = cm.shape[0]
    f1s = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) != 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if len(f1s) else 0.0


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    class_names: Optional[List[str]] = None,
) -> EvalResults:
    cm = compute_confusion_matrix(y_true, y_pred, n_classes=n_classes)
    acc = compute_accuracy(y_true, y_pred)
    macro_f1 = compute_macro_f1(cm)
    report = compute_classification_report(cm, class_names=class_names)
    return EvalResults(
        accuracy=acc,
        macro_f1=macro_f1,
        confusion_matrix=cm.tolist(),
        class_report=report,
    )


def save_confusion_matrix_png(cm: np.ndarray, out_path: Path, class_names: Optional[List[str]] = None) -> None:
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(n_classes)
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    thresh = cm.max() * 0.6 if cm.size else 0
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Selective prediction:
      keep sample if max_prob >= threshold
    Compute coverage, accuracy, macro-F1 on kept samples.
    """
    n = len(y_true)
    if n == 0:
        return np.zeros_like(thresholds), np.zeros_like(thresholds), np.zeros_like(thresholds)

    y_pred = np.argmax(y_prob, axis=1)
    max_prob = np.max(y_prob, axis=1)

    coverages = []
    accs = []
    mf1s = []

    n_classes = y_prob.shape[1]

    for t in thresholds:
        keep = max_prob >= t
        kept = int(np.sum(keep))
        coverage = kept / n
        coverages.append(coverage)

        if kept == 0:
            accs.append(0.0)
            mf1s.append(0.0)
            continue

        yt = y_true[keep]
        yp = y_pred[keep]
        cm = compute_confusion_matrix(yt, yp, n_classes=n_classes)
        accs.append(compute_accuracy(yt, yp))
        mf1s.append(compute_macro_f1(cm))

    return np.array(coverages), np.array(accs), np.array(mf1s)


def save_coverage_curve_png(
    thresholds: np.ndarray,
    coverage: np.ndarray,
    macro_f1: np.ndarray,
    acc: np.ndarray,
    out_path: Path,
) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(coverage, macro_f1, label="Macro-F1 (kept)")
    plt.plot(coverage, acc, label="Accuracy (kept)")
    plt.gca().invert_xaxis()
    plt.xlabel("Coverage (fraction kept)")
    plt.ylabel("Score")
    plt.title("Selective Prediction: Coverage vs Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_class_names(s: Optional[str], n_classes: int) -> Optional[List[str]]:
    if s is None:
        return None
    names = [x.strip() for x in s.split(",")]
    if len(names) != n_classes:
        raise ValueError(f"--class_names must have exactly {n_classes} comma-separated names.")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline evaluation (BatchSampler version).")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional path to saved model state_dict. If not provided, the latest model in --weights_dir is used.",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=None,
        help="Directory containing saved models. If omitted, uses <project_root>/dc1/model_weights.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing X_test.npy and Y_test.npy. If omitted, uses <project_root>/dc1/data.",
    )
    parser.add_argument("--batch_size", type=int, default=100, help="Test batch size")
    parser.add_argument("--n_classes", type=int, default=6, help="Number of classes")
    parser.add_argument(
        "--class_names",
        type=str,
        default=None,
        help='Optional comma-separated class names, e.g. "A,B,C,D,E,F"',
    )
    parser.add_argument(
        "--balanced_test",
        action="store_true",
        help="Use balanced sampling on test set (normally keep this OFF).",
    )
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU even if CUDA/MPS is available")

    args = parser.parse_args()

    # Make all paths robust to where the script is launched from.
    base_dir = Path(__file__).resolve().parent  # .../dc1
    project_root = base_dir.parent              # project root

    def _resolve_path(p: Optional[str], default: Path) -> Path:
        if p is None:
            return default
        path = Path(p)
        if path.is_absolute():
            return path
        # Interpret relative paths as relative to project root for reproducibility
        return (project_root / path).resolve()

    device = pick_device(force_cpu=args.force_cpu)
    print(f"@@@ Using device: {device}")

    data_dir = _resolve_path(args.data_dir, base_dir / "data")
    x_test_path = data_dir / "X_test.npy"
    y_test_path = data_dir / "Y_test.npy"

    if not x_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            f"Could not find test data at:\n  {x_test_path}\n  {y_test_path}\n\n"
            "Tip: run from the project root:\n  python -m dc1.baseline_evaluation"
        )

    test_dataset = ImageDataset(x_test_path, y_test_path)
    test_sampler = BatchSampler(batch_size=args.batch_size, dataset=test_dataset, balanced=args.balanced_test)

    # Decide which model file to load
    weights_dir = _resolve_path(args.weights_dir, base_dir / "model_weights")
    if args.model_path is None:
        model_path = find_latest_model(weights_dir)
        print(f"@@@ --model_path not provided. Using latest model: {model_path}")
    else:
        model_path = _resolve_path(args.model_path, base_dir / "model_weights")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Tip: if you run from the project root with `python -m dc1.baseline_evaluation`, "
            "you can omit --model_path and it will auto-pick the latest model."
        )

    model = Net(n_classes=args.n_classes).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    print(f"@@@ Loaded model weights from: {model_path}")

    class_names = parse_class_names(args.class_names, args.n_classes)

    # Inference
    y_true, y_pred, y_prob = run_inference_with_batch_sampler(model, test_sampler, device=device)

    # Metrics
    results = compute_metrics(y_true, y_pred, n_classes=args.n_classes, class_names=class_names)

    print("\n=== Baseline evaluation results ===")
    print(f"Test Accuracy:  {results.accuracy:.4f}")
    print(f"Test Macro-F1:  {results.macro_f1:.4f}\n")
    print(results.class_report)

    cm = np.array(results.confusion_matrix, dtype=int)
    print("Confusion Matrix:\n", cm)

    # Output folder (always inside <project_root>/dc1/artifacts)
    now = datetime.now()
    out_dir = (base_dir / "artifacts") / f"baseline_eval_{now.month:02}_{now.day:02}_{now.hour:02}_{now.minute:02}"
    os.makedirs(out_dir, exist_ok=True)

    (out_dir / "classification_report.txt").write_text(results.class_report, encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(asdict(results), indent=2), encoding="utf-8")

    save_confusion_matrix_png(cm=cm, out_path=out_dir / "confusion_matrix.png", class_names=class_names)

    thresholds = np.linspace(0.0, 0.99, 50)
    coverage, acc_t, mf1_t = threshold_sweep(y_true=y_true, y_prob=y_prob, thresholds=thresholds)

    print("\n=== Threshold sweep (coverage / acc / macro-F1 on kept samples) ===")
    for t, cov, a, f in zip(thresholds[::10], coverage[::10], acc_t[::10], mf1_t[::10]):
        print(f"t={t:.2f} | coverage={cov:.3f} | acc={a:.3f} | macroF1={f:.3f}")

    save_coverage_curve_png(
        thresholds=thresholds,
        coverage=coverage,
        macro_f1=mf1_t,
        acc=acc_t,
        out_path=out_dir / "threshold_coverage_curve.png",
    )

    print(f"\n@@@ Saved evaluation artifacts to: {out_dir}")


if __name__ == "__main__":
    main()