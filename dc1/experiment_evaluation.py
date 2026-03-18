from __future__ import annotations

"""
experiment_evaluation.py

Evaluate experiment outputs on the official test set.

Supported candidate types:
1. model checkpoints:
   - best_model.pt
   - *best*.pt
   - *best*.pth

2. threshold configs:
   - best_threshold_config.json

Outputs:
- per-candidate evaluation folders under:
    dc1/artifacts/experiment_evaluation_result/eval_<timestamp>/<experiment_group>/<experiment_name>/<run_id>/
- aggregate_summary.json
- aggregate_summary.txt
- ranking.csv

Usage:
    python -m dc1.experiment_evaluation
    python -m dc1.experiment_evaluation --search_dirs dc1/experiments
"""

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net

try:
    from dc1.net_exp_architecture import NetExpArchitecture
except Exception:
    NetExpArchitecture = None  # type: ignore


CLASS_NAMES = [
    "Atelectasis",
    "Effusion",
    "Infiltration",
    "No Finding",
    "Nodule",
    "Pneumothorax",
]

MODEL_PATTERNS = [
    "**/best_model.pt",
    "**/*best*.pt",
    "**/*best*.pth",
]

THRESHOLD_PATTERNS = [
    "**/best_threshold_config.json",
]

ARCHITECTURE_VARIANTS = ["baseline", "deeper", "kernel3", "kernel5"]


@dataclass
class EvaluationResults:
    average_loss: float
    accuracy: float
    macro_f1: float
    confusion_matrix: np.ndarray
    per_class_metrics: List[Dict[str, float]]
    per_class_stats: List[Dict[str, int]]
    selective_curve: List[Dict[str, float]]


@dataclass
class Candidate:
    candidate_type: str  # "model" or "threshold"
    path: Path
    label: str
    experiment_group: str
    experiment_name: str
    run_id: str
    architecture_hint: str
    source_model_path: Optional[Path] = None
    selected_threshold: Optional[float] = None


@dataclass
class AggregateRecord:
    rank: int
    candidate_type: str
    label: str
    source_path: str
    experiment_group: str
    experiment_name: str
    run_id: str
    architecture_hint: str
    average_loss: float
    accuracy: float
    macro_f1: float
    coverage: float
    kept_samples: int
    total_samples: int
    threshold: Optional[float]
    output_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate experiment checkpoints and threshold configs on the official test set."
    )
    parser.add_argument(
        "--search_dirs",
        nargs="+",
        default=None,
        help="Directories to scan recursively. Default: dc1/experiments",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing X_test.npy and Y_test.npy. Default: dc1/data",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root output dir. Default: dc1/artifacts/experiment_evaluation_result",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_classes", type=int, default=6)
    parser.add_argument("--balanced_test", action="store_true")
    parser.add_argument("--force_cpu", action="store_true")
    return parser.parse_args()


def choose_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_path(raw_path: Optional[str], default_path: Path, project_root: Path) -> Path:
    if raw_path is None:
        return default_path.resolve()
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def resolve_paths(raw_paths: Optional[Sequence[str]], default_paths: Sequence[Path], project_root: Path) -> List[Path]:
    if raw_paths is None:
        return [path.resolve() for path in default_paths]
    results: List[Path] = []
    for raw in raw_paths:
        p = Path(raw).expanduser()
        if p.is_absolute():
            results.append(p.resolve())
        else:
            results.append((project_root / p).resolve())
    return results


def load_dataset(data_dir: Path) -> ImageDataset:
    x_test = data_dir / "X_test.npy"
    y_test = data_dir / "Y_test.npy"
    if not x_test.is_file():
        raise FileNotFoundError(f"Missing test array: {x_test}")
    if not y_test.is_file():
        raise FileNotFoundError(f"Missing test labels: {y_test}")
    return ImageDataset(str(x_test), str(y_test))


def safe_load_state_dict(checkpoint_path: Path, device: str):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else float(a / b)


def sanitize_label(text: str) -> str:
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    result = "".join(cleaned)
    while "__" in result:
        result = result.replace("__", "_")
    return result.strip("_") or "item"


def infer_architecture_hint(path: Path) -> str:
    joined = str(path).lower()
    for variant in ARCHITECTURE_VARIANTS:
        if variant in joined:
            return variant
    return "net"


def maybe_parse_experiment_metadata(path: Path) -> Tuple[str, str, str]:
    parts = list(path.parts)
    run_id = path.parent.name
    experiment_group = "unknown_group"
    experiment_name = path.parent.parent.name if len(parts) >= 3 else path.stem

    if "experiments" in parts:
        exp_idx = parts.index("experiments")
        if exp_idx + 1 < len(parts):
            experiment_group = parts[exp_idx + 1]

        if "model_weights" in parts:
            mw_idx = parts.index("model_weights")
            if mw_idx - 1 >= 0:
                experiment_name = parts[mw_idx - 1]
        elif "artifacts" in parts:
            art_idx = parts.index("artifacts")
            if art_idx - 1 >= 0:
                experiment_name = parts[art_idx - 1]

    experiment_group = sanitize_label(experiment_group)
    experiment_name = sanitize_label(experiment_name)
    run_id = sanitize_label(run_id)
    return experiment_group, experiment_name, run_id


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def compute_per_class_metrics(cm: np.ndarray, class_names: Sequence[str]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for i in range(cm.shape[0]):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

        rows.append(
            {
                "class_index": i,
                "class_name": class_names[i],
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
            }
        )
    return rows


def compute_per_class_stats(cm: np.ndarray, class_names: Sequence[str]) -> List[Dict[str, int]]:
    total = int(cm.sum())
    rows: List[Dict[str, int]] = []
    for i in range(cm.shape[0]):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        rows.append(
            {
                "class_index": i,
                "class_name": class_names[i],
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
            }
        )
    return rows


def compute_macro_f1_from_cm(cm: np.ndarray, class_names: Sequence[str]) -> float:
    rows = compute_per_class_metrics(cm, class_names)
    return float(np.mean([r["f1"] for r in rows])) if rows else 0.0


def selective_prediction_curve(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_classes: int,
    class_names: Sequence[str],
) -> List[Dict[str, float]]:
    if probs.size == 0:
        return []

    max_scores = probs.max(axis=1)
    y_pred = probs.argmax(axis=1)
    thresholds = np.linspace(0.0, 0.99, 50)
    rows: List[Dict[str, float]] = []

    for threshold in thresholds:
        keep_mask = max_scores >= threshold
        coverage = float(np.mean(keep_mask)) if keep_mask.size else 0.0

        if np.sum(keep_mask) == 0:
            rows.append(
                {
                    "threshold": float(threshold),
                    "coverage": coverage,
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                    "kept_samples": 0,
                }
            )
            continue

        kept_true = y_true[keep_mask]
        kept_pred = y_pred[keep_mask]
        cm = compute_confusion_matrix(kept_true, kept_pred, n_classes)

        rows.append(
            {
                "threshold": float(threshold),
                "coverage": coverage,
                "accuracy": compute_accuracy(kept_true, kept_pred),
                "macro_f1": compute_macro_f1_from_cm(cm, class_names),
                "kept_samples": int(np.sum(keep_mask)),
            }
        )
    return rows


def save_json(data: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_txt(lines: Sequence[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], output_path: Path, normalize: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = cm.astype(float)
    title = "Confusion Matrix"
    fmt = "d"

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
        title = "Normalized Confusion Matrix"
        fmt = ".2f"

    fig, ax = plt.subplots(figsize=(8.8, 7.2), dpi=150)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)

    threshold = matrix.max() * 0.5 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = format(int(value), fmt) if fmt == "d" else format(value, fmt)
            ax.text(j, i, text, ha="center", va="center", color="white" if value > threshold else "black", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

def plot_selective_curve(selective_curve: Sequence[Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not selective_curve:
        return

    coverage = [r["coverage"] for r in selective_curve]
    accuracy = [r["accuracy"] for r in selective_curve]
    macro_f1 = [r["macro_f1"] for r in selective_curve]

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
    ax.plot(coverage, macro_f1, label="Macro-F1 (kept)", linewidth=2.4)
    ax.plot(coverage, accuracy, label="Accuracy (kept)", linewidth=2.4)
    ax.invert_xaxis()
    ax.set_xlabel("Coverage (fraction kept)")
    ax.set_ylabel("Score")
    ax.set_title("Selective Prediction: Coverage vs Score")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

def save_classification_report(
    per_class_metrics: Sequence[Dict[str, float]],
    per_class_stats: Sequence[Dict[str, int]],
    output_path: Path,
) -> None:
    lines: List[str] = []
    header = (
        f"{'Class':<18} {'Precision':>10} {'Recall':>10} {'F1':>10} "
        f"{'Support':>10} {'TP':>8} {'FP':>8} {'FN':>8} {'TN':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for m, s in zip(per_class_metrics, per_class_stats):
        lines.append(
            f"{m['class_name']:<18} "
            f"{m['precision']:>10.3f} "
            f"{m['recall']:>10.3f} "
            f"{m['f1']:>10.3f} "
            f"{m['support']:>10d} "
            f"{s['TP']:>8d} "
            f"{s['FP']:>8d} "
            f"{s['FN']:>8d} "
            f"{s['TN']:>8d}"
        )

    macro_precision = float(np.mean([r["precision"] for r in per_class_metrics])) if per_class_metrics else 0.0
    macro_recall = float(np.mean([r["recall"] for r in per_class_metrics])) if per_class_metrics else 0.0
    macro_f1 = float(np.mean([r["f1"] for r in per_class_metrics])) if per_class_metrics else 0.0
    total_support = int(np.sum([r["support"] for r in per_class_metrics])) if per_class_metrics else 0

    lines.append("-" * len(header))
    lines.append(
        f"{'MacroAvg':<18} "
        f"{macro_precision:>10.3f} "
        f"{macro_recall:>10.3f} "
        f"{macro_f1:>10.3f} "
        f"{total_support:>10d}"
    )
    save_txt(lines, output_path)


def build_model_from_state_dict(
    architecture_hint: str,
    n_classes: int,
    device: str,
    state_dict,
) -> nn.Module:
    model = Net(n_classes=n_classes).to(device)
    try:
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception:
        pass

    if NetExpArchitecture is not None:
        variants_to_try: List[str] = []
        if architecture_hint in ARCHITECTURE_VARIANTS:
            variants_to_try.append(architecture_hint)
        for variant in ARCHITECTURE_VARIANTS:
            if variant not in variants_to_try:
                variants_to_try.append(variant)

        for variant in variants_to_try:
            model = NetExpArchitecture(n_classes=n_classes, variant=variant).to(device)
            try:
                model.load_state_dict(state_dict)
                model.eval()
                return model
            except Exception:
                continue

    raise RuntimeError("Could not load checkpoint with supported architectures.")


def run_model_forward(
    model: nn.Module,
    sampler: BatchSampler,
    loss_function: nn.Module,
    device: str,
) -> Tuple[List[float], np.ndarray, np.ndarray]:
    all_losses: List[float] = []
    all_true: List[np.ndarray] = []
    all_prob: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in sampler:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1)
            loss = loss_function(logits, y_batch)

            all_losses.append(float(loss.detach().cpu().item()))
            all_true.append(y_batch.detach().cpu().numpy())
            all_prob.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=int)
    probs = np.concatenate(all_prob) if all_prob else np.empty((0, len(CLASS_NAMES)), dtype=float)
    return all_losses, y_true, probs


def evaluate_full_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_classes: int,
) -> EvaluationResults:
    y_pred = probs.argmax(axis=1) if probs.size else np.array([], dtype=int)
    cm = compute_confusion_matrix(y_true, y_pred, n_classes)
    per_class_metrics = compute_per_class_metrics(cm, CLASS_NAMES)
    per_class_stats = compute_per_class_stats(cm, CLASS_NAMES)
    macro_f1 = compute_macro_f1_from_cm(cm, CLASS_NAMES)
    selective_curve = selective_prediction_curve(probs, y_true, n_classes, CLASS_NAMES)

    return EvaluationResults(
        average_loss=0.0,
        accuracy=compute_accuracy(y_true, y_pred),
        macro_f1=macro_f1,
        confusion_matrix=cm,
        per_class_metrics=per_class_metrics,
        per_class_stats=per_class_stats,
        selective_curve=selective_curve,
    )


def evaluate_threshold_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_classes: int,
    selected_threshold: float,
) -> Tuple[EvaluationResults, float, int]:
    if probs.size == 0:
        empty_cm = np.zeros((n_classes, n_classes), dtype=int)
        return EvaluationResults(
            average_loss=0.0,
            accuracy=0.0,
            macro_f1=0.0,
            confusion_matrix=empty_cm,
            per_class_metrics=compute_per_class_metrics(empty_cm, CLASS_NAMES),
            per_class_stats=compute_per_class_stats(empty_cm, CLASS_NAMES),
            selective_curve=[],
        ), 0.0, 0

    max_scores = probs.max(axis=1)
    y_pred = probs.argmax(axis=1)
    keep_mask = max_scores >= selected_threshold

    coverage = float(np.mean(keep_mask))
    kept_samples = int(np.sum(keep_mask))

    if kept_samples == 0:
        empty_cm = np.zeros((n_classes, n_classes), dtype=int)
        return EvaluationResults(
            average_loss=0.0,
            accuracy=0.0,
            macro_f1=0.0,
            confusion_matrix=empty_cm,
            per_class_metrics=compute_per_class_metrics(empty_cm, CLASS_NAMES),
            per_class_stats=compute_per_class_stats(empty_cm, CLASS_NAMES),
            selective_curve=selective_prediction_curve(probs, y_true, n_classes, CLASS_NAMES),
        ), coverage, kept_samples

    kept_true = y_true[keep_mask]
    kept_pred = y_pred[keep_mask]
    cm = compute_confusion_matrix(kept_true, kept_pred, n_classes)
    per_class_metrics = compute_per_class_metrics(cm, CLASS_NAMES)
    per_class_stats = compute_per_class_stats(cm, CLASS_NAMES)

    results = EvaluationResults(
        average_loss=0.0,
        accuracy=compute_accuracy(kept_true, kept_pred),
        macro_f1=compute_macro_f1_from_cm(cm, CLASS_NAMES),
        confusion_matrix=cm,
        per_class_metrics=per_class_metrics,
        per_class_stats=per_class_stats,
        selective_curve=selective_prediction_curve(probs, y_true, n_classes, CLASS_NAMES),
    )
    return results, coverage, kept_samples


def discover_model_candidates(search_dirs: Sequence[Path]) -> List[Candidate]:
    seen: set[Path] = set()
    items: List[Candidate] = []

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in MODEL_PATTERNS:
            for path in search_dir.glob(pattern):
                if not path.is_file():
                    continue
                if path in seen:
                    continue
                seen.add(path)

                experiment_group, experiment_name, run_id = maybe_parse_experiment_metadata(path)
                architecture_hint = infer_architecture_hint(path)
                label = sanitize_label(f"{experiment_group}_{experiment_name}_{run_id}")

                items.append(
                    Candidate(
                        candidate_type="model",
                        path=path.resolve(),
                        label=label,
                        experiment_group=experiment_group,
                        experiment_name=experiment_name,
                        run_id=run_id,
                        architecture_hint=architecture_hint,
                    )
                )
    return items


def read_threshold_config(config_path: Path) -> Tuple[Optional[Path], Optional[float]]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_model = data.get("source_model_path")
    selected_threshold = (
        data.get("selected_threshold")
        or data.get("best_threshold")
        or data.get("threshold")
    )

    source_model_path = Path(source_model).expanduser().resolve() if source_model else None
    threshold_value = float(selected_threshold) if selected_threshold is not None else None
    return source_model_path, threshold_value


def discover_threshold_candidates(search_dirs: Sequence[Path]) -> List[Candidate]:
    seen: set[Path] = set()
    items: List[Candidate] = []

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in THRESHOLD_PATTERNS:
            for path in search_dir.glob(pattern):
                if not path.is_file():
                    continue
                if path in seen:
                    continue
                seen.add(path)

                source_model_path, selected_threshold = read_threshold_config(path)
                if source_model_path is None or selected_threshold is None:
                    continue

                experiment_group, experiment_name, run_id = maybe_parse_experiment_metadata(path)
                architecture_hint = infer_architecture_hint(source_model_path)
                label = sanitize_label(f"{experiment_group}_{experiment_name}_{run_id}_threshold")

                items.append(
                    Candidate(
                        candidate_type="threshold",
                        path=path.resolve(),
                        label=label,
                        experiment_group=experiment_group,
                        experiment_name=experiment_name,
                        run_id=run_id,
                        architecture_hint=architecture_hint,
                        source_model_path=source_model_path,
                        selected_threshold=selected_threshold,
                    )
                )
    return items


def discover_candidates(search_dirs: Sequence[Path]) -> List[Candidate]:
    items = discover_model_candidates(search_dirs) + discover_threshold_candidates(search_dirs)
    items.sort(key=lambda x: (x.path.stat().st_mtime, str(x.path)), reverse=True)
    return items


def save_results_bundle(
    candidate: Candidate,
    results: EvaluationResults,
    out_dir: Path,
    device: str,
    n_classes: int,
    coverage: float,
    kept_samples: int,
    total_samples: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "candidate_type": candidate.candidate_type,
        "label": candidate.label,
        "source_path": str(candidate.path),
        "experiment_group": candidate.experiment_group,
        "experiment_name": candidate.experiment_name,
        "run_id": candidate.run_id,
        "architecture_hint": candidate.architecture_hint,
        "source_model_path": str(candidate.source_model_path) if candidate.source_model_path else None,
        "selected_threshold": candidate.selected_threshold,
        "device": device,
        "n_classes": n_classes,
        "total_samples": total_samples,
        "kept_samples": kept_samples,
        "coverage": coverage,
        "average_loss": results.average_loss,
        "accuracy": results.accuracy,
        "macro_f1": results.macro_f1,
        "confusion_matrix": results.confusion_matrix.tolist(),
        "per_class_metrics": results.per_class_metrics,
        "per_class_stats": results.per_class_stats,
        "selective_curve": results.selective_curve,
    }
    save_json(payload, out_dir / "metrics.json")

    lines = [
        "Experiment Evaluation",
        "=====================",
        "",
        f"Candidate type: {candidate.candidate_type}",
        f"Label: {candidate.label}",
        f"Source path: {candidate.path}",
        f"Experiment group: {candidate.experiment_group}",
        f"Experiment name: {candidate.experiment_name}",
        f"Run id: {candidate.run_id}",
        f"Architecture hint: {candidate.architecture_hint}",
        f"Source model path: {candidate.source_model_path}",
        f"Selected threshold: {candidate.selected_threshold}",
        f"Device: {device}",
        f"Total test samples: {total_samples}",
        f"Kept samples: {kept_samples}",
        f"Coverage: {coverage:.6f}",
        "",
        f"Average loss: {results.average_loss:.6f}",
        f"Accuracy: {results.accuracy:.6f}",
        f"Macro-F1: {results.macro_f1:.6f}",
    ]
    save_txt(lines, out_dir / "summary.txt")
    save_classification_report(results.per_class_metrics, results.per_class_stats, out_dir / "classification_report.txt")
    plot_confusion_matrix(results.confusion_matrix, CLASS_NAMES, out_dir / "confusion_matrix.png", normalize=False)
    plot_confusion_matrix(results.confusion_matrix, CLASS_NAMES, out_dir / "confusion_matrix_normalized.png", normalize=True)
    plot_selective_curve(results.selective_curve, out_dir / "threshold_coverage_curve.png")


def evaluate_candidate(
    candidate: Candidate,
    test_sampler: BatchSampler,
    device: str,
    n_classes: int,
    out_dir: Path,
) -> AggregateRecord:
    loss_function = nn.CrossEntropyLoss()

    if candidate.candidate_type == "model":
        state_dict = safe_load_state_dict(candidate.path, device)
        model = build_model_from_state_dict(candidate.architecture_hint, n_classes, device, state_dict)
        all_losses, y_true, probs = run_model_forward(model, test_sampler, loss_function, device)
        results = evaluate_full_predictions(y_true, probs, n_classes)
        results.average_loss = float(np.mean(all_losses)) if all_losses else 0.0
        coverage = 1.0
        kept_samples = int(len(y_true))
        total_samples = int(len(y_true))

    elif candidate.candidate_type == "threshold":
        if candidate.source_model_path is None or candidate.selected_threshold is None:
            raise ValueError(f"Invalid threshold candidate: {candidate.path}")

        state_dict = safe_load_state_dict(candidate.source_model_path, device)
        model = build_model_from_state_dict(candidate.architecture_hint, n_classes, device, state_dict)
        all_losses, y_true, probs = run_model_forward(model, test_sampler, loss_function, device)
        results, coverage, kept_samples = evaluate_threshold_predictions(
            y_true=y_true,
            probs=probs,
            n_classes=n_classes,
            selected_threshold=float(candidate.selected_threshold),
        )
        results.average_loss = float(np.mean(all_losses)) if all_losses else 0.0
        total_samples = int(len(y_true))

    else:
        raise ValueError(f"Unsupported candidate type: {candidate.candidate_type}")

    save_results_bundle(
        candidate=candidate,
        results=results,
        out_dir=out_dir,
        device=device,
        n_classes=n_classes,
        coverage=coverage,
        kept_samples=kept_samples,
        total_samples=total_samples,
    )

    return AggregateRecord(
        rank=0,
        candidate_type=candidate.candidate_type,
        label=candidate.label,
        source_path=str(candidate.path),
        experiment_group=candidate.experiment_group,
        experiment_name=candidate.experiment_name,
        run_id=candidate.run_id,
        architecture_hint=candidate.architecture_hint,
        average_loss=results.average_loss,
        accuracy=results.accuracy,
        macro_f1=results.macro_f1,
        coverage=coverage,
        kept_samples=kept_samples,
        total_samples=total_samples,
        threshold=candidate.selected_threshold,
        output_dir=str(out_dir),
    )


def rank_records(records: Sequence[AggregateRecord]) -> List[AggregateRecord]:
    ranked = sorted(
        records,
        key=lambda r: (-r.macro_f1, -r.accuracy, -r.coverage, r.average_loss, r.label),
    )

    output: List[AggregateRecord] = []
    for idx, r in enumerate(ranked, start=1):
        output.append(
            AggregateRecord(
                rank=idx,
                candidate_type=r.candidate_type,
                label=r.label,
                source_path=r.source_path,
                experiment_group=r.experiment_group,
                experiment_name=r.experiment_name,
                run_id=r.run_id,
                architecture_hint=r.architecture_hint,
                average_loss=r.average_loss,
                accuracy=r.accuracy,
                macro_f1=r.macro_f1,
                coverage=r.coverage,
                kept_samples=r.kept_samples,
                total_samples=r.total_samples,
                threshold=r.threshold,
                output_dir=r.output_dir,
            )
        )
    return output


def write_aggregate_outputs(records: Sequence[AggregateRecord], output_dir: Path) -> None:
    ranked = rank_records(records)

    save_json(
        {
            "ranked_items": [
                {
                    "rank": r.rank,
                    "candidate_type": r.candidate_type,
                    "label": r.label,
                    "source_path": r.source_path,
                    "experiment_group": r.experiment_group,
                    "experiment_name": r.experiment_name,
                    "run_id": r.run_id,
                    "architecture_hint": r.architecture_hint,
                    "average_loss": r.average_loss,
                    "accuracy": r.accuracy,
                    "macro_f1": r.macro_f1,
                    "coverage": r.coverage,
                    "kept_samples": r.kept_samples,
                    "total_samples": r.total_samples,
                    "threshold": r.threshold,
                    "output_dir": r.output_dir,
                }
                for r in ranked
            ]
        },
        output_dir / "aggregate_summary.json",
    )

    lines = [
        "Experiment Evaluation Aggregate Summary",
        "=======================================",
        "",
        f"Total evaluated items: {len(ranked)}",
        "",
        f"{'Rank':<6} {'Type':<10} {'Label':<40} {'Macro-F1':>10} {'Accuracy':>10} {'Coverage':>10}",
        "-" * 98,
    ]
    for r in ranked:
        lines.append(
            f"{r.rank:<6} "
            f"{r.candidate_type:<10} "
            f"{r.label:<40.40} "
            f"{r.macro_f1:>10.4f} "
            f"{r.accuracy:>10.4f} "
            f"{r.coverage:>10.4f}"
        )
    save_txt(lines, output_dir / "aggregate_summary.txt")


def write_ranking_csv(records: Sequence[AggregateRecord], output_path: Path) -> None:
    ranked = rank_records(records)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "candidate_type",
                "label",
                "experiment_group",
                "experiment_name",
                "run_id",
                "architecture_hint",
                "macro_f1",
                "accuracy",
                "coverage",
                "kept_samples",
                "total_samples",
                "average_loss",
                "threshold",
                "source_path",
                "output_dir",
            ]
        )
        for r in ranked:
            writer.writerow(
                [
                    r.rank,
                    r.candidate_type,
                    r.label,
                    r.experiment_group,
                    r.experiment_name,
                    r.run_id,
                    r.architecture_hint,
                    f"{r.macro_f1:.6f}",
                    f"{r.accuracy:.6f}",
                    f"{r.coverage:.6f}",
                    r.kept_samples,
                    r.total_samples,
                    f"{r.average_loss:.6f}",
                    "" if r.threshold is None else f"{r.threshold:.6f}",
                    r.source_path,
                    r.output_dir,
                ]
            )


def main() -> None:
    args = parse_args()

    if args.n_classes != len(CLASS_NAMES):
        raise ValueError(f"n_classes={args.n_classes}, but CLASS_NAMES has {len(CLASS_NAMES)} classes.")

    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent

    device = choose_device(force_cpu=args.force_cpu)
    print(f"Using device: {device}")

    default_search_dirs = [base_dir / "experiments"]
    search_dirs = resolve_paths(args.search_dirs, default_search_dirs, project_root)

    print("Search directories:")
    for folder in search_dirs:
        print(f"  - {folder}")

    data_dir = resolve_path(args.data_dir, base_dir / "data", project_root)
    output_root = resolve_path(
        args.output_root,
        base_dir / "artifacts" / "experiment_evaluation_result",
        project_root,
    )

    test_dataset = load_dataset(data_dir)
    test_sampler = BatchSampler(
        batch_size=args.batch_size,
        dataset=test_dataset,
        balanced=args.balanced_test,
    )

    candidates = discover_candidates(search_dirs)
    if not candidates:
        raise FileNotFoundError("No model checkpoints or threshold configs were found.")

    print(f"Discovered {len(candidates)} candidate(s).")
    for i, c in enumerate(candidates, start=1):
        print(f"[{i:02d}] {c.candidate_type}: {c.path}")

    aggregate_dir = output_root / datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    records: List[AggregateRecord] = []
    total = len(candidates)

    for idx, candidate in enumerate(candidates, start=1):
        run_dir = aggregate_dir / candidate.experiment_group / candidate.experiment_name / candidate.run_id
        if candidate.candidate_type == "threshold":
            run_dir = run_dir / "threshold_eval"
        else:
            run_dir = run_dir / "model_eval"

        print(f"\n[{idx}/{total}] Evaluating {candidate.candidate_type}: {candidate.path}")
        record = evaluate_candidate(
            candidate=candidate,
            test_sampler=test_sampler,
            device=device,
            n_classes=args.n_classes,
            out_dir=run_dir,
        )
        records.append(record)

    write_aggregate_outputs(records, aggregate_dir)
    write_ranking_csv(records, aggregate_dir / "ranking.csv")

    ranked = rank_records(records)
    print("\n=== Final Ranking ===")
    for r in ranked:
        print(
            f"#{r.rank} | {r.candidate_type} | {r.label} | "
            f"Macro-F1={r.macro_f1:.4f} | Acc={r.accuracy:.4f} | Coverage={r.coverage:.4f}"
        )

    print(f"\nSaved experiment evaluation results to: {aggregate_dir}")


if __name__ == "__main__":
    main()
