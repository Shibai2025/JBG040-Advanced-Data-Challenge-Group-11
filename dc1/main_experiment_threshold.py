from __future__ import annotations

import argparse
import json
import random
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


CLASS_NAMES = [
    "Atelectasis",
    "Effusion",
    "Infiltration",
    "No Finding",
    "Nodule",
    "Pneumothorax",
]

EXPERIMENT_SETTINGS = {
    "balanced_batch": {
        "source_experiment_group": "experiment_balanced_batch_adamw",
        "source_experiment_name": "baseline_balanced_batch_adamw",
        "threshold_experiment_group": "balanced_batch_adamw",
        "threshold_experiment_name": "balanced_batch_adamw_threshold",
        "imbalance_strategy": "balanced_batch",
        "optimizer": "AdamW",
    },
    "severity_weighted_loss": {
        "source_experiment_group": "experiment_severity_weighted_loss_adamw",
        "source_experiment_name": "baseline_severity_weighted_loss_adamw",
        "threshold_experiment_group": "severity_weighted_loss_adamw",
        "threshold_experiment_name": "severity_weighted_loss_adamw_threshold",
        "imbalance_strategy": "severity_weighted_loss",
        "optimizer": "AdamW",
    },
}


@dataclass
class DatasetSubset:
    dataset: ImageDataset
    indices: np.ndarray

    def __post_init__(self) -> None:
        self.indices = np.asarray(self.indices, dtype=int)
        self.targets = np.asarray(self.dataset.targets)[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[int(self.indices[idx])]


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def choose_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dir(raw_dir: str, default_base: Path) -> Path:
    path = Path(raw_dir).expanduser()
    if not path.is_absolute():
        path = default_base / path
    return path.resolve()


def save_json(data: Dict[str, object], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_text(lines: List[str], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def class_distribution(y: np.ndarray) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    y = np.asarray(y).astype(int)
    for class_id, class_name in enumerate(CLASS_NAMES):
        counts[class_name] = int(np.sum(y == class_id))
    return counts


def stratified_split_indices(
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y).reshape(-1).astype(int)

    train_indices: List[int] = []
    val_indices: List[int] = []

    for class_id in np.unique(y):
        class_indices = np.where(y == class_id)[0]
        shuffled = rng.permutation(class_indices)

        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        val_cls = shuffled[:n_val]
        train_cls = shuffled[n_val:]

        if len(train_cls) == 0:
            train_cls = val_cls[:1]
            val_cls = val_cls[1:]

        train_indices.extend(train_cls.tolist())
        val_indices.extend(val_cls.tolist())

    train_indices = rng.permutation(np.array(train_indices, dtype=int))
    val_indices = rng.permutation(np.array(val_indices, dtype=int))
    return train_indices, val_indices


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    sampler: BatchSampler,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    y_true_all: List[int] = []
    y_prob_all: List[np.ndarray] = []

    for xb, yb in sampler:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        probs = torch.softmax(logits, dim=1)

        y_true_all.extend(yb.detach().cpu().numpy().tolist())
        y_prob_all.append(probs.detach().cpu().numpy())

    y_true = np.array(y_true_all, dtype=int)
    y_prob = np.concatenate(y_prob_all, axis=0) if len(y_prob_all) > 0 else np.zeros((0, 0), dtype=float)
    return y_true, y_prob


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

    stats: List[Dict[str, int]] = []
    total = int(cm.sum())

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


def compute_macro_f1(cm: np.ndarray) -> float:
    per_class = compute_per_class_metrics(cm)
    f1s = [item["f1"] for item in per_class]
    return float(np.mean(f1s)) if len(f1s) else 0.0


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> List[Dict[str, object]]:
    """
    Selective prediction based on confidence threshold:
      keep sample if max_prob >= threshold

    For each threshold, compute:
    - coverage
    - kept accuracy
    - kept macro-F1
    - number of kept samples
    """
    results: List[Dict[str, object]] = []

    if len(y_true) == 0 or y_prob.size == 0:
        for t in thresholds:
            results.append(
                {
                    "threshold": float(t),
                    "coverage": 0.0,
                    "kept_samples": 0,
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                }
            )
        return results

    y_pred = np.argmax(y_prob, axis=1)
    max_prob = np.max(y_prob, axis=1)
    n_classes = y_prob.shape[1]
    total_samples = len(y_true)

    for t in thresholds:
        keep = max_prob >= t
        kept = int(np.sum(keep))
        coverage = kept / total_samples

        if kept == 0:
            results.append(
                {
                    "threshold": float(t),
                    "coverage": float(coverage),
                    "kept_samples": kept,
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                }
            )
            continue

        yt = y_true[keep]
        yp = y_pred[keep]
        cm = compute_confusion_matrix(yt, yp, n_classes)

        results.append(
            {
                "threshold": float(t),
                "coverage": float(coverage),
                "kept_samples": kept,
                "accuracy": float(compute_accuracy(yt, yp)),
                "macro_f1": float(compute_macro_f1(cm)),
            }
        )

    return results


def select_best_threshold(
    threshold_results: List[Dict[str, object]],
    min_coverage: float = 0.5,
) -> Dict[str, object]:
    eligible = [
        item for item in threshold_results
        if float(item["coverage"]) >= min_coverage
    ]

    if not eligible:
        eligible = threshold_results

    best_item = max(
        eligible,
        key=lambda x: (
            float(x["macro_f1"]),
            float(x["accuracy"]),
            float(x["coverage"]),
            -float(x["threshold"]),
        ),
    )
    return best_item


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    class_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    if len(y_true) == 0 or y_prob.size == 0:
        return {
            "threshold": float(threshold),
            "coverage": 0.0,
            "kept_samples": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "confusion_matrix": [],
            "per_class_metrics": [],
            "per_class_stats": [],
        }

    y_pred = np.argmax(y_prob, axis=1)
    max_prob = np.max(y_prob, axis=1)

    keep = max_prob >= threshold
    kept = int(np.sum(keep))
    coverage = kept / len(y_true)

    if kept == 0:
        return {
            "threshold": float(threshold),
            "coverage": float(coverage),
            "kept_samples": kept,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "confusion_matrix": [],
            "per_class_metrics": [],
            "per_class_stats": [],
        }

    yt = y_true[keep]
    yp = y_pred[keep]
    n_classes = y_prob.shape[1]
    cm = compute_confusion_matrix(yt, yp, n_classes)

    return {
        "threshold": float(threshold),
        "coverage": float(coverage),
        "kept_samples": kept,
        "accuracy": float(compute_accuracy(yt, yp)),
        "macro_f1": float(compute_macro_f1(cm)),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": compute_per_class_metrics(cm, class_names),
        "per_class_stats": compute_per_class_stats(cm, class_names),
    }


def save_threshold_curve(
    threshold_results: List[Dict[str, object]],
    out_path: Path,
    title: str,
) -> None:
    thresholds = [float(item["threshold"]) for item in threshold_results]
    macro_f1 = [float(item["macro_f1"]) for item in threshold_results]
    accuracy = [float(item["accuracy"]) for item in threshold_results]
    coverage = [float(item["coverage"]) for item in threshold_results]

    fig, ax1 = plt.subplots(figsize=(8, 5.5), dpi=160)

    ax1.plot(thresholds, macro_f1, label="Macro-F1", color="#1f77b4", linewidth=2)
    ax1.plot(thresholds, accuracy, label="Accuracy", color="#ff7f0e", linewidth=2)
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Score")
    ax1.set_title(title)
    ax1.grid(alpha=0.25, linestyle="--")

    ax2 = ax1.twinx()
    ax2.plot(thresholds, coverage, label="Coverage", color="#2ca02c", linestyle="--", linewidth=2)
    ax2.set_ylabel("Coverage")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def ensure_run_directories(experiment_root: Path, experiment_name: str, run_id: str) -> Dict[str, Path]:
    artifacts_root = experiment_root / "artifacts"
    weights_root = experiment_root / "model_weights"

    best_artifacts_dir = artifacts_root / f"{experiment_name}_best" / run_id
    final_artifacts_dir = artifacts_root / f"{experiment_name}_final" / run_id
    best_weights_dir = weights_root / f"{experiment_name}_best" / run_id
    final_weights_dir = weights_root / f"{experiment_name}_final" / run_id

    for path in [best_artifacts_dir, final_artifacts_dir, best_weights_dir, final_weights_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "best_artifacts_dir": best_artifacts_dir,
        "final_artifacts_dir": final_artifacts_dir,
        "best_weights_dir": best_weights_dir,
        "final_weights_dir": final_weights_dir,
    }


def find_latest_run_dir(root_dir: Path) -> Path:
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {root_dir}")

    candidates = [p for p in root_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found in: {root_dir}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_source_best_model_path(
    base_dir: Path,
    source_experiment_group: str,
    source_experiment_name: str,
    explicit_model_path: Optional[str] = None,
) -> Path:
    if explicit_model_path is not None:
        model_path = Path(explicit_model_path).expanduser()
        if not model_path.is_absolute():
            model_path = (base_dir / model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Explicit model path not found: {model_path}")
        return model_path

    best_root = (
        base_dir
        / "experiments"
        / "experiment_imbalance"
        / source_experiment_group
        / "model_weights"
        / f"{source_experiment_name}_best"
    )

    latest_run = find_latest_run_dir(best_root)
    model_path = latest_run / "best_model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Could not find best model checkpoint at: {model_path}")

    return model_path


def load_model(model_path: Path, n_classes: int, device: str) -> Net:
    model = Net(n_classes=n_classes).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_single_threshold_experiment(
    args: argparse.Namespace,
    base_dir: Path,
    dataset: ImageDataset,
    y_all: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    setting_key: str,
    run_id: str,
    device: str,
) -> Dict[str, object]:
    setting = EXPERIMENT_SETTINGS[setting_key]

    val_dataset = DatasetSubset(dataset, val_idx)
    val_sampler = BatchSampler(
        batch_size=args.batch_size,
        dataset=val_dataset,
        balanced=False,
    )

    source_model_path = resolve_source_best_model_path(
        base_dir=base_dir,
        source_experiment_group=setting["source_experiment_group"],
        source_experiment_name=setting["source_experiment_name"],
        explicit_model_path=getattr(args, f"{setting_key}_model_path", None),
    )

    model = load_model(source_model_path, n_classes=args.n_classes, device=device)

    y_true, y_prob = run_inference(model=model, sampler=val_sampler, device=device)

    thresholds = np.arange(args.threshold_min, args.threshold_max + 1e-9, args.threshold_step)
    threshold_results = threshold_sweep(y_true=y_true, y_prob=y_prob, thresholds=thresholds)
    best_threshold_result = select_best_threshold(
        threshold_results=threshold_results,
        min_coverage=args.min_coverage,
    )
    best_threshold = float(best_threshold_result["threshold"])

    detailed_best_metrics = compute_metrics_at_threshold(
        y_true=y_true,
        y_prob=y_prob,
        threshold=best_threshold,
        class_names=CLASS_NAMES,
    )
    final_threshold = float(args.final_threshold) if args.final_threshold is not None else best_threshold
    detailed_final_metrics = compute_metrics_at_threshold(
        y_true=y_true,
        y_prob=y_prob,
        threshold=final_threshold,
        class_names=CLASS_NAMES,
    )

    experiment_root = base_dir / "experiments" / "experiment_threshold" / setting["threshold_experiment_group"]
    experiment_name = setting["threshold_experiment_name"]
    dirs = ensure_run_directories(experiment_root, experiment_name, run_id)

    config = {
        "run_id": run_id,
        "setting_key": setting_key,
        "optimizer": setting["optimizer"],
        "imbalance_strategy": setting["imbalance_strategy"],
        "source_experiment_group": setting["source_experiment_group"],
        "source_experiment_name": setting["source_experiment_name"],
        "source_model_path": str(source_model_path),
        "threshold_experiment_group": setting["threshold_experiment_group"],
        "threshold_experiment_name": experiment_name,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "batch_size": args.batch_size,
        "n_classes": args.n_classes,
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "threshold_step": args.threshold_step,
        "min_coverage": args.min_coverage,
        "final_threshold": final_threshold,
        "device": device,
        "train_distribution": class_distribution(y_all[train_idx]),
        "validation_distribution": class_distribution(y_all[val_idx]),
    }

    threshold_results_payload = {
        "run_id": run_id,
        "setting_key": setting_key,
        "source_model_path": str(source_model_path),
        "best_threshold_selection_rule": "max validation macro-F1 with minimum coverage constraint",
        "best_threshold_result": best_threshold_result,
        "all_threshold_results": threshold_results,
    }

    best_threshold_config = {
        "selected_model_type": "best_threshold",
        "run_id": run_id,
        "setting_key": setting_key,
        "source_model_path": str(source_model_path),
        "selected_threshold": best_threshold,
        "selection_metric": "validation_macro_f1",
        "minimum_coverage_constraint": args.min_coverage,
        "metrics": detailed_best_metrics,
    }

    final_threshold_config = {
        "selected_model_type": "final_threshold",
        "run_id": run_id,
        "setting_key": setting_key,
        "source_model_path": str(source_model_path),
        "selected_threshold": final_threshold,
        "selection_metric": "user_defined_or_best_threshold",
        "metrics": detailed_final_metrics,
    }

    save_json(config, dirs["best_artifacts_dir"] / "config.json")
    save_json(config, dirs["final_artifacts_dir"] / "config.json")

    save_json(threshold_results_payload, dirs["best_artifacts_dir"] / "threshold_results.json")
    save_json(threshold_results_payload, dirs["final_artifacts_dir"] / "threshold_results.json")

    save_json(
        {
            "threshold": best_threshold,
            "per_class_metrics": detailed_best_metrics["per_class_metrics"],
            "per_class_stats": detailed_best_metrics["per_class_stats"],
            "confusion_matrix": detailed_best_metrics["confusion_matrix"],
        },
        dirs["best_artifacts_dir"] / "per_class_threshold_report.json",
    )
    save_json(
        {
            "threshold": final_threshold,
            "per_class_metrics": detailed_final_metrics["per_class_metrics"],
            "per_class_stats": detailed_final_metrics["per_class_stats"],
            "confusion_matrix": detailed_final_metrics["confusion_matrix"],
        },
        dirs["final_artifacts_dir"] / "per_class_threshold_report.json",
    )

    save_json(best_threshold_config, dirs["best_weights_dir"] / "best_threshold_config.json")
    save_json(final_threshold_config, dirs["final_weights_dir"] / "final_threshold_config.json")

    save_threshold_curve(
        threshold_results=threshold_results,
        out_path=dirs["best_artifacts_dir"] / "threshold_curve.png",
        title=f"Threshold Sweep ({setting['threshold_experiment_group']})",
    )
    save_threshold_curve(
        threshold_results=threshold_results,
        out_path=dirs["final_artifacts_dir"] / "threshold_curve.png",
        title=f"Threshold Sweep ({setting['threshold_experiment_group']})",
    )

    best_summary_lines = [
        f"Threshold experiment group: {setting['threshold_experiment_group']}",
        f"Threshold experiment name: {experiment_name}",
        f"Run ID: {run_id}",
        f"Optimizer: {setting['optimizer']}",
        f"Imbalance strategy: {setting['imbalance_strategy']}",
        f"Source model path: {source_model_path}",
        f"Selected best threshold: {best_threshold:.4f}",
        f"Minimum coverage constraint: {args.min_coverage:.4f}",
        f"Validation coverage at best threshold: {float(detailed_best_metrics['coverage']):.6f}",
        f"Validation kept samples at best threshold: {int(detailed_best_metrics['kept_samples'])}",
        f"Validation accuracy at best threshold: {float(detailed_best_metrics['accuracy']):.6f}",
        f"Validation macro-F1 at best threshold: {float(detailed_best_metrics['macro_f1']):.6f}",
        f"Best artifacts dir: {dirs['best_artifacts_dir']}",
        f"Best threshold config path: {dirs['best_weights_dir'] / 'best_threshold_config.json'}",
    ]

    final_summary_lines = [
        f"Threshold experiment group: {setting['threshold_experiment_group']}",
        f"Threshold experiment name: {experiment_name}",
        f"Run ID: {run_id}",
        f"Optimizer: {setting['optimizer']}",
        f"Imbalance strategy: {setting['imbalance_strategy']}",
        f"Source model path: {source_model_path}",
        f"Selected final threshold: {final_threshold:.4f}",
        f"Validation coverage at final threshold: {float(detailed_final_metrics['coverage']):.6f}",
        f"Validation kept samples at final threshold: {int(detailed_final_metrics['kept_samples'])}",
        f"Validation accuracy at final threshold: {float(detailed_final_metrics['accuracy']):.6f}",
        f"Validation macro-F1 at final threshold: {float(detailed_final_metrics['macro_f1']):.6f}",
        f"Final artifacts dir: {dirs['final_artifacts_dir']}",
        f"Final threshold config path: {dirs['final_weights_dir'] / 'final_threshold_config.json'}",
    ]

    save_text(best_summary_lines, dirs["best_artifacts_dir"] / "summary.txt")
    save_text(final_summary_lines, dirs["final_artifacts_dir"] / "summary.txt")

    print(f"\nFinished threshold experiment: {setting_key}")
    print(f"Source checkpoint: {source_model_path}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Best validation macro-F1: {float(detailed_best_metrics['macro_f1']):.6f}")
    print(f"Best artifacts dir: {dirs['best_artifacts_dir']}")

    return {
        "setting_key": setting_key,
        "threshold_experiment_group": setting["threshold_experiment_group"],
        "threshold_experiment_name": experiment_name,
        "source_model_path": str(source_model_path),
        "best_threshold": best_threshold,
        "best_validation_coverage": float(detailed_best_metrics["coverage"]),
        "best_validation_accuracy": float(detailed_best_metrics["accuracy"]),
        "best_validation_macro_f1": float(detailed_best_metrics["macro_f1"]),
        "best_artifacts_dir": str(dirs["best_artifacts_dir"]),
        "best_threshold_config_path": str(dirs["best_weights_dir"] / "best_threshold_config.json"),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select confidence thresholds based on the AdamW optimizer and imbalance experiments."
    )

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--x_file", type=str, default="X_train.npy")
    parser.add_argument("--y_file", type=str, default="Y_train.npy")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--n_classes", type=int, default=6)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    parser.add_argument("--threshold_min", type=float, default=0.10)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    parser.add_argument("--min_coverage", type=float, default=0.50)
    parser.add_argument(
        "--final_threshold",
        type=float,
        default=None,
        help="Optional manual final threshold. If omitted, the selected best threshold is reused.",
    )

    parser.add_argument(
        "--settings",
        nargs="+",
        default=["balanced_batch", "severity_weighted_loss"],
        choices=list(EXPERIMENT_SETTINGS.keys()),
        help="Which threshold experiments to run.",
    )

    parser.add_argument(
        "--balanced_batch_model_path",
        type=str,
        default=None,
        help="Optional explicit checkpoint path for the balanced_batch AdamW model.",
    )
    parser.add_argument(
        "--severity_weighted_loss_model_path",
        type=str,
        default=None,
        help="Optional explicit checkpoint path for the severity_weighted_loss AdamW model.",
    )

    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--non_deterministic", action="store_true")

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    set_seed(args.seed, deterministic=not args.non_deterministic)
    device = choose_device(force_cpu=args.force_cpu)

    base_dir = Path(__file__).resolve().parent
    data_dir = resolve_dir(args.data_dir, base_dir)

    x_path = (data_dir / args.x_file).resolve()
    y_path = (data_dir / args.y_file).resolve()

    if not x_path.exists():
        raise FileNotFoundError(f"Could not find input data file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Could not find label data file: {y_path}")

    dataset = ImageDataset(str(x_path), str(y_path))
    y_all = np.asarray(dataset.targets).astype(int)

    train_idx, val_idx = stratified_split_indices(
        y=y_all,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    print(f"Using device: {device}")
    print(f"Data path: {x_path}")
    print(f"Label path: {y_path}")
    print(f"Run ID: {run_id}")
    print(f"Validation size: {len(val_idx)}")

    results: List[Dict[str, object]] = []
    for setting_key in args.settings:
        result = run_single_threshold_experiment(
            args=args,
            base_dir=base_dir,
            dataset=dataset,
            y_all=y_all,
            train_idx=train_idx,
            val_idx=val_idx,
            setting_key=setting_key,
            run_id=run_id,
            device=device,
        )
        results.append(result)

    comparison_root = base_dir / "experiments" / "experiment_threshold" / "comparison" / run_id
    comparison_root.mkdir(parents=True, exist_ok=True)

    comparison_payload = {
        "run_id": run_id,
        "settings": args.settings,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "threshold_step": args.threshold_step,
        "min_coverage": args.min_coverage,
        "final_threshold": args.final_threshold,
        "results": results,
    }
    save_json(comparison_payload, comparison_root / "comparison_summary.json")

    comparison_lines = [
        "Threshold selection comparison finished.",
        f"Run ID: {run_id}",
        f"Settings: {', '.join(args.settings)}",
        "",
    ]
    for result in results:
        comparison_lines.extend(
            [
                f"Setting: {result['setting_key']}",
                f"  Best threshold: {result['best_threshold']}",
                f"  Best validation coverage: {result['best_validation_coverage']}",
                f"  Best validation accuracy: {result['best_validation_accuracy']}",
                f"  Best validation macro-F1: {result['best_validation_macro_f1']}",
                f"  Best artifacts dir: {result['best_artifacts_dir']}",
                "",
            ]
        )
    save_text(comparison_lines, comparison_root / "comparison_summary.txt")

    print("\nAll threshold experiments finished.")
    print(f"Comparison summary saved to: {comparison_root}")


if __name__ == "__main__":
    main()