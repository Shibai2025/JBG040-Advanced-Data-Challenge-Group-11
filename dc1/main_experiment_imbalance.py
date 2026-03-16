from __future__ import annotations

import argparse
import copy
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
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net
from dc1.train_test import train_model


CLASS_NAMES = [
    "Atelectasis",     # class 0
    "Effusion",        # class 1
    "Infiltration",    # class 2
    "No Finding",      # class 3
    "Nodule",          # class 4
    "Pneumothorax",    # class 5
]

SEVERITY_WEIGHT_SETS: Dict[str, List[float]] = {
    "run1": [2.0, 2.0, 2.0, 1.0, 1.5, 3.0],
    "run2": [2.5, 2.5, 2.5, 1.0, 1.25, 5.0],
    "run3": [3.0, 3.0, 2.5, 1.0, 1.5, 5.0],
}

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "experiments" / "experiment_imbalance"


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


def stratified_split_indices(y: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []

    for class_id in np.unique(y):
        class_indices = np.where(y == class_id)[0]
        rng.shuffle(class_indices)

        n_val = max(1, int(round(len(class_indices) * val_ratio)))
        n_val = min(n_val, len(class_indices) - 1) if len(class_indices) > 1 else 1

        val_parts.append(class_indices[:n_val])
        train_parts.append(class_indices[n_val:])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0))


def evaluate_model(
    model: nn.Module,
    sampler: BatchSampler,
    loss_function: nn.Module,
    device: str,
    n_classes: int,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    with torch.no_grad():
        for x, y in sampler:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_function(logits, y)
            preds = torch.argmax(logits, dim=1)

            losses.append(float(loss.detach().cpu().item()))
            all_true.append(y.detach().cpu().numpy())
            all_pred.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=int)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=int)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": compute_accuracy(y_true, y_pred),
        "macro_f1": compute_macro_f1(y_true, y_pred, n_classes=n_classes),
    }


def save_json(payload: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_text(lines: Sequence[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_loss_plot(train_losses: List[float], val_losses: List[float], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train loss")
    ax.plot(range(1, len(val_losses) + 1), val_losses, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_metric_plot(values: List[float], ylabel: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(range(1, len(values) + 1), values)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def class_distribution(y: np.ndarray) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    y = np.asarray(y).astype(int)
    for class_id, class_name in enumerate(CLASS_NAMES):
        counts[class_name] = int(np.sum(y == class_id))
    return counts


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


def get_severity_weights(weight_set_name: str, n_classes: int, device: str) -> torch.Tensor:
    if weight_set_name not in SEVERITY_WEIGHT_SETS:
        valid = ", ".join(SEVERITY_WEIGHT_SETS.keys())
        raise ValueError(f"Unknown severity_weight_set='{weight_set_name}'. Valid choices: {valid}")

    weights = SEVERITY_WEIGHT_SETS[weight_set_name]
    if len(weights) != n_classes:
        raise ValueError(f"Expected {n_classes} class weights, got {len(weights)}")

    return torch.tensor(weights, dtype=torch.float32, device=device)


def create_experiment_name(strategy: str) -> str:
    mapping = {
        "none": "baseline_no_imbalance_adamw",
        "balanced_batch": "baseline_balanced_batch_adamw",
        "severity_weighted_loss": "baseline_severity_weighted_loss_adamw",
    }
    return mapping[strategy]


def create_experiment_group(strategy: str) -> str:
    mapping = {
        "none": "experiment_no_imbalance_adamw",
        "balanced_batch": "experiment_balanced_batch_adamw",
        "severity_weighted_loss": "experiment_severity_weighted_loss_adamw",
    }
    return mapping[strategy]


def resolve_dir(raw_dir: str, default_base: Path) -> Path:
    path = Path(raw_dir).expanduser()
    if not path.is_absolute():
        path = default_base / path
    return path.resolve()


def mean_tensor_loss(losses: Sequence[torch.Tensor]) -> float:
    if len(losses) == 0:
        return 0.0
    values = [float(loss.detach().cpu().item()) if isinstance(loss, torch.Tensor) else float(loss) for loss in losses]
    return float(np.mean(values))


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed, deterministic=not args.non_deterministic)
    device = choose_device(force_cpu=args.force_cpu)
    print(f"Using device: {device}")

    data_dir = resolve_dir(args.data_dir, BASE_DIR)
    output_root = resolve_dir(args.output_root, BASE_DIR)

    x_path = (data_dir / args.x_file).resolve()
    y_path = (data_dir / args.y_file).resolve()

    print(f"Base directory: {BASE_DIR}")
    print(f"Data directory: {data_dir}")
    print(f"Resolved x path: {x_path}")
    print(f"Resolved y path: {y_path}")

    if not x_path.exists():
        raise FileNotFoundError(f"Could not find input data file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Could not find label data file: {y_path}")

    dataset = ImageDataset(str(x_path), str(y_path))
    y_all = np.asarray(dataset.targets).astype(int)
    n_classes = int(len(np.unique(y_all)))

    train_idx, val_idx = stratified_split_indices(y_all, val_ratio=args.val_ratio, seed=args.seed)
    train_dataset = DatasetSubset(dataset, train_idx)
    val_dataset = DatasetSubset(dataset, val_idx)

    use_balanced_batches = args.imbalance_strategy == "balanced_batch"
    train_sampler = BatchSampler(batch_size=args.batch_size, dataset=train_dataset, balanced=use_balanced_batches)
    val_sampler = BatchSampler(batch_size=args.batch_size, dataset=val_dataset, balanced=False)

    model = Net(n_classes=n_classes).to(device)

    if args.imbalance_strategy == "severity_weighted_loss":
        class_weights = get_severity_weights(args.severity_weight_set, n_classes=n_classes, device=device)
        loss_function = nn.CrossEntropyLoss(weight=class_weights)
        weight_info: Optional[List[float]] = class_weights.detach().cpu().numpy().tolist()
    else:
        loss_function = nn.CrossEntropyLoss()
        weight_info = None

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or create_experiment_name(args.imbalance_strategy)
    experiment_group = args.experiment_group or create_experiment_group(args.imbalance_strategy)
    experiment_root = output_root / experiment_group
    dirs = ensure_run_directories(experiment_root, experiment_name, run_id)

    config = {
        "run_id": run_id,
        "experiment_group": experiment_group,
        "experiment_name": experiment_name,
        "optimizer": "AdamW",
        "imbalance_strategy": args.imbalance_strategy,
        "severity_weight_set": args.severity_weight_set if args.imbalance_strategy == "severity_weighted_loss" else None,
        "severity_weights": weight_info,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "val_ratio": args.val_ratio,
        "device": device,
        "base_dir": str(BASE_DIR),
        "data_dir": str(data_dir),
        "x_file": args.x_file,
        "y_file": args.y_file,
        "resolved_x_path": str(x_path),
        "resolved_y_path": str(y_path),
        "output_root": str(output_root),
        "train_distribution": class_distribution(train_dataset.targets),
        "validation_distribution": class_distribution(val_dataset.targets),
    }

    save_json(config, dirs["best_artifacts_dir"] / "config.json")
    save_json(config, dirs["final_artifacts_dir"] / "config.json")

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    best_score = -float("inf")
    best_epoch = -1
    best_state = None
    best_metrics: Optional[Dict[str, float]] = None

    print(f"Starting training: {experiment_group} / {experiment_name}")

    for epoch in range(1, args.epochs + 1):
        train_losses = train_model(
            model=model,
            train_sampler=train_sampler,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
        )
        train_loss_value = mean_tensor_loss(train_losses)
        val_metrics = evaluate_model(
            model=model,
            sampler=val_sampler,
            loss_function=loss_function,
            device=device,
            n_classes=n_classes,
        )

        history["train_loss"].append(train_loss_value)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        score = val_metrics["macro_f1"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = dict(val_metrics)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss_value:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training finished without a valid best checkpoint.")

    best_model_path = dirs["best_weights_dir"] / "best_model.pt"
    torch.save(best_state, best_model_path)

    final_model_path: Optional[Path] = None
    if args.save_final_model:
        final_model_path = dirs["final_weights_dir"] / "final_model.pt"
        torch.save(model.state_dict(), final_model_path)

    save_json(history, dirs["best_artifacts_dir"] / "history.json")
    save_json(history, dirs["final_artifacts_dir"] / "history.json")

    best_report = {
        "selected_by": "validation_macro_f1",
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "optimizer": "AdamW",
        "imbalance_strategy": args.imbalance_strategy,
        "severity_weight_set": args.severity_weight_set if args.imbalance_strategy == "severity_weighted_loss" else None,
        "best_model_path": str(best_model_path),
    }
    final_report = {
        "final_epoch": args.epochs,
        "final_metrics": {
            "val_loss": history["val_loss"][-1],
            "val_accuracy": history["val_accuracy"][-1],
            "val_macro_f1": history["val_macro_f1"][-1],
        },
        "optimizer": "AdamW",
        "imbalance_strategy": args.imbalance_strategy,
        "severity_weight_set": args.severity_weight_set if args.imbalance_strategy == "severity_weighted_loss" else None,
        "final_model_path": str(final_model_path) if final_model_path is not None else None,
    }

    save_json(best_report, dirs["best_artifacts_dir"] / "report.json")
    save_json(final_report, dirs["final_artifacts_dir"] / "report.json")

    save_loss_plot(history["train_loss"], history["val_loss"], dirs["best_artifacts_dir"] / "loss_curve.png", f"Loss Curve ({experiment_name})")
    save_loss_plot(history["train_loss"], history["val_loss"], dirs["final_artifacts_dir"] / "loss_curve.png", f"Loss Curve ({experiment_name})")
    save_metric_plot(history["val_macro_f1"], "Validation Macro-F1", f"Validation Macro-F1 ({experiment_name})", dirs["best_artifacts_dir"] / "metric_curve.png")
    save_metric_plot(history["val_macro_f1"], "Validation Macro-F1", f"Validation Macro-F1 ({experiment_name})", dirs["final_artifacts_dir"] / "metric_curve.png")

    summary_lines = [
        f"Experiment group: {experiment_group}",
        f"Experiment name: {experiment_name}",
        f"Run ID: {run_id}",
        "Optimizer: AdamW",
        f"Imbalance strategy: {args.imbalance_strategy}",
        f"Best epoch: {best_epoch}",
        f"Best validation loss: {best_metrics['loss']:.6f}",
        f"Best validation accuracy: {best_metrics['accuracy']:.6f}",
        f"Best validation macro-F1: {best_metrics['macro_f1']:.6f}",
        f"Best model path: {best_model_path}",
    ]
    if final_model_path is not None:
        summary_lines.append(f"Final model path: {final_model_path}")
    if weight_info is not None:
        summary_lines.append(f"Severity weights: {weight_info}")

    save_text(summary_lines, dirs["best_artifacts_dir"] / "summary.txt")
    save_text(summary_lines, dirs["final_artifacts_dir"] / "summary.txt")

    print("\nFinished.")
    print(f"Best artifacts saved to:  {dirs['best_artifacts_dir']}")
    print(f"Best weights saved to:    {best_model_path}")
    if final_model_path is not None:
        print(f"Final artifacts saved to: {dirs['final_artifacts_dir']}")
        print(f"Final weights saved to:   {final_model_path}")

    return {
        "strategy": args.imbalance_strategy,
        "experiment_group": experiment_group,
        "experiment_name": experiment_name,
        "best_epoch": best_epoch,
        "best_validation_loss": best_metrics["loss"],
        "best_validation_accuracy": best_metrics["accuracy"],
        "best_validation_macro_f1": best_metrics["macro_f1"],
        "best_model_path": str(best_model_path),
        "best_artifacts_dir": str(dirs["best_artifacts_dir"]),
    }


def run_all_experiments(args: argparse.Namespace) -> None:
    shared_run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    strategies = [
        ("none", "experiment_no_imbalance_adamw", "baseline_no_imbalance_adamw"),
        ("balanced_batch", "experiment_balanced_batch_adamw", "baseline_balanced_batch_adamw"),
        ("severity_weighted_loss", "experiment_severity_weighted_loss_adamw", "baseline_severity_weighted_loss_adamw"),
    ]

    results: List[Dict[str, object]] = []
    for strategy, experiment_group, experiment_name in strategies:
        local_args = argparse.Namespace(**vars(args))
        local_args.imbalance_strategy = strategy
        local_args.experiment_group = experiment_group
        local_args.experiment_name = experiment_name
        local_args.run_id = shared_run_id

        print("\n" + "=" * 90)
        print(f"Running strategy: {strategy}")
        print(f"Experiment group: {experiment_group}")
        print(f"Experiment name: {experiment_name}")
        print("=" * 90)

        result = run_experiment(local_args)
        results.append(result)

    comparison_root = resolve_dir(args.output_root, BASE_DIR) / "comparison" / shared_run_id
    comparison_root.mkdir(parents=True, exist_ok=True)

    comparison_payload = {
        "run_id": shared_run_id,
        "optimizer": "AdamW",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "severity_weight_set": args.severity_weight_set,
        "results": results,
    }
    save_json(comparison_payload, comparison_root / "comparison_summary.json")

    lines = [
        "All imbalance strategy runs finished.",
        f"Run ID: {shared_run_id}",
        "Optimizer: AdamW",
        f"Epochs: {args.epochs}",
        "",
        "Best-checkpoint summary:",
    ]
    for result in results:
        lines.extend([
            f"- {result['strategy']}:",
            f"  experiment_group = {result['experiment_group']}",
            f"  experiment_name = {result['experiment_name']}",
            f"  best_epoch = {result['best_epoch']}",
            f"  best_validation_loss = {result['best_validation_loss']}",
            f"  best_validation_accuracy = {result['best_validation_accuracy']}",
            f"  best_validation_macro_f1 = {result['best_validation_macro_f1']}",
            f"  best_model_path = {result['best_model_path']}",
            f"  best_artifacts_dir = {result['best_artifacts_dir']}",
            "",
        ])
    save_text(lines, comparison_root / "comparison_summary.txt")

    print("\n@@@ All imbalance strategy runs finished.")
    print(f"@@@ Comparison summary saved to: {comparison_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run grouped imbalance experiments with AdamW.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=25, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--severity_weight_set", type=str, default="run2", choices=list(SEVERITY_WEIGHT_SETS.keys()))
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Directory containing X_train.npy and Y_train.npy.")
    parser.add_argument("--x_file", type=str, default="X_train.npy", help="Input image .npy filename.")
    parser.add_argument("--y_file", type=str, default="Y_train.npy", help="Label .npy filename.")
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Root output folder, default: dc1/experiments/experiment_imbalance")
    parser.add_argument("--save_final_model", action="store_true", help="Also save the final-epoch model.")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU even if CUDA/MPS is available.")
    parser.add_argument("--non_deterministic", action="store_true", help="Allow non-deterministic training for speed.")
    parser.add_argument("--run_single_strategy", choices=["none", "balanced_batch", "severity_weighted_loss"], default=None, help="Optional: run only one strategy. Leave empty to run all three.")
    parser.add_argument("--run_id", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--imbalance_strategy", type=str, default="none", help=argparse.SUPPRESS)
    parser.add_argument("--experiment_group", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--experiment_name", type=str, default=None, help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.run_single_strategy is not None:
        args.imbalance_strategy = args.run_single_strategy
        args.experiment_group = create_experiment_group(args.run_single_strategy)
        args.experiment_name = create_experiment_name(args.run_single_strategy)
        run_experiment(args)
    else:
        run_all_experiments(args)
