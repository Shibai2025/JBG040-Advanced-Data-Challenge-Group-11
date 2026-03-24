from __future__ import annotations

import argparse
import copy
import json
import os
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

from dc1.image_dataset import ImageDataset

try:
    from torchvision.models import ResNet18_Weights, resnet18
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "torchvision with ResNet18 support is required for this experiment. "
        "Please ensure torchvision is installed in the active environment."
    ) from exc


CLASS_NAMES = [
    "Atelectasis",
    "Effusion",
    "Infiltration",
    "No Finding",
    "Nodule",
    "Pneumothorax",
]

SETTING_DISPLAY_NAMES = {
    "none": "Fine-tuned ResNet18 + No Imbalance Handling",
    "balanced_batch": "Fine-tuned ResNet18 + Balanced Batch",
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


class BalancedBatchIterator:
    """
    Local balanced/unbalanced batch iterator used only inside this experiment file.

    In balanced mode, each epoch downsamples every class to the minority-class count,
    then shuffles the pooled indices before batching.
    """

    def __init__(self, dataset: DatasetSubset, batch_size: int, balanced: bool) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.balanced = balanced

    def _epoch_indices(self) -> np.ndarray:
        targets = np.asarray(self.dataset.targets).astype(int)

        if not self.balanced:
            indices = np.arange(len(self.dataset), dtype=int)
            np.random.shuffle(indices)
            return indices

        unique_classes = np.unique(targets)
        class_bins: List[np.ndarray] = []
        class_counts: List[int] = []

        for class_id in unique_classes:
            class_idx = np.where(targets == class_id)[0]
            class_bins.append(class_idx)
            class_counts.append(len(class_idx))

        if not class_counts:
            return np.array([], dtype=int)

        minority_count = int(min(class_counts))
        sampled_parts: List[np.ndarray] = []

        for class_idx in class_bins:
            sampled = np.random.choice(class_idx, size=minority_count, replace=False)
            sampled_parts.append(sampled)

        indices = np.concatenate(sampled_parts).astype(int)
        np.random.shuffle(indices)
        return indices

    def __iter__(self):
        indices = self._epoch_indices()
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            x_batch = [self.dataset[int(i)][0] for i in batch_indices]
            y_batch = [self.dataset[int(i)][1] for i in batch_indices]
            yield torch.stack(x_batch).float(), torch.tensor(y_batch).long()

    def __len__(self) -> int:
        targets = np.asarray(self.dataset.targets).astype(int)
        if len(targets) == 0:
            return 0

        if not self.balanced:
            n_items = len(self.dataset)
        else:
            _, counts = np.unique(targets, return_counts=True)
            n_items = int(counts.min()) * len(counts)

        return (n_items + self.batch_size - 1) // self.batch_size


class ResNet18Transfer(nn.Module):
    """
    Fine-tunable ResNet18 for grayscale chest X-rays.

    - Starts from pretrained ImageNet ResNet18.
    - Replaces conv1 to accept 1-channel input by averaging pretrained RGB kernels.
    - Replaces final FC layer with a 6-class classifier.
    - Applies grayscale normalization inside forward().
    """

    def __init__(self, n_classes: int) -> None:
        super().__init__()

        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)

        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        backbone.conv1 = new_conv
        backbone.fc = nn.Linear(backbone.fc.in_features, n_classes)

        self.model = backbone

        # Average ImageNet RGB stats for grayscale normalization.
        self.register_buffer("norm_mean", torch.tensor([0.449], dtype=torch.float32).view(1, 1, 1, 1))
        self.register_buffer("norm_std", torch.tensor([0.226], dtype=torch.float32).view(1, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.norm_mean) / self.norm_std
        return self.model(x)

    def trainable_parameter_groups(
        self,
        head_lr: float,
        backbone_lr: float,
        weight_decay: float,
    ) -> List[Dict[str, object]]:
        backbone_params: List[nn.Parameter] = []
        head_params: List[nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if name.startswith("fc."):
                head_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
            },
        ]


def set_seed(seed: int, deterministic: bool = True) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
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


def stratified_split_indices(
    y: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y).reshape(-1).astype(int)

    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []

    for class_id in np.unique(y):
        class_indices = np.where(y == class_id)[0]
        shuffled = rng.permutation(class_indices)

        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        if len(shuffled) > 1:
            n_val = min(n_val, len(shuffled) - 1)

        val_idx = shuffled[:n_val]
        train_idx = shuffled[n_val:]

        if len(train_idx) == 0:
            train_idx = val_idx[:1]
            val_idx = val_idx[1:]

        train_parts.append(train_idx)
        val_parts.append(val_idx)

    train_indices = np.concatenate(train_parts).astype(int)
    val_indices = np.concatenate(val_parts).astype(int)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def class_distribution(y: np.ndarray) -> Dict[str, int]:
    y = np.asarray(y).astype(int)
    return {
        class_name: int(np.sum(y == class_id))
        for class_id, class_name in enumerate(CLASS_NAMES)
    }


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else float(a / b)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def compute_per_class_metrics(cm: np.ndarray) -> List[Dict[str, float]]:
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
                "class_name": CLASS_NAMES[i],
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
            }
        )
    return rows


def compute_macro_f1_from_cm(cm: np.ndarray) -> float:
    rows = compute_per_class_metrics(cm)
    return float(np.mean([row["f1"] for row in rows])) if rows else 0.0


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
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_metric_plot(values: List[float], ylabel: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(range(1, len(values) + 1), values)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, output_path: Path, normalize: bool = False) -> None:
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
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    threshold = matrix.max() * 0.5 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = format(int(value), fmt) if fmt == "d" else format(value, fmt)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_classification_report(cm: np.ndarray, output_path: Path) -> None:
    rows = compute_per_class_metrics(cm)

    header = f"{'Class':<18} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    lines = [header, "-" * len(header)]

    for row in rows:
        lines.append(
            f"{row['class_name']:<18} "
            f"{row['precision']:>10.3f} "
            f"{row['recall']:>10.3f} "
            f"{row['f1']:>10.3f} "
            f"{int(row['support']):>10d}"
        )

    macro_precision = float(np.mean([row["precision"] for row in rows])) if rows else 0.0
    macro_recall = float(np.mean([row["recall"] for row in rows])) if rows else 0.0
    macro_f1 = float(np.mean([row["f1"] for row in rows])) if rows else 0.0
    total_support = int(np.sum([row["support"] for row in rows])) if rows else 0

    lines.append("-" * len(header))
    lines.append(
        f"{'MacroAvg':<18} "
        f"{macro_precision:>10.3f} "
        f"{macro_recall:>10.3f} "
        f"{macro_f1:>10.3f} "
        f"{total_support:>10d}"
    )
    save_text(lines, output_path)


def train_one_epoch(
    model: nn.Module,
    train_loader: BalancedBatchIterator,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: str,
) -> float:
    model.train()
    losses: List[float] = []

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = loss_function(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else 0.0


def evaluate_model(
    model: nn.Module,
    data_loader: BalancedBatchIterator,
    loss_function: nn.Module,
    device: str,
    n_classes: int,
) -> Dict[str, object]:
    model.eval()

    losses: List[float] = []
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = loss_function(logits, y_batch)
            preds = torch.argmax(logits, dim=1)

            losses.append(float(loss.detach().cpu().item()))
            all_true.append(y_batch.detach().cpu().numpy())
            all_pred.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=int)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=int)

    cm = compute_confusion_matrix(y_true, y_pred, n_classes=n_classes)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": compute_accuracy(y_true, y_pred),
        "macro_f1": compute_macro_f1_from_cm(cm),
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def build_model(n_classes: int) -> ResNet18Transfer:
    return ResNet18Transfer(n_classes=n_classes)


def build_optimizer(
    model: ResNet18Transfer,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    parameter_groups = model.trainable_parameter_groups(
        head_lr=head_lr,
        backbone_lr=backbone_lr,
        weight_decay=weight_decay,
    )
    return optim.AdamW(parameter_groups)


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


def train_single_setting(
    args: argparse.Namespace,
    base_dir: Path,
    full_train_dataset: ImageDataset,
    official_test_dataset: ImageDataset,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    setting_name: str,
    device: str,
    run_group_id: str,
) -> Dict[str, object]:
    deterministic = not args.non_deterministic
    set_seed(seed=args.seed, deterministic=deterministic)

    if setting_name not in {"none", "balanced_batch"}:
        raise ValueError(f"Unsupported setting_name='{setting_name}'")

    balanced_batches = setting_name == "balanced_batch"
    display_name = SETTING_DISPLAY_NAMES[setting_name]
    experiment_name = f"{args.experiment_name_prefix}_{setting_name}"
    experiment_root = base_dir / "experiments" / args.experiment_group / setting_name

    train_dataset = DatasetSubset(full_train_dataset, train_indices)
    val_dataset = DatasetSubset(full_train_dataset, val_indices)

    train_loader = BalancedBatchIterator(
        dataset=train_dataset,
        batch_size=args.batch_size,
        balanced=balanced_batches,
    )
    val_loader = BalancedBatchIterator(
        dataset=val_dataset,
        batch_size=args.val_batch_size,
        balanced=False,
    )

    model = build_model(n_classes=args.n_classes).to(device)
    optimizer = build_optimizer(
        model=model,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
    )
    loss_function = nn.CrossEntropyLoss()

    print("\n" + "=" * 98)
    print(f"Starting balance-effect run: {display_name}")
    print(f"Experiment name        : {experiment_name}")
    print(f"Balanced train batches : {balanced_batches}")
    print(f"Head learning rate     : {args.head_lr}")
    print(f"Backbone learning rate : {args.backbone_lr}")
    print(f"Weight decay           : {args.weight_decay}")
    print("=" * 98)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "validation_macro_f1": [],
    }

    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_epoch: Optional[int] = None
    best_val_loss = float("inf")
    best_val_macro_f1 = -float("inf")
    best_val_accuracy = 0.0
    best_confusion_matrix: Optional[np.ndarray] = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
        )

        val_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            loss_function=loss_function,
            device=device,
            n_classes=args.n_classes,
        )

        val_loss = float(val_metrics["loss"])
        val_accuracy = float(val_metrics["accuracy"])
        val_macro_f1 = float(val_metrics["macro_f1"])
        val_cm = np.asarray(val_metrics["confusion_matrix"])

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(val_loss)
        history["validation_accuracy"].append(val_accuracy)
        history["validation_macro_f1"].append(val_macro_f1)

        is_better = (
            val_macro_f1 > best_val_macro_f1
            or (
                np.isclose(val_macro_f1, best_val_macro_f1)
                and val_loss < best_val_loss
            )
        )

        if is_better:
            best_val_macro_f1 = val_macro_f1
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            best_confusion_matrix = val_cm.copy()

        print(
            f"[{display_name}] Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

    if best_state_dict is None or best_epoch is None or best_confusion_matrix is None:
        raise RuntimeError(f"No valid best checkpoint was created for '{setting_name}'.")

    run_dirs = ensure_run_directories(experiment_root, experiment_name, run_group_id)
    best_artifacts_dir = run_dirs["best_artifacts_dir"]
    final_artifacts_dir = run_dirs["final_artifacts_dir"]
    best_weights_dir = run_dirs["best_weights_dir"]
    final_weights_dir = run_dirs["final_weights_dir"]

    best_model_path = best_weights_dir / "best_model.pt"
    torch.save(best_state_dict, best_model_path)

    final_model_path: Optional[Path] = None
    if args.save_final_model:
        final_model_path = final_weights_dir / "final_model.pt"
        torch.save(model.state_dict(), final_model_path)

    save_loss_plot(
        history["train_loss"],
        history["validation_loss"],
        best_artifacts_dir / "loss_curve.png",
        f"Loss Curve ({display_name})",
    )
    save_loss_plot(
        history["train_loss"],
        history["validation_loss"],
        final_artifacts_dir / "loss_curve.png",
        f"Loss Curve ({display_name})",
    )

    save_metric_plot(
        history["validation_macro_f1"],
        "Validation Macro-F1",
        f"Validation Macro-F1 ({display_name})",
        best_artifacts_dir / "metric_curve_macro_f1.png",
    )
    save_metric_plot(
        history["validation_macro_f1"],
        "Validation Macro-F1",
        f"Validation Macro-F1 ({display_name})",
        final_artifacts_dir / "metric_curve_macro_f1.png",
    )

    save_metric_plot(
        history["validation_accuracy"],
        "Validation Accuracy",
        f"Validation Accuracy ({display_name})",
        best_artifacts_dir / "metric_curve_accuracy.png",
    )
    save_metric_plot(
        history["validation_accuracy"],
        "Validation Accuracy",
        f"Validation Accuracy ({display_name})",
        final_artifacts_dir / "metric_curve_accuracy.png",
    )

    plot_confusion_matrix(best_confusion_matrix, best_artifacts_dir / "confusion_matrix.png", normalize=False)
    plot_confusion_matrix(best_confusion_matrix, best_artifacts_dir / "confusion_matrix_normalized.png", normalize=True)
    save_classification_report(best_confusion_matrix, best_artifacts_dir / "classification_report.txt")

    if final_model_path is not None:
        final_cm_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            loss_function=loss_function,
            device=device,
            n_classes=args.n_classes,
        )
        final_cm = np.asarray(final_cm_metrics["confusion_matrix"])
        plot_confusion_matrix(final_cm, final_artifacts_dir / "confusion_matrix.png", normalize=False)
        plot_confusion_matrix(final_cm, final_artifacts_dir / "confusion_matrix_normalized.png", normalize=True)
        save_classification_report(final_cm, final_artifacts_dir / "classification_report.txt")

    run_config: Dict[str, object] = {
        "run_id": run_group_id,
        "experiment_group": args.experiment_group,
        "experiment_name": experiment_name,
        "architecture_name": "finetuned_resnet18",
        "setting_name": setting_name,
        "setting_display_name": display_name,
        "optimizer": "AdamW",
        "epochs": args.epochs,
        "batch_size_train": args.batch_size,
        "batch_size_val": args.val_batch_size,
        "head_learning_rate": args.head_lr,
        "backbone_learning_rate": args.backbone_lr,
        "weight_decay": args.weight_decay,
        "balanced_train_batches": balanced_batches,
        "seed": args.seed,
        "deterministic": not args.non_deterministic,
        "device": device,
        "n_classes": args.n_classes,
        "class_names": CLASS_NAMES,
        "validation_ratio": args.val_ratio,
        "train_size": len(train_dataset),
        "validation_size": len(val_dataset),
        "official_test_size": len(official_test_dataset),
        "selection_metric": "validation_macro_f1",
        "selection_tiebreaker": "validation_loss",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_macro_f1,
        "best_validation_loss": best_val_loss,
        "best_validation_accuracy": best_val_accuracy,
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path) if final_model_path is not None else None,
        "reference_threshold_macro_f1": args.reference_threshold_macro_f1,
        "beats_reference_thresholded_balanced_batch": bool(best_val_macro_f1 > args.reference_threshold_macro_f1),
        "note": "Official test set was not used for model selection.",
    }

    best_report: Dict[str, object] = {
        "selected_by": "validation_macro_f1",
        "tiebreaker": "validation_loss",
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "best_validation_accuracy": best_val_accuracy,
        "best_validation_macro_f1": best_val_macro_f1,
        "best_model_path": str(best_model_path),
        "reference_threshold_macro_f1": args.reference_threshold_macro_f1,
        "beats_reference_thresholded_balanced_batch": bool(best_val_macro_f1 > args.reference_threshold_macro_f1),
    }

    final_report: Dict[str, object] = {
        "final_epoch": args.epochs,
        "final_validation_loss": history["validation_loss"][-1],
        "final_validation_accuracy": history["validation_accuracy"][-1],
        "final_validation_macro_f1": history["validation_macro_f1"][-1],
        "final_model_path": str(final_model_path) if final_model_path is not None else None,
    }

    save_json(run_config, best_artifacts_dir / "config.json")
    save_json(run_config, final_artifacts_dir / "config.json")
    save_json(history, best_artifacts_dir / "history.json")
    save_json(history, final_artifacts_dir / "history.json")
    save_json(best_report, best_artifacts_dir / "report.json")
    save_json(final_report, final_artifacts_dir / "report.json")

    summary_lines = [
        f"Balance-effect experiment finished: {display_name}",
        f"Experiment group: {args.experiment_group}",
        f"Experiment name: {experiment_name}",
        f"Run ID: {run_group_id}",
        f"Best epoch: {best_epoch}",
        f"Best validation loss: {best_val_loss:.6f}",
        f"Best validation accuracy: {best_val_accuracy:.6f}",
        f"Best validation macro-F1: {best_val_macro_f1:.6f}",
        f"Reference thresholded macro-F1: {args.reference_threshold_macro_f1:.6f}",
        f"Beats thresholded reference: {best_val_macro_f1 > args.reference_threshold_macro_f1}",
        f"Best model path: {best_model_path}",
        f"Final model path: {final_model_path}",
    ]
    save_text(summary_lines, best_artifacts_dir / "summary.txt")
    save_text(summary_lines, final_artifacts_dir / "summary.txt")

    print(f"\nCompleted balance-effect run: {display_name}")
    print(f"Best validation macro-F1: {best_val_macro_f1:.6f}")
    print(f"Best artifacts dir      : {best_artifacts_dir}")
    print(f"Best model path         : {best_model_path}")

    return {
        "setting_name": setting_name,
        "setting_display_name": display_name,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "best_validation_accuracy": best_val_accuracy,
        "best_validation_macro_f1": best_val_macro_f1,
        "best_model_path": str(best_model_path),
        "best_artifacts_dir": str(best_artifacts_dir),
        "final_model_path": str(final_model_path) if final_model_path is not None else None,
        "final_artifacts_dir": str(final_artifacts_dir),
        "beats_reference_thresholded_balanced_batch": bool(best_val_macro_f1 > args.reference_threshold_macro_f1),
    }


def parse_setting_list(raw_items: Sequence[str]) -> List[str]:
    allowed = {"none", "balanced_batch"}
    parsed: List[str] = []

    for item in raw_items:
        for token in item.split(","):
            name = token.strip().lower()
            if not name:
                continue
            if name not in allowed:
                raise ValueError(f"Unsupported setting '{name}'. Choose from: {sorted(allowed)}")
            if name not in parsed:
                parsed.append(name)

    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test 3: compare fine-tuned pretrained ResNet18 with and without balanced batches "
            "to measure whether the current best imbalance-handling strategy still helps once "
            "feature extraction is strengthened."
        )
    )
    parser.add_argument("--experiment_group", type=str, default="experiment_resnet18_balance_effect")
    parser.add_argument("--experiment_name_prefix", type=str, default="architecture_test3")
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["none", "balanced_batch"],
        help="Settings to run. Choices: none, balanced_batch",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--val_batch_size", type=int, default=100)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_classes", type=int, default=6)
    parser.add_argument("--save_final_model", action="store_true")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument(
        "--non_deterministic",
        action="store_true",
        help=(
            "Disable deterministic training. Use this if you encounter CUDA deterministic "
            "issues or prefer maximum speed."
        ),
    )
    parser.add_argument(
        "--reference_threshold_macro_f1",
        type=float,
        default=0.2803132184757912,
        help="Current thresholded balanced-batch validation macro-F1 reference to compare against.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    deterministic = not args.non_deterministic
    set_seed(seed=args.seed, deterministic=deterministic)

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    full_train_dataset = ImageDataset(data_dir / "X_train.npy", data_dir / "Y_train.npy")
    official_test_dataset = ImageDataset(data_dir / "X_test.npy", data_dir / "Y_test.npy")

    full_y = np.asarray(full_train_dataset.targets).reshape(-1).astype(int)
    train_indices, val_indices = stratified_split_indices(
        y=full_y,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print("@@@ Full original training set size:", len(full_train_dataset))
    print("@@@ Train subset size:", len(train_indices))
    print("@@@ Validation subset size:", len(val_indices))
    print("@@@ Official test set size (not used for model selection):", len(official_test_dataset))

    print("\n@@@ Train class distribution:")
    train_targets = full_y[train_indices]
    for class_name, count in class_distribution(train_targets).items():
        print(f"    {class_name}: {count}")

    print("\n@@@ Validation class distribution:")
    val_targets = full_y[val_indices]
    for class_name, count in class_distribution(val_targets).items():
        print(f"    {class_name}: {count}")

    device = choose_device(force_cpu=args.force_cpu)
    print(f"\n@@@ Using device: {device}")

    setting_names = parse_setting_list(args.settings)
    run_group_id = f"run_{datetime.now():%Y%m%d_%H%M%S}"

    results: List[Dict[str, object]] = []
    for setting_name in setting_names:
        result = train_single_setting(
            args=args,
            base_dir=base_dir,
            full_train_dataset=full_train_dataset,
            official_test_dataset=official_test_dataset,
            train_indices=train_indices,
            val_indices=val_indices,
            setting_name=setting_name,
            device=device,
            run_group_id=run_group_id,
        )
        results.append(result)

    comparison_root = base_dir / "experiments" / args.experiment_group / "comparison" / run_group_id
    comparison_root.mkdir(parents=True, exist_ok=True)

    comparison_payload: Dict[str, object] = {
        "run_group_id": run_group_id,
        "experiment_group": args.experiment_group,
        "experiment_name_prefix": args.experiment_name_prefix,
        "architecture_name": "finetuned_resnet18",
        "seed": args.seed,
        "validation_ratio": args.val_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "head_learning_rate": args.head_lr,
        "backbone_learning_rate": args.backbone_lr,
        "weight_decay": args.weight_decay,
        "reference_threshold_macro_f1": args.reference_threshold_macro_f1,
        "results": results,
    }
    save_json(comparison_payload, comparison_root / "comparison_summary.json")

    lines = [
        "ResNet18 balance-effect comparison finished.",
        f"Run group ID: {run_group_id}",
        f"Experiment group: {args.experiment_group}",
        "Architecture: finetuned_resnet18",
        f"Settings: {', '.join(str(r['setting_display_name']) for r in results)}",
        f"Reference thresholded macro-F1: {args.reference_threshold_macro_f1:.6f}",
        "",
        "Best-checkpoint summary:",
    ]

    for result in results:
        lines.extend(
            [
                f"- {result['setting_display_name']}:",
                f"  best_epoch = {result['best_epoch']}",
                f"  best_validation_loss = {result['best_validation_loss']}",
                f"  best_validation_accuracy = {result['best_validation_accuracy']}",
                f"  best_validation_macro_f1 = {result['best_validation_macro_f1']}",
                f"  beats_reference_thresholded_balanced_batch = {result['beats_reference_thresholded_balanced_batch']}",
                f"  best_model_path = {result['best_model_path']}",
                f"  best_artifacts_dir = {result['best_artifacts_dir']}",
                "",
            ]
        )

    save_text(lines, comparison_root / "comparison_summary.txt")

    print("\n@@@ All balance-effect runs finished.")
    print(f"@@@ Comparison summary saved to: {comparison_root}")
    print("@@@ Use experiment_evaluation.py on the saved best_model.pt files for official TEST evaluation.")


if __name__ == "__main__":
    main()