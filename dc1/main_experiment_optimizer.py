from __future__ import annotations

import argparse
import copy
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore

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

VALID_OPTIMIZERS = {"sgd", "adam", "adamw"}
DISPLAY_NAME_MAP = {
    "sgd": "SGD",
    "adam": "Adam",
    "adamw": "AdamW",
}


class DatasetSubset:
    """Lightweight subset wrapper that preserves targets for BatchSampler."""

    def __init__(self, base_dataset: ImageDataset, indices: Sequence[int]) -> None:
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=int)
        self.targets = np.asarray(base_dataset.targets)[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        return self.base_dataset[real_idx]


def set_seed(seed: int, deterministic: bool = True) -> None:
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


def mean_tensor_loss(losses: List[torch.Tensor]) -> float:
    if not losses:
        return 0.0
    return float(torch.stack([loss.detach().cpu() for loss in losses]).mean().item())


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


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1_scores: List[float] = []

    for class_id in range(n_classes):
        tp = np.sum((y_true == class_id) & (y_pred == class_id))
        fp = np.sum((y_true != class_id) & (y_pred == class_id))
        fn = np.sum((y_true == class_id) & (y_pred != class_id))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def evaluate_model(
    model: Net,
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


def save_loss_plot(
    train_losses: List[float],
    val_losses: List[float],
    output_path: Path,
    tag: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train loss", color="#1f77b4")
    ax.plot(range(1, len(val_losses) + 1), val_losses, label="Validation loss", color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    title = "Training and Validation Loss"
    if tag:
        title = f"{title} ({tag})"
    ax.set_title(title)

    ax.legend()
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_metric_plot(values: List[float], ylabel: str, title: str, output_path: Path, color: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(range(1, len(values) + 1), values, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def class_distribution(y: np.ndarray) -> Dict[str, int]:
    counts: Dict[str, int] = {}
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


def save_json(data: Dict[str, object], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_run_text_summary(summary_lines: List[str], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")


def build_optimizer(optimizer_name: str, model: nn.Module, lr: float, momentum: float, weight_decay: float):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def default_lr_for_optimizer(optimizer_name: str, explicit_lr: Optional[float]) -> float:
    if explicit_lr is not None:
        return explicit_lr
    if optimizer_name == "sgd":
        return 0.001
    if optimizer_name in {"adam", "adamw"}:
        return 0.0001
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_run_config(
    args: argparse.Namespace,
    device: str,
    train_size: int,
    val_size: int,
    full_train_size: int,
    official_test_size: int,
    run_id: str,
    optimizer_name: str,
    learning_rate: float,
) -> Dict[str, object]:
    return {
        "run_id": run_id,
        "experiment_group": args.experiment_group,
        "experiment_name": f"{args.experiment_name_prefix}_{optimizer_name}",
        "optimizer": DISPLAY_NAME_MAP[optimizer_name],
        "seed": args.seed,
        "deterministic": not args.non_deterministic,
        "device": device,
        "epochs": args.nb_epochs,
        "learning_rate": learning_rate,
        "momentum": args.momentum if optimizer_name == "sgd" else None,
        "weight_decay": args.weight_decay,
        "loss_function": "CrossEntropyLoss",
        "batch_size_train": args.batch_size,
        "batch_size_val": args.val_batch_size,
        "balanced_train_batches": args.balanced_batches,
        "validation_ratio": args.val_ratio,
        "model_selection_metric": "validation_loss",
        "save_final_model": args.save_final_model,
        "architecture": "dc1.net.Net",
        "n_classes": 6,
        "class_names": CLASS_NAMES,
        "full_train_size": full_train_size,
        "train_size": train_size,
        "validation_size": val_size,
        "official_test_size": official_test_size,
        "note": "Official test set was not used in training or model selection.",
    }


def build_best_report(
    run_id: str,
    experiment_name: str,
    optimizer_name: str,
    best_epoch: Optional[int],
    best_val_loss: float,
    best_model_path: Optional[Path],
    train_loss_history: List[float],
    val_loss_history: List[float],
    val_acc_history: List[float],
    val_macro_f1_history: List[float],
) -> Dict[str, object]:
    display_name = DISPLAY_NAME_MAP[optimizer_name]
    if best_epoch is None or best_model_path is None:
        return {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "optimizer": display_name,
            "selected_model_type": "best",
            "best_epoch": None,
            "best_model_path": None,
            "note": "No best model was saved.",
        }

    idx = best_epoch - 1
    return {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "optimizer": display_name,
        "selected_model_type": "best",
        "best_epoch": best_epoch,
        "best_model_path": str(best_model_path),
        "validation_loss_at_best_epoch": val_loss_history[idx],
        "validation_accuracy_at_best_epoch": val_acc_history[idx],
        "validation_macro_f1_at_best_epoch": val_macro_f1_history[idx],
        "train_loss_at_best_epoch": train_loss_history[idx],
        "best_validation_loss": best_val_loss,
    }


def build_final_report(
    run_id: str,
    experiment_name: str,
    optimizer_name: str,
    final_epoch: int,
    final_model_path: Optional[Path],
    train_loss_history: List[float],
    val_loss_history: List[float],
    val_acc_history: List[float],
    val_macro_f1_history: List[float],
) -> Dict[str, object]:
    idx = final_epoch - 1
    return {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "optimizer": DISPLAY_NAME_MAP[optimizer_name],
        "selected_model_type": "final",
        "final_epoch": final_epoch,
        "final_model_path": str(final_model_path) if final_model_path is not None else None,
        "train_loss_at_final_epoch": train_loss_history[idx],
        "validation_loss_at_final_epoch": val_loss_history[idx],
        "validation_accuracy_at_final_epoch": val_acc_history[idx],
        "validation_macro_f1_at_final_epoch": val_macro_f1_history[idx],
    }


def train_single_optimizer(
    args: argparse.Namespace,
    base_dir: Path,
    full_train_dataset: ImageDataset,
    official_test_dataset: ImageDataset,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    optimizer_name: str,
    device: str,
    run_group_id: str,
) -> Dict[str, object]:
    optimizer_name = optimizer_name.lower()
    display_name = DISPLAY_NAME_MAP[optimizer_name]
    experiment_name = f"{args.experiment_name_prefix}_{optimizer_name}"
    learning_rate = default_lr_for_optimizer(optimizer_name, args.lr)

    deterministic = not args.non_deterministic
    set_seed(seed=args.seed, deterministic=deterministic)

    train_dataset = DatasetSubset(full_train_dataset, train_indices)
    val_dataset = DatasetSubset(full_train_dataset, val_indices)

    print("\n" + "=" * 80)
    print(f"Starting optimizer run: {display_name}")
    print(f"Experiment name        : {experiment_name}")
    print(f"Learning rate          : {learning_rate}")
    if optimizer_name == "sgd":
        print(f"Momentum               : {args.momentum}")
    print(f"Balanced train batches : {args.balanced_batches}")
    print("=" * 80)

    model = Net(n_classes=6)
    optimizer = build_optimizer(
        optimizer_name=optimizer_name,
        model=model,
        lr=learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    loss_function = nn.CrossEntropyLoss()

    model.to(device)
    if device in {"cpu", "cuda"} and args.print_model_summary:
        summary(model, (1, 128, 128), device=device)

    train_sampler = BatchSampler(
        batch_size=args.batch_size,
        dataset=train_dataset,
        balanced=args.balanced_batches,
    )
    val_sampler = BatchSampler(
        batch_size=args.val_batch_size,
        dataset=val_dataset,
        balanced=False,
    )

    train_loss_history: List[float] = []
    val_loss_history: List[float] = []
    val_acc_history: List[float] = []
    val_macro_f1_history: List[float] = []

    best_val_loss = float("inf")
    best_epoch: Optional[int] = None
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    for epoch_idx in range(args.nb_epochs):
        train_losses = train_model(
            model=model,
            train_sampler=train_sampler,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
        )
        train_loss = mean_tensor_loss(train_losses)
        train_loss_history.append(train_loss)

        val_metrics = evaluate_model(
            model=model,
            sampler=val_sampler,
            loss_function=loss_function,
            device=device,
            n_classes=6,
        )
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_macro_f1 = val_metrics["macro_f1"]

        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        val_macro_f1_history.append(val_macro_f1)

        print(
            f"\n[{display_name}] Epoch {epoch_idx + 1}/{args.nb_epochs}"
            f"\n  Train loss          : {train_loss:.6f}"
            f"\n  Validation loss     : {val_loss:.6f}"
            f"\n  Validation accuracy : {val_acc:.6f}"
            f"\n  Validation macro-F1 : {val_macro_f1:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch_idx + 1
            best_state_dict = copy.deepcopy(model.state_dict())
            print(f"  -> New best model at epoch {best_epoch} (validation loss = {best_val_loss:.6f})")

    experiment_root = base_dir / "experiments" / args.experiment_group / optimizer_name
    run_dirs = ensure_run_directories(experiment_root, experiment_name, run_group_id)

    best_artifacts_dir = run_dirs["best_artifacts_dir"]
    final_artifacts_dir = run_dirs["final_artifacts_dir"]
    best_weights_dir = run_dirs["best_weights_dir"]
    final_weights_dir = run_dirs["final_weights_dir"]

    final_model_path: Optional[Path] = None
    if args.save_final_model:
        final_model_path = final_weights_dir / "final_model.pt"
        torch.save(model.state_dict(), final_model_path)

    best_model_path: Optional[Path] = None
    if best_state_dict is not None and best_epoch is not None:
        best_model_path = best_weights_dir / "best_model.pt"
        torch.save(best_state_dict, best_model_path)

    optimizer_tag = display_name

    save_loss_plot(
        train_loss_history,
        val_loss_history,
        best_artifacts_dir / f"loss_curve_{optimizer_tag}.png",
        tag=optimizer_tag,
    )
    save_loss_plot(
        train_loss_history,
        val_loss_history,
        final_artifacts_dir / f"loss_curve_{optimizer_tag}.png",
        tag=optimizer_tag,
    )

    save_metric_plot(
        val_acc_history,
        ylabel="Validation accuracy",
        title=f"Validation Accuracy over Epochs ({optimizer_tag})",
        output_path=best_artifacts_dir / f"metric_curve_accuracy_{optimizer_tag}.png",
        color="#2ca02c",
    )
    save_metric_plot(
        val_acc_history,
        ylabel="Validation accuracy",
        title=f"Validation Accuracy over Epochs ({optimizer_tag})",
        output_path=final_artifacts_dir / f"metric_curve_accuracy_{optimizer_tag}.png",
        color="#2ca02c",
    )

    save_metric_plot(
        val_macro_f1_history,
        ylabel="Validation macro-F1",
        title=f"Validation Macro-F1 over Epochs ({optimizer_tag})",
        output_path=best_artifacts_dir / f"metric_curve_macro_f1_{optimizer_tag}.png",
        color="#9467bd",
    )
    save_metric_plot(
        val_macro_f1_history,
        ylabel="Validation macro-F1",
        title=f"Validation Macro-F1 over Epochs ({optimizer_tag})",
        output_path=final_artifacts_dir / f"metric_curve_macro_f1_{optimizer_tag}.png",
        color="#9467bd",
    )

    run_config = build_run_config(
        args=args,
        device=device,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        full_train_size=len(full_train_dataset),
        official_test_size=len(official_test_dataset),
        run_id=run_group_id,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
    )
    run_config["final_epoch"] = args.nb_epochs
    run_config["best_epoch"] = best_epoch
    run_config["best_validation_loss"] = best_val_loss if best_epoch is not None else None
    run_config["final_model_path"] = str(final_model_path) if final_model_path is not None else None
    run_config["best_model_path"] = str(best_model_path) if best_model_path is not None else None

    history = {
        "train_loss": train_loss_history,
        "validation_loss": val_loss_history,
        "validation_accuracy": val_acc_history,
        "validation_macro_f1": val_macro_f1_history,
    }

    best_report = build_best_report(
        run_id=run_group_id,
        experiment_name=experiment_name,
        optimizer_name=optimizer_name,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_model_path=best_model_path,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        val_acc_history=val_acc_history,
        val_macro_f1_history=val_macro_f1_history,
    )

    if final_model_path is not None:
        final_report = build_final_report(
            run_id=run_group_id,
            experiment_name=experiment_name,
            optimizer_name=optimizer_name,
            final_epoch=args.nb_epochs,
            final_model_path=final_model_path,
            train_loss_history=train_loss_history,
            val_loss_history=val_loss_history,
            val_acc_history=val_acc_history,
            val_macro_f1_history=val_macro_f1_history,
        )
    else:
        final_report = {
            "run_id": run_group_id,
            "experiment_name": experiment_name,
            "optimizer": display_name,
            "selected_model_type": "final",
            "final_epoch": args.nb_epochs,
            "final_model_path": None,
            "note": "Final model saving was disabled for this run.",
        }

    save_json(run_config, best_artifacts_dir / "config.json")
    save_json(run_config, final_artifacts_dir / "config.json")
    save_json(history, best_artifacts_dir / "history.json")
    save_json(history, final_artifacts_dir / "history.json")
    save_json(best_report, best_artifacts_dir / "report.json")
    save_json(final_report, final_artifacts_dir / "report.json")

    text_summary_lines = [
        f"Training finished for optimizer: {display_name}",
        f"Experiment group: {args.experiment_group}",
        f"Experiment name: {experiment_name}",
        f"Run ID: {run_group_id}",
        f"Learning rate: {learning_rate}",
        f"Momentum: {args.momentum if optimizer_name == 'sgd' else 'N/A'}",
        f"Weight decay: {args.weight_decay}",
        f"Best epoch: {best_epoch}",
        f"Best validation loss: {best_val_loss:.6f}" if best_epoch is not None else "Best validation loss: None",
        f"Best artifacts dir: {best_artifacts_dir}",
        f"Final artifacts dir: {final_artifacts_dir}",
        f"Best model path: {best_model_path}",
        f"Final model path: {final_model_path}",
    ]
    save_run_text_summary(text_summary_lines, best_artifacts_dir / "summary.txt")
    save_run_text_summary(text_summary_lines, final_artifacts_dir / "summary.txt")

    print(f"\nCompleted optimizer run: {display_name}")
    print(f"Best artifacts dir : {best_artifacts_dir}")
    print(f"Final artifacts dir: {final_artifacts_dir}")
    print(f"Best model path    : {best_model_path}")
    print(f"Final model path   : {final_model_path}")

    return {
        "optimizer": optimizer_name,
        "optimizer_display_name": display_name,
        "learning_rate": learning_rate,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss if best_epoch is not None else None,
        "best_validation_accuracy": val_acc_history[best_epoch - 1] if best_epoch is not None else None,
        "best_validation_macro_f1": val_macro_f1_history[best_epoch - 1] if best_epoch is not None else None,
        "best_model_path": str(best_model_path) if best_model_path is not None else None,
        "best_artifacts_dir": str(best_artifacts_dir),
        "final_model_path": str(final_model_path) if final_model_path is not None else None,
        "final_artifacts_dir": str(final_artifacts_dir),
    }


def parse_optimizer_list(raw_optimizers: List[str]) -> List[str]:
    parsed: List[str] = []
    for item in raw_optimizers:
        for token in item.split(","):
            name = token.strip().lower()
            if not name:
                continue
            if name not in VALID_OPTIMIZERS:
                raise ValueError(f"Unsupported optimizer '{name}'. Choose from: {sorted(VALID_OPTIMIZERS)}")
            parsed.append(name)

    unique_ordered = []
    for name in parsed:
        if name not in unique_ordered:
            unique_ordered.append(name)
    return unique_ordered


def main(args: argparse.Namespace) -> None:
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

    train_targets = full_y[train_indices]
    val_targets = full_y[val_indices]

    print("\n@@@ Train class distribution:")
    for class_name, count in class_distribution(train_targets).items():
        print(f"    {class_name}: {count}")

    print("\n@@@ Validation class distribution:")
    for class_name, count in class_distribution(val_targets).items():
        print(f"    {class_name}: {count}")

    device = choose_device(force_cpu=args.debug_cpu)
    print(f"\n@@@ Using device: {device}")

    optimizer_names = parse_optimizer_list(args.optimizers)
    run_group_id = f"run_{datetime.now():%Y%m%d_%H%M%S}"

    results: List[Dict[str, object]] = []
    for optimizer_name in optimizer_names:
        result = train_single_optimizer(
            args=args,
            base_dir=base_dir,
            full_train_dataset=full_train_dataset,
            official_test_dataset=official_test_dataset,
            train_indices=train_indices,
            val_indices=val_indices,
            optimizer_name=optimizer_name,
            device=device,
            run_group_id=run_group_id,
        )
        results.append(result)

    comparison_root = base_dir / "experiments" / args.experiment_group / "comparison" / run_group_id
    comparison_root.mkdir(parents=True, exist_ok=True)

    comparison_payload = {
        "run_group_id": run_group_id,
        "experiment_group": args.experiment_group,
        "experiment_name_prefix": args.experiment_name_prefix,
        "seed": args.seed,
        "validation_ratio": args.val_ratio,
        "balanced_train_batches": args.balanced_batches,
        "nb_epochs": args.nb_epochs,
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "weight_decay": args.weight_decay,
        "results": results,
    }
    save_json(comparison_payload, comparison_root / "comparison_summary.json")

    lines = [
        "Optimizer comparison finished.",
        f"Run group ID: {run_group_id}",
        f"Experiment group: {args.experiment_group}",
        f"Optimizers: {', '.join([str(r['optimizer_display_name']) for r in results])}",
        "",
        "Best-checkpoint summary:",
    ]
    for result in results:
        lines.extend([
            f"- {str(result['optimizer_display_name'])}:",
            f"  best_epoch = {result['best_epoch']}",
            f"  best_validation_loss = {result['best_validation_loss']}",
            f"  best_validation_accuracy = {result['best_validation_accuracy']}",
            f"  best_validation_macro_f1 = {result['best_validation_macro_f1']}",
            f"  best_model_path = {result['best_model_path']}",
            f"  best_artifacts_dir = {result['best_artifacts_dir']}",
            "",
        ])
    save_run_text_summary(lines, comparison_root / "comparison_summary.txt")

    print("\n@@@ All optimizer runs finished.")
    print(f"@@@ Comparison summary saved to: {comparison_root}")
    print("@@@ Use your evaluation script on the saved best_model.pt files for final TEST evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train one or more optimizers with the same stratified train/validation split and archive outputs separately."
    )

    parser.add_argument("--experiment_group", type=str, default="experiment_optimizer")
    parser.add_argument("--experiment_name_prefix", type=str, default="baseline")
    parser.add_argument("--optimizers", nargs="+", default=["sgd", "adam", "adamw"])
    parser.add_argument("--nb_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--val_batch_size", type=int, default=100)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Optional shared learning rate. If omitted, optimizer-specific defaults are used.",
    )
    parser.add_argument("--momentum", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    parser.add_argument("--balanced_batches", action="store_true", help="Use balanced sampling for training batches only")
    parser.add_argument("--non_deterministic", action="store_true", help="Disable deterministic PyTorch behavior")
    parser.add_argument("--debug_cpu", action="store_true", help="Force CPU usage for debugging")
    parser.add_argument("--save_final_model", action="store_true", help="Also save the final-epoch model checkpoint")
    parser.add_argument("--print_model_summary", action="store_true", help="Print torchsummary model summary for each optimizer run")

    main(parser.parse_args())
