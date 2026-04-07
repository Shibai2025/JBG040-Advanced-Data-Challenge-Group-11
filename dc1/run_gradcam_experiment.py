from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from dc1.gradcam import GradCAM
from dc1.image_dataset import ImageDataset
from dc1.net import Net
from dc1.resnet import ResNet18Transfer

CLASS_NAME_MAP = {
    0: "Atelectasis",
    1: "Effusion",
    2: "Infiltration",
    3: "No Finding",
    4: "Nodule",
    5: "Pneumothorax",
}


@dataclass
class GradCAMCandidate:
    candidate_type: str  # "model" or "threshold"
    source_path: Path
    model_path: Path
    experiment_group: str
    experiment_name: str
    source_run_id: str
    architecture_hint: str
    threshold_value: Optional[float] = None


@dataclass
class CandidateRunArtifacts:
    candidate: GradCAMCandidate
    run_dir: Path
    reports_dir: Path
    all_records: List[Dict[str, object]]
    selected_entries: List[Dict[str, object]]
    accuracy: float
    coverage: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch Grad-CAM for all best checkpoints and threshold configs, "
            "archived by experiment group / model name / run id, with optional comparison panels."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional: run only one specific model checkpoint.",
    )
    parser.add_argument(
        "--threshold_config",
        type=str,
        default=None,
        help="Optional: run only one specific threshold config JSON.",
    )
    parser.add_argument(
        "--search_dir",
        type=str,
        default="experiments",
        help="Root directory searched recursively for best-model checkpoints and best-threshold configs.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing X_test.npy and Y_test.npy.",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="gradcam_result_experiment",
        help="Root directory where Grad-CAM outputs are archived.",
    )
    parser.add_argument(
        "--correct_samples_per_pair",
        type=int,
        default=2,
        help="Number of correctly classified samples to save per (true, pred) pair.",
    )
    parser.add_argument(
        "--wrong_samples_per_pair",
        type=int,
        default=3,
        help="Number of wrongly classified samples to save per (true, pred) pair.",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=6,
        help="Number of classes.",
    )
    parser.add_argument(
        "--clean_previous_outputs",
        action="store_true",
        help="Delete the whole archive folder for each resolved experiment before creating a new run.",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force CPU even if CUDA or MPS is available.",
    )
    parser.add_argument(
        "--apply_threshold_filter",
        action="store_true",
        help=(
            "For threshold candidates, keep only samples with max probability >= selected threshold. "
            "If not set, threshold candidates are still labeled separately but use the same selection policy as normal models."
        ),
    )
    parser.add_argument(
        "--comparison_mode",
        type=str,
        default="none",
        choices=["none", "same_sample_multi_model"],
        help="Optional comparison output mode.",
    )
    parser.add_argument(
        "--comparison_max_samples",
        type=int,
        default=12,
        help="Maximum number of sample comparison panels to generate.",
    )
    return parser.parse_args()


def choose_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_local_path(raw_path: str) -> Path:
    base_dir = Path(__file__).resolve().parent
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def resolve_flexible_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()

    if path.is_absolute():
        return path.resolve()

    candidate = path.resolve()
    if candidate.exists():
        return candidate

    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent

    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate

    candidate = (project_root / path).resolve()
    if candidate.exists():
        return candidate

    return (base_dir / path).resolve()


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
    return result.strip("_") or "unknown"


def candidate_display_name(candidate: GradCAMCandidate) -> str:
    if candidate.candidate_type == "threshold":
        return (
            f"{candidate.experiment_group}/{candidate.experiment_name}"
            f" [threshold={candidate.threshold_value}]"
        )
    return f"{candidate.experiment_group}/{candidate.experiment_name}"


def infer_architecture_hint(path: Path) -> str:
    if "resnet18" in str(path).lower():
        return "resnet18"
    return "net"

def parse_checkpoint_metadata(checkpoint_path: Path) -> Tuple[str, str, str]:
    parts = list(checkpoint_path.parts)

    experiment_group = "unknown_group"
    experiment_name = "unknown_model"
    source_run_id = checkpoint_path.parent.name

    if "experiments" in parts:
        exp_idx = parts.index("experiments")
        if exp_idx + 1 < len(parts):
            experiment_group = parts[exp_idx + 1]

        if "model_weights" in parts:
            mw_idx = parts.index("model_weights")
            if mw_idx - 1 >= 0:
                experiment_name = parts[mw_idx - 1]
            elif mw_idx + 1 < len(parts):
                experiment_name = parts[mw_idx + 1]

    experiment_group = sanitize_label(experiment_group)
    experiment_name = sanitize_label(experiment_name)
    source_run_id = sanitize_label(source_run_id)
    return experiment_group, experiment_name, source_run_id


def parse_threshold_metadata(config_path: Path) -> Tuple[str, str, str]:
    parts = list(config_path.parts)

    experiment_group = "unknown_group"
    experiment_name = "unknown_threshold_model"
    source_run_id = config_path.parent.name

    if "experiments" in parts:
        exp_idx = parts.index("experiments")
        if exp_idx + 1 < len(parts):
            experiment_group = parts[exp_idx + 1]

        if "artifacts" in parts:
            art_idx = parts.index("artifacts")
            if art_idx - 1 >= 0:
                experiment_name = parts[art_idx - 1]
            elif art_idx + 1 < len(parts):
                experiment_name = parts[art_idx + 1]
        elif "model_weights" in parts:
            mw_idx = parts.index("model_weights")
            if mw_idx - 1 >= 0:
                experiment_name = parts[mw_idx - 1]
            elif mw_idx + 1 < len(parts):
                experiment_name = parts[mw_idx + 1]

    experiment_group = sanitize_label(experiment_group)
    experiment_name = sanitize_label(experiment_name)
    source_run_id = sanitize_label(source_run_id)
    return experiment_group, experiment_name, source_run_id


def safe_load_json(json_path: Path) -> Dict[str, object]:
    with open(json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def debug_print_candidates(candidates: Sequence[GradCAMCandidate]) -> None:
    grouped: Dict[str, List[GradCAMCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.experiment_group].append(candidate)

    print("\nDiscovered candidates by experiment group")
    print("-" * 80)
    for group_name, items in sorted(grouped.items()):
        print(f"{group_name}: {len(items)} candidate(s)")
        for item in items:
            print(
                f"  - type={item.candidate_type:<9} "
                f"name={item.experiment_name:<35} "
                f"run={item.source_run_id:<25} "
                f"source={item.source_path}"
            )
    print("-" * 80)


def discover_candidates(
        checkpoint: Optional[str],
        threshold_config: Optional[str],
        search_dir: str,
) -> List[GradCAMCandidate]:
    if checkpoint and threshold_config:
        raise ValueError("Use either --checkpoint or --threshold_config, not both.")

    if checkpoint:
        checkpoint_path = resolve_flexible_path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        experiment_group, experiment_name, source_run_id = parse_checkpoint_metadata(checkpoint_path)
        return [
            GradCAMCandidate(
                candidate_type="model",
                source_path=checkpoint_path,
                model_path=checkpoint_path,
                experiment_group=experiment_group,
                experiment_name=experiment_name,
                source_run_id=source_run_id,
                architecture_hint=infer_architecture_hint(checkpoint_path),
                threshold_value=None,
            )
        ]

    if threshold_config:
        config_path = resolve_flexible_path(threshold_config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Threshold config not found: {config_path}")

        data = safe_load_json(config_path)
        source_model_path_raw = data.get("source_model_path")
        threshold_value_raw = (
                data.get("selected_threshold")
                or data.get("best_threshold")
                or data.get("threshold")
        )
        if source_model_path_raw is None:
            raise ValueError(f"Threshold config missing source_model_path: {config_path}")
        if threshold_value_raw is None:
            raise ValueError(
                f"Threshold config missing selected_threshold/best_threshold/threshold: {config_path}"
            )

        model_path = resolve_flexible_path(str(source_model_path_raw))
        if not model_path.is_file():
            raise FileNotFoundError(f"Source model from threshold config not found: {model_path}")

        experiment_group, experiment_name, source_run_id = parse_threshold_metadata(config_path)
        return [
            GradCAMCandidate(
                candidate_type="threshold",
                source_path=config_path,
                model_path=model_path,
                experiment_group=experiment_group,
                experiment_name=experiment_name,
                source_run_id=source_run_id,
                architecture_hint=infer_architecture_hint(model_path),
                threshold_value=float(threshold_value_raw),
            )
        ]

    search_root = resolve_flexible_path(search_dir)
    if not search_root.exists():
        raise FileNotFoundError(f"Search directory not found: {search_root}")

    print(f"Resolved search root  : {search_root}")

    model_patterns = [
        "best_model.pt",
        "best_model.pth",
        "*best*.pt",
        "*best*.pth",
        "*best*.ckpt",
        "*best*.pth.tar",
    ]

    model_paths: List[Path] = []
    for pattern in model_patterns:
        model_paths.extend(search_root.rglob(pattern))

    unique_model_paths: List[Path] = []
    seen_models = set()
    for path in model_paths:
        resolved = path.resolve()
        if resolved not in seen_models and resolved.is_file():
            seen_models.add(resolved)
            unique_model_paths.append(resolved)

    threshold_patterns = [
        "best_threshold_config.json",
        "*threshold*config*.json",
        "*best*threshold*.json",
    ]

    threshold_paths: List[Path] = []
    for pattern in threshold_patterns:
        threshold_paths.extend(search_root.rglob(pattern))

    unique_threshold_paths: List[Path] = []
    seen_thresholds = set()
    for path in threshold_paths:
        resolved = path.resolve()
        if resolved not in seen_thresholds and resolved.is_file():
            seen_thresholds.add(resolved)
            unique_threshold_paths.append(resolved)

    candidates: List[GradCAMCandidate] = []

    for checkpoint_path in unique_model_paths:
        experiment_group, experiment_name, source_run_id = parse_checkpoint_metadata(checkpoint_path)
        candidates.append(
            GradCAMCandidate(
                candidate_type="model",
                source_path=checkpoint_path,
                model_path=checkpoint_path,
                experiment_group=experiment_group,
                experiment_name=experiment_name,
                source_run_id=source_run_id,
                architecture_hint=infer_architecture_hint(checkpoint_path),
                threshold_value=None,
            )
        )

    for config_path in unique_threshold_paths:
        try:
            data = safe_load_json(config_path)
        except Exception as exc:
            print(f"Skipping unreadable threshold config: {config_path} ({exc})")
            continue

        source_model_path_raw = data.get("source_model_path")
        threshold_value_raw = (
                data.get("selected_threshold")
                or data.get("best_threshold")
                or data.get("threshold")
        )

        if source_model_path_raw is None or threshold_value_raw is None:
            print(f"Skipping incomplete threshold config: {config_path}")
            continue

        model_path = resolve_flexible_path(str(source_model_path_raw))
        if not model_path.is_file():
            print(f"Skipping threshold config; source model not found: {config_path}")
            continue

        experiment_group, experiment_name, source_run_id = parse_threshold_metadata(config_path)
        candidates.append(
            GradCAMCandidate(
                candidate_type="threshold",
                source_path=config_path.resolve(),
                model_path=model_path,
                experiment_group=experiment_group,
                experiment_name=experiment_name,
                source_run_id=source_run_id,
                architecture_hint=infer_architecture_hint(model_path),
                threshold_value=float(threshold_value_raw),
            )
        )

    if not candidates:
        raise FileNotFoundError(
            f"No best-model checkpoints or best-threshold configs found under: {search_root}"
        )

    candidates.sort(
        key=lambda item: (item.source_path.stat().st_mtime, str(item.source_path)),
        reverse=True,
    )
    return candidates


def ensure_output_dirs(
        save_root: str,
        candidate: GradCAMCandidate,
        clean_previous_outputs: bool,
) -> Dict[str, Path]:
    root_dir = (
            resolve_flexible_path(save_root)
            / candidate.experiment_group
            / candidate.experiment_name
            / candidate.source_run_id
    )

    if candidate.candidate_type == "threshold" and candidate.threshold_value is not None:
        threshold_tag = f"threshold_{str(candidate.threshold_value).replace('.', '_')}"
        root_dir = root_dir / threshold_tag

    if clean_previous_outputs and root_dir.exists():
        shutil.rmtree(root_dir)

    gradcam_run_dir = root_dir / datetime.now().strftime("gradcam_run_%Y%m%d_%H%M%S")
    reports_dir = gradcam_run_dir / "reports"

    gradcam_run_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "root_dir": root_dir,
        "run_dir": gradcam_run_dir,
        "reports_dir": reports_dir,
    }


def ensure_comparison_dir(save_root: str) -> Path:
    root = resolve_flexible_path(save_root) / "comparisons" / datetime.now().strftime("compare_run_%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    return root


def safe_load_state_dict(checkpoint_path: Path, device: str):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_test_dataset(data_dir: str) -> ImageDataset:
    data_root = resolve_flexible_path(data_dir)
    x_test = data_root / "X_test.npy"
    y_test = data_root / "Y_test.npy"
    if not x_test.is_file():
        raise FileNotFoundError(f"Missing test array: {x_test}")
    if not y_test.is_file():
        raise FileNotFoundError(f"Missing test labels: {y_test}")
    return ImageDataset(str(x_test), str(y_test))


def tensor_to_display_image(image_tensor: torch.Tensor) -> np.ndarray:
    return image_tensor.squeeze().detach().cpu().numpy()


def get_case_type(true_class: int, pred_class: int) -> str:
    return "correct_cases" if true_class == pred_class else "wrong_cases"


def get_pair_folder(run_dir: Path, case_type: str, true_class: int, pred_class: int) -> Path:
    folder = run_dir / case_type / f"true_{true_class}_pred_{pred_class}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_class_label(class_idx: int) -> str:
    return f"class {class_idx} ({CLASS_NAME_MAP.get(class_idx, 'Unknown')})"


def save_gradcam_figure(
        image_np: np.ndarray,
        pred_cam: np.ndarray,
        true_cam: np.ndarray,
        true_class: int,
        pred_class: int,
        pred_prob: float,
        true_prob: float,
        sample_index: int,
        run_dir: Path,
) -> Path:
    case_type = get_case_type(true_class, pred_class)
    pair_folder = get_pair_folder(run_dir, case_type, true_class, pred_class)
    correctness = "correct" if true_class == pred_class else "wrong"

    output_path = pair_folder / (
        f"sample_{sample_index:04d}_true_{true_class}_pred_{pred_class}_{correctness}.png"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title(f"Original Image\nTrue: {get_class_label(true_class)}", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(image_np, cmap="gray")
    axes[1].imshow(pred_cam, cmap="jet", alpha=0.4)
    axes[1].set_title(
        f"Predicted-class CAM\nPred: {get_class_label(pred_class)}\nProb: {pred_prob:.3f}",
        fontsize=11,
    )
    axes[1].axis("off")

    axes[2].imshow(image_np, cmap="gray")
    axes[2].imshow(true_cam, cmap="jet", alpha=0.4)
    axes[2].set_title(
        f"True-class CAM\nTrue: {get_class_label(true_class)}\nProb: {true_prob:.3f}",
        fontsize=11,
    )
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_prediction_summary_csv(records: Sequence[Dict[str, object]], reports_dir: Path) -> Path:
    csv_path = reports_dir / "prediction_summary.csv"
    fieldnames = [
        "sample_index",
        "true_class",
        "true_class_name",
        "pred_class",
        "pred_class_name",
        "correctness",
        "pred_prob",
        "true_prob",
        "max_prob",
        "passes_threshold",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "sample_index": row["sample_index"],
                    "true_class": row["true_class"],
                    "true_class_name": CLASS_NAME_MAP[int(row["true_class"])],
                    "pred_class": row["pred_class"],
                    "pred_class_name": CLASS_NAME_MAP[int(row["pred_class"])],
                    "correctness": "correct" if row["true_class"] == row["pred_class"] else "wrong",
                    "pred_prob": f"{float(row['pred_prob']):.6f}",
                    "true_prob": f"{float(row['true_prob']):.6f}",
                    "max_prob": f"{float(row['max_prob']):.6f}",
                    "passes_threshold": row["passes_threshold"],
                }
            )
    return csv_path


def save_selection_summary_csv(selected_entries: Sequence[Dict[str, object]], reports_dir: Path) -> Path:
    csv_path = reports_dir / "selection_summary.csv"
    fieldnames = [
        "filename",
        "case_type",
        "sample_index",
        "true_class",
        "true_class_name",
        "pred_class",
        "pred_class_name",
        "pair",
        "pair_label",
        "pred_prob",
        "true_prob",
        "max_prob",
        "passes_threshold",
        "output_path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_entries:
            true_class = int(row["true_class"])
            pred_class = int(row["pred_class"])
            writer.writerow(
                {
                    "filename": Path(str(row["output_path"])).name,
                    "case_type": row["case_type"],
                    "sample_index": row["sample_index"],
                    "true_class": true_class,
                    "true_class_name": CLASS_NAME_MAP[true_class],
                    "pred_class": pred_class,
                    "pred_class_name": CLASS_NAME_MAP[pred_class],
                    "pair": f"true_{true_class}_pred_{pred_class}",
                    "pair_label": f"{CLASS_NAME_MAP[true_class]} -> {CLASS_NAME_MAP[pred_class]}",
                    "pred_prob": f"{float(row['pred_prob']):.6f}",
                    "true_prob": f"{float(row['true_prob']):.6f}",
                    "max_prob": f"{float(row['max_prob']):.6f}",
                    "passes_threshold": row["passes_threshold"],
                    "output_path": row["output_path"],
                }
            )
    return csv_path


def plot_confusion_matrix(records: Sequence[Dict[str, object]], n_classes: int, reports_dir: Path) -> Path:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for row in records:
        cm[int(row["true_class"]), int(row["pred_class"])] += 1

    class_names = [CLASS_NAME_MAP.get(i, str(i)) for i in range(n_classes)]
    fig, ax = plt.subplots(figsize=(8.8, 7.2), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)

    threshold = cm.max() * 0.5 if cm.max() > 0 else 0
    for i in range(n_classes):
        for j in range(n_classes):
            value = cm[i, j]
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = reports_dir / "confusion_matrix_pretty.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_correct_wrong_counts(records: Sequence[Dict[str, object]], reports_dir: Path) -> Path:
    correct_count = sum(1 for row in records if row["true_class"] == row["pred_class"])
    wrong_count = len(records) - correct_count
    labels = ["Correct", "Wrong"]
    values = [correct_count, wrong_count]

    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=150)
    bars = ax.bar(labels, values)
    ax.set_title("Correct vs Wrong Predictions")
    ax.set_ylabel("Number of samples")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    out_path = reports_dir / "correct_vs_wrong_counts_pretty.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_top_confusion_pairs(
        records: Sequence[Dict[str, object]],
        reports_dir: Path,
        top_k: int = 10,
) -> Path:
    wrong_pairs: Counter = Counter()
    for row in records:
        true_class = int(row["true_class"])
        pred_class = int(row["pred_class"])
        if true_class != pred_class:
            wrong_pairs[(true_class, pred_class)] += 1

    top_items = wrong_pairs.most_common(top_k)
    labels = [f"{CLASS_NAME_MAP[t]} -> {CLASS_NAME_MAP[p]}" for (t, p), _ in top_items]
    values = [count for _, count in top_items]

    fig, ax = plt.subplots(figsize=(10.0, max(4.8, 0.6 * max(len(labels), 1))), dpi=150)

    if values:
        y = np.arange(len(labels))
        bars = ax.barh(y, values)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_title(f"Top {len(labels)} Wrong Confusion Pairs")

        for bar, value in zip(bars, values):
            ax.text(
                value,
                bar.get_y() + bar.get_height() / 2.0,
                f" {value}",
                va="center",
                fontsize=9,
            )
    else:
        ax.text(0.5, 0.5, "No wrong confusion pairs found.", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout()
    out_path = reports_dir / "top_confusion_pairs_pretty.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_text_report(
        candidate: GradCAMCandidate,
        n_classes: int,
        all_records: Sequence[Dict[str, object]],
        selected_entries: Sequence[Dict[str, object]],
        correct_per_pair: int,
        wrong_per_pair: int,
        reports_dir: Path,
        apply_threshold_filter: bool,
) -> Path:
    total = len(all_records)
    correct = sum(1 for row in all_records if row["true_class"] == row["pred_class"])
    wrong = total - correct
    accuracy = correct / total if total else 0.0

    passed_threshold = sum(1 for row in all_records if row["passes_threshold"])
    coverage = passed_threshold / total if total else 0.0

    wrong_pairs: Counter = Counter()
    for row in all_records:
        if row["true_class"] != row["pred_class"]:
            wrong_pairs[(int(row["true_class"]), int(row["pred_class"]))] += 1

    report_path = reports_dir / "report_summary.txt"
    lines = [
        "Grad-CAM Export and Prediction Summary",
        "=" * 50,
        f"Candidate type: {candidate.candidate_type}",
        f"Source path: {candidate.source_path}",
        f"Model path used: {candidate.model_path}",
        f"Experiment group: {candidate.experiment_group}",
        f"Experiment name: {candidate.experiment_name}",
        f"Source run id: {candidate.source_run_id}",
        f"Selected threshold: {candidate.threshold_value}",
        f"Threshold filter applied: {apply_threshold_filter}",
        f"Number of classes: {n_classes}",
        f"Total test samples evaluated: {total}",
        f"Correct predictions: {correct}",
        f"Wrong predictions: {wrong}",
        f"Accuracy: {accuracy:.4f}",
        f"Samples passing threshold: {passed_threshold}",
        f"Threshold coverage: {coverage:.4f}",
        "",
        "Export settings",
        "-" * 25,
        f"Correct samples saved per pair: {correct_per_pair}",
        f"Wrong samples saved per pair: {wrong_per_pair}",
        f"Total Grad-CAM figures saved: {len(selected_entries)}",
        "",
        "Top wrong confusion pairs",
        "-" * 25,
    ]

    if wrong_pairs:
        for (true_class, pred_class), count in wrong_pairs.most_common(10):
            lines.append(
                f"True {true_class} ({CLASS_NAME_MAP[true_class]}) -> "
                f"Pred {pred_class} ({CLASS_NAME_MAP[pred_class]}): {count}"
            )
    else:
        lines.append("No wrong confusion pairs found.")

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return report_path


def load_model_for_candidate(candidate: GradCAMCandidate, device: str, n_classes: int) -> nn.Module:
    if candidate.architecture_hint == "resnet18":
        mode = "frozen_resnet18" if "frozen" in str(candidate.model_path).lower() else "finetuned_resnet18"
        model = ResNet18Transfer(n_classes=n_classes, mode=mode).to(device)
    else:
        model = Net(n_classes=n_classes).to(device)
    state_dict = safe_load_state_dict(candidate.model_path, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_full_records_for_candidate(
        candidate: GradCAMCandidate,
        model: nn.Module,
        test_dataset: ImageDataset,
        device: str,
        apply_threshold_filter: bool,
) -> List[Dict[str, object]]:
    all_records: List[Dict[str, object]] = []

    with torch.no_grad():
        for sample_index in range(len(test_dataset)):
            input_image, label = test_dataset[sample_index]
            true_class = int(label.item()) if torch.is_tensor(label) else int(label)

            batched = input_image.unsqueeze(0).to(device)
            logits = model(batched)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

            pred_class = int(np.argmax(probs))
            pred_prob = float(probs[pred_class])
            true_prob = float(probs[true_class])
            max_prob = float(np.max(probs))

            passes_threshold = True
            if (
                    candidate.candidate_type == "threshold"
                    and apply_threshold_filter
                    and candidate.threshold_value is not None
            ):
                passes_threshold = max_prob >= float(candidate.threshold_value)

            all_records.append(
                {
                    "sample_index": sample_index,
                    "true_class": true_class,
                    "pred_class": pred_class,
                    "pred_prob": pred_prob,
                    "true_prob": true_prob,
                    "max_prob": max_prob,
                    "passes_threshold": passes_threshold,
                }
            )
    return all_records


def select_records_for_export(
        all_records: Sequence[Dict[str, object]],
        correct_samples_per_pair: int,
        wrong_samples_per_pair: int,
) -> List[Dict[str, object]]:
    selected_records: List[Dict[str, object]] = []
    saved_correct_count_per_pair: Dict[tuple, int] = defaultdict(int)
    saved_wrong_count_per_pair: Dict[tuple, int] = defaultdict(int)

    for record in all_records:
        if not bool(record["passes_threshold"]):
            continue

        true_class = int(record["true_class"])
        pred_class = int(record["pred_class"])
        pair_key = (true_class, pred_class)

        if true_class == pred_class:
            if saved_correct_count_per_pair[pair_key] < correct_samples_per_pair:
                selected_records.append(dict(record))
                saved_correct_count_per_pair[pair_key] += 1
        else:
            if saved_wrong_count_per_pair[pair_key] < wrong_samples_per_pair:
                selected_records.append(dict(record))
                saved_wrong_count_per_pair[pair_key] += 1

    return selected_records


def run_single_candidate(
        candidate: GradCAMCandidate,
        device: str,
        test_dataset: ImageDataset,
        args: argparse.Namespace,
) -> CandidateRunArtifacts:
    dirs = ensure_output_dirs(
        save_root=args.save_root,
        candidate=candidate,
        clean_previous_outputs=args.clean_previous_outputs,
    )
    run_dir = dirs["run_dir"]
    reports_dir = dirs["reports_dir"]

    print(f"\nSource path           : {candidate.source_path}")
    print(f"Candidate type        : {candidate.candidate_type}")
    print(f"Model path used       : {candidate.model_path}")
    print(f"Experiment group      : {candidate.experiment_group}")
    print(f"Experiment name       : {candidate.experiment_name}")
    print(f"Source run id         : {candidate.source_run_id}")
    print(f"Selected threshold    : {candidate.threshold_value}")
    print(f"Grad-CAM output dir   : {run_dir}")

    model = load_model_for_candidate(candidate, device=device, n_classes=args.n_classes)

    if candidate.architecture_hint == "resnet18":
        target_layer = model.model.layer4[-1]
    else:
        target_layer = model.cnn_layers[10]

    gradcam = GradCAM(model, target_layer)

    print(f"Scanning {len(test_dataset)} test samples...")
    all_records = compute_full_records_for_candidate(
        candidate=candidate,
        model=model,
        test_dataset=test_dataset,
        device=device,
        apply_threshold_filter=args.apply_threshold_filter,
    )

    selected_records = select_records_for_export(
        all_records=all_records,
        correct_samples_per_pair=args.correct_samples_per_pair,
        wrong_samples_per_pair=args.wrong_samples_per_pair,
    )
    print(f"Selected {len(selected_records)} samples for Grad-CAM export.")

    selected_entries: List[Dict[str, object]] = []
    for record in selected_records:
        sample_index = int(record["sample_index"])
        true_class = int(record["true_class"])
        pred_class = int(record["pred_class"])
        pred_prob = float(record["pred_prob"])
        true_prob = float(record["true_prob"])
        max_prob = float(record["max_prob"])
        passes_threshold = bool(record["passes_threshold"])

        input_image, _ = test_dataset[sample_index]
        input_image = input_image.unsqueeze(0).to(device)

        result = gradcam.generate_pred_true(input_image, true_class_idx=true_class)
        pred_cam = result["pred_cam"]
        true_cam = result["true_cam"]
        image_np = tensor_to_display_image(input_image[0])

        output_path = save_gradcam_figure(
            image_np=image_np,
            pred_cam=pred_cam,
            true_cam=true_cam,
            true_class=true_class,
            pred_class=pred_class,
            pred_prob=pred_prob,
            true_prob=true_prob,
            sample_index=sample_index,
            run_dir=run_dir,
        )

        selected_entries.append(
            {
                "output_path": str(output_path),
                "case_type": get_case_type(true_class, pred_class),
                "sample_index": sample_index,
                "true_class": true_class,
                "pred_class": pred_class,
                "pred_prob": pred_prob,
                "true_prob": true_prob,
                "max_prob": max_prob,
                "passes_threshold": passes_threshold,
            }
        )

    gradcam.remove_hooks()

    prediction_csv = save_prediction_summary_csv(all_records, reports_dir)
    selection_csv = save_selection_summary_csv(selected_entries, reports_dir)
    cm_png = plot_confusion_matrix(all_records, args.n_classes, reports_dir)
    counts_png = plot_correct_wrong_counts(all_records, reports_dir)
    top_pairs_png = plot_top_confusion_pairs(all_records, reports_dir, top_k=10)
    summary_txt = save_text_report(
        candidate=candidate,
        n_classes=args.n_classes,
        all_records=all_records,
        selected_entries=selected_entries,
        correct_per_pair=args.correct_samples_per_pair,
        wrong_per_pair=args.wrong_samples_per_pair,
        reports_dir=reports_dir,
        apply_threshold_filter=args.apply_threshold_filter,
    )

    accuracy = (
        sum(1 for row in all_records if row["true_class"] == row["pred_class"]) / len(all_records)
        if all_records else 0.0
    )
    passed_threshold = sum(1 for row in all_records if row["passes_threshold"])
    coverage = passed_threshold / len(all_records) if all_records else 0.0

    print(f"Prediction summary CSV : {prediction_csv}")
    print(f"Selection summary CSV  : {selection_csv}")
    print(f"Confusion matrix       : {cm_png}")
    print(f"Correct vs wrong plot  : {counts_png}")
    print(f"Top confusion plot     : {top_pairs_png}")
    print(f"Text summary           : {summary_txt}")

    return CandidateRunArtifacts(
        candidate=candidate,
        run_dir=run_dir,
        reports_dir=reports_dir,
        all_records=all_records,
        selected_entries=selected_entries,
        accuracy=accuracy,
        coverage=coverage,
    )


def save_batch_summary(results: Sequence[CandidateRunArtifacts], save_root: str) -> Path:
    root = resolve_flexible_path(save_root)
    root.mkdir(parents=True, exist_ok=True)

    summary_path = root / f"batch_gradcam_summary_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_type",
                "source_path",
                "model_path_used",
                "experiment_group",
                "experiment_name",
                "source_run_id",
                "threshold_value",
                "accuracy",
                "threshold_coverage",
                "saved_gradcam_figures",
                "output_dir",
            ],
        )
        writer.writeheader()
        for artifact in results:
            candidate = artifact.candidate
            writer.writerow(
                {
                    "candidate_type": candidate.candidate_type,
                    "source_path": str(candidate.source_path),
                    "model_path_used": str(candidate.model_path),
                    "experiment_group": candidate.experiment_group,
                    "experiment_name": candidate.experiment_name,
                    "source_run_id": candidate.source_run_id,
                    "threshold_value": candidate.threshold_value,
                    "accuracy": artifact.accuracy,
                    "threshold_coverage": artifact.coverage,
                    "saved_gradcam_figures": len(artifact.selected_entries),
                    "output_dir": str(artifact.run_dir),
                }
            )

    return summary_path


def choose_comparison_sample_indices(
        artifacts: Sequence[CandidateRunArtifacts],
        max_samples: int,
) -> List[int]:
    if not artifacts:
        return []

    common: Optional[set] = None
    for artifact in artifacts:
        selected_ids = {int(row["sample_index"]) for row in artifact.selected_entries}
        if common is None:
            common = selected_ids
        else:
            common &= selected_ids

    if not common:
        sample_counter: Counter[int] = Counter()
        for artifact in artifacts:
            sample_counter.update(int(row["sample_index"]) for row in artifact.selected_entries)
        ranked = [idx for idx, _ in sample_counter.most_common(max_samples)]
        return ranked[:max_samples]

    ranked_common = sorted(common)
    return ranked_common[:max_samples]


def generate_compare_cam(
        candidate: GradCAMCandidate,
        device: str,
        input_tensor: torch.Tensor,
        true_class: int,
        n_classes: int,
) -> Dict[str, object]:
    model = load_model_for_candidate(candidate, device=device, n_classes=n_classes)

    if candidate.architecture_hint == "resnet18":
        target_layer = model.model.layer4[-1]
    else:
        target_layer = model.cnn_layers[10]

    gradcam = GradCAM(model, target_layer)

    with torch.no_grad():
        logits = model(input_tensor.to(device))
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred_class = int(np.argmax(probs))
    pred_prob = float(probs[pred_class])
    true_prob = float(probs[true_class])
    max_prob = float(np.max(probs))

    result = gradcam.generate_pred_true(input_tensor.to(device), true_class_idx=true_class)
    pred_cam = result["pred_cam"]
    true_cam = result["true_cam"]
    gradcam.remove_hooks()

    passes_threshold = True
    if candidate.candidate_type == "threshold" and candidate.threshold_value is not None:
        passes_threshold = max_prob >= float(candidate.threshold_value)

    return {
        "pred_class": pred_class,
        "pred_prob": pred_prob,
        "true_prob": true_prob,
        "max_prob": max_prob,
        "pred_cam": pred_cam,
        "true_cam": true_cam,
        "passes_threshold": passes_threshold,
    }


def save_comparison_panel(
        sample_index: int,
        image_np: np.ndarray,
        true_class: int,
        comparison_rows: Sequence[Tuple[GradCAMCandidate, Dict[str, object]]],
        out_dir: Path,
) -> Tuple[Path, Path]:
    n_models = len(comparison_rows)
    n_cols = 3
    n_rows = n_models + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

    if n_rows == 1:
        axes = np.array([axes])

    axes[0, 0].imshow(image_np, cmap="gray")
    axes[0, 0].set_title(f"Original\nTrue: {get_class_label(true_class)}", fontsize=12)
    axes[0, 0].axis("off")

    axes[0, 1].axis("off")
    axes[0, 2].axis("off")

    summary_lines = [
        f"Sample index: {sample_index}",
        f"True class: {true_class} ({CLASS_NAME_MAP[true_class]})",
        "",
    ]

    for row_idx, (candidate, info) in enumerate(comparison_rows, start=1):
        pred_class = int(info["pred_class"])
        pred_prob = float(info["pred_prob"])
        true_prob = float(info["true_prob"])
        max_prob = float(info["max_prob"])
        passes_threshold = bool(info["passes_threshold"])

        axes[row_idx, 0].imshow(image_np, cmap="gray")
        axes[row_idx, 0].set_title(candidate_display_name(candidate), fontsize=11)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(image_np, cmap="gray")
        axes[row_idx, 1].imshow(info["pred_cam"], cmap="jet", alpha=0.4)
        axes[row_idx, 1].set_title(
            f"Pred-CAM\nPred: {get_class_label(pred_class)}\nProb: {pred_prob:.3f}",
            fontsize=10,
        )
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(image_np, cmap="gray")
        axes[row_idx, 2].imshow(info["true_cam"], cmap="jet", alpha=0.4)
        axes[row_idx, 2].set_title(
            f"True-CAM\nTrue Prob: {true_prob:.3f}\nMax Prob: {max_prob:.3f}",
            fontsize=10,
        )
        axes[row_idx, 2].axis("off")

        summary_lines.extend(
            [
                f"Candidate: {candidate_display_name(candidate)}",
                f"  candidate_type: {candidate.candidate_type}",
                f"  pred_class: {pred_class} ({CLASS_NAME_MAP[pred_class]})",
                f"  pred_prob: {pred_prob:.6f}",
                f"  true_prob: {true_prob:.6f}",
                f"  max_prob: {max_prob:.6f}",
                f"  selected_threshold: {candidate.threshold_value}",
                f"  passes_threshold: {passes_threshold}",
                "",
            ]
        )

    fig.tight_layout()

    image_path = out_dir / f"sample_{sample_index:04d}_compare.png"
    text_path = out_dir / f"sample_{sample_index:04d}_summary.txt"

    fig.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")

    return image_path, text_path


def run_same_sample_multi_model_comparison(
        artifacts: Sequence[CandidateRunArtifacts],
        test_dataset: ImageDataset,
        device: str,
        n_classes: int,
        save_root: str,
        max_samples: int,
) -> Optional[Path]:
    if len(artifacts) < 2:
        print("Skipping comparison: need at least 2 candidates.")
        return None

    comparison_dir = ensure_comparison_dir(save_root)
    chosen_sample_indices = choose_comparison_sample_indices(artifacts, max_samples=max_samples)

    if not chosen_sample_indices:
        print("Skipping comparison: no suitable shared or frequent samples found.")
        return comparison_dir

    for sample_index in chosen_sample_indices:
        input_image, label = test_dataset[sample_index]
        true_class = int(label.item()) if torch.is_tensor(label) else int(label)
        image_np = tensor_to_display_image(input_image)
        input_tensor = input_image.unsqueeze(0)

        comparison_rows: List[Tuple[GradCAMCandidate, Dict[str, object]]] = []
        for artifact in artifacts:
            info = generate_compare_cam(
                candidate=artifact.candidate,
                device=device,
                input_tensor=input_tensor,
                true_class=true_class,
                n_classes=n_classes,
            )
            comparison_rows.append((artifact.candidate, info))

        save_comparison_panel(
            sample_index=sample_index,
            image_np=image_np,
            true_class=true_class,
            comparison_rows=comparison_rows,
            out_dir=comparison_dir,
        )

    return comparison_dir


def main() -> None:
    args = parse_args()
    device = choose_device(force_cpu=args.force_cpu)

    print(f"Resolved data dir     : {resolve_flexible_path(args.data_dir)}")
    print(f"Resolved save root    : {resolve_flexible_path(args.save_root)}")

    candidates = discover_candidates(args.checkpoint, args.threshold_config, args.search_dir)

    print("RUNNING BATCH EXPERIMENT VERSION OF Grad-CAM")
    print(f"Using device          : {device}")
    print(f"Total candidates      : {len(candidates)}")

    debug_print_candidates(candidates)

    for idx, candidate in enumerate(candidates, start=1):
        print(
            f"[{idx:02d}] type={candidate.candidate_type} "
            f"group={candidate.experiment_group} "
            f"name={candidate.experiment_name} "
            f"run={candidate.source_run_id} "
            f"source={candidate.source_path} "
            f"model={candidate.model_path}"
        )

    test_dataset = load_test_dataset(args.data_dir)

    all_artifacts: List[CandidateRunArtifacts] = []
    for idx, candidate in enumerate(candidates, start=1):
        print(f"\n========== [{idx}/{len(candidates)}] Processing ==========")
        artifact = run_single_candidate(
            candidate=candidate,
            device=device,
            test_dataset=test_dataset,
            args=args,
        )
        all_artifacts.append(artifact)

    batch_summary_path = save_batch_summary(all_artifacts, args.save_root)

    comparison_dir = None
    if args.comparison_mode == "same_sample_multi_model":
        comparison_dir = run_same_sample_multi_model_comparison(
            artifacts=all_artifacts,
            test_dataset=test_dataset,
            device=device,
            n_classes=args.n_classes,
            save_root=args.save_root,
            max_samples=args.comparison_max_samples,
        )

    print("\nDone.")
    print(f"Batch summary CSV     : {batch_summary_path}")
    if comparison_dir is not None:
        print(f"Comparison output dir : {comparison_dir}")


if __name__ == "__main__":
    main()