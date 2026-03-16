import os
import glob
import csv
import shutil
import argparse
from datetime import datetime
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from net import Net
from image_dataset import ImageDataset
from gradcam import GradCAM


# ======================================================
# Class mapping
# ======================================================

CLASS_NAME_MAP = {
    0: "Atelectasis",
    1: "Effusion",
    2: "Infiltration",
    3: "No Finding",
    4: "Nodule",
    5: "Pneumothorax",
}


# ======================================================
# Argument parsing
# ======================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Grad-CAM on the stratified best-model checkpoint and export reports."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific best-model checkpoint. If not provided, the script will search automatically."
    )

    parser.add_argument(
        "--model_weights_dir",
        type=str,
        default=None,
        help="Root directory containing model weights. Default: <script_dir>/model_weights"
    )

    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help="Output directory for Grad-CAM results. Default: <script_dir>/gradcam_results_stratified_best"
    )

    parser.add_argument(
        "--correct_samples_per_pair",
        type=int,
        default=2,
        help="Number of correctly classified samples to save per (true, pred) pair."
    )

    parser.add_argument(
        "--wrong_samples_per_pair",
        type=int,
        default=3,
        help="Number of wrongly classified samples to save per (true, pred) pair."
    )

    parser.add_argument(
        "--clean_previous_outputs",
        action="store_true",
        help="If set, delete the previous output folder before running."
    )

    parser.add_argument(
        "--n_classes",
        type=int,
        default=6,
        help="Number of classes in the model."
    )

    return parser.parse_args()


# ======================================================
# Checkpoint selection
# ======================================================

def find_best_checkpoint(manual_path=None, model_weights_dir=None):
    """
    Priority:
    1. Use the manually specified checkpoint if provided.
    2. Otherwise search recursively for the newest best-model checkpoint.

    The search is intended for folders such as:
        model_weights/
            model_weights_baseline_best/
                run_YYYYMMDD_HHMMSS/
                    best_model.pt
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if model_weights_dir is None:
        model_weights_dir = os.path.join(script_dir, "model_weights")

    model_weights_dir = os.path.abspath(model_weights_dir)

    if manual_path is not None and str(manual_path).strip() != "":
        checkpoint_path = os.path.abspath(manual_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Manual checkpoint path not found: {checkpoint_path}")

        mtime = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
        print(f"[Checkpoint] Manual checkpoint selected: {checkpoint_path}")
        print(f"[Checkpoint] Last modified: {mtime}")
        return checkpoint_path

    candidate_patterns = [
        os.path.join(model_weights_dir, "**", "best_model.pt"),
        os.path.join(model_weights_dir, "**", "*best*.pt"),
        os.path.join(model_weights_dir, "**", "*best*.pth"),
    ]

    candidates = []
    for pattern in candidate_patterns:
        candidates.extend(glob.glob(pattern, recursive=True))

    if not candidates:
        raise FileNotFoundError(
            f"No best-model checkpoint files found under: {model_weights_dir}"
        )

    checkpoint_path = max(candidates, key=os.path.getmtime)
    mtime = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))

    print(f"[Checkpoint] Newest best-model checkpoint selected automatically: {checkpoint_path}")
    print(f"[Checkpoint] Last modified: {mtime}")
    return checkpoint_path


# ======================================================
# Dataset loading
# ======================================================

def load_test_dataset():
    """
    Load the official test set.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    x_test_path = os.path.join(script_dir, "data", "X_test.npy")
    y_test_path = os.path.join(script_dir, "data", "Y_test.npy")

    if not os.path.exists(x_test_path):
        raise FileNotFoundError(f"X_test.npy not found at: {x_test_path}")
    if not os.path.exists(y_test_path):
        raise FileNotFoundError(f"Y_test.npy not found at: {y_test_path}")

    return ImageDataset(x_test_path, y_test_path)


# ======================================================
# Utility helpers
# ======================================================

def load_checkpoint_safely(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def tensor_to_display_image(image_tensor):
    return image_tensor.squeeze().detach().cpu().numpy()


def get_case_type(true_class, pred_class):
    return "correct_cases" if true_class == pred_class else "wrong_cases"


def get_pair_folder(save_root, case_type, true_class, pred_class):
    folder = os.path.join(
        save_root,
        case_type,
        f"true_{true_class}_pred_{pred_class}",
    )
    os.makedirs(folder, exist_ok=True)
    return folder


def get_class_label(class_idx):
    return f"class {class_idx} ({CLASS_NAME_MAP.get(class_idx, 'Unknown')})"


# ======================================================
# Save Grad-CAM figure
# ======================================================

def save_gradcam_figure(
    image_np,
    pred_cam,
    true_cam,
    true_class,
    pred_class,
    pred_prob,
    true_prob,
    sample_index,
    save_root,
):
    case_type = get_case_type(true_class, pred_class)
    pair_folder = get_pair_folder(save_root, case_type, true_class, pred_class)
    correctness = "correct" if true_class == pred_class else "wrong"

    file_name = (
        f"sample_{sample_index:04d}"
        f"_true_{true_class}"
        f"_pred_{pred_class}"
        f"_{correctness}.png"
    )
    output_path = os.path.join(pair_folder, file_name)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title(
        f"Original Image\nTrue: {get_class_label(true_class)}",
        fontsize=11
    )
    axes[0].axis("off")

    axes[1].imshow(image_np, cmap="gray")
    axes[1].imshow(pred_cam, cmap="jet", alpha=0.4)
    axes[1].set_title(
        f"Predicted-class CAM\nPred: {get_class_label(pred_class)}\nProb: {pred_prob:.3f}",
        fontsize=11
    )
    axes[1].axis("off")

    axes[2].imshow(image_np, cmap="gray")
    axes[2].imshow(true_cam, cmap="jet", alpha=0.4)
    axes[2].set_title(
        f"True-class CAM\nTrue: {get_class_label(true_class)}\nProb: {true_prob:.3f}",
        fontsize=11
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


# ======================================================
# Report CSVs
# ======================================================

def save_prediction_summary_csv(records, report_dir):
    csv_path = os.path.join(report_dir, "prediction_summary.csv")
    fieldnames = [
        "sample_index",
        "true_class",
        "true_class_name",
        "pred_class",
        "pred_class_name",
        "correctness",
        "pred_prob",
        "true_prob",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in records:
            writer.writerow(
                {
                    "sample_index": r["sample_index"],
                    "true_class": r["true_class"],
                    "true_class_name": CLASS_NAME_MAP.get(r["true_class"], "Unknown"),
                    "pred_class": r["pred_class"],
                    "pred_class_name": CLASS_NAME_MAP.get(r["pred_class"], "Unknown"),
                    "correctness": "correct" if r["true_class"] == r["pred_class"] else "wrong",
                    "pred_prob": f"{r['pred_prob']:.6f}",
                    "true_prob": f"{r['true_prob']:.6f}",
                }
            )

    return csv_path


def save_selection_summary_csv(selected_entries, report_dir):
    csv_path = os.path.join(report_dir, "selection_summary.csv")
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
        "output_path",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for e in selected_entries:
            true_name = CLASS_NAME_MAP.get(e["true_class"], "Unknown")
            pred_name = CLASS_NAME_MAP.get(e["pred_class"], "Unknown")

            writer.writerow(
                {
                    "filename": os.path.basename(e["output_path"]),
                    "case_type": e["case_type"],
                    "sample_index": e["sample_index"],
                    "true_class": e["true_class"],
                    "true_class_name": true_name,
                    "pred_class": e["pred_class"],
                    "pred_class_name": pred_name,
                    "pair": f"true_{e['true_class']}_pred_{e['pred_class']}",
                    "pair_label": f"{true_name} -> {pred_name}",
                    "pred_prob": f"{e['pred_prob']:.6f}",
                    "true_prob": f"{e['true_prob']:.6f}",
                    "output_path": e["output_path"],
                }
            )

    return csv_path


def save_text_report(
    checkpoint_path,
    n_classes,
    all_records,
    selected_entries,
    correct_per_pair,
    wrong_per_pair,
    report_dir,
):
    total = len(all_records)
    correct = sum(1 for r in all_records if r["true_class"] == r["pred_class"])
    wrong = total - correct
    accuracy = correct / total if total > 0 else 0.0

    wrong_pairs = Counter()
    for r in all_records:
        if r["true_class"] != r["pred_class"]:
            wrong_pairs[(r["true_class"], r["pred_class"])] += 1

    report_path = os.path.join(report_dir, "report_summary.txt")

    with open(report_path, "w") as f:
        f.write("Grad-CAM Export and Prediction Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Checkpoint used: {checkpoint_path}\n")
        f.write(f"Number of classes: {n_classes}\n")
        f.write(f"Total test samples evaluated: {total}\n")
        f.write(f"Correct predictions: {correct}\n")
        f.write(f"Wrong predictions: {wrong}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")

        f.write("Class name mapping\n")
        f.write("-" * 25 + "\n")
        for k in sorted(CLASS_NAME_MAP.keys()):
            f.write(f"Class {k}: {CLASS_NAME_MAP[k]}\n")

        f.write("\nExport settings\n")
        f.write("-" * 25 + "\n")
        f.write(f"Correct samples saved per pair: {correct_per_pair}\n")
        f.write(f"Wrong samples saved per pair: {wrong_per_pair}\n")
        f.write(f"Total Grad-CAM figures saved: {len(selected_entries)}\n\n")

        f.write("Top wrong confusion pairs\n")
        f.write("-" * 25 + "\n")
        if wrong_pairs:
            for (t, p), count in wrong_pairs.most_common(10):
                true_name = CLASS_NAME_MAP.get(t, "Unknown")
                pred_name = CLASS_NAME_MAP.get(p, "Unknown")
                f.write(f"True {t} ({true_name}) -> Pred {p} ({pred_name}): {count}\n")
        else:
            f.write("No wrong confusion pairs found.\n")

    return report_path


# ======================================================
# Plots
# ======================================================

def plot_confusion_matrix(all_records, n_classes, report_dir):
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for r in all_records:
        cm[r["true_class"], r["pred_class"]] += 1

    class_names = [CLASS_NAME_MAP.get(i, str(i)) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title("Confusion Matrix", fontsize=15, pad=12)
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("True class", fontsize=12)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    threshold = cm.max() * 0.5 if cm.max() > 0 else 0

    for i in range(n_classes):
        for j in range(n_classes):
            value = cm[i, j]
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()

    out_path = os.path.join(report_dir, "confusion_matrix_stratified.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_correct_wrong_counts(all_records, report_dir):
    correct_count = sum(1 for r in all_records if r["true_class"] == r["pred_class"])
    wrong_count = len(all_records) - correct_count

    labels = ["Correct", "Wrong"]
    values = [correct_count, wrong_count]
    colors = ["#5B8E7D", "#C06C84"]

    fig, ax = plt.subplots(figsize=(6, 4.8))
    bars = ax.bar(labels, values, color=colors, width=0.6)

    ax.set_title("Correct vs Wrong Predictions", fontsize=14, pad=10)
    ax.set_ylabel("Count", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01 if max(values) > 0 else bar.get_height() + 0.1,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    out_path = os.path.join(report_dir, "correct_vs_wrong_counts_stratified.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_top_confusion_pairs(all_records, report_dir, top_k=10):
    pair_counter = Counter()

    for r in all_records:
        if r["true_class"] != r["pred_class"]:
            pair_counter[(r["true_class"], r["pred_class"])] += 1

    top_pairs = pair_counter.most_common(top_k)

    fig, ax = plt.subplots(figsize=(12, 6))

    if top_pairs:
        labels = [
            f"{CLASS_NAME_MAP.get(t, t)} -> {CLASS_NAME_MAP.get(p, p)}"
            for (t, p), _ in top_pairs
        ]
        values = [count for _, count in top_pairs]

        bars = ax.bar(labels, values, color="#4C78A8", width=0.7)

        ax.set_title("Top Wrong Confusion Pairs", fontsize=15, pad=12)
        ax.set_xlabel("True class -> Predicted class", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.tick_params(axis="x", labelrotation=30, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01 if max(values) > 0 else bar.get_height() + 0.1,
                f"{value}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    else:
        ax.text(0.5, 0.5, "No wrong predictions found.", ha="center", va="center", fontsize=13)
        ax.axis("off")

    plt.tight_layout()

    out_path = os.path.join(report_dir, "top_confusion_pairs_stratified.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path


# ======================================================
# Main
# ======================================================

def main():
    args = parse_args()

    print("RUNNING THE STRATIFIED BEST-MODEL VERSION OF Grad-CAM")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = args.n_classes

    script_dir = os.path.dirname(os.path.abspath(__file__))

    save_root = args.save_root
    if save_root is None:
        save_root = os.path.join(script_dir, "gradcam_results_stratified_best")
    save_root = os.path.abspath(save_root)

    report_dir = os.path.join(save_root, "reports")

    if args.clean_previous_outputs and os.path.exists(save_root):
        shutil.rmtree(save_root)

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    checkpoint_path = find_best_checkpoint(
        manual_path=args.checkpoint,
        model_weights_dir=args.model_weights_dir,
    )

    print(f"[run_gradcam_stratified_best] Final checkpoint used: {checkpoint_path}")
    print(f"Grad-CAM output folder: {save_root}")
    print(f"Report output folder: {report_dir}")

    # Model
    model = Net(n_classes=n_classes).to(device)
    state_dict = load_checkpoint_safely(checkpoint_path, device)
    model.load_state_dict(state_dict)
    model.eval()

    # Dataset
    test_dataset = load_test_dataset()

    # Target layer
    target_layer = model.cnn_layers[10]
    gradcam = GradCAM(model, target_layer)

    # --------------------------------------------------
    # First pass: inference over full test set
    # --------------------------------------------------
    all_records = []
    selected_records = []

    saved_correct_count_per_pair = defaultdict(int)
    saved_wrong_count_per_pair = defaultdict(int)

    total_samples = len(test_dataset)
    print(f"Scanning {total_samples} test samples...")

    with torch.no_grad():
        for sample_index in range(total_samples):
            input_image, label = test_dataset[sample_index]

            if torch.is_tensor(label):
                true_class = int(label.item())
            else:
                true_class = int(label)

            batched = input_image.unsqueeze(0).to(device)
            logits = model(batched)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            pred_class = int(np.argmax(probs))
            pred_prob = float(probs[pred_class])
            true_prob = float(probs[true_class])

            record = {
                "sample_index": sample_index,
                "true_class": true_class,
                "pred_class": pred_class,
                "pred_prob": pred_prob,
                "true_prob": true_prob,
            }
            all_records.append(record)

            pair_key = (true_class, pred_class)

            if true_class == pred_class:
                if saved_correct_count_per_pair[pair_key] < args.correct_samples_per_pair:
                    selected_records.append(record)
                    saved_correct_count_per_pair[pair_key] += 1
            else:
                if saved_wrong_count_per_pair[pair_key] < args.wrong_samples_per_pair:
                    selected_records.append(record)
                    saved_wrong_count_per_pair[pair_key] += 1

    print(f"Selected {len(selected_records)} samples for Grad-CAM export.")

    # --------------------------------------------------
    # Second pass: Grad-CAM export
    # --------------------------------------------------
    selected_entries = []

    for r in selected_records:
        sample_index = r["sample_index"]
        true_class = r["true_class"]
        pred_class = r["pred_class"]
        pred_prob = r["pred_prob"]
        true_prob = r["true_prob"]

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
            save_root=save_root,
        )

        case_type = get_case_type(true_class, pred_class)

        selected_entries.append(
            {
                "output_path": output_path,
                "case_type": case_type,
                "sample_index": sample_index,
                "true_class": true_class,
                "pred_class": pred_class,
                "pred_prob": pred_prob,
                "true_prob": true_prob,
            }
        )

        print(f"Saved: {output_path}")

    gradcam.remove_hooks()

    # --------------------------------------------------
    # Save reports
    # --------------------------------------------------
    prediction_csv = save_prediction_summary_csv(all_records, report_dir)
    selection_csv = save_selection_summary_csv(selected_entries, report_dir)

    cm_png = plot_confusion_matrix(all_records, n_classes, report_dir)
    counts_png = plot_correct_wrong_counts(all_records, report_dir)
    top_pairs_png = plot_top_confusion_pairs(all_records, report_dir, top_k=10)

    summary_txt = save_text_report(
        checkpoint_path=checkpoint_path,
        n_classes=n_classes,
        all_records=all_records,
        selected_entries=selected_entries,
        correct_per_pair=args.correct_samples_per_pair,
        wrong_per_pair=args.wrong_samples_per_pair,
        report_dir=report_dir,
    )

    print("\nDone.")
    print(f"Grad-CAM results saved in: {save_root}")
    print(f"Reports saved in: {report_dir}")
    print(f"Prediction summary CSV: {prediction_csv}")
    print(f"Selection summary CSV: {selection_csv}")
    print(f"Confusion matrix: {cm_png}")
    print(f"Correct vs wrong plot: {counts_png}")
    print(f"Top confusion pairs plot: {top_pairs_png}")
    print(f"Text summary: {summary_txt}")

    required_outputs = [
        prediction_csv,
        selection_csv,
        cm_png,
        counts_png,
        top_pairs_png,
        summary_txt,
    ]

    missing_outputs = [p for p in required_outputs if not os.path.exists(p)]

    if missing_outputs:
        print("\nWARNING: Some expected output files were not created:")
        for p in missing_outputs:
            print(f"  Missing: {p}")
    else:
        print("\nAll expected output files were created successfully.")


if __name__ == "__main__":
    main()