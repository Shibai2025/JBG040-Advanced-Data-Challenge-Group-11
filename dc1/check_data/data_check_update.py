"""
data_check.py

Generate a class distribution plot with human-readable class names,
e.g. "class 3 No Finding", in a fixed order matching the reference figure.

Recommended run (from repo root):
    python -m dc1.check_data.data_check

Or run inside dc1/check_data:
    python data_check.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Label -> Name mapping
# (Matches your outputs where class 3 has the largest share and corresponds to "No Finding")
# -------------------------
CLASS_NAME_MAP: Dict[int, str] = {
    5: "Pneumothorax",
    4: "Nodule",
    3: "No Finding",
    2: "Infiltration",
    1: "Effusion",
    0: "Atelectasis",
}

# Plot order to match your reference image:
PLOT_CLASS_ORDER: List[int] = [5, 4, 3, 2, 1, 0]


def base_dir() -> Path:
    """Directory of this script (dc1/check_data)."""
    return Path(__file__).resolve().parent


def load_labels(path: Path) -> np.ndarray:
    """Load label vector from .npy and ensure it is 1D int array."""
    y = np.load(path)
    y = np.asarray(y).reshape(-1).astype(int)
    return y


def count_per_class(y: np.ndarray, class_ids: List[int]) -> np.ndarray:
    """Count occurrences of each class in class_ids."""
    counts = np.zeros(len(class_ids), dtype=int)
    for i, c in enumerate(class_ids):
        counts[i] = int(np.sum(y == c))
    return counts


def print_split_stats(split_name: str, counts: np.ndarray, class_ids: List[int]) -> None:
    total = int(counts.sum())
    print(f"\n--- Label distribution ({split_name}) ---")
    print(f"Total samples: {total}")
    for cid, cnt in zip(class_ids, counts):
        name = CLASS_NAME_MAP.get(cid, f"Class {cid}")
        share = (cnt / total) if total > 0 else 0.0
        print(f"class {cid:>1} ({name:<12}) : {cnt:>5}  ({share*100:>6.2f}%)")


def make_plot(
    train_counts: np.ndarray,
    test_counts: np.ndarray,
    holdout_counts: Optional[np.ndarray],
    class_ids: List[int],
    out_path: Path,
    show_values: bool = True,
) -> None:
    x = np.arange(len(class_ids))

    labels = [f"class {cid} {CLASS_NAME_MAP.get(cid, str(cid))}" for cid in class_ids]

    fig = plt.figure(figsize=(10, 6), dpi=120)
    ax = fig.add_subplot(111)

    # stacked bars
    ax.bar(x, train_counts, label="Training")
    ax.bar(x, test_counts, bottom=train_counts, label="Test")

    if holdout_counts is not None:
        ax.bar(x, holdout_counts, bottom=train_counts + test_counts, label="Hold out")

    ax.set_title("Distribution of image labels")
    ax.set_xlabel("Class label")
    ax.set_ylabel("No. of images")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()

    # optional: write values inside each colored segment
    if show_values:
        for i in range(len(class_ids)):
            # Training label
            if train_counts[i] > 0:
                ax.text(i, train_counts[i] * 0.5, str(int(train_counts[i])), ha="center", va="center", fontsize=9)

            # Test label
            if test_counts[i] > 0:
                ax.text(
                    i,
                    train_counts[i] + test_counts[i] * 0.5,
                    str(int(test_counts[i])),
                    ha="center",
                    va="center",
                    fontsize=9,
                )

            # Holdout label
            if holdout_counts is not None and holdout_counts[i] > 0:
                ax.text(
                    i,
                    train_counts[i] + test_counts[i] + holdout_counts[i] * 0.5,
                    str(int(holdout_counts[i])),
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory that contains Y_train.npy / Y_test.npy (default: dc1/data).",
    )
    parser.add_argument(
        "--y_holdout",
        type=str,
        default=None,
        help="Optional holdout label file (e.g. dc1/data/Y_holdout.npy). If not provided, holdout is skipped.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (default: dc1/check_data/Distribution of image labels.png).",
    )
    parser.add_argument(
        "--no_values",
        action="store_true",
        help="Disable printing the segment values on the bars.",
    )
    args = parser.parse_args()

    # Resolve data dir robustly
    dc1_dir = base_dir().parent  # dc1/
    data_dir = Path(args.data_dir) if args.data_dir else (dc1_dir / "data")
    data_dir = data_dir.resolve()

    y_train_path = data_dir / "Y_train.npy"
    y_test_path = data_dir / "Y_test.npy"

    if not y_train_path.exists():
        raise FileNotFoundError(f"Missing file: {y_train_path}")
    if not y_test_path.exists():
        raise FileNotFoundError(f"Missing file: {y_test_path}")

    y_train = load_labels(y_train_path)
    y_test = load_labels(y_test_path)

    y_holdout: Optional[np.ndarray] = None
    if args.y_holdout:
        holdout_path = Path(args.y_holdout)
        if not holdout_path.is_absolute():
            holdout_path = (dc1_dir / holdout_path).resolve()
        if not holdout_path.exists():
            raise FileNotFoundError(f"Holdout file not found: {holdout_path}")
        y_holdout = load_labels(holdout_path)

    class_ids = PLOT_CLASS_ORDER

    train_counts = count_per_class(y_train, class_ids)
    test_counts = count_per_class(y_test, class_ids)
    holdout_counts = count_per_class(y_holdout, class_ids) if y_holdout is not None else None

    # Print stats
    print_split_stats("train", train_counts, class_ids)
    print_split_stats("test", test_counts, class_ids)
    if holdout_counts is not None:
        print_split_stats("holdout", holdout_counts, class_ids)

    # Output path
    out_path = Path(args.out) if args.out else (base_dir() / "Distribution of image labels.png")
    if not out_path.is_absolute():
        out_path = (dc1_dir / out_path).resolve()

    make_plot(
        train_counts=train_counts,
        test_counts=test_counts,
        holdout_counts=holdout_counts,
        class_ids=class_ids,
        out_path=out_path,
        show_values=not args.no_values,
    )

    print(f"\nSaved figure to: {out_path}")


if __name__ == "__main__":
    main()