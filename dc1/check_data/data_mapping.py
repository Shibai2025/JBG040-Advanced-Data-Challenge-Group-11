import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# 1. Load Data
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent  # adjust if needed

X_train = np.load(BASE_DIR / "data" / "X_train.npy")
Y_train = np.load(BASE_DIR / "data" / "Y_train.npy")

X_test = np.load(BASE_DIR / "data" / "X_test.npy")
Y_test = np.load(BASE_DIR / "data" / "Y_test.npy")

print("Train shape:", X_train.shape, Y_train.shape)
print("Test shape :", X_test.shape, Y_test.shape)

# ======================================================
# 2. Compute Class Counts (Aligned)
# ======================================================

train_labels, train_counts = np.unique(Y_train, return_counts=True)
test_labels, test_counts = np.unique(Y_test, return_counts=True)

# Union ensures the x-axis includes all labels seen in either split
all_labels = np.union1d(train_labels, test_labels)

train_dict = {int(lbl): int(cnt) for lbl, cnt in zip(train_labels, train_counts)}
test_dict = {int(lbl): int(cnt) for lbl, cnt in zip(test_labels, test_counts)}

train_counts_aligned = np.array([train_dict.get(int(lbl), 0) for lbl in all_labels])
test_counts_aligned = np.array([test_dict.get(int(lbl), 0) for lbl in all_labels])

# ======================================================
# 3. Label -> Name Mapping (CONFIRMED BY YOU)
# ======================================================

class_name_map = {
    5: "Pneumothorax",
    4: "Nodule",
    3: "No Finding",
    2: "Infiltration",
    1: "Effusion",
    0: "Atelectasis",
}

# Tick labels as two lines: Class ID + Name
tick_labels = [
    f"Class {int(lbl)}\n{class_name_map.get(int(lbl), 'Unknown')}"
    for lbl in all_labels
]

# ======================================================
# 4. Plot Stacked Bar Chart (Train bottom, Test top)
# ======================================================

x = np.arange(len(all_labels))

plt.figure(figsize=(10, 5))

# Train (deep blue)
plt.bar(
    x,
    train_counts_aligned,
    label="Train",
    color="#1f77b4"
)

# Test (orange) stacked above train
plt.bar(
    x,
    test_counts_aligned,
    bottom=train_counts_aligned,
    label="Test",
    color="#ff7f0e"
)

plt.xticks(x, tick_labels, rotation=0)
plt.xlabel("Class ID and Name")
plt.ylabel("Number of Samples")
plt.title("Class Distribution in Training and Test Sets")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)

# ======================================================
# 5. Value Labels (numbers inside each segment)
# ======================================================

for i in range(len(x)):
    train_val = int(train_counts_aligned[i])
    test_val = int(test_counts_aligned[i])

    # Label train section (centered)
    if train_val > 0:
        plt.text(
            x[i],
            train_val / 2,
            str(train_val),
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold"
        )

    # Label test section (centered)
    if test_val > 0:
        plt.text(
            x[i],
            train_val + test_val / 2,
            str(test_val),
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold"
        )

plt.tight_layout()
plt.show()