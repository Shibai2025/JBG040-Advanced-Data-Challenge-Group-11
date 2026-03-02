import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# 1. Load Data
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent  # Adjust if needed

X_train = np.load(BASE_DIR / "data" / "X_train.npy")
Y_train = np.load(BASE_DIR / "data" / "Y_train.npy")

X_test = np.load(BASE_DIR / "data" / "X_test.npy")
Y_test = np.load(BASE_DIR / "data" / "Y_test.npy")

print("Train shape:", X_train.shape, Y_train.shape)
print("Test shape :", X_test.shape, Y_test.shape)

# ======================================================
# 2. Compute Class Counts (Aligned for Train/Test)
# ======================================================

train_labels, train_counts = np.unique(Y_train, return_counts=True)
test_labels, test_counts = np.unique(Y_test, return_counts=True)

# Union ensures the x-axis includes all labels seen in either split
all_labels = np.union1d(train_labels, test_labels)

train_dict = {int(label): int(count) for label, count in zip(train_labels, train_counts)}
test_dict  = {int(label): int(count) for label, count in zip(test_labels, test_counts)}

train_counts_aligned = np.array([train_dict.get(int(label), 0) for label in all_labels])
test_counts_aligned  = np.array([test_dict.get(int(label), 0)  for label in all_labels])

# ======================================================
# 3. Plot Stacked Bar Chart (Train bottom, Test top)
# ======================================================

x = np.arange(len(all_labels))

plt.figure(figsize=(10, 5))

# Bottom: Train (deep blue)
plt.bar(
    x,
    train_counts_aligned,
    label="Train",
    color="#1f77b4"
)

# Top: Test (orange) stacked above Train
plt.bar(
    x,
    test_counts_aligned,
    bottom=train_counts_aligned,
    label="Test",
    color="#ff7f0e"
)

# X-axis: numeric class IDs only (no name mapping)
plt.xticks(x, [f"Class {int(label)}" for label in all_labels], rotation=0)

plt.xlabel("Class ID")
plt.ylabel("Number of Samples")
plt.title("Class Distribution in Training and Test Sets")
plt.legend()

plt.grid(axis='y', linestyle='--', alpha=0.4)

# ======================================================
# 4. Add Value Labels (White text inside each section)
# ======================================================

for i in range(len(x)):
    train_val = int(train_counts_aligned[i])
    test_val = int(test_counts_aligned[i])

    # Label train section (only if non-zero)
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

    # Label test section (only if non-zero)
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