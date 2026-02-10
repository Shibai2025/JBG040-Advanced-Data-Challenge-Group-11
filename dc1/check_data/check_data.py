import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

X_train = np.load(BASE_DIR / "data" / "X_train.npy")
Y_train = np.load(BASE_DIR / "data" / "Y_train.npy")

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_train dtype:", X_train.dtype)
print("Y_train dtype:", Y_train.dtype)

# ---------- Basic pixel checks ----------
print("\n--- Pixel range check ---")
print("X_train min:", X_train.min(), "max:", X_train.max())
is_uint8_0_255 = (X_train.dtype == np.uint8) and (X_train.min() >= 0) and (X_train.max() <= 255)
print("Looks like uint8 grayscale 0-255:", is_uint8_0_255)

# ---------- Label distribution ----------
print("\n--- Label distribution (train) ---")
unique_labels, counts = np.unique(Y_train, return_counts=True)
n_samples = Y_train.shape[0]

order = np.argsort(unique_labels)
unique_labels = unique_labels[order]
counts = counts[order]

print("Unique labels:", unique_labels.tolist())
print("n_classes:", len(unique_labels))
print("Counts per class:", counts.tolist())
proportions = counts / n_samples
print("Proportions per class:", [float(f"{p:.4f}") for p in proportions])

max_count = counts.max()
min_count = counts.min()
print("\n--- Imbalance indicators ---")
print(f"Most frequent: label={int(unique_labels[np.argmax(counts)])}, count={int(max_count)}, share={max_count/n_samples:.2%}")
print(f"Least frequent: label={int(unique_labels[np.argmin(counts)])}, count={int(min_count)}, share={min_count/n_samples:.2%}")
print(f"Max/Min ratio: {max_count/min_count:.2f}x")

# ---------- Plots ----------
import matplotlib.pyplot as plt

# 1) Bar chart: class counts
plt.figure(figsize=(8, 4))
x = np.arange(len(unique_labels))
plt.bar(x, counts)
plt.xticks(x, [str(int(l)) for l in unique_labels])
plt.xlabel("Class label")
plt.ylabel("Count (train)")
plt.title("Training set class distribution")

for i, c in enumerate(counts):
    plt.text(i, c, str(int(c)), ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.show()

# 2) Sample grid per class: draw K images for each class
def show_samples_per_class(X, y, labels, k_per_class=6, seed=42):
    """
    For each class, randomly sample up to k_per_class images and show as a grid.
    Assumes X shape: (N, 1, H, W) or (N, H, W)
    """
    rng = np.random.default_rng(seed)

    n_classes = len(labels)
    n_cols = k_per_class
    n_rows = n_classes

    plt.figure(figsize=(2.2 * n_cols, 2.2 * n_rows))

    for r, lab in enumerate(labels):
        idxs = np.where(y == lab)[0]
        if len(idxs) == 0:
            # unlikely, but safe
            continue

        take = min(k_per_class, len(idxs))
        chosen = rng.choice(idxs, size=take, replace=False)

        for c in range(k_per_class):
            ax = plt.subplot(n_rows, n_cols, r * n_cols + c + 1)
            ax.axis("off")

            if c < take:
                img = X[chosen[c]]
                img2d = np.squeeze(img)  # (1,H,W)->(H,W)
                ax.imshow(img2d, cmap="gray")
                if c == 0:
                    ax.set_title(f"class {int(lab)}", fontsize=12)
            else:
                # empty slot if not enough images
                if c == 0:
                    ax.set_title(f"class {int(lab)}", fontsize=12)

    plt.suptitle(f"Random samples per class (k={k_per_class})", y=1.01, fontsize=14)
    plt.tight_layout()
    plt.show()

show_samples_per_class(X_train, Y_train, unique_labels, k_per_class=6, seed=42)
