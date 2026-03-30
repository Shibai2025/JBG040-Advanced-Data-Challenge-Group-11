import numpy as np

X = np.load("data/X_train.npy")
means = np.array([np.mean(img) for img in X])

percentiles = [1, 25, 50, 75, 99]
targets = [np.percentile(means, p) for p in percentiles]

representative_indices = []
for t in targets:
    # Find the index where the difference between image mean and target is smallest
    idx = (np.abs(means - t)).argmin()
    representative_indices.append(idx)

labels = ["Darkest", "Lower-Quartile", "Median", "Upper-Quartile", "Brightest"]
for label, idx in zip(labels, representative_indices):
    print(f"{label} Index: {idx} (Brightness Mean: {means[idx]:.2f})")

print(f" Indices for clip limit testing: {representative_indices}")