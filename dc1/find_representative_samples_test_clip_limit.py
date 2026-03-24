import numpy as np

# 1. Load the first 1000 images
X = np.load("data/X_train.npy")
means = np.array([np.mean(img) for img in X])

# 2. Define our 5 target brightness levels (0%, 25%, 50%, 75%, 100%)
# These represent the full spread of your data
percentiles = [1, 25, 50, 75, 99]
targets = [np.percentile(means, p) for p in percentiles]

# 3. Find the index of the image closest to each target brightness
representative_indices = []
for t in targets:
    # Find the index where the difference between image mean and target is smallest
    idx = (np.abs(means - t)).argmin()
    representative_indices.append(idx)

# 4. Print your permanent testing "Panel of Judges"
labels = ["Darkest", "Lower-Quartile", "Median (Average)", "Upper-Quartile", "Brightest"]
for label, idx in zip(labels, representative_indices):
    print(f"{label} Index: {idx} (Brightness Mean: {means[idx]:.2f})")

print(f" Indices for clip limit testing: {representative_indices}")