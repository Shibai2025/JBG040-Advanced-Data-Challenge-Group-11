import numpy as np
import matplotlib.pyplot as plt

data_path = "data/X_train_cropped.npy"
X_cleaned = np.load(data_path)

test_indices = [6360, 10648, 11344, 9860, 9344]
titles = ["Darkest (Switch to 2.5)", "Lower Q (1.5)", "Median (1.5)", "Upper Q (1.5)", "Brightest (Fail-Safe)"]
plt.figure(figsize=(20, 10))

for i, idx in enumerate(test_indices):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_cleaned[idx].squeeze(), cmap='gray')
    plt.title(f"{titles[i]}\nIndex: {idx}")
    plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Verification Complete. Array Shape: {X_cleaned.shape}")
print(f"Data Range: {X_cleaned.min()} to {X_cleaned.max()}")