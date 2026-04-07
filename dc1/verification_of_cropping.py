import numpy as np
import matplotlib.pyplot as plt
import random


def verify_dataset():
    input_path = "data/X_train.npy"
    output_path = "data/X_train_cropped.npy"

    print("Loading original and cropped datasets for verification")
    try:
        x_original = np.load(input_path, mmap_mode='r')
        x_cropped = np.load(output_path, mmap_mode='r')
    except FileNotFoundError:
        print("Could not find the dataset files, make sure the cropped dataset exists.")
        return

    indices_to_check = [9344]
    total_images = len(x_original)
    indices_to_check.extend(random.sample(range(total_images), 4))

    print(f"Generating Visual Report for indices: {indices_to_check}")


    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    fig.suptitle("Verification Report Original vs Cropped Image", fontsize=16, fontweight='bold')

    for col_idx, img_idx in enumerate(indices_to_check):

        orig_img = x_original[img_idx].squeeze()
        crop_img = x_cropped[img_idx].squeeze()

        ax_orig = axes[0, col_idx]
        ax_orig.imshow(orig_img, cmap='gray')
        ax_orig.set_title(f"Original (Index {img_idx})")
        ax_orig.axis('off')

        ax_crop = axes[1, col_idx]
        ax_crop.imshow(crop_img, cmap='gray')
        if img_idx == 9344:
            ax_crop.set_title("Brightest Image Cropped", color='green')
        else:
            ax_crop.set_title("Cropped Output")
        ax_crop.axis('off')

        ax_diff = axes[2, col_idx]
        diff = np.abs(orig_img.astype(float) - crop_img.astype(float))
        ax_diff.imshow(diff, cmap='inferno')
        ax_diff.set_title("Heatmap of Changes in Pixels")
        ax_diff.axis('off')

    plt.tight_layout()
    print("Close the image window to finish the execution.")
    plt.show()


if __name__ == "__main__":
    verify_dataset()