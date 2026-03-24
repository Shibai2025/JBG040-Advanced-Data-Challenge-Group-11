import numpy as np
import cv2
import os


def generate_tuning_slideshows(x_path, output_dir, clip_limits=[1.0, 1.5, 2.0, 3.0],
                               target_indices=[6360, 10648, 11344, 9860, 9344]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading dataset from {x_path}...")
    X_full = np.load(x_path)

    # Mapping our labels to the indices for better folder naming
    labels = ["1_Extreme_Dark", "2_Lower_Quartile", "3_Median", "4_Upper_Quartile", "5_Extreme_Bright"]

    for label, idx in zip(labels, target_indices):
        patient_folder = os.path.join(output_dir, f"{label}_Index_{idx}")
        if not os.path.exists(patient_folder):
            os.makedirs(patient_folder)

        # Get the image (assuming shape [1, 128, 128] or [128, 128])
        img = X_full[idx].squeeze()

        # Convert to 0-255 uint8 if it's float 0-1
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        cv2.imwrite(os.path.join(patient_folder, "0_Original.png"), img)

        for limit in clip_limits:
            # 1. CLAHE (The Contrast Booster)
            clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(img)

            # 2. Blur & Threshold (The Mask Creator)
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 3. Morphology (The Noise Cleaner)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # Save the result
            cv2.imwrite(os.path.join(patient_folder, f"Limit_{limit}.png"), mask)

    print(f"\nDone! Check the folders in: {output_dir}")


# --- RUN ---
# Use the relative path that worked:
generate_tuning_slideshows("data/X_train.npy", "dc1/clip_limits_initial_experiment")