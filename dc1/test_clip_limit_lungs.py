import numpy as np
import cv2
import os


def generate_tuning_slideshows(x_path, output_dir, clip_limits=[1.0, 1.5, 2.0, 3.0],
                               target_indices=[6360, 10648, 11344, 9860, 9344]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading dataset from {x_path}...")
    x_full = np.load(x_path)
    labels = ["1_Extreme_Dark", "2_Lower_Quartile", "3_Median", "4_Upper_Quartile", "5_Extreme_Bright"]

    for label, idx in zip(labels, target_indices):
        patient_folder = os.path.join(output_dir, f"{label}_Index_{idx}")
        if not os.path.exists(patient_folder):
            os.makedirs(patient_folder)
        img = x_full[idx].squeeze()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        cv2.imwrite(os.path.join(patient_folder, "0_Original.png"), img)

        for limit in clip_limits:
            clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(img)

            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            lung_mask = np.zeros_like(img)

            valid_contours = []
            height, width = img.shape

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)

                if cv2.contourArea(cnt) < (height * width * 0.02):
                    continue

                if x <= 2 or y <= 2 or (x + w) >= width - 2 or (y + h) >= height - 2:
                    continue

                valid_contours.append(cnt)
            cv2.drawContours(lung_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
            final_cutout = cv2.bitwise_and(img, img, mask=lung_mask)
            cv2.imwrite(os.path.join(patient_folder, f"Limit_{limit}.png"), final_cutout)

    print(f"\nDone! Check the folders in: {output_dir}")

generate_tuning_slideshows("data/X_train.npy", "dc1/clip_limits_on_lungs")