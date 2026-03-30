import numpy as np
import cv2
import os


def crop_chest_bounding_box(img, padding=5):
    # 1. Setup and Format
    original_height, original_width = img.shape

    if img.max() <= 1.0:
        working_img = (img * 255).astype(np.uint8)
    else:
        working_img = img.astype(np.uint8)

    # 2. THE SMART SWITCH
    mean_brightness = np.mean(working_img)
    if mean_brightness < 100:
        chosen_limit = 2.5  # Heavy-duty laser for dark images
    else:
        chosen_limit = 1.5  # Safe laser for normal/bright images

    # Create the Map
    clahe = cv2.createCLAHE(clipLimit=chosen_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(working_img)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Find and Filter Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Rule A: Ignore tiny specks (< 2% of area)
        if cv2.contourArea(cnt) < (original_height * original_width * 0.02):
            continue
        # Rule B: Ignore background room (touching the 2-pixel border)
        if x <= 2 or y <= 2 or (x + w) >= original_width - 2 or (y + h) >= original_height - 2:
            continue

        valid_contours.append(cnt)

    # 5. Fail-Safe
    if len(valid_contours) == 0:
        # If it's too bright to find lungs, just return the enhanced image safely
        return enhanced

        # 6. Calculate Bounding Box
    min_x = min([cv2.boundingRect(c)[0] for c in valid_contours])
    min_y = min([cv2.boundingRect(c)[1] for c in valid_contours])
    max_x = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid_contours])
    max_y = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in valid_contours])

    # 7. Apply Padding
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(original_width, max_x + padding)
    max_y = min(original_height, max_y + padding)

    # 8. Crop and Resize
    cropped_img = enhanced[min_y:max_y, min_x:max_x]
    final_resized_img = cv2.resize(cropped_img, (original_width, original_height))

    return final_resized_img


if __name__ == "__main__":
    input_path = "data/X_train.npy"
    output_path = "data/X_train_cropped.npy"

    print(f"Loading dataset from {input_path}")
    X_full = np.load(input_path)

    X_cropped_list = []
    total_images = len(X_full)

    print(f"Starting Bounding Box Cropping on {total_images} images")

    for i in range(total_images):
        img = X_full[i].squeeze()

        # Apply our custom crop function
        cropped = crop_chest_bounding_box(img)

        # Add the channel dimension back for the Neural Network (e.g., 128x128 becomes 128x128x1)
        cropped = np.expand_dims(cropped, axis=-1)

        X_cropped_list.append(cropped)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} / {total_images} images")

    print(f"Converting list to Numpy array...")
    X_cropped_array = np.array(X_cropped_list, dtype=np.uint8)

    print(f"Saving new dataset to {output_path}...")
    np.save(output_path, X_cropped_array)
    print("Execution Finished!")
