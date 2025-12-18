import cv2
import numpy as np
import os
import random

# Function to perform random augmentations
def augment_image(image):
    rows, cols, channels = image.shape
    
    # Random rotation
    angle = random.choice([90, 180, 270])
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))
    
    # Random flip (horizontal and vertical)
    flip_type = random.choice([-1, 0, 1])  # -1: both axes, 0: vertical, 1: horizontal
    flipped = cv2.flip(rotated, flip_type)
    
    # Random scaling
    scale = random.uniform(0.8, 1.2)
    scaled = cv2.resize(flipped, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    # Random translation (shifting)
    tx, ty = random.randint(-20, 20), random.randint(-20, 20)
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(scaled, translation_matrix, (cols, rows))
    
    # Random cropping
    crop_size = random.uniform(0.7, 1.0)  # Crop between 70% to 100% of original size
    crop_x = int(cols * crop_size)
    crop_y = int(rows * crop_size)
    x_start = random.randint(0, cols - crop_x)
    y_start = random.randint(0, rows - crop_y)
    cropped = shifted[y_start:y_start + crop_y, x_start:x_start + crop_x]
    cropped_resized = cv2.resize(cropped, (cols, rows))  # Resize back to original dimensions
    
    # Random brightness and contrast adjustments
    brightness = random.randint(-30, 30)
    contrast = random.uniform(0.8, 1.2)
    brightness_contrast_adjusted = cv2.convertScaleAbs(cropped_resized, alpha=contrast, beta=brightness)
    
    # Random color shift (ensure type compatibility)
    b_shift = random.randint(-20, 20)
    g_shift = random.randint(-20, 20)
    r_shift = random.randint(-20, 20)
    color_shift = np.array([b_shift, g_shift, r_shift], dtype=np.int16).reshape(1, 1, 3)  # Create shift array
    
    # Ensure image is in int16 format for adding, then clip and convert back to uint8
    image_int16 = brightness_contrast_adjusted.astype(np.int16)
    color_shifted = image_int16 + color_shift  # Add the color shift
    color_shifted = np.clip(color_shifted, 0, 255)  # Clip values to stay within [0, 255]
    color_shifted = color_shifted.astype(np.uint8)  # Convert back to uint8
    
    return color_shifted


# Function to augment dataset
def augment_dataset(input_dir, output_dir, num_augmentations=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            # Save original image to output dir
            cv2.imwrite(os.path.join(output_dir, filename), image)

            # Perform augmentations
            for i in range(num_augmentations):
                augmented_image = augment_image(image)
                new_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"
                cv2.imwrite(os.path.join(output_dir, new_filename), augmented_image)

# Example usage
input_dir = 'Data/train/0'
output_dir = 'Data/train/0_new'
augment_dataset(input_dir, output_dir, num_augmentations=5)