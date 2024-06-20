import cv2
import numpy as np
import os
import glob


# Function to apply inverse gamma correction
def inverse_gamma_correction(image, gamma):
    # Normalize the image to range [0, 1]
    normalized_image = image / 255.0

    # Apply the inverse gamma correction
    corrected_image = np.power(normalized_image, 1.0 / gamma)

    # Scale back to range [0, 255] and convert to 8-bit
    corrected_image = (corrected_image * 255).astype(np.uint8)

    return corrected_image


# Function to process each image patch
def process_image_patch(image_path, output_dir, gamma):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply inverse gamma correction
    corrected_image = inverse_gamma_correction(image, gamma)

    # Save the image as PNG
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, corrected_image)


# Main function to process all images in a category
def process_category_images(input_dir, output_dir, gamma=0.9):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files in the input directory
    image_files = glob.glob(os.path.join(input_dir, '*.png'))

    for image_file in image_files:
        process_image_patch(image_file, output_dir, gamma)


if __name__ == "__main__":
    # Example usage
    input_directory = '../dataset/test_rgb/cheetah'  # Replace with the path to your 90x90 patches
    output_directory = '../dataset/test/cheetah'  # Replace with the desired output path for PNG images

    process_category_images(input_directory, output_directory)


