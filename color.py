import cv2
import numpy as np
import os
from datetime import datetime

def generate_unique_filename(base_name, extension):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"

def resize_for_display(image, max_width=800, max_height=600):
    h, w = image.shape[:2]
    if h > max_height or w > max_width:
        scaling_factor = min(max_height / h, max_width / w)
        new_size = (int(w * scaling_factor), int(h * scaling_factor))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image

print("Starting the image binarization process...")

# Specify your input image path here
input_image_path = os.path.join("images", "input", "pumpkins.jpg")

# Generate output image path
output_image_path = os.path.join(
    "images", "output", generate_unique_filename("binarized", "png")
)

try:
    # Print the input image path for debugging
    print(f"Input image path: {input_image_path}")

    # Load the image in color
    original_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)

    # Check if the image was loaded successfully
    if original_image is None:
        raise FileNotFoundError(f"File not found or cannot be read: {input_image_path}")

    print("Image loaded successfully.")

    # Convert to grayscale for processing
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply simple thresholding
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    print("Thresholding applied.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    # Save the binary image
    cv2.imwrite(output_image_path, binary_image)
    print(f"Binary image saved at: {output_image_path}")

    # Resize images for display
    display_original = resize_for_display(original_image)
    display_binary = resize_for_display(binary_image)

    # Display the resized original (in color) and binary images
    cv2.imshow("Original Image", display_original)
    cv2.imshow("Binary Image", display_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except FileNotFoundError as fnf_error:
    print(f"File Error: {fnf_error}")

except Exception as e:
    print(f"Error: {e}")