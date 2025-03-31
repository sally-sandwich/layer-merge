from PIL import Image
import cv2
import numpy as np

def compare_images(img1_path, img2_path, output_path):
    # Load first image and convert to RGBA
    with Image.open(img1_path) as left:
        left = left.convert("RGBA")
        left_array = np.array(left, dtype=np.float32)

    # Load second image and convert to RGBA
    with Image.open(img2_path) as right:
        right = right.convert("RGBA")
        # Resize second image to match first if needed
        if right.size != left.size:
            right = right.resize(left.size)
        right_array = np.array(right, dtype=np.float32)

    # Calculate absolute difference
    diff = np.abs(right_array - left_array)
    
    # Normalize difference to 0-255 range
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    
    # Create difference image
    diff_image = Image.fromarray(diff)
    
    # Enhance difference visibility
    diff_array = np.array(diff_image)
    diff_array[diff_array.sum(axis=2) > 0] = [255, 0, 0, 255]  # Highlight differences in red
    
    # Save result
    diff_result = Image.fromarray(diff_array)
    diff_result.save(output_path, optimize=True, quality=95)
    
    return diff_result

# Usage
compare_images("input_1.png", "input_2.png", "output_diff.png")

