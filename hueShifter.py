import cv2
from pathlib import Path
import numpy as np


def shift_hue_to_red(image_path, output_path):
    # Read the image
    img = cv2.imread(str(image_path))

    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Shift the hue to red (hue value for red is around 0 or 180 in OpenCV's HSV)
    # Adjust hue and saturation to achieve a red monochrome effect
    hsv[..., 0] = 0  # Set hue to 0 for red
    hsv[..., 1] = np.clip(hsv[..., 1] * 4, 0, 255)  # Increase saturation by 50%

    # Convert the image back to BGR
    red_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Save the image with the "_r" suffix
    cv2.imwrite(str(output_path), red_img)

def shift_hue_to_blue(image_path, output_path):
    # Read the image
    img = cv2.imread(str(image_path))

    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Shift the hue to red (hue value for red is around 0 or 180 in OpenCV's HSV)
    # Adjust hue and saturation to achieve a red monochrome effect
    hsv[..., 0] = 100  # Set hue to 0 for red
    hsv[..., 1] = np.clip(hsv[..., 1] * 4, 0, 255)  # Increase saturation by 50%

    # Convert the image back to BGR
    blue_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Save the image with the "_r" suffix
    cv2.imwrite(str(output_path), blue_img)

def shift_hue_to_pink(image_path, output_path):
    # Read the image
    img = cv2.imread(str(image_path))

    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Shift the hue to red (hue value for red is around 0 or 180 in OpenCV's HSV)
    # Adjust hue and saturation to achieve a red monochrome effect
    hsv[..., 0] = 160  # Set hue to 0 for red
    hsv[..., 1] = np.clip(hsv[..., 1] * 4, 0, 255)  # Increase saturation by 50%

    # Convert the image back to BGR
    pink_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Save the image with the "_r" suffix
    cv2.imwrite(str(output_path), pink_img)

def process_directory(directory_path):
    # Convert the string path to a Path object
    path = Path(directory_path)

    if not path.exists():
        print(f"The directory does not exist: {directory_path}")
        return

    # Iterate through all image files in the directory
    for file in path.iterdir():
        if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            # Construct the output file path
            red_output_file = file.with_name(f"{file.stem}_r{file.suffix}")
            blue_output_file = file.with_name(f"{file.stem}_b{file.suffix}")
            pink_output_file = file.with_name(f"{file.stem}_p{file.suffix}")

            # Shift the hue to red and save the image
            shift_hue_to_red(file, red_output_file)
            shift_hue_to_blue(file, blue_output_file)
            shift_hue_to_pink(file, pink_output_file)


# Example usage
directory_path = r''
process_directory(directory_path)
