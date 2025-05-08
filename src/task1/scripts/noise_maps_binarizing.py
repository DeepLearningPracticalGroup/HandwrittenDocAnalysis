"""
This script binarizes all images in the noise_maps folder and saves them in a subfolder named binarized.

to execute:
.venv/bin/ipython src/task1/scripts/noise_maps_binarizing.py
"""


import os
from glob import glob
from src.task1.utils.preprocessing import binarize_and_crop_image


def main():
    input_folder = "noise_maps"
    output_subfolder = os.path.join(input_folder, "binarized")
    os.makedirs(output_subfolder, exist_ok=True)

    extensions = ["*.png", "*.jpg", "*.jpeg"]
    image_paths = []

    for ext in extensions:
        pattern = os.path.join(input_folder, ext)
        image_paths.extend(glob(pattern))

    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_subfolder, f"{base_name}_binarized.jpg")
        binarize_and_crop_image(img_path, output_path)

    print(f"Binarized {len(image_paths)} images to {output_subfolder}")

if __name__ == "__main__":
    main()
