import os
from glob import glob
from src.task1.utils.preprocessing import binarize_image
import numpy as np
from PIL import Image


def main():
    input_folder = "noise_maps"
    output_subfolder = os.path.join(input_folder, "noise_maps_binarized")
    os.makedirs(output_subfolder, exist_ok=True)

    extensions = ["*.png", "*.jpg", "*.jpeg"]
    image_paths = []

    for ext in extensions:
        pattern = os.path.join(input_folder, ext)
        image_paths.extend(glob(pattern))

    for img_path in image_paths:
        img = Image.open(img_path).convert("L")

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_subfolder, f"{base_name}_binarized.jpg")
        binarize_image(img_path, output_path)

    print(f"Binarized {len(image_paths)} images to {output_subfolder}")

if __name__ == "__main__":
    main()