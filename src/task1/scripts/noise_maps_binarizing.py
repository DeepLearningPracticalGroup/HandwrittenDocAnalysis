"""
This script binarizes all images in the noise_maps folder and saves them in a subfolder named binarized.
"""

import os
from glob import glob
from src.task1.utils.preprocessing import binarize_and_crop_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def main():
    def show_image(path, title=None):
        img = Image.open(path).convert("L")
        img_np = np.array(img)

        fig, ax = plt.subplots()
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.imshow(img_np, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        if title:
            ax.set_title(title, color="white")
        plt.show()

    input_folder = "noise_maps"
    output_subfolder = os.path.join(input_folder, "binarized")
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
        binarize_and_crop_image(img_path, output_path)

    show_image(output_path, title=base_name)

    print(f"Binarized {len(image_paths)} images to {output_subfolder}")


if __name__ == "__main__":
    main()
