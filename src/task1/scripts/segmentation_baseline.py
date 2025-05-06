"""
Baseline for:
(a) Preprocessing and character segmentation

enter the following command in terminal:
ipython src/task1/scripts/main.py
or
.venv/bin/ipython src/task1/scripts/segmentation_baseline.py
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import perf_counter
from src.task1.utils.preprocessing import *
from src.task1.utils.segmenters import (
    cca_segmentation,
    projection_segmentation,
    projection_segmentation_auto,
)


def main():

    start_time = perf_counter()

    image_path = "image-data"

    dataset = get_binarized_scroll_images(image_path)

    # Image cleaning and denoising
    # boxes = cca_segmentation(dataset, plot=True)
    # boxes = projection_segmentation(dataset, plot=True)
    # boxes = projection_segmentation_auto(dataset, plot=True)

    # Example usage
    segment_lines_projection(
        "image-data/P21-Fg006-R-C01-R01-binarized.jpg", min_line_height=10
    )

    print(f"Running time for task 01: {round(perf_counter() - start_time),2} seconds")


if __name__ == "__main__":
    main()
