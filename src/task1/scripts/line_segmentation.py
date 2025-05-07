"""Line segmentation script for the scrolls dataset.

to execute this script, run the following command:
.venv/bin/ipython src/task1/scripts/line_segmentation.py
"""

from src.task1.utils.line_segmentation import segment_scrolls
import os
import numpy as np
from src.task1.utils.plots import show_line_segmentation_on_image
from PIL import Image

def main():

    # Segment generated training scrolls
    segment_scrolls(
    input_img_dir='generated_scrolls/train/images',
    input_label_dir='generated_scrolls/train/labels',
    output_img_dir='segmented_scrolls/generated_scrolls/train/images',
    output_label_dir='segmented_scrolls/generated_scrolls/train/labels',
    N=80,
    angle_range=(-10, 10),
    angle_step=0.5,
    padding=20
)
    # Segment generated validation scrolls
    segment_scrolls(
    input_img_dir='generated_scrolls/generated_scrolls/val/images',
    input_label_dir='generated_scrolls/generated_scrolls/val/labels',
    output_img_dir='segmented_scrolls/generated_scrolls/val/images',
    output_label_dir='segmented_scrolls/generated_scrolls/val/labels',
    N=80,
    angle_range=(-10, 10),
    angle_step=0.5,
    padding=20
)
    # Segment synthetic training scrolls
    segment_scrolls(
    input_img_dir='synthetic_scrolls/synthetic_scrolls/train/images',
    input_label_dir='synthetic_scrolls/synthetic_scrolls/train/labels',
    output_img_dir='segmented_scrolls/synthetic_scrolls/train/images',
    output_label_dir='segmented_scrolls/synthetic_scrolls/train/labels',
    )

    # Segment synthetic validation scrolls
    segment_scrolls(
    input_img_dir='synthetic_scrolls/synthetic_scrolls/val/images',
    input_label_dir='synthetic_scrolls/synthetic_scrolls/val/labels',
    output_img_dir='segmented_scrolls/synthetic_scrolls/val/images',
    output_label_dir='segmented_scrolls/synthetic_scrolls/val/labels',
    )

if __name__ == "__main__":
    main()