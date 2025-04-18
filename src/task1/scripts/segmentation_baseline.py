"""
Baseline for:
(a) Preprocessing and character segmentation

enter the following command in terminal:
ipython src/task1/scripts/main.py
or
myenv/bin/ipython src/task1/scripts/segmentation_baseline.py
"""

import cv2
from time import perf_counter
from src.task1.utils.preprocessing import *

def main():

    start_time = perf_counter()

    image_path = "image-data"

    dataset = get_binarized_images(image_path)
    
    # Image cleaning and denoising
    lines = segment_lines(dataset,show_preview=True)




    print(f"Running time for task 01: {round(perf_counter() - start_time),2} seconds")

if __name__ == "__main__":
    main()