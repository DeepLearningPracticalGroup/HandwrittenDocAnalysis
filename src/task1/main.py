"""
Task 01: DSS dataset
(a) Preprocessing and character segmentation
(b) Character recognition
"""

from time import perf_counter
from preprocessing import *


def main():

    start_time = perf_counter()

    image_path = "image-data"

    dataset = get_binarized_images(image_path)

    hough_transform(dataset)


    print(f"Running time for task 01: {round(perf_counter() - start_time),2} seconds")

if __name__ == "__main__":
    main()