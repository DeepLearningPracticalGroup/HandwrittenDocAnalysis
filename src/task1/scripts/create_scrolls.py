"""
Task 01: DSS dataset
(a) Preprocessing and character segmentation
(b) Character recognition

to execute this script:
first pip install ipython
then enter the following command in terminal:
ipython src/task1/scripts/create_scrolls.py
or
<env_name>/bin/ipython src/task1/scripts/create_scrolls.py
or
myenv/bin/ipython src/task1/scripts/create_scrolls.py
"""

from time import perf_counter
from src.task1.utils.preprocessing import get_character_images, seperate_character_dataset
from sklearn.model_selection import train_test_split
from src.task1.utils.generate import generate_synthetic_scroll
from src.task1.utils.data_augmentation import baseline_augmentation
from ultralytics import YOLO
import random


def main():

    start_time = perf_counter()

    # Ensure some degree of reproducibility
    random.seed(20)

    # Data image paths
    train_char_path = "monkbrill"

    ## Load training characters

    # Get a dictionary with character images
    char_trainset_dict = get_character_images(root_path=train_char_path)
    print(char_trainset_dict["Bet"][0])
    # Seperate dataset into lists of image_paths and labels
    X_char_train, y_char_train = seperate_character_dataset(char_trainset_dict)
    print(f"First example: X = {X_char_train[0]}, Y={y_char_train[0]}")

    # Split character trainset
    X_char_train, X_char_val, y_char_train, y_char_val = train_test_split(
        X_char_train,
        y_char_train,
        test_size=0.2,
        random_state=42,
        stratify=y_char_train,
    )
    print(f"len of trainset: {len(X_char_train)}")
    print(f"len of valset: {len(X_char_val)}")

    # Character Images to augment
    augment_per_image = 2

    # Scroll Images to generate
    num_train = 8000
    num_val = 2000

    # Data augmentation
    augmented_dir = "augmented_dataset"
    augmented_paths, augmented_labels = baseline_augmentation(
        X_char_train, y_char_train, augmented_dir,
        num_augments=augment_per_image
    )

    # Append new augmented paths and labels
    X_char_train_extended = X_char_train + augmented_paths
    y_char_train_extended = y_char_train + augmented_labels

    print(f"Original training set size: {len(X_char_train)}")
    print(f"Augmented training set size: {len(X_char_train_extended)}")


    # Generate training synthetic scrolls
    X_scroll_train, y_scroll_train = generate_synthetic_scroll(
    output_dir='synthetic_scrolls/train/',
    char_paths = X_char_train_extended,
    char_labels = y_char_train_extended,
    canvas_size=(256, 1024),
    num_images = num_train,
    min_chars = 5,
    max_chars = 8,
    min_lines = 3,
    max_lines = 6
    )
    # Call again to generate validation synthetic scrolls
    # Also change the params a little bit for better generalization
    X_scroll_val, y_scroll_val = generate_synthetic_scroll(
    output_dir='synthetic_scrolls/val/',
    char_paths = X_char_val,
    char_labels = y_char_val,
    canvas_size=(256, 1024),
    num_images = num_val,
    min_chars = 5,
    max_chars = 10,
    min_lines = 2,
    max_lines = 18
    )


    ## To Do's: (only if we want different segmenter and detector)

    ## SEGMENTER TRAINING:

    # Augmentation and merge to X_char_train and y_char_train...

    # Generate training scrolls and validation scrolls from new X_char_train, y_char_train

    # Train a segmenter with the training scrolls

    # Tune the segmenter based on the validation scrolls

    ## PREDICTOR TRAINING:

    # Train a predictor model with X_char_train and y_char_train

    # Tune the predictor using X_char_val and y_char_val

    # Output: Make sure to map each 'English' label with its 'Hebrew' equivalent

    ## FINAL PIPELINE

    # Pass the test scrolls as inputs to the Segmenter and pass the Segmenter's outputs as inputs to the Predictor.


    ## Load test scrolls

    #test_scrolls = get_binarized_scroll_images(image_path=test_scroll_path)

    print(f"Running time for task 01: {round(perf_counter() - start_time,2)} seconds")


if __name__ == "__main__":
    main()
