"""
Task 01: DSS dataset
(a) Preprocessing and character segmentation
(b) Character recognition

to execute this script:
first pip install ipython
then enter the following command in terminal:
ipython src/task1/scripts/create_text_scrolls.py -- --train_char_path "monkbrill_clean" --augmented_char_path "augmented_chars" \
--augment_per_char 1 
or
<env_name>/bin/ipython src/task1/scripts/create_text_scrolls.py -- --train_char_path "monkbrill_clean" --augmented_char_path "augmented_chars" \
--augment_per_char 1
or
.venv/bin/ipython src/task1/scripts/create_text_scrolls.py -- --train_char_path "monkbrill_clean" --augmented_char_path "augmented_chars" \
--augment_per_char 1
"""

from time import perf_counter
from src.task1.utils.preprocessing import (
    get_character_images,
    seperate_character_dataset,
)
from sklearn.model_selection import train_test_split
from src.task1.utils.generate import generate_file_scroll
from src.task1.utils.data_augmentation import (
    imagemorph_augmentation,
)
from ultralytics import YOLO
import random
import argparse



def main(
    train_char_path: str,
    augmented_char_path: str,
    char_val_size: int,
    augment_per_char: int,
):

    start_time = perf_counter()

    # Ensure some degree of reproducibility
    random.seed(20)

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
        test_size=char_val_size,
        random_state=42,
        stratify=y_char_train,
    )
    print(f"len of trainset: {len(X_char_train)}")
    print(f"len of valset: {len(X_char_val)}")

    # Data augmentation
    augmented_paths, augmented_labels = imagemorph_augmentation(
        image_paths=X_char_train, labels=y_char_train, output_dir=augmented_char_path, augment_per_image=augment_per_char
    )

    # Append new augmented paths and labels
    X_char_train_extended = X_char_train + augmented_paths
    y_char_train_extended = y_char_train + augmented_labels

    print(f"Original training set size: {len(X_char_train)}")
    print(f"Augmented training set size: {len(X_char_train_extended)}")

    # Generate training synthetic scrolls from bible text
    generate_file_scroll(
        file_path="text_files/bible_train.txt",
        yaml_file_path="src/hebrew.yaml",
        output_dir="generated_scrolls/train",
        char_paths=X_char_train_extended,
        char_labels=y_char_train_extended,
        canvas_size=(256, 1024),
        max_lines=20,
        noise_prob=0.75,
    )

    # Generate validation synthetic scrolls from bible text
    generate_file_scroll(
        file_path="text_files/bible_val.txt",
        yaml_file_path="src/hebrew.yaml",
        output_dir="generated_scrolls/val",
        char_paths=X_char_val,
        char_labels=y_char_val,
        canvas_size=(256, 1024),
        max_lines=20,
    )

    # Generate training synthetic scrolls from translated aesop fables text
    generate_file_scroll(
        file_path="text_files/hebrew_text/aesops_fables.txt",
        yaml_file_path="src/hebrew.yaml",
        output_dir="generated_scrolls/train",
        char_paths=X_char_train_extended,
        char_labels=y_char_train_extended,
        canvas_size=(256, 1024),
        max_lines=20,
        noise_prob=0.75,
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

    # test_scrolls = get_binarized_scroll_images(image_path=test_scroll_path)

    print(f"Running time for task 01: {round(perf_counter() - start_time,2)} seconds")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create synthetic scrolls for DSS dataset."
    )
    parser.add_argument(
        "--train_char_path",
        type=str,
        required=True,
        help="Path to the training character dataset.",
    )
    parser.add_argument(
        "--augmented_char_path",
        type=str,
        required=True,
        help="Path to save augmented character dataset.",
    )
    parser.add_argument(
        "--char_val_size",
        type=float,
        default=0.2,
        help="Proportion of the character dataset to include in the validation split.",
    )
    parser.add_argument(
        "--augment_per_char",
        type=int,
        default=2,
        help="Number of augmentations per character image.",
    )

    args = parser.parse_args()

    main(
        train_char_path=args.train_char_path,
        augmented_char_path=args.augmented_char_path,
        char_val_size=args.char_val_size,
        augment_per_char=args.augment_per_char,
    )
