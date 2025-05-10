"""
Task 01: DSS dataset
(a) Preprocessing and character segmentation
(b) Character recognition - Scroll Generation (Random & Text-based)

to execute this script:

.venv/bin/ipython src/task1/scripts/create_random_scrolls.py -- --train_char_path "monkbrill_clean" --augmented_char_path "augmented_chars" \
--augment_per_char 1 --num_train_scrolls 1 --num_val_scrolls 1
"""

from time import perf_counter
from src.task1.utils.preprocessing import (
    get_character_images,
    seperate_character_dataset,
)
from src.task1.utils.generate import (
    generate_synthetic_scroll_with_ngrams,
    generate_file_scroll,
)
from src.task1.utils.data_augmentation import imagemorph_augmentation
from sklearn.model_selection import train_test_split
import random
import argparse


def main(
    train_char_path: str,
    augmented_char_path: str,
    char_val_size: float,
    augment_per_char: int,
    num_train_scrolls: int,
    num_val_scrolls: int,
):
    start_time = perf_counter()
    random.seed(20)

    # --- Load and prepare character dataset ---
    char_trainset_dict = get_character_images(root_path=train_char_path)
    X_char_all, y_char_all = seperate_character_dataset(char_trainset_dict)

    X_char_train, X_char_val, y_char_train, y_char_val = train_test_split(
        X_char_all,
        y_char_all,
        test_size=char_val_size,
        random_state=42,
        stratify=y_char_all,
    )

    # --- Augment training data only ---
    augmented_paths, augmented_labels = imagemorph_augmentation(
        image_paths=X_char_train,
        labels=y_char_train,
        output_dir=augmented_char_path,
        augment_per_image=augment_per_char,
    )

    X_char_train_extended = X_char_train + augmented_paths
    y_char_train_extended = y_char_train + augmented_labels

    print(
        f"Train set: {len(X_char_train)} original + {len(augmented_paths)} augmented = {len(X_char_train_extended)}"
    )
    print(f"Val set: {len(X_char_val)}")

    # === 1. Generate Random Scrolls using N-grams ===
    print("\nGenerating RANDOM synthetic scrolls...")
    generate_synthetic_scroll_with_ngrams(
        output_dir="dataset/synthetic_scrolls_random/train/",
        char_paths=X_char_train_extended,
        char_labels=y_char_train_extended,
        canvas_size=(1024, 2048),
        num_images=num_train_scrolls,
        min_lines=5,
        max_lines=15,
        noise_prob=0.75,
        ngram_csv_path="ngrams_frequencies_withNames.csv",
    )

    generate_synthetic_scroll_with_ngrams(
        output_dir="dataset/synthetic_scrolls_random/val/",
        char_paths=X_char_val,
        char_labels=y_char_val,
        canvas_size=(1024, 2048),
        num_images=num_val_scrolls,
        min_lines=5,
        max_lines=15,
        noise_prob=0.75,
        ngram_csv_path="ngrams_frequencies_withNames.csv",
    )

    # === 2. Generate Text-Based Scrolls (e.g. Bible, Aesop) ===
    print("\nGenerating TEXT-based synthetic scrolls...")
    for split, text_file, X_chars, y_chars, noise_prob in [
        (
            "train",
            "text_files/bible_train.txt",
            X_char_train_extended,
            y_char_train_extended,
            0.75,
        ),
        ("val", "text_files/bible_val.txt", X_char_val, y_char_val, 0.50),
        (
            "train",
            "text_files/hebrew_text/aesops_fables.txt",
            X_char_train_extended,
            y_char_train_extended,
            0.75,
        ),
    ]:
        generate_file_scroll(
            file_path=text_file,
            yaml_file_path="src/hebrew.yaml",
            output_dir=f"dataset/synthetic_scrolls_text/{split}",
            char_paths=X_chars,
            char_labels=y_chars,
            canvas_size=(1024, 2048),
            max_lines=20,
            noise_prob=noise_prob,
            words_per_line_range=(5, 10),
        )

    print(
        f"\nScroll generation completed in {round(perf_counter() - start_time, 2)} seconds"
    )


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
    parser.add_argument(
        "--num_train_scrolls",
        type=int,
        default=8000,
        help="Number of synthetic scrolls to generate for training.",
    )
    parser.add_argument(
        "--num_val_scrolls",
        type=int,
        default=2000,
        help="Number of synthetic scrolls to generate for validation.",
    )
    args = parser.parse_args()

    main(
        train_char_path=args.train_char_path,
        augmented_char_path=args.augmented_char_path,
        char_val_size=args.char_val_size,
        augment_per_char=args.augment_per_char,
        num_train_scrolls=args.num_train_scrolls,
        num_val_scrolls=args.num_val_scrolls,
    )
