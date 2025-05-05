"""
.venv/bin/ipython src/task1/scripts/create_font_scrolls.py -- --train_char_path "font_chars" --augmented_char_path "augmented_fonts" \
--augment_per_char 20 --num_train_scrolls 100 --num_val_scrolls 100
"""

from time import perf_counter
from src.task1.utils.preprocessing import get_character_images, seperate_character_dataset
from src.task1.utils.generate import generate_synthetic_scroll
from src.task1.utils.data_augmentation import imagemorph_augmentation
import random
import argparse

def main(
    train_char_path: str,
    augmented_char_path: str,
    augment_per_char: int,
    num_train_scrolls: int,
    num_val_scrolls: int,
):
    start_time = perf_counter()
    random.seed(20)

    # Load all characters (no split needed)
    char_dict = get_character_images(root_path=train_char_path)
    X_chars, y_chars = seperate_character_dataset(char_dict)
    print(f"Loaded {len(X_chars)} character images.")

    # Augment all characters
    augmented_paths, augmented_labels = imagemorph_augmentation(
        X_chars, y_chars, augmented_char_path, augment_per_image=augment_per_char
    )

    # Merge originals and augmentations
    all_X = X_chars + augmented_paths
    all_y = y_chars + augmented_labels
    print(f"Total character images after augmentation: {len(all_X)}")

    # Generate training scrolls
    generate_synthetic_scroll(
        output_dir="font_scrolls/train/",
        char_paths=all_X,
        char_labels=all_y,
        canvas_size=(1024, 2048),
        num_images=num_train_scrolls,
        min_chars=10,
        max_chars=30,
        min_lines=5,
        max_lines=15,
    )

    # Generate validation scrolls (still from full set, different seed internally)
    generate_synthetic_scroll(
        output_dir="font_scrolls/val/",
        char_paths=all_X,
        char_labels=all_y,
        canvas_size=(1024, 2048),
        num_images=num_val_scrolls,
        min_chars=10,
        max_chars=30,
        min_lines=5,
        max_lines=15,
    )

    print(f"Total scroll generation time: {round(perf_counter() - start_time, 2)} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain scroll generator (no character split)")

    parser.add_argument("--train_char_path", type=str, required=True, help="Path to character dataset")
    parser.add_argument("--augmented_char_path", type=str, required=True, help="Path to save augmented characters")
    parser.add_argument("--augment_per_char", type=int, default=5, help="Number of augmentations per character")
    parser.add_argument("--num_train_scrolls", type=int, default=1000, help="Number of training scrolls")
    parser.add_argument("--num_val_scrolls", type=int, default=200, help="Number of validation scrolls")

    args = parser.parse_args()

    main(
        train_char_path=args.train_char_path,
        augmented_char_path=args.augmented_char_path,
        augment_per_char=args.augment_per_char,
        num_train_scrolls=args.num_train_scrolls,
        num_val_scrolls=args.num_val_scrolls,
    )
