import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random

def get_character_images(root_path: str) -> dict[str, list[str]]:
    """
    Load all character images from subfolders into a dictionary.
    """
    all_chars = {}
    # Loop over all subfolders inside root_path
    for class_folder in os.listdir(root_path):
        full_path = os.path.join(root_path, class_folder)
        if os.path.isdir(full_path):
            images = [
                os.path.join(full_path, f) 
                for f in os.listdir(full_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pgm'))
            ]
            if images:
                all_chars[class_folder] = images  # Use clean class name
    return all_chars


def get_scroll_images(image_path: str, endswith: str):
    dataset = []
    for image in os.listdir(image_path):
        if image.endswith(endswith):
            full_path = os.path.join(image_path, image)
            dataset.append(full_path)
    print(f"Found {len(dataset)} {endswith} images in the dataset.")
    return dataset


def get_binarized_scroll_images(image_path):
    return get_scroll_images(image_path=image_path, endswith="binarized.jpg")


def get_fused_images(image_path):
    return get_scroll_images(image_path=image_path, endswith="fused.jpg")


def get_rgb_scroll_images(image_path):
    dataset = []
    for image in os.listdir(image_path):
        if not image.endswith("binarized.jpg") and not image.endswith("fused.jpg"):
            full_path = os.path.join(image_path, image)
            dataset.append(full_path)
    print(f"Found {len(dataset)} rgb images in the dataset.")
    return dataset


# Hough Transform
def hough_transform(dataset):
    img = cv2.imread(dataset[0], cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Image not loaded.")
        return

    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()


# Image cleaning pipeline (Not needed for testset as it is superbly cleaned)
def clean_images_adaptive(
    image_paths: list[str], save_dir: str = None, show_preview: bool = False
) -> list[str]:
    """
    Cleans images using adaptive filtering and morphological closing to reduce noise
    while preserving edges. Optionally saves or previews results.
    """
    cleaned_paths = []
    original_imgs = []
    cleaned_imgs = []

    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    for idx, path in enumerate(image_paths):
        try:
            img = cv2.imread(path)
            if img is None:
                raise ValueError("Unreadable image")

            # Apply bilateral filter (adaptive smoothing)
            cleaned = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
            # Apply morphological closing
            kernel = np.ones((3, 3))
            cleaned = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

            if idx < 4 and show_preview:
                # For preview: save images for later plotting
                original_imgs.append(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                )  # Convert BGR to RGB
                cleaned_imgs.append(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))

            if save_dir:
                filename = os.path.basename(path)
                cleaned_path = os.path.join(save_dir, filename)
                cv2.imwrite(cleaned_path, cleaned)
                cleaned_paths.append(cleaned_path)
            else:
                cleaned_paths.append(path)

        except Exception as e:
            print(f"Skipping corrupted or failed image: {path} ({e})")

    if show_preview and original_imgs:
        fig, axes = plt.subplots(4, 2, figsize=(10, 16))
        fig.suptitle("Original vs Cleaned Images", fontsize=16)

        for i in range(len(original_imgs)):
            axes[i, 0].imshow(original_imgs[i])
            axes[i, 0].set_title(f"Original {i+1}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(cleaned_imgs[i])
            axes[i, 1].set_title(f"Cleaned {i+1}")
            axes[i, 1].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    print(f"Cleaned {len(cleaned_paths)} images successfully.")
    return cleaned_paths




def seperate_character_dataset(
    char_dataset: dict[str, list[str]]
) -> tuple[list[str], list[str]]:
    """
    Seperate a character dataset dictionary into parallel lists of image paths and labels.
    """
    image_paths = []
    labels = []

    for class_name, paths in char_dataset.items():
        image_paths.extend(paths)
        labels.extend([class_name] * len(paths))

    return image_paths, labels
