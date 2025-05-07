import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from glob import glob

from tqdm import tqdm


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
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".pgm"))
            ]
            if images:
                all_chars[class_folder] = images  # Use clean class name
    return all_chars


def get_scroll_images(image_path: str, endswith: str):
    """
    To get the test scroll image files that end with endswith.
    """
    dataset = []
    for image in os.listdir(image_path):
        if image.endswith(endswith):
            full_path = os.path.join(image_path, image)
            dataset.append(full_path)
    print(f"Found {len(dataset)} {endswith} images in the dataset.")
    return dataset


def get_binarized_scroll_images(image_path):
    """
    To get the binarized test scroll images.
    """
    return get_scroll_images(image_path=image_path, endswith="binarized.jpg")


def get_fused_images(image_path):
    """
    To get the fused test scroll images.
    (Never used in the repo)
    """
    return get_scroll_images(image_path=image_path, endswith="fused.jpg")


def get_rgb_scroll_images(image_path):
    """
    To get the rgb test scroll images.
    (Never used in the repo)
    """
    dataset = []
    for image in os.listdir(image_path):
        if not image.endswith("binarized.jpg") and not image.endswith("fused.jpg"):
            full_path = os.path.join(image_path, image)
            dataset.append(full_path)
    print(f"Found {len(dataset)} rgb images in the dataset.")
    return dataset


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
            axes[i, 0].set_title(f"Original {i + 1}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(cleaned_imgs[i])
            axes[i, 1].set_title(f"Cleaned {i + 1}")
            axes[i, 1].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    print(f"Cleaned {len(cleaned_paths)} images successfully.")
    return cleaned_paths


def seperate_character_dataset(
    char_dataset: dict[str, list[str]],
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


def binarize_and_crop_image(input_path, output_path, threshold=200):
    img = Image.open(input_path)

    if img.mode == "RGBA":
        img_np = np.array(img)
        r, g, b, a = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2], img_np[:, :, 3]
        gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
        binary_np = np.where((a > 0) & (gray < threshold), 0, 255).astype(np.uint8)
    else:
        gray = img.convert("L")
        gray_np = np.array(gray)
        binary_np = np.where(gray_np < threshold, 0, 255).astype(np.uint8)

    coords = np.argwhere(binary_np == 0)
    if coords.size == 0:
        print(f"Warning: no content found in {input_path}")
        return None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    cropped = Image.fromarray(binary_np[y0:y1, x0:x1], mode="L")
    cropped.save(output_path)
    return cropped


def clean_character_image(img_path, min_area=50, center_tolerance=0.25):
    """
    Clean a character image by removing small contours and keeping only the largest ones.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Create mask
    mask = (img < 250).astype(np.uint8) * 255

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    # Find the largest component
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return img
    main_label = 1 + np.argmax(areas)

    # Build a mask that only includes the largest component
    final_mask = np.zeros_like(mask)
    final_mask[labels == main_label] = 255

    # Apply mask to the original image
    cleaned = img.copy()
    cleaned[final_mask == 0] = 255

    return cleaned


def clean_character_dataset(input_dir, output_dir):
    """
    Clean character images in the input directory and save them to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    total = 0
    for label in os.listdir(input_dir):
        label_path = os.path.join(input_dir, label)
        if not os.path.isdir(label_path):
            continue

        out_label_path = os.path.join(output_dir, label)
        os.makedirs(out_label_path, exist_ok=True)

        images = [
            f
            for f in os.listdir(label_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".pgm"))
        ]
        for img_name in tqdm(images, desc=f"Cleaning {label}"):
            in_path = os.path.join(label_path, img_name)
            out_path = os.path.join(out_label_path, img_name)

            cleaned = clean_character_image(in_path)
            if cleaned is not None:
                cv2.imwrite(out_path, cleaned)
                total += 1

    print(f"Finished. Cleaned and saved {total} images to '{output_dir}'.")
