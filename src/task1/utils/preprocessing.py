import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def get_images(image_path:str, endswith:str):
    dataset = []
    for image in os.listdir(image_path):
        if image.endswith(endswith):
            full_path = os.path.join(image_path, image)
            dataset.append(full_path)
    print(f"Found {len(dataset)} {endswith} images in the dataset.")
    return dataset

def get_binarized_images(image_path):
    return get_images(image_path=image_path, endswith='binarized.jpg')

def get_fused_images(image_path):
    return get_images(image_path=image_path, endswith='fused.jpg')

def get_rgb_images(image_path):
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

    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.show()

# Image cleaning pipeline (Not needed for testset as it is superbly cleaned)
def clean_images_adaptive(image_paths: list[str], save_dir: str = None, show_preview: bool = False) -> list[str]:
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
            kernel = np.ones((3,3))
            cleaned = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

            if idx < 4 and show_preview:
                # For preview: save images for later plotting
                original_imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
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
        fig.suptitle('Original vs Cleaned Images', fontsize=16)

        for i in range(len(original_imgs)):
            axes[i, 0].imshow(original_imgs[i])
            axes[i, 0].set_title(f'Original {i+1}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(cleaned_imgs[i])
            axes[i, 1].set_title(f'Cleaned {i+1}')
            axes[i, 1].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    print(f"Cleaned {len(cleaned_paths)} images successfully.")
    return cleaned_paths

# Visualize a single image
def plot_image(image_path: str, title: str = "Image", cmap: str = None) -> None:
    """Plots an image from the given file path using matplotlib."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert BGR (OpenCV default) to RGB if not grayscale
    if cmap != 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()





def segment_lines(image_paths: list[str], save_dir: str = None, show_preview: bool = False) -> list[np.ndarray]:
    """Segments an input image into horizontal text lines using contour detection after thresholding."""
    all_lines = []
    preview_lines = []

    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    for img_idx, image_path in enumerate(image_paths):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image: {image_path}")
            continue

        _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        # Sum of black pixels across rows
        projection = np.sum(img_bin == 255, axis=1)

        # Find where text lines exist
        line_indices = []
        in_line = False
        for i, value in enumerate(projection):
            if value > 0 and not in_line:
                start = i
                in_line = True
            elif value == 0 and in_line:
                end = i
                in_line = False
                line_indices.append((start, end))

        for idx, (start, end) in enumerate(line_indices):
            if (end - start) < 10:
                continue  # skip tiny lines

            line_img = img[start:end, :]
            all_lines.append(line_img)

            if save_dir:
                filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_line_{idx+1}.png"
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, line_img)

            if show_preview and len(preview_lines) < 4:
                preview_lines.append(line_img)

    if show_preview and preview_lines:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle('First 4 Segmented Lines (Projection Profile)', fontsize=16)

        for i in range(4):
            ax = axes[i // 2, i % 2]
            ax.imshow(preview_lines[i], cmap='gray')
            ax.axis('off')
            ax.set_title(f"Line {i+1}")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    print(f"Segmented a total of {len(all_lines)} lines from {len(image_paths)} images.")
    return all_lines