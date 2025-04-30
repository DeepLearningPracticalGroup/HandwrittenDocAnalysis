import os
import io
import subprocess
from PIL import Image

from torchvision import transforms
from tqdm import tqdm
import torchvision.transforms.functional as F
import random


class RandomResize(object):
    def __init__(self, scale_range=(0.95, 1.05)):
        self.scale_range = scale_range

    def __call__(self, img):
        width, height = img.size
        scale = random.uniform(*self.scale_range)

        # Calculate crop region
        new_width = int(width / scale)
        new_height = int(height / scale)

        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height

        # Crop the zoomed area
        img = img.crop((left, top, right, bottom))

        # Resize back to original size
        img = img.resize((width, height), resample=Image.BICUBIC)

        return img


def imagemorph_augmentation(
    image_paths,
    labels,
    output_dir,
    morph_exec="./imagemorph",
    num_augments=1,
    alpha="0.9",
    kernel_size="9",
):

    os.makedirs(output_dir, exist_ok=True)
    augmented_image_paths, augmented_labels = [], []

    for idx, input_image_path in enumerate(image_paths):
        label = labels[idx] if labels else ""
        label_output_dir = os.path.join(output_dir, label)
        os.makedirs(label_output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(input_image_path))[0]

        img = Image.open(input_image_path).convert("RGB")
        ppm_buffer = io.BytesIO()
        img.save(ppm_buffer, format="PPM")
        ppm_buffer.seek(0)

        for i in range(num_augments):
            output_filename = f"{base_name}_morph{i+1}.pgm"
            output_image_path = os.path.join(label_output_dir, output_filename)

            with open(output_image_path, "wb") as fout:
                subprocess.run(
                    [morph_exec, alpha, kernel_size],
                    input=ppm_buffer.getvalue(),
                    stdout=fout,
                    stderr=subprocess.DEVNULL,
                )

            augmented_image_paths.append(output_image_path)
            augmented_labels.append(label)

            ppm_buffer.seek(0)

    return augmented_image_paths, augmented_labels


def baseline_augmentation(image_paths, labels, output_dir, num_augments=3, seed=42):
    """
    Applies augmentations to the given dataset of character images.

    Parameters:
    - image_paths: List of paths to original images
    - labels: List of corresponding labels (same length as image_paths)
    - output_dir: Folder to save augmented images, structured by label/class
    - num_augments: Number of augmented versions to generate per image
    - image_size: Output image size as tuple
    - seed: For reproducibility
    """

    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    augmentation_pipeline = transforms.Compose(
        [
            RandomResize(scale_range=(0.95, 1.05)),
            transforms.RandomAffine(degrees=8),
            transforms.ToTensor(),
            transforms.ToPILImage(),
        ]
    )

    augmented_image_paths = []
    augmented_labels = []

    for img_path, label in tqdm(
        zip(image_paths, labels), total=len(image_paths), desc="Augmenting dataset"
    ):
        try:
            img = Image.open(img_path).convert("L")  # grayscale
        except Exception as e:
            print(f"Failed to open {img_path}: {e}")
            continue

        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Save augmented versions
        for i in range(num_augments):
            aug_img = augmentation_pipeline(img)
            aug_save_path = os.path.join(label_dir, f"{base_name}_aug{i}.png")
            aug_img.save(aug_save_path)
            augmented_image_paths.append(aug_save_path)
            augmented_labels.append(label)

    return augmented_image_paths, augmented_labels
