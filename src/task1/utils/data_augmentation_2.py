import os
import io
import subprocess
from PIL import Image, ImageOps
from tqdm import tqdm
import random
import cv2
import numpy as np

def expand_and_transform(img, degrees=5, scale_range=(0.95, 1.05)):
    orig_size = img.size

    # Expand the image with a white border
    expanded = ImageOps.expand(img, border=int(0.2 * min(img.size)), fill=(255, 255, 255))

    # Rotate random
    angle = random.uniform(-degrees, degrees)
    rotated = expanded.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))

    # Resize random (zoom in/out)
    scale = random.uniform(*scale_range)
    new_size = [int(s * scale) for s in rotated.size]
    resized = rotated.resize(new_size, resample=Image.BICUBIC)

    # Crop back to original size
    left = (resized.width - orig_size[0]) // 2
    top = (resized.height - orig_size[1]) // 2
    final = resized.crop((left, top, left + orig_size[0], top + orig_size[1]))

    return final

def imagemorph_augmentation(
    image_paths,
    labels,
    output_dir,
    morph_exec="./imagemorph",
    augment_per_image=1,
):

    os.makedirs(output_dir, exist_ok=True)
    augmented_paths, augmented_labels = [], []

    for idx, input_image_path in tqdm(
        enumerate(image_paths),
        total=len(image_paths),
        desc="Morphing images"
    ):
        label = str(labels[idx]) if labels else "unknown"
        label_output_dir = os.path.join(output_dir, label)
        os.makedirs(label_output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(input_image_path))[0]

        try:
            img = Image.open(input_image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open {input_image_path}: {e}")
            continue

        for i in range(augment_per_image):
            output_filename = f"{base_name}_morph{i+1}.pgm"
            output_image_path = os.path.join(label_output_dir, output_filename)

            # Apply random transformations
            img = expand_and_transform(img, degrees=10, scale_range=(0.95, 1.05))

            ppm_buffer = io.BytesIO()
            img.save(ppm_buffer, format="PPM")
            ppm_buffer.seek(0)

            # Generate a random kernel size and alpha value (to string)           
            kernel_size = str(random.randint(1, 10))
            alpha = str(random.uniform(0.1, 1.0))

            with open(output_image_path, "wb") as fout:
                subprocess.run(
                    [morph_exec, alpha, kernel_size],
                    input=ppm_buffer.getvalue(),
                    stdout=fout,
                    stderr=subprocess.DEVNULL,
                )

            augmented_paths.append(output_image_path)
            augmented_labels.append(label)

            ppm_buffer.seek(0)

    return augmented_paths, augmented_labels