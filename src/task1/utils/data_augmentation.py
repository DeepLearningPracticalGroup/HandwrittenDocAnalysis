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
    expanded = ImageOps.expand(
        img, border=int(0.2 * min(img.size)), fill=(255, 255, 255)
    )

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
        enumerate(image_paths), total=len(image_paths), desc="Morphing images"
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

        i = 0
        attempts = 0
        while i < augment_per_image:
            attempts += 1
            if attempts > 10 * augment_per_image:
                print(f"Too many failed attempts for {base_name}, skipping augmentation.")
                break

            # Apply random transformations
            transformed = expand_and_transform(img)

            # Save as PPM to buffer
            ppm_buffer = io.BytesIO()
            transformed.save(ppm_buffer, format="PPM")
            ppm_buffer.seek(0)

            # Random morph parameters
            kernel_size = str(random.randint(1, 10))
            alpha = str(random.uniform(0.1, 1.0))

            # Run imagemorph and get result
            result = subprocess.run(
                [morph_exec, alpha, kernel_size],
                input=ppm_buffer.getvalue(),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            # Check if output is valid
            try:
                # Read result into numpy array without saving to disk yet
                test_img = Image.open(io.BytesIO(result.stdout)).convert("L")
                np_img = np.array(test_img)

                if np_img is None or np.mean(np_img) < 10:
                    continue  # try again
            except Exception:
                continue  # try again

            # Passed all checks: save final result
            output_filename = f"{base_name}_morph{i+1}.pgm"
            output_image_path = os.path.join(label_output_dir, output_filename)
            with open(output_image_path, "wb") as fout:
                fout.write(result.stdout)

            augmented_paths.append(output_image_path)
            augmented_labels.append(label)
            i += 1

    return augmented_paths, augmented_labels
