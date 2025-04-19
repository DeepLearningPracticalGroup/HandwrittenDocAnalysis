import os
import random
import cv2
import numpy as np
from src.task1.utils.preprocessing import get_character_images


def generate_synthetic_scroll(
    output_dir: str,
    root_path: str,
    canvas_size: tuple[int, int] = (256, 1024),
    num_images: int = 100
) -> None:
    """
    Generate synthetic scroll-like images by pasting individual character images onto a blank canvas.
    """
    # Create output directories if they don't exist
    label_dir = os.path.join(output_dir, 'labels')
    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # Load all character images into a dictionary {class_name: [list of image paths]}
    all_chars = get_character_images(root_path)

    # Create a mapping from character name to class ID (needed for YOLO)
    char_to_id = {char: idx for idx, char in enumerate(sorted(all_chars.keys()))}

    for idx in range(num_images):
        # Create a blank white canvas
        canvas = np.ones(canvas_size, dtype=np.uint8) * 255  # white background
        labels = []

        x_cursor = 20  # Starting x position with margin
        y_cursor = random.randint(30, 70)  # Slight vertical jitter for more variation

        # Randomly choose number of characters to paste
        for _ in range(random.randint(5, 20)):
            # Randomly pick a character class and an image
            char_class = random.choice(list(all_chars.keys()))
            img_path = random.choice(all_chars[char_class])
            char_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if char_img is None:
                continue  # Skip if image failed to load

            h, w = char_img.shape

            # Check if there is space to paste
            if x_cursor + w >= canvas_size[1] - 20:
                break  # No more space on the canvas

            # Paste the character image onto the canvas
            canvas[y_cursor:y_cursor+h, x_cursor:x_cursor+w] = np.minimum(
                canvas[y_cursor:y_cursor+h, x_cursor:x_cursor+w],
                char_img
            )

            # Calculate normalized YOLO bounding box (x_center, y_center, width, height)
            x_center = (x_cursor + w / 2) / canvas_size[1]
            y_center = (y_cursor + h / 2) / canvas_size[0]
            width = w / canvas_size[1]
            height = h / canvas_size[0]
            labels.append(f"{char_to_id[char_class]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Move the cursor for the next character
            x_cursor += w + random.randint(5, 20)  # Random space between characters

        # Save the synthetic image
        img_name = f'scroll_{idx:04d}.png'
        label_name = f'scroll_{idx:04d}.txt'

        cv2.imwrite(os.path.join(image_dir, img_name), canvas)

        # Save the corresponding YOLO label file
        with open(os.path.join(label_dir, label_name), 'w') as f:
            f.write('\n'.join(labels))


def generate_synthetic_multiline_scroll(
    output_dir: str,
    char_folders: list[str],
    canvas_size: tuple[int, int] = (768, 1024),
    num_images: int = 100
) -> None:
    """
    Generate synthetic scroll-like images with multiple lines of characters.
    """
    label_dir = os.path.join(output_dir, 'labels')
    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    all_chars = {}
    for char_folder in char_folders:
        images = [os.path.join(char_folder, f) for f in os.listdir(char_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pgm'))]
        if images:
            all_chars[char_folder] = images

    char_to_id = {char: idx for idx, char in enumerate(sorted(all_chars.keys()))}

    for idx in range(num_images):
        canvas = np.ones(canvas_size, dtype=np.uint8) * 255
        labels = []

        y_cursor = 50
        line_height = 60

        while y_cursor + line_height < canvas_size[0] - 20:
            x_cursor = 20

            for _ in range(random.randint(5, 15)):
                if x_cursor > canvas_size[1] - 50:
                    break

                char_class = random.choice(list(all_chars.keys()))
                img_path = random.choice(all_chars[char_class])
                char_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if char_img is None:
                    continue

                h, w = char_img.shape

                # Skip if character doesn't fit
                if x_cursor + w >= canvas_size[1] - 20 or y_cursor + h >= canvas_size[0] - 20:
                    continue

                # Paste character
                canvas[y_cursor:y_cursor+h, x_cursor:x_cursor+w] = np.minimum(
                    canvas[y_cursor:y_cursor+h, x_cursor:x_cursor+w],
                    char_img
                )

                # Record label
                x_center = (x_cursor + w/2) / canvas_size[1]
                y_center = (y_cursor + h/2) / canvas_size[0]
                width = w / canvas_size[1]
                height = h / canvas_size[0]
                labels.append(f"{char_to_id[char_class]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                x_cursor += w + random.randint(10, 30)  # Space between characters

            y_cursor += line_height + random.randint(20, 40)  # Move to next line

        img_name = f'scroll_{idx:04d}.png'
        label_name = f'scroll_{idx:04d}.txt'

        cv2.imwrite(os.path.join(image_dir, img_name), canvas)

        with open(os.path.join(label_dir, label_name), 'w') as f:
            f.write('\n'.join(labels))
