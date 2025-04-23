import os
import random
import cv2
import numpy as np
from src.task1.utils.preprocessing import get_character_images


def generate_synthetic_scroll(
    output_dir: str,
    char_paths: list[str],
    char_labels: list[str],
    canvas_size: tuple[int, int] = (512, 1024),
    num_images: int = 100,
    min_chars: int = 5,
    max_chars: int = 20,
    min_lines: int = 2,
    max_lines: int = 10,
) -> tuple[list[str], list[str]]:
    """
    Generate synthetic scroll-like images with multiple lines by pasting character images onto a blank canvas.
    """
    label_dir = os.path.join(output_dir, "labels")
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    unique_classes = sorted(set(char_labels))
    char_to_id = {char: idx for idx, char in enumerate(unique_classes)}

    image_paths = []
    label_paths = []

    for idx in range(num_images):
        canvas = np.ones(canvas_size, dtype=np.uint8) * 255
        labels = []

        # Randomly decide number of lines
        num_lines = random.randint(min_lines, max_lines)

        # Line spacing settings
        y_cursor = 20  # Starting y position
        line_spacing = 10  # Extra space between lines

        # Calculate maximum line height roughly (not strict)
        max_line_height = (
            canvas_size[0] - 40
        ) // num_lines  # 40 is margin at top and bottom

        for line_idx in range(num_lines):
            x_cursor = 20  # Reset x position for each line
            # Small random vertical offset for realism
            line_y_offset = random.randint(-5, 5)

            for _ in range(random.randint(min_chars, max_chars)):
                i = random.randint(0, len(char_paths) - 1)
                img_path = char_paths[i]
                char_label = char_labels[i]
                char_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if char_img is None:
                    continue

                h, w = char_img.shape

                if x_cursor + w >= canvas_size[1] - 20:
                    break  # No more space horizontally

                if y_cursor + h >= canvas_size[0] - 20:
                    break  # No more space vertically

                # Paste character
                canvas[
                    y_cursor + line_y_offset : y_cursor + h + line_y_offset,
                    x_cursor : x_cursor + w,
                ] = np.minimum(
                    canvas[
                        y_cursor + line_y_offset : y_cursor + h + line_y_offset,
                        x_cursor : x_cursor + w,
                    ],
                    char_img,
                )

                # Compute YOLO bounding box
                x_center = (x_cursor + w / 2) / canvas_size[1]
                y_center = (y_cursor + line_y_offset + h / 2) / canvas_size[0]
                width = w / canvas_size[1]
                height = h / canvas_size[0]
                labels.append(
                    f"{char_to_id[char_label]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

                x_cursor += w + random.randint(5, 20)  # space between characters

            # After finishing a line, move y_cursor down
            y_cursor += max_line_height + line_spacing

        img_name = f"scroll_{idx:04d}.png"
        label_name = f"scroll_{idx:04d}.txt"

        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, label_name)

        cv2.imwrite(os.path.join(image_dir, img_name), canvas)
        with open(os.path.join(label_dir, label_name), "w") as f:
            f.write("\n".join(labels))

        image_paths.append(img_path)
        label_paths.append(label_path)

    return image_paths, label_paths
