import os
import random
import cv2
import numpy as np
import yaml
import re

def apply_noise_map(scroll_img, noise_maps_dir, prob=1):
    if random.random() > prob:
        return scroll_img

    h, w = scroll_img.shape
    noise_files = [f for f in os.listdir(noise_maps_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    if not noise_files:
        return scroll_img

    max_attempts = 10
    for _ in range(max_attempts):
        noise_path = os.path.join(noise_maps_dir, random.choice(noise_files))
        noise_img = cv2.imread(noise_path, cv2.IMREAD_GRAYSCALE)

        if noise_img is None or noise_img.shape[0] < h or noise_img.shape[1] < w:
            continue

        y = random.randint(0, noise_img.shape[0] - h)
        x = random.randint(0, noise_img.shape[1] - w)
        noise_crop = noise_img[y:y+h, x:x+w]

        mean_val = np.mean(noise_crop)
        if 30 <= mean_val <= 240:
            break
    else:
        return scroll_img

    mask = (noise_crop == 0).astype(np.uint8)
    damaged_scroll = scroll_img.copy()
    damaged_scroll[mask == 1] = 255

    return damaged_scroll

def warp_line(image: np.ndarray, max_displacement: int = 10, cycles: float = 1.0):
    h, w = image.shape
    x_vals = np.linspace(0, 2 * np.pi * cycles, w)
    displacement = (np.sin(x_vals) * max_displacement).astype(np.int32)

    warped = np.ones_like(image) * 255
    for x in range(w):
        dy = displacement[x]
        if dy > 0:
            warped[dy:, x] = image[:h - dy, x]
        elif dy < 0:
            warped[:h + dy, x] = image[-dy:, x]
        else:
            warped[:, x] = image[:, x]
    return warped, displacement

def apply_cutout_noise(image: np.ndarray, num_rects: int = 10, size_range: tuple = (80, 200)) -> np.ndarray:
    h, w = image.shape
    for _ in range(num_rects):
        rh = random.randint(*size_range)
        rw = random.randint(*size_range)
        y = random.randint(0, max(1, h - rh))
        x = random.randint(0, max(1, w - rw))
        mask = np.zeros((rh, rw), dtype=np.uint8)
        for _ in range(random.randint(5, 15)):
            cx = random.randint(0, rw)
            cy = random.randint(0, rh)
            radius = random.randint(10, min(rh, rw) // 2)
            cv2.circle(mask, (cx, cy), radius, 255, -1)
        roi = image[y:y+rh, x:x+rw]
        np.copyto(roi, 255, where=(mask == 255))
    return image

def filter_occluded_labels(canvas: np.ndarray, labels: list[str], visibility_thresh: float = 0.1) -> list[str]:
    H, W = canvas.shape
    valid_labels = []
    for label in labels:
        parts = label.strip().split()
        if len(parts) != 5:
            continue
        class_id, x_center, y_center, width, height = map(float, parts)
        x_center_abs = x_center * W
        y_center_abs = y_center * H
        w_abs = width * W
        h_abs = height * H
        x1 = max(0, int(x_center_abs - w_abs / 2))
        x2 = min(W, int(x_center_abs + w_abs / 2))
        y1 = max(0, int(y_center_abs - h_abs / 2))
        y2 = min(H, int(y_center_abs + h_abs / 2))
        region = canvas[y1:y2, x1:x2]
        visible_fraction = np.mean(region < 250)
        if visible_fraction >= visibility_thresh:
            valid_labels.append(label)
    return valid_labels

def generate_synthetic_scroll(
    output_dir: str,
    char_paths: list[str],
    char_labels: list[str],
    canvas_size: tuple[int, int] = (1024, 2048),
    num_images: int = 100,
    min_chars: int = 5,
    max_chars: int = 20,
    min_lines: int = 2,
    max_lines: int = 10,
    noise_prob: float = 0,
    offset_range: tuple[int, int] = (100, 300),
):
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    unique_classes = sorted(set(char_labels))
    char_to_id = {char: idx for idx, char in enumerate(unique_classes)}

    image_paths = []
    label_paths = []

    for idx in range(num_images):
        canvas = np.ones(canvas_size, dtype=np.uint8) * 255
        labels = []

        num_lines = random.randint(min_lines, max_lines)
        avg_line_height = 60
        estimated_text_height = num_lines * avg_line_height
        y_cursor = max(20, (canvas_size[0] - estimated_text_height) // 2 + random.randint(-30, 30))

        for _ in range(num_lines):
            num_chars = random.randint(min_chars, max_chars)
            line_chars = []
            line_width = 0
            for _ in range(num_chars):
                i = random.randint(0, len(char_paths) - 1)
                img = cv2.imread(char_paths[i], cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    line_chars.append((img, char_labels[i]))
                    line_width += img.shape[1] + 10

            if not line_chars:
                continue

            x_cursor = (canvas_size[1] - line_width) // 2
            line_img = np.ones((avg_line_height, canvas_size[1]), dtype=np.uint8) * 255

            max_disp = random.randint(5, 10)
            padding = max_disp + 5
            padded = np.ones((avg_line_height + 2 * padding, canvas_size[1]), dtype=np.uint8) * 255
            padded[padding:padding + avg_line_height, :] = line_img
            cycles = random.uniform(0.5, 2.0)
            warped_line_img, displacement = warp_line(padded, max_displacement=max_disp, cycles=cycles)

            line_labels = []
            for char_img, char_label in line_chars:
                h, w = char_img.shape
                if x_cursor + w >= canvas_size[1]:
                    break

                x1 = x_cursor
                x2 = x_cursor + w
                dx_range = displacement[x1:x2] if x2 > x1 else [0]
                dy = int(np.mean(dx_range)) if len(dx_range) > 0 else 0

                top = padding + dy
                bottom = top + h
                left = x_cursor
                right = x_cursor + w

                if bottom <= warped_line_img.shape[0]:
                    warped_line_img[top:bottom, left:right] = np.minimum(
                        warped_line_img[top:bottom, left:right], char_img
                    )
                    x_center = (left + w / 2) / canvas_size[1]
                    y_center = (y_cursor + top + h / 2) / canvas_size[0]
                    width = w / canvas_size[1]
                    height = h / canvas_size[0]
                    line_labels.append(
                        f"{char_to_id[char_label]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    )

                x_cursor += w + random.randint(-10, 10)

            if y_cursor + warped_line_img.shape[0] > canvas.shape[0]:
                break

            canvas[y_cursor:y_cursor + warped_line_img.shape[0], :] = np.minimum(
                canvas[y_cursor:y_cursor + warped_line_img.shape[0], :], warped_line_img
            )
            labels.extend(line_labels)
            y_cursor += int(warped_line_img.shape[0] * random.uniform(1.05, 1.2))

        canvas = apply_cutout_noise(canvas, num_rects=10, size_range=(100, 250))

        if noise_prob > 0:
            canvas = apply_noise_map(canvas, noise_maps_dir="noise_maps/binarized", prob=noise_prob)

        labels = filter_occluded_labels(canvas, labels, visibility_thresh=0.1)

        img_name = f"scroll_{idx:04d}.png"
        label_name = f"scroll_{idx:04d}.txt"
        img_path = os.path.join(output_dir, "images", img_name)
        label_path = os.path.join(output_dir, "labels", label_name)

        cv2.imwrite(img_path, canvas)
        with open(label_path, "w") as f:
            f.write("\n".join(labels))

        image_paths.append(img_path)
        label_paths.append(label_path)

    return image_paths, label_paths


def estimate_line_width(line, char_shape_map=None):
        return sum(
            char_shape_map.get(char, (20, 20))[1] + (random.randint(0, 5) if char != " " else random.randint(15, 25))
            for char in line
        )

def generate_file_scroll_alternative(
        file_path: str,
        yaml_file_path: str,
        output_dir: str,
        char_paths: list[str],
        char_labels: list[str],
        canvas_size: tuple[int, int] = (512, 1024),
        max_lines: int = 20,
        line_spacing: int = 10,
        noise_prob: float = 0,
        verbose: bool = False,
    ) -> tuple[list[str], list[str]]:
    """
    Generate scroll images from a Hebrew text file with right-to-left rendering,
    realistic character spacing, and optional noise augmentation.
    """

    # Load Hebrew characters and mapping from YAML
    with open(yaml_file_path, "r", encoding="utf-8") as f:
        hebrew_data = yaml.safe_load(f)
        converted = hebrew_data["converted"]
        hebrew_chars = set(converted.keys())

    # Clean up and filter text
    with open(file_path, "r", encoding="utf-8") as f:
        file_text = re.sub(r"\s+", " ", f.read().strip())
        file_text = "".join([char for char in file_text if char in hebrew_chars or char == " "])

    # Map label names to their image paths
    char_to_path = {label: path for path, label in zip(char_paths, char_labels)}

    # Build image cache and size map
    hebrew_char_to_path = {
        char: char_to_path[converted[char]]
        for char in hebrew_chars
        if converted[char] in char_to_path
    }
    char_img_cache = {
        char: cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        for char, path in hebrew_char_to_path.items()
    }
    char_shape_map = {
        char: img.shape for char, img in char_img_cache.items() if img is not None
    }

    # Split lines based on estimated pixel width
    lines, current_line = [], ""
    for char in file_text:
        if char in hebrew_char_to_path or char == " ":
            test_line = current_line + char
            if estimate_line_width(test_line, char_shape_map) >= canvas_size[1] - 40:
                lines.append(current_line)
                current_line = char
            else:
                current_line = test_line
    if current_line:
        lines.append(current_line)

    # Prepare output dirs
    label_dir = os.path.join(output_dir, "labels")
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    unique_classes = sorted(set(char_labels))
    char_to_id = {char: idx for idx, char in enumerate(unique_classes)}

    image_paths = []
    label_paths = []
    scroll_idx = 0

    while lines:
        canvas = np.ones(canvas_size, dtype=np.uint8) * 255
        labels = []
    
        # Randomly select a top margin
        y_cursor = random.randint(10, 280) 

        for _ in range(min(max_lines, len(lines))):
            line = lines.pop(0)

            # Random starting point (not always flush right)
            initial_margin = random.randint(10, 260)
            x_cursor = canvas_size[1] - initial_margin

            # Random vertical offset for the whole line
            line_y_offset = random.randint(-10, 10)

            if verbose:
                print(f"[Line] {line}")

            for char in line:
                if char == " ":
                    # Vary word spacing more broadly
                    x_cursor -= random.randint(10, 50)
                    continue

                char_img = char_img_cache.get(char)
                if char_img is None:
                    continue

                h, w = char_img.shape

                if x_cursor - w <= 0 or y_cursor + h >= canvas_size[0] - 10:
                    break

                # Character-level jitter
                jitter_y = line_y_offset + random.randint(-3, 3)
                jitter_x = random.randint(-3, 3)

                y1 = max(0, y_cursor + jitter_y)
                y2 = y1 + h
                x1 = max(0, x_cursor - w + jitter_x)
                x2 = x1 + w

                y2 = min(y2, canvas_size[0])
                x2 = min(x2, canvas_size[1])
                cropped_char = char_img[:y2 - y1, :x2 - x1]

                canvas[y1:y2, x1:x2] = np.minimum(canvas[y1:y2, x1:x2], cropped_char)

                label_name = converted[char]
                x_center = (x1 + (x2 - x1) / 2) / canvas_size[1]
                y_center = (y1 + (y2 - y1) / 2) / canvas_size[0]
                width = (x2 - x1) / canvas_size[1]
                height = (y2 - y1) / canvas_size[0]

                labels.append(
                    f"{char_to_id[label_name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

                # Overlap encouraged, especially at word/char level
                x_cursor -= w - random.randint(-4, 16)

            # Randomize vertical spacing between lines
            y_cursor += h + random.randint(line_spacing - 5, line_spacing + 12)


        img_name = f"file_scroll_{scroll_idx:04d}.png"
        label_name = f"file_scroll_{scroll_idx:04d}.txt"

        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, label_name)

        if noise_prob > 0:
            canvas = apply_noise_map(canvas, noise_maps_dir="noise_maps/binarized", prob=noise_prob)

        cv2.imwrite(img_path, canvas)
        with open(label_path, "w") as f:
            f.write("\n".join(labels))

        image_paths.append(img_path)
        label_paths.append(label_path)
        scroll_idx += 1

    return image_paths, label_paths
