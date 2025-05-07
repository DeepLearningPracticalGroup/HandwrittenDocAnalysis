import os
import random
import cv2
import numpy as np
from typing import List, Tuple
import pandas as pd
import yaml
import re
from collections import defaultdict


def apply_noise_map(scroll_img, noise_maps_dir, prob=1):
    if random.random() > prob:
        return scroll_img

    h, w = scroll_img.shape
    noise_files = [
        f for f in os.listdir(noise_maps_dir) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
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
        noise_crop = noise_img[y : y + h, x : x + w]

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
            warped[dy:, x] = image[: h - dy, x]
        elif dy < 0:
            warped[: h + dy, x] = image[-dy:, x]
        else:
            warped[:, x] = image[:, x]
    return warped, displacement


def apply_cutout_noise(
    image: np.ndarray, num_rects: int = 10, size_range: tuple = (80, 200)
) -> np.ndarray:
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
        roi = image[y : y + rh, x : x + rw]
        np.copyto(roi, 255, where=(mask == 255))
    return image


def filter_occluded_labels(
    canvas: np.ndarray, labels: list[str], visibility_thresh: float = 0.1
) -> list[str]:
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


def render_scroll_from_lines(
    lines: List[List[Tuple[np.ndarray, str]]],
    char_to_id: dict,
    canvas_size: Tuple[int, int],
    output_dir: str,
    scroll_idx: int,
    noise_prob: float = 0,
    noise_maps_dir: str = "noise_maps/binarized",
) -> Tuple[str, str]:

    canvas = np.ones(canvas_size, dtype=np.uint8) * 255
    labels = []
    y_cursor = 20

    for line_chars in lines:
        if not line_chars:
            continue

        x_cursor = (canvas_size[1] - sum(img.shape[1] for img, _ in line_chars)) // 2
        line_img = np.ones((60, canvas_size[1]), dtype=np.uint8) * 255

        max_disp = random.randint(10, 25)
        padding = max_disp + 5
        padded = (
            np.ones((line_img.shape[0] + 2 * padding, canvas_size[1]), dtype=np.uint8)
            * 255
        )
        padded[padding : padding + line_img.shape[0], :] = line_img
        cycles = random.uniform(0.3, 3.0)
        warped_line_img, displacement = warp_line(
            padded, max_displacement=max_disp, cycles=cycles
        )

        line_labels = []

        for char_img, char_label in line_chars:
            h, w = char_img.shape
            if x_cursor + w >= canvas_size[1]:
                break  # tronca la riga se non ci sta

            x1, x2 = x_cursor, x_cursor + w
            if (
                w <= 0
                or x1 >= displacement.shape[0]
                or x2 > displacement.shape[0]
                or x2 <= x1
            ):
                continue

            displacement_slice = displacement[x1:x2]
            if displacement_slice.size == 0 or np.isnan(displacement_slice).any():
                continue

            dy = int(np.mean(displacement_slice))

            top = padding + dy
            bottom = top + h
            left = x_cursor
            right = x_cursor + w

            if (
                left < 0
                or top < 0
                or right > warped_line_img.shape[1]
                or bottom > warped_line_img.shape[0]
            ):
                continue

            warped_line_img[top:bottom, left:right] = np.minimum(
                warped_line_img[top:bottom, left:right], char_img
            )

            x_center = (left + w / 2) / canvas_size[1]
            y_center = (y_cursor + top + h / 2) / canvas_size[0]
            width = w / canvas_size[1]
            height = h / canvas_size[0]

            if char_label != "SPACE":
                line_labels.append(
                    f"{char_to_id[char_label]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

            x_cursor += w + random.randint(-10, 10)

        if y_cursor + warped_line_img.shape[0] > canvas.shape[0]:
            break

        canvas[y_cursor : y_cursor + warped_line_img.shape[0], :] = np.minimum(
            canvas[y_cursor : y_cursor + warped_line_img.shape[0], :], warped_line_img
        )
        labels.extend(line_labels)
        y_cursor += int(warped_line_img.shape[0] * random.uniform(0.8, 1))

    canvas = apply_cutout_noise(canvas, num_rects=10, size_range=(100, 250))
    if noise_prob > 0:
        canvas = apply_noise_map(canvas, noise_maps_dir=noise_maps_dir, prob=noise_prob)

    labels = filter_occluded_labels(canvas, labels, visibility_thresh=0.1)

    img_name = f"scroll_{scroll_idx:04d}.png"
    label_name = f"scroll_{scroll_idx:04d}.txt"

    image_dir = os.path.join(output_dir, "images")
    label_dir = os.path.join(output_dir, "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    img_path = os.path.join(image_dir, img_name)
    label_path = os.path.join(label_dir, label_name)

    cv2.imwrite(img_path, canvas)
    with open(label_path, "w") as f:
        f.write("\n".join(labels))

    return img_path, label_path


def generate_synthetic_scroll_with_ngrams(
    output_dir: str,
    char_paths: List[str],
    char_labels: List[str],
    canvas_size: Tuple[int, int] = (1024, 2048),
    num_images: int = 100,
    min_lines: int = 2,
    max_lines: int = 10,
    noise_prob: float = 0,
    ngram_csv_path: str = "ngrams_frequencies_withNames.csv",
):
    # Load n-gram data
    df = pd.read_csv(ngram_csv_path)
    all_ngrams = df["Names"].astype(str).tolist()
    all_weights = df["Frequencies"].astype(float).tolist()

    # Build char to path map
    char_to_paths = defaultdict(list)
    for path, label in zip(char_paths, char_labels):
        char_to_paths[label].append(path)

    # Filter ngrams with all valid labels
    filtered_ngrams = []
    filtered_weights = []
    for ngram, weight in zip(all_ngrams, all_weights):
        labels = ngram.split("_")
        if all(label in char_to_paths for label in labels):
            filtered_ngrams.append(labels)  # store as list of labels
            filtered_weights.append(weight)

    # Prepare class-id mapping
    unique_classes = sorted(set(char_labels))
    char_to_id = {char: idx for idx, char in enumerate(unique_classes)}

    all_image_paths = []
    all_label_paths = []

    for idx in range(num_images):
        lines = []
        for _ in range(random.randint(min_lines, max_lines)):
            line = []
            num_ngrams = random.randint(2, 6)
            selected_ngrams = random.choices(
                filtered_ngrams, weights=filtered_weights, k=num_ngrams
            )

            for ngram_labels in selected_ngrams:
                for label in reversed(ngram_labels):  # RTL
                    version_path = random.choice(char_to_paths[label])
                    char_img = cv2.imread(version_path, cv2.IMREAD_GRAYSCALE)
                    if char_img is not None:
                        line.append((char_img, label))

            lines.append(line)

        img_path, label_path = render_scroll_from_lines(
            lines, char_to_id, canvas_size, output_dir, idx, noise_prob=noise_prob
        )
        all_image_paths.append(img_path)
        all_label_paths.append(label_path)

    return all_image_paths, all_label_paths


def generate_file_scroll(
    file_path: str,
    yaml_file_path: str,
    output_dir: str,
    char_paths: List[str],
    char_labels: List[str],
    canvas_size: Tuple[int, int] = (512, 1024),
    max_lines: int = 20,
    line_spacing: int = 10,
    noise_prob: float = 0,
    words_per_line_range: Tuple[int, int] = (2, 8),
    verbose: bool = False,
) -> Tuple[List[str], List[str]]:

    with open(yaml_file_path, "r", encoding="utf-8") as f:
        hebrew_data = yaml.safe_load(f)
        converted = hebrew_data["converted"]
        hebrew_chars = set(converted.keys())

    with open(file_path, "r", encoding="utf-8") as f:
        text = re.sub(r"\s+", " ", f.read().strip())
        text = "".join([c for c in text if c in hebrew_chars or c == " "])

    words = text.split()

    char_to_paths = defaultdict(list)
    for path, label in zip(char_paths, char_labels):
        char_to_paths[label].append(path)

    hebrew_char_to_paths = {
        c: char_to_paths[converted[c]]
        for c in hebrew_chars
        if converted[c] in char_to_paths
    }

    lines = []
    i = 0
    while i < len(words):
        num_words = random.randint(*words_per_line_range)
        line_words = words[i : i + num_words]
        line = " ".join(line_words)
        lines.append(line)
        i += num_words

    char_to_id = {char: idx for idx, char in enumerate(sorted(set(char_labels)))}
    image_paths, label_paths = [], []
    scroll_idx = 0

    while lines:
        batch = lines[:max_lines]
        lines = lines[max_lines:]

        line_data = []
        for line in batch:
            row = []
            for c in line:
                if c == " ":
                    space_width = random.randint(30, 50)
                    row.append(
                        (np.ones((20, space_width), dtype=np.uint8) * 255, "SPACE")
                    )
                elif c in hebrew_char_to_paths:
                    version_path = random.choice(hebrew_char_to_paths[c])
                    img = cv2.imread(version_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        row.append((img, converted[c]))
            line_data.append(row)

        img_path, label_path = render_scroll_from_lines(
            line_data,
            char_to_id,
            canvas_size,
            output_dir,
            scroll_idx,
            noise_prob=noise_prob,
        )
        image_paths.append(img_path)
        label_paths.append(label_path)
        scroll_idx += 1

    return image_paths, label_paths
