import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from math import sin, cos, radians
import cv2


def find_midpoints(img_array, N=80):
    row_sums = img_array.sum(axis=1)
    rsn = (row_sums - row_sums.min()) / (np.ptp(row_sums) + 1e-8)

    minima = [
        i
        for i in range(N, len(rsn) - N)
        if all(rsn[i] < rsn[i - j] for j in range(1, N + 1))
        and all(rsn[i] < rsn[i + j] for j in range(1, N + 1))
    ]

    midpts = [(minima[i] + minima[i + 1]) // 2 for i in range(len(minima) - 1)]

    if len(midpts) >= 2:
        first_diff = midpts[1] - midpts[0]
        last_diff = midpts[-1] - midpts[-2]
        midpts = (
            [max(0, midpts[0] - first_diff)]
            + midpts
            + [min(img_array.shape[0], midpts[-1] + last_diff)]
        )

    return minima, midpts


def count_intersections(img_array, x1, y1, x2, y2):
    temp_img = Image.new("L", (img_array.shape[1], img_array.shape[0]), 255)
    draw = ImageDraw.Draw(temp_img)
    draw.line([(x1, y1), (x2, y2)], fill=0, width=1)
    line_array = np.array(temp_img)
    intersection = np.logical_and(line_array == 0, img_array == 0)
    return np.sum(intersection)


def rotate_line(center_x, center_y, x, y, angle_deg):
    angle_rad = radians(angle_deg)
    x_translated = x - center_x
    y_translated = y - center_y
    x_rotated = x_translated * cos(angle_rad) - y_translated * sin(angle_rad)
    y_rotated = x_translated * sin(angle_rad) + y_translated * cos(angle_rad)
    return int(x_rotated + center_x), int(y_rotated + center_y)


def optimize_segmentation_lines(
    image_path, midpoints, angle_range=(-10, 10), angle_step=0.5
):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    img_width, img_height = img.size
    center_x = img_width // 2
    center_y = img_height // 2

    line_endpoints = []

    for midpoint in midpoints:
        left_point = (0, midpoint)
        right_point = (img_width, midpoint)
        best_angle = 0
        min_intersection = float("inf")
        best_endpoints = (left_point, right_point)

        for angle in np.arange(angle_range[0], angle_range[1] + angle_step, angle_step):
            new_left_x, new_left_y = rotate_line(center_x, center_y, *left_point, angle)
            new_right_x, new_right_y = rotate_line(
                center_x, center_y, *right_point, angle
            )

            new_left_x = max(0, min(img_width - 1, new_left_x))
            new_left_y = max(0, min(img_height - 1, new_left_y))
            new_right_x = max(0, min(img_width - 1, new_right_x))
            new_right_y = max(0, min(img_height - 1, new_right_y))

            intersections = count_intersections(
                img_array, new_left_x, new_left_y, new_right_x, new_right_y
            )

            if intersections < min_intersection:
                min_intersection = intersections
                best_angle = angle
                best_endpoints = ((new_left_x, new_left_y), (new_right_x, new_right_y))

        line_endpoints.append(best_endpoints)

    return line_endpoints


def extract_line_segments_with_masks(image_path, line_endpoints, padding=10):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    height, width = img_array.shape

    sorted_lines = sorted(line_endpoints, key=lambda x: (x[0][1] + x[1][1]) / 2)
    top_line = ((0, 0), (width - 1, 0))
    bottom_line = ((0, height - 1), (width - 1, height - 1))
    all_lines = [top_line] + sorted_lines + [bottom_line]

    masked_segments = []
    segment_bounds = []

    for i in range(len(all_lines) - 1):
        (ul_x1, ul_y1), (ul_x2, ul_y2) = all_lines[i]
        (ll_x1, ll_y1), (ll_x2, ll_y2) = all_lines[i + 1]

        ul_y1_p = max(0, ul_y1 + padding)
        ul_y2_p = max(0, ul_y2 + padding)
        ll_y1_p = min(height - 1, ll_y1 - padding)
        ll_y2_p = min(height - 1, ll_y2 - padding)

        mask = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(mask)
        polygon = [
            (ul_x1, ul_y1_p),
            (ul_x2, ul_y2_p),
            (ll_x2, ll_y2_p),
            (ll_x1, ll_y1_p),
        ]
        draw.polygon(polygon, fill=0)

        mask_array = np.array(mask)
        masked_img_array = np.copy(img_array)
        masked_img_array[mask_array == 255] = 255
        masked_img = Image.fromarray(masked_img_array)

        rows_with_black = np.where(np.any(masked_img_array < 255, axis=1))[0]
        if len(rows_with_black) > 0:
            top = max(0, rows_with_black[0] - padding)
            bottom = min(height - 1, rows_with_black[-1] + padding)
            cropped = masked_img.crop((0, top, width, bottom + 1))
            masked_segments.append(cropped)
            segment_bounds.append((top, bottom))

    return masked_segments, segment_bounds


def save_line_with_labels(line_img, output_dir, base_name, W, top, bottom, yolo_labels):
    line_img.save(os.path.join(output_dir, "images", base_name + ".png"))
    new_labels = []
    for class_id, x_c_abs, y_c_abs, w_abs, h_abs in yolo_labels:
        y_top = y_c_abs - h_abs / 2
        y_bot = y_c_abs + h_abs / 2
        if y_bot < top or y_top > bottom:
            continue
        new_y_c = (y_c_abs - top) / (bottom - top)
        new_x_c = x_c_abs / W
        new_w = w_abs / W
        new_h = h_abs / (bottom - top)
        if 0 <= new_y_c <= 1:
            new_labels.append(
                f"{int(class_id)} {new_x_c:.6f} {new_y_c:.6f} {new_w:.6f} {new_h:.6f}"
            )

    with open(os.path.join(output_dir, "labels", base_name + ".txt"), "w") as f:
        f.write("\n".join(new_labels))


def segment_image_into_lines(image_path, output_dir, label_path=None, N=80, margin=10):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    W, H = img.size
    scroll_id = os.path.splitext(os.path.basename(image_path))[0]

    yolo_labels = []
    if label_path is not None:
        with open(label_path, "r") as f:
            raw_labels = f.read().strip().splitlines()
        for line in raw_labels:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_c, y_c, w, h = map(float, parts)
                x_c_abs = x_c * W
                y_c_abs = y_c * H
                w_abs = w * W
                h_abs = h * H
                yolo_labels.append((class_id, x_c_abs, y_c_abs, w_abs, h_abs))

    _, midpoints = find_midpoints(img_array, N)
    if len(midpoints) < 2:
        return []

    line_endpoints = optimize_segmentation_lines(image_path, midpoints)
    segments, bounds = extract_line_segments_with_masks(
        image_path, line_endpoints, padding=margin
    )

    if label_path is not None:
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

        for i, (segment, (top, bottom)) in enumerate(zip(segments, bounds)):
            base_name = f"{scroll_id}_line_{i:02d}"
            save_line_with_labels(
                segment, output_dir, base_name, W, top, bottom, yolo_labels
            )
    else:
        for i, segment in enumerate(segments):
            base_name = f"{scroll_id}_line_{i:02d}.png"
            os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
            segment.save(os.path.join(output_dir, "images", base_name))

    return segments


def segment_all_scrolls(
    root_dir="synthetic_scrolls_random",
    output_root="segmented_scrolls",
    N=40,
    margin=10,
):
    for split in ["train", "val"]:
        image_dir = os.path.join(root_dir, split, "images")
        label_dir = os.path.join(root_dir, split, "labels")
        output_dir = os.path.join(output_root, split)
        os.makedirs(output_dir, exist_ok=True)

        counter = 0
        for filename in sorted(os.listdir(image_dir)):
            counter += 1
            if not filename.endswith(".png"):
                continue
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace(".png", ".txt"))
            if not os.path.exists(label_path):
                label_path = None

            # Ensure the output directories for images and labels exist
            os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

            if counter % 50 == 0:
                print(
                    f"Processing {split}/{filename} ({counter}/{len(os.listdir(image_dir))})"
                )

            segment_image_into_lines(
                image_path=image_path,
                output_dir=output_dir,
                label_path=label_path,
                N=N,
                margin=margin,
            )
