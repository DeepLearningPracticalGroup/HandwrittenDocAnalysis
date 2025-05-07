import os
import numpy as np
from PIL import Image


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

def save_line_with_labels(output_dir, base_name, line_img, W, top, bottom, yolo_labels):
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
            new_labels.append(f"{int(class_id)} {new_x_c:.6f} {new_y_c:.6f} {new_w:.6f} {new_h:.6f}")

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

    cropped_lines = []

    if label_path is not None:
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    for i in range(len(midpoints) - 1):
        top = max(0, midpoints[i] - margin)
        bottom = min(H, midpoints[i + 1] + margin)
        line_img = img.crop((0, top, W, bottom))
        cropped_lines.append(line_img)

        if label_path is not None:
            base_name = f"{scroll_id}_line_{i:02d}"
            save_line_with_labels(output_dir, base_name, line_img, W, top, bottom, yolo_labels)

    return cropped_lines

def segment_all_scrolls(root_dir="synthetic_scrolls", output_root="segmented_lines", N=40, margin=10):
    for split in ["train", "val"]:
        image_dir = os.path.join(root_dir, split, "images")
        label_dir = os.path.join(root_dir, split, "labels")
        output_dir = os.path.join(output_root, split)

        os.makedirs(output_dir, exist_ok=True)

        for filename in sorted(os.listdir(image_dir)):
            if not filename.endswith(".png"):
                continue

            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace(".png", ".txt"))

            print(f"Segmenting {split}/{filename}")
            segment_image_into_lines(
                image_path=image_path,
                output_dir=output_dir,
                label_path=label_path,
                N=N,
                margin=margin
            )