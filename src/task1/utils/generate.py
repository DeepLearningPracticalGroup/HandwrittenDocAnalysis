import os
import random
import cv2
import numpy as np



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
    noise_prob: float = 0,
) -> tuple[list[str], list[str]]:
    """
    Generate synthetic scroll-like images with multiple lines by pasting character images onto a blank canvas.

    Also adds noise maps to the generated images.
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

        if noise_prob > 0:
            canvas = apply_noise_map(canvas, noise_maps_dir="noise_maps/", prob=noise_prob)
        cv2.imwrite(os.path.join(image_dir, img_name), canvas)
        with open(os.path.join(label_dir, label_name), "w") as f:
            f.write("\n".join(labels))

        image_paths.append(img_path)
        label_paths.append(label_path)

    return image_paths, label_paths

def generate_file_scroll(
    file_path: str,
    yaml_file_path: str,
    output_dir: str,
    char_paths: list[str],
    char_labels: list[str],
    canvas_size: tuple[int, int] = (512, 1024),
    max_lines: int = 20,
    line_spacing: int = 10,
    noise_prob: float = 0,
) -> tuple[list[str], list[str]]:
    """
    Generate scroll images from a text file, keeping only Hebrew characters 
    as specified in the 'converted' dictionary from hebrew.yaml.
    Properly handles right-to-left (RTL) rendering for Hebrew.
    """
    import os
    import yaml
    import numpy as np
    import cv2
    import random

    # Load Hebrew characters and mapping from YAML
    with open(yaml_file_path, "r", encoding="utf-8") as f:
        hebrew_data = yaml.safe_load(f)
        converted = hebrew_data["converted"]
        hebrew_chars = set(converted.keys())

    # Read and filter file text (keep Hebrew chars and spaces only)
    with open(file_path, "r", encoding="utf-8") as f:
        file_text = "".join([char for char in f.read() if char in hebrew_chars or char == " "])

    # Map label names to their image paths
    char_to_path = {label: path for path, label in zip(char_paths, char_labels)}

    # Map Hebrew characters to image paths using converted labels
    hebrew_char_to_path = {
        char: char_to_path[converted[char]]
        for char in hebrew_chars
        if converted[char] in char_to_path
    }

    label_dir = os.path.join(output_dir, "labels")
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    unique_classes = sorted(set(char_labels))
    char_to_id = {char: idx for idx, char in enumerate(unique_classes)}

    image_paths = []
    label_paths = []

    lines = []
    current_line = ""
    for char in file_text:
        if char in hebrew_char_to_path or char == " ":
            current_line += char
            if len(current_line) >= canvas_size[1] // 20:
                lines.append(current_line)
                current_line = ""

    if current_line:
        lines.append(current_line)

    scroll_idx = 0
    while lines:
        canvas = np.ones(canvas_size, dtype=np.uint8) * 255
        labels = []

        y_cursor = 20
        for _ in range(min(max_lines, len(lines))):
            line = lines.pop(0)
            x_cursor = canvas_size[1] - 20  # start from right for RTL
            line_y_offset = random.randint(-5, 5)

            for char in reversed(line):  # reverse for RTL
                if char == " ":
                    x_cursor -= random.randint(10, 30)
                    continue

                if char not in hebrew_char_to_path:
                    continue

                img_path = hebrew_char_to_path[char]
                char_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if char_img is None:
                    continue

                h, w = char_img.shape

                if x_cursor - w <= 20:
                    break

                if y_cursor + h >= canvas_size[0] - 20:
                    break

                canvas[
                    y_cursor + line_y_offset : y_cursor + h + line_y_offset,
                    x_cursor - w : x_cursor,
                ] = np.minimum(
                    canvas[
                        y_cursor + line_y_offset : y_cursor + h + line_y_offset,
                        x_cursor - w : x_cursor,
                    ],
                    char_img,
                )

                label_name = converted[char]
                x_center = (x_cursor - w / 2) / canvas_size[1]
                y_center = (y_cursor + line_y_offset + h / 2) / canvas_size[0]
                width = w / canvas_size[1]
                height = h / canvas_size[0]
                labels.append(
                    f"{char_to_id[label_name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

                x_cursor -= w + random.randint(5, 20)  # move leftward

            y_cursor += h + line_spacing

        img_name = f"file_scroll_{scroll_idx:04d}.png"
        label_name = f"file_scroll_{scroll_idx:04d}.txt"

        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, label_name)

        if noise_prob > 0:
            canvas = apply_noise_map(canvas, noise_maps_dir="noise_maps/", prob=noise_prob)

        cv2.imwrite(img_path, canvas)
        with open(label_path, "w") as f:
            f.write("\n".join(labels))

        image_paths.append(img_path)
        label_paths.append(label_path)

        scroll_idx += 1

    return image_paths, label_paths
