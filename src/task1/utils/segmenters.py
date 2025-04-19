import cv2
import numpy as np
import matplotlib.pyplot as plt


def cca_segmentation(
    image_paths: list[str],
    area_threshold: int = 50,
    morph_kernel_size: int = 5,
    plot: bool = False,
):
    """
    Perform Connected Component Analysis (CCA) on a list of binarized images.
    """
    all_bboxes = {}

    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load {img_path}")
            continue

        # Ensure binary image
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Invert if necessary
        num_white = np.sum(binary == 255)
        num_black = np.sum(binary == 0)
        if num_white < num_black:
            binary = cv2.bitwise_not(binary)

        # Morphological Closing to connect broken parts
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Connected Components
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        bboxes = []
        height, width = binary.shape
        img_area = height * width

        for i in range(1, n_labels):  # **START FROM 1**, skip label 0
            x, y, w, h, area = stats[i]

            # Ignore giant components (background) and tiny ones (noise)
            if area >= area_threshold and area < 0.95 * img_area:
                bboxes.append((x, y, w, h))

        all_bboxes[img_path] = bboxes

        # Optional plotting
        if plot and idx < 4:
            img_with_boxes = (
                np.ones_like(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)) * 255
            )  # White background
            img_with_boxes[binary == 0] = (0, 0, 0)  # Black text

            for x, y, w, h in bboxes:
                cv2.rectangle(
                    img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3
                )  # green, thicker

            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            axes[0].imshow(binary, cmap="gray")
            axes[0].set_title("Original Binarized Image")
            axes[0].axis("off")

            axes[1].imshow(img_with_boxes)
            axes[1].set_title("Bounding Boxes Visualization")
            axes[1].axis("off")

            plt.suptitle(f"Segmentation Visualization: {img_path}", fontsize=16)
            plt.tight_layout()
            plt.show()

    return all_bboxes


def projection_segmentation(
    image_paths: list[str],
    line_thresh: int = 5,
    char_thresh: int = 5,
    plot: bool = False,
):
    """
    Segment characters using projection profiles.
    """
    all_bboxes = {}

    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load {img_path}")
            continue

        # Ensure binary
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        h_proj = np.sum(binary, axis=1)  # Horizontal projection
        lines = []
        inside_line = False
        for i, val in enumerate(h_proj):
            if val > line_thresh and not inside_line:
                start = i
                inside_line = True
            elif val <= line_thresh and inside_line:
                end = i
                inside_line = False
                lines.append((start, end))

        bboxes = []
        for y1, y2 in lines:
            line_img = binary[y1:y2, :]

            v_proj = np.sum(line_img, axis=0)  # Vertical projection
            inside_char = False
            for j, val in enumerate(v_proj):
                if val > char_thresh and not inside_char:
                    start = j
                    inside_char = True
                elif val <= char_thresh and inside_char:
                    end = j
                    inside_char = False
                    bboxes.append((start, y1, end - start, y2 - y1))

        all_bboxes[img_path] = bboxes

        # Plot
        if plot and idx < 4:
            img_rgb = cv2.cvtColor(255 - binary, cv2.COLOR_GRAY2RGB)  # revert inversion
            for x, y, w, h in bboxes:
                cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
            plt.figure(figsize=(12, 8))
            plt.title(f"Projection Profile Segmentation: {img_path}")
            plt.imshow(img_rgb)
            plt.axis("off")
            plt.show()

    return all_bboxes


# Perform projection segmentation but automatically tune line_thresh, car_thresh
def projection_segmentation_auto(
    image_paths, line_ratio=0.1, char_ratio=0.2, plot=False
):
    """
    Segment characters using projection profiles with automatic threshold tuning.
    """
    all_bboxes = {}

    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load {img_path}")
            continue

        # Ensure binary
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        # Horizontal projection for lines
        h_proj = np.sum(binary, axis=1)
        max_h_proj = np.max(h_proj)
        line_thresh = line_ratio * max_h_proj

        lines = []
        inside_line = False
        for i, val in enumerate(h_proj):
            if val > line_thresh and not inside_line:
                start = i
                inside_line = True
            elif val <= line_thresh and inside_line:
                end = i
                inside_line = False
                lines.append((start, end))

        bboxes = []
        for y1, y2 in lines:
            line_img = binary[y1:y2, :]

            # Vertical projection for characters within a line
            v_proj = np.sum(line_img, axis=0)
            max_v_proj = np.max(v_proj)
            char_thresh = char_ratio * max_v_proj

            inside_char = False
            for j, val in enumerate(v_proj):
                if val > char_thresh and not inside_char:
                    start = j
                    inside_char = True
                elif val <= char_thresh and inside_char:
                    end = j
                    inside_char = False
                    bboxes.append((start, y1, end - start, y2 - y1))

        all_bboxes[img_path] = bboxes

        # Visualize if plot arg is true
        if plot and idx < 4:
            img_rgb = cv2.cvtColor(255 - binary, cv2.COLOR_GRAY2RGB)
            for x, y, w, h in bboxes:
                cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            plt.figure(figsize=(12, 8))
            plt.title(f"Auto Projection Segmentation: {img_path}")
            plt.imshow(img_rgb)
            plt.axis("off")
            plt.show()

    return all_bboxes
