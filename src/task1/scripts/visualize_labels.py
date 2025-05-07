"""

Script to visualize scroll images with bounding boxes from label files.

to execute:
ipython src/task1/scripts/visualize_labels.py -- --image_path "segmented_lines/train/images/scroll_0001_line_00.png" \
--label_path "segmented_lines/train/labels/scroll_0001_line_00.txt"

"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2
import argparse

def visualize_scroll_with_boxes(image_path: str, label_path: str, box_color="red", box_thickness=1):
    """
    Visualizes a scroll image with its corresponding labels as bounding boxes.
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    H, W = img.shape

    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.imshow(img, cmap="gray")
    ax.axis("off")

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, x_center, y_center, width, height = map(float, parts)
                
                x = (x_center - width / 2) * W
                y = (y_center - height / 2) * H
                w = width * W
                h = height * H
                
                rect = patches.Rectangle((x, y), w, h, linewidth=box_thickness,
                                         edgecolor=box_color, facecolor="none")
                ax.add_patch(rect)

    plt.title(f"Bounding Boxes: {os.path.basename(image_path)}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize bounding boxes on an image.")
    parser.add_argument("--image_path", type=str, help="Path to the image file.")
    parser.add_argument("--label_path", type=str, help="Path to the label file.")
    parser.add_argument("--box_color", type=str, default="red", help="Color of the bounding boxes.")
    parser.add_argument("--box_thickness", type=int, default=1, help="Thickness of the bounding box edges.")
    
    args = parser.parse_args()
    
    visualize_scroll_with_boxes(args.image_path, args.label_path, args.box_color, args.box_thickness)

if __name__ == "__main__":
    main()