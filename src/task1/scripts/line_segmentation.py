from src.task1.utils.line_segmentation import segment_all_scrolls, find_midpoints
import numpy as np
from src.task1.utils.plots import show_line_segmentation_on_image
from PIL import Image

def main():
    root_dir = "synthetic_scrolls"
    output_root="segmented_lines"
    N = 40

    image_path = "image-data/P564-Fg003-R-C01-R01-binarized.jpg"
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    segment_all_scrolls(root_dir=root_dir, output_root=output_root, N=N)

    minima, midpoints = find_midpoints(img_array, N)

    show_line_segmentation_on_image(image_path, minima, midpoints, N)
if __name__ == "__main__":
    main()