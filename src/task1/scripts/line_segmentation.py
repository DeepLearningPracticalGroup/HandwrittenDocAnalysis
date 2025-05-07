from src.task1.utils.line_segmentation import (
    process_all_scrolls,
    find_midpoints,
    optimize_segmentation_lines
)
from src.task1.utils.plots import show_line_segmentation_on_image
from PIL import Image
import numpy as np
import os

def main():
    root_dir = "synthetic_scrolls_text"
    output_root = "segmented_lines"
    N = 40

    image_path = "image-data/P564-Fg003-R-C01-R01-binarized.jpg"
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Prepara immagine e array
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)

    # Trova minima e midpoints
    minima, midpoints = find_midpoints(img_array, N=N)

    # Ottimizza le linee di segmentazione
    optimized_lines = optimize_segmentation_lines(
        image_path=image_path,
        midpoints=midpoints,
        angle_range=(-10, 10),
        angle_step=0.5
    )

    # Visualizza tutto insieme
    show_line_segmentation_on_image(
        image_path=image_path,
        minima=minima,
        midpoints=midpoints,
        optimized_lines=optimized_lines,
        N=N
    )

    # Segmenta tutte le immagini nella directory
    process_all_scrolls(root_dir=root_dir, output_root=output_root, N=N)

if __name__ == "__main__":
    main()