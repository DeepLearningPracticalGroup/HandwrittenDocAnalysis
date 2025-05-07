import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



def img_visualization(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Image not loaded.")
        return
    
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()


# Visualize a single image
def plot_image(image_path: str, title: str = "Image", cmap: str = None) -> None:
    """Plots an image from the given file path using matplotlib."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert BGR (OpenCV default) to RGB if not grayscale
    if cmap != "gray":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()

def show_line_segmentation_on_image(image_path, minima=None, midpoints=None, optimized_lines=None, N=None):
    """
    Visualizza la segmentazione delle righe su un'immagine.
    Supporta sia linee orizzontali (midpoints) sia linee inclinate (optimized_lines).
    """
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_array, cmap="gray", vmin=0, vmax=255)

    if minima is not None:
        for y in minima:
            plt.axhline(y, color="red", linewidth=1.5, linestyle="--", alpha=0.6)

    if midpoints is not None:
        for y in midpoints:
            plt.axhline(y, color="cyan", linewidth=1)

    if optimized_lines is not None:
        for (x1, y1), (x2, y2) in optimized_lines:
            plt.plot([x1, x2], [y1, y2], color="lime", linewidth=2)

    title = "Segmentazione in righe"
    if N is not None:
        title += f" (N={N})"
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()