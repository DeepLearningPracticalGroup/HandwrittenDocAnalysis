import cv2
import matplotlib.pyplot as plt


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
