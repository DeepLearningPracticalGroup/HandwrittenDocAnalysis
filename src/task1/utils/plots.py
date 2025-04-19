import cv2
import matplotlib.pyplot as plt

def img_visualization(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Image not loaded.")
        return

    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.show()