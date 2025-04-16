import cv2
import os
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    dataset = []
    for image in os.listdir(image_path):
        if image.endswith("binarized.jpg"):
            full_path = os.path.join(image_path, image)
            dataset.append(full_path)
    print(f"Found {len(dataset)} images in the dataset.")
    return dataset

def hough_transform(dataset):
    img = cv2.imread(dataset[0], cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Image not loaded.")
        return

    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.show()