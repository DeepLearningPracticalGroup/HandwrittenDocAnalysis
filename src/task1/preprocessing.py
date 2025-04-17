import cv2
import os
import matplotlib.pyplot as plt

def get_images(image_path:str, endswith:str):
    dataset = []
    for image in os.listdir(image_path):
        if image.endswith(endswith):
            full_path = os.path.join(image_path, image)
            dataset.append(full_path)
    print(f"Found {len(dataset)} {endswith} images in the dataset.")
    return dataset

def get_binarized_images(image_path):
    return get_images(image_path=image_path, endswith='binarized.jpg')

def get_fused_images(image_path):
    return get_images(image_path=image_path, endswith='fused.jpg')

def get_rgb_images(image_path):
    dataset = []
    for image in os.listdir(image_path):
        if not image.endswith("binarized.jpg") and not image.endswith("fused.jpg"):
            full_path = os.path.join(image_path, image)
            dataset.append(full_path)
    print(f"Found {len(dataset)} rgb images in the dataset.")
    return dataset 


def hough_transform(dataset):
    img = cv2.imread(dataset[0], cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Image not loaded.")
        return

    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.show()