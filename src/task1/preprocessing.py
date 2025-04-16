import cv2
import os

def preprocess_image(image_path):
    dataset = []
    for image in os.listdir(image_path):
        if image.endswith("binarized.jpg"):
            dataset.append(image)
    print(f"Found {len(dataset)} images in the dataset.")
    return dataset