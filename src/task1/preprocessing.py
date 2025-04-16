import cv2
import os

def preprocess_image(image_path):
    dataset = []
    for image in os.listdir(image_path):
        if image.endswith("binarized.jpg"):
            full_path = os.path.join(image_path, image)
            dataset.append(full_path)
    print(f"Found {len(dataset)} images in the dataset.")
    return dataset

def hough_transform(dataset):
    print("Trying to read:", dataset[0])
    img = cv2.imread(dataset[0], cv2.IMREAD_GRAYSCALE)
    print("Loaded image:", img)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()