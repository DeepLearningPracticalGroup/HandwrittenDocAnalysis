from preprocessing import *

image_path = "image-data"

dataset = preprocess_image(image_path)

hough_transform(dataset)