import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def preprocess_image(image_path):
    dataset = []
    for image in os.listdir(image_path):
        if image.endswith("binarized.jpg"):
            full_path = os.path.join(image_path, image)
            dataset.append(full_path)
    print(f"Found {len(dataset)} images in the dataset.")
    return dataset

def hough_transform(dataset):
    img = cv2.imread(dataset[1])
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,100)

    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
    cv2.imwrite('houghlines3.jpg',img)


    if img is None:
        print("Image not loaded.")
        return

    #plt.imshow(edges, cmap='gray')
    #plt.axis("off")
    #plt.show()