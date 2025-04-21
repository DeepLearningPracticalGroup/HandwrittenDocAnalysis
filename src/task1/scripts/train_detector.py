"""
Task 01: DSS dataset
(a) Preprocessing and character segmentation
(b) Character recognition

to execute this script:
first pip install ipython
then enter the following command in terminal:
ipython src/task1/scripts/main.py
or
<env_name>/bin/ipython src/task1/scripts/main.py
or
myenv/bin/ipython src/task1/scripts/main.py
"""

from time import perf_counter
import random
import os
from src.task1.utils.preprocessing import *
from sklearn.model_selection import train_test_split
from src.task1.utils.generate import generate_synthetic_scroll
# Segmenter (YOLO)
from ultralytics import YOLO


def main():

    start_time = perf_counter()


    # Load YoLo nano (YOLOv8)
    ## It is a detector, meaning no need of two models in the pipeline
    ## But if we need a segmenter first, we can get 'yolov8n-seg.pt'
    model = YOLO('yolov8n.pt') 


    # Detector model params
    input_size = 640 # YOLO does the resize automatically
    batch_size = 128
    optimizer_name = 'SGD' # can also do 'Adam'
    patience = 10
    epochs = 4

    # Fine-tune YOLO on scroll dataset:
    ## YOLO will look at the YAML file where we specify the training and validation set
    ## along with the labels
    ### Best model weights will be stored inside runs/
    model.train(
        task = 'detect',
        data='src/hebrew.yaml',
        epochs=epochs,                
        imgsz=input_size,                 
        batch=batch_size,                  
        patience=patience,                 
        optimizer=optimizer_name,   
        workers=1, 
        save=True 
    )



    ## To Do's:

    ## SEGMENTER TRAINING:

    # Augmentation and merge to X_char_train and y_char_train...

    # Generate training scrolls and validation scrolls from new X_char_train, y_char_train

    # Train a segmenter with the training scrolls

    # Tune the segmenter based on the validation scrolls

    ## PREDICTOR TRAINING:

    # Train a predictor model with X_char_train and y_char_train

    # Tune the predictor using X_char_val and y_char_val

    # Output: Make sure to map each 'English' label with its 'Hebrew' equivalent

    ## FINAL PIPELINE

    # Pass the test scrolls as inputs to the Segmenter and pass the Segmenter's outputs as inputs to the Predictor.


    ## Load test scrolls

    #test_scrolls = get_binarized_scroll_images(image_path=test_scroll_path)

    print(f"Running time for task 01: {round(perf_counter() - start_time,2)} seconds")


if __name__ == "__main__":
    main()
