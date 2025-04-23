"""
Task 01: DSS dataset
(a) Preprocessing and character segmentation
(b) Character recognition

to execute this script:
first pip install ipython
then enter the following command in terminal:
ipython src/task1/scripts/train_detector.py -- --yaml_file_path='src/hebrew.yaml' --input_size=280 \
--batch_size 128 --optimizer 'SGD' --patience 15 --epochs 200 --workers 1
or
<env_name>/bin/ipython src/task1/scripts/train_detector.py -- --yaml_file_path 'src/hebrew.yaml' --input_size 280 \
--batch_size 128 --optimizer 'SGD' --patience 15 --epochs 200 --workers 1
or
myenv/bin/ipython src/task1/scripts/train_detector.py -- --yaml_file_path 'src/hebrew.yaml' --input_size 280 \
--batch_size 128 --optimizer 'SGD' --patience 15 --epochs 200 --workers 1
"""

from time import perf_counter
from src.task1.utils.preprocessing import *
from sklearn.model_selection import train_test_split
from src.task1.utils.generate import generate_synthetic_scroll
# Segmenter (YOLO)
from ultralytics import YOLO
import argparse

def main(
        
        input_size: int,
        batch_size: int,
        optimizer: str,
        patience: int,
        epochs: int,
        yaml_file_path: str,
        workers: int

):

    start_time = perf_counter()


    # Load YoLo nano (YOLOv8)
    ## It is a detector, meaning no need of two models in the pipeline
    ## But if we need a segmenter first, we can get 'yolov8n-seg.pt'
    model = YOLO('yolov8n.pt') 


    # Fine-tune YOLO on scroll dataset:
    ## YOLO will look at the YAML file where we specify the training and validation set
    ## along with the labels
    ### Best model weights will be stored inside runs/
    model.train(
        task = 'detect',
        data=yaml_file_path,
        epochs=epochs,                
        imgsz=input_size,                 
        batch=batch_size,                  
        patience=patience,                 
        optimizer=optimizer,   
        workers=workers, 
        save=True 
    )



    print(f"Running time for task 01: {round(perf_counter() - start_time,2)} seconds")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a YOLO detector on the DSS dataset.")

    parser.add_argument("--yaml_file_path", type=str, required=True, help="Path to the YAML file specifying dataset details.")
    parser.add_argument("--input_size", type=int, required=True, help="Input image size for YOLO model.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training.")
    parser.add_argument("--optimizer", type=str, required=True, choices=["SGD", "Adam"], help="Optimizer to use for training.")
    parser.add_argument("--patience", type=int, required=True, help="Early stopping patience.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers for data loading.")

    args = parser.parse_args()
    main(
    yaml_file_path=args.yaml_file_path,
    input_size=args.input_size,
    batch_size=args.batch_size,
    optimizer=args.optimizer,
    patience=args.patience,
    epochs=args.epochs,
    workers=args.workers
    )
