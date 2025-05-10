"""
Task 01: DSS dataset
(a) Preprocessing and character segmentation
(b) Character recognition

This script trains a YOLOv8n detector on randomly generated scrolls and fine-tunes it on text scrolls (eg. Bible scrolls).

to execute this script:
first pip install ipython
then enter the following command in terminal (adjust the virtual environment name as needed):

.venv/bin/ipython src/task1/scripts/train_detector.py -- --input_size 640 \
--batch_size 16 --optimizer 'Adam' --patience 15 --epochs 1 --workers 1
"""

from time import perf_counter
from src.task1.utils.preprocessing import *
from ultralytics import YOLO
import argparse


def main(
    input_size: int,
    batch_size: int,
    optimizer: str,
    patience: int,
    epochs: int,
    workers: int,
    pretrained_model_path: str = None,
):
    start_time = perf_counter()

    if pretrained_model_path:
        print(f"[INFO] Fine-tuning from pretrained model: {pretrained_model_path}")
        model = YOLO(pretrained_model_path)

        model.train(
            task="detect",
            data="src/hebrew_ft.yaml",
            epochs=epochs,
            imgsz=input_size,
            batch=batch_size,
            patience=patience,
            optimizer=optimizer,
            workers=workers,
            save=True,
        )
    else:
        print("[INFO] Training from scratch on random scrolls")
        model = YOLO("yolov8s.pt")

        model.train(
            task="detect",
            data="src/hebrew.yaml",
            epochs=epochs,
            imgsz=input_size,
            batch=batch_size,
            patience=patience,
            optimizer=optimizer,
            workers=workers,
            save=True,
        )

    print(f"Training completed in {round(perf_counter() - start_time, 2)} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a YOLO detector on the DSS dataset."
    )

    parser.add_argument(
        "--input_size", type=int, required=True, help="Input image size for YOLO model."
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="Batch size for training."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        required=True,
        choices=["SGD", "Adam"],
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--patience", type=int, required=True, help="Early stopping patience."
    )
    parser.add_argument(
        "--epochs", type=int, required=True, help="Number of training epochs."
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers for data loading."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Path to a pretrained model to fine-tune)",
    )

    args = parser.parse_args()
    main(
        input_size=args.input_size,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        patience=args.patience,
        epochs=args.epochs,
        workers=args.workers,
        pretrained_model_path=args.pretrained_model_path,
    )
