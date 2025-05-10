"""
This script predicts characters from a scroll image using a trained YOLO model. 
It also visualizes the results.

.venv/bin/ipython src/task1/scripts/make_prediction.py -- \
--model_path "runs/detect/train3/weights/best.pt" \
--image_path "test/25-Fg001.pbm" \
--yaml_file_path "src/hebrew.yaml" \
--confidence 0.395
"""

import argparse
import os
from PIL import Image
import numpy as np
from ultralytics import YOLO
import yaml


def load_hebrew_names(yaml_file_path):
    with open(yaml_file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        return {int(k): v for k, v in config["hebrew_names"].items()}


def main(model_path, image_path, label_path, yaml_file_path, confidence):
    # Load model and label names
    model = YOLO(model_path)
    hebrew_names = load_hebrew_names(yaml_file_path)

    # Load image manually to avoid format issues (like with .pbm)
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    # Predict with visualization and saving
    results = model.predict(
        source=img_array,
        conf=confidence,
        show=True,  # Show annotated image in a window
        save=True,  # Save annotated image to disk
        project="results",  # Folder to save visual output
        name="prediction_output",  # Subfolder
        exist_ok=True,
    )

    # Extract and print predictions
    if len(results) == 0 or results[0].boxes is None:
        print("No detections found.")
        return

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    preds = [(int(cls), int(box[0])) for box, cls in zip(boxes, classes)]
    preds.sort(key=lambda x: x[1])
    chars = [hebrew_names[cls] for cls, _ in preds]

    print("Predicted text:", "".join(chars))

    # Optional ground truth handling
    if label_path:
        print("Ground truth comparison not implemented yet.")
    else:
        print("No ground truth label path provided. Skipping ground truth decoding.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run YOLO prediction on a single image."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained YOLO model"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to input image file"
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default=None,
        help="(Optional) Ground truth label path",
    )
    parser.add_argument(
        "--yaml_file_path", type=str, required=True, help="Path to Hebrew label YAML"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.40,
        help="Confidence threshold for prediction",
    )
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        image_path=args.image_path,
        label_path=args.label_path,
        yaml_file_path=args.yaml_file_path,
        confidence=args.confidence,
    )
