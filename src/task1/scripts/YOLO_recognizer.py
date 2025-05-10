"""
.venv/bin/ipython src/task1/scripts/YOLO_recognizer.py -- --model_path "runs/detect/train3/weights/best.pt" --input_dir "test" --output_dir "results/"
"""

import os
import logging
from ultralytics import YOLO
import numpy as np
from src.task1.utils.line_segmentation import segment_image_into_lines
import argparse
import yaml


def predict_text_from_image(image_path, model, hebrew_names):
    lines = segment_image_into_lines(image_path, output_dir=None, label_path=None)
    predictions = []

    for line_img in lines:
        line_array = np.array(line_img.convert("RGB"))
        results = model.predict(source=line_array, conf=0.1, verbose=False)

        if len(results) == 0 or results[0].boxes is None:
            predictions.append("")
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        line_pred = [(int(cls), int(box[0])) for box, cls in zip(boxes, classes)]
        line_pred.sort(key=lambda x: x[1])

        chars = [hebrew_names[cls] for cls, _ in line_pred]
        predictions.append("".join(chars))

    return predictions


def main(input_dir: str, model_path: str, output_dir: str):
    """
    Main function to process images and predict text.

    Args:
        input_dir (str): Path to the input directory containing images.
        model_path (str): Path to the trained YOLO model.
        output_dir (str): Path to the output directory for saving predictions.
    """
    try:
        model = YOLO(model_path)
    except Exception as e:
        logging.error(f"Failed to load YOLO model: {e}")
        return

    with open("src/hebrew.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        hebrew_names = {int(k): v for k, v in config["hebrew_names"].items()}

    os.makedirs("results", exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".pbm")):
            logging.info(f"Processing {filename}...")
            image_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            output_txt_path = os.path.join(output_dir, f"{base_name}_characters.txt")

        predictions = predict_text_from_image(image_path, model, hebrew_names)

        with open(output_txt_path, "w", encoding="utf-8") as f:
            for line in predictions:
                f.write(line + "\n")

        print(f"Saved prediction: {output_txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recognize DSS scroll text using a trained YOLO model."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to folder containing test images.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained YOLO model file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the folder where output text files will be saved.",
    )
    args = parser.parse_args()

    main(args.input_dir, args.model_path, args.output_dir)
