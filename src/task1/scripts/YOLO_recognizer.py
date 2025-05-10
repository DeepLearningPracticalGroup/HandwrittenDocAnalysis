"""
.venv/bin/ipython src/task1/scripts/YOLO_recognizer.py -- --input test
"""

import os
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
        results = model.predict(source=line_array, conf=0.40, verbose=False)

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


def main(input_dir: str):

    model_path = "runs/detect/yolov8n_640_ft/weights/best.pt"

    model = YOLO(model_path)

    with open("src/hebrew.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        hebrew_names = {int(k): v for k, v in config["hebrew_names"].items()}

    os.makedirs("results", exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".pbm")):
            print(f"Processing {filename}...")
            continue
        image_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        output_txt_path = os.path.join("results", f"{base_name}_characters.txt")

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
        "--input",
        type=str,
        required=True,
        help="Path to folder containing test images.",
    )
    args = parser.parse_args()

    main(args.input)
