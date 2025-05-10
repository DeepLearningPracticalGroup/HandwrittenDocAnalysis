"""
This script predicts characters from a scroll image using a trained YOLO model. 
It also visualizes the results.

.venv/bin/ipython src/task1/scripts/make_prediction.py -- \
--model_path "runs/detect/train3/weights/best.pt" \
--image_path "image-data/P22-Fg008-R-C01-R01-binarized.jpg" \
--yaml_file_path "src/hebrew.yaml" \
--confidence 0.395
"""

import argparse
from ultralytics import YOLO
import cv2
import yaml


def visualize_and_read(results, label_names):
    # Visualize results
    for r in results:
        im = r.orig_img.copy()
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()

        # Build output list
        predictions = []
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            label = label_names[int(cls)]
            predictions.append((x1, y1, label, conf))
            cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                im,
                f"{label} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Sort predictions top-to-bottom (based on y1) and then left-to-right (based on x1)
        predictions.sort(key=lambda tup: (tup[1], tup[0]))

        # Group predictions into rows based on y1 proximity
        row_threshold = 25
        rows = []
        current_row = [predictions[0]]

        for i in range(1, len(predictions)):
            if abs(predictions[i][1] - current_row[-1][1]) < row_threshold:
                current_row.append(predictions[i])
            else:
                rows.append(current_row)
                current_row = [predictions[i]]
        rows.append(current_row)

        # Construct the final text with spaces between characters
        final_text = []
        for row in rows:
            row.sort(key=lambda x: x[0])  # Sort left-to-right
            final_text.append(" ".join([char[2] for char in row]))

        # Print the final predicted text
        print("\nPredicted Text:")
        for line in final_text:
            print(english_to_hebrew(line, "src/hebrew.yaml"))

        # Show and save the visualization
        output_path = "output_scroll_prediction.png"
        cv2.imwrite(output_path, im)
        print(f"Saved annotated image to {output_path}")


def english_to_hebrew(english_line, yaml_file_path):
    # Load the YAML file
    with open(yaml_file_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    converted_map = {v: k for k, v in yaml_data["converted"].items()}

    # Split the English line into character names
    hebrew_sequence = []
    buffer = ""
    for char in english_line.split():  # Split by spaces to handle individual words
        buffer += char
        if buffer in converted_map:
            hebrew_sequence.append(converted_map[buffer])
            buffer = ""

    # Return the Hebrew sequence as a string
    return "".join(hebrew_sequence)


def decode_labels(label_path, yaml_file_path):
    # Read the label file
    with open(label_path, "r") as f:
        label_lines = f.readlines()

    # Read the YAML file to decode character numbers
    with open(yaml_file_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    character_map = yaml_data["hebrew_names"]

    # Decode the labels
    decoded_labels = []
    for line in label_lines:
        parts = line.strip().split()
        char_number = int(parts[0])
        x_center, y_center, height, width = map(float, parts[1:])
        character = character_map[char_number]
        decoded_labels.append((character, x_center, y_center, height, width))
    # Group characters into rows based on their y_center
    row_threshold = 0.09  # Adjust this threshold as needed
    decoded_labels.sort(key=lambda x: x[2])  # Sort by y_center
    rows = []
    current_row = [decoded_labels[0]]

    for i in range(1, len(decoded_labels)):
        if abs(decoded_labels[i][2] - current_row[-1][2]) < row_threshold:
            current_row.append(decoded_labels[i])
        else:
            rows.append(current_row)
            current_row = [decoded_labels[i]]
    rows.append(current_row)

    # Sort characters within each row by x_center and construct text
    text_lines = []
    for row in rows:
        row.sort(key=lambda x: x[1])  # Sort by x_center
        text_lines.append(" ".join([char[0] for char in row]))

    # Print the reconstructed text
    print("\nGround Truth:")
    for line in text_lines:
        print(line)

    return decoded_labels


def main(
    model_path: str, image_path: str, label_path: str, yaml_file_path, confidence: float
):
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    print(f"Predicting on image: {image_path}")
    print(f"Confidence threshold: {confidence}")

    if label_path:
        decode_labels(label_path, yaml_file_path)
    else:
        print("No ground truth label path provided. Skipping ground truth decoding.")

    results = model.predict(source=image_path, conf=confidence, save=False)

    # Get label names from model
    label_names = model.names  # dict: {0: 'A', 1: 'B', ...}
    visualize_and_read(results, label_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict characters from a scroll image using a trained YOLO model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained YOLO model (e.g., best.pt)",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to scroll image for prediction.",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        required=False,
        help="Path to the label file for the image. If not provided, ground truth will be skipped.",
    )
    parser.add_argument(
        "--yaml_file_path",
        type=str,
        required=True,
        help="Path to the YAML file specifying dataset details.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.40,
        help="Confidence threshold for predictions.",
    )

    args = parser.parse_args()
    main(
        model_path=args.model_path,
        image_path=args.image_path,
        label_path=args.label_path,
        yaml_file_path=args.yaml_file_path,
        confidence=args.confidence,
    )
