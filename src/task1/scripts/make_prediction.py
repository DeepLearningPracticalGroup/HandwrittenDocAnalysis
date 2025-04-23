"""
myenv/bin/ipython src/task1/scripts/make_prediction.py -- \
--model_path "runs/detect/train4/weights/best.pt" \
--image_path "image-data/P123-Fg002-R-C01-R01-binarized.jpg"

"""


import argparse
from ultralytics import YOLO
import cv2
import torch

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
            predictions.append((x1, label, conf))
            cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(im, f"{label} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Sort predictions left-to-right (based on x1)
        predictions.sort(key=lambda tup: tup[0])
        sequence = "".join([p[1] for p in predictions])
        print(f"\n🧾 Detected Character Sequence:\n{sequence}")

        # Show and save the visualization
        output_path = "output_scroll_prediction.png"
        cv2.imwrite(output_path, im)
        print(f"📸 Saved annotated image to {output_path}")

def main(model_path, image_path):
    print(f"🔍 Loading model from: {model_path}")
    model = YOLO(model_path)

    print(f"Predicting on image: {image_path}")
    results = model.predict(source=image_path, conf=0.25, save=False)

    # Get label names from model
    label_names = model.names  # dict: {0: 'A', 1: 'B', ...}
    
    visualize_and_read(results, label_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict characters from a scroll image using a trained YOLO model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained YOLO model (e.g., best.pt)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to scroll image for prediction")

    args = parser.parse_args()
    main(args.model_path, args.image_path)
