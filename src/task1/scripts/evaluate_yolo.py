"""Evaluate YOLO model and extract class probabilities from the last layer.

to execute:
ipython src/task1/scripts/evaluate_yolo.py -- --model_path "runs/detect/train/weights/best.pt" \
--input_image "dataset/segmented_scrolls/val/images/random_scroll_0001_line_00.png"
"""

from ultralytics import YOLO
import torch
import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the YOLO model file."
    )
    parser.add_argument(
        "--input_image", type=str, required=True, help="Path to the input image."
    )
    args = parser.parse_args()

    model = YOLO(args.model_path)

    print(model.model.model[-1].nc)

    extracted_probs = []

    def extract_class_probs(module, input, output):
        for i, pred in enumerate(output):
            if isinstance(pred, torch.Tensor) and pred.ndim >= 2:
                class_logits = pred[..., 5:]
                class_probs = torch.softmax(class_logits, dim=-1)
                print(f"[HOOK] Image {i} class probs shape: {class_probs.shape}")
                extracted_probs.append(class_probs)

    model.model.model[-1].register_forward_hook(extract_class_probs)

    # Run inference
    results = model(args.input_image)

    # After inference
    if extracted_probs:
        print(f"[OUTSIDE] Extracted class probs shape: {extracted_probs[0].shape}")
    else:
        print("[OUTSIDE] No class probabilities extracted.")


if __name__ == "__main__":
    main()
