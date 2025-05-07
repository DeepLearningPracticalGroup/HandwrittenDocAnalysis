from ultralytics import YOLO
import torch

def main():
    model_path = "runs/detect/train6/weights/best.pt"
    model = YOLO(model_path)

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
    results = model('segmented_lines/val/images/scroll_0000_line_00.png')

    # After inference
    if extracted_probs:
        print(f"[OUTSIDE] Extracted class probs shape: {extracted_probs[0].shape}")
    else:
        print("[OUTSIDE] No class probabilities extracted.")

if __name__ == "__main__":
    main()