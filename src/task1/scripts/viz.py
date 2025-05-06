import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2

def visualize_scroll_with_boxes(image_path: str, label_path: str, box_color="red", box_thickness=1):
    """
    Visualizza un'immagine scroll con i bounding box YOLO disegnati sopra.
    
    Args:
        image_path: path all'immagine (es. .png)
        label_path: path al file di annotazioni YOLO (.txt)
        box_color: colore dei bounding box (es. "red", "green")
        box_thickness: spessore della linea
    """
    # Carica immagine
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    H, W = img.shape

    # Crea figura
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.imshow(img, cmap="gray")
    ax.axis("off")

    # Carica le label
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, x_center, y_center, width, height = map(float, parts)
                
                # Converti in coordinate assolute
                x = (x_center - width / 2) * W
                y = (y_center - height / 2) * H
                w = width * W
                h = height * H
                
                # Disegna rettangolo
                rect = patches.Rectangle((x, y), w, h, linewidth=box_thickness,
                                         edgecolor=box_color, facecolor="none")
                ax.add_patch(rect)

    plt.title(f"Bounding Boxes: {os.path.basename(image_path)}")
    plt.show()

def main():
    # Esempio di utilizzo
    image_path = "synthetic_scrolls/train/images/scroll_0006.png"
    label_path = "synthetic_scrolls/train/labels/scroll_0006.txt"
    visualize_scroll_with_boxes(image_path, label_path)

if __name__ == "__main__":
    main()