import numpy as np


def IoU(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """
    Measure the Intersection over Union between two bounding boxes
    Bounding boxes should be represented as [x1, y1, x2, y2]
    IoU should be a float in [0,1]
    """
    xA = np.maximum(boxA[0], boxB[0])
    yA = np.maximum(boxA[1], boxB[1])
    xB = np.minimum(boxA[2], boxB[2])
    yB = np.minimum(boxA[3], boxB[3])

    inter_width = np.maximum(0, xB - xA)
    inter_height = np.maximum(0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union_area = boxA_area + boxB_area - inter_area

    # Avoid dividing by zero
    if union_area == 0:
        return 0.0

    return inter_area / union_area
