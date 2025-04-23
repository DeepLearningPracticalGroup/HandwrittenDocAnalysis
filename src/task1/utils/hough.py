import cv2
import matplotlib.pyplot as plt
import numpy as np


def group_lines_by_rho(lines, tolerance=20):
    rhos = [line[0][0] for line in lines if line is not None]
    rhos.sort()

    groups = []
    current_group = []

    for rho in rhos:
        if not current_group:
            current_group.append(rho)
        elif abs(rho - current_group[-1]) <= tolerance:
            current_group.append(rho)
        else:
            groups.append(current_group)
            current_group = [rho]

    if current_group:
        groups.append(current_group)

    return [np.mean(group) for group in groups]


def hough_transform(img_path, theta_tolerance_deg=10, hough_threshold=70):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Image not loaded.")
        return

    # Step 1: Canny edge detection
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    # Step 2: Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_threshold)
    if lines is None:
        print("⚠️ No lines detected")
        return

    # Step 3: Convert image to color
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Step 4: Filter lines by angle (near-horizontal)
    theta_target = np.pi / 2
    tolerance = np.deg2rad(theta_tolerance_deg)
    filtered_lines = [
        line for line in lines if np.abs(line[0][1] - theta_target) < tolerance
    ]

    # Step 5: Group rho values to detect true text lines
    mean_rhos = group_lines_by_rho(filtered_lines, tolerance=20)
    print(f"📏 Estimated {len(mean_rhos)} text lines after grouping.")

    # Step 6: Draw one line per detected text row
    a = np.cos(theta_target)
    b = np.sin(theta_target)
    for rho in mean_rhos:
        x1 = 0
        y1 = int((rho - x1 * a) / b)
        x2 = img.shape[1]
        y2 = int((rho - x2 * a) / b)
        cv2.line(
            img_color, (x1, y1), (x2, y2), (0, 255, 0), 2
        )  # green lines for actual rows

    # Step 7: Show the result
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Detected Text Lines (Grouped Hough)")
    plt.show()
