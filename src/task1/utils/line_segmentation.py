

import os
import numpy as np
from PIL import Image, ImageDraw, ImageChops
import matplotlib.pyplot as plt
from math import sin, cos, radians
import cv2
from pathlib import Path



def find_minima_and_draw_all_lines(image_path, N=80):
    """Find minima and draw red lines at minima and blue lines between them."""
    # Load the original image
    img = Image.open(image_path).convert("L")  # 'L' = 8-bit pixels, black and white
    img_array = np.array(img)
    
    # Sum pixel values across each row
    row_sums = img_array.sum(axis=1)  # Sum along width (axis 1)
    
    # Normalize row sums to [0, 1]
    row_sums_normalized = (row_sums - row_sums.min()) / (np.ptp(row_sums) + 1e-8)  
    # Find minima based on the surrounding N rows
    minima = []
    for i in range(N, len(row_sums_normalized) - N):
        # Check if current point is a local minima
        is_minima = all(row_sums_normalized[i] < row_sums_normalized[i-j] for j in range(1, N+1)) and \
                    all(row_sums_normalized[i] < row_sums_normalized[i+j] for j in range(1, N+1))
        if is_minima:
            minima.append(i)
    
    # Convert to RGB mode before drawing colored lines
    img_with_lines = img.convert("RGB")
    draw = ImageDraw.Draw(img_with_lines)
    
    # Calculate midpoints between minima
    midpoints = []
    if len(minima) >= 2:
        for i in range(len(minima) - 1):
            # Calculate the midpoint between two minima
            midpoint = (minima[i] + minima[i+1]) // 2
            midpoints.append(midpoint)
            # Draw a blue line at the midpoint
            draw.line([(0, midpoint), (img.width, midpoint)], fill=(255, 0, 0), width=4)  # Blue in RGB
    
    return img_with_lines, minima, midpoints

def count_intersections(img_array, x1, y1, x2, y2):
    """Count black pixels that a line from (x1,y1) to (x2,y2) intersects with."""
    # Create a temporary image with just the line
    temp_img = Image.new('L', (img_array.shape[1], img_array.shape[0]), 255)
    draw = ImageDraw.Draw(temp_img)
    draw.line([(x1, y1), (x2, y2)], fill=0, width=1)
    
    # Convert to numpy array
    line_array = np.array(temp_img)
    
    # Where the line is drawn (0) and the original image is black (0), that's an intersection
    # In our binary image, black pixels are 0 and white pixels are 255
    intersection = np.logical_and(line_array == 0, img_array == 0)
    
    # Count intersections
    return np.sum(intersection)

def rotate_line(center_x, center_y, x, y, angle_deg):
    """Rotate a point (x,y) around center (center_x,center_y) by angle_deg degrees."""
    # Convert angle to radians
    angle_rad = radians(angle_deg)
    
    # Translate point to origin
    x_translated = x - center_x
    y_translated = y - center_y
    
    # Rotate point
    x_rotated = x_translated * cos(angle_rad) - y_translated * sin(angle_rad)
    y_rotated = x_translated * sin(angle_rad) + y_translated * cos(angle_rad)
    
    # Translate point back
    x_final = x_rotated + center_x
    y_final = y_rotated + center_y
    
    return int(x_final), int(y_final)

def optimize_segmentation_lines(image_path, midpoints, angle_range=(-10, 10), angle_step=0.5):
    """
    Optimize segmentation lines by rotating them to minimize black pixel intersections.
    """
    # Load image and convert to binary array (0=black, 255=white)
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    img_width, img_height = img.size
    
    # Center point of the image
    center_x = img_width // 2
    center_y = img_height // 2
    
    # For each midpoint, find the optimal rotation angle
    optimized_angles = []
    min_intersections = []
    line_endpoints = []
    
    for midpoint in midpoints:
        # Original horizontal line endpoints
        left_point = (0, midpoint)
        right_point = (img_width, midpoint)
        
        # Test different angles
        best_angle = 0
        min_intersection = float('inf')
        best_endpoints = (left_point, right_point)
        intersection_counts = []
        angle_values = []
        
        for angle in np.arange(angle_range[0], angle_range[1] + angle_step, angle_step):
            # Rotate line endpoints around center of image
            new_left_x, new_left_y = rotate_line(center_x, center_y, left_point[0], left_point[1], angle)
            new_right_x, new_right_y = rotate_line(center_x, center_y, right_point[0], right_point[1], angle)
            
            # Keep points within image boundaries
            new_left_x = max(0, min(img_width-1, new_left_x))
            new_left_y = max(0, min(img_height-1, new_left_y))
            new_right_x = max(0, min(img_width-1, new_right_x))
            new_right_y = max(0, min(img_height-1, new_right_y))
            
            # Count black pixel intersections
            intersections = count_intersections(img_array, new_left_x, new_left_y, new_right_x, new_right_y)
            intersection_counts.append(intersections)
            angle_values.append(angle)
            
            if intersections < min_intersection:
                min_intersection = intersections
                best_angle = angle
                best_endpoints = ((new_left_x, new_left_y), (new_right_x, new_right_y))
        
        optimized_angles.append(best_angle)
        min_intersections.append(min_intersection)
        line_endpoints.append(best_endpoints)
    
    # Create image with optimized lines
    optimized_img = img.convert("RGB")
    draw = ImageDraw.Draw(optimized_img)
    
    # Draw original horizontal lines for comparison
    for midpoint in midpoints:
        draw.line([(0, midpoint), (img_width, midpoint)], fill=(255, 0, 0), width=2)  # Red
    
    # Draw optimized rotated lines
    for idx, ((x1, y1), (x2, y2)) in enumerate(line_endpoints):
        draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 0), width=3)  # Green
    
    return optimized_angles, optimized_img, line_endpoints

def extract_line_segments_with_masks(image_path, line_endpoints, padding=10):
    """
    Extract text line segments using polygon masks between consecutive lines.
    Each segment is created by masking out everything except the current segment,
    and then cropping to include only rows with black pixels.
    """
    # Load the original image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)
    height, width = img_array.shape
    
    # Sort lines by y-position (top to bottom)
    sorted_lines = sorted(line_endpoints, key=lambda x: (x[0][1] + x[1][1]) / 2)
    
    # Add top and bottom image boundaries
    top_line = ((0, 0), (width-1, 0))
    bottom_line = ((0, height-1), (width-1, height-1))
    all_lines = [top_line] + sorted_lines + [bottom_line]
    
    # Extract segments between consecutive lines
    masked_segments = []
    
    for i in range(len(all_lines) - 1):
        # Get consecutive lines
        upper_line = all_lines[i]
        lower_line = all_lines[i+1]
        
        # Unpack line coordinates
        (ul_x1, ul_y1), (ul_x2, ul_y2) = upper_line
        (ll_x1, ll_y1), (ll_x2, ll_y2) = lower_line
        
        # Add padding to upper and lower lines
        ul_y1_padded = max(0, ul_y1 + padding)  # Add padding below upper line
        ul_y2_padded = max(0, ul_y2 + padding)
        ll_y1_padded = min(height-1, ll_y1 - padding)  # Subtract padding above lower line
        ll_y2_padded = min(height-1, ll_y2 - padding)
        
        # Create mask for the entire image (white background)
        mask = Image.new('L', (width, height), 255)  # White background (255)
        draw = ImageDraw.Draw(mask)
        
        # Draw a black polygon for the area between the lines (this is the area we want to keep)
        polygon = [
            (ul_x1, ul_y1_padded), 
            (ul_x2, ul_y2_padded),
            (ll_x2, ll_y2_padded), 
            (ll_x1, ll_y1_padded)
        ]
        draw.polygon(polygon, fill=0)  # Black polygon (0)
        
        # Apply mask to original image - keep original pixels where mask is black (0), make white elsewhere
        mask_array = np.array(mask)
        masked_img_array = np.copy(img_array)
        masked_img_array[mask_array == 255] = 255  # Set everything outside mask to white
        
        # Convert back to PIL Image
        masked_img = Image.fromarray(masked_img_array)
        
        # Find the first and last rows with black pixels (0) for vertical cropping
        rows_with_black = np.where(np.any(masked_img_array < 255, axis=1))[0]
        
        if len(rows_with_black) > 0:
            first_black_row = rows_with_black[0]
            last_black_row = rows_with_black[-1]
            
            # Add some padding to the crop
            first_black_row = max(0, first_black_row - padding)
            last_black_row = min(height-1, last_black_row + padding)
            
            # Crop the image vertically to include only rows with black pixels
            cropped_segment = masked_img.crop((0, first_black_row, width, last_black_row + 1))
            
            masked_segments.append(cropped_segment)
    
    return masked_segments

def save_line_segments(segments, output_dir, base_filename):
    """Save the extracted line segments to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    for i, segment in enumerate(segments):
        # Create filename
        filename = f"{base_filename}_line_{i+1:02d}.png"
        output_path = os.path.join(output_dir, filename)
        
        # Save the segment
        segment.save(output_path)
        saved_paths.append(output_path)
    
    return saved_paths


def convert_labels_to_segments(original_label_path, line_start, line_end, image_height):
    """
    Filter and adjust character boxes from full scroll label to this segment.
    """
    new_boxes = []
    if not os.path.exists(original_label_path):
        return new_boxes  # if no labels, skip

    with open(original_label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x_center, y_center, width, height = map(float, parts)
            abs_y = y_center * image_height

            if line_start <= abs_y < line_end:
                # Adjust y relative to the segment and re-normalize
                new_y = (abs_y - line_start) / (line_end - line_start)
                new_h = height * image_height / (line_end - line_start)
                new_boxes.append(f"{int(cls)} {x_center:.6f} {new_y:.6f} {width:.6f} {new_h:.6f}")

    return new_boxes

def segment_scrolls(
    input_img_dir,
    input_label_dir,
    output_img_dir,
    output_label_dir,
    N=80,
    angle_range=(-10, 10),
    angle_step=0.5,
    padding=20
):
    """
    Segment each scroll image into lines and generate per-line YOLO labels.

    Args:
        input_img_dir (str): Directory with scroll images.
        input_label_dir (str): Directory with full-scroll YOLO labels.
        output_img_dir (str): Directory for output line images.
        output_label_dir (str): Directory for output line label files.
        N (int): Minima detection window size.
        angle_range (tuple): Angle search range for line rotation.
        angle_step (float): Step size for angle optimization.
        padding (int): Vertical padding for each line segment.
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for fname in sorted(os.listdir(input_img_dir)):
        if not fname.endswith('.png'):
            continue

        name_base = os.path.splitext(fname)[0]
        image_path = os.path.join(input_img_dir, fname)
        label_path = os.path.join(input_label_dir, name_base + '.txt')

        # Find and optimize segmentation lines
        _, minima, midpoints = find_minima_and_draw_all_lines(image_path, N=N)
        _, _, line_endpoints = optimize_segmentation_lines(
            image_path,
            midpoints,
            angle_range=angle_range,
            angle_step=angle_step
        )
        segments = extract_line_segments_with_masks(
            image_path,
            line_endpoints,
            padding=padding
        )

        full_img = Image.open(image_path).convert('L')
        image_height = full_img.height

        # Compute vertical bounds for each line
        sorted_lines = sorted(line_endpoints, key=lambda l: (l[0][1] + l[1][1]) / 2)
        all_lines = [((0, 0), (full_img.width - 1, 0))] + sorted_lines + [((0, image_height - 1), (full_img.width - 1, image_height - 1))]
        line_limits = [
            (int((all_lines[i][0][1] + all_lines[i][1][1]) // 2),
             int((all_lines[i+1][0][1] + all_lines[i+1][1][1]) // 2))
            for i in range(len(all_lines) - 1)
        ]

        # Save each line segment and label
        for i, (segment_img, (top_y, bot_y)) in enumerate(zip(segments, line_limits)):
            segment_fname = f"{name_base}_line{i:03d}"
            segment_img_path = os.path.join(output_img_dir, segment_fname + '.png')
            segment_lbl_path = os.path.join(output_label_dir, segment_fname + '.txt')

            # Ensure parent folders exist (useful if custom output structure is added later)
            Path(segment_img_path).parent.mkdir(parents=True, exist_ok=True)
            Path(segment_lbl_path).parent.mkdir(parents=True, exist_ok=True)

            segment_img.save(segment_img_path)

            segment_labels = convert_labels_to_segments(label_path, top_y, bot_y, image_height)
            with open(segment_lbl_path, 'w') as f:
                f.write('\n'.join(segment_labels))