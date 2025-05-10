# Deep Learning Practical 2024–2025  
**Handwritten Document Analysis — Group Project**

## 📝 Overview

This repository contains our submission for the Deep Learning Practical course. The project is divided into three tasks:

- **Task 1** focuses on Hebrew character segmentation and recognition using images from the Dead Sea Scrolls.
- **Task 2** (not yet implemented) will involve document layout analysis of Dutch manuscripts from the Staten van Overijssel.
- **Task 3** (coming later) will focus on English handwriting recognition using the IAM dataset.

---

## ⚙️ Environment Setup

To set up the environment and install all necessary dependencies, use the provided `Makefile`.

By default, the Makefile uses **Python 3.11** and creates a virtual environment in the `.venv/` directory. It then installs required packages from `requirements.txt`. You can customize the Python version by changing the `PYTHON` variable at the top of the Makefile (e.g., to Python 3.10).

### Basic Installation

Run one of the following:

- `make all` – Installs everything including the **CPU version** of PyTorch and torchvision.
- `make all-gpu` – Installs everything including the **GPU version** of PyTorch (CUDA 11.8).
- `make torch-help` – Prints instructions for choosing between CPU and GPU versions of PyTorch.

### Individual Commands

- `make venv` – Creates the Python virtual environment at `.venv/`.
- `make install` – Installs all required Python packages listed in `requirements.txt`.
- `make torch-cpu` – Installs the CPU version of PyTorch and torchvision.
- `make torch-gpu` – Installs the GPU version of PyTorch and torchvision using CUDA 11.8.
- `make clean` – Removes the virtual environment and Python cache files.

---

### Running the Task 1 Pipeline

Once the environment is ready, you can run the full Task 1 pipeline using:

- `make task_1` – Runs all steps: data cleaning, noise map binarization, scroll generation (random + text), line segmentation, YOLO training, and optional fine-tuning if a pretrained model is available.

You can also:

- `make test_task_1` – Runs a small test with fewer augmentations, fewer scrolls, and only 1 training epoch, to quickly verify that the pipeline works.
- `make reset_task_1` – Deletes all generated folders related to Task 1 (`augmented_chars`, `dataset`, `noise_maps/binarized`, and `monkbrill_clean`) to start fresh.

---

## 📁 More Documentation

Inside `src/taskX`(where X the number of the task), you can find instructions on how to execute the task-specific pipeline, information about the content of the scripts, and other comments such as limitations.

## 🔍 Task 1 Prediction

To perform predictions on scroll images using your trained YOLO model, use the script located at:

`src/task1/scripts/YOLO_recognizer.py`

This script segments each scroll into horizontal lines, applies the YOLO model to detect Hebrew characters, and writes the predicted characters into a `.txt` file for each image.

---

### ▶️ How to Run

Activate your virtual environment and run the following command:

`.venv/bin/ipython src/task1/scripts/YOLO_recognizer.py -- --input_dir "<path_to_input_images>"`

This script segments each scroll into horizontal lines, applies the YOLO model to detect Hebrew characters, and writes the predicted characters into a `.txt` file for each image.

---

### Fixed Arguments

When using the script, make sure `yolov8n_640_ft/weights/best.pt` is located at the repo's root. The default args look like:

1. **`--model_path`**:  
   Path to the trained YOLO `.pt` model file (default is `yolov8n_640_ft/weights/best.pt`).

2. **`--output_dir`**:  
   Directory where output `.txt` files will be saved. The directory will be created if it does not exist. Default is `final_predictions/`.

3. **`--confidence`**:  
   Confidence threshold for YOLO prediction. Default is 0.39.

---

### Output Format

For every input image, the script generates a `.txt` file in the output directory.  
Each line in the image (segmented horizontally) is written as a separate line of predicted characters.

**Example:**

- Input image: `test/25-Fg001.pbm`  
- Output file: `results/25-Fg001_characters.txt`

---

### ⚠️ Potential Bugs and Troubleshooting

- Be carefult. Supported image formats include `.pbm`, `.png`, `.jpg`, and `.jpeg`.
- If YOLO does not predict anything, an error will likely be returned. If this happens, try lowering the confidence threshold.

---
