# Deep Learning Practical 2024–2025  
**Handwritten Document Analysis — Group Project**

## 📝 Overview

This repository contains our submission for the Deep Learning Practical course. The project is divided into three tasks:

- **Task 1** focuses on Hebrew character segmentation and recognition using images from the Dead Sea Scrolls.
- **Task 2** (not yet implemented) will involve document layout analysis of Dutch manuscripts from the Staten van Overijssel.
- **Task 3** (coming later) will focus on English handwriting recognition using the IAM dataset.

---

## ⚙️ Environment Setup

To set up the environment, use the provided Makefile.

By default, the Makefile uses Python 3.11 and creates a virtual environment in the `.venv` directory. It then installs all required packages from `requirements.txt`.

The Makefile also provides commands to install either the CPU or GPU version of PyTorch and torchvision. The GPU installation uses CUDA 11.8. A help command is included to guide users in choosing the appropriate PyTorch version.

If you want to use a different Python version (e.g., Python 3.10), you can change the `PYTHON` variable at the top of the Makefile. This will not significantly affect the project. The translation script used in Task 1 is optional and already processed text files are provided in the `text_files/` folder.

The Makefile also includes a clean command to remove the virtual environment and cache files.

---

## More Documentation

Inside `src/taskX`(where X the number of the task), you can find instructions on how to execute the task-specific pipeline, information about the content of the scripts, and other comments such as limitations.