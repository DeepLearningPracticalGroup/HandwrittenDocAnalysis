"""

myenv/bin/ipython src/task1/scripts/scroll_generation_baseline.py

"""


from src.task1.utils.generate import generate_synthetic_multiline_scroll
import os

# Paths to training data (characters)
root_folder = 'monkbrill/'
char_folders = [os.path.join(root_folder, d) for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]

# Generate scroll lines from characters
generate_synthetic_multiline_scroll(
    output_dir='synthetic_scrolls/',
    char_folders=char_folders,
    canvas_size=(256, 1024),
    num_images=10
)
