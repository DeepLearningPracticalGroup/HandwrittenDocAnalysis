"""
.venv/bin/ipython src/task1/scripts/create_font_chars.py -- --font_path "Habbakuk.ttf" --output_root "font_chars"
"""

import os
import argparse
from PIL import Image, ImageFont, ImageDraw

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate character images from a font.")
parser.add_argument("--font_path", type=str, required=True, help="Path to the font file (e.g., .ttf file).")
parser.add_argument("--output_root", type=str, required=True, help="Directory to save the generated character images.")
args = parser.parse_args()

font_path = args.font_path
output_root = args.output_root

# Load the font and set the font size to 42
font = ImageFont.truetype(font_path, 42)

# Character mapping for each of the 27 tokens
char_map = {'Alef': ')',
            'Ayin': '(',
            'Bet': 'b',
            'Dalet': 'd',
            'Gimel': 'g',
            'He': 'x',
            'Het': 'h',
            'Kaf': 'k',
            'Kaf-final': '\\',
            'Lamed': 'l',
            'Mem': '{',
            'Mem-medial': 'm',
            'Nun-final': '}',
            'Nun-medial': 'n',
            'Pe': 'p',
            'Pe-final': 'v',
            'Qof': 'q',
            'Resh': 'r',
            'Samekh': 's',
            'Shin': '$',
            'Taw': 't',
            'Tet': '+',
            'Tsadi-final': 'j',
            'Tsadi-medial': 'c',
            'Waw': 'w',
            'Yod': 'y',
            'Zayin': 'z'}

# Returns a grayscale image based on specified label of img_size
def create_image(label, img_size):
    """
    Create a blank image with the specified label and size.
    """
    if label not in char_map:
        raise KeyError('Unknown label!')

    # Create blank image and create a draw interface
    img = Image.new('L', img_size, 255)
    draw = ImageDraw.Draw(img)

    # Get size of the font and draw the token in the center of the blank image
    left, top, right, bottom = font.getbbox(char_map[label])
    w, h = right - left, bottom - top
    draw.text(((img_size[0] - w) / 2, (img_size[1] - h) / 2), char_map[label], 0, font)

    return img

# Directory to save characters
os.makedirs(output_root, exist_ok=True)

# Create one image per letter
for label in char_map:
    char_dir = os.path.join(output_root, label)
    os.makedirs(char_dir, exist_ok=True)

    img = create_image(label, (50, 50))
    save_path = os.path.join(char_dir, f"{label}_001.png")
    img.save(save_path)

print("All character images saved.")