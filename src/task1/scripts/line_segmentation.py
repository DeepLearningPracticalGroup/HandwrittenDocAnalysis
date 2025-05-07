"""
.venv/bin/ipython src/task1/scripts/line_segmentation.py
"""

from src.task1.utils.line_segmentation import segment_all_scrolls


def main():
    root_dir = "generated_scrolls/"
    output_root = "segmented_scrolls/"

    # Segment Lines from scrolls
    segment_all_scrolls(root_dir=root_dir, output_root=output_root, N=40)


if __name__ == "__main__":
    main()
