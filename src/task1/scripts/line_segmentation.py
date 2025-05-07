"""
.venv/bin/ipython src/task1/scripts/line_segmentation.py
"""

from src.task1.utils.line_segmentation import segment_all_scrolls
from time import perf_counter


def main():
    # Measure the time taken to run the script
    start_time = perf_counter()

    # Segment Lines from both random and 'from text' scrolls
    segment_all_scrolls(
        root_dir="dataset/synthetic_scrolls_random/",
        output_root="dataset/segmented_scrolls",
        N=40,
    )
    segment_all_scrolls(
        root_dir="dataset/synthetic_scrolls_text/",
        output_root="dataset/segmented_scrolls",
        N=40,
    )

    # Print the time taken
    end_time = perf_counter()
    elapsed_time = end_time - start_time
    print(f"Time taken to segment lines: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
