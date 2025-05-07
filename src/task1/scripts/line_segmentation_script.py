from src.task1.utils.line_segmentation import process_all_scrolls

def main():
    root_dir = "synthetic_scrolls"
    output_root="segmented_lines"
    N = 40

    process_all_scrolls(root_dir=root_dir, output_root=output_root, N=N)

if __name__ == "__main__":
    main()