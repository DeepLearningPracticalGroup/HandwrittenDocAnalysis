"""
.venv/bin/ipython src/task1/scripts/data_cleaning.py
"""

from src.task1.utils.preprocessing import clean_character_dataset

def main():
    clean_character_dataset("monkbrill", "monkbrill_clean")

if __name__ == "__main__":
    main()