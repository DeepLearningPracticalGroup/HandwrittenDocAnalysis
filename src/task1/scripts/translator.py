"""
 
.venv/bin/ipython src/task1/scripts/translator.py -- --input text_files/english_text/aaesops_fables.txt --output text_files/hebrew_text/aaesops_fables.txt

"""
import argparse
from transformers import MarianMTModel, MarianTokenizer
import os
from time import perf_counter

def translate_text(text: str) -> str:
    model_name = "Helsinki-NLP/opus-mt-en-he"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    translated_tokens = model.generate(**inputs)
    hebrew_translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    
    return hebrew_translation

def main(input_path: str, output_path: str):
    # Measure the start time
    start_time = perf_counter()
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The input file '{input_path}' does not exist. Please provide a valid file path.")
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            english_text = line.strip()
            if english_text:  # Skip empty lines
                hebrew_text = translate_text(english_text)
                outfile.write(hebrew_text + "\n")
    
    print(f"Translation complete. Hebrew text saved to: {output_path}")
    # Measure the end time
    print(f"Time taken: {perf_counter() - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate a .txt file from English to Hebrew.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input .txt file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output .txt file.")
    args = parser.parse_args()

    main(args.input, args.output)

