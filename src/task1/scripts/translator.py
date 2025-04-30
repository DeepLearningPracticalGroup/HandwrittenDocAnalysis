"""
 
myenv/bin/python src/task1/scripts/translator.py \
--text "Hello world" \
"""
import argparse
from transformers import MarianMTModel, MarianTokenizer

def main(text):
    model_name = 'Helsinki-NLP/opus-mt-en-he'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    print(tokenizer.decode(translated[0], skip_special_tokens=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate text using MarianMTModel.")
    parser.add_argument("--text", type=str, help="Text to translate")
    args = parser.parse_args()
    main(args.text)
