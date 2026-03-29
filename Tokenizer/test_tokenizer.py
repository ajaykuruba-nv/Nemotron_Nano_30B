#!/usr/bin/env python3
"""
Test script to verify the expanded Nemotron tokenizer against the base tokenizer.
It compares token counts and displays the actual subwords to ensure the BPE is working efficiently.

Usage:
  python test_tokenizer.py --new ./nemotron-indic-expanded --base nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
"""

import argparse
from transformers import AutoTokenizer

# Sample sentences across the trained languages
SAMPLES = {
    "English": "Artificial intelligence is rapidly transforming the modern world.",
    "Hindi": "कृत्रिम बुद्धिमत्ता तेजी से आधुनिक दुनिया को बदल रही है।",
    "Bengali": "কৃত্রিম বুদ্ধিমত্তা দ্রুত আধুনিক বিশ্বকে বদলে দিচ্ছে।",
    "Tamil": "செயற்கை நுண்ணறிவு நவீன உலகத்தை வேகமாக மாற்றி வருகிறது.",
    "Telugu": "కృత్రిమ మేధస్సు ఆధునిక ప్రపంచాన్ని వేగంగా మారుస్తోంది."
}

def analyze_text(base_tok, new_tok, language, text):
    print(f"\n{'='*60}")
    print(f"🌍 Language: {language}")
    print(f"📝 Original: {text}")
    print(f"{'='*60}")

    # --- BASE TOKENIZER ---
    base_ids = base_tok.encode(text, add_special_tokens=False)
    base_tokens = base_tok.convert_ids_to_tokens(base_ids)
    
    # --- NEW TOKENIZER ---
    new_ids = new_tok.encode(text, add_special_tokens=False)
    new_tokens = new_tok.convert_ids_to_tokens(new_ids)
    decoded_text = new_tok.decode(new_ids)

    # --- RESULTS ---
    print("\n[ BASE TOKENIZER (Original Nemotron) ]")
    print(f"Count:   {len(base_ids)} tokens")
    # We only show the first 15 base tokens so it doesn't flood the terminal with raw bytes
    print(f"Subwords (first 15): {base_tokens[:15]} ...") 

    print("\n[ NEW TOKENIZER (Expanded Indic) ]")
    print(f"Count:   {len(new_ids)} tokens")
    print(f"Subwords: {new_tokens}")
    
    # Verification check
    print("\n[ VERIFICATION ]")
    if decoded_text == text:
        print("✅ Decode matches original text perfectly (No data loss).")
    else:
        print("❌ Decode mismatch!")
        print(f"Got: {decoded_text}")

    # Calculate Efficiency
    if len(base_ids) > 0:
        saved = len(base_ids) - len(new_ids)
        improvement = (saved / len(base_ids)) * 100
        print(f"\n🚀 EFFICIENCY GAIN: {improvement:.1f}% fewer tokens used! ({saved} tokens saved)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", default="./nemotron-indic-expanded", help="Path to your newly trained tokenizer")
    parser.add_argument("--base", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", help="Original model ID")
    args = parser.parse_args()

    print(f"Loading base tokenizer from: {args.base}...")
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    except Exception as e:
        print(f"Failed to load base tokenizer: {e}")
        return

    print(f"Loading new tokenizer from:  {args.new}...")
    try:
        new_tokenizer = AutoTokenizer.from_pretrained(args.new, use_fast=True)
    except Exception as e:
        print(f"Failed to load new tokenizer. Did the training script finish successfully? Error: {e}")
        return

    print(f"\nBase Vocab Size: {len(base_tokenizer):,}")
    print(f"New Vocab Size:  {len(new_tokenizer):,}")

    for lang, text in SAMPLES.items():
        analyze_text(base_tokenizer, new_tokenizer, lang, text)

if __name__ == "__main__":
    main()