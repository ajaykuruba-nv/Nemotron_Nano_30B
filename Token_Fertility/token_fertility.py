#!/usr/bin/env python3
"""
Token fertility for Nemotron 3 Nano 30B tokenizer on Hindi, Bengali, Tamil, Telugu.
Corpus-level fertility: sum(tokens) / sum(words) over samples. Low ≈1–1.3 = efficient; high >2 = many splits.
Dataset: ai4bharat/samanantar. No model or GPU: tokenizer only.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

LANG_CONFIG = {
    "hindi": ("hi", "Hindi"),
    "bengali": ("bn", "Bengali"),
    "tamil": ("ta", "Tamil"),
    "telugu": ("te", "Telugu"),
}


def count_words(text: str) -> int:
    """Word count: non-empty tokens from whitespace split."""
    return len([w for w in text.split() if w.strip()])


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def fertility(tokens: int, words: int) -> float:
    """Token fertility = T/W (tokens per word). Returns 0.0 if words <= 0."""
    if words <= 0:
        return 0.0
    return tokens / words


def main():
    parser = argparse.ArgumentParser(description="Token fertility (tokenizer only) for Nemotron 30B on Indic languages")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples per language (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--output", type=str, default="fertility_results.json", help="Output JSON path")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token (or set HF_TOKEN)")
    args = parser.parse_args()

    hf_token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    torch.manual_seed(args.seed)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token)

    results = {}
    lang_names = list(LANG_CONFIG.keys())

    for lang_key, (config_name, lang_label) in LANG_CONFIG.items():
        print(f"\n--- {lang_label} (config: {config_name}) ---")
        try:
            ds = load_dataset("ai4bharat/samanantar", config_name, split="train", token=hf_token)
        except Exception as e:
            print(f"  Failed to load dataset config {config_name}: {e}")
            results[lang_key] = {"error": str(e), "fertility": None}
            continue

        n_total = len(ds)
        n_samples = min(args.samples, n_total)
        indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed))[:n_samples].tolist()
        subset = ds.select(indices)

        token_counts = []
        word_counts = []

        for i, row in enumerate(subset):
            tgt = (row.get("tgt") or row.get("translation", {}).get(config_name) or "").strip()
            if not tgt:
                continue
            w = count_words(tgt)
            if w == 0:
                continue
            t = count_tokens(tokenizer, tgt)
            token_counts.append(t)
            word_counts.append(w)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{n_samples}")

        if not token_counts:
            results[lang_key] = {"error": "No valid samples", "fertility": None}
            continue

        total_tokens = sum(token_counts)
        total_words = sum(word_counts)
        corpus_fertility = total_tokens / total_words
        results[lang_key] = {
            "fertility": round(corpus_fertility, 4),
            "samples": len(token_counts),
            "avg_tokens": round(total_tokens / len(token_counts), 2),
            "avg_words": round(total_words / len(word_counts), 2),
        }
        print(f"  Fertility (T/W): {corpus_fertility:.4f}  avg_tokens={total_tokens/len(token_counts):.1f} avg_words={total_words/len(word_counts):.1f}  (n={len(token_counts)})")

    summary = {
        "tokenizer": MODEL_ID,
        "dataset": "ai4bharat/samanantar",
        "fertility_definition": "T/W (tokens per word), corpus-level: sum(tokens)/sum(words)",
        "languages": lang_names,
        "samples_per_lang": args.samples,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {out_path}")
    return summary


if __name__ == "__main__":
    main()
