#!/usr/bin/env python3
"""
Token fertility (tokenizer only) for medical domain: English, Hindi, Bengali, Tamil, Telugu.
Corpus-level fertility: sum(tokens) / sum(words) over samples. Low ≈1–1.3 = efficient; high >2 = many splits.
Dataset: ekacare/MedMCQA-Indic. No model or GPU.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# MedMCQA-Indic subset names and display names
LANG_CONFIG: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
}


def _sample_text(row: Dict[str, Any], question_only: bool = False) -> str:
    """Build one string: question only, or question + 4 options (default)."""
    if question_only:
        return (row.get("question") or "").strip()
    parts = [
        row.get("question") or "",
        row.get("opa") or "",
        row.get("opb") or "",
        row.get("opc") or "",
        row.get("opd") or "",
    ]
    return " ".join(p for p in parts if isinstance(p, str) and p.strip())


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
    parser = argparse.ArgumentParser(
        description="Medical token fertility (tokenizer only) for Nemotron 30B: en, hi, bn, ta, te"
    )
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="fertility_medical_results.json", help="Output JSON path")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token (or set HF_TOKEN)")
    parser.add_argument("--question-only", action="store_true", help="Count only question text (lower avg tokens); default is question + 4 options")
    args = parser.parse_args()

    hf_token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    question_only = getattr(args, "question_only", False)
    torch.manual_seed(args.seed)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token)

    # Load MedMCQA-Indic for each language (parallel by row index)
    datasets: Dict[str, Any] = {}
    for subset_id, label in LANG_CONFIG.items():
        print(f"Loading MedMCQA-Indic subset: {subset_id} ({label})...")
        try:
            ds = load_dataset(
                "ekacare/MedMCQA-Indic",
                subset_id,
                split="test",
                trust_remote_code=True,
                token=hf_token,
            )
            datasets[subset_id] = ds
        except Exception as e:
            print(f"  Failed: {e}")
            datasets[subset_id] = None

    if datasets.get("en") is None:
        print("Error: English subset required. Aborting.")
        return

    en_ds = datasets["en"]
    n_total = len(en_ds)
    n_samples = min(args.samples, n_total)
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed))[:n_samples].tolist()

    results: Dict[str, Any] = {
        "tokenizer": MODEL_ID,
        "dataset": "ekacare/MedMCQA-Indic",
        "domain": "medical",
        "fertility_definition": "T/W (tokens per word), corpus-level: sum(tokens)/sum(words)",
        "languages": list(LANG_CONFIG.keys()),
        "samples": n_samples,
        "text_unit": "question_only" if question_only else "question_plus_options",
        "results": {},
    }

    for subset_id, label in LANG_CONFIG.items():
        ds = datasets.get(subset_id)
        if ds is None:
            results["results"][subset_id] = {"error": "dataset load failed", "fertility": None}
            continue

        token_counts = []
        word_counts = []
        for idx in indices:
            row = ds[idx]
            text = _sample_text(row, question_only=question_only)
            w = count_words(text)
            if w == 0:
                continue
            t = count_tokens(tokenizer, text)
            token_counts.append(t)
            word_counts.append(w)

        if not token_counts:
            results["results"][subset_id] = {"error": "No valid samples", "fertility": None}
            continue

        total_tokens = sum(token_counts)
        total_words = sum(word_counts)
        corpus_fertility = total_tokens / total_words
        results["results"][subset_id] = {
            "fertility": round(corpus_fertility, 4),
            "avg_tokens": round(total_tokens / len(token_counts), 2),
            "avg_words": round(total_words / len(word_counts), 2),
            "samples": len(token_counts),
        }
        print(f"\n--- {label} --- Fertility (T/W): {corpus_fertility:.4f}  avg_tokens={total_tokens/len(token_counts):.1f} avg_words={total_words/len(word_counts):.1f}  (n={len(token_counts)})")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")
    return results


if __name__ == "__main__":
    main()
