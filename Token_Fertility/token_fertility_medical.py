#!/usr/bin/env python3
"""
Token fertility (tokenizer only) for medical domain: English, Hindi, Bengali, Tamil, Telugu.
Same logic as token_fertility_all_models.py: corpus-level fertility = sum(tokens)/sum(words).
Dataset: ekacare/MedMCQA-Indic. No GPU.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# MedMCQA-Indic: subset_id -> display name
LANG_CONFIG = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
}


def count_words(text):
    """Same as token_fertility_all_models.py."""
    return len(text.split())


def get_medical_text(row, question_only=False):
    """Extract text from a MedMCQA row: question only, or question + 4 options."""
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


def load_medical_subsets(hf_token, samples, seed):
    """Load MedMCQA-Indic per language; return subsets[lang] = list of rows (same indices across langs)."""
    datasets = {}
    for subset_id, label in LANG_CONFIG.items():
        try:
            ds = load_dataset(
                "ekacare/MedMCQA-Indic",
                subset_id,
                split="test",
                token=hf_token,
            )
            datasets[subset_id] = ds
        except Exception as e:
            print(f"Failed to load {label} ({subset_id}): {e}")
            datasets[subset_id] = None

    if datasets.get("en") is None:
        return {}

    n_total = len(datasets["en"])
    n = min(samples, n_total)
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))[:n].tolist()

    subsets = {}
    for subset_id, ds in datasets.items():
        if ds is None:
            continue
        subsets[subset_id] = [ds[i] for i in indices]
    return subsets


def compute_fertility(tokenizer, rows, question_only=False):
    """
    Same logic as token_fertility_all_models.compute_fertility: corpus-level T/W.
    rows: list of MedMCQA row dicts; text from get_medical_text(row, question_only).
    """
    total_tokens = 0
    total_words = 0
    valid = 0

    for row in rows:
        text = get_medical_text(row, question_only)
        if not text:
            continue

        words = count_words(text)
        if words == 0:
            continue

        tokens = len(tokenizer.encode(text, add_special_tokens=False))
        total_tokens += tokens
        total_words += words
        valid += 1

    if valid == 0 or total_words == 0:
        return {"error": "No valid samples", "fertility": None}

    return {
        "fertility": round(total_tokens / total_words, 4),
        "samples": valid,
        "avg_tokens": round(total_tokens / valid, 2),
        "avg_words": round(total_words / valid, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Medical token fertility (Nemotron 30B): en, hi, bn, ta, te")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="fertility_medical_results.json")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--question-only", action="store_true", help="Use only question text; default is question + 4 options")
    args = parser.parse_args()

    hf_token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    question_only = getattr(args, "question_only", False)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token)

    print("Loading MedMCQA-Indic subsets...")
    subsets = load_medical_subsets(hf_token, args.samples, args.seed)
    if not subsets:
        print("No dataset loaded. Aborting.")
        return

    results = {
        "tokenizer": MODEL_ID,
        "dataset": "ekacare/MedMCQA-Indic",
        "domain": "medical",
        "fertility_definition": "sum(tokens)/sum(words)",
        "languages": list(LANG_CONFIG.keys()),
        "samples_per_lang": args.samples,
        "seed": args.seed,
        "text_unit": "question_only" if question_only else "question_plus_options",
        "results": {},
    }

    for subset_id, label in LANG_CONFIG.items():
        rows = subsets.get(subset_id, [])
        print(f"\n  --- {label} ({subset_id}) ---")

        if not rows:
            results["results"][subset_id] = {"error": "No rows loaded"}
            print("    No rows loaded")
            continue

        res = compute_fertility(tokenizer, rows, question_only)
        results["results"][subset_id] = res

        if res.get("fertility") is not None:
            print(f"    Fertility (T/W): {res['fertility']}  (n={res['samples']})")
        else:
            print(f"    Error: {res.get('error', 'unknown')}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
