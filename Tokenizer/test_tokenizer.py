#!/usr/bin/env python3
"""
Test script: compare three Nemotron tokenizers (base vs indic-expanded vs continued BPE).

Usage:
  python test_tokenizer.py
  python test_tokenizer.py --base nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \\
      --indic-expanded /path/to/nemotron-indic-expanded \\
      --continued-bpe /path/to/continued_bpe_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import os

from transformers import AutoTokenizer

DEFAULT_BASE = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_INDIC_EXPANDED = "/home/admin/nvidia/Nemotron_Nano_30B/Tokenizer/nemotron-indic-expanded"
DEFAULT_CONTINUED_BPE = "/home/admin/nvidia/Nemotron_Nano_30B/Tokenizer/outputs/continued_bpe_20260331_150643"

# Sample sentences across the trained languages
SAMPLES = {
    "English": "Artificial intelligence is rapidly transforming the modern world.",
    "Hindi": "कृत्रिम बुद्धिमत्ता तेजी से आधुनिक दुनिया को बदल रही है।",
    "Bengali": "কৃত্রিম বুদ্ধিমত্তা দ্রুত আধুনিক বিশ্বকে বদলে দিচ্ছে।",
    "Tamil": "செயற்கை நுண்ணறிவு நவீன உலகத்தை வேகமாக மாற்றி வருகிறது.",
    "Telugu": "కృత్రిమ మేధస్సు ఆధునిక ప్రపంచాన్ని వేగంగా మారుస్తోంది.",
}


def load_tokenizer(label: str, path_or_id: str):
    print(f"Loading [{label}] from: {path_or_id}...")
    try:
        kwargs = dict(use_fast=True, trust_remote_code=True)
        try:
            return AutoTokenizer.from_pretrained(path_or_id, fix_mistral_regex=True, **kwargs)
        except TypeError:
            return AutoTokenizer.from_pretrained(path_or_id, **kwargs)
    except Exception as e:
        print(f"  Failed: {e}")
        return None


def analyze_three(
    tokenizers: list[tuple[str, object]],
    language: str,
    text: str,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Language: {language}")
    print(f"Original: {text}")
    print(f"{'=' * 60}")

    base_tok = tokenizers[0][1]
    base_ids = base_tok.encode(text, add_special_tokens=False)

    for label, tok in tokenizers:
        ids = tok.encode(text, add_special_tokens=False)
        toks = tok.convert_ids_to_tokens(ids)
        decoded = tok.decode(ids)
        print(f"\n[ {label} ]")
        print(f"  Token count: {len(ids)}")
        preview = toks[:20]
        suffix = " ..." if len(toks) > 20 else ""
        print(f"  Subwords (first 20): {preview}{suffix}")
        if decoded == text:
            print("  Decode: OK (matches original)")
        else:
            print("  Decode: mismatch")
            print(f"  Got: {decoded!r}")

        if tok is base_tok:
            continue
        if len(base_ids) > 0:
            saved = len(base_ids) - len(ids)
            pct = (saved / len(base_ids)) * 100
            print(f"  vs base: {saved:+d} tokens ({pct:+.1f}% rel. to base count)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare base, indic-expanded, and continued-BPE Nemotron tokenizers."
    )
    parser.add_argument("--base", default=DEFAULT_BASE, help="Base model ID or local path")
    parser.add_argument(
        "--indic-expanded",
        default=DEFAULT_INDIC_EXPANDED,
        help="Path to nemotron-indic-expanded tokenizer",
    )
    parser.add_argument(
        "--continued-bpe",
        default=DEFAULT_CONTINUED_BPE,
        help="Path to continued_bpe output folder (saved tokenizer)",
    )
    args = parser.parse_args()

    specs = [
        ("BASE (Nemotron)", args.base),
        ("INDIC-EXPANDED", args.indic_expanded),
        ("CONTINUED-BPE", args.continued_bpe),
    ]

    def _missing_local(p: str) -> bool:
        if not p:
            return True
        if os.path.isabs(p) or p.startswith("./") or p.startswith("../"):
            return not os.path.isdir(p)
        return False

    loaded: list[tuple[str, object]] = []
    for label, path in specs:
        if not path:
            print(f"Skipping {label}: empty path")
            continue
        if _missing_local(path):
            print(f"Skipping {label}: local path missing or not a directory: {path}")
            continue
        tok = load_tokenizer(label, path)
        if tok is not None:
            loaded.append((label, tok))

    if len(loaded) < 2:
        print("Need at least base plus one local tokenizer. Fix paths and retry.")
        return

    print("\n--- Vocab sizes ---")
    for label, tok in loaded:
        print(f"  {label}: {len(tok):,}")

    for lang, text in SAMPLES.items():
        analyze_three(loaded, lang, text)

    print(f"\n{'=' * 60}")
    print("Done.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
