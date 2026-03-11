#!/usr/bin/env python3

import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_CONFIGS = {
    "Aya Fire": "CohereLabs/tiny-aya-fire",
    "Nemotron": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "Sarvam 30B": "sarvamai/sarvam-30b",
    "Mistral Nemo Base 2407": "mistralai/Mistral-Nemo-Base-2407",
    "GPT-OSS-20B": "openai/gpt-oss-20b",
    "Gemma 3 4B": "google/gemma-3-4b-it",
    "Qwen 3.5": "Qwen/Qwen3.5-397B-A17B",
}

LANG_CONFIG = {
    "hindi": ("hi", "Hindi"),
    "bengali": ("bn", "Bengali"),
    "tamil": ("ta", "Tamil"),
    "telugu": ("te", "Telugu"),
}


def count_words(text):
    return len(text.split())


def get_text(row, config_name):
    if row.get("tgt"):
        return row["tgt"].strip()

    translation = row.get("translation")
    if isinstance(translation, dict) and translation.get(config_name):
        return translation[config_name].strip()

    return ""


def load_subsets(hf_token, samples, seed):
    subsets = {}
    for lang_key, (config_name, lang_label) in LANG_CONFIG.items():
        try:
            ds = load_dataset(
                "ai4bharat/samanantar",
                config_name,
                split="train",
                token=hf_token,
            )
        except Exception as e:
            print(f"Failed to load {lang_label} ({config_name}): {e}")
            continue

        n = min(samples, len(ds))
        indices = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:n].tolist()
        subset = ds.select(indices)
        subsets[lang_key] = [subset[i] for i in range(n)]

    return subsets


def load_tokenizer(model_name, model_id, hf_token):
    kwargs = {"token": hf_token}
    if model_name in {"Aya Fire", "Nemotron", "Sarvam 30B", "Mistral Nemo Base 2407"}:
        kwargs["trust_remote_code"] = True
    if model_name == "Sarvam 30B":
        kwargs["fix_mistral_regex"] = True
    return AutoTokenizer.from_pretrained(model_id, **kwargs)


def compute_fertility(tokenizer, config_name, rows):
    total_tokens = 0
    total_words = 0
    valid = 0

    for row in rows:
        text = get_text(row, config_name)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="fertility_all_models_results.json")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--models", type=str, nargs="*", default=None)
    parser.add_argument("--verify-sample", action="store_true")
    args = parser.parse_args()

    hf_token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    models_to_run = args.models if args.models else list(MODEL_CONFIGS.keys())
    models_to_run = [m for m in models_to_run if m in MODEL_CONFIGS]

    print("Loading dataset subsets...")
    subsets = load_subsets(hf_token, args.samples, args.seed)

    if not subsets:
        print("No dataset loaded. Aborting.")
        return

    if args.verify_sample:
        for lang_key, (config_name, lang_label) in LANG_CONFIG.items():
            rows = subsets.get(lang_key, [])
            if rows:
                print(f"\n{lang_label} ({config_name})")
                print(rows[0])
                print("Extracted text:", repr(get_text(rows[0], config_name)[:200]))
        return

    all_results = {
        "dataset": "ai4bharat/samanantar",
        "fertility_definition": "sum(tokens)/sum(words)",
        "samples_per_lang": args.samples,
        "seed": args.seed,
        "models": {},
    }

    for model_name in models_to_run:
        model_id = MODEL_CONFIGS[model_name]
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name} ({model_id})")
        print(f"{'=' * 60}")

        try:
            tokenizer = load_tokenizer(model_name, model_id, hf_token)
        except Exception as e:
            print(f"  Failed to load tokenizer: {e}")
            all_results["models"][model_name] = {"model_id": model_id, "error": str(e)}
            continue

        model_results = {}
        for lang_key, (config_name, lang_label) in LANG_CONFIG.items():
            rows = subsets.get(lang_key, [])
            print(f"  --- {lang_label} ({config_name}) ---")

            if not rows:
                print("    No rows loaded")
                model_results[lang_key] = {"error": "No rows loaded"}
                continue

            res = compute_fertility(tokenizer, config_name, rows)
            model_results[lang_key] = res

            if res.get("fertility") is not None:
                print(f"    Fertility (T/W): {res['fertility']}  (n={res['samples']})")
            else:
                print(f"    Error: {res['error']}")

        all_results["models"][model_name] = {
            "model_id": model_id,
            "results": model_results,
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()