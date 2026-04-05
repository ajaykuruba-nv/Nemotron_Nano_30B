#!/usr/bin/env python3

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


# Default local tokenizers (repo root-relative).
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NEMOTRON_EXTENDED_TOKENIZER = REPO_ROOT / "Tokenizer" / "nemotron-indic-expanded"
# After `continued_bpe.py` save_pretrained (timestamped outputs folder).
DEFAULT_NEMOTRON_CONTINUED_BPE_TOKENIZER = (
    REPO_ROOT / "Tokenizer" / "outputs" / "continued_bpe_20260404_041116"
)

NEMOTRON_BASE_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

MODEL_CONFIGS = {
    "Aya Fire": "CohereLabs/tiny-aya-fire",
    "Nemotron": NEMOTRON_BASE_ID,
    "Nemotron Indic Expanded": NEMOTRON_BASE_ID,
    "Nemotron Continued BPE": NEMOTRON_BASE_ID,
    "Sarvam 30B": "sarvamai/sarvam-30b",
    "Mistral Nemo Base 2407": "mistralai/Mistral-Nemo-Base-2407",
    "GPT-OSS-20B": "openai/gpt-oss-20b",
    "Gemma 3 4B": "google/gemma-3-4b-it",
    "Gemma 4 31B": "google/gemma-4-31B",
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


def _needs_gemma4_extra_special_tokens_dict_fix(model_id: str) -> bool:
    """google/gemma-4-* tokenizer_config.json may list extra_special_tokens as a JSON array; transformers expects a dict."""
    return model_id.startswith("google/gemma-4-")


def _normalize_tokenizer_config_extra_special_tokens(cfg: dict) -> dict:
    est = cfg.get("extra_special_tokens")
    if not isinstance(est, list):
        return cfg
    mapping: dict[str, str] = {}
    for i, tok in enumerate(est):
        t = str(tok)
        if t == "<|video|>":
            mapping["video_token"] = t
        else:
            mapping[f"extra_special_token_{i}"] = t
    cfg = dict(cfg)
    cfg["extra_special_tokens"] = mapping
    return cfg


def load_tokenizer_from_hub_with_tokenizer_config_patch(model_id: str, hf_token: str | None, **from_pretrained_kwargs) -> AutoTokenizer:
    """
    Work around: AttributeError 'list' object has no attribute 'keys' when loading Gemma 4 fast tokenizers
    (extra_special_tokens is a list in tokenizer_config.json on the Hub).
    """
    with tempfile.TemporaryDirectory(prefix="hf_tok_cfg_patch_") as tmp:
        snapshot_download(
            repo_id=model_id,
            local_dir=tmp,
            allow_patterns=["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"],
            token=hf_token,
        )
        cfg_path = Path(tmp) / "tokenizer_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg = _normalize_tokenizer_config_extra_special_tokens(cfg)
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        return AutoTokenizer.from_pretrained(
            tmp,
            local_files_only=True,
            token=from_pretrained_kwargs.pop("token", hf_token),
            **from_pretrained_kwargs,
        )


def load_tokenizer(model_name, model_id, hf_token, *, local_tokenizer_path=None):
    if local_tokenizer_path is not None:
        kwargs = {}
        if model_name in {
            "Aya Fire",
            "Nemotron",
            "Nemotron Indic Expanded",
            "Nemotron Continued BPE",
            "Sarvam 30B",
            "Mistral Nemo Base 2407",
        }:
            kwargs["trust_remote_code"] = True
        if model_name in {"Sarvam 30B", "Nemotron", "Nemotron Indic Expanded", "Nemotron Continued BPE"}:
            kwargs["fix_mistral_regex"] = True
        return AutoTokenizer.from_pretrained(local_tokenizer_path, **kwargs)

    kwargs = {"token": hf_token}
    if model_name in {
        "Aya Fire",
        "Nemotron",
        "Nemotron Indic Expanded",
        "Nemotron Continued BPE",
        "Sarvam 30B",
        "Mistral Nemo Base 2407",
    }:
        kwargs["trust_remote_code"] = True
    if model_name in {"Sarvam 30B", "Nemotron", "Nemotron Indic Expanded", "Nemotron Continued BPE"}:
        kwargs["fix_mistral_regex"] = True
    if _needs_gemma4_extra_special_tokens_dict_fix(model_id):
        kwargs["trust_remote_code"] = True
        return load_tokenizer_from_hub_with_tokenizer_config_patch(model_id, hf_token, **kwargs)
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
    parser.add_argument(
        "--nemotron-extended-only",
        action="store_true",
        help=(
            "Run token fertility only for Nemotron, using the locally saved extended "
            "tokenizer (see --nemotron-tokenizer-path). Ignores --models."
        ),
    )
    parser.add_argument(
        "--nemotron-tokenizer-path",
        type=str,
        default=None,
        help=(
            "Directory with save_pretrained() output for the extended Nemotron tokenizer. "
            f"Default when --nemotron-extended-only: {DEFAULT_NEMOTRON_EXTENDED_TOKENIZER}"
        ),
    )
    parser.add_argument(
        "--nemotron-continued-bpe-path",
        type=str,
        default=None,
        help=(
            "Directory with save_pretrained() output for Nemotron continued-BPE tokenizer. "
            f"Used for model name 'Nemotron Continued BPE'. Default: {DEFAULT_NEMOTRON_CONTINUED_BPE_TOKENIZER}"
        ),
    )
    args = parser.parse_args()

    hf_token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    continued_bpe_path = args.nemotron_continued_bpe_path or str(DEFAULT_NEMOTRON_CONTINUED_BPE_TOKENIZER)
    local_tokenizer_overrides = {
        "Nemotron Indic Expanded": str(DEFAULT_NEMOTRON_EXTENDED_TOKENIZER),
        "Nemotron Continued BPE": continued_bpe_path,
    }

    nemotron_local_tok = None
    if args.nemotron_extended_only:
        models_to_run = ["Nemotron"]
        nemotron_local_tok = args.nemotron_tokenizer_path or str(DEFAULT_NEMOTRON_EXTENDED_TOKENIZER)
    else:
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
        if args.nemotron_extended_only and model_name == "Nemotron":
            print(f"Tokenizer (local extended): {nemotron_local_tok}")
        if model_name in local_tokenizer_overrides:
            print(f"Tokenizer (local): {local_tokenizer_overrides[model_name]}")
        print(f"{'=' * 60}")

        try:
            local_path = None
            if args.nemotron_extended_only and model_name == "Nemotron":
                local_path = nemotron_local_tok
            elif model_name in local_tokenizer_overrides:
                local_path = local_tokenizer_overrides[model_name]
            tokenizer = load_tokenizer(model_name, model_id, hf_token, local_tokenizer_path=local_path)
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

        entry = {"model_id": model_id, "results": model_results}
        if args.nemotron_extended_only and model_name == "Nemotron":
            entry["tokenizer_source"] = "local_extended"
            entry["tokenizer_path"] = nemotron_local_tok
        elif model_name in local_tokenizer_overrides:
            entry["tokenizer_source"] = "local"
            entry["tokenizer_path"] = local_tokenizer_overrides[model_name]
        all_results["models"][model_name] = entry

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()