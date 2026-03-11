#!/usr/bin/env python3
"""
Compare tokenizers: mistralai/Mistral-Nemo-Base-2407 vs nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.
Checks if they are the same (vocab size, type, and identical tokenization on sample strings).
"""

import os
from transformers import AutoTokenizer

MODELS = {
    "Mistral Nemo Base 2407": "mistralai/Mistral-Nemo-Base-2407",
    "Nemotron": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
}

TEST_STRINGS = [
    "भारत एक बड़ा देश है।",
    "বাংলা একটি সুন্দর ভাষা।",
    "தமிழ் ஒரு பழமையான மொழி.",
    "తెలుగు ఒక అందమైన భాష.",
    "Hello world",
    "GPU tokenization test 123",
    "भारत",
    "বাংলা",
    "தமிழ்",
    "తెలుగు",
]


def load_tokenizer(name, model_id, hf_token=None):
    kwargs = {"token": hf_token}
    if name in {"Mistral Nemo Base 2407", "Nemotron"}:
        kwargs["trust_remote_code"] = True
    return AutoTokenizer.from_pretrained(model_id, **kwargs)


def inspect_tokenizer(name, tok):
    print(f"\n{name}")
    print("-" * 80)
    print("class       :", type(tok))
    print("name_or_path:", getattr(tok, "name_or_path", "N/A"))
    try:
        print("vocab size  :", len(tok))
    except Exception:
        print("vocab size  : N/A")


def compare_tokenization(tok1, name1, tok2, name2, text):
    ids1 = tok1.encode(text, add_special_tokens=False)
    ids2 = tok2.encode(text, add_special_tokens=False)
    toks1 = tok1.convert_ids_to_tokens(ids1)
    toks2 = tok2.convert_ids_to_tokens(ids2)
    same_ids = ids1 == ids2
    same_tokens = toks1 == toks2
    print(f"\nTEXT: {text}")
    print(f"{name1} ids   :", ids1)
    print(f"{name2} ids   :", ids2)
    print(f"{name1} toks  :", toks1)
    print(f"{name2} toks  :", toks2)
    print("same ids?      :", same_ids)
    print("same tokens?   :", same_tokens)
    print(f"{name1} count :", len(ids1))
    print(f"{name2} count :", len(ids2))
    return same_ids and same_tokens


def main():
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    mistral = load_tokenizer("Mistral Nemo Base 2407", MODELS["Mistral Nemo Base 2407"], hf_token)
    nemotron = load_tokenizer("Nemotron", MODELS["Nemotron"], hf_token)

    inspect_tokenizer("Mistral Nemo Base 2407", mistral)
    inspect_tokenizer("Nemotron", nemotron)

    # Quick same-vocab check
    try:
        len_m = len(mistral)
        len_n = len(nemotron)
        print("\n" + "-" * 80)
        print(f"Vocab size: Mistral Nemo Base 2407 = {len_m}, Nemotron = {len_n}")
        print("Same vocab size?" if len_m == len_n else "Different vocab size.")
    except Exception as e:
        print("Could not compare vocab size:", e)

    all_same = True
    for text in TEST_STRINGS:
        same = compare_tokenization(
            mistral, "Mistral Nemo Base 2407",
            nemotron, "Nemotron",
            text,
        )
        all_same = all_same and same

    print("\n" + "=" * 80)
    if all_same:
        print("RESULT: Mistral Nemo Base 2407 and Nemotron tokenize all tested strings identically.")
        print("This strongly suggests they use the same tokenizer (e.g. same base).")
    else:
        print("RESULT: Mistral Nemo Base 2407 and Nemotron do NOT tokenize all tested strings identically.")
        print("They are different tokenizers.")
    print("=" * 80)


if __name__ == "__main__":
    main()
