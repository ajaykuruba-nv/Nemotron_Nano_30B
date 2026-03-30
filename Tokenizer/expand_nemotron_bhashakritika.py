#!/usr/bin/env python3
"""
Correctly expand NVIDIA Nemotron 3 Nano 30B tokenizer using krutrim-ai-labs/BhashaKritika.

This script trains a new tokenizer to discover optimal subwords, decodes them to unicode,
and adds them via `add_tokens()` to bypass BPE merge conflicts. It includes regex fixes
for Mistral-based tokenizers to ensure proper Indic character grouping, and resizes/mean-initializes
both input and output embeddings.
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from typing import Any

import torch
from datasets import interleave_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Regex
# ---------------------------------------------------------------------------
DATASET_ID = "krutrim-ai-labs/BhashaKritika"
DEFAULT_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEVANAGARI_LANGS = frozenset({"hindi", "marathi"})

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\[\]{}|\\^`\"']+")
EMAIL_RE = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")
MULTISPACE_RE = re.compile(r"[ \t]+")

def get_devanagari_normalizer():
    from indicnlp.normalize.indic_normalize import DevanagariNormalizer
    return DevanagariNormalizer()

def clean_text(text: str, devanagari_norm: Any) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "").replace("\ufeff", "")
    text = unicodedata.normalize("NFKC", text)
    if devanagari_norm:
        text = devanagari_norm.normalize(text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()

def get_mixed_language_stream(languages: list[str], max_samples_per_lang: int, devanagari_norm: Any, pbar: tqdm):
    datasets = []
    for lang in languages:
        ds = load_dataset(DATASET_ID, name=lang, split="train", streaming=True)
        ds = ds.take(max_samples_per_lang)
        datasets.append(ds)
    
    mixed_ds = interleave_datasets(datasets)
    
    for example in mixed_ds:
        raw_text = example.get("response", "") or example.get("prompt", "")
        if isinstance(raw_text, str) and raw_text.strip():
            cleaned = clean_text(raw_text, devanagari_norm)
            if len(cleaned) > 50:
                yield cleaned
                pbar.update(1)

def batch_iterator(stream, batch_size=1000):
    batch = []
    for item in stream:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default="./nemotron-indic-expanded")
    parser.add_argument("--languages", default="hindi,bengali,tamil,telugu")
    parser.add_argument("--new-tokens", type=int, default=64000)
    parser.add_argument("--samples-per-lang", type=int, default=500000)
    parser.add_argument("--tokenizer-only", action="store_true")
    args = parser.parse_args()

    langs = [l.strip() for l in args.languages.split(",")]
    devanagari_norm = get_devanagari_normalizer()

    logger.info("Loading base tokenizer with regex fixes...")
    
    # CRITICAL FIX: fix_mistral_regex=True prevents the pre-tokenizer from shattering Indic words
    base_tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, 
        use_fast=True, 
        trust_remote_code=True,
        fix_mistral_regex=True
    )
    original_vocab_size = len(base_tokenizer)

    total_docs = len(langs) * args.samples_per_lang
    logger.info(f"Streaming ~{total_docs} docs to discover optimal Indic subwords...")
    
    with tqdm(total=total_docs, desc="Training BPE", unit="docs") as pbar:
        stream = get_mixed_language_stream(langs, args.samples_per_lang, devanagari_norm, pbar)
        
        # 1. Train a temporary tokenizer to discover the best subwords
        # FIXED: Directly use args.new_tokens so we don't accidentally add 150k+ tokens
        trained_tokenizer = base_tokenizer.train_new_from_iterator(
            batch_iterator(stream), 
            vocab_size=args.new_tokens
        )

    # 2. Extract the learned tokens and convert them back to strings
    logger.info("Extracting new tokens and overriding BPE priorities...")
    base_vocab_keys = set(base_tokenizer.get_vocab().keys())
    new_vocab_keys = set(trained_tokenizer.get_vocab().keys()) - base_vocab_keys
    
    # We must decode them to proper Unicode strings so add_tokens handles spaces accurately
    new_tokens_unicode = set()
    for byte_token in new_vocab_keys:
        decoded_str = trained_tokenizer.convert_tokens_to_string([byte_token])
        if decoded_str.strip(): # Ignore empty/junk parses
            new_tokens_unicode.add(decoded_str)

    # 3. Add them safely to the base tokenizer
    added_count = base_tokenizer.add_tokens(list(new_tokens_unicode))
    new_vocab_size = len(base_tokenizer)
    
    logger.info(f"Base Vocabulary Size: {original_vocab_size}")
    logger.info(f"New Vocabulary Size:  {new_vocab_size}")
    logger.info(f"Total Tokens Added:   +{added_count}")

    if args.tokenizer_only:
        base_tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Tokenizer saved to {args.output_dir}. Skipping model load.")
        return

    # 4. Load Model & Resize Embeddings
    logger.info("Loading 30B model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    logger.info("Resizing token embeddings...")
    model.resize_token_embeddings(new_vocab_size)

    # 5. Mean Initialization (Crucial for preventing gibberish generation)
    logger.info("Applying mean initialization...")
    with torch.no_grad():
        in_emb = model.get_input_embeddings().weight
        mean_in = in_emb[:original_vocab_size].mean(dim=0)
        in_emb[original_vocab_size:].copy_(mean_in)
        
        out_emb = model.get_output_embeddings().weight
        mean_out = out_emb[:original_vocab_size].mean(dim=0)
        out_emb[original_vocab_size:].copy_(mean_out)

    logger.info(f"Saving model and tokenizer to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    base_tokenizer.save_pretrained(args.output_dir)
    logger.info("Complete. Model is ready for Indic continual pre-training.")

if __name__ == "__main__":
    main()