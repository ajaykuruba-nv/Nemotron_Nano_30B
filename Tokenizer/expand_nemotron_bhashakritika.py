#!/usr/bin/env python3
"""
Correctly expand NVIDIA Nemotron 3 Nano 30B tokenizer using krutrim-ai-labs/BhashaKritika.

This script trains a *joint* BPE model across multiple Indic languages to maximize 
subword sharing, explicitly merges the BPE rules (vocab + merges) into the base 
Fast Tokenizer's JSON state, and mean-initializes both input and output embeddings.

Setup:
  pip install -r requirements.txt tqdm
  # Optional for Devanagari normalization:
  # pip install indic-nlp-library

Example:
  python expand_nemotron_indic.py --languages hindi,bengali,tamil,telugu --new-tokens 32000
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from typing import Any

import torch
from datasets import interleave_datasets, load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
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

def _maybe_devanagari_normalizer():
    try:
        from indicnlp.normalize.indic_normalize import DevanagariNormalizer
        return DevanagariNormalizer()
    except ImportError:
        logger.warning("indic-nlp-library not installed; skipping Devanagari Normalizer.")
        return None

def clean_text(text: str, devanagari_norm: Any) -> str:
    """Standardizes text and strips junk before tokenizer training."""
    if not text:
        return ""
    
    # 1. Strip raw byte artifacts
    text = text.replace("\x00", "").replace("\ufeff", "")
    
    # 2. NFKC Normalization: Crucial for Indic scripts to standardize combining characters (Matras)
    text = unicodedata.normalize("NFKC", text)

    # 3. Devanagari specific hygiene (Nuktas, ZWJ/ZWNJ cleanup)
    if devanagari_norm:
        text = devanagari_norm.normalize(text)

    # 4. Strip boilerplate links and excessive spacing
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()

def get_mixed_language_stream(languages: list[str], max_samples_per_lang: int, devanagari_norm: Any, pbar: tqdm):
    """Interleaves multiple languages into a single randomized stream for joint BPE training."""
    datasets = []
    for lang in languages:
        ds = load_dataset(DATASET_ID, name=lang, split="train", streaming=True)
        # Take a slice to ensure balanced representation
        ds = ds.take(max_samples_per_lang)
        datasets.append(ds)
    
    mixed_ds = interleave_datasets(datasets)
    
    for example in mixed_ds:
        # Prefer response, fallback to prompt
        raw_text = example.get("response", "") or example.get("prompt", "")
        if isinstance(raw_text, str) and raw_text.strip():
            cleaned = clean_text(raw_text, devanagari_norm)
            if len(cleaned) > 50: # Basic length filter to ignore useless fragments
                yield cleaned
                pbar.update(1) # Update the progress bar for every valid document yielded

def train_and_merge_bpe(base_tokenizer: PreTrainedTokenizerFast, stream, new_tokens_count: int) -> PreTrainedTokenizerFast:
    """
    Trains a new BPE on the Indic stream and cleanly injects its vocab and merges 
    into the base tokenizer's underlying JSON state.
    """
    logger.info(f"Training new BPE model targetting {new_tokens_count} tokens (This may take a while)...")
    
    # 1. Train a vanilla BPE model on our Indic text using a ByteLevel pre-tokenizer
    # ByteLevel is vital for Indic languages to prevent <UNK> tokens on rare conjuncts
    new_bpe = Tokenizer(models.BPE())
    new_bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=new_tokens_count, special_tokens=[])
    
    # The Tokenizers library will consume our tracked stream
    new_bpe.train_from_iterator(stream, trainer=trainer)
    
    # Extract the new rules
    new_vocab = new_bpe.get_vocab()
    new_merges = []
    
    # Tokenizers library doesn't expose merges easily, so we parse its JSON
    new_bpe_json = json.loads(new_bpe.to_str())
    if "merges" in new_bpe_json["model"]:
        new_merges = new_bpe_json["model"]["merges"]

    # 2. Extract base tokenizer's JSON state
    base_json = json.loads(base_tokenizer.backend_tokenizer.to_str())
    base_vocab = base_json["model"]["vocab"]
    base_merges_list = base_json["model"].get("merges", [])
    
    existing_tokens = set(base_vocab.keys())
    existing_merges = set(base_merges_list)
    
    # 3. Carefully inject new tokens and merges
    current_id = max(base_vocab.values()) + 1
    added_tokens = 0
    added_merges = 0
    
    # Add merges first (BPE needs rules)
    for merge in new_merges:
        if merge not in existing_merges:
            base_merges_list.append(merge)
            added_merges += 1
            
    # Add vocabulary mapping
    for token in new_vocab.keys():
        if token not in existing_tokens:
            base_vocab[token] = current_id
            current_id += 1
            added_tokens += 1
            
    # Update the base JSON dictionary
    base_json["model"]["vocab"] = base_vocab
    base_json["model"]["merges"] = base_merges_list
    
    logger.info(f"Successfully merged {added_tokens} UNIQUE new tokens and {added_merges} new BPE merges.")
    
    # 4. Reload the updated JSON into a new Hugging Face Fast Tokenizer
    updated_backend = Tokenizer.from_str(json.dumps(base_json))
    
    # Wrap it back into the Transformers FastTokenizer format
    return PreTrainedTokenizerFast(
        tokenizer_object=updated_backend,
        unk_token=base_tokenizer.unk_token,
        bos_token=base_tokenizer.bos_token,
        eos_token=base_tokenizer.eos_token,
        pad_token=base_tokenizer.pad_token,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default="./nemotron-indic-expanded")
    parser.add_argument("--languages", default="hindi,bengali,tamil,telugu")
    parser.add_argument("--new-tokens", type=int, default=32000, help="Total new subwords to learn across ALL combined languages")
    parser.add_argument("--samples-per-lang", type=int, default=150000)
    parser.add_argument("--tokenizer-only", action="store_true")
    args = parser.parse_args()

    langs = [l.strip() for l in args.languages.split(",")]
    devanagari_norm = _maybe_devanagari_normalizer()

    logger.info("Loading base tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    original_vocab_size = len(base_tokenizer)

    # 1. Stream, Train, and Merge (With Progress Bar)
    total_estimated_docs = len(langs) * args.samples_per_lang
    logger.info(f"Initializing stream for ~{total_estimated_docs} documents across {len(langs)} languages...")
    
    with tqdm(total=total_estimated_docs, desc="Extracting & Training BPE", unit="docs") as pbar:
        stream = get_mixed_language_stream(langs, args.samples_per_lang, devanagari_norm, pbar)
        expanded_tokenizer = train_and_merge_bpe(base_tokenizer, stream, args.new_tokens)
    
    new_vocab_size = len(expanded_tokenizer)
    logger.info(f"Base Vocabulary Size: {original_vocab_size}")
    logger.info(f"New Vocabulary Size:  {new_vocab_size}")
    logger.info(f"Total Expansion:      +{new_vocab_size - original_vocab_size} tokens")

    if args.tokenizer_only:
        expanded_tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Tokenizer saved to {args.output_dir}. Skipping model load.")
        return

    # 2. Load Model & Resize
    logger.info("Loading 30B model weights (Requires high RAM/VRAM)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    logger.info("Resizing token embeddings...")
    model.resize_token_embeddings(new_vocab_size)

    # 3. Mean Initialization for BOTH Input and Output Embeddings
    logger.info("Applying mean initialization to input and output embeddings...")
    with torch.no_grad():
        # Input Embeddings
        in_emb = model.get_input_embeddings().weight
        mean_in = in_emb[:original_vocab_size].mean(dim=0)
        in_emb[original_vocab_size:].copy_(mean_in)
        
        # Output Embeddings (lm_head)
        out_emb = model.get_output_embeddings().weight
        mean_out = out_emb[:original_vocab_size].mean(dim=0)
        out_emb[original_vocab_size:].copy_(mean_out)

    # 4. Save
    logger.info(f"Saving model and tokenizer to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    expanded_tokenizer.save_pretrained(args.output_dir)
    logger.info("Complete. Model is ready for Indic continual pre-training.")

if __name__ == "__main__":
    main()