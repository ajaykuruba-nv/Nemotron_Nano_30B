#!/usr/bin/env python3
"""
Complete Nemotron Tokenizer Extension Toolkit (Unified).

This script implements all logics from taidopurason/tokenizer-extension:
1. Continued BPE Training (train_vocab_extension & bpe_extension)
2. Safe Extension Application (extension.py)
3. "Mean of Constituents" Embedding Modification (models.py)
4. Unreachable Token Benchmarking (benchmarking.py)
5. Vocabulary Pruning Strategies (pruning.py)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import re
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import torch
from datasets import interleave_datasets, load_dataset
from tokenizers import Tokenizer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# 1. CONSTANTS & DATA STREAMING (data.py)
# =============================================================================
DATASET_ID = "krutrim-ai-labs/BhashaKritika"
DEFAULT_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\[\]{}|\\^`\"']+")
EMAIL_RE = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")
MULTISPACE_RE = re.compile(r"[ \t]+")

def get_devanagari_normalizer() -> Any:
    try:
        from indicnlp.normalize.indic_normalize import DevanagariNormalizer
        return DevanagariNormalizer()
    except ImportError:
        logger.warning("indic-nlp-library missing; skipping Devanagari Normalizer.")
        return None

def clean_text(text: str, devanagari_norm: Any) -> str:
    if not text: return ""
    text = text.replace("\x00", "").replace("\ufeff", "")
    text = unicodedata.normalize("NFKC", text)
    if devanagari_norm: text = devanagari_norm.normalize(text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()

def mixed_language_text_stream(
    languages: list[str], max_samples_per_lang: int, devanagari_norm: Any, pbar: tqdm | None
) -> Iterator[str]:
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
                if pbar is not None: pbar.update(1)

def batch_iterator(stream: Iterable[str], batch_size: int = 1000) -> Iterator[list[str]]:
    batch: list[str] = []
    for item in stream:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch: yield batch


# =============================================================================
# 2. BPE ARTIFACTS EXTRACTION (bpe_extension.py & train_vocab_extension.py)
# =============================================================================
@dataclass(frozen=True)
class ExtensionArtifacts:
    new_vocab: dict[str, int]
    new_merges: list[str]

def _get_bpe_state(tokenizer_backend_str: str) -> tuple[dict[str, int], list[str]]:
    obj = json.loads(tokenizer_backend_str)
    model = obj.get("model", {})
    if model.get("type") != "BPE":
        raise ValueError(f"Expected BPE backend, got {model.get('type')!r}")
    
    vocab = model.get("vocab", {})
    merges_raw = model.get("merges", [])
    merges: list[str] = []
    for m in merges_raw:
        if isinstance(m, str):
            merges.append(m)
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            merges.append(f"{m[0]} {m[1]}")
    return vocab, merges

def compute_continued_bpe_artifacts(base_backend: Tokenizer, trained_backend: Tokenizer) -> ExtensionArtifacts:
    base_vocab, base_merges = _get_bpe_state(base_backend.to_str())
    trained_vocab, trained_merges = _get_bpe_state(trained_backend.to_str())

    base_merges_set = set(base_merges)
    new_merges = [m for m in trained_merges if m not in base_merges_set]

    base_vocab_set = set(base_vocab.keys())
    new_vocab = {t: int(trained_vocab[t]) for t in trained_vocab.keys() if t not in base_vocab_set}
    return ExtensionArtifacts(new_vocab=new_vocab, new_merges=new_merges)


# =============================================================================
# 3. EXTENSION APPLICATION (extension.py)
# =============================================================================
def _apply_bpe_extension_backend(
    base_backend: Tokenizer, new_vocab: dict[str, int], new_merges: list[str] | None, 
    n_tokens: int, keep_added_token_positions: bool
) -> Tokenizer:
    obj = json.loads(base_backend.to_str())
    model = obj.get("model", {})
    vocab: dict[str, int] = model["vocab"]
    
    merges_raw = model.get("merges", [])
    merges: list[str] = [m if isinstance(m, str) else f"{m[0]} {m[1]}" for m in merges_raw]
    base_size = len(vocab)

    selected_tokens = [t for (t, _) in sorted(new_vocab.items(), key=lambda kv: kv[1])][: int(n_tokens)]
    selected_set = set(selected_tokens)

    if keep_added_token_positions:
        rank_map = {tok: i for i, tok in enumerate(selected_tokens)}
        for tok in selected_tokens:
            if tok not in vocab: vocab[tok] = base_size + rank_map[tok]
    else:
        next_id = (max(vocab.values()) + 1) if vocab else 0
        for tok in selected_tokens:
            if tok not in vocab:
                vocab[tok] = next_id
                next_id += 1

    if not new_merges:
        model["vocab"] = vocab
        return Tokenizer.from_str(json.dumps(obj))

    existing_merges = set(merges)

    def ensure_token(tok: str) -> None:
        if tok not in vocab: vocab[tok] = (max(vocab.values()) + 1) if vocab else 0

    filtered_merges: list[str] = []
    for pair in new_merges:
        parts = pair.split(" ")
        if len(parts) == 2 and (parts[0] in selected_set or parts[1] in selected_set or (parts[0]+parts[1]) in selected_set):
            filtered_merges.append(pair)

    for rule in filtered_merges:
        parts = rule.split(" ")
        if len(parts) == 2:
            ensure_token(parts[0])
            ensure_token(parts[1])
            ensure_token(parts[0] + parts[1])
            if rule not in existing_merges:
                merges.append(rule)
                existing_merges.add(rule)

    model["vocab"] = vocab
    model["merges"] = merges
    return Tokenizer.from_str(json.dumps(obj))

def extend_tokenizer(
    tokenizer: PreTrainedTokenizerBase, new_vocab: dict[str, int], new_merges: list[tuple[str, str]] | None,
    n_tokens: int, keep_added_token_positions: bool = False
) -> PreTrainedTokenizerFast:
    merges_list = [" ".join(x) for x in new_merges] if new_merges else None
    updated_backend = _apply_bpe_extension_backend(
        base_backend=Tokenizer.from_str(tokenizer.backend_tokenizer.to_str()),
        new_vocab=new_vocab, new_merges=merges_list, n_tokens=n_tokens, keep_added_token_positions=keep_added_token_positions
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=updated_backend,
        unk_token=getattr(tokenizer, "unk_token", None), bos_token=getattr(tokenizer, "bos_token", None),
        eos_token=getattr(tokenizer, "eos_token", None), pad_token=getattr(tokenizer, "pad_token", None)
    )


# =============================================================================
# 4. MODEL MODIFICATION (models.py)
# =============================================================================
InitMethod = Literal["mean", "mean_of_constituents"]

def modify_embeddings(
    model: PreTrainedModel, old_tokenizer: PreTrainedTokenizerBase, new_tokenizer: PreTrainedTokenizerBase,
    init_method: InitMethod = "mean_of_constituents", ignore_size_mismatch: bool = False
) -> dict[str, Any]:
    old_n, new_n = len(old_tokenizer), len(new_tokenizer)

    if not ignore_size_mismatch and model.get_input_embeddings().weight.shape[0] != old_n:
        raise ValueError(f"Model rows ({model.get_input_embeddings().weight.shape[0]}) != old tokenizer ({old_n}).")

    changes: dict[str, Any] = {"old_vocab_size": old_n, "new_vocab_size": new_n, "initialized": []}
    model.resize_token_embeddings(new_n)
    
    in_emb = model.get_input_embeddings().weight
    global_mean_in = in_emb[:old_n].mean(dim=0)

    out_layer = model.get_output_embeddings()
    out_emb = out_layer.weight if out_layer is not None else None
    global_mean_out = out_emb[:old_n].mean(dim=0) if out_emb is not None else None

    with torch.no_grad():
        for tid in tqdm(range(old_n, new_n), desc="Initializing Embeddings"):
            tok = new_tokenizer.convert_ids_to_tokens(tid)
            vec_in, vec_out = global_mean_in, global_mean_out

            if init_method == "mean_of_constituents":
                text = new_tokenizer.convert_tokens_to_string([tok])
                enc = old_tokenizer(text, add_special_tokens=False)
                ids = [i for i in enc.get("input_ids", []) if 0 <= i < old_n]
                if ids:
                    vec_in = in_emb[ids].mean(dim=0)
                    if out_emb is not None: vec_out = out_emb[ids].mean(dim=0)

            in_emb[tid].copy_(vec_in)
            if out_emb is not None and vec_out is not None: out_emb[tid].copy_(vec_out)
            changes["initialized"].append({"id": tid, "token": tok, "method": init_method})

    return changes


# =============================================================================
# 5. BENCHMARKING & PRUNING (benchmarking.py & pruning.py)
# =============================================================================
def find_unreachable_tokens_merges(tokenizer: PreTrainedTokenizerBase) -> list[str]:
    vocab, merges = _get_bpe_state(tokenizer.backend_tokenizer.to_str())
    vocab_tokens = set(vocab.keys())
    
    merge_pairs, merge_outputs = [], set()
    for rule in merges:
        parts = rule.split(" ")
        if len(parts) == 2:
            out = parts[0] + parts[1]
            merge_pairs.append((parts[0], parts[1], out))
            merge_outputs.add(out)

    reachable = set(vocab_tokens - merge_outputs) # Leaves
    changed = True
    while changed:
        changed = False
        for a, b, out in merge_pairs:
            if out not in reachable and a in reachable and b in reachable and out in vocab_tokens:
                reachable.add(out)
                changed = True

    return sorted(vocab_tokens - reachable)

class BasePruner(ABC):
    @abstractmethod
    def train(self, tokenizer: PreTrainedTokenizerBase, corpus: Iterable[str] | None = None) -> None: ...
    @abstractmethod
    def prune(self, tokenizer: PreTrainedTokenizerBase, n_tokens: int) -> None: ...
    @abstractmethod
    def save(self, path: str | Path) -> None: ...

@dataclass
class FrequencyPruner(BasePruner):
    freq: dict[int, int] | None = None
    token_ids_sorted: list[int] | None = None

    def train(self, tokenizer: PreTrainedTokenizerBase, corpus: Iterable[str] | None = None) -> None:
        freq = Counter()
        for text in tqdm(corpus, desc="Computing Pruner Frequencies"):
            freq.update(tokenizer(text, add_special_tokens=False).get("input_ids", []))
        self.freq = dict(freq)
        self.token_ids_sorted = sorted(freq.keys(), key=lambda tid: (freq[tid], tid))

    def prune(self, tokenizer: PreTrainedTokenizerBase, n_tokens: int) -> None:
        self.token_ids_to_prune = self.token_ids_sorted[: int(n_tokens)]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"token_ids_to_prune": self.token_ids_to_prune}, indent=2))


# =============================================================================
# 6. MAIN ORCHESTRATOR
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified Nemotron Tokenizer Extension Toolkit")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--languages", default="hindi,bengali,tamil,telugu")
    parser.add_argument("--samples-per-lang", type=int, default=200000)
    parser.add_argument("--extension-size", type=int, default=64000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save final model/tokenizer/artifacts. If omitted, a new timestamped folder is created.",
    )
    parser.add_argument("--tokenizer-only", action="store_true")
    
    # Advanced features from repo
    parser.add_argument("--keep-added-token-positions", action="store_true", help="Preserve dense relative ID placements.")
    parser.add_argument("--init-method", choices=["mean", "mean_of_constituents"], default="mean_of_constituents")
    parser.add_argument("--benchmark", action="store_true", help="Calculate and print unreachable tokens.")
    parser.add_argument("--prune-size", type=int, default=0, help="If >0, calculates the lowest-frequency N tokens to prune downstream.")

    args = parser.parse_args()
    if not args.out_dir:
        script_dir = Path(__file__).resolve().parent
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = str(script_dir / "outputs" / f"continued_bpe_{stamp}")
    os.makedirs(args.out_dir, exist_ok=True)

    langs = [l.strip() for l in args.languages.split(",")]
    total_docs = len(langs) * args.samples_per_lang
    devanagari_norm = get_devanagari_normalizer()

    logger.info("Loading Base Tokenizer with Mistral Regex Fix...")
    base_tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True, fix_mistral_regex=True)

    # STEP 1: Stream and Train Temporary Artifacts
    logger.info(f"Phase 1: Streaming ~{total_docs} docs to learn optimal subwords...")
    with tqdm(total=total_docs, desc="Corpus Streaming", unit="docs") as pbar:
        stream = mixed_language_text_stream(langs, args.samples_per_lang, devanagari_norm, pbar)
        trained_tok = base_tok.train_new_from_iterator(batch_iterator(stream, args.batch_size), vocab_size=len(base_tok) + args.extension_size)
    
    logger.info("Phase 2: Extracting Diff Artifacts...")
    artifacts = compute_continued_bpe_artifacts(base_tok.backend_tokenizer, trained_tok.backend_tokenizer)
    
    # Save artifacts directly mimicking the repo
    merges_pairs = [tuple(x.split(" ")) for x in artifacts.new_merges if len(x.split(" ")) == 2]
    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f: json.dump(artifacts.new_vocab, f, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "merges.json"), "w") as f: json.dump(artifacts.new_merges, f, ensure_ascii=False)

    # STEP 2: Extend Tokenizer
    logger.info("Phase 3: Splicing merges into base tokenizer...")
    expanded_tok = extend_tokenizer(
        base_tok, artifacts.new_vocab, merges_pairs, 
        n_tokens=args.extension_size, keep_added_token_positions=args.keep_added_token_positions
    )
    expanded_tok.save_pretrained(args.out_dir)

    # STEP 3: Benchmarking (Optional)
    if args.benchmark:
        logger.info("Phase 3.5: Benchmarking Unreachable Graph Tokens...")
        unreachable = find_unreachable_tokens_merges(expanded_tok)
        logger.info(f"Found {len(unreachable)} mathematically unreachable tokens in the BPE graph.")
        with open(os.path.join(args.out_dir, "unreachable.json"), "w") as f: json.dump(unreachable, f, ensure_ascii=False)

    # STEP 4: Pruning (Optional)
    if args.prune_size > 0:
        logger.info(f"Phase 3.5: Calculating {args.prune_size} tokens to prune via FrequencyPruner...")
        pruner = FrequencyPruner()
        # We need a small fresh stream to count frequencies
        prune_stream = mixed_language_text_stream(langs, min(args.samples_per_lang, 10000), devanagari_norm, None)
        pruner.train(expanded_tok, list(prune_stream))
        pruner.prune(expanded_tok, args.prune_size)
        pruner.save(os.path.join(args.out_dir, "pruned_tokens.json"))
        logger.info("Pruned tokens list saved (Downstream scripts must filter these IDs during CPT).")

    if args.tokenizer_only:
        logger.info("Tokenizer-only mode active. Exiting before Model initialization.")
        return

    # STEP 5: Modify Embeddings
    logger.info(f"Phase 4: Loading Model & Applying '{args.init_method}' embeddings...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    
    modify_embeddings(model, old_tokenizer=base_tok, new_tokenizer=expanded_tok, init_method=args.init_method)
    
    model.save_pretrained(args.out_dir)
    logger.info(f"Success! Model and Tokenizer fully extended and saved to: {args.out_dir}")

if __name__ == "__main__":
    main()