#!/usr/bin/env python3
"""
Fast tokenizer extension script for Indic languages.

What this version does:
- Streams BhashaKritika from Hugging Face.
- Stops as soon as it collects K accepted samples per language.
- Deduplicates lines.
- Mines candidate words in the target script.
- Scores all candidate words against the *base* tokenizer first
  (avoids language-order dependence).
- Adds all selected tokens once at the end.

Example:
    python fix_tokenizer_update_fast.py --dry-run
    python fix_tokenizer_update_fast.py \
        --languages hindi tamil \
        --corpus-top-k 50000 \
        --max-new-tokens-per-lang 8000 \
        --out-dir ./nemotron_nano_tokenizer_indic_extended
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


LANG_TO_CONFIG = {
    "hindi": "hindi",
    "bengali": "bengali",
    "tamil": "tamil",
    "telugu": "telugu",
}

SCRIPT_RANGES = {
    "hindi": (0x0900, 0x097F),    # Devanagari
    "bengali": (0x0980, 0x09FF),  # Bengali
    "tamil": (0x0B80, 0x0BFF),    # Tamil
    "telugu": (0x0C00, 0x0C7F),   # Telugu
}

# Strip punctuation/symbols from edges, but keep letters/numbers/marks inside.
EDGE_JUNK_RE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)


def _script_range_for_language(lang: str) -> tuple[int, int]:
    try:
        return SCRIPT_RANGES[lang.lower()]
    except KeyError:
        raise ValueError(f"Unknown language: {lang!r}")


def _safe_int(x: Any, default: int = 0) -> int:
    if x is None:
        return default
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if math.isnan(x):
            return default
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return default
        try:
            return int(float(s))
        except Exception:
            return default
    return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        try:
            v = float(x)
            return default if math.isnan(v) else v
        except Exception:
            return default
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return default
        try:
            v = float(s)
            return default if math.isnan(v) else v
        except Exception:
            return default
    return default


def _as_json_dict(x: Any) -> dict[str, Any]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _parse_flags(x: Any) -> dict[str, Any]:
    return _as_json_dict(x)


def _extract_response_text(row: dict[str, Any]) -> str:
    # Tries a few likely text keys used by datasets of this style.
    for key in ("response", "text", "output", "content", "answer"):
        val = row.get(key)
        if isinstance(val, str):
            s = val.strip()
            if s:
                return s

    # Some datasets keep conversation turns as nested objects/lists.
    conv = row.get("messages") or row.get("conversation")
    if isinstance(conv, list):
        parts: list[str] = []
        for item in conv:
            if isinstance(item, dict):
                for k in ("content", "text", "value"):
                    v = item.get(k)
                    if isinstance(v, str) and v.strip():
                        parts.append(v.strip())
                        break
        if parts:
            return "\n".join(parts)

    return ""


def _row_passes_hard_filters(flags: dict[str, Any]) -> bool:
    # Keep this permissive and robust to schema drift.
    # Reject only when a known bad flag is explicitly true.
    known_bad = [
        "is_garbled",
        "garbled",
        "is_toxic",
        "toxic",
        "is_repetitive",
        "repetitive",
        "is_low_quality",
        "low_quality",
    ]
    for key in known_bad:
        if flags.get(key) is True:
            return False
    return True


def _normalize_word(w: str) -> str:
    w = unicodedata.normalize("NFC", w).strip()
    w = EDGE_JUNK_RE.sub("", w)
    return w.strip()


def _char_in_script(ch: str, lo: int, hi: int) -> bool:
    code = ord(ch)
    return lo <= code <= hi


def _is_word_dominantly_in_script(w: str, lo: int, hi: int, threshold: float = 0.7) -> bool:
    """
    Accept the word if at least threshold fraction of letter/mark characters
    belong to the target script. Digits/punctuation are ignored for the ratio.
    """
    total = 0
    in_script = 0
    for ch in w:
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("M"):
            total += 1
            if _char_in_script(ch, lo, hi):
                in_script += 1

    if total == 0:
        return False
    return (in_script / total) >= threshold


def _has_script_text(text: str, lang: str) -> bool:
    lo, hi = _script_range_for_language(lang)
    return any(_char_in_script(ch, lo, hi) for ch in text)


def _select_first_k_texts_streaming(
    row_iter: Iterator[dict[str, Any]],
    k: int,
    *,
    min_word_count: int,
    min_chars: int,
) -> list[str]:
    """
    Fast mode:
    - stream rows
    - apply filters
    - keep accepted unique texts
    - stop as soon as k are collected
    """
    if k <= 0:
        return []

    out: list[str] = []
    seen: set[str] = set()

    for row in row_iter:
        flags = _parse_flags(row.get("flags"))
        if not _row_passes_hard_filters(flags):
            continue

        text = _extract_response_text(row)
        if not text or len(text) < min_chars:
            continue

        wc = _safe_int(row.get("word_count"), 0)
        if wc > 0 and wc < min_word_count:
            continue

        if text in seen:
            continue

        out.append(text)
        seen.add(text)

        if len(out) >= k:
            break

    return out


def _fetch_first_k_from_hub(
    lang: str,
    k: int,
    *,
    min_word_count: int,
    min_chars: int,
) -> list[str]:
    from datasets import load_dataset

    cfg = LANG_TO_CONFIG[lang.lower()]
    ds = load_dataset(
        "krutrim-ai-labs/BhashaKritika",
        name=cfg,
        split="train",
        streaming=True,
    )
    return _select_first_k_texts_streaming(
        iter(ds),
        k,
        min_word_count=min_word_count,
        min_chars=min_chars,
    )


def _mine_candidates_for_lang(
    lines: list[str],
    lang: str,
    *,
    min_freq: int,
    max_word_chars: int,
) -> Counter[str]:
    lo, hi = _script_range_for_language(lang)

    ctr: Counter[str] = Counter()
    for line in lines:
        for raw in line.split():
            w = _normalize_word(raw)
            if not w:
                continue
            if len(w) > max_word_chars:
                continue
            if not _is_word_dominantly_in_script(w, lo, hi):
                continue
            ctr[w] += 1

    return Counter({w: c for w, c in ctr.items() if c >= min_freq})


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fast early-stop tokenizer extension for Indic languages."
    )
    p.add_argument(
        "--model-id",
        type=str,
        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
        help="Base tokenizer/model id on the Hugging Face Hub.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "nemotron_nano_tokenizer_indic_extended",
        help="Directory for save_pretrained(tokenizer).",
    )
    p.add_argument(
        "--languages",
        nargs="*",
        default=sorted(LANG_TO_CONFIG.keys()),
        help="Languages to extend (default: hindi bengali tamil telugu).",
    )
    p.add_argument(
        "--max-new-tokens-per-lang",
        type=int,
        default=16_384,
        help="Max new strings to select per language.",
    )
    p.add_argument(
        "--corpus-top-k",
        type=int,
        default=150_000,
        help="Number of accepted Hub samples to collect per language (stops early after k).",
    )
    p.add_argument(
        "--min-freq",
        type=int,
        default=3,
        help="Minimum corpus frequency for a word to be considered.",
    )
    p.add_argument(
        "--min-subwords",
        type=int,
        default=2,
        help="Only add words that encode to at least this many ids (base tokenizer, add_special_tokens=False).",
    )
    p.add_argument(
        "--max-word-chars",
        type=int,
        default=48,
        help="Skip words longer than this (characters).",
    )
    p.add_argument(
        "--min-word-count",
        type=int,
        default=32,
        help="Minimum dataset word_count metadata (when parseable).",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=120,
        help="Minimum response character length.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats only; do not modify tokenizer or write files.",
    )
    args = p.parse_args()

    try:
        from transformers import AutoTokenizer, AddedToken
    except ImportError:
        print("Install: pip install transformers", file=sys.stderr)
        raise SystemExit(1)

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("Install: pip install datasets", file=sys.stderr)
        raise SystemExit(1)

    langs = [lang.lower() for lang in args.languages]
    for lang in langs:
        if lang not in LANG_TO_CONFIG:
            print(
                f"Unknown language {lang!r}; expected one of {sorted(LANG_TO_CONFIG)}",
                file=sys.stderr,
            )
            raise SystemExit(1)

    # Base tokenizer used only for scoring candidates.
    base_tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    per_lang_selected: dict[str, list[tuple[float, str, int, int]]] = {}

    for lang in langs:
        print(f"\n=== {lang} (streaming Hub, stop after {args.corpus_top_k} accepted samples) ===")

        lines = _fetch_first_k_from_hub(
            lang,
            args.corpus_top_k,
            min_word_count=args.min_word_count,
            min_chars=args.min_chars,
        )

        script_hits = sum(1 for ln in lines if _has_script_text(ln, lang))
        print(f"Collected {len(lines)} unique accepted lines ({script_hits} with {lang} script chars)")

        ctr = _mine_candidates_for_lang(
            lines,
            lang,
            min_freq=args.min_freq,
            max_word_chars=args.max_word_chars,
        )
        print(f"Unique candidate words with freq>={args.min_freq}: {len(ctr)}")

        scored: list[tuple[float, str, int, int]] = []
        for w, freq in ctr.items():
            ids = base_tokenizer.encode(w, add_special_tokens=False)
            n = len(ids)
            if n < args.min_subwords:
                continue

            # Estimated token savings if the full word becomes one token.
            savings = freq * (n - 1)
            scored.append((float(savings), w, freq, n))

        scored.sort(reverse=True, key=lambda x: x[0])

        if args.max_new_tokens_per_lang > 0:
            scored = scored[: args.max_new_tokens_per_lang]

        per_lang_selected[lang] = scored
        print(f"Selected {len(scored)} candidate additions for {lang}")

        if args.dry_run:
            for row in scored[:10]:
                score, word, freq, toklen = row
                print(f"  score={score:.0f} freq={freq} toklen={toklen} text={word[:60]!r}")

    if args.dry_run:
        print("\nDry run complete; no tokenizer written.")
        return

    # Fresh tokenizer instance for actual mutation.
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    base_len = len(tokenizer)

    # Flatten candidates in requested language order, preserving order.
    all_candidates: list[str] = []
    seen_global: set[str] = set()
    for lang in langs:
        for _, w, _, _ in per_lang_selected.get(lang, []):
            if w in seen_global:
                continue
            seen_global.add(w)
            all_candidates.append(w)

    # Use AddedToken to reduce accidental partial matching behavior.
    added_tokens = [
        AddedToken(
            w,
            single_word=True,
            lstrip=False,
            rstrip=False,
            normalized=True,
        )
        for w in all_candidates
    ]

    added = tokenizer.add_tokens(added_tokens)

    print(f"\nVocab size: {base_len} -> {len(tokenizer)}")
    print(f"Requested {len(all_candidates)} total strings; tokenizer added {added} new ids.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved tokenizer to {args.out_dir}")
    print(
        "\nNext: load the base model and call "
        "model.resize_token_embeddings(len(tokenizer)), then train new embeddings."
    )


if __name__ == "__main__":
    main()