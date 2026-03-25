# Nemotron 3 Nano — Indic tokenizer extension

This folder builds an **extended Hugging Face tokenizer** for
[`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16)
by adding frequent **multi-piece** words in **Hindi, Bengali, Tamil, and Telugu**, mined from synthetic Indic pretraining text.

The base checkpoint uses a large SentencePiece-style vocabulary (~131k). This workflow **adds** tokens instead of replacing the tokenizer from scratch, so vocabulary layout stays compatible with the Hub model until you resize embeddings and train the new rows.

## Data

- **Corpus:** [`krutrim-ai-labs/BhashaKritika`](https://huggingface.co/datasets/krutrim-ai-labs/BhashaKritika) (configs `hindi`, `bengali`, `tamil`, `telugu`).
- **Streaming:** rows are read with `streaming=True`; the script stops after collecting enough accepted lines per language (see `--corpus-top-k`).
- **Filtering:** responses must pass basic checks (`flags` must not mark obvious bad content; minimum length / `word_count` when present). Lines are **deduplicated** while streaming.

## Setup

```bash
cd /path/to/Nemotron_Nano_30B/tokenizer
pip install -r requirements.txt
```

Authenticate to the Hub if downloads are rate-limited:

```bash
huggingface-cli login
# or: export HF_TOKEN=...
```

## Script: `extend_nemotron_tokenizer_indic.py`

1. Loads the **base** tokenizer once to **score** candidate words (token savings if a whole word became one token).
2. Streams BhashaKritika per language until `--corpus-top-k` unique accepted lines are collected.
3. Mines words that are **mostly** in that language’s Unicode script.
4. Picks up to `--max-new-tokens-per-lang` strings per language (by savings), then adds them in one batch via `AddedToken(..., single_word=True, ...)` and `save_pretrained`.

### Common options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-id` | `nvidia/...-Base-BF16` | Hub id for the base tokenizer. |
| `--out-dir` | `./nemotron_nano_tokenizer_indic_extended` | Output directory for `save_pretrained`. |
| `--languages` | all four | Subset, e.g. `hindi tamil`. |
| `--corpus-top-k` | `150000` | Target number of accepted unique lines per language (early stop). |
| `--max-new-tokens-per-lang` | `16384` | Cap on new strings selected per language. |
| `--min-freq` | `3` | Minimum word count in the mined corpus. |
| `--min-subwords` | `2` | Only add if base tokenizer splits into at least this many ids. |
| `--max-word-chars` | `48` | Skip longer words. |
| `--min-word-count` | `32` | Minimum dataset `word_count` when parseable. |
| `--min-chars` | `120` | Minimum response length. |
| `--dry-run` | off | Print stats only; do not write a tokenizer. |

### Examples

```bash
# Quick sanity check
python3 extend_nemotron_tokenizer_indic.py --dry-run --corpus-top-k 5000 --languages hindi

# Full run (long-running; use Slurm or nohup on clusters)
python3 extend_nemotron_tokenizer_indic.py \
  --corpus-top-k 150000 \
  --max-new-tokens-per-lang 16384 \
  --out-dir ./nemotron_nano_tokenizer_indic_extended
```

## After saving: model embeddings

New token rows are **untrained** until you continue pretrain or fine-tune. Load the base model, load your saved tokenizer, then resize:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "./nemotron_nano_tokenizer_indic_extended",
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
    trust_remote_code=True,
)
model.resize_token_embeddings(len(tok))
```

## Measuring token fertility (before / after)

From the sibling **`Token_Fertility`** folder, run multi-model fertility on Samanantar and compare the Hub Nemotron tokenizer to your extended directory:

```bash
cd ../Token_Fertility
python3 token_fertility_all_models.py --nemotron-extended-only
```

See [`../Token_Fertility/README.md`](../Token_Fertility/README.md) for flags (`--nemotron-tokenizer-path`, `--samples`, etc.).
