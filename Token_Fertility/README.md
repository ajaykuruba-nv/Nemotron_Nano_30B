# Nemotron 30B — Token fertility (Indic)

Scripts measure **token fertility** on **Hindi, Bengali, Tamil, and Telugu** text using [Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) (Indic sides of the parallel corpus). **No GPU** is required: only tokenizers and datasets.

**Fertility (as implemented here)** = **sum(tokens) / sum(words)** over sampled lines, where **tokens** come from `tokenizer.encode(..., add_special_tokens=False)` and **words** come from whitespace splitting of the Indic `tgt` (or equivalent) field. Lower values usually mean fewer subwords per surface word for that tokenizer on that text.

## Setup

```bash
cd /path/to/Nemotron_Nano_30B/Token_Fertility
pip install -r requirements.txt
```

Avoid Hub rate limits:

```bash
huggingface-cli login
# or: export HF_TOKEN=your_read_token
```

---

## Single model: `token_fertility.py`

Nemotron 3 Nano tokenizer only ([`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)).

```bash
python3 token_fertility.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--samples` | 500 | Random training rows per language |
| `--seed` | 42 | Sampling seed |
| `--output` | `fertility_results.json` | Output JSON (relative paths resolve next to this script) |
| `--token` | `HF_TOKEN` env | Hugging Face token |

---

## Multiple models: `token_fertility_all_models.py`

Compares several Hub tokenizers on the **same** sampled Samanantar subsets.

```bash
python3 token_fertility_all_models.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--samples` | 500 | Rows per language |
| `--seed` | 42 | Sampling seed |
| `--output` | `fertility_all_models_results.json` | Output JSON |
| `--token` | `HF_TOKEN` env | Hugging Face token |
| `--models` | all configured | Subset of model **keys**, e.g. `--models Nemotron "Sarvam 30B"` |
| `--verify-sample` | off | Print one raw row + extracted text per language, then exit |
| `--nemotron-extended-only` | off | Run **only** Nemotron, loading the **local extended** tokenizer (ignores `--models`) |
| `--nemotron-tokenizer-path` | (see below) | Directory from `save_pretrained()` for the extended tokenizer |

When `--nemotron-extended-only` is set, the default tokenizer path is the expanded Nemotron save directory:

`../Tokenizer/nemotron-indic-expanded`

(relative to the **repo root** `Nemotron_Nano_30B`, i.e. output of `Tokenizer/expand_nemotron_bhashakritika.py`). Override with `--nemotron-tokenizer-path /your/path`.

The results JSON includes `tokenizer_source: "local_extended"` and `tokenizer_path` for that run.

**Example — extended Nemotron tokenizer only:**

```bash
python3 token_fertility_all_models.py --nemotron-extended-only --samples 1000
```

**Example — subset of Hub models:**

```bash
python3 token_fertility_all_models.py --models Nemotron "Mistral Nemo Base 2407"
```

---

## Medical domain: `token_fertility_medical.py`

Token fertility for **medical** text in **English, Hindi, Bengali, Tamil, Telugu** using [MedMCQA-Indic](https://huggingface.co/datasets/ekacare/MedMCQA-Indic). Fertility compares Indic tokenization to English on parallel items. No GPU.

```bash
python3 token_fertility_medical.py [--samples 1000] [--output fertility_medical_results.json]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--samples` | 1000 | Parallel medical samples |
| `--output` | `fertility_medical_results.json` | Output path |
| `--token` | `HF_TOKEN` env | Hugging Face token |

---

## Extended Nemotron tokenizer (Indic)

To **build** the extended tokenizer used with `--nemotron-extended-only`, run the script in the sibling folder:

- [`../tokenizer/README.md`](../tokenizer/README.md) — `extend_nemotron_tokenizer_indic.py`, BhashaKritika, and embedding resize notes.

---

## Running on Slurm

Tokenizer-only jobs fit CPU nodes. Example: `sbatch run_slurm.sh` (adjust partition, time, and account for your site).

---

## Datasets

- **General:** [Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) (`hi`, `bn`, `ta`, `te` configs).
- **Medical:** [MedMCQA-Indic](https://huggingface.co/datasets/ekacare/MedMCQA-Indic).
- **Tokenizer extension corpus:** [BhashaKritika](https://huggingface.co/datasets/krutrim-ai-labs/BhashaKritika) (used under `../tokenizer/`, not in these fertility scripts).

## Output files

- `fertility_results.json` — single-model (Nemotron) per-language fertility.
- `fertility_all_models_results.json` — per-model, per-language fertility and averages.
- `fertility_medical_results.json` — medical domain, vs English baseline where applicable.
