# Nemotron 30B – Token fertility (Hindi, Bengali, Tamil, Telugu)

Evaluates **token fertility** of [Nemotron 3 Nano 30B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) on four Indic languages using the [Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) parallel corpus (English → Indic).

**Token fertility** = (number of generated translation tokens) / (number of source English tokens).  
Higher values mean the model tends to use more tokens in that language for the same English content.

## Setup

```bash
cd /home/admin/Nemotron30B
pip install -r requirements.txt
```

**Avoid 429 rate limit:** log in or set a token so requests are authenticated:

```bash
huggingface-cli login
# or: export HF_TOKEN=your_read_token
```

## Usage

```bash
python token_fertility.py [OPTIONS]
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--samples` | 500 | Number of sentence pairs per language |
| `--seed` | 42 | Random seed |
| `--output` | fertility_results.json | Output JSON path |
| `--token` | (HF_TOKEN env) | Hugging Face token; avoids 429 rate limit |

Results are written to `fertility_results.json`.

---

## Medical domain (separate script)

**`token_fertility_medical.py`** – Token fertility for **medical** text in **English, Hindi, Bengali, Tamil, Telugu** using [MedMCQA-Indic](https://huggingface.co/datasets/ekacare/MedMCQA-Indic) (same medical MCQA in each language). Fertility = language_tokens / english_tokens. No GPU.

```bash
python token_fertility_medical.py [--samples 1000] [--output fertility_medical_results.json]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--samples` | 1000 | Number of parallel medical samples |
| `--output` | fertility_medical_results.json | Output path |
| `--token` | (HF_TOKEN env) | Hugging Face token |

Results: per-language fertility vs English and avg token counts.

---

## Running on Slurm

No GPU needed (tokenizer only). `sbatch run_slurm.sh` runs the general script.

## Datasets

- **General:** [Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) (English → hi, bn, ta, te).
- **Medical:** [MedMCQA-Indic](https://huggingface.co/datasets/ekacare/MedMCQA-Indic) (en, hi, bn, ta, te).

## Output

- `fertility_results.json`: per-language fertility (target/source), samples, avg tokens.
- `fertility_medical_results.json`: same for medical domain, fertility vs English.
