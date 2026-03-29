# Nemotron tokenizer expansion (BhashaKritika)

This folder expands the **NVIDIA Nemotron 3 Nano 30B** tokenizer using the Hugging Face dataset [`krutrim-ai-labs/BhashaKritika`](https://huggingface.co/datasets/krutrim-ai-labs/BhashaKritika). The Python entry point is `expand_nemotron_bhashakritika.py`.

## Prerequisites

- Accept the dataset license on Hugging Face (if prompted).
- Authenticate when needed: `huggingface-cli login`
- Optional: for Hindi/Marathi Devanagari normalization, install [Indic NLP](https://github.com/anoopkunchukuttan/indic_nlp_library) resources and set `INDIC_RESOURCES_PATH` if your environment requires it.

## Install dependencies

```bash
cd /home/admin/nvidia/Nemotron_Nano_30B/Tokenizer
pip install -r requirements.txt
```

The training code uses the `tokenizers` library (typically installed with `transformers`). If imports fail, run: `pip install tokenizers`.

## Run the Python script (foreground)

```bash
cd /home/admin/nvidia/Nemotron_Nano_30B/Tokenizer

# Tokenizer only (recommended first; avoids loading the 30B weights)
python expand_nemotron_bhashakritika.py --tokenizer-only

# Full pipeline: load model, resize embeddings, mean-init new rows, save
python expand_nemotron_bhashakritika.py
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-id` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | Base model (tokenizer + optional weights). |
| `--output-dir` | `./nemotron-indic-expanded` | Where the tokenizer (and model, if not `--tokenizer-only`) is saved. |
| `--languages` | `hindi,bengali,tamil,telugu` | Comma-separated BhashaKritika config names (e.g. `marathi`, `punjabi`, `gujarati`, `malayalam`, `telugu`). |
| `--new-tokens` | `32000` | Total new subwords to learn across **all** selected languages (joint BPE). |
| `--samples-per-lang` | `150000` | Cap on streamed training documents **per** language. |
| `--tokenizer-only` | off | Save only the expanded tokenizer; do not load or save the full model. |

Examples:

```bash
python expand_nemotron_bhashakritika.py --tokenizer-only --languages hindi,bengali --new-tokens 16000
python expand_nemotron_bhashakritika.py --tokenizer-only --output-dir ./my-tokenizer
```

## Run in the background (survive SSH disconnect)

Use `run_expand.sh` so the job keeps running after you close the session **as long as the cluster allows `nohup` jobs on that node** (see Slurm note below).

```bash
cd /home/admin/nvidia/Nemotron_Nano_30B/Tokenizer
chmod +x run_expand.sh   # once

# Background (default): arguments are forwarded to the Python script
./run_expand.sh --tokenizer-only
./run_expand.sh --tokenizer-only --languages hindi,bengali,tamil --new-tokens 32000
```

### Foreground (no `nohup`)

```bash
./run_expand.sh fg --tokenizer-only
```

### Check progress after you log back in

```bash
# Stream the latest log
tail -f /home/admin/nvidia/Nemotron_Nano_30B/Tokenizer/logs/latest.log

# Or print the log path
/home/admin/nvidia/Nemotron_Nano_30B/Tokenizer/run_expand.sh logs

# See whether the PID is still running
/home/admin/nvidia/Nemotron_Nano_30B/Tokenizer/run_expand.sh status
```

Logs are written under `logs/expand_YYYYMMDD_HHMMSS.log`; `logs/latest.log` is a symlink to the most recent run. The process ID is stored in `expand_nemotron.pid`.

### Help for the shell wrapper

```bash
./run_expand.sh --help
```

## Slurm / login node warning

Many HPC sites **kill long or heavy processes on login nodes**, even with `nohup`. If your job stops after logout, submit it with **`sbatch`** (or `srun` on a compute node) instead, and point Slurm `#SBATCH -o` / `-e` at a log file you can `tail -f` after reconnecting.

## Outputs

- With `--tokenizer-only`: tokenizer files under `--output-dir` (default `./nemotron-indic-expanded`).
- Without it: model weights and tokenizer under `--output-dir` (large; needs sufficient GPU/RAM).
