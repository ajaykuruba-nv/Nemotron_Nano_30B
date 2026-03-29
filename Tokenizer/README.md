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

## Quick start (`run_expand.sh`)

From `/home/admin/nvidia/Nemotron_Nano_30B/Tokenizer`:

| Goal | Command |
|------|---------|
| **Quick test** (small sample, tokenizer only, logs to your terminal) | `./run_expand.sh fg --test --tokenizer-only` |
| Same test in **background** | `./run_expand.sh --test --tokenizer-only` |
| **Full run** (production settings, **background** + `nohup`, survives SSH disconnect if the node allows it) | `./run_expand.sh` |

- **`--test`** is interpreted only by **`run_expand.sh`**: it turns into `--samples-per-lang 1000` so the pipeline finishes quickly.
- **Full `./run_expand.sh`** does **not** pass `--tokenizer-only`, so it runs the **full** Python path: expand tokenizer, load the **30B** weights, resize embeddings, and save model + tokenizer (high RAM/VRAM). For a long tokenizer-only job in the background, use `./run_expand.sh --tokenizer-only` instead.
- If you omit **`--languages`**, the script injects `hindi,bengali,tamil,telugu` (no spaces) so arguments are not split wrongly in bash.

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
| `--trust-remote-code` / `--no-trust-remote-code` | on | Nemotron’s Hub repo uses custom modeling code; keep **on** so `nohup`/Slurm jobs do not hang on an interactive prompt. |

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

# Full production run (default languages, full model save) — same as Quick start table
./run_expand.sh

# Tokenizer-only in background (no 30B load)
./run_expand.sh --tokenizer-only

# Optional: override languages or token budget (comma-separated, no spaces)
./run_expand.sh --tokenizer-only --languages hindi,bengali,tamil --new-tokens 32000
```

### Foreground (no `nohup`)

```bash
# Quick test (see Quick start)
./run_expand.sh fg --test --tokenizer-only

# Foreground full pipeline (still loads 30B)
./run_expand.sh fg

# Foreground tokenizer-only
./run_expand.sh fg --tokenizer-only
```

### `run_expand.sh` flags (wrapper)

| Flag / behavior | Meaning |
|-----------------|--------|
| `--test` | Shell-only: adds `--samples-per-lang 1000` for a fast dry run. |
| (no `--languages`) | Injects `--languages hindi,bengali,tamil,telugu`. |
| `fg` / `foreground` | Run in the foreground instead of `nohup`. |

### Check progress after you log back in

```bash
# Stream the latest log
tail -f /home/admin/nvidia/Nemotron_Nano_30B/Tokenizer/logs/latest.log

# Or print the log path
/home/admin/nvidia/Nemotron_Nano_30B/Tokenizer/run_expand.sh logs
```

Logs are written under `logs/expand_YYYYMMDD_HHMMSS.log`; `logs/latest.log` is a symlink to the most recent run. The process ID is stored in `expand_nemotron.pid`.

### Is the job still running?

```bash
cd /home/admin/nvidia/Nemotron_Nano_30B/Tokenizer

# Wrapper (shows PID, elapsed time, command if alive)
./run_expand.sh status

# Same PID file, manual check
test -f expand_nemotron.pid && ps -p "$(cat expand_nemotron.pid)" -o pid,etime,cmd

# Find the Python process by script name (if you lost the PID file)
pgrep -af "expand_nemotron_bhashakritika.py"
```

If you submitted via **Slurm**, use `squeue -u "$USER"` or `sacct -j <jobid>` instead.

### Stop or kill the job

Graceful stop (asks the process to exit; prefer this first):

```bash
cd /home/admin/nvidia/Nemotron_Nano_30B/Tokenizer
kill "$(cat expand_nemotron.pid)"
```

If it does not exit within a short wait, force kill:

```bash
kill -9 "$(cat expand_nemotron.pid)"
```

If `expand_nemotron.pid` is missing or wrong, use `pgrep` to find the PID, then `kill <pid>` or `kill -9 <pid>`.

```bash
pgrep -af "expand_nemotron_bhashakritika.py"
kill <PID_FROM_ABOVE>
```

### Help for the shell wrapper

```bash
./run_expand.sh --help
```

## Slurm / login node warning

Many HPC sites **kill long or heavy processes on login nodes**, even with `nohup`. If your job stops after logout, submit it with **`sbatch`** (or `srun` on a compute node) instead, and point Slurm `#SBATCH -o` / `-e` at a log file you can `tail -f` after reconnecting.

## Outputs

- With `--tokenizer-only`: tokenizer files under `--output-dir` (default `./nemotron-indic-expanded`).
- Without it: model weights and tokenizer under `--output-dir` (large; needs sufficient GPU/RAM).

## Token fertility check (extended Nemotron tokenizer)

From the repo, run `Token_Fertility/token_fertility_all_models.py` against the saved tokenizer (default path matches `./nemotron-indic-expanded`):

```bash
cd /home/admin/nvidia/Nemotron_Nano_30B/Token_Fertility
python3 token_fertility_all_models.py --nemotron-extended-only \
  --nemotron-tokenizer-path ../Tokenizer/nemotron-indic-expanded
```

If you used a different `--output-dir`, pass that path instead. With only `--nemotron-extended-only`, the script uses `../Tokenizer/nemotron-indic-expanded` by default.
