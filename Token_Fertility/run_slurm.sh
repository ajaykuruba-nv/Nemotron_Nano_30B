#!/bin/bash
#SBATCH --job-name=nemotron-fertility
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=fertility_%j.out

# Tokenizer-only: no GPU. Avoid 429: huggingface-cli login or export HF_TOKEN=...
cd /home/admin/Nemotron30B
pip install -q -r requirements.txt
python token_fertility.py --samples 500 --output fertility_results.json
