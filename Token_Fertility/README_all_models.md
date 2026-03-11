# Token fertility – all models

## Reference

- **Single-model script:** `token_fertility.py`  
  Same fertility logic (T/W = tokens per word), one tokenizer, Samanantar (hi, bn, ta, te).

## Multi-model script

- **Script:** `token_fertility_all_models.py`  
  Runs token fertility for **multiple models** on the **same dataset subset** (same seed/samples) so results are comparable.

### Models (default)

| Display name   | Model ID |
|----------------|----------|
| Aya Fire       | `CohereLabs/tiny-aya-fire` |
| Nemotron       | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Sarvam M       | `sarvamai/sarvam-m` |
| GPT-OSS-20B    | `openai/gpt-oss-20b` |
| Gemma 3 4B     | `google/gemma-3-4b-it` |
| Qwen 3.5       | `Qwen/Qwen3.5-397B-A17B` |

### Usage

```bash
cd /home/admin/Nemotron30B/Token_Fertility

# All models (default 500 samples per language)
python token_fertility_all_models.py --output fertility_all_models_results.json

# Fewer samples
python token_fertility_all_models.py --samples 200

# Only specific models
python token_fertility_all_models.py --models "Nemotron" "Gemma 3 4B"

# With HF token (avoid 429)
export HF_TOKEN=your_token
python token_fertility_all_models.py
```

### Output

`fertility_all_models_results.json` (or `--output` path):

- `reference_script`, `dataset`, `fertility_definition`, `languages`, `samples_per_lang`, `seed`
- `models`: for each model, `model_id` and `results` (per language: `fertility`, `samples`, `avg_tokens`, `avg_words`)

### Calculation (same as `token_fertility.py`)

- **Corpus-level fertility** = **sum(T_i) / sum(W_i)** over samples (not mean of per-sample T/W).
- **Words** = `len([w for w in text.split() if w.strip()])`.
- **Tokens** = `len(tokenizer.encode(text, add_special_tokens=False))`.
- One dataset subset per language (fixed by `--seed`), reused for every model.
- **Sarvam M:** tokenizer loaded with `fix_mistral_regex=True`.

### Verify target field

To confirm `tgt` is the intended Indic text per language (and not English):

```bash
python token_fertility_all_models.py --verify-sample
```

Prints one row per language and exits; no model evaluation.
