# DCA Val/Test Set Evaluation

This folder contains scripts for running DCA inference on the held-out val and test splits and computing the full composite metric. Used to compare human-prompt runs, DSPy-optimized programs, and direct model calls on an equal footing.

All scripts use the same scoring metric (`dca_metric.py`): `0.6 × classification + 0.4 × change summary (LLM judge)`.

---

## Folder Structure

```
dca_run_val_test_set_eval/
├── run_val_test_set_eval_on_human_runs.py      # Evaluate human-prompt runs stored in pair JSONs
├── run_val_test_set_eval_on_optimized_prompt.py # Evaluate a saved DSPy optimized model
├── run_human_prompt_direct_eval.py             # Direct inference eval for a single human prompt
├── run_dspy_mimic_direct_eval.py               # Direct inference eval mimicking DSPy program format
├── run_human_prompt_repeatability.py           # Repeatability study across temperatures/seeds
├── run_cache_grid_eval.py                      # Batch evaluation across many cached DSPy models
├── run_all_human_and_optimized_eval.py         # Orchestrate all human + optimized evals in one pass
└── eval_results/
    ├── human_runs/
    │   └── eval_results_{val|test}.json        # Human run results
    ├── optimized_prompt_runs/
    │   └── eval_results_<run_key>_{val|test}.json
    ├── human_prompt_runs/
    ├── human_prompt_direct_runs/
    └── dspy_mimic_runs/
```

---

## Script Reference

### `run_val_test_set_eval_on_human_runs.py`

Evaluates all human-prompt runs stored as keys inside the pair JSON files (e.g., `v2_prompt_cr_gpt41`, `v3_prompt_cr_gpt52`).

Reads existing model outputs from `data/dca_data/image_pairs_llm_output_exp1/` — no new inference is performed. Computes classification score, LLM-judge change score, and composite score for each pair.

**Output** (written to `eval_results/human_runs/`):
```json
{
  "meta": { "split": "val", "weights": {...}, "runs": [...] },
  "summary": [ { "run_key": "...", "total_score": 0.82, ... } ],
  "details": [ { "run_key": "...", "per_pair": [...] } ]
}
```

**Usage:**
```bash
# Evaluate val set
python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_human_runs.py --which-set val

# Evaluate test set (default)
python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_human_runs.py

# Evaluate both
python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_human_runs.py --which-set both

# Skip LLM judge (fast, rule-based scores only; breakdown cases marked None)
python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_human_runs.py --skip-llm

# Overwrite existing entries
python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_human_runs.py --overwrite
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--which-set` | `test` | `val`, `test`, or `both` |
| `--split-file` | `.cache/dca_split_seed42.json` | Path to the frozen split manifest |
| `--json-dir` | `data/dca_data/image_pairs_llm_output_exp1` | Directory of pair JSON files |
| `--output-dir` | `eval_results/human_runs/` | Where to write result JSON files |
| `--skip-llm` | False | Skip LLM judge for breakdown cases |
| `--overwrite` | False | Re-evaluate runs that already have results |

---

### `run_val_test_set_eval_on_optimized_prompt.py`

Loads a saved DSPy optimized model (`.json` file from `.cache/`) and runs live inference on the val or test split. Computes per-pair and aggregated scores using the same metric as the tournament pipeline.

Captures per-pair: `classification_score`, `change_score`, `total_score`, `response_time_seconds`, `total_tokens`.

Output format matches `run_val_test_set_eval_on_human_runs.py` for side-by-side comparison.

**Usage:**
```bash
# Evaluate on test set
python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py \
    --model-path .cache/gpt-4.1_MIPROv2_DCA_Light_dca_optimized_2026032716.json \
    --model-name azure/gpt-4.1 \
    --split-file .cache/dca_split_seed42.json

# Evaluate on val set
python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py \
    --model-path .cache/my_model.json \
    --model-name azure/gpt-4.1-mini \
    --split-file .cache/dca_split_seed42.json \
    --which-set val

# Evaluate both sets
python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py \
    --model-path .cache/my_model.json \
    --model-name azure/gpt-4.1-mini \
    --split-file .cache/dca_split_seed42.json \
    --which-set both

# Different temperature
python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py \
    --model-path .cache/my_model.json \
    --model-name azure/gpt-4.1-mini \
    --split-file .cache/dca_split_seed42.json \
    --temperature 0.0

# Overwrite an existing result entry
python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py \
    --model-path .cache/my_model.json \
    --model-name azure/gpt-4.1-mini \
    --split-file .cache/dca_split_seed42.json \
    --overwrite
```

**Required arguments:**

| Argument | Description |
|----------|-------------|
| `--model-path` | Path to the saved DSPy program `.json` file |
| `--model-name` | LM identifier for inference (e.g. `azure/gpt-4.1-mini`) |
| `--split-file` | Path to the frozen split manifest (`.cache/dca_split_seed42.json`) |

**Optional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--which-set` | `test` | `val`, `test`, or `both` |
| `--temperature` | `0.1` | Inference temperature |
| `--reasoning-effort` | None | For reasoning models: `low`, `medium`, `high` |
| `--image-dir` | `data/dca_data/image_pairs` | Directory of image pair folders |
| `--gt-dir` | `data/dca_data/image_pairs_llm_output_exp1` | Ground truth JSON directory |
| `--output-dir` | `eval_results/optimized_prompt_runs/` | Where to write results |
| `--overwrite` | False | Re-run if result file already exists |

The `RUN_KEY` and `RUN_LABEL` are automatically derived from `--model-path` and `--model-name` — no manual editing needed.

---

### `run_cache_grid_eval.py`

Batch evaluates many saved DSPy model files from a directory. Useful for comparing many optimization runs at once.

```bash
python evaluation/dca_run_val_test_set_eval/run_cache_grid_eval.py \
    --model-dir .cache/ \
    --model-name azure/gpt-4.1-mini \
    --split-file .cache/dca_split_seed42.json \
    --which-set val
```

---

### `run_human_prompt_direct_eval.py`

Runs live inference for a single human prompt markdown file directly (no stored outputs). Useful for evaluating new prompt versions without going through the full baseline run pipeline.

```bash
python evaluation/dca_run_val_test_set_eval/run_human_prompt_direct_eval.py \
    --prompt evaluation/dca_baseline_runs/human_prompts/v5_prompt_cr.md \
    --model-name azure/gpt-4.1-mini \
    --split-file .cache/dca_split_seed42.json \
    --which-set val
```

---

### `run_human_prompt_repeatability.py`

Runs the same human prompt multiple times at different temperatures or seeds to measure output stability.

```bash
python evaluation/dca_run_val_test_set_eval/run_human_prompt_repeatability.py \
    --prompt evaluation/dca_baseline_runs/human_prompts/v2_prompt_cr.md \
    --model-name azure/gpt-4.1-mini \
    --split-file .cache/dca_split_seed42.json \
    --reps 5 \
    --temperature 0.7
```

---

## Output Format

All result files follow this structure:

```json
{
  "meta": {
    "split": "val",
    "n_samples": 86,
    "weights": { "cls": 0.6, "change": 0.4 },
    "split_file": ".cache/dca_split_seed42.json"
  },
  "summary": [
    {
      "run_key": "my_model_run_gpt41mini",
      "run_label": "my_model / azure/gpt-4.1-mini",
      "total_score": 0.847,
      "cls_score": 0.919,
      "change_score": 0.739,
      "total_tokens_avg": 2847.3,
      "response_time_avg": 3.21
    }
  ],
  "details": [
    {
      "run_key": "my_model_run_gpt41mini",
      "per_pair": [
        {
          "pair_id": "pair_001",
          "classification_score": 1.0,
          "change_score": 0.85,
          "total_score": 0.94,
          "response_time_seconds": 2.7,
          "total_tokens": 2941
        }
      ]
    }
  ]
}
```

---

## Data and Split

All scripts use the frozen split manifest `.cache/dca_split_seed42.json` (created on first run of `extract_dca.py` or the tournament pipeline). This ensures every script evaluates on exactly the same val and test pairs.

- Val set: ~86 pairs (20% of 429 total)
- Test set: ~86 pairs (20% of 429 total)
- Train set: ~257 pairs (60% of 429 total)

Stratification: `expected_discard_image` × `human_label_category` × `doubt`.

---

## See Also

- [DCA Metric](../README.md) — composite scoring metric details
- [DCA Baseline Runs](../dca_baseline_runs/README.md) — running and evaluating human prompts
- [Tournament Pipeline](../../dca_tournaments/README.md) — automated multi-stage optimization
