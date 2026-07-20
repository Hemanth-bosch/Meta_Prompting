# DCA Human Prompt Baseline Runs — Guide

This directory contains scripts for running LLM inference on DCA document image pairs using human-authored prompts, and evaluating the results with classification and change-summary metrics.

---

## Overview

The human prompts pipeline runs inference on all image pairs using different prompt versions and LLM models, storing outputs back into the per-pair JSON files. Separate evaluation scripts then compute classification accuracy, change-summary quality, and token usage statistics.

```
evaluation/dca_baseline_runs/
├── human_prompts/           # Prompt markdown files (v0–v5)
├── run_human_prompts/       # Inference scripts
│   ├── run_with_human_prompts.py              # Single run
│   ├── run_all_experiments.py                 # Batch orchestration
│   ├── run_v4_human_prompt.py                 # V4 multi-step pipeline
│   └── run_support_prompt_for_categorization_of_data.py  # Label coding
└── eval/
    ├── discard_image_classification_eval/
    │   ├── run_eval/eval_human_runs.py        # Classification metrics + plots
    │   └── eval_results/                      # Generated outputs
    └── change_summary_eval/
        ├── run_eval/midas_summary_eval.py     # LLM judge + semantic similarity
        ├── run_eval/visualize_all_metrics.py  # Heatmap + bar charts
        ├── run_eval/fbeta_quartile_analysis.py # F-beta quartile breakdown
        └── eval_results/                      # Generated outputs
```

---

## Prompt Files

Located in `human_prompts/`:

| File | Description |
|------|-------------|
| `v0_prompt_midas.md` | Original MIDAS prompt |
| `v1_prompt_midas.md` | Adapted to latest rules and decision table after rounds of discussion |
| `v2_prompt_cr.md` | Decision tree style format (LLM-assisted, human-reviewed) |
| `v3_prompt_cr.md` | Same as v2 but adds an extra `thinking_area` key in the JSON output (not reasoning tokens — just an output field instructing the model to explain its reasoning in the JSON) |
| `v4_prompt_cr.md` | Multi-step triage + diff pipeline — uses `run_v4_human_prompt.py` |
| `v5_prompt_cr.md` | Latest version — updated rules and formatting |
| `support_prompt_for_categorization_of_data.md` | Maps `human_label_discard_image` + `human_label_changes` to category codes A1/A2/B1/B2/B3/B4/B5a/B5b |

---

## Prerequisites

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate         # Linux/Mac
   .venv\Scripts\activate            # Windows cmd
   & ".venv\Scripts\Activate.ps1"    # Windows PowerShell
   ```

2. Set Azure credentials (or use `az login`):
   ```powershell
   $env:AZURE_OPENAI_ENDPOINT    = "https://your-endpoint.openai.azure.com/"
   $env:AZURE_OPENAI_API_VERSION = "2024-08-01-preview"
   # Then either set AZURE_OPENAI_API_KEY or run: az login
   ```

3. Data must be present:
   - Image pairs: `data/dca_data/image_pairs/pair_NNN/`
   - Ground truth JSON: `data/dca_data/image_pairs_llm_output_exp1/pair_NNN.json`

---

## Running Inference

### Single Run: `run_human_prompts/run_with_human_prompts.py`

Runs one prompt + model combination and writes outputs directly into the pair JSON files under a key derived from `--key-prefix`.

**Syntax:**
```bash
python evaluation/dca_baseline_runs/run_human_prompts/run_with_human_prompts.py \
    --key-prefix <KEY_PREFIX> \
    --prompt <PROMPT_FILE> \
    --deployment <MODEL_NAME> \
    [--no-thinking-area] \
    [--save-full-response] \
    [--verbose]
```

**Required arguments:**

| Argument | Description | Example |
|----------|-------------|---------|
| `--key-prefix` | Key written into pair JSON for this run's outputs | `v2_prompt_cr_gpt41mini` |
| `--prompt` | Prompt filename (looked up in `human_prompts/`) | `v2_prompt_cr.md` |
| `--deployment` | Azure OpenAI deployment name | `gpt-4.1-mini` |

**Optional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--no-thinking-area` | False | Must be passed for v0, v1, v2, v5. Do NOT pass for v3 (v3 outputs an extra `thinking_area` JSON key) |
| `--reasoning-effort` | None | For reasoning models (gpt-5.2, gpt-5-mini): `minimal`, `low`, `medium`, `high` |
| `--save-full-response` | False | Save complete raw API response alongside parsed output |
| `--overwrite` | False | Re-run pairs that already have this key |
| `--start` / `--end` | None (all) | Process only pairs in index range (e.g., `--start 1 --end 50`) |
| `--limit` | None | Max number of pairs to process |
| `--only-doubt` | False | Process only pairs where `doubt: true` |
| `--dry-run` | False | Print what would be run without calling the API |
| `--verbose` | False | Print per-pair inference details |
| `--sleep-seconds` | `0.0` | Delay between API calls (use to avoid rate limits) |
| `--max-retries` | `3` | Retries on transient API errors |
| `--json-dir` | `data/dca_data/image_pairs_llm_output_exp1` | Directory of pair JSON files |
| `--image-pairs-dir` | `data/dca_data/image_pairs` | Directory of image pair folders |

**Examples:**

Run v2 prompt with gpt-4.1-mini on all pairs:
```bash
python evaluation/dca_baseline_runs/run_human_prompts/run_with_human_prompts.py \
    --key-prefix v2_prompt_cr_gpt41mini \
    --prompt v2_prompt_cr.md \
    --deployment gpt-4.1-mini \
    --no-thinking-area \
    --verbose \
    --save-full-response
```

Run v3 prompt with gpt-5.2 (reasoning model, note: no `--no-thinking-area`):
```bash
python evaluation/dca_baseline_runs/run_human_prompts/run_with_human_prompts.py \
    --key-prefix v3_prompt_cr_gpt52_reason_low \
    --prompt v3_prompt_cr.md \
    --deployment gpt-5.2 \
    --reasoning-effort low \
    --verbose \
    --save-full-response
```

Run v5 prompt with gpt-4.1:
```bash
python evaluation/dca_baseline_runs/run_human_prompts/run_with_human_prompts.py \
    --key-prefix v5_prompt_cr_gpt41 \
    --prompt v5_prompt_cr.md \
    --deployment gpt-4.1 \
    --no-thinking-area \
    --verbose
```

---

### Batch Runs: `run_human_prompts/run_all_experiments.py`

Sequentially runs all entries in the `RUNS` list defined inside the script, with full logging.

```bash
python evaluation/dca_baseline_runs/run_human_prompts/run_all_experiments.py
```

Edit the `RUNS` list in the script to configure which prompt/model combinations to run. Each entry is a dict with `label` and `args` keys. The script continues on individual run failure and logs all stdout/stderr to `run_all_experiments.log`.

---

### V4 Multi-Step Pipeline: `run_human_prompts/run_v4_human_prompt.py`

A two-step triage + diff pipeline. Step 1 classifies the pair (discard/keep). For "keep" pairs, Step 2 generates the detailed change description. Uses `v4_prompt_cr.md`.

```bash
python evaluation/dca_baseline_runs/run_human_prompts/run_v4_human_prompt.py \
    --deployment gpt-4.1-mini \
    --key-prefix v4_prompt_cr_gpt41mini \
    --verbose
```

---

### Category Coding: `run_human_prompts/run_support_prompt_for_categorization_of_data.py`

Assigns the `human_label_category` code (A1/A2/B1/B2/B3/B4/B5a/B5b) to each pair based on existing `human_label_discard_image` and `human_label_changes` fields, using the LLM to interpret the human labels.

```bash
python evaluation/dca_baseline_runs/run_human_prompts/run_support_prompt_for_categorization_of_data.py \
    --prompt support_prompt_for_categorization_of_data.md \
    --deployment gpt-4.1-mini \
    --verbose
```

---

## Output Format

Each pair JSON (`data/dca_data/image_pairs_llm_output_exp1/pair_NNN.json`) stores:

```json
{
  "doubt": false,
  "human_label_discard_image": true,
  "human_label_changes": "Images are identical - no changes",
  "human_label_category": "A2",
  "is_synthetic": false,

  "v2_prompt_cr_gpt41mini": {
    "discard_image": "true",
    "changes": "Images are identical - no changes",
    "total_tokens": 1843,
    "response_time_seconds": 2.1
  },
  "v3_prompt_cr_gpt52_reason_low": {
    "discard_image": "true",
    "changes": "Images are identical - no changes",
    "thinking_area": "Both images show the same circuit diagram...",
    "total_tokens": 3201,
    "response_time_seconds": 4.7
  }
}
```

---

## Evaluation

After running inference, compute metrics using:

### Classification Evaluation

```bash
python evaluation/dca_baseline_runs/eval/discard_image_classification_eval/run_eval/eval_human_runs.py
```

Outputs to `eval/discard_image_classification_eval/eval_results/`:
- `classification_metrics.png` — grouped bar chart of accuracy / precision / recall / F1 per run
- `confusion_matrices.png` — confusion matrix per run
- `token_usage.png` — token usage comparison
- `response_time.png` — response time comparison
- `eval_summary.csv` — all metrics in CSV

### Change Summary Evaluation

```bash
python evaluation/dca_baseline_runs/eval/change_summary_eval/run_eval/midas_summary_eval.py
```

Then visualize:
```bash
python evaluation/dca_baseline_runs/eval/change_summary_eval/run_eval/visualize_all_metrics.py
python evaluation/dca_baseline_runs/eval/change_summary_eval/run_eval/fbeta_quartile_analysis.py
```

Outputs to `eval/change_summary_eval/eval_results/`:
- `summary_eval_results.json` — per-pair + averaged F-beta, precision, recall, hallucination rate
- `heatmap_quality.png` — quality heatmap across models and prompt versions
- `bars_precision.png`, `bars_recall.png`, `bars_f1.png`, `bars_hallucination_rate.png`

### Combined Val/Test Evaluation

To evaluate on the frozen val/test split (comparable with optimized model results):

```bash
python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_human_runs.py --which-set val
```

See [evaluation/dca_run_val_test_set_eval/README.md](../dca_run_val_test_set_eval/README.md) for full details.

---

## Pipeline at a Glance

1. Pick a prompt from `human_prompts/`.
2. Run `run_human_prompts/run_with_human_prompts.py` (single run) or `run_human_prompts/run_all_experiments.py` (batch).
3. Outputs are written into the pair JSONs under `data/dca_data/image_pairs_llm_output_exp1/`.
4. Evaluate classification with `eval/discard_image_classification_eval/run_eval/eval_human_runs.py`.
5. Evaluate change summaries with `eval/change_summary_eval/run_eval/midas_summary_eval.py`, then visualize.
6. Optionally compare with optimized-model runs via `evaluation/dca_run_val_test_set_eval/`.

---

## See Also

- [Val/Test Evaluation](../dca_run_val_test_set_eval/README.md) — compare human prompts vs optimized models on frozen splits
- [DCA Metric](../README.md) — scoring formula details
- [Tournament Pipeline](../../dca_tournaments/README.md) — automated multi-stage prompt optimization
- [Data Generation](../../utils/dca_data_generation/ReadMe.md) — how the dataset was created and augmented
