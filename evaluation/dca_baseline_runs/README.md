# DCA Human Prompt Baseline Runs — Guide

This directory contains scripts for executing LLM-based experiments on document image pairs using various prompts and model configurations.

---

## Overview

The human prompts pipeline runs inference on image pair datasets using different prompt versions and LLM models:

- **Scripts**:
  - `run_human_prompts/run_with_human_prompts.py` — Execute a single prompt on image pairs
  - `run_human_prompts/run_all_experiments.py` — Orchestrate multiple runs sequentially with logging
  - `run_human_prompts/run_v4_human_prompt.py` — V4 multi-step triage + diff pipeline
  - `run_human_prompts/run_support_prompt_for_categorization_of_data.py` — Assign `human_label_category` codes from existing human labels

- **Prompts** (in `human_prompts/` directory):
  - `v0_prompt_midas.md` — MIDAS prompt version 0 - this is original MIDAS prompt
  - `v1_prompt_midas.md` — MIDAS prompt version 1 - this has been adapted to latest rules and decision table - after rounds of discussion
  - `v2_prompt_cr.md` — prompt version 2 - made using claude 4.6 opus and using decision tree style format
  - `v3_prompt_cr.md` — prompt version 3 - made using claude 4.6 opus and using decision tree style format but with extra thinking area key in output json. This thinking area is nothing related to reasoning tokens, but in output json we instruct LLM to output this extra key.
  - `v4_prompt_cr.md` — prompt version 4 - multi-step triage + diff pipeline (uses `run_v4_human_prompt.py`)
  - `support_prompt_for_categorization_of_data.md` — maps `human_label_discard_image` + `human_label_changes` to category codes `A1/A2/B1/B2/B3/B4/B5a/B5b`

- **Models** (via Azure OpenAI):
  - `gpt-4.1` — Standard GPT-4.1
  - `gpt-4.1-mini` — Lightweight GPT-4.1
  - `gpt-5.2` — Reasoning-capable model
  - `gpt-5-mini` — Reasoning-capable mini model

### `eval/` Folder Structure

```
eval/
├── discard_image_classification_eval/
│   ├── run_eval/
│   │   └── eval_human_runs.py           # Classification metrics, grouped bar plots, confusion matrices, CSV
│   └── eval_results/                    # Generated outputs (created on first run)
│       ├── classification_metrics.png
│       ├── confusion_matrices.png
│       ├── token_usage.png
│       ├── response_time.png
│       ├── failures.png
│       └── eval_summary.csv
└── change_summary_eval/
    ├── run_eval/
    │   ├── midas_summary_eval.py        # LLM judge + semantic similarity across all model/version combos
    │   ├── visualize_all_metrics.py     # Heatmap + bar charts from summary_eval_results.json
    │   └── fbeta_quartile_analysis.py   # Precision-60/Recall-40 F-beta quartile breakdown (gpt41m, gpt5m)
    └── eval_results/                    # Generated outputs (created on first run)
        ├── summary_eval_results.json    # Per-pair + averaged metrics (written by midas_summary_eval.py)
        ├── heatmap_quality.png
        ├── bars_precision.png
        ├── bars_recall.png
        ├── bars_f1.png
        ├── bars_hallucination_rate.png
        └── bars_llm_score.png
```

### Pipeline at a Glance

1. Pick a prompt from `human_prompts/`.
2. Run `run_human_prompts/run_with_human_prompts.py` (single run) or `run_human_prompts/run_all_experiments.py` (batch).
3. Outputs are written back to pair JSONs under `data/dca_data/image_pairs_llm_output_exp1/`.
4. Evaluate classification with `eval/discard_image_classification_eval/run_eval/eval_human_runs.py`.
5. Evaluate change summaries with `eval/change_summary_eval/run_eval/midas_summary_eval.py`, then visualize with `visualize_all_metrics.py`.

---

## Prerequisites

1. **Python Environment**:
   ```powershell
   & ".venv\Scripts\Activate.ps1"
   ```

2. **Azure OpenAI Credentials**:
   ```powershell
   $env:AZURE_OPENAI_ENDPOINT = "https://aiq212-openai.openai.azure.com/"
   $env:AZURE_OPENAI_API_VERSION = "2024-08-01-preview"
   # Then: az login
   ```

3. **Data Structure**:
   - Image pairs in: `data/dca_data/image_pairs/`
   - Output JSONs in: `data/dca_data/image_pairs_llm_output_exp1/`
  - Note: DCA data is currently placed under `data/dca_data/` for active experimentation; this placement is temporary and may be removed or reorganized later.

---

## Single Run: `run_human_prompts/run_with_human_prompts.py`

Execute a single prompt with a specific model and store results in JSON files.

### Basic Syntax

```powershell
python evaluation/dca_baseline_runs/run_human_prompts/run_with_human_prompts.py `
    --key-prefix <KEY_PREFIX> `
    --prompt <PROMPT_FILE> `
    --deployment <MODEL_NAME> `
    --no-thinking-area (Needed - except for v3) `
    --save-full-response `
    --verbose `
    --overwrite (options to overwrite existing keys in JSON) `
    [--reasoning-effort <LEVEL>] `
    [other options]
```

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--key-prefix` | Prefix for output JSON keys | `v3_prompt_cr_gpt41`, `v3_prompt_cr_gpt52` |
| `--prompt` | Prompt markdown filename | `v3_prompt_cr.md`, `v1_prompt_midas.md` |
| `--deployment` | LLM model to use | `gpt-4.1`, `gpt-5.2`, `gpt-4.1-mini`, `gpt-5-mini` |

### Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model-name` | Alias for `--deployment` (overrides if provided) | None |
| `--reasoning-effort` | For reasoning models (gpt-5.2, gpt-5-mini): `minimal`, `low`, `medium`, `high` | None (unset) |
| `--json-dir` | Directory with pair JSONs | `data/dca_data/image_pairs_llm_output_exp1` |
| `--image-pairs-dir` | Directory with image pair files | `data/dca_data/image_pairs` |
| `--save-full-response` | Save complete API response | False |
| `--overwrite` | Overwrite key if exists | False |
| `--no-thinking-area` | We do not need thinking area for v0, v1, and v2. Hence, we need to pass this argument every time we run v0, v1, and v2. Only in case of v3, we do not want this. |
| `--max-retries` | Retries on transient errors | 3 |
| `--sleep-seconds` | Sleep between API calls (seconds) | 0.0 |
| `--start` | Start pair index (e.g., 1 for pair_001) | None (all) |
| `--end` | End pair index (e.g., 50 for pair_050) | None (all) |
| `--limit` | Max pairs to process | None (all) |
| `--only-doubt` | Process only pairs where `doubt: true` | False |
| `--dry-run` | Simulate without writing files | False |
| `--verbose` | Enable verbose logging | False |
| `--endpoint` | Azure OpenAI endpoint URL | From env or default |
| `--api-version` | Azure OpenAI API version | From env or default |

### Examples

#### Example 1: Run v3_prompt_cr with GPT-5.2

```powershell
python evaluation/dca_baseline_runs/run_human_prompts/run_with_human_prompts.py --key-prefix v3_prompt_cr_gpt52 --deployment gpt-5.2 --model-name gpt-5.2 --verbose --save-full-response --prompt v3_prompt_cr.md
```

#### Example 2: Run v2_prompt_cr with GPT-5.2

```powershell
python evaluation/dca_baseline_runs/run_human_prompts/run_with_human_prompts.py --key-prefix v2_prompt_cr_gpt52 --deployment gpt-5.2 --model-name gpt-5.2 --verbose --save-full-response --prompt v2_prompt_cr.md --no-thinking-area
```

#### Example 3: Run v1_prompt_midas on subset (pairs 1-50)

```powershell
python evaluation/dca_baseline_runs/run_human_prompts/run_with_human_prompts.py `
    --key-prefix v1_prompt_midas_gpt41_mini `
    --prompt v1_prompt_midas.md `
    --deployment gpt-4.1-mini `
    --start 1 `
    --end 50 `
    --verbose
    --save-full-response
    --no-thinking-area
```

---

## Batch Runs: `run_human_prompts/run_all_experiments.py`

Sequentially execute multiple predefined experiments with full logging.

### Usage

```powershell
python evaluation/dca_baseline_runs/run_human_prompts/run_all_experiments.py
```

### Features

- Runs all `RUNS` list sequentially
- Creates log file: `run_all_experiments.log` (in workspace root)
- Captures combined stdout/stderr for each run
- Continues on failure (but reports it)

### Configuration

Edit `run_all_experiments.py` to customize the `RUNS` list:

```python
RUNS = [
    {
        "label": "v3_prompt_cr_gpt52_reason_low",
        "args": [
            PYTHON, "evaluation/dca_baseline_runs/run_human_prompts/run_with_human_prompts.py",
            "--key-prefix", "v3_prompt_cr_gpt52_reason_low",
            "--deployment", "gpt-5.2",
            "--model-name", "gpt-5.2",
            "--verbose", "--save-full-response",
            "--prompt", "v3_prompt_cr.md",
            "--reasoning-effort", "low",
        ],
    },
    # ... more runs ...
]
```

### Example Output

```
======================================================================
run_all_experiments.py  started at 2026-03-16 10:30:45
Total runs: 3
Log file  : C:\...\run_all_experiments.log
======================================================================

──────────────────────────────────────────────────────────────────────
[1/3]  v3_prompt_cr_gpt52_reason_low   started at 2026-03-16 10:30:45
──────────────────────────────────────────────────────────────────────
[2026-03-16 10:30:46] Processing pair_001.json ...
[2026-03-16 10:30:47] Saved metadata to pair_001.json
...
[1/3]  v3_prompt_cr_gpt52_reason_low   finished at 2026-03-16 10:45:00  →  OK
...
======================================================================
All runs completed at 2026-03-16 11:00:00.  Overall: SUCCESS
======================================================================
```

---

## Human Label Category Coding

Assign branch codes to `human_label_category` using existing human labels (`human_label_discard_image`, `human_label_changes`):

```powershell
python evaluation/dca_baseline_runs/run_human_prompts/run_support_prompt_for_categorization_of_data.py `
  --prompt support_prompt_for_categorization_of_data.md `
  --deployment gpt-4.1-mini `
  --start 1 `
  --end 100 `
  --verbose
```

Allowed codes written to JSON:
- `A1`, `A2`, `B1`, `B2`, `B3`, `B4`, `B5a`, `B5b`

Then analyze category distribution and cross-tabs:

```powershell
python evaluation/dca_baseline_runs/run_human_prompts/analyze_labels.py `
  --input-dir data/dca_data/image_pairs_llm_output_exp1 `
  --output-dir data/dca_data/human_label_stats
```

---

## Output Format

### Key Fields

```json
{
  "doubt": false,
  "human_label_category": "B5a",
  "human_label_discard_image": true,
  "human_label_changes": "Images are identical - no changes",
  "discard_image": true,
  "changes": "<LLM-generated change description>",
  "is_synthetic": false
}
```

- `doubt` (`bool`): whether this pair has uncertain/disputed ground truth labels
- `human_label_discard_image` (`bool`): ground truth label
  - `true`  = discard — images are identical or differ only trivially
  - `false` = keep — at least one meaningful change exists
- `human_label_changes` (`str`): human-verified change description
- `human_label_category` (`str`): category code (`A1/A2/B1/B2/B3/B4/B5a/B5b`)
- `is_synthetic` (`bool`): whether this pair was generated by `augment_rare_categories.py` (present only for synthetic pairs)
- `discard_image` / `changes`: LLM-generated outputs

---

## Next Steps

After running experiments:

1. **Evaluate Results**: See `evaluation/dca_baseline_runs/eval/discard_image_classification_eval/run_eval/eval_human_runs.py` (outputs to `eval/discard_image_classification_eval/eval_results/`)
2. **Analyze Metrics**: Generate classification report, confusion matrix, token usage plots
3. **Compare Models**: Run multiple models and visualize differences
4. **Data Augmentation**: If class imbalance is an issue, generate synthetic pairs with `utils/dca_data_generation/augment_rare_categories.py` (see root README for details)

---

## Environment Variables

Set these before running (or let defaults apply):

```powershell
$env:AZURE_OPENAI_ENDPOINT = "https://aiq212-openai.openai.azure.com/"
$env:AZURE_OPENAI_API_VERSION = "2024-08-01-preview"
$env:AZURE_OPENAI_DEPLOYMENT = "gpt-4.1-mini"  # Default if not specified
```

For CA certificate on corporate networks:
```powershell
$env:AOAI_CA_BUNDLE = "C:\path\to\ca-bundle.crt"
```

---

## See Also

- **Evaluation**: `evaluation/dca_baseline_runs/eval/discard_image_classification_eval/run_eval/eval_human_runs.py`
- **Prompt Files**: `evaluation/dca_baseline_runs/human_prompts/v*.md`
- **Data**: `data/dca_data/image_pairs_llm_output_exp1/`
