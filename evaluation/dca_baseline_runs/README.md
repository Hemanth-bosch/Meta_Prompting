# DCA Human Prompt Baseline Runs — Guide

This directory contains scripts for executing LLM-based experiments on document image pairs using various prompts and model configurations.

---

## Overview

The human prompts pipeline runs inference on image pair datasets using different prompt versions and LLM models:

- **Scripts**:
  - `human_prompt_runs/run_with_human_prompts.py` — Execute a single prompt on image pairs
  - `human_prompt_runs/run_all_experiments.py` — Orchestrate multiple runs sequentially with logging
  - `human_prompt_runs/run_v4_human_prompt.py` — V4 multi-step triage + diff pipeline

- **Prompts** (in `human_prompts/` directory):
  - `v0_prompt_midas.md` — MIDAS prompt version 0 - this is original MIDAS prompt
  - `v1_prompt_midas.md` — MIDAS prompt version 1 - this has been adapted to latest rules and decision table - after rounds of discussion
  - `v2_prompt_cr.md` — prompt version 2 - made using claude 4.6 opus and using decision tree style format
  - `v3_prompt_cr.md` — prompt version 3 - made using claude 4.6 opus and using decision tree style format but with extra thinking area key in output json. This thinking area is nothing related to reasoning tokens, but in output json we instruct LLM to output this extra key.

- **Models** (via Azure OpenAI):
  - `gpt-4.1` — Standard GPT-4.1
  - `gpt-4.1-mini` — Lightweight GPT-4.1
  - `gpt-5.2` — Reasoning-capable model
  - `gpt-5-mini` — Reasoning-capable mini model

### Pipeline at a Glance

1. Pick a prompt from `human_prompts/`.
2. Run `human_prompt_runs/run_with_human_prompts.py` (single run) or `human_prompt_runs/run_all_experiments.py` (batch).
3. Outputs are written back to pair JSONs under `data/dca_data/image_pairs_llm_output_exp1/`.
4. Evaluate and compare runs with `eval_runs.py`.

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

## Single Run: `human_prompt_runs/run_with_human_prompts.py`

Execute a single prompt with a specific model and store results in JSON files.

### Basic Syntax

```powershell
python evaluation/dca_baseline_runs/human_prompt_runs/run_with_human_prompts.py `
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
python evaluation/dca_baseline_runs/human_prompt_runs/run_with_human_prompts.py --key-prefix v3_prompt_cr_gpt52 --deployment gpt-5.2 --model-name gpt-5.2 --verbose --save-full-response --prompt v3_prompt_cr.md
```

#### Example 2: Run v2_prompt_cr with GPT-5.2

```powershell
python evaluation/dca_baseline_runs/human_prompt_runs/run_with_human_prompts.py --key-prefix v2_prompt_cr_gpt52 --deployment gpt-5.2 --model-name gpt-5.2 --verbose --save-full-response --prompt v2_prompt_cr.md --no-thinking-area
```

#### Example 3: Run v1_prompt_midas on subset (pairs 1-50)

```powershell
python evaluation/dca_baseline_runs/human_prompt_runs/run_with_human_prompts.py `
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

## Batch Runs: `human_prompt_runs/run_all_experiments.py`

Sequentially execute multiple predefined experiments with full logging.

### Usage

```powershell
python evaluation/dca_baseline_runs/human_prompt_runs/run_all_experiments.py
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
            PYTHON, "evaluation/dca_baseline_runs/human_prompt_runs/run_with_human_prompts.py",
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

## Output Format

### Key Fields

- **`discard_image`**: Boolean classification (true=discard, false=keep)
- **`changes`**: String description of detected changes
- **`thinking_area`**: (Optional) Extra key from model
- **`metadata`**: Complete request/response data + timing + retry log

### Check JSON Output

View results in `data/dca_data/image_pairs_llm_output_exp1/pair_*.json`:

```powershell
# View one pair
cat data/dca_data/image_pairs_llm_output_exp1/pair_001.json | jq '.v3_prompt_cr_gpt52'
```

---

## Next Steps

After running experiments:

1. **Evaluate Results**: See `evaluation/dca_baseline_runs/eval_runs.py`
2. **Analyze Metrics**: Generate classification report, confusion matrix, token usage plots
3. **Compare Models**: Run multiple models and visualize differences

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

- **Evaluation**: `evaluation/dca_baseline_runs/eval_runs.py`
- **Prompt Files**: `evaluation/dca_baseline_runs/human_prompts/v*.md`
- **Data**: `data/dca_data/image_pairs_llm_output_exp1/`
