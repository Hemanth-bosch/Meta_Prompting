# Onboarding a New Use Case — Meta-Prompting Framework

**Who this guide is for:** A developer adding a new document-processing or image-analysis use case to this repository.

**What you will end up with:** A fully working pipeline that loads your data, optimizes a prompt automatically using DSPy, evaluates it, and saves the best result — following the same patterns as the existing DCA use case.

---

## How This Framework Works — 2-Minute Overview

The framework has two layers. You choose based on how mature your use case is:

**Layer 1 — Single optimizer run (`extract_<usecase>.py`):**
Run one optimizer (MIPROv2, Bootstrap, GEPA, or SIMBA) on your dataset and get a saved optimized prompt. Fast to set up, good for initial experiments.

**Layer 2 — Tournament pipeline (`dca_tournaments/`):**
Run all four optimizers in sequence. Each stage takes the best results from the previous stage as its starting point. Produces a leaderboard of all candidates and selects the overall best. More effort to set up, but gives better results. See [dca_tournaments/README.md](../dca_tournaments/README.md).

Both layers share the same building blocks: Signature → Data Loader → Splitter → Metric → Script.

```
Your use case needs:
  signatures/<usecase>.py      ← what inputs/outputs the model should produce
  utils/data_loading.py        ← how to load your data
  utils/splitting.py           ← how to split train/val/test
  evaluation/<usecase>_metric.py  ← how to score model outputs
  extract_<usecase>.py         ← the main script that ties it all together
```

Reference use case to model yours on: **DCA** (image-pair comparison) — `signatures/dca.py`, `evaluation/dca_metric.py`, `utils/data_loading.py`, `extract_dca.py`, `dca_tournaments/`.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Prepare the Data](#2-prepare-the-data)
3. [Define the DSPy Signature](#3-define-the-dspy-signature)
4. [Implement the Data Loader](#4-implement-the-data-loader)
5. [Implement the Data Splitting Strategy](#5-implement-the-data-splitting-strategy)
6. [Define the Evaluation Metric](#6-define-the-evaluation-metric)
7. [Add Post-hoc Analysis](#7-add-post-hoc-analysis)
8. [Create the Main Pipeline Script](#8-create-the-main-pipeline-script)
9. [Update Shared Configuration](#9-update-shared-configuration)
10. [Optional: Baseline and Eval Infrastructure](#10-optional-baseline-and-eval-infrastructure)
11. [Verification and Testing](#11-verification-and-testing)
12. [Complete File Checklist](#12-complete-file-checklist)
13. [Architecture Diagram](#13-architecture-diagram)

---

## 1. Prerequisites

| Item | What to do |
|------|------------|
| **Branch** | Create a new feature branch **from `dca`** — it has the most complete pipeline (stratified splits, LLM-judge metric, grid runner, val/test eval scripts). |
| **Python environment** | Run `uv sync` from the repo root, then activate with `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows). |
| **Azure / LLM access** | Copy `env.template` to `.env` and fill in your credentials. Run `az login` if using Azure CLI auth without an API key. |
| **Verify setup** | Run `python -c "import dspy; print(dspy.__version__)"` to confirm the environment is working. |

---

## 2. Prepare the Data

### 2.1 Organize raw data

Create a directory under `data/` for your use case:

```
data/<usecase>_data/
├── images/              # or image_pairs/, documents, etc.
├── ground_truth/        # one JSON file per sample with expected labels
└── README.md            # schema documentation and data provenance
```

Each ground-truth JSON should contain the expected output values for your task, using consistent key names (e.g. `human_label_discard_image`, `human_label_changes` in DCA).

### 2.2 Profile the data before writing any code

Before touching code, understand your dataset:

- **Class balance** — what is the distribution of your primary output field?
- **Sub-categories** — are there named category codes that need to be preserved across splits?
- **Ambiguous samples** — do any samples have uncertain labels? Mark them (e.g. `doubt: true`).
- **Volume** — aim for at least 50 samples total, 20 per split. If you have rare categories, plan for augmentation.

Store statistics in `data/<usecase>_data/human_label_stats/` so they are version-controlled.

> **DCA example:** 429 pairs across 8 categories (A1/A2/B1/B2/B3/B4/B5a/B5b). Category distribution is tracked in `data/dca_data/human_label_stats/`.

### 2.3 (Optional) Data augmentation for rare categories

If some categories have very few samples (fewer than 20), create augmentation scripts:

```
utils/<usecase>_data_generation/
├── generate_from_source.py       # initial dataset creation
└── augment_rare_categories.py    # synthetic oversampling
```

> **DCA example:** `utils/dca_data_generation/augment_rare_categories.py` generates synthetic B1/B2/B4 pairs using blur, cross-out overlays, and image recombination. See [utils/dca_data_generation/ReadMe.md](../utils/dca_data_generation/ReadMe.md).

---

## 3. Define the DSPy Signature

**File:** `signatures/<usecase>.py`

A DSPy Signature is the **task definition** — it tells the model what inputs it receives and what outputs it must produce. The docstring becomes the instruction block (system prompt). The optimizers will try to improve this docstring automatically.

```python
# signatures/<usecase>.py
import dspy
from typing import Optional

class MyUseCaseSignature(dspy.Signature):
    """Write a detailed task description here.
    Include: what the task is, decision rules, edge cases,
    and the exact format expected for each output field.
    The optimizers will rewrite this docstring — a strong
    starting point leads to better optimization results."""

    # Inputs
    image: dspy.Image = dspy.InputField(desc="...")
    # For optional inputs (e.g. one side of a pair may be missing):
    # image_b: Optional[dspy.Image] = dspy.InputField(desc="...", default=None)

    # Outputs
    label: str = dspy.OutputField(desc="Output exactly 'A' or 'B'. ...")
    explanation: str = dspy.OutputField(desc="Free-text explanation of the decision. ...")
```

**Register it** in `signatures/__init__.py`:
```python
from signatures.<usecase> import MyUseCaseSignature
```

**Tips:**
- Use `Optional[dspy.Image]` when an input image may be absent (DCA does this for one-sided pairs).
- Be very specific in output field `desc` values: list allowed values, format requirements, and edge cases. These directly affect prompt quality before any optimization.
- For structured sub-objects (like a list of line items), use a nested `dspy.Signature` / Pydantic model as the output field type.
- For the DCA use case, all guidance lives in the docstring and field `desc` values are empty (`desc=""`). This is the P1 "instructions-only" mode. New use cases can start with full `desc` values and switch later.

---

## 4. Implement the Data Loader

**File:** `utils/data_loading.py` — add a new function

```python
def load_<usecase>_dataset(
    data_dir: str = "data/<usecase>_data/images",
    gt_dir: str = "data/<usecase>_data/ground_truth",
) -> list:  # list of dspy.Example
    examples = []
    for gt_file in sorted(Path(gt_dir).glob("*.json")):
        gt = json.loads(gt_file.read_text())
        example = dspy.Example(
            # Inputs (must match your Signature's InputField names)
            image=dspy.Image.from_file(...),
            # Expected outputs — prefix with "expected_" by convention
            expected_label=gt["human_label"],
            expected_explanation=gt["human_explanation"],
        ).with_inputs("image")   # tells DSPy which fields are inputs
        examples.append(example)
    print(f"Loaded {len(examples)} examples")
    return examples
```

**Key conventions:**

| Convention | Why |
|---|---|
| Prefix ground-truth fields with `expected_` | DSPy uses this convention. The metric function receives `example.expected_*` as the ground truth and `prediction.<field>` as the model output. |
| Call `.with_inputs(...)` on every Example | Tells DSPy which fields to pass to the model vs. which are labels. Missing this causes silent errors. |
| Print a load summary | Shows counts and class distribution — makes it easy to spot data issues early. |
| Handle missing data gracefully | Check for missing images or null values and skip or substitute (see `load_dca_dataset` for the pattern). |

**Register it** in `utils/__init__.py`:
```python
from utils.data_loading import load_<usecase>_dataset
```

---

## 5. Implement the Data Splitting Strategy

**File:** `utils/splitting.py` — add a new function

```python
def stratified_split_<usecase>_dataset(
    dataset: list,
    train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
    random_seed=42, verbose=True,
) -> tuple:  # (train, val, test) lists
    ...
```

**Checklist:**
- Stratify on the primary classification field AND any category codes to preserve class balance in each split.
- Handle small groups (fewer than 3 examples in a stratum) — assign the whole group to the split with the largest deficit.
- Save a split manifest JSON to `.cache/<usecase>_split_seed42.json`. This makes the split reproducible: any script that loads the manifest gets exactly the same val/test sets.
- Print per-split distribution statistics when `verbose=True`.

> **DCA example:** Stratifies on `["expected_discard_image", "expected_human_label_category", "doubt"]`.

**Register it** in `utils/__init__.py`:
```python
from utils.splitting import stratified_split_<usecase>_dataset
```

---

## 6. Define the Evaluation Metric

**File:** `evaluation/<usecase>_metric.py`

The metric is the **objective function** that all DSPy optimizers maximize. Higher is better. It must return a float in [0, 1].

```python
def <usecase>_metric(example, prediction, trace=None) -> float:
    """Score model output against ground truth. Return float in [0, 1]."""
    # Compare prediction.<field> against example.expected_<field>
    ...
    return score
```

**Design decisions:**

| Decision | Guidance |
|---|---|
| Single score vs. weighted composite | If you have multiple output fields, combine them with explicit weights. DCA uses `0.6 × classification + 0.4 × change_summary`. Assign higher weight to the business-critical field. |
| Exact match | Use for categorical fields (labels, codes, boolean flags). |
| Fuzzy match | Use `SequenceMatcher` or token overlap for fields where near-matches should get partial credit. See `common.py`. |
| LLM-as-judge | For free-text outputs where similarity matters (summaries, explanations), an LLM judge is most accurate. Cache results to `.cache/<usecase>_judge_cache.json` to avoid re-running the same calls. See `dca_metric.py → judge_summary_metric`. |
| Rule-based fast paths | For canonical output strings (e.g. "NoMatchFound", "Images are identical"), use string matching before the LLM judge. Much faster and cheaper. |

**Metric with feedback (required for GEPA optimizer):**

```python
def <usecase>_metric_with_feedback(example, prediction, trace=None,
                                   pred_name=None, pred_trace=None):
    """Same as above, but returns dspy.Prediction(score=..., feedback=...)
    when pred_name is set. The feedback string is shown to the reflection LM
    to explain what was wrong."""
    score = <usecase>_metric(example, prediction, trace)
    if pred_name:
        feedback = f"Score: {score:.2f}. ..."  # human-readable explanation
        return dspy.Prediction(score=score, feedback=feedback)
    return score
```

**Register in** `evaluation/__init__.py`:
```python
from evaluation.<usecase>_metric import (
    <usecase>_metric,
    <usecase>_metric_with_feedback,
)
```

---

## 7. Add Post-hoc Analysis

**File:** `evaluation/posthoc_analysis.py` — add a function

```python
def analyze_<usecase>_model_performance(extractor, dataset, verbose=False):
    """Run inference on every example and compute per-field scores.
    Returns (field_scores_dict, total_scores_list)."""
    field_scores = {}
    total_scores = []
    for example in dataset:
        prediction = extractor(**example.inputs())
        # Score each output field individually
        field_scores["label"].append(...)
        total_scores.append(total_score)
    # Print a summary table
    ...
    return field_scores, total_scores
```

This is called after optimization to produce a detailed per-field breakdown on the val and test sets. It helps identify which output fields are still underperforming after optimization.

---

## 8. Create the Main Pipeline Script

**File:** `extract_<usecase>.py` (at the repo root)

This script ties everything together. Use `extract_dca.py` as the template — it is the most complete example.

**Script structure:**

```
1. Module docstring       — describe the use case, usage examples, supported optimizers
2. Argument parsing       — model name, optimizer choice, data paths, CoT, save/load, cache
3. Logging (TeeLogger)    — mirrors stdout to a log file (copy from extract_dca.py)
4. LM configuration       — call create_lm() from config.py
5. Data loading           — call load_<usecase>_dataset()
6. Data splitting         — call stratified_split_<usecase>_dataset(), save manifest to .cache/
7. DSPy module creation   — dspy.Predict(MyUseCaseSignature) or dspy.ChainOfThought(...)
8. Load cached model      — optional: --load-extractor path
9. Build strategies       — define optimizer configurations
10. Run OptimizationRunner — from optimization/opt.py
11. Save to .cache/        — --save-model flag
12. Close logger
```

**Key CLI arguments to support:**

| Argument | Description |
|----------|-------------|
| `--model-name` | LLM to use (overrides `.env MODEL_NAME`) |
| `--optimizer` | `miprov2`, `bootstrap`, `gepa`, `simba`, or `all` |
| `--miprov2-auto` | `light`, `medium`, or `heavy` |
| `--no-cot` | Use `dspy.Predict` instead of `dspy.ChainOfThought` |
| `--reasoning-effort` | `low`, `medium`, `high` for reasoning models |
| `--save-model` | Save the optimized prompt to `.cache/` |
| `--load-extractor` | Resume from a previously saved `.json` program |
| `--no-cache` | Disable LLM response caching |

### Option: Use the Tournament Pipeline Instead

For production-level optimization, use the tournament pipeline rather than a single-optimizer script. The tournament pipeline runs all four stages sequentially, tracks all candidates in a leaderboard, and selects the best by val score and optionally by inference token cost.

To adapt the tournament pipeline to your use case, see the "Adapting to a New Use Case" section in [dca_tournaments/README.md](../dca_tournaments/README.md).

---

## 9. Update Shared Configuration

### `config.py`

**Usually no changes needed.** `config.py` handles LM creation and Azure auth generically.

The only time you touch it: if your metric uses an LLM-as-judge, add a constant for the judge model name:
```python
MYUSECASE_JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "azure/gpt-4.1")
```

### `optimization/opt.py`

**No changes needed.** `OptimizationRunner` and all strategy builders (`create_miprov2_strategy`, etc.) are use-case-agnostic. They accept any metric function and any DSPy module.

### `.env` and `env.template`

Add your data paths:
```env
MYUSECASE_DATA_DIR=data/<usecase>_data/images
MYUSECASE_GT_DIR=data/<usecase>_data/ground_truth
```

Update `env.template` with the same entries so other developers know these variables exist.

---

## 10. Optional: Baseline and Eval Infrastructure

For a mature use case, create evaluation infrastructure so human-authored prompts and optimized prompts can be compared side by side.

### Human prompt baselines

```
evaluation/<usecase>_baseline_runs/
├── README.md
├── human_prompts/              # versioned prompt markdown files
├── run_human_prompts/          # scripts to run inference with each prompt
└── eval/                       # evaluation scripts and result visualizations
```

> **DCA example:** See [evaluation/dca_baseline_runs/README.md](../evaluation/dca_baseline_runs/README.md).

### Val/test evaluation scripts

```
evaluation/<usecase>_run_val_test_set_eval/
├── run_val_test_set_eval_on_optimized_prompt.py   # evaluate a saved .json model
├── run_val_test_set_eval_on_human_runs.py         # evaluate stored human-prompt outputs
└── eval_results/
```

These scripts load a saved model, run live inference on the frozen val/test split, compute per-pair scores, and write structured JSON results for comparison.

> **DCA example:** See [evaluation/dca_run_val_test_set_eval/README.md](../evaluation/dca_run_val_test_set_eval/README.md).

### Grid runner

For systematic experiments across multiple models and hyperparameter combinations:
```
run_<usecase>_grid.py           # parallel combinatorial grid runner
```

> **DCA example:** `run_dca_grid.py` runs combinations of models × MIPROv2 intensity × demo counts in parallel using a `ThreadPoolExecutor`.

---

## 11. Verification and Testing

Run these checks in order before considering the setup complete:

```bash
# Step 1: Verify data loads correctly
python -c "
from utils import load_<usecase>_dataset, stratified_split_<usecase>_dataset
dataset = load_<usecase>_dataset()
train, val, test = stratified_split_<usecase>_dataset(dataset)
print(f'train={len(train)}  val={len(val)}  test={len(test)}')
"

# Step 2: Run a fast zero-shot baseline (no optimization, just score the initial prompt)
python extract_<usecase>.py --optimizer miprov2 --miprov2-auto light --no-cache

# Step 3: Full optimization run
python extract_<usecase>.py --optimizer miprov2 --miprov2-auto medium --save-model --no-cache

# Step 4: Verify saved model loads and runs
python extract_<usecase>.py --load-extractor .cache/<saved_model>.json
```

**What to check:**

| Check | What it tells you |
|-------|------------------|
| Data loads without errors | Correct image count, expected class distribution printed |
| Split is reproducible | Running twice with the same seed produces identical `.cache/<usecase>_split_seed42.json` |
| Zero-shot score is non-trivial | If score is near 0%, the signature instructions or metric logic likely have a bug |
| Optimization improves score | At least one strategy should beat the zero-shot baseline by a few percent |
| Saved model loads and produces same score | Confirms the save/load round-trip works correctly |

---

## 12. Complete File Checklist

### New files to create

| # | File | Purpose |
|---|------|---------|
| 1 | `signatures/<usecase>.py` | DSPy Signature — inputs and outputs |
| 2 | `evaluation/<usecase>_metric.py` | Metric function(s) for optimization and evaluation |
| 3 | `extract_<usecase>.py` | Main pipeline script |
| 4 | `data/<usecase>_data/README.md` | Dataset schema and provenance documentation |

### Existing files to modify

| # | File | What to add |
|---|------|-------------|
| 5 | `utils/data_loading.py` | `load_<usecase>_dataset()` function |
| 6 | `utils/splitting.py` | `stratified_split_<usecase>_dataset()` function |
| 7 | `evaluation/posthoc_analysis.py` | `analyze_<usecase>_model_performance()` function |
| 8 | `signatures/__init__.py` | Import of new signature class |
| 9 | `evaluation/__init__.py` | Import of new metric functions |
| 10 | `utils/__init__.py` | Import of new loader and splitter |
| 11 | `env.template` | New data path variables |

### Files that need no changes

| File | Why |
|------|-----|
| `config.py` | Generic LM configuration — shared across all use cases |
| `optimization/opt.py` | Generic optimizer runner — works with any metric and any DSPy module |

---

## 13. Architecture Diagram

How the pieces connect:

```
extract_<usecase>.py
  │   (entry point: argparse, logging, pipeline orchestration)
  │
  ├── config.py
  │     └── create_lm()            ← creates the Azure/OpenAI/Vertex LM client
  │
  ├── signatures/<usecase>.py
  │     └── MyUseCaseSignature     ← inputs/outputs contract
  │           └── dspy.ChainOfThought(Signature)  or  dspy.Predict(Signature)
  │
  ├── utils/
  │     ├── data_loading.py        ← load_<usecase>_dataset() → List[dspy.Example]
  │     └── splitting.py           ← stratified_split() → train / val / test
  │
  ├── evaluation/
  │     ├── <usecase>_metric.py    ← metric(example, prediction) → float [0, 1]
  │     └── posthoc_analysis.py    ← per-field performance breakdown
  │
  └── optimization/opt.py
        └── OptimizationRunner     ← runs Bootstrap / MIPROv2 / GEPA / SIMBA
              └── saves optimized program to .cache/<model>_<strategy>_<timestamp>.json


For multi-stage optimization, the tournament pipeline wraps the above:

dca_tournaments/run_tournament.py
  │   (orchestrates 4 stages: seed → bootstrap → miprov2 → gepa)
  │
  ├── seed_prompt_generator.py   ← Stage 0: LLM writes improved instructions
  ├── stage_runner.py            ← Stages 1-3: subprocess-isolated compile() + eval
  ├── leaderboard.py             ← tracks all candidates, Pareto selection
  └── tour_utils.py              ← evaluate_candidate(), token tracking, logging
```

---

*For questions or issues, open a ticket or check the DCA reference implementation: `signatures/dca.py`, `extract_dca.py`, `evaluation/dca_metric.py`, `dca_tournaments/`.*
