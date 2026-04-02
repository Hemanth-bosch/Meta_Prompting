# Onboarding a New Use Case — Meta-Prompting Framework

> **Audience:** Developer adding a new document-processing use case to the `dca` branch.
> **Existing references:** `dca` (image-pair change analysis), `jo` (Japanese order extraction), `yanase` (YANASE order extraction).

---

## 0. Prerequisites

| Item | Detail |
|------|--------|
| **Branch** | Create a new feature branch **from `dca`** — it has the most mature pipeline (stratified splits, LLM-judge metric, grid runner, val/test eval scripts). |
| **Python env** | Activate the existing `.meta-prompting` venv, or create a new one with `uv sync` / `pip install -e .` from `pyproject.toml`. |
| **Azure / LLM access** | Ensure your `.env` has valid credentials (see `env.template`). Run `az login` if using Azure CLI auth. |

---

## 1. Prepare the Data

### 1.1 Collect & organise raw data

Create a directory under `data/` for your use case:

```
data/<usecase>_data/
├── images/          # or image_pairs/, PDFs, etc.
├── ground_truth/    # JSON labels per sample
└── README.md        # schema docs + provenance
```

Each ground-truth JSON should contain the **expected output fields** your signature will predict (Step 2), keyed consistently (e.g. `human_label_*` or plain field names).

### 1.2 Analyse class distribution & label quality

Before writing any code, profile the labels to understand:

- **Class balance** — what is the distribution of your primary classification target?
- **Label categories** — are there sub-categories that need stratification?
- **Ambiguous / doubt labels** — do human annotators disagree on some samples?
- **Data volume** — is the dataset large enough for train/val/test? (aim for ≥ 50 total; ≥ 20 per split).

Store summary statistics in `data/<usecase>_data/human_label_stats/` (CSV, JSON) so they're version-controlled.

> **DCA example:** `data/dca_data/human_label_stats/` contains category counts, classification-vs-category cross-tabs, and a JSON summary.

### 1.3 (Optional) Data generation / augmentation

If rare categories exist, consider writing augmentation scripts under:

```
utils/<usecase>_data_generation/
├── __init__.py
├── generate_from_source.py     # initial dataset creation
└── augment_rare_categories.py  # synthetic oversampling
```

> **DCA example:** `utils/dca_data_generation/augment_rare_categories.py` generates synthetic B1/B2/B4 pairs via blur, cross-out overlays, and recombination.

---

## 2. Define the DSPy Signature

**File:** `signatures/<usecase>.py`

A DSPy `Signature` declares the **inputs** (what the LLM sees) and **outputs** (what the LLM must produce). This is the contract between your data, metric, and optimizer.

```python
# signatures/<usecase>.py
import dspy
from typing import Optional, List

class MyUseCaseSignature(dspy.Signature):
    """Docstring becomes the system prompt preamble.
    Write detailed instructions and decision rules here —
    the optimizer may rewrite this, but a strong starting
    point helps all strategies."""

    # ── Inputs ──
    image: dspy.Image = dspy.InputField(desc="...")
    # or: image_a: Optional[dspy.Image], image_b: Optional[dspy.Image], text: str, ...

    # ── Outputs ──
    classification: str = dspy.OutputField(desc="Exactly 'A' or 'B'. ...")
    summary: str = dspy.OutputField(desc="Free-text explanation. ...")
    # Use List[...] or nested Signature for structured outputs (see yanase.py).
```

**Register it** in `signatures/__init__.py`:

```python
from signatures.<usecase> import MyUseCaseSignature
```

### Tips
- Use `Optional[dspy.Image]` if an input may be absent (DCA does this for one-sided pairs).
- Output field `desc` heavily influences prompt quality — be specific about format, allowed values, and edge cases.
- If your task has structured sub-objects, define a helper Signature class (like `yanase.py → ReceiverAndOrder`).

---

## 3. Implement the Data Loader

**File:** `utils/data_loading.py` — add a function:

```python
def load_<usecase>_dataset(
    data_dir: str = "data/<usecase>_data/images",
    gt_dir: str = "data/<usecase>_data/ground_truth",
) -> List[dspy.Example]:
    ...
```

### Key conventions

| Convention | Why |
|---|---|
| Prefix ground-truth fields with `expected_` | DSPy treats `expected_*` as labels — the metric compares `prediction.<field>` against `example.expected_<field>`. |
| Call `.with_inputs(...)` on each Example | Tells DSPy which fields are model inputs vs. expected outputs. |
| Return `List[dspy.Example]` | All downstream code (splitting, evaluation, optimization) expects this type. |
| Print load summary at the end | Counts, class distribution, skipped samples — makes debugging easy. |

> **DCA example:** `load_dca_dataset()` handles one-sided pairs (original or reimage missing → `None`), normalises category codes, and prints class balance.

**Register** in `utils/__init__.py`:

```python
from utils.data_loading import load_<usecase>_dataset
```

---

## 4. Implement the Data Splitting Strategy

**File:** `utils/splitting.py` — add a function:

```python
def stratified_split_<usecase>_dataset(
    dataset: List[dspy.Example],
    train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
    random_seed=42, verbose=True,
    stratify_by=None,   # list of field names for stratification
) -> Tuple[List, List, List]:
    ...
```

### Checklist

- [ ] **Stratify** on the primary classification field + any category/difficulty labels to preserve class balance across splits.
- [ ] **Handle small groups** (< 3 examples) — assign whole group to the split with the largest deficit (see `stratified_split_dca_dataset` for the pattern).
- [ ] **Save a split manifest** (JSON) to `.cache/` so val/test sets are reproducible across runs.
- [ ] **Print distribution stats** per split when `verbose=True`.

> **DCA example:** Stratifies on `["expected_discard_image", "expected_human_label_category", "doubt"]`.
> **YANASE example:** Stratifies on `shipPostal` (recipient location) to ensure geographic diversity.

**Register** in `utils/__init__.py`:

```python
from utils.splitting import stratified_split_<usecase>_dataset
```

---

## 5. Define the Evaluation Metric

**File:** `evaluation/<usecase>_metric.py`

The metric is the **objective function** that all DSPy optimizers maximise. It must have this signature:

```python
def <usecase>_metric(example, prediction, trace=None) -> float:
    """Return a score in [0, 1]."""
    ...
```

### Design decisions

| Decision | Guidance |
|---|---|
| **Single score vs. weighted composite** | If your task has multiple outputs (e.g. classification + summary), combine with explicit weights. DCA uses `0.6 * cls + 0.4 * change_summary`. |
| **Exact match vs. fuzzy** | Use exact match for categorical/numeric fields; fuzzy (SequenceMatcher, LLM-judge) for free-text. |
| **LLM-as-judge** | If one output is free-text, consider an LLM judge (see `dca_metric.py → judge_summary_metric`). Cache results to avoid redundant API calls. |
| **Rule-based fast paths** | For known canonical outputs (e.g. "NoMatchFound"), use rules before falling back to the LLM judge. |
| **Field weights** | Assign higher weights to business-critical fields (see `jo_metric.py` field_weights dict and `yanase_metric.py`). |

### (Optional) Metric with feedback — for GEPA optimizer

If you plan to use the GEPA optimizer, also implement:

```python
def <usecase>_metric_with_feedback(example, prediction, trace=None,
                                    pred_name=None, pred_trace=None):
    """Return dspy.Prediction(score=..., feedback=...) when pred_name is set."""
    ...
```

This provides textual feedback that the reflection LM uses to improve the prompt. See `jo_metric.py → japanese_order_extraction_metric_with_feedback` for the pattern.

**Register** in `evaluation/__init__.py`:

```python
from evaluation.<usecase>_metric import (
    <usecase>_metric,
    <usecase>_metric_with_feedback,   # if implemented
)
```

---

## 6. Add Post-hoc Analysis

**File:** `evaluation/posthoc_analysis.py` — add a function:

```python
def analyze_<usecase>_model_performance(extractor, dataset, verbose=False):
    """Run inference on every example, compute per-field scores, return (field_scores, total_scores)."""
    ...
```

This is called **after** optimization to produce a detailed field-by-field performance breakdown on val/test sets.

### Pattern (from existing use cases)

1. Loop over `dataset`, run `extractor(**inputs)` on each example.
2. Compare each output field against `example.expected_*` using your metric helpers.
3. Collect per-field score lists and per-example total scores.
4. Print a summary table: field name, avg score, perfect/partial/poor counts.
5. Return `(field_scores_dict, total_scores_list)`.

---

## 7. Create the Main Extraction / Pipeline Script

**File:** `extract_<usecase>.py` (top-level)

This is the **entry point** that ties everything together. Use `extract_dca.py` as the template — it's the most complete.

### Structure

```
1. Docstring           — describe the use case, supported optimizers, dataset, usage examples
2. Argument parsing    — model name, cache, optimizer choice, data paths, save/load, CoT toggle
3. Logging / TeeLogger — capture stdout + file log (copy from extract_dca.py)
4. Model configuration — call reconfigure_lm() from config.py
5. Data loading        — call load_<usecase>_dataset()
6. Data splitting      — call stratified_split_<usecase>_dataset(); save split manifest to .cache/
7. Create DSPy module  — dspy.ChainOfThought(MyUseCaseSignature) or dspy.Predict(...)
8. (Optional) Load cached extractor
9. Set up dspy.Evaluate
10. Define optimization strategies (bootstrap, miprov2, gepa, simba)
11. Create OptimizationRunner and run
12. Save optimized model to .cache/
13. Close logger, rename log file
```

### Key CLI flags to support

```
--model-name          LLM to use (overrides .env MODEL_NAME)
--optimizer           miprov2 | bootstrap | gepa | simba | all
--miprov2-auto        light | medium | heavy
--no-cot              Use dspy.Predict instead of dspy.ChainOfThought
--reasoning-effort    low | medium | high (for reasoning models like gpt-5)
--save-model          Persist optimized prompt to .cache/
--load-extractor      Resume from a previously saved .json
--no-cache            Disable LLM response caching
```

---

## 8. Update Shared Configuration

### 8.1 `config.py`

Generally **no changes needed** — `config.py` handles LM creation, Azure auth, and `reconfigure_lm()` generically for all use cases. Only touch it if your use case needs a different judge model or special LM settings.

If your metric uses an **LLM-as-judge** (like DCA), define a constant:

```python
<USECASE>_JUDGE_MODEL_NAME = "gpt-4.1"
```

### 8.2 `optimization/opt.py`

Generally **no changes needed** — `OptimizationRunner` and all strategy helpers (`create_miprov2_strategy`, etc.) are use-case-agnostic. They accept any metric function and any DSPy module.

Only extend `opt.py` if you need a new optimizer type not already supported (bootstrap, miprov2, gepa, simba are all covered).

### 8.3 `.env` / `env.template`

Add your use-case-specific data paths:

```env
# <UseCase> Dataset Paths
<USECASE>_DATA_DIR=data/<usecase>_data/images
<USECASE>_GT_DIR=data/<usecase>_data/ground_truth
```

Update `env.template` so other developers know about the new variables.

---

## 9. (Optional) Set Up Baseline & Eval Runs

For mature use cases, create evaluation infrastructure:

### 9.1 Human prompt baselines

```
evaluation/<usecase>_baseline_runs/
├── README.md
├── human_prompts/          # versioned prompt markdown files
├── run_human_prompts/      # scripts to run inference with human-written prompts
└── eval/                   # evaluation scripts + result visualisations
```

### 9.2 Val/test eval on optimized prompts

```
evaluation/<usecase>_run_val_test_set_eval/
├── run_val_test_set_eval_on_optimized_prompt.py
├── run_val_test_set_eval_on_human_runs.py
└── eval_results/
```

These scripts load a saved `.json` model, run inference on the val/test split, compute per-pair metrics, and write structured JSON results.

### 9.3 Grid runner (for systematic experiments)

If you need to sweep over multiple models × optimizer settings × CoT modes, create `run_<usecase>_grid.py` (see `run_dca_grid.py` for the pattern).

---

## 10. Checklist — Files to Create / Modify

### New files

| # | File | Purpose |
|---|------|---------|
| 1 | `signatures/<usecase>.py` | DSPy Signature (input/output contract) |
| 2 | `evaluation/<usecase>_metric.py` | Metric function(s) for optimization |
| 3 | `extract_<usecase>.py` | Main pipeline script |
| 4 | `data/<usecase>_data/README.md` | Dataset documentation |

### Modified files

| # | File | Change |
|---|------|--------|
| 5 | `utils/data_loading.py` | Add `load_<usecase>_dataset()` |
| 6 | `utils/splitting.py` | Add `stratified_split_<usecase>_dataset()` |
| 7 | `evaluation/posthoc_analysis.py` | Add `analyze_<usecase>_model_performance()` |
| 8 | `signatures/__init__.py` | Register new signature import |
| 9 | `evaluation/__init__.py` | Register new metric import |
| 10 | `utils/__init__.py` | Register new loader + splitter imports |
| 11 | `env.template` | Add data path variables |

### No changes needed (usually)

| File | Reason |
|------|--------|
| `config.py` | Generic LM config — shared across all use cases |
| `optimization/opt.py` | Generic optimizer runner — shared across all use cases |
| `optimization/__init__.py` | Already exports everything needed |

---

## 11. Verification & Testing

### Quick smoke test

```bash
# 1. Load data + check split
python -c "
from utils import load_<usecase>_dataset, stratified_split_<usecase>_dataset
ds = load_<usecase>_dataset()
tr, va, te = stratified_split_<usecase>_dataset(ds)
print(f'train={len(tr)} val={len(va)} test={len(te)}')
"

# 2. Baseline eval (no optimization, just measure zero-shot)
python extract_<usecase>.py --optimizer miprov2 --miprov2-auto light --no-cache

# 3. Full optimization
python extract_<usecase>.py --optimizer miprov2 --miprov2-auto medium --save-model --no-cache
```

### What to look for

- [ ] **Data loads without errors** — correct image count, expected class distribution printed.
- [ ] **Split is reproducible** — running twice with same seed produces identical `.cache/<usecase>_split_seed42.json`.
- [ ] **Baseline score is reasonable** — if zero-shot is near 0%, check signature descriptions and metric logic.
- [ ] **Optimization improves score** — at least one strategy should beat baseline by a few percent.
- [ ] **Saved model loads** — verify with `--load-extractor .cache/<saved>.json --skip-load false`.

---

## Appendix: Architecture Diagram

```
extract_<usecase>.py          ← entry point (argparse, logging, pipeline orchestration)
  │
  ├── config.py               ← LM creation, Azure auth, reconfigure_lm()
  │
  ├── signatures/<usecase>.py ← DSPy Signature (inputs → outputs)
  │     └── dspy.ChainOfThought(Signature) or dspy.Predict(Signature)
  │
  ├── utils/
  │     ├── data_loading.py   ← load_<usecase>_dataset() → List[dspy.Example]
  │     └── splitting.py      ← stratified_split_<usecase>_dataset() → train/val/test
  │
  ├── evaluation/
  │     ├── <usecase>_metric.py       ← metric(example, prediction) → float [0,1]
  │     ├── common.py                 ← shared helpers (normalize, similarity, etc.)
  │     └── posthoc_analysis.py       ← analyze_<usecase>_model_performance()
  │
  └── optimization/
        └── opt.py             ← OptimizationRunner (generic — no changes needed)
              ├── BootstrapFewShotWithRandomSearch
              ├── MIPROv2
              ├── GEPA
              └── SIMBA
```

---

*Last updated: 2025-04-01*
