# Evaluation

This folder contains all evaluation metrics, analysis scripts, and evaluation runners for the DCA, Japanese Order, and YANASE tasks.

---

## Folder Structure

```
evaluation/
├── dca_metric.py              # DCA composite scoring metric (used everywhere)
├── jo_metric.py               # Japanese Order field extraction metric
├── yanase_metric.py           # YANASE order extraction metric
├── common.py                  # Shared helpers: text normalization, fuzzy matching
├── posthoc_analysis.py        # Post-hoc performance analysis and reporting
├── dca_baseline_runs/         # Human-prompt baseline experiments for DCA
│   ├── human_prompts/         #   Prompt markdown files (v0–v5)
│   ├── run_human_prompts/     #   Single run, batch orchestration, category coding scripts
│   ├── eval/                  #   Classification and change-summary evaluation scripts
│   └── README.md              #   Detailed usage guide
└── dca_run_val_test_set_eval/ # Val/test set evaluation scripts for DCA
    ├── run_val_test_set_eval_on_human_runs.py
    ├── run_val_test_set_eval_on_optimized_prompt.py
    ├── run_cache_grid_eval.py
    └── README.md
```

---

## DCA Metric — `dca_metric.py`

The core scoring function for all DCA experiments. Used by the tournament pipeline, `extract_dca.py`, and all evaluation scripts.

### Scoring Formula

```
total_score = 0.6 * classification_score + 0.4 * change_score
```

Both components are in [0, 1]. The composite score is in [0, 1].

### Classification Score

`compute_classification_score(expected: bool, predicted: bool) -> float`

Returns `1.0` if the predicted `discard_image` value matches the expected ground truth, `0.0` otherwise. Parsed from the model output string (`"true"` → `True`, anything else → `False`).

### Change Summary Score

`judge_summary_metric(human: str, model: str) -> float`

Two-tier scoring:

**Tier 1 — Rule-based fast path** (no LLM call):
Handles special-case outputs by exact/fuzzy string matching:

| Human label | Model output | Score |
|-------------|-------------|-------|
| Special case A | Correct special case string | 1.0 |
| Special case A | Different special case or breakdown | 0.0 |
| Breakdown expected | Special case string returned | 0.0 |

Special cases recognized: `NoMatchFound`, `Could not infer`, `Image has been crossed out`, `Both images are non informational...`, `Images are identical - no changes`.

**Tier 2 — LLM judge** (for breakdown changes, i.e., `OVERVIEW: ... | DETAILS: ...`):

Sends a structured prompt to the Azure OpenAI judge model (default: `DCA_JUDGE_MODEL_NAME` from environment). The judge returns JSON:
```json
{
  "correct_matches": [...],
  "missed_changes": [...],
  "hallucinated_changes": [...]
}
```

Score = weighted F-beta with precision weight 60%, recall weight 40%:
```
precision = correct / (correct + hallucinated)
recall    = correct / (correct + missed)
f_beta    = 5 * P * R / (2 * P + 3 * R)
```

Results are cached in `.cache/judge_summary_cache.json` keyed by SHA-256 of `(human, model)` to avoid duplicate LLM calls across repeated evaluations.

### Main Metric Function

```python
from evaluation.dca_metric import dca_comparison_metric

score = dca_comparison_metric(example, prediction, trace=None)
# Returns float in [0, 1]
```

- `example`: DSPy Example with `expected_discard_image` (str "true"/"false") and `expected_changes` (str)
- `prediction`: DSPy Prediction with `discard_image` (str) and `changes` (str)
- `trace`: If not None (optimization context), returns a bool threshold pass/fail instead of raw score

### Metric with Feedback

```python
from evaluation.dca_metric import dca_comparison_metric_with_feedback

result = dca_comparison_metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None)
# Returns dspy.Prediction(score=..., feedback=...) when pred_name is set
# Returns float when pred_name is None
```

Used by GEPA and MIPROv2. The `feedback` field contains a natural-language explanation of what was correct and what was wrong:
- Classification: `CORRECT`, `FALSE NEGATIVE (should be 'false', predicted 'true')`, or `FALSE POSITIVE`
- Change summary: `Exact match`, `Good (score X.XX)`, `Partial (score X.XX)`, or `No match (score 0.00)`

### Hybrid Metric

```python
from evaluation.dca_metric import make_hybrid_metric

hybrid_metric, hybrid_metric_with_feedback = make_hybrid_metric(
    base_metric=dca_comparison_metric,
    base_metric_with_feedback=dca_comparison_metric_with_feedback,
    score_weight=0.5,
    length_weight=0.5,
    max_prompt_tokens=4000,
)
```

The hybrid metric adds a prompt-length penalty during optimization:
```
hybrid_score = score_weight * dca_score + length_weight * max(0, 1 - prompt_tokens / max_prompt_tokens)
```

During plain evaluation (`trace=None`) the hybrid metric returns the raw DCA score unchanged — the length penalty only applies during `compile()`. Enable via `--use-hybrid-metric` in `run_tournament.py` or `extract_dca.py`.

### Constants

| Name | Value | Description |
|------|-------|-------------|
| `FIELD_WEIGHT_CLS` | `0.6` | Weight for classification in composite score |
| `FIELD_WEIGHT_CHANGE` | `0.4` | Weight for change summary in composite score |
| `MODEL` | from env | Azure OpenAI deployment used as LLM judge |

---

## See Also

- [DCA Baseline Runs](dca_baseline_runs/README.md) — human-prompt experiments
- [Val/Test Set Evaluation](dca_run_val_test_set_eval/README.md) — evaluate saved models on held-out sets
- [Tournament Pipeline](../dca_tournaments/README.md) — automated multi-stage prompt optimization
