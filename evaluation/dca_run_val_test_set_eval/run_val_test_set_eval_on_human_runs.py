#!/usr/bin/env python3
"""Full DCA evaluation on the test split for all human-prompt runs.

Computes for each run × test-pair:
  - classification_score  (1.0 / 0.0)  — discard_image correct?
  - change_score          (0.0–1.0)    — change-summary quality via judge_summary_metric
  - total_score           (0.6 × cls + 0.4 × change)
  - response_time_seconds (from stored metadata)
  - total_tokens          (from stored metadata)

Aggregates per run, saves JSON results, and produces grouped bar-chart plots.

Usage
-----
# With LLM judge (requires Azure auth):
    python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_human_runs.py

# Skip LLM judge (rule-based only; breakdown cases marked None):
    python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_human_runs.py --skip-llm

# Overwrite existing results (otherwise aborts if results already exist):
    python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_human_runs.py --overwrite

# Re-use a previously saved result JSON (just re-plot):
    python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_human_runs.py --replot
"""

# TODO Full file new

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Path Setup — ensure repo root is on sys.path so we can import config/eval
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
# layout: evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_human_runs.py
#   parents[0] = evaluation/
#   parents[1] = repo root
REPO_ROOT   = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Import DCA metric functions and constants directly.
# dca_metric.py uses a lazy Azure client, so importing it here does NOT spin
# up any Azure connection until judge_summary_metric is actually called.
# ---------------------------------------------------------------------------
from evaluation.dca_metric import (
    FIELD_WEIGHT_CLS,
    FIELD_WEIGHT_CHANGE,
    _try_rule_based_match,
    judge_summary_metric,
    compute_classification_score,
    compute_total_score,
)

# ---------------------------------------------------------------------------
# Matplotlib — non-interactive backend before any other mpl import
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_JSON_DIR   = REPO_ROOT / "data" / "dca_data" / "image_pairs_llm_output_exp1"
DEFAULT_SPLIT_FILE = REPO_ROOT / ".cache" / "dca_split_seed42.json"
OUTPUT_DIR         = SCRIPT_DIR / "eval_results"


# ---------------------------------------------------------------------------
# Runs and display labels
# ---------------------------------------------------------------------------
RUNS: list[str] = [
    "v1_prompt_midas_gpt41_mini",
    # "v1_prompt_midas_gpt41",
    # "v1_prompt_midas_gpt52",
    # "v1_prompt_midas_gpt52_reason_low",
    # "v1_prompt_midas_gpt_5_mini_reason_low",
    "v2_prompt_cr_gpt41_mini",
    # "v2_prompt_cr_gpt41",
    # "v2_prompt_cr_gpt52",
    # "v2_prompt_cr_gpt52_reason_low",
    # "v2_prompt_cr_gpt_5_mini_reason_low",
    "v3_prompt_cr_gpt41_mini",
    # "v3_prompt_cr_gpt41",
    # "v3_prompt_cr_gpt52",
    # "v3_prompt_cr_gpt52_reason_low",
    # "v3_prompt_cr_gpt_5_mini_reason_low",
    "v4_prompt_cr_gpt41_mini",
    # "v4_prompt_cr_gpt41",
    # "v4_prompt_cr_gpt52",
    # "v4_prompt_cr_gpt52_reason_low",
    # "v4_prompt_cr_gpt_5_mini_reason_low",
]

RUN_LABELS: dict[str, str] = {
    "v1_prompt_midas_gpt41_mini":          "v1/GPT-4.1-mini",
    "v1_prompt_midas_gpt41":               "v1/GPT-4.1",
    "v1_prompt_midas_gpt52":               "v1/GPT-5.2",
    "v1_prompt_midas_gpt52_reason_low":    "v1/GPT-5.2-RL",
    "v1_prompt_midas_gpt_5_mini_reason_low": "v1/GPT-5-mini-RL",
    "v2_prompt_cr_gpt41_mini":             "v2/GPT-4.1-mini",
    "v2_prompt_cr_gpt41":                  "v2/GPT-4.1",
    "v2_prompt_cr_gpt52":                  "v2/GPT-5.2",
    "v2_prompt_cr_gpt52_reason_low":       "v2/GPT-5.2-RL",
    "v2_prompt_cr_gpt_5_mini_reason_low":  "v2/GPT-5-mini-RL",
    "v3_prompt_cr_gpt41_mini":             "v3/GPT-4.1-mini",
    "v3_prompt_cr_gpt41":                  "v3/GPT-4.1",
    "v3_prompt_cr_gpt52":                  "v3/GPT-5.2",
    "v3_prompt_cr_gpt52_reason_low":       "v3/GPT-5.2-RL",
    "v3_prompt_cr_gpt_5_mini_reason_low":  "v3/GPT-5-mini-RL",
    "v4_prompt_cr_gpt41_mini":             "v4/GPT-4.1-mini",
    "v4_prompt_cr_gpt41":                  "v4/GPT-4.1",
    "v4_prompt_cr_gpt52":                  "v4/GPT-5.2",
    "v4_prompt_cr_gpt52_reason_low":       "v4/GPT-5.2-RL",
    "v4_prompt_cr_gpt_5_mini_reason_low":  "v4/GPT-5-mini-RL",
}

# ---------------------------------------------------------------------------
# Model groups — colour-coded families, shade per prompt version
# ---------------------------------------------------------------------------
MODEL_GROUPS: list[dict] = [
    {
        "name": "GPT-4.1-mini",
        "runs": [
            "v1_prompt_midas_gpt41_mini",
            "v2_prompt_cr_gpt41_mini",
            "v3_prompt_cr_gpt41_mini",
            "v4_prompt_cr_gpt41_mini",
        ],
        "shades": ["#90CAF9", "#42A5F5", "#1565C0", "#0D2D6B"],
    },
    {
        "name": "GPT-4.1",
        "runs": [
            "v1_prompt_midas_gpt41",
            "v2_prompt_cr_gpt41",
            "v3_prompt_cr_gpt41",
            "v4_prompt_cr_gpt41",
        ],
        "shades": ["#A5D6A7", "#4CAF50", "#1B5E20", "#0A3D0A"],
    },
    {
        "name": "GPT-5.2",
        "runs": [
            "v1_prompt_midas_gpt52",
            "v2_prompt_cr_gpt52",
            "v3_prompt_cr_gpt52",
            "v4_prompt_cr_gpt52",
        ],
        "shades": ["#FFCC80", "#FF9800", "#E65100", "#7F2900"],
    },
    {
        "name": "GPT-5.2\nReason-Low",
        "runs": [
            "v1_prompt_midas_gpt52_reason_low",
            "v2_prompt_cr_gpt52_reason_low",
            "v3_prompt_cr_gpt52_reason_low",
            "v4_prompt_cr_gpt52_reason_low",
        ],
        "shades": ["#CE93D8", "#AB47BC", "#6A1B9A", "#2D004D"],
    },
    {
        "name": "GPT-5-mini\nReason-Low",
        "runs": [
            "v1_prompt_midas_gpt_5_mini_reason_low",
            "v2_prompt_cr_gpt_5_mini_reason_low",
            "v3_prompt_cr_gpt_5_mini_reason_low",
            "v4_prompt_cr_gpt_5_mini_reason_low",
        ],
        "shades": ["#EF9A9A", "#EF5350", "#B71C1C", "#600000"],
    },
]

PROMPT_VERSION_LABELS = ["v1", "v2", "v3", "v4"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "1"}:
            return True
        if v in {"false", "no", "0"}:
            return False
    if isinstance(value, int):
        return bool(value)
    return None


def parse_metadata(meta_str: Any) -> Optional[dict]:
    if meta_str is None:
        return None
    if isinstance(meta_str, dict):
        return meta_str
    if isinstance(meta_str, str):
        try:
            return json.loads(meta_str)
        except Exception:
            return None
    return None


def extract_tokens_and_time(meta: dict) -> tuple[Optional[int], Optional[float]]:
    """Extract (total_tokens, response_time_seconds) from parsed metadata dict.

    Compatible with v1/v2/v3 (key: response.usage) and v4 (same top-level key).
    """
    if not meta:
        return None, None
    usage = (meta.get("response") or {}).get("usage") or {}
    total_tokens = usage.get("total_tokens")
    response_time = (meta.get("timing") or {}).get("response_time_seconds")
    try:
        total_tokens = int(total_tokens) if total_tokens is not None else None
        response_time = float(response_time) if response_time is not None else None
    except (TypeError, ValueError):
        pass
    return total_tokens, response_time


def compute_change_score(
    human: str,
    model: str,
    use_llm: bool,
) -> tuple[Optional[float], str]:
    """Return (score, source) where source ∈ {'rule', 'llm', 'skipped'}.

    If use_llm=False, breakdown cases return (None, 'skipped').
    """

    # 1) Rule-based fast-path
    score, reasoning, matched = _try_rule_based_match(human, model)
    if matched:
        return score, "rule"

    # 3) LLM judge
    if not use_llm:
        return None, "skipped"

    try:
        score = judge_summary_metric(human, model)
    except Exception:
        score = None

    return score, "llm"


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
# FIELD_WEIGHT_CLS / FIELD_WEIGHT_CHANGE / _try_rule_based_match / judge_summary_metric /
# compute_classification_score / compute_total_score are all imported from evaluation.dca_metric above.


def evaluate_run(
    run: str,
    test_records: list[dict],
    use_llm: bool,
) -> dict:
    """Evaluate a single run on all test records.

    Returns a dict with per-pair details and aggregated metrics.
    """
    discard_key = f"{run}_discard_image"
    changes_key = f"{run}_changes"
    meta_key    = f"{run}_metadata"

    per_pair: list[dict] = []
    n_total_pairs = len(test_records)

    for idx, rec in enumerate(test_records, 1):
        pair_name = rec.get("_pair_name", "unknown")

        # Ground truth
        human_discard  = normalize_bool(rec.get("human_label_discard_image"))
        human_changes  = str(rec.get("human_label_changes") or "").strip()

        # Prediction
        pred_discard_raw = rec.get(discard_key)
        pred_changes     = str(rec.get(changes_key) or "").strip()

        if pred_discard_raw is None:
            print(f"    [{idx:03d}/{n_total_pairs}] {pair_name}  [SKIP] no prediction for this run")
            per_pair.append({"pair_name": pair_name, "missing": True})
            continue

        pred_discard = normalize_bool(pred_discard_raw)
        if human_discard is None or pred_discard is None:
            print(f"    [{idx:03d}/{n_total_pairs}] {pair_name}  [SKIP] unparseable discard value")
            per_pair.append({"pair_name": pair_name, "missing": True})
            continue

        # Classification score
        cls_score = compute_classification_score(human_discard, pred_discard)

        # Change score
        change_score, change_source = compute_change_score(
            human_changes, pred_changes, use_llm
        )

        # Total score — match extract_dca.py / dca_comparison_metric exactly:
        #   - LLM was called but returned None (edge case: judge output had 0 counts) →
        #     treat as 0.0 for total, same as `if change_score is None: change_score = 0.0`
        #   - Skipped (--skip-llm): keep total_score=None so it is excluded from avg
        if change_score is not None:
            total_score = compute_total_score(cls_score, change_score)
        elif change_source == "llm" and change_score is None:
            # judge was called but returned None — use 0.0 fallback (matches extract_dca.py)
            total_score = compute_total_score(cls_score, 0.0)
        else:
            total_score = None  # --skip-llm: change score unavailable

        # Metadata: tokens + timing
        meta_str = rec.get(meta_key)
        meta = parse_metadata(meta_str)
        total_tokens, response_time = extract_tokens_and_time(meta or {})

        chg_str = f"{change_score:.3f}" if change_score is not None else "N/A "
        tot_str = f"{total_score:.3f}"  if total_score  is not None else "N/A "
        print(f"    [{idx:03d}/{n_total_pairs}] {pair_name}  "
              f"cls={cls_score:.0f}  chg={chg_str}  tot={tot_str}  src={change_source}",
              flush=True)

        per_pair.append({
            "pair_name":      pair_name,
            "missing":        False,
            "cls_score":      cls_score,
            "change_score":   change_score,
            "total_score":    total_score,
            "change_source":  change_source,
            "total_tokens":   total_tokens,
            "response_time":  response_time
        })

    # Aggregate
    valid = [p for p in per_pair if not p.get("missing")]
    n_total   = len(test_records)
    n_missing = sum(1 for p in per_pair if p.get("missing"))
    n_valid   = len(valid)

    cls_scores    = [p["cls_score"]   for p in valid]
    change_scores = [p["change_score"] for p in valid if p["change_score"] is not None]
    total_scores  = [p["total_score"]  for p in valid if p["total_score"]  is not None]
    tokens        = [p["total_tokens"] for p in valid if p["total_tokens"] is not None]
    times         = [p["response_time"] for p in valid if p["response_time"] is not None]

    source_counts = {"rule": 0, "llm": 0, "skipped": 0}
    for p in valid:
        src = p.get("change_source", "skipped")
        source_counts[src] = source_counts.get(src, 0) + 1

    return {
        "run":                run,
        "label":              RUN_LABELS.get(run, run),
        "n_test_pairs":       n_total,
        "n_evaluated":        n_valid,
        "n_missing":          n_missing,
        "avg_cls_score":      float(np.mean(cls_scores))    if cls_scores    else None,
        "avg_change_score":   float(np.mean(change_scores)) if change_scores else None,
        "avg_total_score":    float(np.mean(total_scores))  if total_scores  else None,
        "avg_total_tokens":   float(np.mean(tokens))        if tokens        else None,
        "avg_response_time":  float(np.mean(times))         if times         else None,
        "n_cls":              len(cls_scores),
        "n_change":           len(change_scores),
        "n_total_score":      len(total_scores),
        "change_source_counts": source_counts,
        "per_pair":           per_pair,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _grouped_metric_bar(ax, metric_key, title, ylabel, results: list[dict], groups: list[dict]):
    """Draw a grouped bar chart for a single metric across model groups."""
    n_groups  = len(groups)
    bar_width = 0.18

    group_centers = np.arange(n_groups)
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * (bar_width + 0.01)

    result_by_run = {r["run"]: r for r in results}

    handles_for_legend = []
    labels_for_legend  = []

    for vi, version_label in enumerate(PROMPT_VERSION_LABELS):
        bars_x = []
        bars_h = []
        bars_c = []
        for gi, group in enumerate(groups):
            run = group["runs"][vi] if vi < len(group["runs"]) else None
            shade = group["shades"][vi] if vi < len(group["shades"]) else "#999999"
            bars_c.append(shade)
            if run and run in result_by_run:
                val = result_by_run[run].get(metric_key)
                bars_h.append(val if val is not None else 0.0)
            else:
                bars_h.append(0.0)
            bars_x.append(group_centers[gi] + offsets[vi])

        ax.bar(bars_x, bars_h, width=bar_width, color=bars_c, edgecolor="white", linewidth=0.5)

        # One legend entry per version (colour from first group)
        patch = plt.Rectangle((0, 0), 1, 1, fc=bars_c[0])
        handles_for_legend.append(patch)
        labels_for_legend.append(version_label)

        # Value labels on bars
        for bx, bh, bc in zip(bars_x, bars_h, bars_c):
            if bh > 0:
                ax.text(bx, bh + 0.005, f"{bh:.2f}", ha="center", va="bottom",
                        fontsize=6, color="black")

    ax.set_xticks(group_centers)
    ax.set_xticklabels([g["name"] for g in groups], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.10)
    ax.yaxis.grid(True, alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(handles_for_legend, labels_for_legend,
              title="Prompt Version", fontsize=7, title_fontsize=8,
              loc="upper right", ncol=2)


def _flat_metric_bar(ax, metric_key, title, ylabel, results: list[dict],
                     unit_scale: float = 1.0, ymax: Optional[float] = None):
    """Draw a flat (ungrouped) bar chart for token / time metrics, coloured by model group."""
    run_keys = [r["run"] for r in results]
    labels   = [RUN_LABELS.get(r, r).replace("/", "\n") for r in run_keys]
    values   = [(res.get(metric_key) or 0.0) * unit_scale for res in results]

    # Assign colour by model group membership
    run_to_shade = {}
    for group in MODEL_GROUPS:
        for vi, run in enumerate(group["runs"]):
            shade = group["shades"][vi] if vi < len(group["shades"]) else "#999999"
            run_to_shade[run] = shade
    colours = [run_to_shade.get(r, "#607D8B") for r in run_keys]

    x = np.arange(len(run_keys))
    ax.bar(x, values, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    if ymax:
        ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)

    for xi, v in zip(x, values):
        if v > 0:
            ax.text(xi, v + ax.get_ylim()[1] * 0.01, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=6, rotation=0)

    # Group dividers (one per model change)
    group_sizes = [len(g["runs"]) for g in MODEL_GROUPS]
    boundary = 0
    for gs in group_sizes[:-1]:
        boundary += gs
        ax.axvline(boundary - 0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)


def plot_all(results: list[dict], output_dir: Path, n_test_pairs: int,
             set_name: str = "test") -> None:
    """Generate and save all plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    set_label = set_name.capitalize()

    # ── Figure 1: Score metrics (3 grouped bar charts stacked vertically) ──
    fig1, axes1 = plt.subplots(3, 1, figsize=(14, 15))
    fig1.suptitle(
        f"DCA Human-Prompt Baseline Evaluation — {set_label} Set (n={n_test_pairs})\n"
        "Grouped by Model Family | Prompt Version v1→v4 (lighter→darker shade)",
        fontsize=11, fontweight="bold", y=0.98,
    )

    _grouped_metric_bar(
        axes1[0], "avg_total_score",
        "Total Score  (0.6 × Classification + 0.4 × Change)",
        "Avg Total Score",
        results, MODEL_GROUPS,
    )
    _grouped_metric_bar(
        axes1[1], "avg_cls_score",
        "Classification Score  (discard_image correct?)",
        "Avg Classification Score",
        results, MODEL_GROUPS,
    )
    _grouped_metric_bar(
        axes1[2], "avg_change_score",
        "Change Score  (judge_summary_metric, precision-60/recall-40 F-beta)",
        "Avg Change Score",
        results, MODEL_GROUPS,
    )

    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    score_path = output_dir / "scores_grouped.png"
    fig1.savefig(score_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {score_path}")

    # ── Figure 2: Efficiency metrics (tokens + time) ──
    fig2, axes2 = plt.subplots(2, 1, figsize=(16, 10))
    fig2.suptitle(
        f"DCA Baseline — Response Time & Token Usage — {set_label} Set (n={n_test_pairs})",
        fontsize=11, fontweight="bold",
    )

    _flat_metric_bar(
        axes2[0], "avg_response_time",
        "Avg Response Time per Pair  (seconds)",
        "Seconds",
        results,
    )
    _flat_metric_bar(
        axes2[1], "avg_total_tokens",
        "Avg Total Tokens per Pair  (prompt + completion)",
        "Tokens",
        results,
    )

    fig2.tight_layout()
    eff_path = output_dir / "efficiency_metrics.png"
    fig2.savefig(eff_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {eff_path}")

    # ── Figure 3: Summary heatmap (all metrics × all runs) ──
    run_labels_short = [RUN_LABELS.get(r["run"], r["run"]) for r in results]
    metric_names = ["Total Score", "Cls Score", "Change Score", "Avg Time (s)", "Avg Tokens (k)"]
    data_matrix  = []
    for r in results:
        row = [
            r.get("avg_total_score")    or 0.0,
            r.get("avg_cls_score")      or 0.0,
            r.get("avg_change_score")   or 0.0,
            r.get("avg_response_time")  or 0.0,
            (r.get("avg_total_tokens")  or 0.0) / 1000.0,
        ]
        data_matrix.append(row)

    data_arr = np.array(data_matrix, dtype=float)  # shape: (n_runs, n_metrics)

    fig3, ax3 = plt.subplots(figsize=(10, max(8, len(results) * 0.45)))
    fig3.suptitle("DCA Baseline — Summary Heatmap (All Runs × Metrics)",
                  fontsize=11, fontweight="bold")

    # Normalise each column separately for colour scaling
    normed = np.zeros_like(data_arr)
    for c in range(data_arr.shape[1]):
        col = data_arr[:, c]
        col_min, col_max = col.min(), col.max()
        span = col_max - col_min
        normed[:, c] = (col - col_min) / span if span > 0 else np.zeros_like(col)

    # Use viridis for scores (higher = better), reverse for time/tokens (lower = better)
    ax3.imshow(normed, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax3.set_xticks(range(len(metric_names)))
    ax3.set_xticklabels(metric_names, fontsize=9)
    ax3.set_yticks(range(len(run_labels_short)))
    ax3.set_yticklabels(run_labels_short, fontsize=8)

    # Annotate cells with actual values
    for ri in range(len(results)):
        for ci, col_vals in enumerate(data_arr[ri]):
            txt = f"{col_vals:.2f}" if ci < 3 else (
                f"{col_vals:.1f}s" if ci == 3 else f"{col_vals:.1f}k"
            )
            ax3.text(ci, ri, txt, ha="center", va="center", fontsize=7,
                     color="black")

    fig3.tight_layout()
    heat_path = output_dir / "summary_heatmap.png"
    fig3.savefig(heat_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {heat_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full DCA evaluation on test split.")
    parser.add_argument("--json-dir",    type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--split-file",  type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--output-dir",  type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-llm",   action="store_true",
                        help="Skip LLM judge calls; breakdown cases get change_score=None")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Overwrite existing eval_results.json if it already exists")
    parser.add_argument("--replot",     action="store_true",
                        help="Load existing results JSON and re-generate plots only")
    parser.add_argument("--results-json", type=Path, default=None,
                        help="Path to existing results JSON for --replot")
    parser.add_argument("--which-set", type=str, choices=["test", "val", "both"], default="test",
                        help="Which split to evaluate: test, val, or both (default: test)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_llm = not args.skip_llm

    # ── Guard: abort if results already exist and --overwrite not set ─────────
    if not args.overwrite and not args.replot:
        _sets_to_check = ["val", "test"] if args.which_set == "both" else [args.which_set]
        for _sname in _sets_to_check:
            _out = args.output_dir / f"eval_results_{_sname}.json"
            if _out.exists():
                print(f"[ERROR] Results already exist: {_out}")
                print("  Use --overwrite to re-run and overwrite, or --replot to just regenerate plots.")
                sys.exit(1)

    # ── Replot mode ──────────────────────────────────────────────────────────
    if args.replot:
        results_file = args.results_json or (args.output_dir / "eval_results.json")
        if not results_file.exists():
            print(f"[ERROR] --replot: results file not found: {results_file}")
            sys.exit(1)
        data = json.loads(results_file.read_text(encoding="utf-8"))
        results  = data["results"]
        n_test   = data["meta"]["n_test_pairs"]
        set_name = data["meta"].get("which_set", "test")
        print(f"Replotting {len(results)} runs from {results_file.name} …")
        plot_all(results, args.output_dir / set_name, n_test, set_name=set_name)
        return
    
    # ── Load split (support both old and new manifest formats) ────────────────
    if not args.split_file.exists():
        print(f"[ERROR] Split file not found: {args.split_file}")
        sys.exit(1)

    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    # Manifest may be either: {"test": {"pair_names": [...]}, "val": {...}} or
    # the older format that also contains a top-level "train" key. Be flexible.
    test_pair_names = split.get("test", {}).get("pair_names") or split.get("test_pair_numbers") or []
    if isinstance(test_pair_names, list) and test_pair_names and isinstance(test_pair_names[0], int):
        # accidental numeric list stored; convert to strings with prefix if needed
        test_pair_names = [f"pair_{n:03d}" for n in test_pair_names]
    val_pair_names = split.get("val", {}).get("pair_names") or split.get("val_pair_numbers") or []
    if isinstance(val_pair_names, list) and val_pair_names and isinstance(val_pair_names[0], int):
        val_pair_names = [f"pair_{n:03d}" for n in val_pair_names]

    def _load_records(pair_names: list[str]) -> tuple[list[dict], int]:
        records: list[dict] = []
        missing = 0
        for pname in pair_names:
            f = args.json_dir / f"{pname}.json"
            if not f.exists():
                print(f"  [WARN] Missing: {f.name}")
                missing += 1
                continue
            rec = json.loads(f.read_text(encoding="utf-8"))
            rec["_pair_name"] = pname
            records.append(rec)
        return records, missing

    # Determine which sets to evaluate
    which = args.which_set
    evaluate_test = which in ("test", "both")
    evaluate_val = which in ("val", "both")

    if evaluate_test:
        print(f"Test split: {len(test_pair_names)} pairs (seed={split.get('seed','?')})")
    if evaluate_val:
        print(f"Validation split: {len(val_pair_names)} pairs (seed={split.get('seed','?')})")

    if not args.json_dir.exists():
        print(f"[ERROR] JSON dir not found: {args.json_dir}")
        sys.exit(1)

    test_records, test_missing = ([], 0)
    val_records, val_missing = ([], 0)
    if evaluate_test:
        test_records, test_missing = _load_records(test_pair_names)
        print(f"Loaded {len(test_records)} test pair JSONs ({test_missing} missing files)")
    if evaluate_val:
        val_records, val_missing = _load_records(val_pair_names)
        print(f"Loaded {len(val_records)} val pair JSONs ({val_missing} missing files)")

    def _evaluate_and_save(records: list[dict], set_name: str) -> list[dict]:
        if not records:
            print(f"\nNo records for set: {set_name} — skipping.")
            return []

        print(f"\nEvaluating {len(RUNS)} runs on {len(records)} {set_name} pairs …")
        print(f"  LLM judge: {'ENABLED' if use_llm else 'DISABLED (--skip-llm)'}\n")

        results: list[dict] = []
        for i, run in enumerate(RUNS, 1):
            t0 = time.time()
            print(f"\n  [{i:02d}/{len(RUNS)}] {run}", flush=True)
            res = evaluate_run(run, records, use_llm)
            elapsed = time.time() - t0
            chg_str = f"{res['avg_change_score']:.3f}" if res['avg_change_score'] is not None else "N/A"
            tot_str = f"{res['avg_total_score']:.3f}"  if res['avg_total_score']  is not None else "N/A"
            print(
                f"  → avg: cls={res['avg_cls_score']:.3f}  "
                f"chg={chg_str:>6}  "
                f"tot={tot_str:>6}  "
                f"src={res['change_source_counts']}  "
                f"[{elapsed:.1f}s]"
            )
            results.append(res)

        # Save summary JSON (no per-pair)
        results_no_per_pair = []
        for r in results:
            r_clean = {k: v for k, v in r.items() if k != "per_pair"}
            results_no_per_pair.append(r_clean)

        results_path = args.output_dir / f"eval_results_{set_name}.json"
        saved_data = {
            "meta": {
                "n_test_pairs": len(records),
                "split_file":    str(args.split_file),
                "json_dir":      str(args.json_dir),
                "which_set":     set_name,
                "skip_llm":      not use_llm,
                "runs":          RUNS,
                "field_weight_cls":    FIELD_WEIGHT_CLS,
                "field_weight_change": FIELD_WEIGHT_CHANGE,
            },
            "results": results_no_per_pair,
        }
        results_path.write_text(json.dumps(saved_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResults JSON -> {results_path}")

        # Full per-pair results
        full_path = args.output_dir / f"eval_results_full_{set_name}.json"
        full_data = saved_data.copy()
        full_data["results"] = results
        full_path.write_text(json.dumps(full_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Full results -> {full_path}")

        # Summary table
        print("\n" + "=" * 100)
        print(f"{'Run':<42} {'Cls':>6} {'Chg':>6} {'Tot':>6} {'Time(s)':>8} {'Tokens':>8} {'n_eval':>7}")
        print("-" * 100)
        for r in results:
            c  = f"{r['avg_cls_score']:.3f}"   if r['avg_cls_score']    is not None else "N/A"
            ch = f"{r['avg_change_score']:.3f}" if r['avg_change_score'] is not None else "N/A"
            t  = f"{r['avg_total_score']:.3f}"  if r['avg_total_score']  is not None else "N/A"
            ti = f"{r['avg_response_time']:.1f}" if r['avg_response_time'] is not None else "N/A"
            tk = f"{r['avg_total_tokens']:.0f}"  if r['avg_total_tokens']  is not None else "N/A"
            print(f"  {r['label']:<40} {c:>6} {ch:>6} {t:>6} {ti:>8} {tk:>8} {r['n_evaluated']:>7}")
        print("=" * 100)

        # Plots in a subdirectory per set
        out_plot_dir = args.output_dir / set_name
        print("\nGenerating plots …")
        plot_all(results, out_plot_dir, len(records), set_name=set_name)

        return results

    # Run requested sets
    if evaluate_test:
        _ = _evaluate_and_save(test_records, "test")
    if evaluate_val:
        _ = _evaluate_and_save(val_records, "val")

    print("\nDone.")


if __name__ == "__main__":
    main()
