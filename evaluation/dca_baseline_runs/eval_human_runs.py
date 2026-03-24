"""
Evaluation script for DCA LLM runs.

Computes for each of the 7 runs:
  1. Binary classification metrics for discard_image vs human_label_discard_image
  2. Average token usage (prompt, completion, total) from metadata
  3. Average response_time_seconds from metadata
  4. Failure count (status != "success" OR missing key)

Generates:
  - Console summary table
  - classification_metrics.png
  - token_usage.png
  - response_time.png
  - confusion_matrices.png
  - eval_summary.csv

Usage:
    python evaluation/dca_baseline_runs/eval_human_runs.py
    python evaluation/dca_baseline_runs/eval_human_runs.py --json-dir path/to/jsons --output-dir evaluation/dca_baseline_runs/eval_results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for all environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_JSON_DIR = REPO_ROOT / "data" / "dca_data" / "image_pairs_llm_output_exp1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation" / "dca_baseline_runs" / "eval_results"

# ---------------------------------------------------------------------------
# The 7 runs to evaluate (key prefixes)
# ---------------------------------------------------------------------------
RUNS = [
    # "v0_prompt_midas_OLD_gpt41",
    # "v0_prompt_midas_gpt41_mini",
    # "v0_prompt_midas_gpt41",
    "v1_prompt_midas_gpt41_mini",
    "v1_prompt_midas_gpt41",
    "v1_prompt_midas_gpt_5_mini_reason_low",
    "v1_prompt_midas_gpt52",
    "v1_prompt_midas_gpt52_reason_low",
    "v2_prompt_cr_gpt41_mini",
    "v2_prompt_cr_gpt41",
    "v2_prompt_cr_gpt_5_mini_reason_low",
    "v2_prompt_cr_gpt52",
    "v2_prompt_cr_gpt52_reason_low",
    # "v3_prompt_cr_OLD_gpt52",
    # "v3_prompt_cr_OLD_gpt41",
    "v3_prompt_cr_gpt41_mini",
    "v3_prompt_cr_gpt41",
    "v3_prompt_cr_gpt_5_mini_reason_low",
    "v3_prompt_cr_gpt52",
    "v3_prompt_cr_gpt52_reason_low",
    # ---- v4 multi-step pipeline (uncomment after running run_v4_human_prompt.py) ----
    "v4_prompt_cr_gpt41_mini",
    "v4_prompt_cr_gpt41",
    "v4_prompt_cr_gpt52",
    "v4_prompt_cr_gpt52_reason_low",
    "v4_prompt_cr_gpt_5_mini_reason_low",
]

# Short display labels for plots
RUN_LABELS = {
    # "v0_prompt_midas_OLD_gpt41":  "v0OLD/gpt41",
    # "v0_prompt_midas_gpt41_mini": "v0/gpt41m",
    # "v0_prompt_midas_gpt41":      "v0/gpt41",
    "v1_prompt_midas_gpt41_mini": "v1/gpt41m",
    "v1_prompt_midas_gpt41":      "v1/gpt41",
    "v1_prompt_midas_gpt52":      "v1/gpt52",
    "v2_prompt_cr_gpt41_mini":    "v2/gpt41m",
    "v2_prompt_cr_gpt41":         "v2/gpt41",
    "v2_prompt_cr_gpt52":         "v2/gpt52",
    # "v3_prompt_cr_OLD_gpt52":     "v3OLD/gpt52",
    # "v3_prompt_cr_OLD_gpt41":     "v3OLD/gpt41",
    "v3_prompt_cr_gpt41_mini":    "v3/gpt41m",
    "v3_prompt_cr_gpt41":         "v3/gpt41",
    "v3_prompt_cr_gpt52":         "v3/gpt52",
    "v1_prompt_midas_gpt52_reason_low": "v1/gpt52_r_low",
    "v2_prompt_cr_gpt52_reason_low": "v2/gpt52_r_low",
    "v3_prompt_cr_gpt52_reason_low": "v3/gpt52_r_low",
    "v1_prompt_midas_gpt_5_mini_reason_low": "v1/gpt5_m_r_low",
    "v2_prompt_cr_gpt_5_mini_reason_low": "v2/gpt5_m_r_low",
    "v3_prompt_cr_gpt_5_mini_reason_low": "v3/gpt5_m_r_low",
    # v4 labels
    "v4_prompt_cr_gpt41_mini":    "v4/gpt41m",
    "v4_prompt_cr_gpt41":         "v4/gpt41",
    "v4_prompt_cr_gpt52":         "v4/gpt52",
    "v4_prompt_cr_gpt52_reason_low":    "v4/gpt52_r_low",
    "v4_prompt_cr_gpt_5_mini_reason_low": "v4/gpt5_m_r_low",
}

# ---------------------------------------------------------------------------
# Model groups for grouped bar plots
# Each group = one model family.  Runs within a group share a color family,
# progressing light→dark for v1→v4 (older→newer prompt version).
# Only runs present in the active RUNS list are rendered; missing slots skipped.
# ---------------------------------------------------------------------------
MODEL_GROUPS = [
    {
        "name": "GPT-4.1-mini",
        "runs": [
            "v1_prompt_midas_gpt41_mini",
            "v2_prompt_cr_gpt41_mini",
            "v3_prompt_cr_gpt41_mini",
            "v4_prompt_cr_gpt41_mini",
        ],
        "shades": ["#90CAF9", "#42A5F5", "#1565C0", "#0D2D6B"],  # Blues
    },
    {
        "name": "GPT-4.1",
        "runs": [
            "v1_prompt_midas_gpt41",
            "v2_prompt_cr_gpt41",
            "v3_prompt_cr_gpt41",
            "v4_prompt_cr_gpt41",
        ],
        "shades": ["#A5D6A7", "#4CAF50", "#1B5E20", "#0A3D0A"],  # Greens
    },
    {
        "name": "GPT-5.2",
        "runs": [
            "v1_prompt_midas_gpt52",
            "v2_prompt_cr_gpt52",
            "v3_prompt_cr_gpt52",
            "v4_prompt_cr_gpt52",
        ],
        "shades": ["#FFCC80", "#FF9800", "#E65100", "#7F2900"],  # Oranges
    },
    {
        "name": "GPT-5.2\nReason Low",
        "runs": [
            "v1_prompt_midas_gpt52_reason_low",
            "v2_prompt_cr_gpt52_reason_low",
            "v3_prompt_cr_gpt52_reason_low",
            "v4_prompt_cr_gpt52_reason_low",
        ],
        "shades": ["#CE93D8", "#AB47BC", "#6A1B9A", "#2D004D"],  # Purples
    },
    {
        "name": "GPT-5-mini\nReason Low",
        "runs": [
            "v1_prompt_midas_gpt_5_mini_reason_low",
            "v2_prompt_cr_gpt_5_mini_reason_low",
            "v3_prompt_cr_gpt_5_mini_reason_low",
            "v4_prompt_cr_gpt_5_mini_reason_low",
        ],
        "shades": ["#EF9A9A", "#EF5350", "#B71C1C", "#600000"],  # Reds
    },
]

_FALLBACK_COLORS = ["#607D8B", "#795548", "#00BCD4", "#FF5722", "#3F51B5", "#009688"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsons(json_dir: Path) -> list[dict[str, Any]]:
    """Load all pair JSON files sorted by pair index."""
    files = sorted(json_dir.glob("pair_*.json"))
    if not files:
        print(f"[ERROR] No pair_*.json files found in {json_dir}", file=sys.stderr)
        sys.exit(1)
    records = []
    for f in files:
        try:
            records.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"[WARN] Could not read {f.name}: {exc}", file=sys.stderr)
    print(f"Loaded {len(records)} JSON files from {json_dir}")
    return records


def parse_metadata(meta_str: Any) -> Optional[dict]:
    """Parse the {prefix}_metadata field (stored as compact JSON string)."""
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


def normalize_bool(value: Any) -> Optional[bool]:
    """Coerce various representations to bool."""
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


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_run_data(records: list[dict], run: str) -> dict:
    """
    Extract all metrics for one run from all pair records.

    Returns dict with:
        y_true, y_pred           : lists of booleans (for pairs where both present)
        prompt_tokens            : list of ints (from metadata)
        completion_tokens        : list of ints
        total_tokens             : list of ints
        response_times           : list of floats (response_time_seconds)
        n_total                  : total pairs in JSON dir
        n_missing_pred           : pairs where {run}_discard_image is absent
        n_failed                 : pairs where metadata status == "failed"
        n_no_metadata            : pairs where metadata key entirely absent
        pairs_with_metadata      : count of pairs that have metadata
    """
    discard_key = f"{run}_discard_image"
    meta_key = f"{run}_metadata"

    y_true: list[bool] = []
    y_pred: list[bool] = []
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    total_tokens: list[int] = []
    reasoning_tokens: list[int] = []
    response_times: list[float] = []

    n_total = len(records)
    n_missing_pred = 0
    n_failed = 0
    n_no_metadata = 0

    for rec in records:
        human = normalize_bool(rec.get("human_label_discard_image"))
        pred_raw = rec.get(discard_key)

        # Classification pair
        if pred_raw is None:
            n_missing_pred += 1
        else:
            pred = normalize_bool(pred_raw)
            if human is not None and pred is not None:
                y_true.append(human)
                y_pred.append(pred)

        # Metadata
        meta_str = rec.get(meta_key)
        if meta_str is None:
            n_no_metadata += 1
        else:
            meta = parse_metadata(meta_str)
            if meta is None:
                n_no_metadata += 1
                continue

            # Status
            status = meta.get("status", "success")
            if status != "success":
                n_failed += 1

            # Tokens — from response.usage
            usage = (meta.get("response") or {}).get("usage") or {}
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            tt = usage.get("total_tokens")
            if pt is not None:
                prompt_tokens.append(int(pt))
            if ct is not None:
                completion_tokens.append(int(ct))
            if tt is not None:
                total_tokens.append(int(tt))
            rt_tok = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            if rt_tok is not None:
                reasoning_tokens.append(int(rt_tok))

            # Response time
            rt = (meta.get("timing") or {}).get("response_time_seconds")
            if rt is not None:
                try:
                    response_times.append(float(rt))
                except (TypeError, ValueError):
                    pass

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "response_times": response_times,
        "n_total": n_total,
        "n_missing_pred": n_missing_pred,
        "n_failed": n_failed,
        "n_no_metadata": n_no_metadata,
        "pairs_with_metadata": n_total - n_no_metadata,
    }


# ---------------------------------------------------------------------------
# Classification metrics (no sklearn dependency)
# ---------------------------------------------------------------------------

def classification_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    """
    Binary classification metrics.
    Positive class = True (discard_image=True → keep/discard).
    """
    if not y_true:
        return {k: None for k in ["accuracy", "precision", "recall", "f1",
                                   "tp", "fp", "tn", "fn", "n_evaluated",
                                   "support_pos", "support_neg"]}

    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    n = len(y_true)

    accuracy  = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "accuracy":    round(accuracy,  4),
        "precision":   round(precision, 4),
        "recall":      round(recall,    4),
        "f1":          round(f1,        4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_evaluated": n,
        "support_pos": tp + fn,   # actual positives
        "support_neg": tn + fp,   # actual negatives
    }


def safe_mean(lst: list) -> Optional[float]:
    return round(sum(lst) / len(lst), 2) if lst else None


def extended_metrics(cm: dict) -> Optional[dict]:
    """Per-class + macro + weighted metrics (equivalent to sklearn classification_report)."""
    if cm["n_evaluated"] is None:
        return None
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    n = cm["n_evaluated"]
    sup_false = tn + fp
    sup_true  = tp + fn

    # False class (negative)
    prec_f = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_f  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_f   = 2 * prec_f * rec_f / (prec_f + rec_f) if (prec_f + rec_f) > 0 else 0.0

    # True class (positive — already in cm)
    prec_t = cm["precision"]
    rec_t  = cm["recall"]
    f1_t   = cm["f1"]

    # Macro avg: simple average of both classes
    mac_p = (prec_f + prec_t) / 2
    mac_r = (rec_f  + rec_t)  / 2
    mac_f = (f1_f   + f1_t)   / 2

    # Weighted avg: weighted by support (sample count per class)
    w_p = (prec_f * sup_false + prec_t * sup_true) / n
    w_r = (rec_f  * sup_false + rec_t  * sup_true) / n
    w_f = (f1_f   * sup_false + f1_t   * sup_true) / n

    return {
        "false":        {"precision": round(prec_f, 2), "recall": round(rec_f, 2), "f1": round(f1_f, 2), "support": sup_false},
        "true":         {"precision": round(prec_t, 2), "recall": round(rec_t, 2), "f1": round(f1_t, 2), "support": sup_true},
        "accuracy":     cm["accuracy"],
        "n":            n,
        "macro_avg":    {"precision": round(mac_p, 2), "recall": round(mac_r, 2), "f1": round(mac_f, 2), "support": n},
        "weighted_avg": {"precision": round(w_p, 2),   "recall": round(w_r, 2),   "f1": round(w_f, 2),   "support": n},
    }


# ---------------------------------------------------------------------------
# Classification report printer
# ---------------------------------------------------------------------------

def print_classification_reports(results: dict) -> None:
    """Print per-class breakdown in sklearn-style classification_report format."""
    sep = "=" * 70
    hdr = f"  {'Class / Metric':<22}  {'Precision':>9}  {'Recall':>9}  {'F1-score':>9}  {'Support':>8}"
    div = "  " + "-" * 62

    print("\n" + sep)
    print("CLASSIFICATION REPORTS  (per-class breakdown)")
    print(sep)

    for run in RUNS:
        label = RUN_LABELS.get(run, run)
        d   = results[run]
        cm  = d["clf"]
        ext = extended_metrics(cm)
        if ext is None:
            continue
        print(f"\nRun: {label}")
        print(div)
        print(hdr)
        print(div)
        for class_name, class_key in [("Discard=False", "false"), ("Discard=True", "true")]:
            c = ext[class_key]
            print(f"  {class_name:<22}  {c['precision']:>9.2f}  {c['recall']:>9.2f}  {c['f1']:>9.2f}  {c['support']:>8}")
        print()
        print(f"  {'accuracy':<22}  {'':>9}  {'':>9}  {ext['accuracy']:>9.4f}  {ext['n']:>8}")
        for avg_name, avg_key in [("macro avg", "macro_avg"), ("weighted avg", "weighted_avg")]:
            c = ext[avg_key]
            print(f"  {avg_name:<22}  {c['precision']:>9.2f}  {c['recall']:>9.2f}  {c['f1']:>9.2f}  {c['support']:>8}")

    print("\n" + sep + "\n")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(results: dict[str, dict]) -> None:
    """Print a formatted summary table to stdout."""
    sep = "-" * 127
    header = (
        f"{'Run':<28} {'N_eval':>6} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} "
        f"{'AvgPTok':>8} {'AvgCTok':>8} {'AvgRTok':>8} {'AvgTTok':>8} {'AvgRT(s)':>9} "
        f"{'Failed':>7} {'NoMeta':>7}"
    )
    print("\n" + sep)
    print(header)
    print(sep)
    for run, d in results.items():
        cm = d["clf"]
        label = RUN_LABELS.get(run, run)
        n_eval   = cm["n_evaluated"] if cm["n_evaluated"] is not None else "-"
        acc      = f"{cm['accuracy']:.4f}"  if cm["accuracy"]  is not None else "-"
        prec     = f"{cm['precision']:.4f}" if cm["precision"] is not None else "-"
        rec      = f"{cm['recall']:.4f}"    if cm["recall"]    is not None else "-"
        f1       = f"{cm['f1']:.4f}"        if cm["f1"]        is not None else "-"
        avg_pt   = f"{d['avg_prompt_tokens']:.0f}"      if d["avg_prompt_tokens"]      is not None else "-"
        avg_ct   = f"{d['avg_completion_tokens']:.0f}"  if d["avg_completion_tokens"]  is not None else "-"
        avg_rt_k = f"{d['avg_reasoning_tokens']:.0f}"   if d["avg_reasoning_tokens"]   is not None else "-"
        avg_tt   = f"{d['avg_total_tokens']:.0f}"       if d["avg_total_tokens"]       is not None else "-"
        avg_rt   = f"{d['avg_response_time']:.2f}"      if d["avg_response_time"]      is not None else "-"
        failed   = d["n_failed"]
        no_meta  = d["n_no_metadata"]
        print(
            f"{label:<28} {n_eval:>6} {acc:>6} {prec:>6} {rec:>6} {f1:>6} "
            f"{avg_pt:>8} {avg_ct:>8} {avg_rt_k:>8} {avg_tt:>8} {avg_rt:>9} "
            f"{failed:>7} {no_meta:>7}"
        )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(results: dict[str, dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "eval_summary.csv"
    rows = []
    header = [
        "run", "n_evaluated", "accuracy", "precision", "recall", "f1",
        "tp", "fp", "tn", "fn",
        "avg_prompt_tokens", "avg_completion_tokens", "avg_reasoning_tokens", "avg_total_tokens",
        "avg_response_time_s", "n_failed", "n_no_metadata", "n_total"
    ]
    rows.append(",".join(header))
    for run, d in results.items():
        cm = d["clf"]
        row = [
            run,
            cm["n_evaluated"],
            cm["accuracy"],
            cm["precision"],
            cm["recall"],
            cm["f1"],
            cm["tp"],
            cm["fp"],
            cm["tn"],
            cm["fn"],
            d["avg_prompt_tokens"],
            d["avg_completion_tokens"],
            d["avg_reasoning_tokens"],
            d["avg_total_tokens"],
            d["avg_response_time"],
            d["n_failed"],
            d["n_no_metadata"],
            d["n_total"],
        ]
        rows.append(",".join("" if v is None else str(v) for v in row))
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"CSV saved → {csv_path}")


# ---------------------------------------------------------------------------
# Plots — grouped by model family
# ---------------------------------------------------------------------------

import matplotlib.patches as mpatches


def _build_grouped_layout(results: dict) -> dict:
    """
    Compute x-axis positions, bar colors, and tick labels for grouped bar charts.

    Runs belonging to the same MODEL_GROUP are placed adjacent; a wider gap
    separates groups.  Runs not in any MODEL_GROUP are appended as "Other".

    Returns dict:
      positions     : list[float]  — x centre per run, in plot order
      colors        : list[str]    — bar colour per run
      labels        : list[str]    — x-tick label per run
      runs_ordered  : list[str]    — run keys in plot order
      group_spans   : list[dict]   — {name, x_start, x_end, x_center, mid_color}
      group_sep_x   : list[float]  — x midpoints between groups  (separator lines)
    """
    BAR_SPACING = 1.0   # distance between bar centres within a group
    GROUP_GAP   = 1.8   # extra space inserted between consecutive groups

    positions:    list[float] = []
    colors:       list[str]   = []
    labels:       list[str]   = []
    runs_ordered: list[str]   = []
    group_spans:  list[dict]  = []
    group_sep_x:  list[float] = []

    results_runs  = set(results.keys())
    already_placed: set[str] = set()
    cur_x = 0.0
    last_end_x: float | None = None

    for group in MODEL_GROUPS:
        active = [(i, r) for i, r in enumerate(group["runs"]) if r in results_runs]
        if not active:
            continue

        grp_start = cur_x
        for shade_idx, run in active:
            positions.append(cur_x)
            colors.append(group["shades"][shade_idx % len(group["shades"])])
            labels.append(RUN_LABELS.get(run, run))
            runs_ordered.append(run)
            already_placed.add(run)
            cur_x += BAR_SPACING

        grp_end   = cur_x - BAR_SPACING
        mid_shade = group["shades"][len(active) // 2 % len(group["shades"])]
        group_spans.append({
            "name":     group["name"],
            "x_start":  grp_start,
            "x_end":    grp_end,
            "x_center": (grp_start + grp_end) / 2,
            "mid_color": mid_shade,
        })
        if last_end_x is not None:
            group_sep_x.append((last_end_x + grp_start) / 2)
        last_end_x = grp_end
        cur_x += GROUP_GAP

    # Orphan runs not captured by any MODEL_GROUP
    orphans = [r for r in results if r not in already_placed]
    if orphans:
        grp_start = cur_x
        for ci, run in enumerate(orphans):
            positions.append(cur_x)
            colors.append(_FALLBACK_COLORS[ci % len(_FALLBACK_COLORS)])
            labels.append(RUN_LABELS.get(run, run))
            runs_ordered.append(run)
            cur_x += BAR_SPACING
        grp_end = cur_x - BAR_SPACING
        group_spans.append({
            "name":     "Other",
            "x_start":  grp_start,
            "x_end":    grp_end,
            "x_center": (grp_start + grp_end) / 2,
            "mid_color": _FALLBACK_COLORS[0],
        })
        if last_end_x is not None:
            group_sep_x.append((last_end_x + grp_start) / 2)

    return {
        "positions":    positions,
        "colors":       colors,
        "labels":       labels,
        "runs_ordered": runs_ordered,
        "group_spans":  group_spans,
        "group_sep_x":  group_sep_x,
    }


def _vals_for_layout(results: dict, layout: dict,
                     key: str, clf_key: str | None = None) -> list:
    """Extract metric values in layout["runs_ordered"] order."""
    vals = []
    for run in layout["runs_ordered"]:
        d = results[run]
        v = d["clf"].get(clf_key) if clf_key else d.get(key)
        vals.append(v if v is not None else 0.0)
    return vals


def _setup_grouped_ax(ax, layout: dict, bar_width: float = 0.7) -> None:
    """
    Apply xticks, group separator lines, faint background rectangles,
    and group-name annotations to an already-drawn axes.
    Call this AFTER drawing bars so ylim is set correctly.
    """
    positions = layout["positions"]
    ax.set_xticks(positions)
    ax.set_xticklabels(layout["labels"], rotation=90, ha="center", fontsize=11)
    ax.set_xlim(positions[0] - bar_width * 1.0,
                positions[-1] + bar_width * 1.0)

    ylo, yhi = ax.get_ylim()

    # Vertical dashed separator between groups
    for sx in layout["group_sep_x"]:
        ax.axvline(sx, color="#CCCCCC", linewidth=1.0,
                   linestyle="--", zorder=0, alpha=0.9)

    # Faint background tint + bold group-name label at top of each group
    for span in layout["group_spans"]:
        ax.axvspan(
            span["x_start"] - bar_width * 0.52,
            span["x_end"]   + bar_width * 0.52,
            alpha=0.06, color=span["mid_color"], zorder=0,
        )
        ax.text(
            span["x_center"], yhi * 0.995,
            span["name"],
            ha="center", va="top",
            fontsize=11, fontweight="bold",
            color=span["mid_color"], alpha=0.95,
        )


def _version_legend(fig: plt.Figure) -> None:
    """
    Add a compact figure-level legend explaining the light→dark shade convention
    (v1 lightest = oldest prompt version, v4 darkest = newest / multi-step).
    """
    slot_colors = ["#BDBDBD", "#757575", "#424242", "#111111"]
    slot_labels = [
        "v1 — baseline prompt",
        "v2 — prompt v2",
        "v3 — prompt v3 (thinking)",
        "v4 — multi-step pipeline (new)",
    ]
    patches = [mpatches.Patch(facecolor=c, label=l, edgecolor="white")
               for c, l in zip(slot_colors, slot_labels)]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=4,
        fontsize=10,
        framealpha=0.75,
        bbox_to_anchor=(0.5, 0.0),
        title="Shade = prompt version (lighter → older, darker → newer)."
              "  Color family = model.",
        title_fontsize=9,
    )


def _figwidth(layout: dict, min_w: float = 18.0, scale: float = 1.1) -> float:
    """Derive a sensible figure width from the number of x positions."""
    n = len(layout["positions"])
    g = len(layout["group_sep_x"]) * 1.8  # extra width for inter-group gaps
    return max(min_w, n * scale + g)


BAR_W = 0.68   # bar width used consistently across all plots


def plot_classification_metrics(results: dict, output_dir: Path) -> None:
    layout  = _build_grouped_layout(results)
    metrics = ["accuracy", "precision", "recall", "f1"]
    fw      = _figwidth(layout)

    fig, axes = plt.subplots(2, 2, figsize=(fw, fw * 0.85))
    fig.suptitle("Classification Metrics — discard_image vs Human Label",
                 fontsize=16, fontweight="bold", y=1.01)

    for ax, metric in zip(axes.flatten(), metrics):
        vals = _vals_for_layout(results, layout, key=None, clf_key=metric)
        bars = ax.bar(layout["positions"], vals,
                      width=BAR_W, color=layout["colors"],
                      edgecolor="white", linewidth=0.6)
        ax.set_ylim(0, 1.14)
        ax.set_title(metric.capitalize(), fontsize=13)
        ax.set_ylabel("Score", fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=10)
        _setup_grouped_ax(ax, layout, bar_width=BAR_W)

    _version_legend(fig)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    path = output_dir / "classification_metrics.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {path}")


def plot_token_usage(results: dict, output_dir: Path) -> None:
    layout = _build_grouped_layout(results)
    fw     = _figwidth(layout, min_w=16)

    avg_pt  = _vals_for_layout(results, layout, "avg_prompt_tokens")
    avg_ct  = _vals_for_layout(results, layout, "avg_completion_tokens")
    avg_rkt = _vals_for_layout(results, layout, "avg_reasoning_tokens")
    avg_tt  = _vals_for_layout(results, layout, "avg_total_tokens")

    # 4 sub-bars per run position; shrink individual sub-bar width
    sub_w  = BAR_W / 4.5
    pos    = np.array(layout["positions"], dtype=float)
    bcolors = layout["colors"]

    fig, ax = plt.subplots(figsize=(fw, 11))

    off = sub_w * 1.1
    b1 = ax.bar(pos - 1.5 * off, avg_pt,  sub_w * 1.0,
                color=bcolors, alpha=1.0, edgecolor="white", linewidth=0.4, label="Prompt")
    b2 = ax.bar(pos - 0.5 * off, avg_ct,  sub_w * 1.0,
                color=bcolors, alpha=0.65, edgecolor="white", linewidth=0.4, label="Completion", hatch="///")
    b3 = ax.bar(pos + 0.5 * off, avg_rkt, sub_w * 1.0,
                color=bcolors, alpha=0.40, edgecolor="white", linewidth=0.4, label="Reasoning", hatch="xxx")
    b4 = ax.bar(pos + 1.5 * off, avg_tt,  sub_w * 1.0,
                color=bcolors, alpha=0.20, edgecolor="grey",  linewidth=0.4, label="Total",  hatch="...")

    def _lbl(bars, data):
        for bar, v in zip(bars, data):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=9, rotation=90)

    _lbl(b1, avg_pt); _lbl(b2, avg_ct); _lbl(b3, avg_rkt); _lbl(b4, avg_tt)

    ax.set_title("Average Token Usage per Run", fontsize=16, fontweight="bold")
    ax.set_ylabel("Tokens", fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", alpha=0.3)
    # Token-type legend (hatch patterns)
    token_patches = [
        mpatches.Patch(facecolor="#888", alpha=1.0, label="Avg Prompt Tokens"),
        mpatches.Patch(facecolor="#888", alpha=0.65, hatch="///", label="Avg Completion Tokens"),
        mpatches.Patch(facecolor="#888", alpha=0.40, hatch="xxx", label="Avg Reasoning Tokens"),
        mpatches.Patch(facecolor="#888", alpha=0.20, hatch="...", label="Avg Total Tokens"),
    ]
    ax.legend(handles=token_patches, fontsize=11, loc="upper right")
    _setup_grouped_ax(ax, layout, bar_width=BAR_W)
    _version_legend(fig)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    path = output_dir / "token_usage.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {path}")


def plot_response_time(results: dict, output_dir: Path) -> None:
    layout = _build_grouped_layout(results)
    fw     = _figwidth(layout, min_w=12)
    vals   = _vals_for_layout(results, layout, "avg_response_time")

    fig, ax = plt.subplots(figsize=(fw, 9))
    bars = ax.bar(layout["positions"], vals,
                  width=BAR_W, color=layout["colors"],
                  edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{val:.1f}s", ha="center", va="bottom", fontsize=11)
    ax.set_title("Average Response Time per Run", fontsize=16, fontweight="bold")
    ax.set_ylabel("Seconds", fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", alpha=0.3)
    _setup_grouped_ax(ax, layout, bar_width=BAR_W)
    _version_legend(fig)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    path = output_dir / "response_time.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {path}")


def plot_confusion_matrices(results: dict, output_dir: Path) -> None:
    # Reorder by MODEL_GROUPS so related models are adjacent in the grid
    layout   = _build_grouped_layout(results)
    ordered  = layout["runs_ordered"]
    n_runs   = len(ordered)
    ncols    = 4
    nrows    = (n_runs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.8, nrows * 4.8))
    fig.suptitle("Confusion Matrices — discard_image (Positive = True)",
                 fontsize=16, fontweight="bold")
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, run in enumerate(ordered):
        ax  = axes_flat[idx]
        d   = results[run]
        cm  = d["clf"]
        bar_color = layout["colors"][idx]
        matrix = np.array([[cm["tn"] or 0, cm["fp"] or 0],
                            [cm["fn"] or 0, cm["tp"] or 0]])
        # Use the run's group color as the colormap tint
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "grp", ["#FFFFFF", bar_color], N=256)
        ax.imshow(matrix, interpolation="nearest", cmap=cmap)
        ax.set_title(RUN_LABELS.get(run, run), fontsize=11, fontweight="bold",
                     color=bar_color)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred F", "Pred T"], fontsize=10)
        ax.set_yticklabels(["Act. F", "Act. T"], fontsize=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual",    fontsize=10)
        for i in range(2):
            for j in range(2):
                val   = int(matrix[i, j])
                color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=16, color=color, fontweight="bold")

    for idx in range(n_runs, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    path = output_dir / "confusion_matrices.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {path}")


def plot_failures(results: dict, output_dir: Path) -> None:
    layout     = _build_grouped_layout(results)
    fw         = _figwidth(layout, min_w=13)
    n_failed   = _vals_for_layout(results, layout, "n_failed")
    n_no_meta  = _vals_for_layout(results, layout, "n_no_metadata")
    n_missing  = _vals_for_layout(results, layout, "n_missing_pred")

    sub_w = BAR_W / 3.3
    pos   = np.array(layout["positions"], dtype=float)

    fig, ax = plt.subplots(figsize=(fw, 9))
    ax.bar(pos - sub_w, n_failed,  sub_w, label="Failed (status=failed)",
           color="#F44336", edgecolor="white")
    ax.bar(pos,          n_no_meta, sub_w, label="No metadata key",
           color="#FF9800", edgecolor="white")
    ax.bar(pos + sub_w, n_missing, sub_w, label="Missing prediction key",
           color="#9E9E9E", edgecolor="white")
    ax.set_title("Failures / Missing Data per Run", fontsize=16, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    _setup_grouped_ax(ax, layout, bar_width=BAR_W)
    _version_legend(fig)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    path = output_dir / "failures.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DCA LLM runs against human labels")
    parser.add_argument("--json-dir",   type=Path, default=DEFAULT_JSON_DIR,   help="Directory with pair_*.json files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for plots and CSV")
    args = parser.parse_args()

    records = load_jsons(args.json_dir)

    # -----------------------------------------------------------------------
    # Extract data for each run
    # -----------------------------------------------------------------------
    results: dict[str, dict] = {}
    for run in RUNS:
        raw = extract_run_data(records, run)
        cm  = classification_metrics(raw["y_true"], raw["y_pred"])
        results[run] = {
            "clf":                   cm,
            "avg_prompt_tokens":     safe_mean(raw["prompt_tokens"]),
            "avg_completion_tokens": safe_mean(raw["completion_tokens"]),
            "avg_reasoning_tokens":  safe_mean(raw["reasoning_tokens"]),
            "avg_total_tokens":      safe_mean(raw["total_tokens"]),
            "avg_response_time":     safe_mean(raw["response_times"]),
            "n_total":               raw["n_total"],
            "n_failed":              raw["n_failed"],
            "n_no_metadata":         raw["n_no_metadata"],
            "n_missing_pred":        raw["n_missing_pred"],
            "pairs_with_metadata":   raw["pairs_with_metadata"],
            # Keep raw lists for per-run detail
            "y_true": raw["y_true"],
            "y_pred": raw["y_pred"],
        }

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print_summary_table(results)

    # -----------------------------------------------------------------------
    # Per-class breakdown (macro / weighted avg)
    # -----------------------------------------------------------------------
    print_classification_reports(results)

    # -----------------------------------------------------------------------
    # Additional per-run detail
    # -----------------------------------------------------------------------
    print("Confusion matrix details:")
    for run, d in results.items():
        cm = d["clf"]
        n = cm["n_evaluated"]
        if n:
            print(f"  {RUN_LABELS.get(run, run):<12} → TP={cm['tp']} TN={cm['tn']} FP={cm['fp']} FN={cm['fn']}"
                  f"  |  +support={cm['support_pos']}  -support={cm['support_neg']}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_classification_metrics(results, args.output_dir)
    plot_token_usage(results, args.output_dir)
    plot_response_time(results, args.output_dir)
    plot_confusion_matrices(results, args.output_dir)
    plot_failures(results, args.output_dir)

    # -----------------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------------
    save_csv(results, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
