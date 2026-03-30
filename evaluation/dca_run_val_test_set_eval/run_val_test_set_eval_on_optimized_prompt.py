#!/usr/bin/env python3
"""Evaluate the saved MIPROv2-optimized DCA model on val/test splits.

Runs real inference through the optimized DSPy model and evaluates with
the same classification + LLM-judge change metric as run_val_test_set_eval_on_human_runs.py.

Captures per-pair:
  - classification_score     (1.0 / 0.0)
  - change_score             (rule-based or LLM judge, 0.0-1.0)
  - total_score              (0.6 x cls + 0.4 x change)
  - response_time_seconds    (inference only, not including judge calls)
  - total_tokens             (from DSPy LM history for the inference call)

Output JSON is compatible with eval_results_full_<set>.json from
run_val_test_set_eval_on_human_runs.py so results can be compared in the same table.

Usage
-----
    # Evaluate on val set (default):
    python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py

    # Evaluate on both val and test:
    python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py --which-set both

    # Use a specific saved model:
    python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py \\
        --model-path .cache/gpt-4.1-mini_MIPROv2_DCA_Light_dca_optimized_2026032517.json

    # Skip LLM judge (rule-based change score only):
    python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py --skip-llm

    # Overwrite existing outputs:
    python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Path Setup — ensure repo root is on sys.path
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
# layout: evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py
#   parents[0] = evaluation/
#   parents[1] = repo root
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Imports (after sys.path is set)
# ---------------------------------------------------------------------------
import dspy
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import create_lm, DEFAULT_MODEL_NAME, AZURE_API_KEY
from signatures.dca import DCAImageComparison
from evaluation.dca_metric import (
    FIELD_WEIGHT_CLS,
    FIELD_WEIGHT_CHANGE,
    _try_rule_based_match,
    judge_summary_metric,
    compute_classification_score,
    compute_total_score,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = REPO_ROOT / ".cache" / "gpt-4.1_MIPROv2_DCA_Light_dca_optimized_2026032716.json"
DEFAULT_SPLIT_FILE = REPO_ROOT / ".cache" / "dca_split_seed42.json"
DEFAULT_IMAGE_DIR  = REPO_ROOT / "data" / "dca_data" / "image_pairs"
DEFAULT_GT_DIR     = REPO_ROOT / "data" / "dca_data" / "image_pairs_llm_output_exp1"
OUTPUT_DIR         = SCRIPT_DIR / "eval_results"

RUN_KEY   = "miprov2_light_gpt41_2026032716_run_using_gpt41"
RUN_LABEL = "MIPROv2-Light/GPT-4.1_2026032716_run_using_gpt41"


# ---------------------------------------------------------------------------
# Infer the inference model name from the optimized-model filename.
# Convention: <model>_<Strategy>_dca_optimized_<timestamp>.json
# e.g. gpt-4.1-mini_MIPROv2_DCA_Light_dca_optimized_2026032517.json → azure/gpt-4.1-mini
# ---------------------------------------------------------------------------

def _infer_model_name_from_path(path: Path) -> Optional[str]:
    """Return 'azure/<model>' extracted from the filename, or None if unparseable."""
    stem = path.stem  # strip .json extension
    for marker in ["_MIPROv2_", "_GEPA_", "_SimBA_", "_Bootstrap_"]:
        idx = stem.find(marker)
        if idx > 0:
            return f"azure/{stem[:idx]}"
    return None


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


def _get_tokens_from_lm_history(lm: dspy.LM, history_len_before: int) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Return (total, input, output, reasoning) summed from new LM history entries.

    Uses plain getattr/dict access — no litellm-specific methods.
    reasoning_tokens comes from completion_tokens_details.reasoning_tokens (o-series models only).
    """
    def _get(obj, key):
        """Read key from dict or object attribute."""
        if obj is None:
            return None
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    try:
        new_entries = lm.history[history_len_before:]
        if not new_entries:
            return None, None, None, None
        total = input_t = output_t = reasoning_t = 0
        for entry in new_entries:
            # DSPy stores usage either at top-level or inside response object
            usage = entry.get("usage") or _get(entry.get("response"), "usage")
            pt  = int(_get(usage, "prompt_tokens")     or _get(usage, "input_tokens")  or 0)
            ct  = int(_get(usage, "completion_tokens") or _get(usage, "output_tokens") or 0)
            tt  = _get(usage, "total_tokens")
            details = _get(usage, "completion_tokens_details")
            rt  = int(_get(details, "reasoning_tokens") or 0)
            input_t     += pt
            output_t    += ct
            reasoning_t += rt
            total += int(tt) if tt is not None else (pt + ct)
        return (
            total       if total       > 0 else None,
            input_t     if input_t     > 0 else None,
            output_t    if output_t    > 0 else None,
            reasoning_t if reasoning_t > 0 else None,
        )
    except Exception:
        return None, None, None, None


def compute_change_score(
    human: str,
    model: str,
    use_llm: bool,
) -> tuple[Optional[float], str]:
    """Return (score, source) where source is 'rule', 'llm', or 'skipped'."""
    score, _reasoning, matched = _try_rule_based_match(human, model)
    if matched:
        return score, "rule"

    if not use_llm:
        return None, "skipped"

    try:
        score = judge_summary_metric(human, model)
    except Exception:
        score = None

    return score, "llm"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    comparator: dspy.ChainOfThought,
    pair_names: list[str],
    image_dir: Path,
) -> list[dict]:
    """Run the optimized model on all pairs. Returns raw predictions with timing."""
    lm = dspy.settings.lm
    per_pair: list[dict] = []

    for i, pair_name in enumerate(pair_names, 1):
        pair_dir   = image_dir / pair_name
        orig_path  = pair_dir / "original.png"
        reim_path  = pair_dir / "reimage.png"

        # Load images (None if absent — same behaviour as load_dca_dataset)
        try:
            image_original = dspy.Image.from_file(str(orig_path)) if orig_path.exists() else None
            image_reimage  = dspy.Image.from_file(str(reim_path)) if reim_path.exists() else None
        except Exception as exc:
            print(f"  [{i:03d}/{len(pair_names)}] {pair_name}  [WARN] image load failed: {exc}")
            per_pair.append({"pair_name": pair_name, "missing": True, "error": str(exc)})
            continue

        # Inference — timed and token-tracked
        history_len = len(lm.history) if hasattr(lm, "history") else 0
        t0 = time.time()
        try:
            pred = comparator(image_original=image_original, image_reimage=image_reimage)
        except Exception as exc:
            print(f"  [{i:03d}/{len(pair_names)}] {pair_name}  [WARN] inference error: {exc}")
            per_pair.append({"pair_name": pair_name, "missing": True, "error": str(exc)})
            continue
        response_time = time.time() - t0
        total_tokens, input_tokens, output_tokens, reasoning_tokens = _get_tokens_from_lm_history(lm, history_len)

        pred_discard = str(getattr(pred, "discard_image", "") or "").strip()
        pred_changes = str(getattr(pred, "changes", "")        or "").strip()

        tok_str = f"  tok={total_tokens}" if total_tokens else ""
        print(f"  [{i:03d}/{len(pair_names)}] {pair_name}  "
              f"discard={pred_discard}  t={response_time:.1f}s{tok_str}")

        per_pair.append({
            "pair_name":        pair_name,
            "missing":          False,
            "pred_discard":     pred_discard,
            "pred_changes":     pred_changes,
            "response_time":    response_time,
            "total_tokens":     total_tokens,
            "input_tokens":     input_tokens,
            "output_tokens":    output_tokens,
            "reasoning_tokens": reasoning_tokens,
        })

    return per_pair


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    per_pair_inf: list[dict],
    gt_records: dict[str, dict],
    use_llm: bool,
) -> list[dict]:
    """Match predictions with ground truth and compute cls + change scores."""
    per_pair_eval: list[dict] = []

    n = len(per_pair_inf)
    for i, rec in enumerate(per_pair_inf, 1):
        pair_name = rec["pair_name"]

        if rec.get("missing"):
            per_pair_eval.append({"pair_name": pair_name, "missing": True})
            continue

        gt = gt_records.get(pair_name, {})
        human_discard = normalize_bool(gt.get("human_label_discard_image"))
        human_changes = str(gt.get("human_label_changes") or "").strip()
        pred_discard  = normalize_bool(rec["pred_discard"])

        if human_discard is None or pred_discard is None:
            per_pair_eval.append({
                "pair_name": pair_name,
                "missing":   True,
                "note":      f"unparseable discard: human={gt.get('human_label_discard_image')!r}  pred={rec['pred_discard']!r}",
            })
            continue

        cls_score = compute_classification_score(human_discard, pred_discard)

        print(f"  [{i:03d}/{n}] {pair_name}  cls={cls_score:.0f}", end="  ", flush=True)
        change_score, change_source = compute_change_score(human_changes, rec["pred_changes"], use_llm)
        chg_str = f"{change_score:.3f}" if change_score is not None else "N/A"
        print(f"chg={chg_str}  src={change_source}")

        if change_score is not None:
            total_score = compute_total_score(cls_score, change_score)
        elif change_source == "llm":
            # judge was called but returned None (all-zero counts edge case) → 0.0 fallback
            total_score = compute_total_score(cls_score, 0.0)
        else:
            total_score = None  # --skip-llm: change score unavailable

        per_pair_eval.append({
            "pair_name":        pair_name,
            "missing":          False,
            "cls_score":        cls_score,
            "change_score":     change_score,
            "total_score":      total_score,
            "change_source":    change_source,
            "total_tokens":     rec.get("total_tokens"),
            "input_tokens":     rec.get("input_tokens"),
            "output_tokens":    rec.get("output_tokens"),
            "reasoning_tokens": rec.get("reasoning_tokens"),
            "response_time":    rec.get("response_time"),
        })

    return per_pair_eval


def aggregate_results(
    per_pair: list[dict],
    run_key: str = RUN_KEY,
    run_label: str = RUN_LABEL,
) -> dict:
    """Aggregate per-pair scores into a run-level result dict (same format as evaluate_run)."""
    valid = [p for p in per_pair if not p.get("missing")]

    cls_scores    = [p["cls_score"]    for p in valid]
    change_scores = [p["change_score"] for p in valid if p["change_score"] is not None]
    total_scores  = [p["total_score"]  for p in valid if p["total_score"]  is not None]
    tokens           = [p["total_tokens"]     for p in valid if p.get("total_tokens")     is not None]
    input_tok_list   = [p["input_tokens"]     for p in valid if p.get("input_tokens")     is not None]
    output_tok_list  = [p["output_tokens"]    for p in valid if p.get("output_tokens")    is not None]
    reason_tok_list  = [p["reasoning_tokens"] for p in valid if p.get("reasoning_tokens") is not None]
    times            = [p["response_time"]    for p in valid if p.get("response_time")    is not None]

    source_counts = {"rule": 0, "llm": 0, "skipped": 0}
    for p in valid:
        src = p.get("change_source", "skipped")
        source_counts[src] = source_counts.get(src, 0) + 1

    return {
        "run":                  run_key,
        "label":                run_label,
        "n_test_pairs":         len(per_pair),
        "n_evaluated":          len(valid),
        "n_missing":            len(per_pair) - len(valid),
        "avg_cls_score":        float(np.mean(cls_scores))       if cls_scores       else None,
        "avg_change_score":     float(np.mean(change_scores))    if change_scores    else None,
        "avg_total_score":      float(np.mean(total_scores))     if total_scores     else None,
        "avg_total_tokens":     float(np.mean(tokens))           if tokens           else None,
        "avg_input_tokens":     float(np.mean(input_tok_list))   if input_tok_list   else None,
        "avg_output_tokens":    float(np.mean(output_tok_list))  if output_tok_list  else None,
        "avg_reasoning_tokens": float(np.mean(reason_tok_list))  if reason_tok_list  else None,
        "avg_response_time":    float(np.mean(times))            if times            else None,
        "n_cls":                len(cls_scores),
        "n_change":             len(change_scores),
        "n_total_score":        len(total_scores),
        "change_source_counts": source_counts,
        "per_pair":             per_pair,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all(results: list[dict], output_dir: Path, n_pairs: int, set_name: str = "val") -> None:
    """Generate and save score, efficiency, and heatmap plots for optimized-prompt runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    set_label = set_name.capitalize()
    run_labels = [r.get("label", r["run"]) for r in results]
    x = np.arange(len(results))
    bar_w = 0.5

    # ── Figure 1: Score metrics (3 subplots) ─────────────────────────────────
    fig1, axes1 = plt.subplots(3, 1, figsize=(max(8, len(results) * 2.5), 12))
    fig1.suptitle(
        f"DCA Optimized Prompt Evaluation — {set_label} Set (n={n_pairs})",
        fontsize=11, fontweight="bold",
    )
    for ax, key, title, ylabel in [
        (axes1[0], "avg_total_score",  "Total Score  (0.6 × Classification + 0.4 × Change)", "Avg Total Score"),
        (axes1[1], "avg_cls_score",    "Classification Score  (discard_image correct?)",      "Avg Cls Score"),
        (axes1[2], "avg_change_score", "Change Score  (judge F-beta, precision-60/recall-40)", "Avg Change Score"),
    ]:
        values = [(r.get(key) or 0.0) for r in results]
        ax.bar(x, values, width=bar_w, color="#4CAF50", edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.10)
        ax.yaxis.grid(True, alpha=0.4, linestyle="--")
        ax.set_axisbelow(True)
        for xi, v in zip(x, values):
            if v > 0:
                ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig1.tight_layout()
    p = output_dir / "scores.png"
    fig1.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {p}")

    # ── Figure 2: Efficiency metrics ─────────────────────────────────────────
    fig2, axes2 = plt.subplots(2, 1, figsize=(max(8, len(results) * 2.5), 8))
    fig2.suptitle(
        f"DCA Optimized — Response Time & Token Usage — {set_label} Set (n={n_pairs})",
        fontsize=11, fontweight="bold",
    )
    for ax, key, title, ylabel in [
        (axes2[0], "avg_response_time", "Avg Response Time per Pair  (seconds)", "Seconds"),
        (axes2[1], "avg_total_tokens",  "Avg Total Tokens per Pair  (prompt + completion)", "Tokens"),
    ]:
        values = [(r.get(key) or 0.0) for r in results]
        ax.bar(x, values, width=bar_w, color="#42A5F5", edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, alpha=0.4, linestyle="--")
        ax.set_axisbelow(True)
        for xi, v in zip(x, values):
            if v > 0:
                ax.text(xi, v + ax.get_ylim()[1] * 0.01, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig2.tight_layout()
    p = output_dir / "efficiency_metrics.png"
    fig2.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {p}")

    # ── Figure 3: Summary heatmap ─────────────────────────────────────────────
    metric_names = ["Total Score", "Cls Score", "Change Score", "Avg Time (s)", "Avg Tokens (k)"]
    data_matrix = []
    for r in results:
        data_matrix.append([
            r.get("avg_total_score")   or 0.0,
            r.get("avg_cls_score")     or 0.0,
            r.get("avg_change_score")  or 0.0,
            r.get("avg_response_time") or 0.0,
            (r.get("avg_total_tokens") or 0.0) / 1000.0,
        ])
    data_arr = np.array(data_matrix, dtype=float)

    normed = np.zeros_like(data_arr)
    for c in range(data_arr.shape[1]):
        col = data_arr[:, c]
        span = col.max() - col.min()
        normed[:, c] = (col - col.min()) / span if span > 0 else np.zeros_like(col)

    fig3, ax3 = plt.subplots(figsize=(10, max(4, len(results) * 0.6)))
    fig3.suptitle("DCA Optimized — Summary Heatmap (All Runs × Metrics)",
                  fontsize=11, fontweight="bold")
    ax3.imshow(normed, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax3.set_xticks(range(len(metric_names)))
    ax3.set_xticklabels(metric_names, fontsize=9)
    ax3.set_yticks(range(len(results)))
    ax3.set_yticklabels(run_labels, fontsize=8)
    for ri, row in enumerate(data_matrix):
        for ci, v in enumerate(row):
            txt = f"{v:.3f}" if ci < 3 else (f"{v:.1f}" if ci == 3 else f"{v:.2f}k")
            ax3.text(ci, ri, txt, ha="center", va="center", fontsize=7, color="black")
    fig3.tight_layout()
    p = output_dir / "summary_heatmap.png"
    fig3.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {p}")


# ---------------------------------------------------------------------------
# Azure pre-flight check
# ---------------------------------------------------------------------------

def _preflight_azure_check(n_pairs: int, avg_seconds_per_pair: float = 20.0) -> None:
    """Validate Azure CLI credentials before a long inference run.

    - Exits immediately with a clear message if credentials are already expired.
    - Warns if the token will likely expire before all pairs are processed.
    Skipped when an API key is configured (no CLI auth needed).
    """
    if AZURE_API_KEY:
        return  # Using API key auth — CLI token not needed

    try:
        import time as _time
        from azure.identity import AzureCliCredential
        token = AzureCliCredential().get_token("https://cognitiveservices.azure.com/.default")
        expires_in_s = token.expires_on - _time.time()
        estimated_s  = n_pairs * avg_seconds_per_pair
        if expires_in_s < estimated_s + 300:  # 5-min safety buffer
            print(
                f"  [WARN] Azure CLI token expires in {expires_in_s/60:.0f} min "
                f"but run will take ~{estimated_s/60:.0f} min ({n_pairs} pairs). "
                f"Run 'az login' now to refresh before proceeding."
            )
        else:
            print(
                f"  Azure CLI token: valid for {expires_in_s/60:.0f} min "
                f"(est. run ~{estimated_s/60:.0f} min) — OK"
            )
    except Exception as exc:
        print(f"\n[ERROR] Azure CLI credentials are not valid: {exc}")
        print("  Please run 'az login' and retry.\n")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MIPROv2-optimized DCA model on val/test splits."
    )
    parser.add_argument("--model-path",  type=Path, default=DEFAULT_MODEL_PATH,
                        help="Path to saved DSPy optimized model JSON")
    parser.add_argument("--model-name",  type=str, default=None,
                        help="Override LM name for inference (default: MODEL_NAME from .env)")
    parser.add_argument("--split-file",  type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--image-dir",   type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--gt-dir",      type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--output-dir",  type=Path, default=OUTPUT_DIR)
    parser.add_argument("--which-set",   type=str, choices=["val", "test", "both"], default="val",
                        help="Which split to evaluate (default: val)")
    parser.add_argument("--skip-llm",    action="store_true",
                        help="Skip LLM judge; breakdown cases get change_score=None")
    parser.add_argument("--overwrite",   action="store_true",
                        help="Overwrite the current run's entry if it already exists in the results file")
    parser.add_argument("--replot",      action="store_true",
                        help="Load existing results JSON and re-generate plots only (no inference)")
    parser.add_argument("--results-json", type=Path, default=None,
                        help="Path to existing results JSON for --replot")
    parser.add_argument("--no-cot",      action="store_true",
                        help="Use dspy.Predict instead of dspy.ChainOfThought "
                             "(keeps few-shot demos + optimized instruction, removes reasoning step)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_llm = not args.skip_llm
    no_cot  = args.no_cot

    if no_cot:
        active_run_key   = RUN_KEY + "_nocot"
        active_run_label = RUN_LABEL + " (no CoT)"
        file_prefix      = "eval_results" + "_no_CoT_" + RUN_KEY
    else:
        active_run_key   = RUN_KEY
        active_run_label = RUN_LABEL
        file_prefix      = "eval_results" + RUN_KEY

    sets_to_run = ["val", "test"] if args.which_set == "both" else [args.which_set]

    # ── Replot mode ───────────────────────────────────────────────────────────
    if args.replot:
        for set_name in sets_to_run:
            results_file = args.results_json or (args.output_dir / f"{file_prefix}_{set_name}.json")
            if not results_file.exists():
                print(f"[ERROR] --replot: results file not found: {results_file}")
                sys.exit(1)
            data = json.loads(results_file.read_text(encoding="utf-8"))
            n_pairs  = data["meta"].get("n_pairs", 0)
            set_lbl  = data["meta"].get("which_set", set_name)
            results  = data["results"]
            print(f"Replotting {len(results)} run(s) from {results_file.name} …")
            plot_all(results, args.output_dir / f"opt_{set_lbl}", n_pairs, set_name=set_lbl)
        return

    # ── Guard: abort if the current run_key already exists and --overwrite not set ──
    if not args.overwrite:
        for set_name in sets_to_run:
            out = args.output_dir / f"{file_prefix}_{set_name}.json"
            if out.exists():
                _old = json.loads(out.read_text(encoding="utf-8"))
                _existing = {r["run"] for r in _old.get("results", [])}
                if active_run_key in _existing:
                    print(f"[ERROR] Run '{active_run_key}' already exists in {out.name}")
                    print("  Use --overwrite to re-evaluate and replace it.")
                    sys.exit(1)

    # ── Load optimized model ──────────────────────────────────────────────────
    if not args.model_path.exists():
        print(f"[ERROR] Model file not found: {args.model_path}")
        sys.exit(1)

    # Determine which model to use for inference.
    # Priority: 1) --model-name arg  2) model name inferred from filename  3) MODEL_NAME from .env
    if args.model_name:
        model_name = args.model_name
        model_source = "--model-name arg"
    else:
        inferred = _infer_model_name_from_path(args.model_path)
        if inferred:
            model_name = inferred
            model_source = f"inferred from filename ({args.model_path.name})"
        else:
            model_name = DEFAULT_MODEL_NAME
            model_source = f"MODEL_NAME from .env  [WARNING: could not infer from filename — verify this matches the model used during optimization]"

    print(f"Inference LM : {model_name}  ({model_source})")
    print(f"              cache=False, temperature=0.1")
    lm = create_lm(model_name, cache=False, temperature=0.1)
    dspy.configure(lm=lm)

    mode_label = "Predict (no CoT — few-shots only)" if no_cot else "ChainOfThought (few-shots + reasoning)"
    print(f"Mode         : {mode_label}")
    print(f"Loading optimized program: {args.model_path.name}")
    comparator = dspy.Predict(DCAImageComparison) if no_cot else dspy.ChainOfThought(DCAImageComparison)
    comparator.load(str(args.model_path))
    print("Model ready.\n")

    # ── Load split ────────────────────────────────────────────────────────────
    if not args.split_file.exists():
        print(f"[ERROR] Split file not found: {args.split_file}")
        sys.exit(1)
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    seed  = split.get("seed", "?")

    def _get_pair_names(key: str) -> list[str]:
        names = split.get(key, {}).get("pair_names") or split.get(f"{key}_pair_numbers") or []
        if names and isinstance(names[0], int):
            names = [f"pair_{n:03d}" for n in names]
        return names

    # ── Validate paths ────────────────────────────────────────────────────────
    for d, label in [(args.image_dir, "image-dir"), (args.gt_dir, "gt-dir")]:
        if not d.exists():
            print(f"[ERROR] {label} not found: {d}")
            sys.exit(1)

    # ── Pre-flight: validate Azure CLI credentials before the long inference run ──
    total_pairs = sum(len(_get_pair_names(s)) for s in sets_to_run)
    _preflight_azure_check(total_pairs)

    # ── Per-set evaluation ────────────────────────────────────────────────────
    def _run_set(set_name: str) -> None:
        pair_names = _get_pair_names(set_name)
        print(f"\n{'='*70}")
        print(f"Set: {set_name.upper()}  ({len(pair_names)} pairs, seed={seed})")
        print(f"LLM judge: {'ENABLED' if use_llm else 'DISABLED (--skip-llm)'}")
        print(f"{'='*70}\n")

        # Load ground truth
        gt_records: dict[str, dict] = {}
        for pname in pair_names:
            f = args.gt_dir / f"{pname}.json"
            if f.exists():
                gt_records[pname] = json.loads(f.read_text(encoding="utf-8"))

        # ── Step 1: Inference ────────────────────────────────────────────────
        print("Step 1/2 — Running inference …\n")
        per_pair_inf = run_inference(comparator, pair_names, args.image_dir)

        # ── Step 2: Evaluate (change score + cls) ────────────────────────────
        print(f"\nStep 2/2 — Evaluating scores (LLM judge: {use_llm}) …\n")
        per_pair_eval = evaluate_predictions(per_pair_inf, gt_records, use_llm)

        # Aggregate
        result = aggregate_results(per_pair_eval, run_key=active_run_key, run_label=active_run_label)

        # Attach per-run model provenance directly to the result entry
        result["model_path"] = str(args.model_path)
        result["model_name"] = model_name
        result["no_cot"]     = no_cot

        # ── Load existing JSONs and merge by run key ──────────────────────────
        summary_path = args.output_dir / f"{file_prefix}_{set_name}.json"
        full_path    = args.output_dir / f"{file_prefix}_full_{set_name}.json"

        old_run_order:        list[str]       = []
        existing_summary_map: dict[str, dict] = {}
        existing_full_map:    dict[str, dict] = {}

        if summary_path.exists():
            _old = json.loads(summary_path.read_text(encoding="utf-8"))
            for r in _old.get("results", []):
                old_run_order.append(r["run"])
                existing_summary_map[r["run"]] = r

        if full_path.exists():
            _old_full = json.loads(full_path.read_text(encoding="utf-8"))
            for r in _old_full.get("results", []):
                existing_full_map[r["run"]] = r

        # New result overrides matching old entry; all others are preserved
        r_clean = {k: v for k, v in result.items() if k != "per_pair"}
        existing_summary_map[active_run_key] = r_clean
        existing_full_map[active_run_key]    = result

        # Preserve original order, append truly new run at the end
        merged_run_names: list[str] = list(old_run_order)
        if active_run_key not in merged_run_names:
            merged_run_names.append(active_run_key)

        merged_summary = [existing_summary_map[r] for r in merged_run_names if r in existing_summary_map]
        merged_full    = [existing_full_map[r]    for r in merged_run_names if r in existing_full_map]

        # ── File-level meta (not run-specific) ───────────────────────────────
        saved_meta = {
            "split_file":          str(args.split_file),
            "which_set":           set_name,
            "n_pairs":             len(pair_names),
            "skip_llm":            not use_llm,
            "field_weight_cls":    FIELD_WEIGHT_CLS,
            "field_weight_change": FIELD_WEIGHT_CHANGE,
            "runs":                merged_run_names,
        }

        summary_path.write_text(
            json.dumps({"meta": saved_meta, "results": merged_summary}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nResults JSON  -> {summary_path}  ({len(merged_summary)} run(s) total)")

        full_path.write_text(
            json.dumps({"meta": saved_meta, "results": merged_full}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Full results  -> {full_path}  ({len(merged_full)} run(s) total)")

        # ── Summary table (all merged runs) ──────────────────────────────────
        print(f"\n{'='*100}")
        print(f"{'Run':<42} {'Cls':>6} {'Chg':>6} {'Tot':>6} {'Time(s)':>8} {'Tokens':>8} {'n_eval':>7}")
        print(f"{'-'*100}")
        for r in merged_summary:
            c  = f"{r['avg_cls_score']:.3f}"    if r.get('avg_cls_score')    is not None else "N/A"
            ch = f"{r['avg_change_score']:.3f}" if r.get('avg_change_score') is not None else "N/A"
            t  = f"{r['avg_total_score']:.3f}"  if r.get('avg_total_score')  is not None else "N/A"
            ti = f"{r['avg_response_time']:.1f}" if r.get('avg_response_time') is not None else "N/A"
            tk = f"{r['avg_total_tokens']:.0f}"  if r.get('avg_total_tokens')  is not None else "N/A"
            src = r.get("change_source_counts", {})
            print(f"  {r['label']:<40} {c:>6} {ch:>6} {t:>6} {ti:>8} {tk:>8} {r['n_evaluated']:>7}  src={src}")
            inp = f"{r['avg_input_tokens']:.0f}"     if r.get('avg_input_tokens')     is not None else "N/A"
            out = f"{r['avg_output_tokens']:.0f}"    if r.get('avg_output_tokens')    is not None else "N/A"
            rsn = f"{r['avg_reasoning_tokens']:.0f}" if r.get('avg_reasoning_tokens') is not None else "N/A"
            print(f"    tok breakdown: input={inp}  output={out}  reasoning={rsn}")
        print(f"{'='*100}")

        # ── Plots from all merged runs ────────────────────────────────────────
        out_plot_dir = args.output_dir / f"opt_{set_name}"
        print("\nGenerating plots …")
        plot_all(merged_summary, out_plot_dir, len(pair_names), set_name=set_name)

    for set_name in sets_to_run:
        _run_set(set_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
