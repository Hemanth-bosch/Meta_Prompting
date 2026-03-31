#!/usr/bin/env python3
"""Evaluate all human-prompt DCA runs on val/test splits.

Computes for each run × pair:
  - classification_score  (1.0 / 0.0)  — discard_image correct?
  - change_score          (0.0–1.0)    — change-summary quality via LLM judge
  - total_score           (0.6 × cls + 0.4 × change)
  - response_time_seconds (from stored metadata)
  - total_tokens          (from stored metadata)

Output layout (under --output-dir = eval_results/human_runs/):
  eval_results_{val|test}.json
    ├── meta      — split info, weights, run list
    ├── summary   — one entry per run, aggregated metrics only  (easy to scan)
    └── details   — one entry per run, aggregated + per_pair    (full data)

Usage
-----
    # Evaluate test set (default):
    python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_human_runs.py

    # Evaluate val set:
    python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_human_runs.py --which-set val

    # Evaluate both:
    python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_human_runs.py --which-set both

    # Skip LLM judge (rule-based only; breakdown cases marked None):
    python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_human_runs.py --skip-llm

    # Overwrite existing run entries:
    python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_human_runs.py --overwrite
"""

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

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_JSON_DIR   = REPO_ROOT / "data" / "dca_data" / "image_pairs_llm_output_exp1"
DEFAULT_SPLIT_FILE = REPO_ROOT / ".cache" / "dca_split_seed42.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "eval_results" / "human_runs"


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


def extract_tokens_and_time(meta: dict) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[float]]:
    """Extract (total, input, output, reasoning, response_time_seconds) from parsed metadata dict.

    Compatible with v1/v2/v3 (key: response.usage) and v4 (same top-level key).
    reasoning_tokens comes from completion_tokens_details.reasoning_tokens (o-series models).
    """
    if not meta:
        return None, None, None, None, None
    usage = (meta.get("response") or {}).get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    total_tokens     = usage.get("total_tokens")
    input_tokens     = usage.get("prompt_tokens")
    output_tokens    = usage.get("completion_tokens")
    reasoning_tokens = details.get("reasoning_tokens")
    response_time    = (meta.get("timing") or {}).get("response_time_seconds")
    try:
        total_tokens     = int(total_tokens)     if total_tokens     is not None else None
        input_tokens     = int(input_tokens)     if input_tokens     is not None else None
        output_tokens    = int(output_tokens)    if output_tokens    is not None else None
        reasoning_tokens = int(reasoning_tokens) if reasoning_tokens is not None else None
        response_time    = float(response_time)  if response_time    is not None else None
    except (TypeError, ValueError):
        pass
    return total_tokens, input_tokens, output_tokens, reasoning_tokens, response_time


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

    # 2) LLM judge
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
        total_tokens, input_tokens, output_tokens, reasoning_tokens, response_time = extract_tokens_and_time(meta or {})

        chg_str = f"{change_score:.3f}" if change_score is not None else "N/A "
        tot_str = f"{total_score:.3f}"  if total_score  is not None else "N/A "
        print(f"    [{idx:03d}/{n_total_pairs}] {pair_name}  "
              f"cls={cls_score:.0f}  chg={chg_str}  tot={tot_str}  src={change_source}",
              flush=True)

        per_pair.append({
            "pair_name":        pair_name,
            "missing":          False,
            "cls_score":        cls_score,
            "change_score":     change_score,
            "total_score":      total_score,
            "change_source":    change_source,
            "total_tokens":     total_tokens,
            "input_tokens":     input_tokens,
            "output_tokens":    output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "response_time":    response_time,
        })

    # Aggregate
    valid = [p for p in per_pair if not p.get("missing")]
    n_total   = len(test_records)
    n_missing = sum(1 for p in per_pair if p.get("missing"))
    n_valid   = len(valid)

    cls_scores        = [p["cls_score"]        for p in valid]
    change_scores     = [p["change_score"]     for p in valid if p["change_score"]     is not None]
    total_scores      = [p["total_score"]      for p in valid if p["total_score"]      is not None]
    tokens            = [p["total_tokens"]     for p in valid if p["total_tokens"]     is not None]
    input_tok_list    = [p["input_tokens"]     for p in valid if p.get("input_tokens")     is not None]
    output_tok_list   = [p["output_tokens"]    for p in valid if p.get("output_tokens")    is not None]
    reasoning_tok_list= [p["reasoning_tokens"] for p in valid if p.get("reasoning_tokens") is not None]
    times             = [p["response_time"]    for p in valid if p["response_time"]    is not None]

    source_counts = {"rule": 0, "llm": 0, "skipped": 0}
    for p in valid:
        src = p.get("change_source", "skipped")
        source_counts[src] = source_counts.get(src, 0) + 1

    return {
        "run":                    run,
        "label":                  RUN_LABELS.get(run, run),
        "n_test_pairs":           n_total,
        "n_evaluated":            n_valid,
        "n_missing":              n_missing,
        "avg_cls_score":          float(np.mean(cls_scores))         if cls_scores         else None,
        "avg_change_score":       float(np.mean(change_scores))      if change_scores      else None,
        "avg_total_score":        float(np.mean(total_scores))       if total_scores       else None,
        "avg_total_tokens":       float(np.mean(tokens))             if tokens             else None,
        "avg_input_tokens":       float(np.mean(input_tok_list))     if input_tok_list     else None,
        "avg_output_tokens":      float(np.mean(output_tok_list))    if output_tok_list    else None,
        "avg_reasoning_tokens":   float(np.mean(reasoning_tok_list)) if reasoning_tok_list else None,
        "avg_response_time":      float(np.mean(times))              if times              else None,
        "n_cls":                  len(cls_scores),
        "n_change":               len(change_scores),
        "n_total_score":          len(total_scores),
        "change_source_counts":   source_counts,
        "per_pair":               per_pair,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate human-prompt DCA runs on val/test splits.")
    parser.add_argument("--json-dir",   type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--which-set",  type=str, choices=["test", "val", "both"], default="test",
                        help="Which split to evaluate: test, val, or both (default: test)")
    parser.add_argument("--skip-llm",  action="store_true",
                        help="Skip LLM judge calls; breakdown cases get change_score=None")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing run entries if they already exist")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_llm = not args.skip_llm

    # ── Guard: abort if any of the current RUNS already exist and --overwrite not set ──
    if not args.overwrite:
        _sets_to_check = ["val", "test"] if args.which_set == "both" else [args.which_set]
        for _sname in _sets_to_check:
            _out = args.output_dir / f"eval_results_{_sname}.json"
            if _out.exists():
                _old = json.loads(_out.read_text(encoding="utf-8"))
                _existing_runs = {r["run"] for r in _old.get("summary", [])}
                _conflicting = [r for r in RUNS if r in _existing_runs]
                if _conflicting:
                    print(f"[ERROR] The following runs already exist in {_out.name}:")
                    for _r in _conflicting:
                        print(f"  {_r}")
                    print("  Use --overwrite to re-evaluate and replace these runs.")
                    sys.exit(1)

    # ── Load split ────────────────────────────────────────────────────────────
    if not args.split_file.exists():
        print(f"[ERROR] Split file not found: {args.split_file}")
        sys.exit(1)

    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    test_pair_names = split.get("test", {}).get("pair_names") or split.get("test_pair_numbers") or []
    if isinstance(test_pair_names, list) and test_pair_names and isinstance(test_pair_names[0], int):
        test_pair_names = [f"pair_{n:03d}" for n in test_pair_names]
    val_pair_names = split.get("val", {}).get("pair_names") or split.get("val_pair_numbers") or []
    if isinstance(val_pair_names, list) and val_pair_names and isinstance(val_pair_names[0], int):
        val_pair_names = [f"pair_{n:03d}" for n in val_pair_names]

    if not args.json_dir.exists():
        print(f"[ERROR] JSON dir not found: {args.json_dir}")
        sys.exit(1)

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

    which = args.which_set
    evaluate_test = which in ("test", "both")
    evaluate_val  = which in ("val",  "both")

    seed = split.get("seed", "?")
    if evaluate_test:
        print(f"Test split: {len(test_pair_names)} pairs  (seed={seed})")
    if evaluate_val:
        print(f"Val  split: {len(val_pair_names)} pairs  (seed={seed})")

    test_records, test_missing = ([], 0)
    val_records,  val_missing  = ([], 0)
    if evaluate_test:
        test_records, test_missing = _load_records(test_pair_names)
        print(f"Loaded {len(test_records)} test pair JSONs  ({test_missing} missing)")
    if evaluate_val:
        val_records, val_missing = _load_records(val_pair_names)
        print(f"Loaded {len(val_records)} val pair JSONs  ({val_missing} missing)")

    def _evaluate_and_save(records: list[dict], set_name: str) -> None:
        if not records:
            print(f"\nNo records for set: {set_name} — skipping.")
            return

        print(f"\nEvaluating {len(RUNS)} runs on {len(records)} {set_name} pairs …")
        print(f"  LLM judge: {'ENABLED' if use_llm else 'DISABLED (--skip-llm)'}\n")

        new_results: list[dict] = []
        for i, run in enumerate(RUNS, 1):
            t0 = time.time()
            print(f"\n  [{i:02d}/{len(RUNS)}] {run}", flush=True)
            res = evaluate_run(run, records, use_llm)
            elapsed = time.time() - t0
            chg_str = f"{res['avg_change_score']:.3f}" if res['avg_change_score'] is not None else "N/A"
            tot_str = f"{res['avg_total_score']:.3f}"  if res['avg_total_score']  is not None else "N/A"
            print(
                f"  → avg: cls={res['avg_cls_score']:.3f}  "
                f"chg={chg_str:>6}  tot={tot_str:>6}  "
                f"src={res['change_source_counts']}  [{elapsed:.1f}s]"
            )
            new_results.append(res)

        # ── Load existing file and merge by run name ──────────────────────────
        results_path = args.output_dir / f"eval_results_{set_name}.json"

        old_run_order:     list[str]       = []
        existing_summary_map: dict[str, dict] = {}
        existing_details_map: dict[str, dict] = {}

        if results_path.exists():
            _old = json.loads(results_path.read_text(encoding="utf-8"))
            for r in _old.get("summary", []):
                old_run_order.append(r["run"])
                existing_summary_map[r["run"]] = r
            for r in _old.get("details", []):
                existing_details_map[r["run"]] = r

        # New results override matching old entries; all others are preserved
        for r in new_results:
            existing_summary_map[r["run"]] = {k: v for k, v in r.items() if k != "per_pair"}
            existing_details_map[r["run"]] = r

        # Preserve original order, append truly new runs at the end
        merged_run_names: list[str] = list(old_run_order)
        for run in RUNS:
            if run not in merged_run_names:
                merged_run_names.append(run)

        merged_summary = [existing_summary_map[r] for r in merged_run_names if r in existing_summary_map]
        merged_details = [existing_details_map[r] for r in merged_run_names if r in existing_details_map]

        saved_meta = {
            "n_pairs":             len(records),
            "which_set":           set_name,
            "split_file":          str(args.split_file),
            "json_dir":            str(args.json_dir),
            "skip_llm":            not use_llm,
            "runs":                merged_run_names,
            "field_weight_cls":    FIELD_WEIGHT_CLS,
            "field_weight_change": FIELD_WEIGHT_CHANGE,
        }
        results_path.write_text(
            json.dumps(
                {"meta": saved_meta, "summary": merged_summary, "details": merged_details},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nResults JSON -> {results_path}  ({len(merged_summary)} runs total)")

        # ── Summary table ────────────────────────────────────────────────────
        print("\n" + "=" * 100)
        print(f"{'Run':<42} {'Cls':>6} {'Chg':>6} {'Tot':>6} {'Time(s)':>8} {'Tokens':>8} {'n_eval':>7}")
        print("-" * 100)
        for r in merged_summary:
            c  = f"{r['avg_cls_score']:.3f}"     if r['avg_cls_score']    is not None else "N/A"
            ch = f"{r['avg_change_score']:.3f}"  if r['avg_change_score'] is not None else "N/A"
            t  = f"{r['avg_total_score']:.3f}"   if r['avg_total_score']  is not None else "N/A"
            ti = f"{r['avg_response_time']:.1f}" if r['avg_response_time'] is not None else "N/A"
            tk = f"{r['avg_total_tokens']:.0f}"  if r['avg_total_tokens']  is not None else "N/A"
            print(f"  {r['label']:<40} {c:>6} {ch:>6} {t:>6} {ti:>8} {tk:>8} {r['n_evaluated']:>7}")
            inp = f"{r['avg_input_tokens']:.0f}"     if r.get('avg_input_tokens')     is not None else "N/A"
            out = f"{r['avg_output_tokens']:.0f}"    if r.get('avg_output_tokens')    is not None else "N/A"
            rsn = f"{r['avg_reasoning_tokens']:.0f}" if r.get('avg_reasoning_tokens') is not None else "N/A"
            print(f"    tok breakdown: input={inp}  output={out}  reasoning={rsn}")
        print("=" * 100)

    if evaluate_test:
        _evaluate_and_save(test_records, "test")
    if evaluate_val:
        _evaluate_and_save(val_records, "val")

    print("\nDone.")


if __name__ == "__main__":
    main()
