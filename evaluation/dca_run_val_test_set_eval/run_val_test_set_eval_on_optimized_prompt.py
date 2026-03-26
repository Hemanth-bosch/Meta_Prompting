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

from config import create_lm, DEFAULT_MODEL_NAME
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
DEFAULT_MODEL_PATH = REPO_ROOT / ".cache" / "gpt-4.1-mini_MIPROv2_DCA_Light_dca_optimized_2026032517.json"
DEFAULT_SPLIT_FILE = REPO_ROOT / ".cache" / "dca_split_seed42.json"
DEFAULT_IMAGE_DIR  = REPO_ROOT / "data" / "dca_data" / "image_pairs"
DEFAULT_GT_DIR     = REPO_ROOT / "data" / "dca_data" / "image_pairs_llm_output_exp1"
OUTPUT_DIR         = SCRIPT_DIR / "eval_results"

RUN_KEY   = "miprov2_light_gpt41_mini"
RUN_LABEL = "MIPROv2-Light/GPT-4.1-mini"


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


def _get_tokens_from_lm_history(lm: dspy.LM, history_len_before: int) -> Optional[int]:
    """Sum total_tokens from all new LM history entries added since history_len_before."""
    try:
        new_entries = lm.history[history_len_before:]
        if not new_entries:
            return None
        total = 0
        for entry in new_entries:
            usage = entry.get("usage") or {}
            t = usage.get("total_tokens")
            if t is not None:
                total += int(t)
            else:
                # fallback: prompt + completion
                pt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                ct = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
                total += pt + ct
        return total if total > 0 else None
    except Exception:
        return None


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
        total_tokens  = _get_tokens_from_lm_history(lm, history_len)

        pred_discard = str(getattr(pred, "discard_image", "") or "").strip()
        pred_changes = str(getattr(pred, "changes", "")        or "").strip()

        tok_str = f"  tok={total_tokens}" if total_tokens else ""
        print(f"  [{i:03d}/{len(pair_names)}] {pair_name}  "
              f"discard={pred_discard}  t={response_time:.1f}s{tok_str}")

        per_pair.append({
            "pair_name":    pair_name,
            "missing":      False,
            "pred_discard": pred_discard,
            "pred_changes": pred_changes,
            "response_time": response_time,
            "total_tokens": total_tokens,
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

        total_score = compute_total_score(cls_score, change_score) if change_score is not None else None

        per_pair_eval.append({
            "pair_name":     pair_name,
            "missing":       False,
            "cls_score":     cls_score,
            "change_score":  change_score,
            "total_score":   total_score,
            "change_source": change_source,
            "total_tokens":  rec["total_tokens"],
            "response_time": rec["response_time"],
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
    tokens        = [p["total_tokens"] for p in valid if p["total_tokens"] is not None]
    times         = [p["response_time"] for p in valid if p["response_time"] is not None]

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
        "avg_cls_score":        float(np.mean(cls_scores))    if cls_scores    else None,
        "avg_change_score":     float(np.mean(change_scores)) if change_scores else None,
        "avg_total_score":      float(np.mean(total_scores))  if total_scores  else None,
        "avg_total_tokens":     float(np.mean(tokens))        if tokens        else None,
        "avg_response_time":    float(np.mean(times))         if times         else None,
        "n_cls":                len(cls_scores),
        "n_change":             len(change_scores),
        "n_total_score":        len(total_scores),
        "change_source_counts": source_counts,
        "per_pair":             per_pair,
    }


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
                        help="Overwrite existing output files")
    parser.add_argument("--no-cot",      action="store_true",
                        help="Use dspy.Predict instead of dspy.ChainOfThought "
                             "(keeps few-shot demos + optimized instruction, removes reasoning step)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_llm = not args.skip_llm
    no_cot  = args.no_cot

    if no_cot:
        active_run_key   = "miprov2_light_gpt41_mini_nocot"
        active_run_label = "MIPROv2-Light/GPT-4.1-mini (no CoT)"
        file_prefix      = "eval_results_miprov2_nocot"
    else:
        active_run_key   = RUN_KEY
        active_run_label = RUN_LABEL
        file_prefix      = "eval_results_miprov2"

    # Guard: abort if outputs already exist
    sets_to_run = ["val", "test"] if args.which_set == "both" else [args.which_set]
    for set_name in sets_to_run:
        out = args.output_dir / f"{file_prefix}_{set_name}.json"
        if out.exists() and not args.overwrite:
            print(f"[ERROR] Results already exist: {out}")
            print("  Use --overwrite to re-run.")
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
    print(f"              cache=False  (no cached responses)")
    lm = create_lm(model_name, cache=False)
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

        # Summary table (same format as run_val_test_set_eval_on_human_runs.py)
        cls_s  = f"{result['avg_cls_score']:.3f}"    if result['avg_cls_score']    is not None else "N/A"
        chg_s  = f"{result['avg_change_score']:.3f}" if result['avg_change_score'] is not None else "N/A"
        tot_s  = f"{result['avg_total_score']:.3f}"  if result['avg_total_score']  is not None else "N/A"
        time_s = f"{result['avg_response_time']:.1f}" if result['avg_response_time'] is not None else "N/A"
        tok_s  = f"{result['avg_total_tokens']:.0f}"  if result['avg_total_tokens']  is not None else "N/A"

        print(f"\n{'='*100}")
        print(f"{'Run':<42} {'Cls':>6} {'Chg':>6} {'Tot':>6} {'Time(s)':>8} {'Tokens':>8} {'n_eval':>7}")
        print(f"{'-'*100}")
        print(f"  {result['label']:<40} {cls_s:>6} {chg_s:>6} {tot_s:>6} {time_s:>8} {tok_s:>8} {result['n_evaluated']:>7}")
        print(f"{'='*100}")
        print(f"  src={result['change_source_counts']}")

        # ── Save summary JSON ─────────────────────────────────────────────────
        meta = {
            "run":                active_run_key,
            "label":              active_run_label,
            "model_path":         str(args.model_path),
            "model_name":         model_name,
            "which_set":          set_name,
            "n_pairs":            len(pair_names),
            "skip_llm":           not use_llm,
            "no_cot":             no_cot,
            "field_weight_cls":   FIELD_WEIGHT_CLS,
            "field_weight_change": FIELD_WEIGHT_CHANGE,
        }

        r_clean = {k: v for k, v in result.items() if k != "per_pair"}
        summary_path = args.output_dir / f"{file_prefix}_{set_name}.json"
        summary_path.write_text(
            json.dumps({"meta": meta, "results": [r_clean]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nResults JSON  -> {summary_path}")

        full_path = args.output_dir / f"{file_prefix}_full_{set_name}.json"
        full_path.write_text(
            json.dumps({"meta": meta, "results": [result]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Full results  -> {full_path}")

    for set_name in sets_to_run:
        _run_set(set_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
