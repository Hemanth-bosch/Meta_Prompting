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

Output JSON uses the same per-pair scoring fields as
run_val_test_set_eval_on_human_runs.py so results can be compared side-by-side.

Usage
-----
Required arguments (must always be specified):
  --model-path   Path to the saved DSPy optimized model JSON
  --model-name   LM to use for inference  (e.g. azure/gpt-4.1)
  --split-file   Path to the train/val/test split JSON

Optional arguments (sensible repo-relative defaults):
  --temperature       Inference temperature (default: 0.1)
  --reasoning-effort  Reasoning effort for reasoning models: low|medium|high (default: None)
  --image-dir         Default: data/dca_data/image_pairs
  --gt-dir       Default: data/dca_data/image_pairs_llm_output_exp1
  --output-dir   Default: <this script's dir>/eval_results/optimized_prompt_runs

RUN_KEY and RUN_LABEL are automatically derived from --model-path and
--model-name, so you never need to edit them manually.

Output layout (all under --output-dir):
  eval_results/optimized_prompt_runs/
  ├── eval_results_<run_key>_val.json   ← all runs merged, includes per_pair
  └── eval_results_<run_key>_test.json

Examples
--------
    # Evaluate on test set (default):
    python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py \\
        --model-path .cache/gpt-4.1_MIPROv2_DCA_Light_dca_optimized_2026032716.json \\
        --model-name azure/gpt-4.1 \\
        --split-file .cache/splits/dca_split_seed42_<fingerprint>.json

    # Evaluate on val set:
    python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py \\
        --model-path .cache/gpt-4.1_MIPROv2_DCA_Light_dca_optimized_2026032716.json \\
        --model-name azure/gpt-4.1 \\
        --split-file .cache/splits/dca_split_seed42_<fingerprint>.json \
        --which-set val

    # Evaluate on both val and test:
    python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py \\
        --model-path .cache/gpt-4.1_MIPROv2_DCA_Light_dca_optimized_2026032716.json \\
        --model-name azure/gpt-4.1 \\
        --split-file .cache/splits/dca_split_seed42_<fingerprint>.json \
        --which-set both

    # Use a different temperature:
    python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py \\
        --model-path .cache/gpt-4.1_MIPROv2_DCA_Light_dca_optimized_2026032716.json \\
        --model-name azure/gpt-4.1 \\
        --split-file .cache/splits/dca_split_seed42_<fingerprint>.json \
        --temperature 0.0

    # Overwrite an existing run:
    python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py \\
        --model-path .cache/gpt-4.1_MIPROv2_DCA_Light_dca_optimized_2026032716.json \\
        --model-name azure/gpt-4.1 \\
        --split-file .cache/splits/dca_split_seed42_<fingerprint>.json \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from evaluation.dca_run_val_test_set_eval.split_utils import resolve_default_split_file

# ---------------------------------------------------------------------------
# Imports (after sys.path is set)
# ---------------------------------------------------------------------------
import dspy
import numpy as np

from config import create_lm, AZURE_API_KEY
from signatures.dca import DCAImageComparison
from dca_tournaments.tour_utils import normalize_demos
from evaluation.dca_metric import (
    FIELD_WEIGHT_CLS,
    FIELD_WEIGHT_CHANGE,
    _try_rule_based_match,
    judge_summary_metric,
    compute_classification_score,
    compute_total_score,
)

# ---------------------------------------------------------------------------
# Defaults for paths that are fixed by repo layout (no need to pass every run)
# ---------------------------------------------------------------------------
DEFAULT_IMAGE_DIR   = REPO_ROOT / "data" / "dca_data" / "image_pairs"
DEFAULT_GT_DIR      = REPO_ROOT / "data" / "dca_data" / "image_pairs_llm_output_exp1"
DEFAULT_OUTPUT_DIR  = SCRIPT_DIR / "eval_results" / "optimized_prompt_runs"
DEFAULT_SPLIT_FILE = resolve_default_split_file(REPO_ROOT)
DEFAULT_TEMPERATURE = 0.1

# ---------------------------------------------------------------------------
# Derive RUN_KEY and RUN_LABEL from the model filename + inference model name.
#
# Filename convention:
#   {train_model}_{Strategy}_dca_optimized_{timestamp}.json
#   e.g. gpt-4.1_MIPROv2_DCA_Light_dca_optimized_2026032716.json
#
# Produces:
#   run_key   → "miprov2_dca_light_gpt41_2026032716_run_gpt41"
#   run_label → "MIPROv2-DCA-Light/gpt-4.1_2026032716_run_gpt-4.1"
# ---------------------------------------------------------------------------

def _derive_run_ids(model_path: Path, inference_model_name: str) -> tuple[str, str]:
    """Return (run_key, run_label) derived from the model filename and inference model."""
    stem = model_path.stem  # e.g. gpt-4.1_MIPROv2_DCA_Light_dca_optimized_2026032716

    train_model = strategy = timestamp = None
    for marker in ["_MIPROv2_", "_GEPA_", "_SimBA_", "_Bootstrap_"]:
        idx = stem.find(marker)
        if idx > 0:
            train_model = stem[:idx]                                 # gpt-4.1
            strategy_prefix = marker[1:-1]                           # MIPROv2
            remainder = stem[idx + len(marker):]                     # DCA_Light_dca_optimized_2026032716
            opt_marker = "_dca_optimized_"
            opt_idx = remainder.find(opt_marker)
            if opt_idx >= 0:
                strategy_suffix = remainder[:opt_idx]                # DCA_Light
                timestamp = remainder[opt_idx + len(opt_marker):]    # 2026032716
                strategy = f"{strategy_prefix}_{strategy_suffix}"    # MIPROv2_DCA_Light
            else:
                strategy = strategy_prefix
                timestamp = remainder
            break

    # Inference model display name — strip "azure/" prefix
    inf_display = inference_model_name.split("/")[-1] if "/" in inference_model_name else inference_model_name

    if train_model is None:
        # Filename doesn't match convention — fall back to stem + model name
        key = stem.lower().replace("-", "").replace(".", "") + f"_run_{inf_display.replace('-', '').replace('.', '')}"
        label = f"{stem}_run_{inf_display}"
        return key, label

    def _slug(s: str) -> str:
        return s.lower().replace("-", "").replace(".", "")

    run_key   = f"{_slug(strategy)}_{_slug(train_model)}_{timestamp}_run_{_slug(inf_display)}"
    run_label = f"{strategy.replace('_', '-')}/{train_model}_{timestamp}_run_{inf_display}"
    return run_key, run_label


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
    run_key: str,
    run_label: str,
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
    parser.add_argument("--model-path",  type=Path, required=True,
                        help="Path to saved DSPy optimized model JSON  (e.g. .cache/gpt-4.1_MIPROv2_DCA_Light_dca_optimized_2026032716.json)")
    parser.add_argument("--model-name",  type=str, required=True,
                        help="LM to use for inference  (e.g. azure/gpt-4.1)")
    parser.add_argument("--split-file",  type=Path, default=DEFAULT_SPLIT_FILE,
                        help="Path to train/val/test split JSON")
    parser.add_argument("--temperature",  type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Inference temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--image-dir",   type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--gt-dir",      type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--output-dir",  type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--which-set",   type=str, choices=["val", "test", "both"], default="test",
                        help="Which split to evaluate (default: test)")
    parser.add_argument("--skip-llm",    action="store_true",
                        help="Skip LLM judge; breakdown cases get change_score=None")
    parser.add_argument("--overwrite",   action="store_true",
                        help="Overwrite the current run's entry if it already exists in the results file")
    parser.add_argument("--no-cot",      action="store_true",
                        help="Use dspy.Predict instead of dspy.ChainOfThought "
                             "(keeps few-shot demos + optimized instruction, removes reasoning step)")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["None", "low", "medium", "high"],
                        help="Reasoning effort for o-series / reasoning models (e.g. gpt-5). "
                             "Passed to create_lm as reasoning_effort kwarg.")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Optional suffix appended to run_key and output filename "
                             "(e.g. 'rep01'). Used automatically by --n-repeats.")
    parser.add_argument("--n-repeats", type=int, default=None,
                        help="Launch N independent parallel repeats (each gets --run-id repNN). "
                             "Results are written to separate files for easy comparison.")
    args = parser.parse_args()

    # ── n-repeats: spawn N subprocesses, each with a unique --run-id ─────────
    if args.n_repeats is not None:
        if args.n_repeats < 1:
            print("[ERROR] --n-repeats must be >= 1")
            sys.exit(1)
        base_argv = [a for a in sys.argv[1:]
                     if not a.startswith("--n-repeats") and a != str(args.n_repeats)]
        # Remove any existing --run-id passed alongside --n-repeats
        filtered: list[str] = []
        skip_next = False
        for tok in base_argv:
            if skip_next:
                skip_next = False
                continue
            if tok == "--run-id":
                skip_next = True
                continue
            if tok.startswith("--run-id="):
                continue
            filtered.append(tok)
        base_argv = filtered

        pad = len(str(args.n_repeats))
        cmds = [
            [sys.executable, __file__] + base_argv + ["--run-id", f"rep{i:0{pad}d}"]
            for i in range(1, args.n_repeats + 1)
        ]

        # Build a unique log-tag from the run identity so different experiments
        # never share the same log filenames.
        _rk, _ = _derive_run_ids(args.model_path, args.model_name)
        if args.reasoning_effort:
            _rk += f"_re{args.reasoning_effort}"
        if args.no_cot:
            _rk += "_nocot"
        if args.run_id:          # shouldn't be set together with --n-repeats, but guard anyway
            _rk += f"_{args.run_id}"
        log_tag = _rk            # e.g. miprov2_dca_light_gpt41mini_20260402145119_run_gpt41mini_nocot
        
        log_tag += f"_temp{args.temperature}" if args.temperature != DEFAULT_TEMPERATURE else str(args.temperature)
        # Log files go to <output-dir>/repeat_logs/<log_tag>/repN.log
        logs_dir = args.output_dir / "repeat_logs" / log_tag
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_paths = {
            i: logs_dir / f"rep{i:0{pad}d}.log"
            for i in range(1, args.n_repeats + 1)
        }

        print(f"Launching {args.n_repeats} parallel repeats …")
        print(f"Logs directory: {logs_dir}")
        for i, cmd in enumerate(cmds, 1):
            print(f"  rep {i:0{pad}d}: {' '.join(cmd[-6:])} …  -> {log_paths[i].name}")

        def _run_one(cmd: list[str], rep_idx: int) -> tuple[int, str, str, str]:
            result = subprocess.run(cmd, capture_output=True, text=True)
            # Write full output to log immediately
            log_path = log_paths[rep_idx]
            with log_path.open("w", encoding="utf-8") as fh:
                fh.write(f"=== rep {rep_idx:0{pad}d} — exit_code={result.returncode} ===\n\n")
                fh.write("--- STDOUT ---\n")
                fh.write(result.stdout or "(empty)\n")
                if result.stderr.strip():
                    fh.write("\n--- STDERR ---\n")
                    fh.write(result.stderr)
            return rep_idx, result.returncode, result.stdout, result.stderr

        results_list: list[tuple[int, int, str, str]] = []
        with ThreadPoolExecutor(max_workers=args.n_repeats) as pool:
            futures = {pool.submit(_run_one, cmd, i): i for i, cmd in enumerate(cmds, 1)}
            for fut in as_completed(futures):
                rep_idx, rc, stdout, stderr = fut.result()
                results_list.append((rep_idx, rc, stdout, stderr))
                tag = f"[rep {rep_idx:0{pad}d}]"
                status = "OK" if rc == 0 else f"FAILED (exit {rc})"
                print(f"  {tag} finished — {status}  log -> {log_paths[rep_idx]}")

        # Print full output of every repeat once all are done
        results_list.sort(key=lambda x: x[0])
        print(f"\n{'#'*80}")
        print("FULL OUTPUT OF ALL REPEATS")
        print(f"{'#'*80}")
        for rep_idx, rc, stdout, stderr in results_list:
            tag = f"rep {rep_idx:0{pad}d}"
            print(f"\n{'='*80}")
            print(f"  {tag}  exit_code={rc}  (log: {log_paths[rep_idx]})")
            print(f"{'='*80}")
            print(stdout or "(no stdout)")
            if stderr.strip():
                print(f"--- STDERR ---\n{stderr.strip()}")

        print(f"\n{'#'*80}")
        n_ok   = sum(1 for _, rc, _, _ in results_list if rc == 0)
        n_fail = len(results_list) - n_ok
        print(f"All {args.n_repeats} repeats finished — {n_ok} OK, {n_fail} FAILED")
        print(f"Logs saved to: {logs_dir}")
        print(f"{'#'*80}")
        sys.exit(0 if n_fail == 0 else 1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_llm = not args.skip_llm
    no_cot  = args.no_cot

    # Derive run_key and run_label from the model path and inference model name.
    run_key, run_label = _derive_run_ids(args.model_path, args.model_name)
    model_name = args.model_name

    # Append reasoning effort suffix so different efforts don't collide
    if args.reasoning_effort:
        re_suffix_key   = f"_re{args.reasoning_effort}"       # e.g. _relow
        re_suffix_label = f" (RE={args.reasoning_effort})"    # e.g. (RE=low)
        run_key   += re_suffix_key
        run_label += re_suffix_label

    if no_cot:
        active_run_key   = run_key + "_nocot"
        active_run_label = run_label + " (no CoT)"
        file_prefix      = "eval_results_no_CoT_" + run_key
    else:
        active_run_key   = run_key
        active_run_label = run_label
        file_prefix      = "eval_results_" + run_key

    # Append --run-id suffix so parallel repeats write to separate files
    if args.run_id:
        rid = args.run_id
        active_run_key   = active_run_key   + f"_{rid}"
        active_run_label = active_run_label + f" [{rid}]"
        file_prefix      = file_prefix      + f"_{rid}"

    # Append temperature — same pattern as log_tag: "0.1" for default, "_temp0.7" for others
    # This ensures different temperatures produce separate JSON files (mirrors log folder naming)
    _temp_suffix   = f"_temp{args.temperature}" if args.temperature != DEFAULT_TEMPERATURE else str(args.temperature)
    active_run_key += _temp_suffix
    file_prefix    += _temp_suffix

    sets_to_run = ["val", "test"] if args.which_set == "both" else [args.which_set]

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

    lm_kwargs = {}
    if args.reasoning_effort is not None:
        lm_kwargs["reasoning_effort"] = args.reasoning_effort

    print(f"Inference LM : {model_name}  (--model-name)")
    print(f"              cache=False, temperature={args.temperature}"
          + (f", reasoning_effort={args.reasoning_effort}" if args.reasoning_effort else ""))
    lm = create_lm(model_name, cache=False, temperature=args.temperature, **lm_kwargs)
    dspy.configure(lm=lm)

    mode_label = "Predict (no CoT — few-shots only)" if no_cot else "ChainOfThought (few-shots + reasoning)"
    print(f"Mode         : {mode_label}")
    print(f"Loading optimized program: {args.model_path.name}")
    comparator = dspy.Predict(DCAImageComparison) if no_cot else dspy.ChainOfThought(DCAImageComparison)
    comparator.load(str(args.model_path))
    # Normalize expected_* demo keys from old/buggy saved files
    _n_fixed = normalize_demos(comparator)
    if _n_fixed:
        print(f"[demo-fix] Normalized {_n_fixed} demo(s) with expected_* keys")
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
        result["model_path"]        = str(args.model_path)
        result["model_name"]        = model_name
        result["temperature"]       = args.temperature
        result["reasoning_effort"]  = args.reasoning_effort
        result["no_cot"]            = no_cot

        # ── Load existing JSON and merge by run key ───────────────────────────
        results_path = args.output_dir / f"{file_prefix}_{set_name}.json"

        old_run_order:     list[str]       = []
        existing_runs_map: dict[str, dict] = {}

        if results_path.exists():
            _old = json.loads(results_path.read_text(encoding="utf-8"))
            for r in _old.get("results", []):
                old_run_order.append(r["run"])
                existing_runs_map[r["run"]] = r

        # New result overrides matching old entry; all others are preserved
        existing_runs_map[active_run_key] = result

        # Preserve original order, append truly new run at the end
        merged_run_names: list[str] = list(old_run_order)
        if active_run_key not in merged_run_names:
            merged_run_names.append(active_run_key)

        merged_results = [existing_runs_map[r] for r in merged_run_names if r in existing_runs_map]

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

        results_path.write_text(
            json.dumps({"meta": saved_meta, "results": merged_results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nResults JSON  -> {results_path}  ({len(merged_results)} run(s) total)")

        # ── Summary table (all merged runs) ──────────────────────────────────
        print(f"\n{'='*100}")
        print(f"{'Run':<42} {'Cls':>6} {'Chg':>6} {'Tot':>6} {'Time(s)':>8} {'Tokens':>8} {'n_eval':>7}")
        print(f"{'-'*100}")
        for r in merged_results:
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

    for set_name in sets_to_run:
        _run_set(set_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
