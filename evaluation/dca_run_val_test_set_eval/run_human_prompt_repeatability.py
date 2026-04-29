#!/usr/bin/env python3
"""Repeatability evaluation for human-written DCA prompts.

Runs live inference using a plain human prompt file (no DSPy demos, no optimized
model) and evaluates with the same classification + LLM-judge change metric.
Supports --n-repeats for parallel repeatability testing, mirroring the behaviour
of run_val_test_set_eval_on_optimized_prompt.py.

Configurable grid
-----------------
Edit the RUNS dict below to add/remove prompt × model combinations.
Each entry is: "run_name": RunConfig(prompt_file, model_name, [no_cot])

Run selection
-------------
    # All runs in RUNS (default):
    python run_human_prompt_repeatability.py --which-set val

    # Specific runs only:
    python run_human_prompt_repeatability.py --runs v1_gpt41mini v2_gpt41mini --which-set val

    # 10 parallel repeats of two runs:
    python run_human_prompt_repeatability.py --runs v1_gpt41mini v2_gpt41mini --n-repeats 10 --which-set val

Output layout
-------------
  eval_results/human_prompt_runs/
  ├── eval_results_<run_key>_<set>.json       ← per-run results JSON
  └── repeat_logs/
      └── <run_key>_<temp>/
          ├── rep01.log
          ├── rep02.log
          └── ...

Usage examples
--------------
    python evaluation/dca_run_val_test_set_eval/run_human_prompt_repeatability.py \\
        --runs v1_gpt41mini v1_gpt41 --which-set val --n-repeats 10

    python evaluation/dca_run_val_test_set_eval/run_human_prompt_repeatability.py \\
        --runs v2_gpt41mini --which-set val --temperature 0.7 --n-repeats 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dspy
import numpy as np

from config import create_lm, AZURE_API_KEY
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
HUMAN_PROMPTS_DIR = REPO_ROOT / "evaluation" / "dca_baseline_runs" / "human_prompts"
DEFAULT_IMAGE_DIR  = REPO_ROOT / "data" / "dca_data" / "image_pairs"
DEFAULT_GT_DIR     = REPO_ROOT / "data" / "dca_data" / "image_pairs_llm_output_exp1"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "eval_results" / "human_prompt_runs"
DEFAULT_SPLIT_FILE = REPO_ROOT / ".cache" / "dca_split_seed42.json"
DEFAULT_TEMPERATURE = 0.1


# ---------------------------------------------------------------------------
# CONFIGURABLE GRID — edit this to add/remove runs
#
# Each entry: "run_name": RunConfig(
#     prompt_file = path to .md prompt (relative to HUMAN_PROMPTS_DIR, or absolute),
#     model_name  = LM identifier passed to create_lm(),
# )
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    prompt_file: str   # relative to HUMAN_PROMPTS_DIR, or absolute path
    model_name:  str


RUNS: dict[str, RunConfig] = {
    # ── v1 prompt (midas) ────────────────────────────────────────────
    "v1_gpt41mini": RunConfig("v1_prompt_midas.md",  "azure/gpt-4.1-mini"),
    "v1_gpt41":     RunConfig("v1_prompt_midas.md",  "azure/gpt-4.1"),

    # ── v2 prompt (cr) ───────────────────────────────────────────────
    "v2_gpt41mini": RunConfig("v2_prompt_cr.md",     "azure/gpt-4.1-mini"),
    "v2_gpt41":     RunConfig("v2_prompt_cr.md",     "azure/gpt-4.1"),

    # # ── v3 prompt (cr) ───────────────────────────────────────────────
    # "v3_gpt41mini": RunConfig("v3_prompt_cr.md",     "azure/gpt-4.1-mini"),
    # "v3_gpt41":     RunConfig("v3_prompt_cr.md",     "azure/gpt-4.1"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_key(run_name: str, temperature: float) -> str:
    """Unique key for file/folder naming from run name + temperature."""
    temp_sfx = f"_temp{temperature}" if temperature != DEFAULT_TEMPERATURE else str(temperature)
    return f"{run_name}{temp_sfx}"


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


def _get_tokens_from_lm_history(lm: dspy.LM, history_len_before: int):
    """Return (total, input, output, reasoning) from new LM history entries."""
    def _get(obj, key):
        if obj is None:
            return None
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    try:
        new_entries = lm.history[history_len_before:]
        if not new_entries:
            return None, None, None, None
        total = input_t = output_t = reasoning_t = 0
        for entry in new_entries:
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


def compute_change_score(human: str, model: str, use_llm: bool):
    """Return (score, source) where source ∈ {'rule', 'llm', 'skipped'}."""
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
# Azure pre-flight
# ---------------------------------------------------------------------------

def _preflight_azure_check(n_pairs: int, avg_seconds_per_pair: float = 20.0) -> None:
    if AZURE_API_KEY:
        return
    try:
        from azure.identity import AzureCliCredential
        token = AzureCliCredential().get_token("https://cognitiveservices.azure.com/.default")
        expires_in_s = token.expires_on - time.time()
        estimated_s  = n_pairs * avg_seconds_per_pair
        if expires_in_s < estimated_s + 300:
            print(
                f"  [WARN] Azure CLI token expires in {expires_in_s/60:.0f} min "
                f"but run will take ~{estimated_s/60:.0f} min. Run 'az login' to refresh."
            )
        else:
            print(f"  Azure CLI token: valid for {expires_in_s/60:.0f} min "
                  f"(est. run ~{estimated_s/60:.0f} min) — OK")
    except Exception as exc:
        print(f"\n[ERROR] Azure CLI credentials are not valid: {exc}")
        print("  Please run 'az login' and retry.\n")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    comparator: dspy.Predict,
    pair_names: list[str],
    image_dir: Path,
) -> list[dict]:
    """Run the human-prompt model on all pairs. Returns raw predictions with timing."""
    lm = dspy.settings.lm
    per_pair: list[dict] = []

    for i, pair_name in enumerate(pair_names, 1):
        pair_dir  = image_dir / pair_name
        orig_path = pair_dir / "original.png"
        reim_path = pair_dir / "reimage.png"

        try:
            image_original = dspy.Image.from_file(str(orig_path)) if orig_path.exists() else None
            image_reimage  = dspy.Image.from_file(str(reim_path)) if reim_path.exists() else None
        except Exception as exc:
            print(f"  [{i:03d}/{len(pair_names)}] {pair_name}  [WARN] image load failed: {exc}")
            per_pair.append({"pair_name": pair_name, "missing": True, "error": str(exc)})
            continue

        history_len = len(lm.history) if hasattr(lm, "history") else 0
        t0 = time.time()
        try:
            pred = comparator(image_original=image_original, image_reimage=image_reimage)
        except Exception as exc:
            print(f"  [{i:03d}/{len(pair_names)}] {pair_name}  [WARN] inference error: {exc}")
            per_pair.append({"pair_name": pair_name, "missing": True, "error": str(exc)})
            continue
        response_time = time.time() - t0
        total_tokens, input_tokens, output_tokens, reasoning_tokens = \
            _get_tokens_from_lm_history(lm, history_len)

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
            total_score = compute_total_score(cls_score, 0.0)
        else:
            total_score = None

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


def aggregate_results(per_pair: list[dict], run_key: str, run_label: str) -> dict:
    valid = [p for p in per_pair if not p.get("missing")]

    cls_scores       = [p["cls_score"]        for p in valid]
    change_scores    = [p["change_score"]      for p in valid if p["change_score"]     is not None]
    total_scores     = [p["total_score"]       for p in valid if p["total_score"]      is not None]
    tokens           = [p["total_tokens"]      for p in valid if p.get("total_tokens") is not None]
    input_tok_list   = [p["input_tokens"]      for p in valid if p.get("input_tokens") is not None]
    output_tok_list  = [p["output_tokens"]     for p in valid if p.get("output_tokens") is not None]
    reason_tok_list  = [p["reasoning_tokens"]  for p in valid if p.get("reasoning_tokens") is not None]
    times            = [p["response_time"]     for p in valid if p.get("response_time") is not None]

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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate human prompt DCA runs on val/test splits with repeatability support."
    )
    parser.add_argument(
        "--runs", nargs="+", default=None,
        metavar="RUN",
        help=f"Which runs to evaluate (default: all). Choices: {list(RUNS.keys())}"
    )
    parser.add_argument("--split-file",  type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--image-dir",   type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--gt-dir",      type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--output-dir",  type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--which-set",   type=str, choices=["val", "test", "both"], default="val")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Inference temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--skip-llm",    action="store_true",
                        help="Skip LLM judge for change score")
    parser.add_argument("--overwrite",   action="store_true",
                        help="Overwrite existing run entries")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Repeat suffix (e.g. 'rep01'). Set automatically by --n-repeats.")
    parser.add_argument("--n-repeats", type=int, default=None,
                        help="Launch N parallel independent repeats.")
    args = parser.parse_args()

    # ── Validate run selection ────────────────────────────────────────────────
    selected_run_names: list[str] = args.runs if args.runs else list(RUNS.keys())
    unknown = [r for r in selected_run_names if r not in RUNS]
    if unknown:
        print(f"[ERROR] Unknown run name(s): {unknown}")
        print(f"  Available: {list(RUNS.keys())}")
        sys.exit(1)

    # ── n-repeats: identical pattern to run_val_test_set_eval_on_optimized_prompt.py
    #
    # For each run_name (sequentially): launch N repeats ALL IN PARALLEL,
    # wait for all N to finish, print their output, then move to the next run.
    #
    # Layout (one log dir per run, one file per repeat):
    #   repeat_logs/
    #     v1_gpt41mini0.1/  rep01.log … rep10.log   ← all 10 launched at once
    #     v1_gpt410.1/      rep01.log … rep10.log   ← then these 10
    #     v2_gpt41mini0.1/  ...                     ← then these 10, etc.
    if args.n_repeats is not None:
        if args.n_repeats < 1:
            print("[ERROR] --n-repeats must be >= 1")
            sys.exit(1)

        # Build base argv: strip --n-repeats, --run-id, and --runs + its values.
        # We inject --runs <single_run_name> fresh for each batch.
        skip_next = False
        skip_runs_values = False
        base_argv_no_runs: list[str] = []
        for tok in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if tok in ("--n-repeats", "--run-id"):
                skip_next = True
                continue
            if tok.startswith("--n-repeats=") or tok.startswith("--run-id="):
                continue
            if tok == "--runs":
                skip_runs_values = True
                continue
            if skip_runs_values:
                if tok.startswith("--"):
                    skip_runs_values = False   # end of --runs values; fall through
                else:
                    continue                   # a run name value — drop it
            base_argv_no_runs.append(tok)

        pad = len(str(args.n_repeats))
        temp_sfx = f"_temp{args.temperature}" if args.temperature != DEFAULT_TEMPERATURE else str(args.temperature)

        def _run_one_repeat(rep_idx: int, cmd: list[str], log_path: Path) -> tuple[int, int, str, str]:
            result = subprocess.run(cmd, capture_output=True, text=True)
            with log_path.open("w", encoding="utf-8") as fh:
                fh.write(f"=== rep {rep_idx:0{pad}d} — exit_code={result.returncode} ===\n\n")
                fh.write("--- STDOUT ---\n")
                fh.write(result.stdout or "(empty)\n")
                if result.stderr.strip():
                    fh.write("\n--- STDERR ---\n")
                    fh.write(result.stderr)
            return rep_idx, result.returncode, result.stdout, result.stderr

        overall_ok = overall_fail = 0

        for run_name in selected_run_names:
            log_tag  = f"{run_name}{temp_sfx}"
            logs_dir = args.output_dir / "repeat_logs" / log_tag
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_paths = {i: logs_dir / f"rep{i:0{pad}d}.log" for i in range(1, args.n_repeats + 1)}
            cmds = [
                [sys.executable, __file__]
                + base_argv_no_runs
                + ["--runs", run_name, "--run-id", f"rep{i:0{pad}d}"]
                for i in range(1, args.n_repeats + 1)
            ]

            print(f"\nLaunching {args.n_repeats} parallel repeats for: {run_name} …")
            print(f"Logs directory: {logs_dir}")
            for i, cmd in enumerate(cmds, 1):
                print(f"  rep {i:0{pad}d}: {' '.join(cmd[-4:])} …  -> {log_paths[i].name}")

            results_list: list[tuple[int, int, str, str]] = []
            with ThreadPoolExecutor(max_workers=args.n_repeats) as pool:
                futures = {
                    pool.submit(_run_one_repeat, i, cmd, log_paths[i]): i
                    for i, cmd in enumerate(cmds, 1)
                }
                for fut in as_completed(futures):
                    rep_idx, rc, stdout, stderr = fut.result()
                    results_list.append((rep_idx, rc, stdout, stderr))
                    status = "OK" if rc == 0 else f"FAILED (exit {rc})"
                    print(f"  [rep {rep_idx:0{pad}d}] finished — {status}  log -> {log_paths[rep_idx]}")

            # Print full output of every repeat once all are done (same as optimized script)
            results_list.sort(key=lambda x: x[0])
            print(f"\n{'#'*80}")
            print(f"FULL OUTPUT — {run_name}")
            print(f"{'#'*80}")
            for rep_idx, rc, stdout, stderr in results_list:
                print(f"\n{'='*80}")
                print(f"  {run_name}  rep {rep_idx:0{pad}d}  exit_code={rc}  (log: {log_paths[rep_idx]})")
                print(f"{'='*80}")
                print(stdout or "(no stdout)")
                if stderr.strip():
                    print(f"--- STDERR ---\n{stderr.strip()}")

            n_ok   = sum(1 for _, rc, _, _ in results_list if rc == 0)
            n_fail = len(results_list) - n_ok
            overall_ok   += n_ok
            overall_fail += n_fail
            print(f"\n{'#'*80}")
            print(f"{run_name}: {args.n_repeats} repeats finished — {n_ok} OK, {n_fail} FAILED")
            print(f"Logs saved to: {logs_dir}")
            print(f"{'#'*80}")

        total = len(selected_run_names) * args.n_repeats
        print(f"\n{'='*70}")
        print(f"ALL RUNS DONE — {total} total: {overall_ok} OK, {overall_fail} FAILED")
        print(f"{'='*70}")
        sys.exit(0 if overall_fail == 0 else 1)

    # ── Single evaluation pass ───────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_llm = not args.skip_llm

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

    for d, label in [(args.image_dir, "image-dir"), (args.gt_dir, "gt-dir")]:
        if not d.exists():
            print(f"[ERROR] {label} not found: {d}")
            sys.exit(1)

    sets_to_run = ["val", "test"] if args.which_set == "both" else [args.which_set]

    # ── Eval each selected run ────────────────────────────────────────────────
    for run_name in selected_run_names:
        cfg = RUNS[run_name]

        # Resolve prompt file path
        prompt_path = Path(cfg.prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = HUMAN_PROMPTS_DIR / cfg.prompt_file
        if not prompt_path.exists():
            print(f"[ERROR] Prompt file not found: {prompt_path}")
            sys.exit(1)

        prompt_content = prompt_path.read_text(encoding="utf-8")

        # Build run_key / run_label / file_prefix
        temp_sfx   = f"_temp{args.temperature}" if args.temperature != DEFAULT_TEMPERATURE else str(args.temperature)
        run_key    = f"{run_name}{temp_sfx}"
        run_label  = f"{run_name} ({cfg.model_name})"
        file_prefix = f"eval_human_{run_key}"

        # Append --run-id for parallel repeats
        if args.run_id:
            run_key    = run_key   + f"_{args.run_id}"
            run_label  = run_label + f" [{args.run_id}]"
            file_prefix = file_prefix + f"_{args.run_id}"

        # ── Configure LM + DSPy ───────────────────────────────────────────────
        print(f"\n{'='*80}")
        print(f"Run         : {run_name}")
        print(f"Prompt file : {prompt_path.name}")
        print(f"Model       : {cfg.model_name}  temperature={args.temperature}")
        print(f"{'='*80}")

        lm = create_lm(cfg.model_name, cache=False, temperature=args.temperature)
        dspy.configure(lm=lm)

        # Build comparator with human prompt injected as instruction (no demos)
        comparator = dspy.Predict(DCAImageComparison)
        comparator.signature = comparator.signature.with_instructions(prompt_content)

        # Clear field descs so they don't add boilerplate to the prompt
        for fname in comparator.signature.fields:
            comparator.signature = comparator.signature.with_updated_fields(fname, desc="")

        print(f"Prompt injected ({len(prompt_content)} chars).  No demos loaded.")

        # ── Per-set evaluation ─────────────────────────────────────────────────
        total_pairs = sum(len(_get_pair_names(s)) for s in sets_to_run)
        _preflight_azure_check(total_pairs)

        for set_name in sets_to_run:
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

            # Guard: check if run already exists
            results_path = args.output_dir / f"{file_prefix}_{set_name}.json"
            if not args.overwrite and results_path.exists():
                _old = json.loads(results_path.read_text(encoding="utf-8"))
                _existing = {r["run"] for r in _old.get("results", [])}
                if run_key in _existing:
                    print(f"[ERROR] Run '{run_key}' already exists in {results_path.name}")
                    print("  Use --overwrite to re-evaluate.")
                    sys.exit(1)

            # Step 1: Inference
            print("Step 1/2 — Running inference …\n")
            per_pair_inf = run_inference(comparator, pair_names, args.image_dir)

            # Step 2: Evaluate
            print(f"\nStep 2/2 — Evaluating scores (LLM judge: {use_llm}) …\n")
            per_pair_eval = evaluate_predictions(per_pair_inf, gt_records, use_llm)

            # Aggregate
            result = aggregate_results(per_pair_eval, run_key=run_key, run_label=run_label)
            result["run_name"]    = run_name
            result["prompt_file"] = str(prompt_path)
            result["model_name"]  = cfg.model_name
            result["temperature"] = args.temperature

            # Merge with existing JSON
            old_run_order:     list[str]       = []
            existing_runs_map: dict[str, dict] = {}
            if results_path.exists():
                _old = json.loads(results_path.read_text(encoding="utf-8"))
                for r in _old.get("results", []):
                    old_run_order.append(r["run"])
                    existing_runs_map[r["run"]] = r

            existing_runs_map[run_key] = result
            merged_run_names = list(old_run_order)
            if run_key not in merged_run_names:
                merged_run_names.append(run_key)

            merged_results = [existing_runs_map[r] for r in merged_run_names if r in existing_runs_map]

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
            print(f"\nResults JSON -> {results_path}  ({len(merged_results)} run(s) total)")

            # Summary table
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

    print("\nDone.")


if __name__ == "__main__":
    main()
