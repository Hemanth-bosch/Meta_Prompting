"""
DCA Optimization Grid Runner
=============================

Launches extract_dca.py with up to --parallel workers running simultaneously:

  MIPROv2 intensity  : light, medium                    (2)  [heavy excluded by default]
  Demo config        : (max_labeled=6, max_bootstrapped=6)
                       (max_labeled=3, max_bootstrapped=3) (2)
  Model              : gpt-4.1, gpt-4.1-mini             (2)
  CoT mode           : NoCot only                         (1)
  ──────────────────────────────────────────────────────────
  Total              : 2 × 2 × 2 × 1 = 8 runs  (default)

Starting instruction: human prompt from
  evaluation/dca_baseline_runs/human_prompts/v1_prompt_midas.md

Usage:
    uv run python run_dca_grid.py                     # 8 runs, 4 parallel
    uv run python run_dca_grid.py --parallel 2        # 2 at a time (rate-limit friendly)
    uv run python run_dca_grid.py --parallel 1        # fully sequential
    uv run python run_dca_grid.py --dry-run           # print plan only
    uv run python run_dca_grid.py --miprov2-auto light  # filter intensity
    uv run python run_dca_grid.py --include-heavy     # add heavy runs (12 total)
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from itertools import product

# ── Grid definition ──────────────────────────────────────────────

MODELS = [
    # (model_name,            reasoning_effort)
    ("azure/gpt-4.1-mini",    None),
    ("azure/gpt-4.1",         None),
    
    
]

INTENSITIES = [
    # "light", 
               "medium"
               ]   # MIPROv2 --miprov2-auto  (heavy excluded by default)

COT_MODES = [
    # (label,    no_cot_flag)
    ("NoCot",  True),
]

DEMO_CONFIGS = [
    # (max_labeled_demos, max_bootstrapped_demos)
    (6, 6),
    (3, 3),
]

# Human prompt used as the initial instruction for all runs
INITIAL_INSTRUCTION_FILE = "evaluation/dca_baseline_runs/human_prompts/v2_prompt_cr.md"

# ── CLI for the grid runner itself ───────────────────────────────
parser = argparse.ArgumentParser(description="Run DCA optimization grid (12 runs)")
parser.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing them")
parser.add_argument("--parallel", type=int, default=4,
                    help="Max number of runs to execute in parallel (default: 4)")
parser.add_argument("--miprov2-auto", type=str, default=None,
                    choices=["light", "medium", "heavy"],
                    help="Run only this intensity (default: light + medium)")
parser.add_argument("--include-heavy", action="store_true",
                    help="Include heavy intensity runs (excluded by default)")
parser.add_argument("--cot-only", action="store_true",
                    help="Run only CoT variants")
parser.add_argument("--nocot-only", action="store_true",
                    help="Run only NoCot variants")
args = parser.parse_args()

# Apply filters
if args.miprov2_auto:
    intensities = [args.miprov2_auto]
elif args.include_heavy:
    intensities = INTENSITIES + ["heavy"]
else:
    intensities = INTENSITIES
cot_modes   = COT_MODES
if args.cot_only:
    cot_modes = [m for m in COT_MODES if not m[1]]
if args.nocot_only:
    cot_modes = [m for m in COT_MODES if m[1]]

# ── Build run list ───────────────────────────────────────────────
# Pre-generate a unique timestamp for every run so that parallel workers
# never collide on log / saved-model filenames.  Each run gets base_time
# offset by its sequential index (1 s apart) which keeps the YYYYmmddHHMMSS
# format intact and is immediately human-readable.
_grid_base_time = datetime.now()

runs = []
for _run_idx, (intensity, (cot_label, no_cot), (model_name, reasoning_effort), (max_lab, max_boot)) in enumerate(
    product(intensities, cot_modes, MODELS, DEMO_CONFIGS)
):
    model_label    = model_name.split("/")[-1]
    re_label       = f" RE={reasoning_effort}" if reasoning_effort else ""
    label          = f"MIPROv2-{intensity} | {cot_label} | {model_label}{re_label} | d{max_lab}b{max_boot}"
    run_timestamp  = (_grid_base_time + timedelta(seconds=_run_idx)).strftime("%Y%m%d%H%M%S")

    cmd = [
        "extract_dca.py",
        "--model-name", model_name,
        "--no-cache",
        "--save-model",
        "--optimizer", "miprov2",
        "--miprov2-auto", intensity,
        "--skip-load",
        "--max-labeled-demos",      str(max_lab),
        "--max-bootstrapped-demos", str(max_boot),
        "--initial-instruction-file", INITIAL_INSTRUCTION_FILE,
        "--clear-field-descs",   # seed from instruction file only; no fixed field desc boilerplate
        "--run-timestamp", run_timestamp,  # unique id so parallel runs never share filenames
    ]
    if no_cot:
        cmd.append("--no-cot")
    if reasoning_effort is not None:
        cmd.extend(["--reasoning-effort", reasoning_effort])

    runs.append({"label": label, "cmd": cmd})

# ── Print plan ───────────────────────────────────────────────────

print(f"\n{'='*80}")
print(f" DCA Optimization Grid — {len(runs)} runs planned")
print(f"{'='*80}")
for i, run in enumerate(runs, 1):
    print(f"  {i:2d}. {run['label']}")
print(f"{'='*80}\n")

if args.dry_run:
    print("Dry-run mode — commands that would be executed:\n")
    for i, run in enumerate(runs, 1):
        print(f"  [{i:2d}] python {' '.join(run['cmd'])}")
    print()
    sys.exit(0)

# ── Execute with bounded parallelism ─────────────────────────────

grid_start = time.time()

def _run_one(idx: int, run: dict) -> dict:
    label = run["label"]
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[START {idx:2d}/{len(runs)}] {label}  ({started})")
    actual_cmd = [sys.executable] + run["cmd"]
    t0 = time.time()
    try:
        proc = subprocess.run(actual_cmd, check=False)
        elapsed = time.time() - t0
        success = proc.returncode == 0
        status  = "OK" if success else f"FAILED (exit {proc.returncode})"
    except Exception as e:
        elapsed = time.time() - t0
        success = False
        status  = f"ERROR: {e}"
    print(f"[DONE  {idx:2d}/{len(runs)}] {label}  [{status}]  {elapsed/60:.1f} min")
    return {"label": label, "success": success, "status": status, "elapsed": elapsed}

print(f"Executing {len(runs)} runs with max {args.parallel} in parallel …\n")
with ThreadPoolExecutor(max_workers=args.parallel) as pool:
    futures = {pool.submit(_run_one, i, run): i for i, run in enumerate(runs, 1)}
    results_list = []
    for fut in as_completed(futures):
        results_list.append(fut.result())

# Restore original order for the summary
results_list.sort(key=lambda r: next(i for i, run in enumerate(runs, 1) if run["label"] == r["label"]))

# ── Summary ──────────────────────────────────────────────────────

total_elapsed = time.time() - grid_start
n_ok   = sum(1 for r in results_list if r["success"])
n_fail = len(results_list) - n_ok

print(f"\n{'='*80}")
print(f" Grid complete — {n_ok}/{len(runs)} succeeded  |  total wall-clock: {total_elapsed/60:.1f} min")
print(f"{'='*80}")
for r in results_list:
    icon = "OK  " if r["success"] else "FAIL"
    print(f"  {icon}  {r['elapsed']/60:5.1f} min  {r['label']}  [{r['status']}]")
print(f"{'='*80}\n")

if n_fail > 0:
    sys.exit(1)
    sys.exit(1)
