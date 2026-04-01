"""
DCA Optimization Grid Runner
=============================

Launches extract_dca.py sequentially for all 12 strategy combinations:

  MIPROv2 intensity  : light, medium                   (2)
  CoT mode           : CoT, NoCot                       (2)
  Model              : gpt-4.1, gpt-4.1-mini, gpt-5-mini  (3)
  ──────────────────────────────────────────────────────────
  Total              : 2 × 2 × 3 = 12 runs

Each run saves its own optimized model and log file.
Failed runs are skipped and reported at the end — the grid continues.

Usage:
    uv run python run_dca_grid.py

    # Dry-run: print commands without executing
    uv run python run_dca_grid.py --dry-run

    # Run only a subset (e.g. light intensity only)
    uv run python run_dca_grid.py --miprov2-auto light

    # Skip CoT or NoCot runs
    uv run python run_dca_grid.py --cot-only
    uv run python run_dca_grid.py --nocot-only
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from itertools import product

# ── Grid definition ──────────────────────────────────────────────

MODELS = [
    # (model_name,            reasoning_effort)
    # ("azure/gpt-4.1",         None),
    # ("azure/gpt-4.1-mini",    None),
    ("azure/gpt-5-mini",      "low"),   # reasoning model → pass reasoning_effort
]

INTENSITIES = ["heavy", 
            #    "medium"
               ]   # MIPROv2 --miprov2-auto

COT_MODES = [
    # (label,    no_cot_flag)
    # ("CoT",    False),
    ("NoCot",  True),
]

# ── CLI for the grid runner itself ───────────────────────────────
parser = argparse.ArgumentParser(description="Run DCA optimization grid (12 runs)")
parser.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing them")
parser.add_argument("--miprov2-auto", type=str, default=None,
                    choices=["light", "medium"],
                    help="Run only this intensity (default: both)")
parser.add_argument("--cot-only", action="store_true",
                    help="Run only CoT variants")
parser.add_argument("--nocot-only", action="store_true",
                    help="Run only NoCot variants")
args = parser.parse_args()

# Apply filters
intensities = [args.miprov2_auto] if args.miprov2_auto else INTENSITIES
cot_modes   = COT_MODES
if args.cot_only:
    cot_modes = [m for m in COT_MODES if not m[1]]   # keep only CoT
if args.nocot_only:
    cot_modes = [m for m in COT_MODES if m[1]]        # keep only NoCot

# ── Build run list ───────────────────────────────────────────────

runs = []
for intensity, (cot_label, no_cot), (model_name, reasoning_effort) in product(
    intensities, cot_modes, MODELS
):
    model_label = model_name.split("/")[-1]
    re_label    = f" RE={reasoning_effort}" if reasoning_effort else ""
    label       = f"MIPROv2-{intensity} | {cot_label} | {model_label}{re_label}"

    cmd = [
        "extract_dca.py",
        "--model-name", model_name,
        "--no-cache",               # always: optimizer needs fresh evaluations
        "--save-model",             # save each optimized prompt
        "--optimizer", "miprov2",
        "--miprov2-auto", intensity,
        "--skip-load",              # start fresh each run, don't load previous extractor
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

# ── Execute sequentially ─────────────────────────────────────────

results = []
grid_start = time.time()

for i, run in enumerate(runs, 1):
    label = run["label"]
    print(f"\n{'='*80}")
    print(f" Run {i}/{len(runs)}: {label}")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    # Use sys.executable (current interpreter) — already in the right venv,
    # no need to invoke uv which may not be on subprocess PATH on Windows.
    actual_cmd = [sys.executable] + run["cmd"]

    run_start = time.time()
    try:
        proc = subprocess.run(actual_cmd, check=False)
        elapsed = time.time() - run_start
        success = proc.returncode == 0
        status  = "OK" if success else f"FAILED (exit code {proc.returncode})"
    except Exception as e:
        elapsed = time.time() - run_start
        success = False
        status  = f"ERROR: {e}"

    results.append({"label": label, "success": success, "status": status, "elapsed": elapsed})
    print(f"\n  [{status}]  elapsed: {elapsed/60:.1f} min")

# ── Summary ──────────────────────────────────────────────────────

total_elapsed = time.time() - grid_start
n_ok   = sum(1 for r in results if r["success"])
n_fail = len(results) - n_ok

print(f"\n{'='*80}")
print(f" Grid complete — {n_ok}/{len(runs)} succeeded  |  total: {total_elapsed/60:.1f} min")
print(f"{'='*80}")
for r in results:
    icon = "OK  " if r["success"] else "FAIL"
    print(f"  {icon}  {r['elapsed']/60:5.1f} min  {r['label']}  [{r['status']}]")
print(f"{'='*80}\n")

if n_fail > 0:
    sys.exit(1)
