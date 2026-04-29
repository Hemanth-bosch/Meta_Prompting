#!/usr/bin/env python3
"""Run all optimized-prompt + human-prompt evaluations sequentially.

Launches each command one after the other, streams output live to the
terminal AND writes every line to a timestamped log file.

Log location:
    log/human_and_optimized_eval/run_<YYYYMMDD_HHMMSS>.log
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Commands to run — one per entry.  Fix the "ython" typo in the originals.
# ---------------------------------------------------------------------------
PYTHON = sys.executable

COMMANDS: list[tuple[str, list[str]]] = [
    # (
    #     "gpt-4.1-mini  MIPROv2-Light  v1",
    #     [
    #         PYTHON,
    #         "evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py",
    #         "--model-path",  ".cache/gpt-4.1-mini_MIPROv2_DCA_Lig_NoCot_max_3_3_v1_cfd_20260413191910.json",
    #         "--model-name",  "azure/gpt-4.1-mini",
    #         "--split-file",  ".cache/dca_split_seed42.json",
    #         "--which-set",   "val",
    #         "--no-cot",
    #         "--temperature", "0.1",
    #         "--n-repeats",   "10",
    #         "--overwrite",
    #     ],
    # ),
    (
        "gpt-5-mini  MIPROv2-Light  v2",
        [
            PYTHON,
            "evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py",
            "--model-path",  ".cache/azure_gpt-4.1-mini_MIPROv2_DCA_Light_NoCot_dca_optimized_max_6_6_20260402141458.json",
            "--model-name",  "azure/gpt-5-mini",
            "--split-file",  ".cache/dca_split_seed42.json",
            "--which-set",   "test",
            "--no-cot",
            "--temperature", "1",
            "--reasoning-effort", "low",
        ],
    ),
    (
        "gpt-5.2  MIPROv2-Light  v2",
        [
            PYTHON,
            "evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py",
            "--model-path",  ".cache/azure_gpt-4.1-mini_MIPROv2_DCA_Light_NoCot_dca_optimized_max_6_6_20260402141458.json",
            "--model-name",  "azure/gpt-5.2",
            "--split-file",  ".cache/dca_split_seed42.json",
            "--which-set",   "test",
            "--no-cot",
            "--temperature", "1",
            "--reasoning-effort", "none",
        ],
    ),
    (
        "gpt-4.1-mini  MIPROv2-Light  v2",
        [
            PYTHON,
            "evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py",
            "--model-path",  ".cache/azure_gpt-4.1-mini_MIPROv2_DCA_Light_NoCot_dca_optimized_max_6_6_20260402141458.json",
            "--model-name",  "azure/gpt-4.1-mini",
            "--split-file",  ".cache/dca_split_seed42.json",
            "--which-set",   "test",
            "--no-cot",
            "--temperature", "0.1",
            # "--reasoning-effort", "low",
            "--overwrite",
        ],
    ),

    # (
    #     "human-prompts  direct  all runs  10 repeats",
    #     [
    #         PYTHON,
    #         "evaluation/dca_run_val_test_set_eval/run_human_prompt_direct_eval.py",
    #         "--which-set",  "val",
    #         "--n-repeats",  "10",
    #     ],
    # ),
]

# ---------------------------------------------------------------------------
# Log setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR   = REPO_ROOT / "log" / "human_and_optimized_eval"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE  = LOG_DIR / f"run_{TIMESTAMP}.log"


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(text: str, fh) -> None:
    print(text, flush=True)
    fh.write(text + "\n")
    fh.flush()


def _run_one(label: str, cmd: list[str], fh) -> int:
    """Stream one subprocess to stdout + log file, return exit code."""
    _log(f"\n{'='*90}", fh)
    _log(f"  START  [{label}]  at {_ts()}", fh)
    _log(f"  CMD: {' '.join(cmd)}", fh)
    _log(f"{'='*90}", fh)

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,      # merge stderr into stdout
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO_ROOT),
    )

    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    proc.wait()
    elapsed = time.time() - t0
    status  = "OK" if proc.returncode == 0 else f"FAILED (exit={proc.returncode})"
    _log(f"\n  END    [{label}]  {status}  elapsed={elapsed:.1f}s  at {_ts()}", fh)
    _log(f"{'='*90}\n", fh)
    return proc.returncode


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    with open(LOG_FILE, "w", encoding="utf-8") as fh:
        _log(f"Log file : {LOG_FILE}", fh)
        _log(f"Started  : {_ts()}", fh)
        _log(f"Commands : {len(COMMANDS)}", fh)

        overall_ok = overall_fail = 0
        results: list[tuple[str, int, float]] = []

        for label, cmd in COMMANDS:
            t0 = time.time()
            rc = _run_one(label, cmd, fh)
            elapsed = time.time() - t0
            results.append((label, rc, elapsed))
            if rc == 0:
                overall_ok += 1
            else:
                overall_fail += 1

        # ── Final summary ──────────────────────────────────────────────────
        _log(f"\n{'#'*90}", fh)
        _log(f"  SUMMARY — {len(COMMANDS)} commands  |  {overall_ok} OK  |  {overall_fail} FAILED", fh)
        _log(f"{'#'*90}", fh)
        for label, rc, elapsed in results:
            status = "OK" if rc == 0 else f"FAILED(exit={rc})"
            _log(f"  [{status:>14}]  {elapsed:6.0f}s  {label}", fh)
        _log(f"\nLog saved -> {LOG_FILE}", fh)
        _log(f"Finished  : {_ts()}", fh)

    return 0 if overall_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
