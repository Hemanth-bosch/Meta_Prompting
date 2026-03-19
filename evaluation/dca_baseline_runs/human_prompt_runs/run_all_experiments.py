"""
Run all LLM experiment commands sequentially, logging stdout+stderr for each run.

Log file: run_all_experiments.log  (next to this script)
Usage:
    python evaluation/dca_baseline_runs/human_prompt_runs/run_all_experiments.py
"""

import subprocess
import sys
import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PYTHON = sys.executable
LOG_PATH = REPO_ROOT / "run_all_experiments.log"

print(PYTHON)

RUNS = [
    # {
    #     "label": "v1_prompt_midas_gpt52_reason_low",
    #     "args": [
    #         PYTHON, "evaluation/dca_baseline_runs/human_prompt_runs/run_with_human_prompts.py",
    #         "--key-prefix", "v1_prompt_midas_gpt52_reason_low",
    #         "--deployment", "gpt-5.2",
    #         "--model-name", "gpt-5.2",
    #         "--verbose", "--save-full-response",
    #         "--prompt", "v1_prompt_midas.md",
    #         "--no-thinking-area", "--reasoning-effort", "low",
    #     ],
    # },
    # {
    #     "label": "v2_prompt_cr_gpt52_reason_low",
    #     "args": [
    #         PYTHON, "evaluation/dca_baseline_runs/human_prompt_runs/run_with_human_prompts.py",
    #         "--key-prefix", "v2_prompt_cr_gpt52_reason_low",
    #         "--deployment", "gpt-5.2",
    #         "--model-name", "gpt-5.2",
    #         "--verbose", "--save-full-response",
    #         "--prompt", "v2_prompt_cr.md",
    #         "--no-thinking-area", "--reasoning-effort", "low",
    #     ],
    # },
    {
        "label": "v3_prompt_cr_gpt52_reason_low",
        "args": [
            PYTHON, "evaluation/dca_baseline_runs/human_prompt_runs/run_with_human_prompts.py",
            "--key-prefix", "v3_prompt_cr_gpt52_reason_low",
            "--deployment", "gpt-5.2",
            "--model-name", "gpt-5.2",
            "--verbose", "--save-full-response",
            "--prompt", "v3_prompt_cr.md",
            "--reasoning-effort", "low",
        ],
    },
]


def ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_all():
    log_file = LOG_PATH.open("w", encoding="utf-8")

    def log(text: str):
        print(text, flush=True)
        log_file.write(text + "\n")
        log_file.flush()

    log(f"{'='*70}")
    log(f"run_all_experiments.py  started at {ts()}")
    log(f"Total runs: {len(RUNS)}")
    log(f"Log file  : {LOG_PATH}")
    log(f"{'='*70}\n")

    overall_ok = True

    for i, run in enumerate(RUNS, start=1):
        label = run["label"]
        log(f"\n{'─'*70}")
        log(f"[{i}/{len(RUNS)}]  {label}   started at {ts()}")
        log(f"{'─'*70}")

        proc = subprocess.Popen(
            run["args"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(line, flush=True)
            log_file.write(line + "\n")
            log_file.flush()
        proc.wait()
        result_returncode = proc.returncode

        status = "OK" if result_returncode == 0 else f"FAILED (exit {result_returncode})"
        log(f"\n[{i}/{len(RUNS)}]  {label}   finished at {ts()}  →  {status}")

        if result_returncode != 0:
            overall_ok = False
            log(f"  *** Non-zero exit code — check output above ***")

    log(f"\n{'='*70}")
    log(f"All runs completed at {ts()}.  Overall: {'SUCCESS' if overall_ok else 'SOME FAILURES'}")
    log(f"{'='*70}")

    log_file.close()
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(run_all())
