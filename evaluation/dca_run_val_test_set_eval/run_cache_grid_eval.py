#!/usr/bin/env python3
"""Evaluate all new .cache/ model JSON files on val set and print a comparison table.

Finds every *.json in .cache/ dated >= CUTOFF_DATE (default: 2026-04-13),
runs each with:
    run_val_test_set_eval_on_optimized_prompt.py
        --model-name azure/gpt-4.1-mini
        --split-file .cache/dca_split_seed42.json
        --which-set val --no-cot --temperature 0.1 --overwrite

Then reads back each result JSON and renders a comparison table.

Usage
-----
    python run_cache_grid_eval.py                    # run all, batch-4
    python run_cache_grid_eval.py --dry-run          # show what would run
    python run_cache_grid_eval.py --batch-size 2     # 2 parallel at a time
    python run_cache_grid_eval.py --skip-eval        # skip inference, just build table from existing JSONs
    python run_cache_grid_eval.py --cutoff 2026-04-14  # only files from Apr 14+
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

def _find_repo_root() -> Path:
    """Walk up from this file until we find the directory containing pyproject.toml."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: two levels up (evaluation/dca_run_val_test_set_eval/ -> repo root)
    return here.parents[1]

REPO_ROOT   = _find_repo_root()
EVAL_SCRIPT = REPO_ROOT / "evaluation" / "dca_run_val_test_set_eval" / "run_val_test_set_eval_on_optimized_prompt.py"
CACHE_DIR   = REPO_ROOT / ".cache"
OUTPUT_DIR  = REPO_ROOT / "evaluation" / "dca_run_val_test_set_eval" / "eval_results" / "optimized_prompt_runs"
LOG_DIR     = REPO_ROOT / "log" / "cache_grid_eval"
SKIP_FILES  = {"dca_split_seed42.json", "judge_summary_cache.json"}

DEFAULT_CUTOFF       = "2026-04-13"
DEFAULT_INFERENCE_LM = "azure/gpt-4.1-mini"
DEFAULT_SPLIT_FILE   = ".cache/dca_split_seed42.json"
DEFAULT_TEMPERATURE  = 0.1
DEFAULT_BATCH_SIZE   = 4

# Regex patterns for filename parsing
RE_OPT_MODEL = re.compile(r'^(azure_gpt-[\d.]+(?:-mini)?)')
RE_DEMO      = re.compile(r'max_(\d+)_(\d+)')
RE_SEED      = re.compile(r'_(v\d+)_cfd')
RE_TS        = re.compile(r'(\d{14})$')
# Use .+? (non-greedy, handles spaces in Windows paths) followed by .json lookahead
RE_JSON_OUT  = re.compile(r'Results JSON\s+->\s+(.+?\.json)')

SEED_LABELS = {"v1": "v1_prompt_midas", "v2": "v2_prompt_cr"}


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def parse_model_filename(stem: str) -> dict:
    """Extract table metadata from a model JSON stem."""
    m_model    = RE_OPT_MODEL.search(stem)
    m_demo     = RE_DEMO.search(stem)
    m_seed     = RE_SEED.search(stem)
    m_ts       = RE_TS.search(stem)

    opt_model  = m_model.group(1) if m_model else stem
    demo_raw   = m_demo.group(0) if m_demo else "?"
    labeled    = m_demo.group(1) if m_demo else "?"
    bootstrap  = m_demo.group(2) if m_demo else "?"
    seed_key   = m_seed.group(1) if m_seed else None
    timestamp  = m_ts.group(1) if m_ts else stem

    # Strategy: text between optimizer model prefix and _max_
    if m_model and "_max_" in stem:
        strategy = stem[len(opt_model) + 1 : stem.find("_max_")]
    else:
        strategy = stem

    seed_label = SEED_LABELS.get(seed_key, "sig_only (no seed)")

    return {
        "stem":       stem,
        "opt_model":  opt_model.replace("azure_", ""),   # e.g. gpt-4.1
        "strategy":   strategy,                           # e.g. MIPROv2_DCA_Light_NoCot
        "optimizer":  _strategy_short(strategy),          # e.g. MIPROv2-Light
        "demo_tag":   demo_raw,                           # e.g. max_3_3
        "labeled":    labeled,
        "bootstrap":  bootstrap,
        "seed":       seed_label,
        "timestamp":  timestamp,
    }


def _strategy_short(strategy: str) -> str:
    s = strategy.lower()
    if "medium" in s:    return "MIPROv2-Medium"
    if "light" in s:     return "MIPROv2-Light"
    if "miprov2" in s:   return "MIPROv2"
    if "baseline" in s:  return "Bootstrap-Baseline"
    return strategy


# ---------------------------------------------------------------------------
# Log / result health audit
# ---------------------------------------------------------------------------

_RE_TRACEBACK = re.compile(r'\b(Traceback|Exception)\b')
_RE_STDERR_WARN = re.compile(r'\b(warning|warn|error)\b', re.IGNORECASE)
# STDERR lines that are known-harmless startup noise (LiteLLM config probing)
_RE_STDERR_NOISE = re.compile(r'^Key \w+ not found in .+\.env', re.IGNORECASE)
_EXPECTED_N = 82


def _audit_log(log_path: Optional[Path],
               n_test_pairs: Optional[int],
               n_missing: Optional[int]) -> str:
    """Return a compact health string for a single row.

    Checks:
      - n_test_pairs == 82
      - n_missing == 0
      - log exit code != 0  (exit=1 from "already exists / skip" is treated as OK)
      - STDERR lines after filtering known-harmless LiteLLM startup noise
      - Traceback / Exception in the STDOUT section
    """
    issues: list[str] = []

    if n_test_pairs is not None and n_test_pairs != _EXPECTED_N:
        issues.append(f"n={n_test_pairs}\u226082")
    if n_missing is not None and n_missing > 0:
        issues.append(f"miss={n_missing}")

    lp = Path(log_path) if log_path else None
    if lp is None or not lp.exists():
        issues.append("no-log")
    else:
        try:
            text = lp.read_text(encoding="utf-8", errors="replace")

            # --- Split into STDOUT / STDERR sections ----------------------------
            stdout_part = ""
            stderr_part = ""
            if "STDERR\n" in text:
                before_stderr, _, after_stderr = text.partition("STDERR\n")
                before_stderr = before_stderr.rsplit("=" * 10, 1)[0]
                stdout_part = before_stderr
                stderr_part = after_stderr
            else:
                stdout_part = text

            # --- Check exit code — but ignore benign "already exists" skips ----
            exit_code = "0"
            for line in text.splitlines()[:5]:
                if line.startswith("EXIT:"):
                    exit_code = line.split(":", 1)[1].strip()
                    break
            already_exists = "already exists" in stdout_part
            if exit_code != "0" and not already_exists:
                issues.append(f"exit={exit_code}")

            # --- STDOUT: only flag genuine Python crashes -----------------------
            crash_line: Optional[str] = None
            for raw in stdout_part.splitlines():
                if _RE_TRACEBACK.search(raw):
                    crash_line = raw.strip()[:80]
                    break
            if crash_line:
                issues.append(f"ERR: {crash_line}")

            # --- STDERR: skip known LiteLLM .env noise, flag the rest ----------
            real_stderr = [
                l.strip() for l in stderr_part.splitlines()
                if l.strip() and not l.startswith("=") and not _RE_STDERR_NOISE.match(l.strip())
            ]
            if real_stderr:
                first_sig = next(
                    (l for l in real_stderr if _RE_STDERR_WARN.search(l)),
                    real_stderr[0]
                )
                issues.append(f"STDERR: {first_sig[:80]}")

        except Exception as exc:
            issues.append(f"log-read-err: {exc}")

    return " | ".join(issues) if issues else "OK"


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_model_files(cutoff: datetime) -> list[Path]:
    files = []
    for f in sorted(CACHE_DIR.glob("*.json")):
        if f.name in SKIP_FILES:
            continue
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
        if mtime >= cutoff:
            files.append(f)
    return files


# ---------------------------------------------------------------------------
# Run one evaluation subprocess
# ---------------------------------------------------------------------------

def run_eval(model_path: Path, dry_run: bool,
             status_dict: Optional[dict] = None) -> dict:
    """Run the eval script for one model file. Returns result dict."""
    meta = parse_model_filename(model_path.stem)

    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--model-path",  str(model_path),
        "--model-name",  DEFAULT_INFERENCE_LM,
        "--split-file",  DEFAULT_SPLIT_FILE,
        "--which-set",   "val",
        "--no-cot",
        "--temperature", str(DEFAULT_TEMPERATURE),
        # "--overwrite",
    ]

    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd[-8:])}")
        return {**meta, "status": "dry-run", "result_path": None,
                "cls": None, "chg": None, "tot": None,
                "time": None, "inp": None, "out": None, "rsn": None}

    t0 = time.time()
    if status_dict is not None:
        status_dict[model_path.name] = {"start": t0, "done": False, "status": "running"}

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))

    elapsed = time.time() - t0

    # --- Write full output to a per-run log file --------------------------------
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{model_path.stem}.log"
    log_file.write_text(
        f"CMD: {' '.join(cmd)}\n"
        f"EXIT: {result.returncode}\n"
        f"ELAPSED: {elapsed:.1f}s\n"
        f"{'='*80}\nSTDOUT\n{'='*80}\n{result.stdout}\n"
        f"{'='*80}\nSTDERR\n{'='*80}\n{result.stderr}\n",
        encoding="utf-8",
    )

    if status_dict is not None:
        status_dict[model_path.name].update({"done": True, "elapsed": elapsed, "log": log_file})

    status = "OK" if result.returncode == 0 else f"FAILED(exit={result.returncode})"
    if result.returncode == 0 and result.stderr.strip():
        status = "OK+WARN"

    if result.returncode != 0:
        # Show last 30 lines of combined output inline so you see the error immediately
        err_lines = [l for l in (result.stderr + result.stdout).splitlines() if l.strip()][-30:]
        print(f"\n  [FAIL] {model_path.name}  exit={result.returncode}  elapsed={elapsed:.1f}s")
        print(f"  Full log -> {log_file}")
        print(f"  --- last 30 lines ---")
        for el in err_lines:
            print(f"    {el}")
        print(f"  --- end of tail ---")
        sys.stdout.flush()
    elif result.stderr.strip():
        # Exit=0 but stderr has content (warnings, retries, deprecation notices, etc.)
        stderr_lines = [l for l in result.stderr.splitlines() if l.strip()]
        n_warn = len(stderr_lines)
        preview = stderr_lines[:10]
        print(f"\n  [WARN] {model_path.name}  exit=0 but {n_warn} stderr line(s)  elapsed={elapsed:.1f}s")
        for wl in preview:
            print(f"    {wl}")
        if n_warn > 10:
            print(f"    ... ({n_warn - 10} more lines in log)")
        print(f"  Full log -> {log_file}")
        sys.stdout.flush()

    # --- Find the result JSON path from stdout ----------------------------------
    # The eval script prints:  Results JSON  -> <path>  (N run(s) total)
    # Path may contain spaces (Windows OneDrive paths), so use non-greedy .+?
    result_path: Optional[Path] = None
    m = RE_JSON_OUT.search(result.stdout)
    if m:
        result_path = Path(m.group(1).strip())

    # --- Read metrics from the result JSON (most reliable source) ---------------
    cls_ = chg = tot = time_ = inp = out = rsn = None
    if result_path and result_path.exists():
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
            # The run entry for THIS eval is the most recently written one.
            # active_run_key is what the eval script appended, but we don't know
            # it here — just take the last entry (it was just written / overwritten).
            matched = None
            for r in reversed(data.get("results", [])):
                if r.get("avg_cls_score") is not None:
                    matched = r
                    break
            if matched:
                cls_  = matched["avg_cls_score"]
                chg   = matched.get("avg_change_score")
                tot   = matched.get("avg_total_score")
                time_ = matched.get("avg_response_time")
                inp   = matched.get("avg_input_tokens")
                out   = matched.get("avg_output_tokens")
                rsn   = matched.get("avg_reasoning_tokens")
        except Exception as exc:
            print(f"  [WARN] Could not read result JSON for {model_path.name}: {exc}")
    elif result.returncode == 0:
        print(f"  [WARN] Could not locate result JSON for {model_path.name}")
        print(f"    Searched stdout for 'Results JSON  ->' — no match found.")
        print(f"    Check full log: {log_file}")

    return {
        **meta,
        "status":      status,
        "result_path": result_path,
        "cls":  round(cls_,  3) if cls_  is not None else None,
        "chg":  round(chg,   3) if chg   is not None else None,
        "tot":  round(tot,   3) if tot   is not None else None,
        "time": round(time_, 1) if time_ is not None else None,
        "inp":  round(inp,   0) if inp   is not None else None,
        "out":  round(out,   0) if out   is not None else None,
        "rsn":  rsn,
    }


# ---------------------------------------------------------------------------
# Table print
# ---------------------------------------------------------------------------

def print_table(rows: list[dict]) -> None:
    def f(v, fmt=".3f"):
        return f"{v:{fmt}}" if v is not None else "N/A"

    # Sort: strategy, opt_model, demo_tag, seed
    rows_s = sorted(rows, key=lambda r: (r["strategy"], r["opt_model"], r["demo_tag"], r["seed"]))

    has_health = any("health" in r for r in rows_s)

    C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14 = (
        20, 10, 18, 20, 22, 7, 7, 7, 7, 7, 7, 7, 46, 55
    )
    header = (
        f"{'Optimizer':<{C1}} "
        f"{'HParams':<{C2}} "
        f"{'Opt-Model':<{C3}} "
        f"{'Inf-Model':<{C4}} "
        f"{'Seed Prompt':<{C5}} "
        f"{'Cls':>{C6}} {'Chg':>{C7}} {'Tot':>{C8}} "
        f"{'Time':>{C9}} {'In':>{C10}} {'Out':>{C11}} {'Rsn':>{C12}}"
    )
    if has_health:
        header += f"  {'Health':<{C13}}"
    header += f"  {'Cache File':<{C14}}"
    sep = "-" * len(header)
    print()
    print(f"Inference model : {DEFAULT_INFERENCE_LM}  |  which-set: val  |  temperature: {DEFAULT_TEMPERATURE}  |  no-cot")
    print(sep)
    print(header)
    print(sep)
    for r in rows_s:
        line = (
            f"  {r['optimizer']:<{C1-2}} "
            f"{r['demo_tag']:<{C2}} "
            f"{r['opt_model']:<{C3}} "
            f"{DEFAULT_INFERENCE_LM:<{C4}} "
            f"{r['seed']:<{C5}} "
            f"{f(r.get('cls')):>{C6}} "
            f"{f(r.get('chg')):>{C7}} "
            f"{f(r.get('tot')):>{C8}} "
            f"{f(r.get('time'),'.1f'):>{C9}} "
            f"{f(r.get('inp'),'.0f'):>{C10}} "
            f"{f(r.get('out'),'.0f'):>{C11}} "
            f"{str(r.get('rsn','N/A')):>{C12}}"
        )
        if has_health:
            raw_h = r.get("health", "")
            cell  = (raw_h[:C13-1] + "\u2026") if len(raw_h) > C13 else raw_h
            line += f"  {cell:<{C13}}"
        stem_val = r.get("stem", "")
        line += f"  {stem_val}"
        print(line)
    print(sep)
    print(f"Total rows: {len(rows_s)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Ensure UTF-8 output on Windows terminals (avoids UnicodeEncodeError for special chars)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Batch-eval all new .cache/ model files.")
    parser.add_argument("--cutoff",     default=DEFAULT_CUTOFF,
                        help=f"Only files on or after this date (YYYY-MM-DD, default: {DEFAULT_CUTOFF})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Number of parallel evals per batch (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print commands without running them")
    parser.add_argument("--skip-eval",  action="store_true",
                        help="Skip running evals; just read existing result JSONs and build table")
    args = parser.parse_args()

    cutoff = datetime.strptime(args.cutoff, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    model_files = find_model_files(cutoff)

    print(f"Found {len(model_files)} model file(s) in .cache/ dated >= {args.cutoff}")
    for f in model_files:
        meta = parse_model_filename(f.stem)
        print(f"  {f.name}")
        print(f"    -> optimizer={meta['optimizer']}  opt_model={meta['opt_model']}  "
              f"demo={meta['demo_tag']}  seed={meta['seed']}")

    if args.dry_run:
        print(f"\n--- DRY RUN ({len(model_files)} files, batch-size={args.batch_size}) ---")
        for mf in model_files:
            run_eval(mf, dry_run=True)
        print("\n(No evals were run)")
        return

    if args.skip_eval:
        print("\n--skip-eval: reading existing result JSONs …")
        rows = []
        for mf in model_files:
            meta = parse_model_filename(mf.stem)
            log_path = LOG_DIR / f"{mf.stem}.log"
            # Try to find any result JSON matching this file
            pattern = f"*{meta['timestamp']}*val.json"
            matches = list(OUTPUT_DIR.glob(pattern))
            if not matches:
                print(f"  [MISS] No result JSON for {mf.name}")
                health = _audit_log(log_path, None, None)
                rows.append({**meta, "status": "missing", "health": health,
                              "cls": None, "chg": None, "tot": None,
                              "time": None, "inp": None, "out": None, "rsn": None})
                continue
            # Use the most recent match
            result_path = sorted(matches)[-1]
            try:
                data = json.loads(result_path.read_text(encoding="utf-8"))
                r = next((x for x in data.get("results", []) if x.get("avg_cls_score") is not None), None)
                if r:
                    n_test = r.get("n_test_pairs")
                    n_miss = r.get("n_missing")
                    health = _audit_log(log_path, n_test, n_miss)
                    rows.append({**meta, "status": "OK", "result_path": result_path, "health": health,
                                 "cls":  round(r["avg_cls_score"], 3),
                                 "chg":  round(r["avg_change_score"], 3) if r.get("avg_change_score") is not None else None,
                                 "tot":  round(r["avg_total_score"], 3) if r.get("avg_total_score") is not None else None,
                                 "time": round(r["avg_response_time"], 1) if r.get("avg_response_time") is not None else None,
                                 "inp":  round(r["avg_input_tokens"], 0) if r.get("avg_input_tokens") is not None else None,
                                 "out":  round(r["avg_output_tokens"], 0) if r.get("avg_output_tokens") is not None else None,
                                 "rsn":  r.get("avg_reasoning_tokens")})
                else:
                    health = _audit_log(log_path, None, None)
                    rows.append({**meta, "status": "empty", "health": health,
                                 "cls": None, "chg": None, "tot": None,
                                 "time": None, "inp": None, "out": None, "rsn": None})
            except Exception as e:
                print(f"  [ERR] {result_path.name}: {e}")
                health = _audit_log(log_path, None, None)
                rows.append({**meta, "status": "error", "health": health,
                             "cls": None, "chg": None, "tot": None,
                             "time": None, "inp": None, "out": None, "rsn": None})
        print_table(rows)
        # Detail block for any non-OK row
        non_ok = [r for r in rows if r.get("health", "OK") != "OK"]
        if non_ok:
            print(f"\n{'='*70}")
            print(f"HEALTH DETAILS  ({len(non_ok)} row(s) with issues)")
            print(f"{'='*70}")
            for r in non_ok:
                print(f"  {r['stem']}")
                print(f"    Health : {r['health']}")
                lp = LOG_DIR / f"{r['stem']}.log"
                if lp.exists():
                    print(f"    Log    : {lp}")
                print()
        else:
            print(f"\nAll {len(rows)} rows: OK (82 samples, 0 missing, no log warnings/errors)")
        return

    # ── Run in batches ────────────────────────────────────────────────────────
    rows: list[dict] = []
    total = len(model_files)
    batch_sz = args.batch_size

    for batch_start in range(0, total, batch_sz):
        batch = model_files[batch_start : batch_start + batch_sz]
        bn    = batch_start // batch_sz + 1
        nb    = (total + batch_sz - 1) // batch_sz
        print(f"\n{'='*70}")
        print(f"Batch {bn}/{nb}  ({len(batch)} runs in parallel)")
        print(f"{'='*70}")
        for mf in batch:
            print(f"  {mf.name}")

        batch_results: list[dict] = []
        status_dict: dict = {}
        stop_hb = threading.Event()

        def _heartbeat() -> None:
            """Print a status line every 15 s while the batch is running."""
            while not stop_hb.wait(15):
                still = [n for n, s in status_dict.items() if not s["done"]]
                if not still:
                    continue
                elapsed_strs = [
                    f"  [RUNNING] {n[-58:]}  ({time.time()-status_dict[n]['start']:.0f}s)"
                    for n in still
                ]
                print(f"\n  --- heartbeat: {len(still)}/{len(batch)} still running ---")
                for line in elapsed_strs:
                    print(line)
                sys.stdout.flush()

        hb_thread = threading.Thread(target=_heartbeat, daemon=True)
        hb_thread.start()

        with ThreadPoolExecutor(max_workers=batch_sz) as pool:
            future_map = {pool.submit(run_eval, mf, False, status_dict): mf for mf in batch}
            for fut in as_completed(future_map):
                mf  = future_map[fut]
                res = fut.result()
                batch_results.append(res)
                run_status = res["status"]
                elapsed_s  = status_dict.get(mf.name, {}).get("elapsed", 0.0)
                log_path   = status_dict.get(mf.name, {}).get("log", "?")
                c = f"{res['cls']:.3f}" if res.get("cls") is not None else "N/A"
                t = f"{res['tot']:.3f}" if res.get("tot") is not None else "N/A"
                print(f"  [DONE] {mf.name[-50:]}  cls={c}  tot={t}  "
                      f"status={run_status}  elapsed={elapsed_s:.1f}s")
                print(f"         log -> {log_path}")
                sys.stdout.flush()

        stop_hb.set()
        hb_thread.join()

        rows.extend(batch_results)

        # Print incremental table after each batch
        print(f"\n--- Incremental table after batch {bn}/{nb} ---")
        print_table([r for r in rows if r.get("cls") is not None])

    # Final full table
    print(f"\n{'='*70}")
    print("FINAL TABLE — ALL RUNS")
    print(f"{'='*70}")
    print_table(rows)

    # Count failures
    failed = [r for r in rows if "FAILED" in str(r.get("status", ""))]
    if failed:
        print(f"\n[WARN] {len(failed)} run(s) failed:")
        for r in failed:
            print(f"  {r['stem']}")


if __name__ == "__main__":
    main()
