#!/usr/bin/env python3
"""Evaluate human-written DCA prompts with DIRECT Azure OpenAI calls (no DSPy).

This is the non-DSPy counterpart to run_human_prompt_repeatability.py.
It calls the Azure OpenAI API directly (same style as run_with_human_prompts.py)
and evaluates with the same cls / change / total metrics used by the optimized
prompt eval scripts.

Configurable grid
-----------------
Edit the RUNS dict to add/remove prompt x model combinations.

Usage examples
--------------
    # Evaluate all runs on val set (default):
    python evaluation/dca_run_val_test_set_eval/run_human_prompt_direct_eval.py

    # Specific runs only:
    python evaluation/dca_run_val_test_set_eval/run_human_prompt_direct_eval.py \\
        --runs v1_gpt41mini v2_gpt41mini --which-set val

    # 5 parallel repeats of two runs:
    python evaluation/dca_run_val_test_set_eval/run_human_prompt_direct_eval.py \\
        --runs v1_gpt41mini v2_gpt41mini --n-repeats 5 --which-set val

    # Skip LLM judge (faster, no judge API calls):
    python evaluation/dca_run_val_test_set_eval/run_human_prompt_direct_eval.py \\
        --skip-llm

Output layout
-------------
  eval_results/human_prompt_direct_runs/
  ├── eval_human_direct_<run_key>_val.json
  └── repeat_logs/<run_key>/
      ├── rep01.log
      └── ...

Table columns
-------------
  Prompt | Inf-Model | Cls | Chg | Tot | Time(s) | InTok | OutTok | RsnTok
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import certifi
import dotenv
import numpy as np
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI, DefaultHttpxClient
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

dotenv.load_dotenv(dotenv.find_dotenv(), override=True)

# ---------------------------------------------------------------------------
# Repo root & imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

def _find_repo_root() -> Path:
    here = SCRIPT_DIR
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return SCRIPT_DIR.parents[1]

REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.dca_metric import (
    FIELD_WEIGHT_CLS,
    FIELD_WEIGHT_CHANGE,
    _try_rule_based_match,
    judge_summary_metric,
    compute_classification_score,
    compute_total_score,
)
from evaluation.dca_run_val_test_set_eval.split_utils import resolve_default_split_file

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------
HUMAN_PROMPTS_DIR  = REPO_ROOT / "evaluation" / "dca_baseline_runs" / "human_prompts"
DEFAULT_IMAGE_DIR  = REPO_ROOT / "data" / "dca_data" / "image_pairs"
DEFAULT_GT_DIR     = REPO_ROOT / "data" / "dca_data" / "image_pairs_llm_output_exp1"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "eval_results" / "human_prompt_direct_runs"
DEFAULT_SPLIT_FILE = resolve_default_split_file(REPO_ROOT)
DEFAULT_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT")
DEFAULT_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
DEFAULT_TEMPERATURE = 0.1

# ---------------------------------------------------------------------------
# CONFIGURABLE GRID — edit this to add/remove runs
#
# Each entry: "run_name": RunConfig(
#     prompt_file      = filename inside HUMAN_PROMPTS_DIR (or absolute path)
#     model_name       = Azure deployment name
#     no_thinking_area = True if the prompt does NOT produce a "thinking_area" field
#     reasoning_effort = None | "low" | "medium" | "high"  (for o-series / gpt-5.x)
# )
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    prompt_file:      str
    model_name:       str
    no_thinking_area: bool          = True
    reasoning_effort: Optional[str] = None


RUNS: dict[str, RunConfig] = {
    # ── v5 prompt (midas) ────────────────────────────────────────────
    "v5_gpt41mini": RunConfig("v5_prompt_cr.md", "azure/gpt-4.1-mini", True, None),
    "v5_gpt41":     RunConfig("v5_prompt_cr.md", "azure/gpt-4.1", True, None),
    # "v5_gpt5mini":  RunConfig("v5_gepa_0.9_opt_cr.md", "azure/gpt-5-mini", True, "low"),
    # "v5_gpt52":     RunConfig("v5_gepa_0.9_opt_cr.md", "azure/gpt-5.2", True, "low"),

    # # ── v2 prompt (cr) ───────────────────────────────────────────────
    # "v2_gpt41mini": RunConfig("v2_prompt_cr.md",    "azure/gpt-4.1-mini"),
    # "v2_gpt41":     RunConfig("v2_prompt_cr.md",    "azure/gpt-4.1"),

    # ── v3 prompt (cr) ───────────────────────────────────────────────
    # "v3_gpt41mini": RunConfig("v3_prompt_cr.md",    "azure/gpt-4.1-mini"),
    # "v3_gpt41":     RunConfig("v3_prompt_cr.md",    "azure/gpt-4.1"),
}

# ---------------------------------------------------------------------------
# Azure client helpers  (ported from run_with_human_prompts.py)
# ---------------------------------------------------------------------------

def _build_ca_bundle() -> str:
    for var in ("AOAI_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        val = os.getenv(var)
        if val and Path(val).exists():
            return val
    return certifi.where()


def build_client(endpoint: str, api_version: str) -> AzureOpenAI:
    ca_bundle = _build_ca_bundle()
    api_key   = os.getenv("AZURE_OPENAI_API_KEY")
    if api_key:
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            http_client=DefaultHttpxClient(verify=ca_bundle),
        )
    credential      = AzureCliCredential()
    token_provider  = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
        http_client=DefaultHttpxClient(verify=ca_bundle),
    )


def _image_to_data_url(path: Path) -> str:
    ext_to_mime = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
        ".tif": "image/tiff", ".tiff": "image/tiff",
    }
    media_type = ext_to_mime.get(path.suffix.lower(), "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{media_type};base64,{b64}"


def _find_image(pair_dir: Path, stem: str) -> Optional[Path]:
    allowed = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
    matches = [p for p in sorted(pair_dir.glob(f"{stem}.*")) if p.suffix.lower() in allowed]
    return matches[0] if matches else None


def _normalize_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "1"}:  return True
        if v in {"false", "no", "0"}: return False
    if isinstance(value, int):
        return bool(value)
    return None


# ---------------------------------------------------------------------------
# Single-pair model call
# ---------------------------------------------------------------------------

class ModelCallFailure(Exception):
    pass


def call_model(
    client:           AzureOpenAI,
    deployment:       str,
    original_image:   Optional[Path],
    reimage:          Optional[Path],
    prompt_text:      str,
    temperature:      float = 0.1,
    max_retries:      int = 3,
    reasoning_effort: Optional[str] = None,
    no_thinking_area: bool = True,
) -> dict:
    """Call the model and return a result dict with parsed fields + token usage + timing."""
    user_content: list[dict] = []
    if original_image is not None:
        user_content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(original_image)}})
    if reimage is not None:
        user_content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(reimage)}})
    user_content.append({"type": "text", "text": prompt_text})

    messages = [
        {"role": "system", "content": "You are an expert assistant for comparing document image pairs."},
        {"role": "user",   "content": user_content},
    ]

    extra_kwargs: dict = {}
    if reasoning_effort is not None:
        extra_kwargs["reasoning_effort"] = reasoning_effort
    else:
        # reasoning models (o-series / gpt-5.x) do not accept temperature
        extra_kwargs["temperature"] = temperature

    _RATELIMIT_WAIT = 30.0
    last_exc: Optional[Exception] = None
    response = None
    t0 = time.time()

    for attempt in range(1, max_retries + 2):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=messages,
                response_format={"type": "json_object"},
                **extra_kwargs,
            )
            last_exc = None
            break
        except RateLimitError as exc:
            last_exc = exc
            wait = _RATELIMIT_WAIT
            try:
                hdr = exc.response.headers.get("Retry-After")
                if hdr:
                    wait = float(hdr)
            except Exception:
                pass
            print(f"    [attempt {attempt}] RateLimitError — waiting {wait:.0f}s …", flush=True)
            if attempt <= max_retries:
                time.sleep(wait)
        except (APITimeoutError, APIConnectionError) as exc:
            last_exc = exc
            wait = 2.0 ** attempt
            print(f"    [attempt {attempt}] {type(exc).__name__} — waiting {wait:.0f}s …", flush=True)
            if attempt <= max_retries:
                time.sleep(wait)
        except APIStatusError as exc:
            last_exc = exc
            if exc.status_code in (500, 502, 503, 504):
                wait = 2.0 ** attempt
                print(f"    [attempt {attempt}] APIStatusError {exc.status_code} — waiting {wait:.0f}s …", flush=True)
                if attempt <= max_retries:
                    time.sleep(wait)
            else:
                raise ModelCallFailure(f"APIStatusError {exc.status_code}: {exc}") from exc

    elapsed = time.time() - t0

    if response is None:
        raise ModelCallFailure(f"All {max_retries + 1} attempts failed. Last: {last_exc}")

    # ── Parse response ────────────────────────────────────────────────────────
    first_choice = response.choices[0] if response.choices else None
    raw = (first_choice.message.content or "").strip() if first_choice else ""

    pred_discard: Optional[bool] = None
    pred_changes = ""

    if raw:
        try:
            payload      = json.loads(raw)
            pred_discard = _normalize_bool(payload.get("discard_image"))
            pred_changes = str(payload.get("changes") or "").strip()
        except json.JSONDecodeError:
            pred_changes = raw  # store raw as changes for any rule-based matching

    # ── Token usage ───────────────────────────────────────────────────────────
    usage = getattr(response, "usage", None)
    input_tokens    = int(getattr(usage, "prompt_tokens",     0) or 0) or None
    output_tokens   = int(getattr(usage, "completion_tokens", 0) or 0) or None
    total_tokens    = int(getattr(usage, "total_tokens",      0) or 0) or None
    details         = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0) or None

    return {
        "pred_discard":     pred_discard,
        "pred_changes":     pred_changes,
        "response_time":    elapsed,
        "total_tokens":     total_tokens,
        "input_tokens":     input_tokens,
        "output_tokens":    output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "raw_response":     raw,
    }


# ---------------------------------------------------------------------------
# Azure pre-flight check
# ---------------------------------------------------------------------------

def _preflight_azure_check(n_pairs: int, avg_s: float = 20.0) -> None:
    if os.getenv("AZURE_OPENAI_API_KEY"):
        return
    try:
        from azure.identity import AzureCliCredential as _C
        token      = _C().get_token("https://cognitiveservices.azure.com/.default")
        expires_in = token.expires_on - time.time()
        estimated  = n_pairs * avg_s
        if expires_in < estimated + 300:
            print(f"  [WARN] Azure CLI token expires in {expires_in/60:.0f} min "
                  f"but run will take ~{estimated/60:.0f} min. Run 'az login' to refresh.")
        else:
            print(f"  Azure CLI token: valid for {expires_in/60:.0f} min "
                  f"(est. run ~{estimated/60:.0f} min) — OK")
    except Exception as exc:
        print(f"\n[ERROR] Azure CLI credentials are not valid: {exc}")
        print("  Please run 'az login' and retry.\n")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Per-set evaluation
# ---------------------------------------------------------------------------

def run_and_evaluate(
    cfg:         RunConfig,
    pair_names:  list[str],
    image_dir:   Path,
    gt_dir:      Path,
    client:      AzureOpenAI,
    deployment:  str,
    use_llm:     bool,
    prompt_text: str,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> list[dict]:
    """Run inference + evaluation for every pair. Returns per-pair dicts."""
    gt_records: dict[str, dict] = {}
    for pname in pair_names:
        f = gt_dir / f"{pname}.json"
        if f.exists():
            gt_records[pname] = json.loads(f.read_text(encoding="utf-8"))

    per_pair: list[dict] = []
    n = len(pair_names)

    for i, pair_name in enumerate(pair_names, 1):
        pair_dir = image_dir / pair_name
        original = _find_image(pair_dir, "original")
        reimage  = _find_image(pair_dir, "reimage")

        if original is None and reimage is None:
            print(f"  [{i:03d}/{n}] {pair_name}  [SKIP] no images found")
            per_pair.append({"pair_name": pair_name, "missing": True, "note": "no images"})
            continue

        # ── Inference ─────────────────────────────────────────────────────────
        try:
            res = call_model(
                client=client,
                deployment=deployment,
                original_image=original,
                reimage=reimage,
                prompt_text=prompt_text,
                temperature=temperature,
                max_retries=max_retries,
                reasoning_effort=cfg.reasoning_effort,
                no_thinking_area=cfg.no_thinking_area,
            )
        except ModelCallFailure as exc:
            print(f"  [{i:03d}/{n}] {pair_name}  [FAIL] {exc}")
            per_pair.append({"pair_name": pair_name, "missing": True, "note": str(exc)})
            continue

        # ── Scoring ───────────────────────────────────────────────────────────
        gt = gt_records.get(pair_name, {})
        human_discard = _normalize_bool(gt.get("human_label_discard_image"))
        human_changes = str(gt.get("human_label_changes") or "").strip()
        pred_discard  = res["pred_discard"]

        if human_discard is None or pred_discard is None:
            print(f"  [{i:03d}/{n}] {pair_name}  [SKIP] "
                  f"unparseable: human={gt.get('human_label_discard_image')!r}  "
                  f"pred={res['pred_discard']!r}")
            per_pair.append({
                "pair_name": pair_name, "missing": True,
                "note": f"unparseable discard",
                **{k: res.get(k) for k in ("response_time","total_tokens","input_tokens","output_tokens","reasoning_tokens")},
            })
            continue

        cls_score = compute_classification_score(human_discard, pred_discard)

        # Change score
        change_score, change_source = None, "skipped"
        rb_score, _reason, matched = _try_rule_based_match(human_changes, res["pred_changes"])
        if matched:
            change_score, change_source = rb_score, "rule"
        elif use_llm:
            try:
                change_score = judge_summary_metric(human_changes, res["pred_changes"])
                change_source = "llm"
            except Exception as exc:
                print(f"    [judge WARN] {pair_name}: {exc}")
                change_source = "llm_error"

        if change_score is not None:
            total_score = compute_total_score(cls_score, change_score)
        elif change_source in ("llm", "llm_error"):
            # LLM was called but returned None or raised — penalise change with 0
            # mirrors run_human_prompt_repeatability.py: elif change_source == "llm"
            total_score = compute_total_score(cls_score, 0.0)
        else:
            total_score = None

        tok_str = f"  tok={res['total_tokens']}" if res.get("total_tokens") else ""
        chg_str = f"{change_score:.3f}" if change_score is not None else "N/A"
        print(f"  [{i:03d}/{n}] {pair_name}  "
              f"cls={cls_score:.0f}  chg={chg_str}  src={change_source}  "
              f"t={res['response_time']:.1f}s{tok_str}", flush=True)

        per_pair.append({
            "pair_name":        pair_name,
            "missing":          False,
            "cls_score":        cls_score,
            "change_score":     change_score,
            "total_score":      total_score,
            "change_source":    change_source,
            "response_time":    res["response_time"],
            "total_tokens":     res.get("total_tokens"),
            "input_tokens":     res.get("input_tokens"),
            "output_tokens":    res.get("output_tokens"),
            "reasoning_tokens": res.get("reasoning_tokens"),
        })

    return per_pair


# ---------------------------------------------------------------------------
# Aggregate per-pair results
# ---------------------------------------------------------------------------

def aggregate_results(per_pair: list[dict], run_key: str, run_label: str) -> dict:
    valid = [p for p in per_pair if not p.get("missing")]

    cls_scores      = [p["cls_score"]       for p in valid]
    change_scores   = [p["change_score"]     for p in valid if p["change_score"]     is not None]
    total_scores    = [p["total_score"]      for p in valid if p["total_score"]      is not None]
    tokens          = [p["total_tokens"]     for p in valid if p.get("total_tokens") is not None]
    in_toks         = [p["input_tokens"]     for p in valid if p.get("input_tokens") is not None]
    out_toks        = [p["output_tokens"]    for p in valid if p.get("output_tokens") is not None]
    rsn_toks        = [p["reasoning_tokens"] for p in valid if p.get("reasoning_tokens") is not None]
    times           = [p["response_time"]    for p in valid if p.get("response_time") is not None]

    source_counts = {}
    for p in valid:
        src = p.get("change_source", "skipped")
        source_counts[src] = source_counts.get(src, 0) + 1

    return {
        "run":                  run_key,
        "label":                run_label,
        "n_test_pairs":         len(per_pair),
        "n_evaluated":          len(valid),
        "n_missing":            len(per_pair) - len(valid),
        "avg_cls_score":        float(np.mean(cls_scores))   if cls_scores    else None,
        "avg_change_score":     float(np.mean(change_scores)) if change_scores else None,
        "avg_total_score":      float(np.mean(total_scores))  if total_scores  else None,
        "avg_total_tokens":     float(np.mean(tokens))        if tokens        else None,
        "avg_input_tokens":     float(np.mean(in_toks))       if in_toks       else None,
        "avg_output_tokens":    float(np.mean(out_toks))      if out_toks      else None,
        "avg_reasoning_tokens": float(np.mean(rsn_toks))      if rsn_toks      else None,
        "avg_response_time":    float(np.mean(times))         if times         else None,
        "n_cls":                len(cls_scores),
        "n_change":             len(change_scores),
        "n_total_score":        len(total_scores),
        "change_source_counts": source_counts,
        "per_pair":             per_pair,
    }


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def print_table(results: list[dict], which_set: str, temperature: float) -> None:
    def f(v, fmt=".3f"):
        return f"{v:{fmt}}" if v is not None else "N/A"

    C = (22, 20, 7, 7, 7, 8, 7, 7, 7)
    header = (
        f"{'Prompt':<{C[0]}} {'Inf-Model':<{C[1]}} "
        f"{'Cls':>{C[2]}} {'Chg':>{C[3]}} {'Tot':>{C[4]}} "
        f"{'Time(s)':>{C[5]}} {'InTok':>{C[6]}} {'OutTok':>{C[7]}} {'RsnTok':>{C[8]}}"
    )
    sep = "-" * len(header)
    print()
    print(f"  which-set: {which_set}  |  temperature: {temperature}  |  direct API (no DSPy)")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        prompt_name = r.get("prompt_file_name", r.get("run_name", r["run"]))
        model_name  = r.get("model_name",        r["run"])
        print(
            f"  {prompt_name:<{C[0]-2}} {model_name:<{C[1]}} "
            f"{f(r.get('avg_cls_score')):>{C[2]}} "
            f"{f(r.get('avg_change_score')):>{C[3]}} "
            f"{f(r.get('avg_total_score')):>{C[4]}} "
            f"{f(r.get('avg_response_time'), '.1f'):>{C[5]}} "
            f"{f(r.get('avg_input_tokens'), '.0f'):>{C[6]}} "
            f"{f(r.get('avg_output_tokens'), '.0f'):>{C[7]}} "
            f"{str(r.get('avg_reasoning_tokens', 'N/A') if r.get('avg_reasoning_tokens') is not None else 'N/A'):>{C[8]}}"
        )
    print(sep)
    print(f"Total rows: {len(results)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Evaluate human prompts via direct Azure OpenAI call + latest eval metrics."
    )
    parser.add_argument("--runs", nargs="+", default=None,
                        help=f"Which runs (default: all). Choices: {list(RUNS.keys())}")
    parser.add_argument("--split-file",  type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--image-dir",   type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--gt-dir",      type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--output-dir",  type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--which-set",   choices=["val", "test", "both"], default="val")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--endpoint",    default=DEFAULT_ENDPOINT)
    parser.add_argument("--api-version", default=DEFAULT_API_VER)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--skip-llm",    action="store_true",
                        help="Skip LLM judge for change score (faster)")
    parser.add_argument("--overwrite",   action="store_true")
    parser.add_argument("--run-id",     type=str, default=None,
                        help="Repeat suffix e.g. 'rep01'. Set automatically by --n-repeats.")
    parser.add_argument("--n-repeats",  type=int, default=None,
                        help="Launch N parallel independent repeats per run.")
    args = parser.parse_args()

    selected_run_names: list[str] = args.runs if args.runs else list(RUNS.keys())
    unknown = [r for r in selected_run_names if r not in RUNS]
    if unknown:
        print(f"[ERROR] Unknown run name(s): {unknown}")
        print(f"  Available: {list(RUNS.keys())}")
        sys.exit(1)

    # ── n-repeats —————————————————————————————————————————————————————————————
    # For each run_name sequentially: launch all N repeats in parallel,
    # wait for all N, print combined output, then next run.
    if args.n_repeats is not None:
        if args.n_repeats < 1:
            print("[ERROR] --n-repeats must be >= 1")
            sys.exit(1)

        # Rebuild base argv without --n-repeats / --run-id / --runs
        skip_next = False
        skip_runs_vals = False
        base_argv: list[str] = []
        for tok in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if tok in ("--n-repeats", "--run-id"):
                skip_next = True
                continue
            if tok.startswith(("--n-repeats=", "--run-id=")):
                continue
            if tok == "--runs":
                skip_runs_vals = True
                continue
            if skip_runs_vals:
                if tok.startswith("--"):
                    skip_runs_vals = False
                else:
                    continue
            base_argv.append(tok)

        pad = len(str(args.n_repeats))
        temp_sfx = (f"_temp{args.temperature}"
                    if args.temperature != DEFAULT_TEMPERATURE
                    else str(args.temperature))

        def _run_repeat(rep_idx: int, cmd: list[str], log_path: Path) -> tuple:
            result = subprocess.run(cmd, capture_output=True, text=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                f"=== rep {rep_idx:0{pad}d} — exit_code={result.returncode} ===\n\n"
                f"--- STDOUT ---\n{result.stdout or '(empty)'}\n"
                + (f"\n--- STDERR ---\n{result.stderr}" if result.stderr.strip() else ""),
                encoding="utf-8",
            )
            return rep_idx, result.returncode, result.stdout, result.stderr

        overall_ok = overall_fail = 0

        for run_name in selected_run_names:
            logs_dir = args.output_dir / "repeat_logs" / f"{run_name}{temp_sfx}"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_paths = {i: logs_dir / f"rep{i:0{pad}d}.log" for i in range(1, args.n_repeats + 1)}
            cmds = [
                [sys.executable, __file__]
                + base_argv
                + ["--runs", run_name, "--run-id", f"rep{i:0{pad}d}"]
                for i in range(1, args.n_repeats + 1)
            ]

            print(f"\nLaunching {args.n_repeats} parallel repeats for: {run_name} ...")
            print(f"  Logs dir: {logs_dir}")

            with ThreadPoolExecutor(max_workers=args.n_repeats) as pool:
                futures = {
                    pool.submit(_run_repeat, i, cmd, log_paths[i]): i
                    for i, cmd in enumerate(cmds, 1)
                }
                results_list = []
                for fut in as_completed(futures):
                    rep_idx, rc, stdout, stderr = fut.result()
                    results_list.append((rep_idx, rc, stdout, stderr))
                    status = "OK" if rc == 0 else f"FAILED(exit={rc})"
                    print(f"  [rep {rep_idx:0{pad}d}] {status}  log -> {log_paths[rep_idx]}")

            results_list.sort(key=lambda x: x[0])
            print(f"\n{'#'*80}")
            print(f"FULL OUTPUT — {run_name}")
            print(f"{'#'*80}")
            for rep_idx, rc, stdout, stderr in results_list:
                print(f"\n{'='*80}")
                print(f"  {run_name}  rep {rep_idx:0{pad}d}  exit={rc}  log: {log_paths[rep_idx]}")
                print(f"{'='*80}")
                print(stdout or "(no stdout)")
                if stderr.strip():
                    print(f"--- STDERR ---\n{stderr.strip()}")

            n_ok   = sum(1 for _, rc, _, _ in results_list if rc == 0)
            n_fail = len(results_list) - n_ok
            overall_ok   += n_ok
            overall_fail += n_fail
            print(f"\n{run_name}: {args.n_repeats} repeats — {n_ok} OK, {n_fail} FAILED")

        total = len(selected_run_names) * args.n_repeats
        print(f"\n{'='*70}")
        print(f"ALL DONE — {total} total: {overall_ok} OK, {overall_fail} FAILED")
        print(f"{'='*70}")
        sys.exit(0 if overall_fail == 0 else 1)

    # ── Single evaluation pass ────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_llm = not args.skip_llm

    if not args.split_file.exists():
        print(f"[ERROR] Split file not found: {args.split_file}")
        sys.exit(1)
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    seed  = split.get("seed", "?")

    def _get_pair_names(key: str) -> list[str]:
        names = (split.get(key, {}).get("pair_names")
                 or split.get(f"{key}_pair_numbers") or [])
        if names and isinstance(names[0], int):
            names = [f"pair_{n:03d}" for n in names]
        return names

    for d, label in [(args.image_dir, "image-dir"), (args.gt_dir, "gt-dir")]:
        if not d.exists():
            print(f"[ERROR] {label} not found: {d}")
            sys.exit(1)

    sets_to_run = ["val", "test"] if args.which_set == "both" else [args.which_set]

    # Build ONE Azure client for all runs
    client = build_client(args.endpoint, args.api_version)

    # Collect summary rows for final table
    summary_rows: list[dict] = []

    for run_name in selected_run_names:
        cfg = RUNS[run_name]

        # Resolve prompt
        prompt_path = Path(cfg.prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = HUMAN_PROMPTS_DIR / cfg.prompt_file
        if not prompt_path.exists():
            print(f"[ERROR] Prompt file not found: {prompt_path}")
            sys.exit(1)
        prompt_text = prompt_path.read_text(encoding="utf-8").strip()

        # Deployment name: strip "azure/" prefix for raw API calls
        deployment = cfg.model_name.removeprefix("azure/")

        # run_key naming mirrors run_human_prompt_repeatability.py
        temp_sfx   = (f"_temp{args.temperature}"
                      if args.temperature != DEFAULT_TEMPERATURE
                      else str(args.temperature))
        run_key    = f"{run_name}{temp_sfx}"
        run_label  = f"{run_name} (direct/{cfg.model_name})"
        file_prefix = f"eval_human_direct_{run_key}"

        if args.run_id:
            run_key    = f"{run_key}_{args.run_id}"
            run_label  = f"{run_label} [{args.run_id}]"
            file_prefix = f"{file_prefix}_{args.run_id}"

        print(f"\n{'='*80}")
        print(f"Run          : {run_name}")
        print(f"Prompt file  : {prompt_path.name}  ({len(prompt_text)} chars)")
        print(f"Model        : {cfg.model_name}  temperature={args.temperature}")
        if cfg.reasoning_effort:
            print(f"Reason effort: {cfg.reasoning_effort}")
        print(f"Mode         : DIRECT Azure OpenAI (no DSPy)")
        print(f"{'='*80}")

        for set_name in sets_to_run:
            pair_names = _get_pair_names(set_name)
            total_pairs = len(pair_names)

            print(f"\nSet: {set_name.upper()}  ({total_pairs} pairs, seed={seed})")
            print(f"LLM judge: {'ENABLED' if use_llm else 'DISABLED (--skip-llm)'}\n")

            _preflight_azure_check(total_pairs)

            results_path = args.output_dir / f"{file_prefix}_{set_name}.json"

            # Guard against overwriting existing run
            if not args.overwrite and results_path.exists():
                _old = json.loads(results_path.read_text(encoding="utf-8"))
                if run_key in {r["run"] for r in _old.get("results", [])}:
                    print(f"[ERROR] Run '{run_key}' already exists in {results_path.name}")
                    print("  Use --overwrite to re-evaluate.")
                    sys.exit(1)

            # Run inference + evaluation
            per_pair = run_and_evaluate(
                cfg=cfg,
                pair_names=pair_names,
                image_dir=args.image_dir,
                gt_dir=args.gt_dir,
                client=client,
                deployment=deployment,
                use_llm=use_llm,
                prompt_text=prompt_text,
                temperature=args.temperature,
                max_retries=args.max_retries,
            )

            # Aggregate
            result = aggregate_results(per_pair, run_key=run_key, run_label=run_label)
            result["run_name"]         = run_name
            result["prompt_file_name"] = prompt_path.name
            result["model_name"]       = cfg.model_name
            result["temperature"]      = args.temperature
            result["reasoning_effort"] = cfg.reasoning_effort

            # Merge with existing JSON
            old_order: list[str]       = []
            existing:  dict[str, dict] = {}
            if results_path.exists():
                _old = json.loads(results_path.read_text(encoding="utf-8"))
                for r in _old.get("results", []):
                    old_order.append(r["run"])
                    existing[r["run"]] = r

            existing[run_key] = result
            merged_order = list(old_order)
            if run_key not in merged_order:
                merged_order.append(run_key)
            merged = [existing[r] for r in merged_order if r in existing]

            meta = {
                "split_file":          str(args.split_file),
                "which_set":           set_name,
                "n_pairs":             total_pairs,
                "skip_llm":            not use_llm,
                "field_weight_cls":    FIELD_WEIGHT_CLS,
                "field_weight_change": FIELD_WEIGHT_CHANGE,
                "runs":                merged_order,
            }
            results_path.write_text(
                json.dumps({"meta": meta, "results": merged}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"\nResults JSON -> {results_path}  ({len(merged)} run(s) total)")

            # Per-run summary line
            c  = f"{result['avg_cls_score']:.3f}"    if result.get("avg_cls_score")    is not None else "N/A"
            ch = f"{result['avg_change_score']:.3f}" if result.get("avg_change_score") is not None else "N/A"
            t  = f"{result['avg_total_score']:.3f}"  if result.get("avg_total_score")  is not None else "N/A"
            ti = f"{result['avg_response_time']:.1f}" if result.get("avg_response_time") is not None else "N/A"
            tk = f"{result['avg_input_tokens']:.0f}"  if result.get("avg_input_tokens")  is not None else "N/A"
            to = f"{result['avg_output_tokens']:.0f}" if result.get("avg_output_tokens") is not None else "N/A"
            tr = (f"{result['avg_reasoning_tokens']:.0f}"
                  if result.get("avg_reasoning_tokens") is not None else "N/A")
            print(f"  {run_key:<40} cls={c}  chg={ch}  tot={t}  "
                  f"time={ti}s  in={tk}  out={to}  rsn={tr}")

            if set_name == (sets_to_run[-1]):
                summary_rows.append(result)

    # ── Final summary table ────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("FINAL TABLE")
    print(f"{'='*90}")
    print_table(summary_rows, which_set=args.which_set, temperature=args.temperature)

    print("\nDone.")


if __name__ == "__main__":
    main()
