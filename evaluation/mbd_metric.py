"""Reference-free evaluation metric for MBD test case generation.

DESIGN PHILOSOPHY
-----------------
This metric requires NO ground truth test cases. All scoring is computed from:
  - The equation graph (equations.json)
  - The signal sheet (signalsheet.csv / model_interface.csv)
  - The DLL (optional, for functional verification)

METRIC COMPONENTS                              Evaluated on
-----------------                              ----------------------------------
  M1: I/O Classification    (15%) — Rule-based     Prompt 2 raw output
  M2: Signal Resolution     (15%) — String lookup  Final pipeline output
  M3: Structural Validity   (10%) — JSON / regex   Prompt 2 raw output
  M4: Test Coverage         (20%) — KB LLM-judge   Prompt 2 raw output
  M6: Boundary Values        (0%) — DISABLED       Prompt 2 raw output
  M7: DLL Pass Rate         (20%) — DLL inline 1/N Final pipeline output
  M8: Functional Correctness(20%) — DLL sequential Final pipeline output

  Composite: Prompt-2 side (M1+M3+M4+M6 = 15+10+20+0 = 45%) vs
  DLL/final side (M2+M7+M8 = 15+20+20 = 55%). Sum = 100%.

  GEPA group dispatch — M2 + M8 are end-of-chain outcomes, so their feedback is shown
  to every stage predictor (the whole sequence influences them). IMPORTANT: dspy GEPA
  requires the returned score to EQUAL the module-level composite (gepa_utils.py asserts
  fb.score == module_score), so EVERY predictor receives the SAME composite score; only the
  per-predictor FEEDBACK TEXT differs (see mbd_metric_with_feedback). The text emphasises:
    testcase_generator    M1+M3+M4+M6 (Prompt 2 raw) + M2+M8 (final)
    equation_solver_dll   M7 (Prompt 4) + M2+M8 (final)
    equation_solver_retry same as equation_solver_dll (solves the same equations)
    context_extractor     full composite (all components)

M7 vs M8 distinction
--------------------
  M7: Evaluates Prompt 4 equation-solving quality. Each step runs INDEPENDENTLY
      (no input accumulation). Provides directional GEPA signal for equation_solver_dll.
  M8: Evaluates the final post-processed test case correctness. Steps run
      SEQUENTIALLY (state and Wait semantics preserved). Catches dedup/post-
      processing corruption that M7 (stateless) would miss.

GEPA COMPATIBILITY
------------------
  mbd_metric_with_feedback(example, prediction, trace, pred_name, pred_trace)
    → float (when pred_name is None)
    → dspy.Prediction(score=float, feedback=str) (when pred_name is set)

  Staged dispatch by predictor name (the SCORE is ALWAYS the composite total — required by
  GEPA's fb.score == module_score check; only the FEEDBACK TEXT is tailored):
    testcase_generator    → text on M1/M3/M4/M6 (Prompt 2 raw) + M2/M8 (final)
    equation_solver_dll   → text on M2/M7/M8  (Prompt 4 + post-proc)
    equation_solver_retry → same signal as equation_solver_dll
    context_extractor     → text on the full composite

RELATIONSHIP TO run.py (fidelity audit)
---------------------------------------
  This file has NO counterpart in MIDAS run.py — run.py only GENERATES test cases and writes an
  XLSM; it computes no score. This is the wrapper-added evaluation layer GEPA optimizes against.
  Its INPUTS, however, are the run.py-faithful artifacts produced by the pipeline: the Example
  fields built in utils/mbd_data.py (equation_json, signalsheet/model_interface, equation_results_json,
  weakspot_op_key, relations_json, instance_names, raster_time, dll_path) and the forward() outputs
  in extract_mbd_full.py (test_cases_pre_postprocess, test_cases, inline m7_score). M7/M8 verify
  through the SAME FunctionCaller DLL mechanism run.py uses (run.py:821-826).

  M7/M8 DLL backbone — ONE-TO-ONE with run.py _step_4a (508-600), reusing the SAME imported helpers:
    eq_to_solve = assignment_chain.split("\\n")[::-1]   run.py:546
    _attempt_solve(simulated, eq_to_solve, fc)          run.py:581
    _build_substitution_dict(attempt.final_assignments) run.py:582
    _substitute_ops(final_substituted_equation, subs)   run.py:584-591 (inline regex copy)
    generate_bodmas_chain(expr)                          run.py:594
    resolve_equation_block(eq_list, fc)                  run.py:597
    actual = final_check[-1].split("=")[-1].strip()      run.py:600
  M7 runs this PER STEP independently (stateless, measures Prompt-4 quality); M8 runs it
  SEQUENTIALLY with DLL state carried across steps (measures the final post-processed test case,
  incl. function-call weakspots via fc.functionCall). run.py itself does NOT compare actual vs a
  per-step expected — that comparison is the wrapper-added scoring layer.
"""

from __future__ import annotations

import importlib  # [wrapper-only] lazy DLL/module import in the scoring layer (run.py has no metric)
import json  # [wrapper-only] parse Prompt-2 test_cases / Example JSON fields for scoring
import logging  # [wrapper-only] step-by-step metric trace logger
import os  # [wrapper-only] path handling for sys.path bootstrap + DLL work dirs
import re  # [wrapper-only] JSON-fence / assignment-string parsing in the scoring helpers
import sys  # [wrapper-only] insert project root on sys.path so utils/signatures import
import threading  # [wrapper-only] locks for the shared DLL/judge caches (thread-safe metric)
from typing import Any, Dict, List, Optional, Tuple  # [wrapper-only] type hints for scoring helpers

import dspy  # [wrapper-only] GEPA metric returns dspy.Prediction(score, feedback); no run.py analog

# ── Debug logger for step-by-step metric evaluation ──────────────────────────
# Silent by default (no handlers attached here, propagates to root). A caller
# that wants the full step-by-step trace configures logging (basicConfig /
# adds a file handler) and sets this logger's level to DEBUG.
logger = logging.getLogger("mbd_metric")  # [wrapper-only] named metric logger, silent unless caller configures it

# ── Project root on sys.path so utils.kb_parser and signatures.mbd are importable ──
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))  # [wrapper-only] this file's dir (evaluation/)
_PROJ_ROOT_METRIC = os.path.normpath(os.path.join(_EVAL_DIR, ".."))  # [wrapper-only] repo root (parent of evaluation/)
if _PROJ_ROOT_METRIC not in sys.path:  # [wrapper-only] only prepend once
    sys.path.insert(0, _PROJ_ROOT_METRIC)  # [wrapper-only] make utils/ and signatures/ importable at scoring time

try:  # [wrapper-only] KB is optional — M4 coverage judge degrades gracefully if utils.kb_parser is absent
    # NEW: layered lookup — resolve_guidelines() tries the NEW (BU Excel) KB first, then the OLD
    # (HTML) KB (see utils/kb_parser.resolve_guidelines). _KB_LOADED reflects EITHER KB present.
    from utils.kb_parser import (  # [wrapper-only] KB accessors used only by M4 (run.py has no KB scoring)
        WEAKSPOT_GUIDELINES,       # [wrapper-only] old (HTML-derived) KB dict {block_type: [guidelines]}
        NEW_WEAKSPOT_GUIDELINES,   # [wrapper-only] new (BU Excel-derived) KB dict
        find_block_type as _find_block_type,       # [wrapper-only] fn-name → block type resolver
        resolve_guidelines as _resolve_guidelines,  # [wrapper-only] fn-name → (block_type, kb_source, guidelines)
    )
    _KB_LOADED = bool(WEAKSPOT_GUIDELINES) or bool(NEW_WEAKSPOT_GUIDELINES)  # [wrapper-only] True if EITHER KB is non-empty
except ImportError:  # [wrapper-only] KB module missing → install no-op fallbacks so scoring still runs
    WEAKSPOT_GUIDELINES: dict = {}  # [wrapper-only] empty old-KB fallback
    NEW_WEAKSPOT_GUIDELINES: dict = {}  # [wrapper-only] empty new-KB fallback
    def _find_block_type(fn, kb=None): return None  # type: ignore[misc]  # [wrapper-only] no-op block-type resolver
    def _resolve_guidelines(fn): return None, None, []  # type: ignore[misc]  # [wrapper-only] no-op guideline resolver
    _KB_LOADED = False  # [wrapper-only] no KB available → M4 will fall back to neutral 0.5

try:  # [wrapper-only] Required-I/O KB is optional — M1/M2 fall back to default behavior if absent
    from utils.required_io import resolve_required_io as _resolve_required_io  # [wrapper-only] fc#ws → {required_inputs/preconds/outputs} | None
except ImportError:  # [wrapper-only] module missing → no-op (every weakspot skips the completeness/override)
    def _resolve_required_io(fc_name, ws_id): return None  # type: ignore[misc]  # [wrapper-only] no-op resolver

# ── DLL FunctionCaller registry (shared across metric calls, thread-safe) ───
_dll_caller_registry: dict = {}  # [wrapper-only] cache {dll_path: FunctionCaller} reused by M7/M8 (run.py builds one at 821-826)
_dll_registry_lock = threading.Lock()  # [wrapper-only] guards the DLL registry across metric threads

# ── LLM judge predictor (lazy, thread-safe) ──────────────────────────────────
_judge_predictor = None  # [wrapper-only] lazily-built dspy judge for M4 coverage (no run.py analog)
_judge_predictor_lock = threading.Lock()  # [wrapper-only] guards one-time judge construction

# ── M4 judge result memo (deterministic coverage — thread-safe) ──────────────
# GEPA scores the SAME prediction twice (module-score + feedback) and asserts the two
# composites are EQUAL (dspy gepa_utils.py: `assert fb["score"] == module_score`). On the
# Azure az-login path the judge LM is uncached (config.py forces cache=False), so identical
# judge inputs could return e.g. 4/8 then 3/8 -> composite 0.500 vs 0.475 -> AssertionError
# (that GEPA reflection iteration is discarded). Memoizing the judge's parsed result by content
# key makes M4 — and therefore the whole composite — deterministic per prediction. A sentinel
# (_JUDGE_MISS) distinguishes "not cached" from a cached neutral-0.5 (judge-unavailable) outcome.
_JUDGE_MISS = object()  # [wrapper-only] sentinel: "not cached" vs a cached neutral-0.5 judge-unavailable result
_judge_result_cache: dict = {}  # [wrapper-only] memoize M4 judge result by content key → deterministic composite
_judge_reasoning_cache: dict = {}  # [wrapper-only] memoize the judge's per-guideline coverage_reasoning (for M4 detail + GEPA feedback)
_judge_result_lock = threading.Lock()  # [wrapper-only] guards the judge-result memo across GEPA's twin scoring calls

# ── M8 DLL-safety guards (wrapper-only; run.py NEVER calls fc.functionCall on the weakspot fn) ──
# M8's functional-verification loop drives the 32-bit DLL, which run.py's pipeline never does for
# the top-level weakspot function. Two DATA-DEPENDENT pathologies (both PROVEN from run logs on
# Srv_TrnOffDly_out) could otherwise FREEZE an entire run — the generated test-case inputs decide
# whether they trigger, which is why some runs finish and some hang on the same code:
#   Guard 1 (tick cap): a boundary test with a huge "Wait" (e.g. 2_147_483_647 µs) expands to
#       wait//raster ≈ 107k DLL ticks for ONE step (observed: a 50 MB / 78k-call log). Cap the
#       ticks — a stateful block settles within a few ticks, so the M8 verdict is unchanged.
#   Guard 2 (call watchdog + poison): some arg combos make a SINGLE fc.functionCall spin forever
#       inside the DLL, and the bridge (function_caller.py:390) has NO read timeout. Run each call
#       under a timeout; on expiry, record the fn as hung (skip it thereafter) and drop the shared
#       FunctionCaller so a later example rebuilds a clean one. The step scores FAIL (None) — the
#       same honest "unverifiable weakspot" outcome already documented for non-exported library fns.
_M8_MAX_ADVANCE_TICKS = 2000   # [wrapper-only] Guard 1: max DLL ticks simulated per step (block settles long before)
_M8_FNCALL_TIMEOUT_S = 15.0    # [wrapper-only] Guard 2: max seconds for one fc.functionCall (healthy call returns in ms)
_M8_HUNG_FUNCTIONS: set = set()  # [wrapper-only] Guard 2: weakspot fn names that hung the DLL this run → skip on sight (guarded by _dll_registry_lock)


def _run_with_timeout(fn, timeout_s: float):  # [wrapper-only] Guard 2 helper — no run.py analog
    """Run fn() on a daemon thread and wait at most timeout_s. Returns (completed, result):
    completed=False means fn() was still blocked (caller treats it as a DLL hang and moves on; the
    daemon thread is abandoned — daemon=True so it never blocks interpreter exit). Any exception fn()
    raises is re-raised here so the caller's existing `except Exception: return None` still applies."""
    box: dict = {}
    def _target():
        try:
            box["result"] = fn()
        except BaseException as _e:  # noqa: BLE001 — capture to re-raise on the caller thread
            box["error"] = _e
    _th = threading.Thread(target=_target, daemon=True)
    _th.start()
    _th.join(timeout_s)
    if _th.is_alive():
        return False, None  # timed out — fn() still blocked inside the DLL bridge
    if "error" in box:
        raise box["error"]  # propagate fn()'s exception unchanged
    return True, box.get("result")


# ── DLL SOLVE watchdog (extends Guard 2 to the ARITHMETIC _attempt_solve/resolve_equation_block path) ──
# The 2026-07-14 GEPA trial FROZE for ~2h inside a Rail Prompt-4 solve: certain LLM-proposed values make the
# 32-bit DLL bridge (_rpc/_read_exact, function_caller.py:387-410) block forever holding its _io_lock, with NO
# read timeout. Guard 2 (_M8_FNCALL_TIMEOUT_S) only covered fc.functionCall; this covers the ARITHMETIC solve
# path (_attempt_solve / resolve_equation_block) used by forward() (Prompt 4) AND by M7/M8. On timeout we KILL
# the 32-bit helper subprocess: the blocked worker's stdout read then raises EOFError, which unwinds out of
# `with _io_lock` (releasing the lock) and reaps the zombie process — and the NEXT _rpc on the SAME
# FunctionCaller sees the dead process and auto-restarts a clean helper (function_caller.py:397). So the caller
# simply treats this one solve as a FAILED solve and keeps going; the DLL self-heals for the next step.
_DLL_SOLVE_TIMEOUT_S = float(os.getenv("MBD_DLL_SOLVE_TIMEOUT_S", "120") or "120")  # healthy solve is <5s; 120s = generous hang cutoff


class _DLLSolveTimeout(Exception):  # [wrapper-only] raised when a DLL solve/verify exceeds _DLL_SOLVE_TIMEOUT_S
    """A DLL _attempt_solve / resolve_equation_block call exceeded its wall-clock budget and was abandoned;
    the caller treats the step as a FAILED solve — never a crash, never an infinite hang."""


def solve_with_dll_timeout(fn, fc, timeout_s: float = None):  # [wrapper-only] watchdog around ONE DLL solve thunk
    """Run fn() (a DLL _attempt_solve / resolve_equation_block call) under a hard timeout. On expiry, KILL the
    FunctionCaller `fc`'s 32-bit helper (releases the bridge _io_lock held by the abandoned worker + reaps the
    zombie; the next _rpc auto-restarts a clean helper) and raise _DLLSolveTimeout. See the note above."""
    _t = _DLL_SOLVE_TIMEOUT_S if timeout_s is None else timeout_s
    completed, result = _run_with_timeout(fn, _t)          # [wrapper-only] reuse Guard-2's daemon-thread watchdog
    if not completed:                                      # [wrapper-only] still blocked inside the DLL bridge → hang
        try:
            _proc = getattr(fc, "process", None)           # [wrapper-only] the 32-bit helper subprocess (killable, unlike an in-thread C call)
            if _proc is not None:
                _proc.kill()                               # [wrapper-only] unblock the abandoned worker (EOF → releases _io_lock) + reap zombie
        except Exception:                                  # [wrapper-only] best-effort; never mask the timeout
            pass
        raise _DLLSolveTimeout(f"DLL solve exceeded {_t:.0f}s (helper killed; FunctionCaller self-restarts next call)")
    return result


# ═══════════════════════════════════════════════════════════════════
# SCHEMA NORMALIZER
# ═══════════════════════════════════════════════════════════════════
def _parse_assign_str(s: str) -> Tuple[str, str]:  # [wrapper-only] metric helper (run.py has no scoring layer)
    """Parse 'name = value' string → (name, value_str) or ('','')."""
    m = re.match(r"(.+?)\s*=\s*(.+)", s.strip())  # [wrapper-only] split "lhs = rhs" (assignment strings the pipeline emits)
    return (m.group(1).strip(), m.group(2).strip()) if m else ("", "")  # [wrapper-only] (name, value) or ("","") on no match


def _dict_from_anything(raw) -> Dict[str, str]:  # [wrapper-only] metric helper — normalize {name:val} from list/dict/str
    """Convert any representation of {name→value} to a plain str→str dict."""
    if isinstance(raw, dict):  # [wrapper-only] dict schema (HYBRID/NEW): {name: value}
        return {str(k): str(v) for k, v in raw.items()}  # [wrapper-only] stringify keys+values
    if isinstance(raw, list):  # [wrapper-only] list schema (OLD): ["name = val", ...] or [{name:val}, ...]
        out: Dict[str, str] = {}  # [wrapper-only] accumulate the flattened {name: value}
        for item in raw:  # [wrapper-only] each list entry may be an assignment string or a nested dict
            if isinstance(item, str):  # [wrapper-only] "name = val" string entry
                n, v = _parse_assign_str(item)  # [wrapper-only] split into (name, value)
                if n:  # [wrapper-only] keep only entries with a parsed name
                    out[n] = v  # [wrapper-only] record name→value
            elif isinstance(item, dict):  # [wrapper-only] nested {name: value} entry
                for k, v in item.items():  # [wrapper-only] merge each pair in
                    out[str(k)] = str(v)  # [wrapper-only] stringify
        return out  # [wrapper-only] normalized dict from the list
    return {}  # [wrapper-only] unrecognized type → empty dict


def _strip_json_comments(s: str) -> str:  # [wrapper-only] tolerate // and /* */ comments some LLMs add to JSON — no run.py analog
    """Remove // line and /* */ block comments an LLM may add inside JSON, WITHOUT touching text
    inside string literals (so a value like "a // b" or "http://x" is preserved). String-aware so
    it never corrupts real data. Weaker models (e.g. gpt-4.1-mini) sometimes annotate values with
    `"x = 251"  // 0xFB` — invalid JSON that otherwise collapses the whole pipeline + metrics to 0."""
    if not isinstance(s, str) or ("//" not in s and "/*" not in s):  # [wrapper-only] fast path: nothing to strip
        return s
    out = []  # [wrapper-only] rebuilt characters
    i, n = 0, len(s)  # [wrapper-only] cursor + length
    in_str = False  # [wrapper-only] currently inside a string literal?
    quote = ""  # [wrapper-only] the opening quote char (" or ')
    while i < n:  # [wrapper-only] single pass
        c = s[i]  # [wrapper-only] current char
        if in_str:  # [wrapper-only] copy string content verbatim (comments inside strings are DATA)
            out.append(c)  # [wrapper-only]
            if c == "\\" and i + 1 < n:  # [wrapper-only] keep an escaped char with its backslash
                out.append(s[i + 1]); i += 2; continue  # [wrapper-only]
            if c == quote:  # [wrapper-only] string closed
                in_str = False  # [wrapper-only]
            i += 1; continue  # [wrapper-only]
        if c in ('"', "'"):  # [wrapper-only] string opens
            in_str = True; quote = c; out.append(c); i += 1; continue  # [wrapper-only]
        if c == "/" and i + 1 < n and s[i + 1] == "/":  # [wrapper-only] // line comment → skip to newline
            j = s.find("\n", i); i = n if j == -1 else j; continue  # [wrapper-only]
        if c == "/" and i + 1 < n and s[i + 1] == "*":  # [wrapper-only] /* block comment */ → skip to close
            j = s.find("*/", i + 2); i = n if j == -1 else j + 2; continue  # [wrapper-only]
        out.append(c); i += 1  # [wrapper-only] ordinary char
    return "".join(out)  # [wrapper-only] comment-free JSON text


def _extract_json_str(s: str) -> str:  # [wrapper-only] metric helper — mirrors run.py bridge regex (test_generator.py:64) but for scoring
    """Pull a parseable JSON array out of an LLM/test_cases string.

    Mirrors the pipeline bridge (run.py generate_and_parse_test_cases / forward()):
    raw Prompt-2 output is commonly wrapped in a ```json ... ``` markdown fence and/or
    surrounded by prose. M1/M3/M4/M6 evaluate that RAW output, so the metric must be as
    tolerant as the pipeline — otherwise a harmless code fence silently scores them 0.
    """
    if not isinstance(s, str):  # [wrapper-only] non-string input → nothing to extract
        return ""  # [wrapper-only] empty on non-string
    s = _strip_json_comments(s.strip())  # [wrapper-only] trim + remove // /* */ comments (string-aware) before matching
    # 1) strip a markdown code fence if present (```json ... ``` or ``` ... ```)
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL)  # [wrapper-only] fence-strip (pipeline regex only handles the array)
    if m:  # [wrapper-only] fence found
        s = m.group(1).strip()  # [wrapper-only] keep the fenced body
    # 2) prefer the first top-level JSON array if extra prose surrounds it
    m2 = re.search(r"\[\s*{.*}\s*\]", s, re.DOTALL)  # test_generator.py:64 SAME regex the pipeline bridge uses to pull the JSON array
    if m2:  # test_generator.py:65 array matched
        return m2.group()  # test_generator.py:64 .group() → the matched JSON array text
    return s  # [wrapper-only] no array match → hand back the (fence-stripped) string as-is


def _normalize_test_cases(test_cases_str: str) -> Tuple[List[Dict], str]:  # [wrapper-only] schema normalizer for scoring — no run.py analog
    """Parse test_cases JSON (any schema variant) → canonical list + error str.

    GEPA has evolved at least three schemas during this run:

    OLD (original prompt, Seed):
        { "Pre-Condition": ["name = val", ...],
          "Steps": [{"Actions": ["name = val", "Wait N"], "Expected Results": ["name = val"]}],
          "Detailed Explanation": "..." }

    HYBRID iter-24 (dict-valued old keys):
        { "Pre-Condition": {name: number, ...},
          "Steps": [{"Wait": N, "Actions": {name: number}, "Expected Results": {name: number}}] }

    NEW iter-2 (fully renamed keys):
        { "pre_condition": {name: number},
          "steps": [{"actions": {name: number}, "wait_time": N, "expected_results": {name: number}}],
          "explanation": "..." }

    Canonical output per test case:
        {
          "precond":  {name: str_val, ...},           # pre-condition assignments
          "steps":    [{"wait": int|None,
                        "act":  {name: str_val},      # actions
                        "exp":  {name: str_val}}],    # expected results
          "explanation": str,
        }
    """
    try:  # [wrapper-only] parse test_cases JSON for scoring (run.py 5×-retries at generation time instead)
        raw = json.loads(_extract_json_str(test_cases_str))  # [wrapper-only] extract array (fence-tolerant) then json.loads
    except (json.JSONDecodeError, TypeError) as e:  # [wrapper-only] bad/absent JSON → scoreable error string
        logger.error("_normalize_test_cases: JSON parse failed: %s | input[:200]=%r", e, str(test_cases_str)[:200])  # [wrapper-only] trace
        return [], f"Invalid JSON: {e}"  # [wrapper-only] empty list + error → structural-validity failure

    if not isinstance(raw, list) or not raw:  # [wrapper-only] pipeline expects a non-empty JSON array of test cases
        logger.error("_normalize_test_cases: not a non-empty JSON array (type=%s)", type(raw).__name__)  # [wrapper-only] trace
        return [], "Not a JSON array or empty array"  # [wrapper-only] error → structural failure

    logger.debug("_normalize_test_cases: parsed %d raw test case(s) from input", len(raw))  # [wrapper-only] trace

    canonical: List[Dict] = []  # [wrapper-only] accumulate normalized test cases
    for tc in raw:  # [wrapper-only] normalize each raw test case into the canonical schema
        if not isinstance(tc, dict):  # [wrapper-only] skip non-dict entries defensively
            continue  # [wrapper-only]

        c: Dict = {"precond": {}, "steps": [], "explanation": ""}  # [wrapper-only] canonical skeleton for this test case

        # ── Explanation ──────────────────────────────────────────────
        # [wrapper-only] accept any of the three schema key spellings for the explanation field
        c["explanation"] = str(
            tc.get("Detailed Explanation")  # [wrapper-only] OLD schema key
            or tc.get("explanation")        # [wrapper-only] NEW schema key
            or tc.get("description")        # [wrapper-only] alt key
            or ""                           # [wrapper-only] default empty
        )

        # ── Pre-condition: any of three key names, list or dict value ─
        # [wrapper-only] pre-condition under OLD/NEW/alt key spellings
        pc_raw = (
            tc.get("Pre-Condition")   # [wrapper-only] OLD key (list of "name = val")
            or tc.get("pre_condition")  # [wrapper-only] NEW key (dict)
            or tc.get("precondition")   # [wrapper-only] alt key
            or []                       # [wrapper-only] default empty
        )
        c["precond"] = _dict_from_anything(pc_raw)  # [wrapper-only] normalize list/dict → {name: value}

        # ── Steps ────────────────────────────────────────────────────
        steps_raw = tc.get("Steps") or tc.get("steps") or []  # [wrapper-only] steps under OLD("Steps")/NEW("steps") key
        for step in steps_raw:  # [wrapper-only] normalize each step
            if not isinstance(step, dict):  # [wrapper-only] skip non-dict steps defensively
                continue  # [wrapper-only]
            cs: Dict = {"wait": None, "act": {}, "exp": {}}  # [wrapper-only] canonical step skeleton

            # Detect schema variant by key presence
            has_new_keys = "actions" in step or "wait_time" in step or "expected_results" in step  # [wrapper-only] NEW schema marker
            has_hybrid_keys = "Wait" in step  # hybrid: "Wait": N (int), "Actions": dict  # [wrapper-only] HYBRID marker (int Wait key)

            if has_new_keys:  # [wrapper-only] NEW schema branch
                # NEW: actions/wait_time/expected_results
                wt = step.get("wait_time")  # [wrapper-only] NEW wait field (rasters/µs int)
                cs["wait"] = int(wt) if wt is not None else None  # [wrapper-only] coerce to int or leave None
                cs["act"] = _dict_from_anything(step.get("actions") or {})  # [wrapper-only] normalize actions dict
                cs["exp"] = _dict_from_anything(step.get("expected_results") or {})  # [wrapper-only] normalize expected results

            elif has_hybrid_keys:  # [wrapper-only] HYBRID schema branch
                # HYBRID: Wait (int key), Actions (dict), Expected Results (dict)
                wt = step.get("Wait")  # [wrapper-only] HYBRID wait field (int)
                cs["wait"] = int(wt) if wt is not None else None  # [wrapper-only] coerce to int or leave None
                cs["act"] = _dict_from_anything(step.get("Actions") or {})  # [wrapper-only] normalize actions dict
                cs["exp"] = _dict_from_anything(step.get("Expected Results") or {})  # [wrapper-only] normalize expected results

            else:
                # OLD: Actions is a list of strings OR a list of lists of strings.
                # After run.py's extract_and_replace_equations, UUIDs are replaced
                # with sub-lists like [["sig=val", ...], ["sig=val", ...]] before
                # flatten_precondition_and_actions runs.  Flatten one level here so
                # actions are never silently dropped.
                actions_raw = step.get("Actions", [])  # [wrapper-only] OLD "Actions" list (strings or sub-lists)
                actions_flat: List[str] = []  # [wrapper-only] one-level-flattened action strings
                for item in actions_raw:  # [wrapper-only] flatten nested UUID-replacement sub-lists
                    if isinstance(item, list):  # [wrapper-only] sub-list [["sig=val", ...]] → extend
                        actions_flat.extend(str(x) for x in item)  # [wrapper-only] stringify each inner element
                    elif isinstance(item, str):  # [wrapper-only] plain "name = val" string
                        actions_flat.append(item)  # [wrapper-only] keep as-is

                for a in actions_flat:  # [wrapper-only] classify each action: Wait directive vs assignment
                    a = a.strip()  # [wrapper-only] trim before matching
                    # "Wait N*@" → N rasters.  Store as -N so M8 knows the unit.
                    # "Wait @"   → 1 raster.   Store as -1 (existing convention).
                    # "Wait N"   → N µs.        Store as +N (legacy absolute time).
                    wm_mult = re.match(r"Wait\s+(\d+)\s*\*\s*@", a, re.IGNORECASE)  # [wrapper-only] "Wait N*@" → N rasters
                    if wm_mult:  # [wrapper-only] N-raster form matched
                        cs["wait"] = -int(wm_mult.group(1))  # -N = N rasters  # [wrapper-only] store negated so M8 reads the raster unit
                    elif re.match(r"Wait\s*@", a, re.IGNORECASE):  # [wrapper-only] "Wait @" → single raster
                        cs["wait"] = -1   # 1 raster  # [wrapper-only] -1 = 1 raster (existing convention)
                    else:  # [wrapper-only] not a raster-based Wait
                        wm = re.match(r"Wait\s+(\d+)", a, re.IGNORECASE)  # [wrapper-only] "Wait N" → absolute µs
                        if wm:  # [wrapper-only] absolute-time Wait matched
                            cs["wait"] = int(wm.group(1))  # absolute µs  # [wrapper-only] +N = N µs (legacy)
                        else:  # [wrapper-only] not a Wait at all → treat as an assignment
                            n, v = _parse_assign_str(a)  # [wrapper-only] parse "name = val"
                            if n:  # [wrapper-only] parsed name → record as an action
                                cs["act"][n] = v  # [wrapper-only] name→value into this step's actions
                cs["exp"] = _dict_from_anything(step.get("Expected Results") or [])  # [wrapper-only] OLD expected results (list of "name = val")

            # [wrapper-only] per-step trace
            logger.debug(
                "_normalize_test_cases: TC%d step%d -> wait=%s act=%s exp=%s",
                len(canonical) + 1, len(c["steps"]) + 1, cs["wait"], cs["act"], cs["exp"],
            )
            c["steps"].append(cs)  # [wrapper-only] add normalized step to this test case

        # [wrapper-only] per-test-case trace
        logger.debug(
            "_normalize_test_cases: TC%d -> precond=%s, %d step(s), explanation=%r",
            len(canonical) + 1, c["precond"], len(c["steps"]), c["explanation"][:80],
        )
        canonical.append(c)  # [wrapper-only] add normalized test case to the output list

    logger.debug("_normalize_test_cases: produced %d canonical test case(s)", len(canonical))  # [wrapper-only] trace
    return canonical, ""  # [wrapper-only] canonical list + no-error (empty string)


# ═══════════════════════════════════════════════════════════════════
# WEIGHTS
# ═══════════════════════════════════════════════════════════════════
WEIGHT_M1_IO = 0.20  # [wrapper-only] M1 output-role weight (up 0.15->0.20: most reliable / expert-aligned signal)
WEIGHT_M2_SIGNAL = 0.15  # [wrapper-only] M2 signal-resolution + input/precond completeness weight
WEIGHT_M3_STRUCTURAL = 0.15  # [wrapper-only] M3 structural-validity weight (up 0.10->0.15)
WEIGHT_M4_COVERAGE = 0.20  # [wrapper-only] M4 KB-coverage (LLM judge) weight
WEIGHT_M7_DLL = 0.15  # [wrapper-only] M7 per-step DLL pass-rate weight (down 0.20->0.15: neutral 0.5 for fn-call weakspots)
# M6 (boundary values) REMOVED 2026-07-09: it was weight 0 and unused. No WEIGHT_M6 constant.
WEIGHT_M8_FUNCTIONAL = 0.15  # [wrapper-only] M8 sequential functional-correctness weight (down 0.20->0.15: ~0/gated for disconnected weakspots)
# Sum = 0.20+0.15+0.15+0.20+0.0+0.15+0.15 = 1.00

# [wrapper-only] M3 weight of the two "don't verify a value you set" TAUTOLOGY-overlap checks — act↔expected
# (a var both driven and verified in the same step) and precond↔expected (a var both pinned and verified).
# ONLY ON FAILURE each counts as W failed checks (a clean pass counts as ONE normal check, so it does NOT
# inflate the denominator / dilute other failures). So a violation weighs W× a normal check. Set to 1 to
# weigh them like any other check.
M3_OVERLAP_WEIGHT = 2

# [wrapper-only] M1 weight of the two I/O-role failures PS-GB experts flag most: (c) the test verifies NO real
# weakspot output (checks a fabricated / internal signal instead of the measurable output — e.g. testing
# `_flgDlydlnPostDrive` when the real output is `InjPwrRly_r`), and (d) an OUTPUT signal is driven as an INPUT
# (e.g. `GbxFan_r ... is an output, should not be in input`). ONLY ON FAILURE each counts as W failed checks
# (a clean pass counts as one normal check — no denominator inflation). Set to 1 to weigh like any other check.
M1_IO_CHECK_WEIGHT = 1

# [wrapper-only] M8 hard gate: a test that verifies a NON-measurable output (no Expected-Results signal is a
# real weakspot output / instance_name) or drives a weakspot OUTPUT as an INPUT is functionally invalid — its
# DLL "verification" would confirm the WRONG signal. When True, M8 short-circuits to 0.0 WITHOUT running the
# DLL in those cases (aligned with PS-GB expert remarks: "Final output should be InjPwrRly_r"). Set False to
# always run the DLL regardless of I/O correctness (old behaviour).
M8_GATE_ON_WRONG_IO = True


# ═══════════════════════════════════════════════════════════════════
# M3: STRUCTURAL VALIDITY (rule-based, schema-agnostic)
# ═══════════════════════════════════════════════════════════════════
def _m3_structural_validity(test_cases_str: str, raster_time: int) -> Tuple[float, str]:  # [wrapper-only] M3 metric (rule-based) — no run.py analog
    """Check JSON validity, schema correctness, numeric values, Wait times.

    Uses _normalize_test_cases() so all GEPA-evolved schemas are handled equally.
    """
    logger.info("M3: ===== Structural Validity (raster_time=%s) =====", raster_time)  # [wrapper-only] trace
    tcs, err = _normalize_test_cases(test_cases_str)  # [wrapper-only] parse+canonicalize the model's test cases (schema-agnostic)
    if err:  # [wrapper-only] RULE: unparseable output cannot be structurally valid
        logger.error("M3: parse error -> score=0.0 (%s)", err)  # [wrapper-only] trace
        return 0.0, err  # [wrapper-only] hard-fail: no JSON -> score 0

    details = []  # [wrapper-only] human-readable list of every failed check (returned as feedback)
    checks_passed = 0  # [wrapper-only] numerator: count of passing checks across all TCs
    checks_total = 0  # [wrapper-only] denominator: total checks attempted (score = passed/total)
    precond_out_violations = 0  # [wrapper-only] # of TCs pinning an output as precondition AND verifying it (the fundamental tautology)

    for i, tc in enumerate(tcs):  # [wrapper-only] score each canonical test case independently
        # NOTE: "must have at least one step" check intentionally removed —
        # not every FC/weakspot needs explicit steps every time.

        # Must have an explanation (non-empty)
        checks_total += 1  # [wrapper-only] register the explanation check
        if tc["explanation"].strip():  # [wrapper-only] RULE: every TC must carry a non-empty rationale (traceability)
            checks_passed += 1  # [wrapper-only] pass: explanation present
            logger.debug("M3: TC%d explanation check PASS", i + 1)  # [wrapper-only] trace
        else:  # [wrapper-only] RULE violated: explanation empty/whitespace
            details.append(f"TC{i+1}: missing explanation")  # [wrapper-only] record failure detail
            logger.debug("M3: TC%d explanation check FAIL (empty)", i + 1)  # [wrapper-only] trace

        for j, step in enumerate(tc["steps"]):  # [wrapper-only] apply the per-step structural rules to each step
            # Wait-timing is validated ONLY WHEN a Wait is present. A step with no Wait is a valid
            # INSTANTANEOUS step (apply action, check expected on the next tick) — not every step needs a
            # Wait — so it is NOT penalised and NOT counted (no denominator inflation). The multiple-of-dT
            # rule applies only to steps that DO specify a Wait.
            w = step["wait"]  # [wrapper-only] normalized Wait value (int; None=unspecified, negative=raster-encoded)
            if w is None:  # [wrapper-only] no Wait -> valid instantaneous step; skip the timing check entirely
                logger.debug("M3: TC%d.S%d Wait check SKIP (no Wait — instantaneous step, not penalised)", i + 1, j + 1)  # [wrapper-only] trace
                # (fall through to the act/exp overlap check below; no checks_total increment)
            elif w < 0:  # [wrapper-only] RULE: raster-relative Wait ("Wait @"/"Wait N*@") is always valid
                # Negative = raster-based wait: "Wait @" (w=-1) or "Wait N*@" (w=-N)
                checks_total += 1  # [wrapper-only] register the Wait-timing check (a Wait IS present)
                checks_passed += 1  # [wrapper-only] pass: raster-encoded wait is inherently dT-aligned
                logger.debug("M3: TC%d.S%d Wait check PASS (raster-encoded w=%d)", i + 1, j + 1, w)  # [wrapper-only] trace
            elif raster_time <= 0:  # [wrapper-only] RULE: dT unknown for this FC -> can't judge, so don't penalise
                # raster_time not reliably known for this FC — don't penalise the
                # Wait value; give the benefit of the doubt.
                checks_total += 1  # [wrapper-only] register the Wait-timing check (a Wait IS present)
                checks_passed += 1  # [wrapper-only] pass: benefit of the doubt when dT is unavailable
                logger.debug("M3: TC%d.S%d Wait check PASS (raster_time<=0, w=%s not penalised)", i + 1, j + 1, w)  # [wrapper-only] trace
            elif w % raster_time == 0:  # [wrapper-only] RULE: absolute Wait must be an integer multiple of the raster period dT
                checks_total += 1  # [wrapper-only] register the Wait-timing check (a Wait IS present)
                checks_passed += 1  # [wrapper-only] pass: Wait is dT-aligned (schedulable on the raster)
                logger.debug("M3: TC%d.S%d Wait check PASS (w=%d is multiple of dT=%d)", i + 1, j + 1, w, raster_time)  # [wrapper-only] trace
            else:  # [wrapper-only] RULE violated: Wait not on the raster grid -> unschedulable timing
                checks_total += 1  # [wrapper-only] register the Wait-timing check (a Wait IS present)
                details.append(f"TC{i+1}.S{j+1}: Wait {w} not multiple of dT={raster_time}")  # [wrapper-only] fail detail
                logger.debug("M3: TC%d.S%d Wait check FAIL (w=%d not multiple of dT=%d)", i + 1, j + 1, w, raster_time)  # [wrapper-only] trace

            # All action values must be numeric
            for name, val_str in step["act"].items():  # [wrapper-only] check every stimulus (action) value in this step
                checks_total += 1  # [wrapper-only] register one numeric check per action var
                try:  # [wrapper-only] RULE: action values must be numeric (they drive the DLL / M7-M8 simulation)
                    float(val_str)  # [wrapper-only] parse test: numeric if float() succeeds
                    checks_passed += 1  # [wrapper-only] pass: value is a valid number
                    logger.debug("M3: TC%d.S%d action %s=%s numeric check PASS", i + 1, j + 1, name, val_str)  # [wrapper-only] trace
                except (ValueError, TypeError):  # [wrapper-only] RULE violated: non-numeric action value can't be applied
                    details.append(f"TC{i+1}.S{j+1}: non-numeric action {name}={val_str[:30]}")  # [wrapper-only] fail detail (truncated)
                    logger.debug("M3: TC%d.S%d action %s=%s numeric check FAIL", i + 1, j + 1, name, val_str)  # [wrapper-only] trace

            # All expected-result values must be numeric
            for name, val_str in step["exp"].items():  # [wrapper-only] check every expected-result value in this step
                checks_total += 1  # [wrapper-only] register one numeric check per expected var
                try:  # [wrapper-only] RULE: expected values must be numeric (they are compared against DLL outputs)
                    float(val_str)  # [wrapper-only] parse test: numeric if float() succeeds
                    checks_passed += 1  # [wrapper-only] pass: value is a valid number
                    logger.debug("M3: TC%d.S%d exp %s=%s numeric check PASS", i + 1, j + 1, name, val_str)  # [wrapper-only] trace
                except (ValueError, TypeError):  # [wrapper-only] RULE violated: non-numeric expected value is uncheckable
                    details.append(f"TC{i+1}.S{j+1}: non-numeric exp {name}={val_str[:30]}")  # [wrapper-only] fail detail (truncated)
                    logger.debug("M3: TC%d.S%d exp %s=%s numeric check FAIL", i + 1, j + 1, name, val_str)  # [wrapper-only] trace

            # No variable in both act and exp within same step (tautology). Weight-on-failure (M3_OVERLAP_WEIGHT).
            both = set(step["act"]) & set(step["exp"])  # [wrapper-only] vars appearing as BOTH stimulus and expected in one step
            if not both:  # [wrapper-only] RULE: a var you SET (act) must not also be a var you VERIFY (exp) in the same step
                checks_total += 1  # [wrapper-only] pass: one normal check (no inflation)
                checks_passed += 1  # [wrapper-only] pass: act and exp are disjoint
                logger.debug("M3: TC%d.S%d act/exp-overlap check PASS (no overlap)", i + 1, j + 1)  # [wrapper-only] trace
            else:  # [wrapper-only] RULE violated: setting and checking the same var is a tautology
                checks_total += M3_OVERLAP_WEIGHT  # [wrapper-only] fail: counts as W failed checks (bites W× harder)
                details.append(f"TC{i+1}.S{j+1}: var in both act+exp (weighs {M3_OVERLAP_WEIGHT}x): {list(both)[:3]}")  # [wrapper-only] fail detail (first 3 offenders)
                logger.debug("M3: TC%d.S%d act/exp-overlap check FAIL (weighs %dx): %s", i + 1, j + 1, M3_OVERLAP_WEIGHT, both)  # [wrapper-only] trace

        # [wrapper-only] Role disjointness that MUST hold: precond↔expected and action↔expected — i.e. "don't
        # verify a value you set." NOTE: precond↔action is DELIBERATELY NOT checked — setting a signal in the
        # precondition (initial state) and then driving it in an action (e.g. precond X=0, then action X=4) is
        # the legitimate "initialize-then-drive" test pattern, not a flaw. So this is a 2-way, not 3-way, rule.
        # (action↔expected is checked per step above; precond↔expected is checked here.)
        pre_vars = set(tc["precond"])  # [wrapper-only] variables fixed by the precondition
        exp_vars: set = set()  # [wrapper-only] accumulate all expected-result vars across the TC's steps
        for step in tc["steps"]:  # [wrapper-only] gather exp vars from every step
            exp_vars.update(step["exp"])  # [wrapper-only] collect this step's expected vars
        overlap = pre_vars & exp_vars  # [wrapper-only] vars pinned by precond yet also verified as outputs
        # [wrapper-only] Weight this fundamental flaw ONLY ON FAILURE: a violation counts as W FAILED checks
        # (denominator += W, numerator += 0) so the MISTAKE weighs W× a normal check, but a clean PASS counts
        # as ONE normal check (denominator += 1) — so a passing check does NOT inflate the denominator and
        # dilute other failures. Per TC with ~9 other checks: violation -> 9/(9+W) (W=4 -> 0.69); clean -> no effect.
        if not overlap:  # [wrapper-only] RULE: a var fixed in the precondition must not also be an expected OUTPUT (can't verify a var you pinned)
            checks_total += 1  # [wrapper-only] pass: a normal single check (no denominator inflation)
            checks_passed += 1  # [wrapper-only] pass: precond and expected are disjoint
            logger.debug("M3: TC%d precond/expected-overlap check PASS (no overlap)", i + 1)  # [wrapper-only] trace
        else:  # [wrapper-only] RULE violated: precond pins a var that's also checked as output
            checks_total += M3_OVERLAP_WEIGHT  # [wrapper-only] the fundamental flaw counts as W FAILED checks (bites W× harder)
            precond_out_violations += 1  # [wrapper-only] count for the trace (not used for compounding)
            details.append(f"TC{i+1}: OUTPUT PINNED AS PRECONDITION (also verified, weighs {M3_OVERLAP_WEIGHT}x): {list(overlap)[:3]}")  # [wrapper-only] fail detail
            logger.debug("M3: TC%d precond/expected-overlap check FAIL (weighs %dx): %s", i + 1, M3_OVERLAP_WEIGHT, overlap)  # [wrapper-only] trace

    score = checks_passed / checks_total if checks_total > 0 else 0.0  # [wrapper-only] M3 = weighted fraction of structural checks passed (the flaw check weighs W×)
    if precond_out_violations:  # [wrapper-only] trace the flaw's effect
        logger.info("M3: OUTPUT-PINNED-AS-PRECONDITION: %d TC(s), each weighed %dx a normal check", precond_out_violations, M3_OVERLAP_WEIGHT)  # [wrapper-only]
    logger.info("M3: RESULT score=%.4f (%d/%d weighted checks passed)", score, checks_passed, checks_total)  # [wrapper-only] trace
    if details:  # [wrapper-only] only log the failure list when something failed
        logger.info("M3: failed checks: %s", "; ".join(details))  # [wrapper-only] trace all failures
    return score, "; ".join(details[:8]) if details else "All structural checks passed"  # [wrapper-only] score + feedback (first 8 failures, or success msg)


# ═══════════════════════════════════════════════════════════════════
# M1: I/O CLASSIFICATION (rule-based, schema-agnostic)
# ═══════════════════════════════════════════════════════════════════
def _m1_io_classification(   # [wrapper-only] M1 metric (rule-based I/O classification) — no run.py analog
    test_cases_str: str,
    equation_json: str,
    signalsheet_data: str,
    instance_names: str = "",
    required_outputs: Optional[List[str]] = None,  # [wrapper-only] expert-curated outputs (required_io KB) — OVERRIDES instance_names
) -> Tuple[float, str]:
    """M1 = OUTPUT-ROLE correctness vs ground truth (instance_names / fixed consts). Checks: (a) fixed
    consts aren't DRIVEN or VERIFIED; (c) the test VERIFIES a real weakspot output; (d) a real output isn't
    SET as an input. Role-AGNOSTIC disjointness (action/precond/expected overlaps) lives in M3, not here.
    Input-role is NOT judged here: at Prompt-2 output the inputs are still intermediate/arg names that get
    BACKTRACED to physical inputs later (update_test_cases + Prompt 4), so judging them here is premature.
    If ``required_outputs`` is provided (expert-curated required_io KB), it OVERRIDES instance_names as the
    ground-truth output set."""
    logger.info("M1: ===== I/O Classification =====")  # [wrapper-only] trace banner
    tcs, err = _normalize_test_cases(test_cases_str)  # [wrapper-only] parse Prompt-2 JSON into normalized TC dicts (helper)
    if err:  # [wrapper-only] unparseable test cases -> nothing to classify
        logger.error("M1: parse error -> score=0.0 (%s)", err)  # [wrapper-only] trace
        return 0.0, f"Cannot parse test_cases: {err}"  # [wrapper-only] score 0.0 + feedback on parse failure

    try:  # [wrapper-only] equation graph is optional context for the fixed-var rule
        equation = json.loads(equation_json)  # [wrapper-only] parse equation graph JSON (var -> element dict)
    except (json.JSONDecodeError, TypeError):  # [wrapper-only] missing/malformed -> treat as empty graph
        equation = {}  # [wrapper-only] no equation info available

    # Real weakspot OUTPUT signals (the checkable outputs). Expert-curated required_outputs (required_io
    # KB) OVERRIDE the auto-derived instance_names when present; else fall back to instance_names.
    # Used for the output-role checks below; empty -> those checks are skipped (can't judge role).
    if required_outputs:  # [wrapper-only] expert KB present -> authoritative output set
        output_set = {str(n).strip() for n in required_outputs if str(n).strip()}  # [wrapper-only] curated outputs
        logger.debug("M1: weakspot output signals (required_io KB) = %s", output_set)  # [wrapper-only] trace
    else:  # [wrapper-only] no KB -> auto-derived instance_names
        output_set = {n.strip() for n in (instance_names or "").split(",") if n.strip()}  # [wrapper-only] declared weakspot outputs from Example instance_names (comma-split, stripped)
        logger.debug("M1: weakspot output signals (instance_names) = %s", output_set)  # [wrapper-only] trace

    # Build set of fixed (non-alterable) variables from equation graph
    fixed_vars: set = set()  # [wrapper-only] compile-time constants the weakspot never computes as I/O
    for key, elem in equation.items():  # [wrapper-only] scan each element of the equation graph
        if not isinstance(elem, dict):  # [wrapper-only] skip non-dict graph entries
            continue  # [wrapper-only]
        etype = elem.get("type", "")  # [wrapper-only] element type (e.g. SystemConstant/Literal)
        instance = elem.get("instance_name", "")  # [wrapper-only] the signal/var name for this element
        if not instance:  # [wrapper-only] unnamed elements can't be matched against I/O vars
            continue  # [wrapper-only]
        if etype in ("SystemConstant", "Literal"):  # [wrapper-only] RULE: SystemConstant/Literal are fixed consts
            fixed_vars.add(instance)  # [wrapper-only] mark as fixed
        elif instance.lower() in ("dt",):  # [wrapper-only] RULE: raster time dT is also a fixed const
            fixed_vars.add(instance)  # [wrapper-only] mark dT as fixed

    # From signalsheet: sysconstant kind → fixed
    for line in signalsheet_data.split("\n")[1:]:  # [wrapper-only] iterate signalsheet rows (skip header row 0)
        parts = line.split(",")  # [wrapper-only] CSV columns: [name, ..., kind, ...]
        if len(parts) >= 3 and parts[2].strip() == "sysconstant":  # [wrapper-only] RULE: kind col == "sysconstant" -> fixed
            fixed_vars.add(parts[0].strip())  # [wrapper-only] add signal name (col 0) to fixed set

    logger.debug("M1: fixed_vars (SystemConstant/Literal/dT) = %s", fixed_vars)  # [wrapper-only] trace

    details = []  # [wrapper-only] accumulates human-readable failure messages
    checks_passed = 0  # [wrapper-only] numerator of the M1 score
    checks_total = 0  # [wrapper-only] denominator of the M1 score

    for i, tc in enumerate(tcs):  # [wrapper-only] classify each normalized test case
        precond_vars = set(tc["precond"])  # [wrapper-only] vars pinned by preconditions (inputs)
        action_vars: set = set()  # [wrapper-only] vars driven by test-step actions (inputs)
        expected_vars: set = set()  # [wrapper-only] vars asserted by test-step expected results (outputs)
        for step in tc["steps"]:  # [wrapper-only] aggregate action/expected vars across all steps
            action_vars.update(step["act"])  # [wrapper-only] collect this step's action vars
            expected_vars.update(step["exp"])  # [wrapper-only] collect this step's expected-result vars

        # [wrapper-only] trace the classified var sets for this TC
        logger.debug(
            "M1: TC%d precond_vars=%s action_vars=%s expected_vars=%s",
            i + 1, precond_vars, action_vars, expected_vars,
        )

        # Fixed vars (SystemConstant / Literal / dT) must NOT be DRIVEN as an input (action) nor VERIFIED
        # as an output (expected) — they are compile-time constants the weakspot never computes/alters.
        # They MAY, however, be DECLARED in a Pre-Condition (a test legitimately states a fixed const's
        # config value, e.g. SY_TWAN), so precond_vars is EXCLUDED from this check (aligns with Prompt 1
        # and lets required_preconds carry SystemConstants).
        checks_total += 1  # [wrapper-only] count the fixed-vars check
        bad_fixed = fixed_vars & (action_vars | expected_vars)  # [wrapper-only] fixed const DRIVEN(action) or VERIFIED(expected) — precond allowed
        if not bad_fixed:  # [wrapper-only] RULE satisfied: no fixed const driven/verified
            checks_passed += 1  # [wrapper-only] pass
            logger.debug("M1: TC%d fixed-vars check PASS (no fixed var used as I/O)", i + 1)  # [wrapper-only] trace
        else:  # [wrapper-only] RULE violated: a compile-time const appears as an input/output
            details.append(f"TC{i+1}: fixed vars used as input/output: {list(bad_fixed)[:3]}")  # [wrapper-only] fail detail (first 3)
            logger.debug("M1: TC%d fixed-vars check FAIL: %s", i + 1, bad_fixed)  # [wrapper-only] trace

        # NOTE: action↔expected disjointness is a ROLE-AGNOSTIC structural check — it lives in M3
        # (act/exp-overlap, per step). Removed from M1 to avoid double-counting; M1 is now output-role only.

        # ── Output-role checks (only when we know the weakspot's real outputs) ──
        if output_set:  # [wrapper-only] skip output-role checks when no declared outputs (can't judge role)
            # (c) The test must VERIFY a real weakspot output — at least one Expected-Results
            #     var must be a declared output signal. Catches tests that check fabricated /
            #     non-output names (e.g. 'bit_out', 'out') instead of the real signal. Lenient:
            #     a single real-output check passes.
            # [wrapper-only] weight-on-failure (M1_IO_CHECK_WEIGHT): experts flag "wrong output" as a major fault
            if expected_vars & output_set:  # [wrapper-only] RULE satisfied: an expected var is a real declared output
                checks_total += 1  # [wrapper-only] pass: one normal check (no inflation)
                checks_passed += 1  # [wrapper-only] pass
                logger.debug("M1: TC%d output-verified check PASS", i + 1)  # [wrapper-only] trace
            else:  # [wrapper-only] RULE violated: TC verifies no real weakspot output
                checks_total += M1_IO_CHECK_WEIGHT  # [wrapper-only] fail: counts as W failed checks (bites W× harder)
                # [wrapper-only] fail detail: expected declared outputs vs what the TC actually checked
                details.append(
                    f"TC{i+1}: Expected Results verify no real output {sorted(output_set)[:2]} "
                    f"(weighs {M1_IO_CHECK_WEIGHT}x); got {list(expected_vars)[:3]}"
                )
                logger.debug("M1: TC%d output-verified check FAIL (weighs %dx, expected=%s)", i + 1, M1_IO_CHECK_WEIGHT, expected_vars)  # [wrapper-only] trace

            # (d) A weakspot OUTPUT must not be SET as an input (precond/action). Weight-on-failure (experts flag this).
            out_as_input = output_set & (precond_vars | action_vars)  # [wrapper-only] declared outputs wrongly driven as inputs
            if not out_as_input:  # [wrapper-only] RULE satisfied: no output signal is set as an input
                checks_total += 1  # [wrapper-only] pass: one normal check (no inflation)
                checks_passed += 1  # [wrapper-only] pass
                logger.debug("M1: TC%d output-not-set-as-input check PASS", i + 1)  # [wrapper-only] trace
            else:  # [wrapper-only] RULE violated: an output signal is pinned/driven as an input
                checks_total += M1_IO_CHECK_WEIGHT  # [wrapper-only] fail: counts as W failed checks (bites W× harder)
                details.append(f"TC{i+1}: output signal set as input (weighs {M1_IO_CHECK_WEIGHT}x): {list(out_as_input)[:3]}")  # [wrapper-only] fail detail (first 3)
                logger.debug("M1: TC%d output-not-set-as-input check FAIL (weighs %dx): %s", i + 1, M1_IO_CHECK_WEIGHT, out_as_input)  # [wrapper-only] trace

    score = checks_passed / checks_total if checks_total > 0 else 0.5  # [wrapper-only] M1 = fraction of I/O-role checks passed (neutral 0.5 if nothing to check)
    logger.info("M1: RESULT score=%.4f (%d/%d checks passed)", score, checks_passed, checks_total)  # [wrapper-only] trace
    if details:  # [wrapper-only] only log the failure list when something failed
        logger.info("M1: failed checks: %s", "; ".join(details))  # [wrapper-only] trace all failures
    return score, "; ".join(details[:5]) if details else "I/O classification correct"  # [wrapper-only] score + feedback (first 5 failures, or success msg)


# ═══════════════════════════════════════════════════════════════════
# M2: SIGNAL RESOLUTION (string lookup, schema-agnostic)
# ═══════════════════════════════════════════════════════════════════
def _m2_signal_resolution(   # [wrapper-only] M2 metric (signal-name resolution + input/precond completeness) — no run.py analog
    test_cases_str: str,
    signalsheet_data: str,
    model_interface_data: str,
    equation_json: str,
    required_io: Optional[dict] = None,  # [wrapper-only] expert-curated required_io KB {required_inputs, required_preconds} — enables completeness
) -> Tuple[float, str]:
    """M2 = signal-name validity, and (when an expert required_io KB entry exists) input/precond
    COMPLETENESS. name_validity = fraction of DISTINCT used names that resolve to a real signal.
    completeness = fraction of KB-required inputs (driven in precond∪actions) + preconds (in precond)
    that are present. When a KB entry exists: M2 = (name_validity + completeness) / 2; else M2 =
    name_validity (unchanged). The MISSING required inputs/preconds are appended to the feedback string."""
    logger.info("M2: ===== Signal Resolution (final JSON) =====")  # [wrapper-only] trace banner
    tcs, err = _normalize_test_cases(test_cases_str)  # [wrapper-only] parse final-JSON test cases into canonical {precond, steps[act/exp]} shape
    if err:  # [wrapper-only] test-case JSON was unparseable
        logger.error("M2: parse error -> score=0.0 (%s)", err)  # [wrapper-only] trace
        return 0.0, f"Cannot parse JSON: {err}"  # [wrapper-only] hard fail: cannot resolve signals in unparseable JSON

    # Build set of known signal names — STRUCTURED parse (mirrors the original
    # signal-name source: signalsheet is indexed on its "signal" column
    # [input_loader.set_index('signal')]; model_interface names live in its "Name"
    # column). The previous naive split(",")[0]/[1] included the CSV HEADER row and
    # arbitrary columns (e.g. the data-type column "sint32"), over-crediting non-signal
    # tokens as if they were valid signal names.
    import csv as _csv  # [wrapper-only] structured CSV parse (replaces brittle split(",")[0])
    import io as _io  # [wrapper-only] wrap CSV text as a file object for DictReader

    known_names: set = set()  # [wrapper-only] namespace of every valid signal name (signalsheet + model_interface + equation)

    def _csv_rows(text: str):  # [wrapper-only] parse CSV text -> list of column-keyed row dicts
        try:  # [wrapper-only] tolerate malformed/empty CSV
            return list(_csv.DictReader(_io.StringIO(text or "")))  # [wrapper-only] header-aware rows (mirrors input_loader.py:79 signalsheet column semantics)
        except Exception:  # [wrapper-only] any CSV error -> treat as no rows
            return []  # [wrapper-only] empty namespace contribution

    # signalsheet names come from its "signal" column (mirrors input_loader.py:79 df.set_index("signal"))
    for _row in _csv_rows(signalsheet_data):  # [wrapper-only] iterate signalsheet rows
        name = _row.get("signal") or _row.get("Signal")  # [wrapper-only] read the "signal" column (case variants)
        if not name and _row:  # [wrapper-only] no "signal" column present but row has data
            name = next(iter(_row.values()), "")  # fallback: first column  # [wrapper-only] use first column value as the name
        if name:  # [wrapper-only] only add non-empty names
            known_names.add(str(name).strip())  # [wrapper-only] register signalsheet signal name (whitespace-trimmed)

    # model_interface names come from its "Name" column (mirrors input_loader.py:24/209 item["Name"])
    for _row in _csv_rows(model_interface_data):  # [wrapper-only] iterate model_interface rows
        for _col in ("Name", "name"):  # [wrapper-only] try both header-case variants
            if _row.get(_col):  # [wrapper-only] this row has a Name value
                known_names.add(str(_row[_col]).strip())  # [wrapper-only] register model_interface signal name
                break  # [wrapper-only] first matching column wins

    try:  # [wrapper-only] equation JSON is the third signal-name source
        equation = json.loads(equation_json)  # [wrapper-only] parse the equation map (op_key -> op dict)
        for key, elem in equation.items():  # [wrapper-only] iterate every operation entry
            if not isinstance(elem, dict):  # [wrapper-only] skip non-dict values (metadata)
                continue  # [wrapper-only] next entry
            inst = elem.get("instance_name", "")  # [wrapper-only] op instance name is a valid signal (mirrors run.py:378 instance_name usage)
            if inst:  # [wrapper-only] only add non-empty instance names
                known_names.add(inst)  # [wrapper-only] register op instance name
            # Pull EVERY signal-bearing field of every op so the namespace is complete.
            for _fld in ("unique_variables", "unique_params", "unique_consts",  # [wrapper-only] all signal-bearing op fields
                         "unique_class_items"):  # [wrapper-only] (continued field list)
                for nm in elem.get(_fld, []) or []:  # [wrapper-only] each name listed in that field
                    known_names.add(nm)  # [wrapper-only] register variable/param/const/class-item name
    except (json.JSONDecodeError, TypeError):  # [wrapper-only] equation JSON missing/malformed
        pass  # [wrapper-only] equation contributes no names; signalsheet/model_interface still count

    known_names.discard("")  # [wrapper-only] drop the empty-string name if any slipped in

    # ROBUST RESOLUTION (not brittle exact-string match): resolve case- and
    # whitespace-insensitively, so a real signal written with trivial format differences
    # still resolves. Genuine fabrications ('out', 'bit_out', 'return') are NOT in the
    # namespace under any normalization, so they still correctly fail.
    def _norm(s: str) -> str:  # [wrapper-only] normalization key: strip all whitespace, lowercase
        return re.sub(r"\s+", "", str(s)).lower()  # [wrapper-only] case/whitespace-insensitive form for robust matching
    known_norm = {_norm(n) for n in known_names}  # [wrapper-only] pre-normalize the whole namespace for O(1) lookup

    # Generic placeholder/output names the LLM invents instead of a real signal.
    abstract_patterns = [r"^\$op\d+", r"^signal$", r"^impl_cast$",  # [wrapper-only] regex list flagging invented placeholder names
                         r"^input$", r"^output$", r"^out$", r"^return$", r"^result$"]  # [wrapper-only] (continued placeholder patterns)

    logger.debug("M2: known_names count=%d (signalsheet+model_interface+equation)", len(known_names))  # [wrapper-only] trace namespace size

    details = []  # [wrapper-only] per-variable failure notes (Abstract/Unknown)
    total_vars = 0  # [wrapper-only] denominator: every variable name checked
    resolved_vars = 0  # [wrapper-only] numerator: names found in the namespace

    # [wrapper-only] Collect every UNIQUE variable name across the WHOLE test suite (occurrence-independent):
    # a signal used in many steps/TCs is counted ONCE. This makes M2 fair — one fabricated name costs the
    # same as any other distinct name (1/#unique), instead of being diluted by repeated valid names
    # (the old per-occurrence denominator under-penalized a rare fabrication buried among many repeats).
    seen_norm: set = set()          # [wrapper-only] normalized names already counted (global dedup key)
    unique_names: List[str] = []    # [wrapper-only] first-seen spelling of each distinct name (for the trace)
    driven_norm: set = set()        # [wrapper-only] normalized precond∪action signal names (for input-completeness)
    precond_norm: set = set()       # [wrapper-only] normalized precond signal names (for precond-completeness)
    for ti, tc in enumerate(tcs):  # [wrapper-only] scan every test case
        all_names = list(tc["precond"])  # [wrapper-only] precondition signal names
        for _pk in tc["precond"]:  # [wrapper-only] record precond signals (both driven and precond sets)
            precond_norm.add(_norm(_pk)); driven_norm.add(_norm(_pk))  # [wrapper-only]
        for step in tc["steps"]:  # [wrapper-only] then each test step
            all_names.extend(step["act"])  # [wrapper-only] action (input) signal names
            for _ak in step["act"]:  # [wrapper-only] record action signals (driven set only)
                driven_norm.add(_norm(_ak))  # [wrapper-only]
            all_names.extend(step["exp"])  # [wrapper-only] expected-result (output) signal names
        for var_name in all_names:  # [wrapper-only] register each name once (by normalized key)
            if not var_name:  # [wrapper-only] skip empty names
                continue  # [wrapper-only] next name
            key = _norm(var_name)  # [wrapper-only] case/whitespace-insensitive dedup key
            if key in seen_norm:  # [wrapper-only] already counted this distinct signal
                continue  # [wrapper-only] don't count occurrences
            seen_norm.add(key)  # [wrapper-only] mark this distinct name counted
            unique_names.append(var_name)  # [wrapper-only] keep first spelling for the trace

    for var_name in unique_names:  # [wrapper-only] resolve each DISTINCT name exactly once
        total_vars += 1  # [wrapper-only] denominator = # of DISTINCT names (unique, not occurrences)
        # name-resolution rule 1: an invented placeholder never resolves (counts as unresolved)
        if any(re.match(pat, var_name.strip(), re.IGNORECASE) for pat in abstract_patterns):  # [wrapper-only] matches a placeholder pattern
            details.append(f"Abstract: {var_name}")  # [wrapper-only] record placeholder failure
            logger.debug("M2: var=%r -> ABSTRACT (generic placeholder, not a real signal)", var_name)  # [wrapper-only] trace
            continue  # [wrapper-only] not counted as resolved
        # name-resolution rule 2: normalized name present in namespace -> resolved
        if _norm(var_name) in known_norm:  # [wrapper-only] case/whitespace-insensitive membership test
            resolved_vars += 1  # [wrapper-only] count as resolved (numerator)
            logger.debug("M2: var=%r -> RESOLVED (normalized match in namespace)", var_name)  # [wrapper-only] trace
        else:  # [wrapper-only] name-resolution rule 3: no normalized match -> fabricated/unknown signal
            details.append(f"Unknown: {var_name}")  # [wrapper-only] record unknown-signal failure
            logger.debug("M2: var=%r -> UNKNOWN (not in signalsheet/model_interface/equation)", var_name)  # [wrapper-only] trace

    name_validity = resolved_vars / total_vars if total_vars > 0 else 0.0  # [wrapper-only] M2 base: fraction of DISTINCT names that resolve
    score = name_validity  # [wrapper-only] default when there is no required_io KB entry (unchanged behavior)

    # [wrapper-only] INPUT/PRECOND COMPLETENESS (expert-curated required_io KB). Only runs when a curated
    # entry exists with ≥1 required item; else M2 stays pure name-validity (no penalty on uncurated weakspots).
    if required_io:  # [wrapper-only] curated KB entry present
        req_in = [str(r).strip() for r in (required_io.get("required_inputs") or []) if str(r).strip()]  # [wrapper-only] required driven inputs
        req_pc = [str(r).strip() for r in (required_io.get("required_preconds") or []) if str(r).strip()]  # [wrapper-only] required preconditions
        req_total = len(req_in) + len(req_pc)  # [wrapper-only] completeness denominator
        if req_total > 0:  # [wrapper-only] only when the KB actually lists requirements
            missing_in = [r for r in req_in if _norm(r) not in driven_norm]  # [wrapper-only] required inputs NOT driven (precond∪actions)
            missing_pc = [r for r in req_pc if _norm(r) not in precond_norm]  # [wrapper-only] required preconds NOT set in precondition
            present = (len(req_in) - len(missing_in)) + (len(req_pc) - len(missing_pc))  # [wrapper-only] completeness numerator
            completeness = present / req_total  # [wrapper-only] fraction of required I/O present in the final test
            score = (name_validity + completeness) / 2.0  # [wrapper-only] BLEND (average) of validity + completeness
            logger.info("M2: input/precond completeness = %d/%d = %.4f  (name_validity=%.4f -> M2=%.4f)",  # [wrapper-only] trace
                         present, req_total, completeness, name_validity, score)
            if missing_in:  # [wrapper-only] surface missing inputs as feedback (GEPA + BU)
                details.append(f"MISSING required inputs: {missing_in}")  # [wrapper-only]
            if missing_pc:  # [wrapper-only] surface missing preconds as feedback
                details.append(f"MISSING required preconds: {missing_pc}")  # [wrapper-only]

    _completeness_applied = bool(required_io) and (len(required_io.get("required_inputs") or []) + len(required_io.get("required_preconds") or []) > 0)  # [wrapper-only] only when the KB actually lists inputs/preconds
    logger.info("M2: RESULT score=%.4f (%d/%d UNIQUE vars resolved%s)", score, resolved_vars, total_vars,  # [wrapper-only] trace result
                 "; +completeness" if _completeness_applied else "")  # [wrapper-only] label completeness ONLY when it ran
    if details:  # [wrapper-only] only log the failure list when something failed
        logger.info("M2: unresolved/abstract/missing: %s", "; ".join(details))  # [wrapper-only] trace all failures
    return score, "; ".join(details[:6]) if details else "All signals resolved"  # [wrapper-only] score + feedback (first 6 items, or success msg)


# ═══════════════════════════════════════════════════════════════════
# M4: BRANCH COVERAGE (equation-graph, schema-agnostic)
# ═══════════════════════════════════════════════════════════════════
def _m4_branch_coverage(   # [wrapper-only] M4 rule-based fallback (now unused) — no run.py analog
    test_cases_str: str,
    equation_json: str,
) -> Tuple[float, str]:
    """Check if test cases cover branches in the equation graph.

    Rule-based heuristic: count MUX/IfThenElse/comparison operations,
    compare against number of unique input value combinations.
    """
    logger.info("M4: ----- branch coverage fallback -----")  # [wrapper-only] trace
    tcs, err = _normalize_test_cases(test_cases_str)  # [wrapper-only] parse Prompt-2 output into canonical test cases
    if err:  # [wrapper-only] broken JSON -> can't score
        logger.error("M4(branch): parse error -> score=0.0 (%s)", err)  # [wrapper-only] trace
        return 0.0, f"Cannot parse JSON: {err}"  # [wrapper-only] parse failure scores 0.0

    try:  # [wrapper-only] equation map is optional here
        equation = json.loads(equation_json)  # [wrapper-only] parse equation graph JSON
    except (json.JSONDecodeError, TypeError):  # [wrapper-only] tolerate missing/malformed equation
        equation = {}  # [wrapper-only] fall back to empty graph

    # Find branch operations
    branch_ops = []  # [wrapper-only] collect (key, fname) of branching nodes in the equation graph
    for key, elem in equation.items():  # [wrapper-only] scan every equation node
        if not isinstance(elem, dict):  # [wrapper-only] skip non-dict entries
            continue  # [wrapper-only] next node
        etype = elem.get("type", "")  # [wrapper-only] node kind (Operation/Function/…)
        fname = (elem.get("function_name") or elem.get("class_method_name") or elem.get("method_name") or "")  # [wrapper-only] the node's function/method name
        if etype in ("Operation", "Function"):  # [wrapper-only] only Operation/Function nodes can branch
            # [wrapper-only] treat MUX/IfThenElse/comparison/boolean ops as branch points
            if any(op in fname for op in ["MUX", "IfThenElse", "If", "Equal", "NotEqual",
                                          "GreaterOrEqual", "Smaller", "Greater",
                                          "LessOrEqual", "And", "Or", "Not"]):
                branch_ops.append((key, fname))  # [wrapper-only] record this branch point

    logger.debug("M4(branch): branch_ops found = %s", branch_ops)  # [wrapper-only] trace

    if not branch_ops:  # [wrapper-only] no branches in graph
        logger.info("M4(branch): RESULT score=1.0 (no branch ops -> trivially covered)")  # [wrapper-only] trace
        return 1.0, "No branch operations found — trivially covered"  # [wrapper-only] nothing to cover -> full credit

    n_branches = len(branch_ops)  # [wrapper-only] number of branch points = denominator

    # Collect unique input value fingerprints per test case
    value_sets = []  # [wrapper-only] one input-value dict per test case
    for tc in tcs:  # [wrapper-only] each test case
        values: Dict[str, str] = dict(tc["precond"])  # [wrapper-only] seed with precondition inputs
        for step in tc["steps"]:  # [wrapper-only] add each step's action inputs
            values.update(step["act"])  # [wrapper-only] merge action (input) signal values
        value_sets.append(values)  # [wrapper-only] store this TC's input fingerprint

    for i, vs in enumerate(value_sets):  # [wrapper-only] trace loop
        logger.debug("M4(branch): TC%d input fingerprint = %s", i + 1, vs)  # [wrapper-only] trace

    unique_combos = len(set(json.dumps(vs, sort_keys=True) for vs in value_sets))  # [wrapper-only] count distinct input combinations (JSON-canonicalized)
    coverage_ratio = min(1.0, unique_combos / max(1, n_branches))  # [wrapper-only] heuristic: distinct combos / branch points, capped at 1.0
    details = f"{unique_combos} unique input combos for {n_branches} branch points"  # [wrapper-only] human-readable detail
    logger.info("M4(branch): RESULT score=%.4f (%s)", coverage_ratio, details)  # [wrapper-only] trace result
    return coverage_ratio, details  # [wrapper-only] score + detail


# ═══════════════════════════════════════════════════════════════════
# M4 (KB path): LLM-as-judge helpers
# ═══════════════════════════════════════════════════════════════════

def _get_judge_predictor():   # [wrapper-only] M4 LLM coverage judge — no run.py analog
    """Return a lazily initialised, thread-safe dspy.Predict(TestCoverageJudge)."""
    global _judge_predictor  # [wrapper-only] module-level singleton for the KB-coverage judge
    if _judge_predictor is None:  # [wrapper-only] fast path — already built
        with _judge_predictor_lock:  # [wrapper-only] serialize first-time construction across threads
            if _judge_predictor is None:  # [wrapper-only] double-checked lock (still unset inside lock)
                from signatures.mbd import TestCoverageJudge  # lazy — avoids circular import
                _jp = dspy.Predict(TestCoverageJudge)  # [wrapper-only] build the LLM coverage-grader predictor once
                # DEDICATED judge LM: M4 is always graded by a FIXED model (default azure/gpt-5.2),
                # INDEPENDENT of the inference LM, so coverage scores are comparable across inference
                # models (e.g. gpt-4.1-mini vs gpt-5-mini). Falls back to the global LM if it can't build.
                try:  # [wrapper-only] best-effort — never break M4 if the judge LM can't be constructed
                    from config import create_lm as _create_lm  # [wrapper-only] repo LM builder (Azure endpoint/token)
                    _judge_model = os.getenv("M4_JUDGE_MODEL_NAME", "azure/gpt-5.2") or "azure/gpt-5.2"  # [wrapper-only] fixed judge model
                    _jp.set_lm(_create_lm(_judge_model))  # [wrapper-only] pin the judge to gpt-5.2 (gpt-5 -> temp auto 1.0)
                    logger.info("M4: judge LM dedicated to %s (independent of the inference LM)", _judge_model)  # [wrapper-only] trace
                except Exception as _e:  # noqa: BLE001  # [wrapper-only] fall back to global LM on any failure
                    logger.warning("M4: could not build dedicated judge LM -> using global inference LM (%s)", _e)  # [wrapper-only] trace
                _judge_predictor = _jp  # [wrapper-only] publish the singleton last (double-checked lock)
    return _judge_predictor  # [wrapper-only] shared predictor instance


def _parse_covered_indices(covered_str: str, total: int) -> set:   # [wrapper-only] M4 judge-output parse — no run.py analog
    """Parse '1,3,4' → {1, 3, 4}, filtering out-of-range numbers."""
    nums = set()  # [wrapper-only] accumulate valid 1-based guideline indices from judge output
    for tok in re.findall(r"\b(\d+)\b", covered_str):  # [wrapper-only] pull every integer token from the judge's covered_indices string
        n = int(tok)  # [wrapper-only] to int
        if 1 <= n <= total:  # [wrapper-only] keep only indices within the guideline count (drops hallucinated numbers)
            nums.add(n)  # [wrapper-only] record covered guideline index
    return nums  # [wrapper-only] set of covered 1-based indices


def _m4_test_coverage(   # [wrapper-only] M4 metric (KB-guidelines LLM judge) — no run.py analog
    test_cases_str: str,
    equation_json: str,
    weakspot_op_key: str,
) -> Tuple[float, str]:
    """M4 Test Coverage: KB-guidelines LLM-judge ONLY (no branch-coverage fallback).

    When ``weakspot_guidelines.json`` has guidelines for the block type:
      1. Look up the block's guidelines from the KB.
      2. Extract test case explanations via ``_normalize_test_cases()``.
      3. Call TestCoverageJudge to determine which guidelines are covered.
      4. Return score = covered / total, plus a detail string.

    The old rule-based ``_m4_branch_coverage`` fallback was REMOVED — it awarded a free
    1.0 to function-call/no-branch blocks (no real coverage judgment). Instead:
      - No KB guidelines (block not in KB / empty) -> 0.5 NEUTRAL ("coverage not judgeable").
      - No explanation text / LLM judge error      -> 0.5 NEUTRAL.
      - Test-case parse error                       -> 0.0 (broken output, like M1/M2/M3).
    So M4 is either a REAL KB-judged ratio, a neutral 0.5 when un-judgeable, or 0.0 if broken.
    """
    logger.info("M4: ===== Test Coverage (weakspot_op_key=%s, KB_LOADED=%s) =====", weakspot_op_key, _KB_LOADED)  # [wrapper-only] trace

    # ── Step 0: parse test cases first — broken output -> 0.0 (like M1/M2/M3), ──
    #            regardless of whether the block is in the KB.
    tcs, err = _normalize_test_cases(test_cases_str)  # [wrapper-only] parse Prompt-2 output into canonical test cases
    if err or not tcs:  # [wrapper-only] broken/empty output can't be graded
        logger.warning("M4: parse error (%s) -> score=0.0", err)  # [wrapper-only] trace
        return 0.0, f"[parse-err] cannot parse test cases: {err}"  # [wrapper-only] parse failure scores 0.0 (like M1/M2/M3)

    # ── Step 1: look up block guidelines — NEW KB first, then OLD KB ──────────
    # resolve_guidelines(fn) tries the NEW (BU FCI Excel) KB, then the OLD (HTML) KB, using the
    # SAME find_block_type() suffix-stripping for both (see utils/kb_parser). kb_source is
    # "newKB" / "KB" / None — surfaced in the detail string so a run shows WHICH KB scored M4.
    block_type: Optional[str] = None  # [wrapper-only] resolved KB block type (None until matched)
    guidelines: List[str] = []  # [wrapper-only] KB coverage guidelines for this block
    kb_source: Optional[str] = None  # [wrapper-only] which KB scored it ("newKB"/"KB"/None)

    if _KB_LOADED and weakspot_op_key:  # [wrapper-only] only look up when a KB is loaded and we know the op key
        try:  # [wrapper-only] tolerate malformed equation_json
            eq = json.loads(equation_json) if isinstance(equation_json, str) else equation_json  # [wrapper-only] accept JSON string or already-parsed dict
            op_entry = eq.get(weakspot_op_key, {}) if isinstance(eq, dict) else {}  # [wrapper-only] the weakspot op's node in the equation graph
            fn = (  # [wrapper-only] resolve the op's function/method name — the KB lookup key
                op_entry.get("function_name")
                or op_entry.get("class_method_name")
                or op_entry.get("method_name")
                or ""
            )
            logger.debug("M4: weakspot op_entry function_name/class_method_name/method_name = %r", fn)  # [wrapper-only] trace
            if fn:  # [wrapper-only] only resolve guidelines when a fn name exists
                kb_source, block_type, guidelines = _resolve_guidelines(fn)  # NEW -> OLD -> (None,None,[])
                logger.debug("M4: resolve_guidelines(%r) -> source=%s block_type=%r (%d guideline(s))",  # [wrapper-only] trace
                             fn, kb_source, block_type, len(guidelines or []))
        except (json.JSONDecodeError, TypeError, AttributeError) as e:  # [wrapper-only] KB lookup best-effort — never crash the metric
            logger.warning("M4: KB lookup failed: %s", e)  # [wrapper-only] trace (leaves guidelines empty -> neutral 0.5 below)

    if not guidelines:
        # No KB coverage spec for this block (not in KB, or empty guidelines). Coverage is
        # NOT judgeable — return a NEUTRAL 0.5 rather than the old branch-coverage heuristic
        # (which awarded a free 1.0 to function-call/no-branch blocks). No fabricated score.
        kb_state = "no-kb" if _KB_LOADED else "kb-missing"  # [wrapper-only] distinguish "KB loaded but block absent" from "no KB at all"
        logger.info("M4: no KB guidelines (%s) -> neutral 0.5 (coverage not judgeable)", kb_state)  # [wrapper-only] trace
        return 0.5, f"[{kb_state}] coverage not judgeable — no KB spec (neutral 0.5)"  # [wrapper-only] no spec -> neutral, not a fabricated score

    # ── Step 2: build the FULL test-case text for the judge (tcs parsed in Step 0) ─
    # Pass the ACTUAL test content — Pre-Conditions + ordered Steps (Actions/Wait/Expected Results)
    # + the Detailed Explanation — so the judge grades coverage against real stimuli/expected results,
    # not just the prose explanation (previously only the explanation was sent).
    def _fmt_tc(idx: int, tc: Dict) -> str:  # [wrapper-only] render one canonical test case for the judge
        lines = [f"TC{idx}:"]  # [wrapper-only] TC header
        _pc = tc.get("precond", {})  # [wrapper-only] preconditions
        if _pc:
            lines.append(f"  Pre-Condition: {_pc}")  # [wrapper-only]
        for _si, _st in enumerate(tc.get("steps", []), 1):  # [wrapper-only] ordered steps
            lines.append(f"  Step {_si}: Wait={_st.get('wait')}, Actions={_st.get('act', {})}, Expected Results={_st.get('exp', {})}")  # [wrapper-only]
        _expl = (tc.get("explanation") or "").strip()  # [wrapper-only] detailed explanation
        if _expl:
            lines.append(f"  Explanation: {_expl}")  # [wrapper-only]
        return "\n".join(lines)  # [wrapper-only]

    tc_text = "\n".join(_fmt_tc(i + 1, tc) for i, tc in enumerate(tcs))  # [wrapper-only] FULL test cases = judge input
    if not tc_text.strip():  # [wrapper-only] nothing to grade (shouldn't happen; Step 0 ensured tcs non-empty)
        logger.info("M4: no test-case content -> neutral 0.5 (judge cannot operate)")  # [wrapper-only] trace
        return 0.5, "[no-tc] coverage not judgeable — no test-case content (neutral 0.5)"  # [wrapper-only] neutral

    # ── Step 3: build inputs for LLM judge ───────────────────────────────────
    guidelines_text = "\n".join(f"{i + 1}: {g}" for i, g in enumerate(guidelines))  # [wrapper-only] number KB guidelines 1..N for the judge prompt
    total = len(guidelines)  # [wrapper-only] denominator = guideline count
    logger.debug("M4: guidelines_text=\n%s", guidelines_text)  # [wrapper-only] trace
    logger.debug("M4: full test-case text for judge=\n%s", tc_text)  # [wrapper-only] trace

    # ── Step 4: call LLM judge (MEMOIZED for determinism) ────────────────────
    # Keyed by the exact judge inputs. A cache hit reuses the prior parsed result so GEPA's
    # paired module-score / feedback calls on the SAME prediction always agree (see _JUDGE_MISS
    # note above). Cache the neutral-0.5 outcome too (sentinel None) so a transient judge failure
    # between the two paired calls cannot make one path 0.5 and the other a real ratio.
    cache_key = (block_type, guidelines_text, tc_text)  # [wrapper-only] memo key = exact judge inputs (deterministic reuse)
    with _judge_result_lock:  # [wrapper-only] guard shared cache across threads
        _cached = _judge_result_cache.get(cache_key, _JUDGE_MISS)  # [wrapper-only] look up prior parsed result (sentinel _JUDGE_MISS if absent)

    if _cached is not _JUDGE_MISS:  # [wrapper-only] cache HIT — reuse without an LLM call
        if _cached is None:  # [wrapper-only] None sentinel = judge previously failed for this input
            logger.info("M4: judge cache HIT -> neutral 0.5 (judge previously unavailable for this input)")  # [wrapper-only] trace
            return 0.5, "[judge-err] coverage not judgeable — judge unavailable (cached neutral 0.5)"  # [wrapper-only] cached neutral keeps paired GEPA calls consistent
        covered = _cached  # [wrapper-only] reuse cached covered-index set
        _reasoning = _judge_reasoning_cache.get(cache_key, "")  # [wrapper-only] reuse cached judge reasoning
        logger.info("M4: judge cache HIT -> covered=%s (deterministic reuse, no LLM call)", covered)  # [wrapper-only] trace
    else:  # [wrapper-only] cache MISS — call the LLM judge
        try:  # [wrapper-only] judge call is best-effort -> neutral 0.5 on any failure
            logger.info("M4: calling TestCoverageJudge LLM (block_type=%s, %d guideline(s), %d test case(s))",  # [wrapper-only] trace
                         block_type, total, len(tcs))
            judge = _get_judge_predictor()  # [wrapper-only] lazily-built shared coverage-grader predictor
            judge_result = judge(  # [wrapper-only] ask the LLM which guidelines the full test cases cover
                block_type=block_type,
                guidelines=guidelines_text,
                test_cases=tc_text,
            )
            logger.debug("M4: judge raw covered_indices=%r", judge_result.covered_indices)  # [wrapper-only] trace raw judge output
            covered = _parse_covered_indices(judge_result.covered_indices, total)  # [wrapper-only] parse "1,3,4" -> valid index set
            _reasoning = (getattr(judge_result, "coverage_reasoning", "") or "").strip()  # [wrapper-only] capture the judge's per-guideline explanation
            logger.debug("M4: judge parsed covered set = %s", covered)  # [wrapper-only] trace
            logger.debug("M4: judge coverage_reasoning = %s", _reasoning)  # [wrapper-only] trace
            with _judge_result_lock:  # [wrapper-only] guard cache write
                _judge_result_cache[cache_key] = covered  # [wrapper-only] memoize parsed result for deterministic reuse
                _judge_reasoning_cache[cache_key] = _reasoning  # [wrapper-only] memoize the reasoning too (paired with covered)
        except Exception as e:  # [wrapper-only] any judge/LLM failure -> neutral 0.5 (cached)
            # LLM judge call failed — coverage not judgeable this run -> neutral 0.5.
            # Cache the neutral outcome (None) so GEPA's paired calls on the SAME prediction
            # can't diverge (one failing -> 0.5, one succeeding -> real ratio).
            with _judge_result_lock:  # [wrapper-only] guard cache write
                _judge_result_cache[cache_key] = None  # [wrapper-only] cache None so paired GEPA calls can't diverge (0.5 vs real ratio)
            logger.warning("M4: LLM judge call failed (%s) -> neutral 0.5 (cached)", e)  # [wrapper-only] trace
            return 0.5, f"[judge-err] coverage not judgeable — judge failed: {e} (neutral 0.5)"  # [wrapper-only] judge error -> neutral

    # ── Step 5: score and detail ─────────────────────────────────────────────
    score = len(covered) / total if total > 0 else 1.0  # [wrapper-only] M4 = covered guidelines / total (1.0 if no guidelines)
    missing_indices = sorted(set(range(1, total + 1)) - covered)  # [wrapper-only] guideline indices NOT covered
    missing_texts = [f'"{guidelines[i - 1][:60]}"' for i in missing_indices[:3]]  # [wrapper-only] first 3 missing guideline texts (truncated) for feedback
    missing_str = (", ".join(missing_texts) + (" …" if len(missing_indices) > 3 else "")) if missing_texts else "none"  # [wrapper-only] join + "…" if more than 3 missing
    detail = f"[{kb_source or 'KB'}:{block_type}] {len(covered)}/{total} covered | missing: {missing_str}"  # [wrapper-only] detail names WHICH KB scored + coverage ratio
    if _reasoning:  # [wrapper-only] append the judge's per-guideline reasoning -> flows into M4 detail AND GEPA feedback (mbd_metric_with_feedback uses m4_details)
        detail = f"{detail}\n    JUDGE REASONING: {_reasoning}"  # [wrapper-only] the judge's explanation of what each TC covers / what's missing and why
    logger.info("M4: RESULT score=%.4f %s", score, detail)  # [wrapper-only] trace result
    return score, detail  # [wrapper-only] KB-judged coverage score + detail (incl. judge reasoning)


# ═══════════════════════════════════════════════════════════════════
# M6: BOUNDARY VALUES (schema-agnostic)
# ═══════════════════════════════════════════════════════════════════
def _m6_boundary_values(   # [wrapper-only] M6 metric (boundary values, DISABLED weight 0) — no run.py analog
    test_cases_str: str,
    signalsheet_data: str,
    equation_json: str,
) -> Tuple[float, str]:
    """Check if test values include boundary/edge cases."""
    logger.info("M6: ===== Boundary Values (DISABLED, weight=0, computed for visibility) =====")  # [wrapper-only] trace
    tcs, err = _normalize_test_cases(test_cases_str)  # [wrapper-only] parse Prompt-2 output into canonical test cases
    if err:  # [wrapper-only] broken JSON -> can't score
        logger.error("M6: parse error -> score=0.0 (%s)", err)  # [wrapper-only] trace
        return 0.0, f"Cannot parse JSON: {err}"  # [wrapper-only] parse failure scores 0.0

    # Extract min/max from signalsheet header
    signal_ranges: Dict[str, Dict[str, float]] = {}  # [wrapper-only] signal name -> {min,max} allowed range
    lines = signalsheet_data.split("\n")  # [wrapper-only] split the CSV signalsheet into rows
    header = lines[0].split(",") if lines else []  # [wrapper-only] first row = column headers
    min_col = max_col = -1  # [wrapper-only] column indices for min/max range (-1 = not found)
    for idx, col in enumerate(header):  # [wrapper-only] locate the min-range and max-range columns by header text
        cl = col.strip().lower()  # [wrapper-only] normalize header cell
        if "min" in cl and "range" in cl:  # [wrapper-only] header names the min-range column
            min_col = idx  # [wrapper-only] record min column index
        elif "max" in cl and "range" in cl:  # [wrapper-only] header names the max-range column
            max_col = idx  # [wrapper-only] record max column index

    if min_col >= 0 and max_col >= 0:  # [wrapper-only] only parse ranges if both columns were found
        for line in lines[1:]:  # [wrapper-only] each data row after the header
            parts = line.split(",")  # [wrapper-only] split row into cells
            if len(parts) > max(min_col, max_col):  # [wrapper-only] row has enough columns to read min/max
                name = parts[0].strip()  # [wrapper-only] signal name is first column
                try:  # [wrapper-only] tolerate non-numeric range cells
                    signal_ranges[name] = {  # [wrapper-only] record this signal's numeric min/max range
                        "min": float(parts[min_col].strip()),
                        "max": float(parts[max_col].strip()),
                    }
                except (ValueError, IndexError):  # [wrapper-only] skip signals with unparseable/missing range cells
                    pass  # [wrapper-only] leave signal without a known range

    # Also extract min_max_range from equation_json
    try:  # [wrapper-only] equation-graph ranges are supplementary; tolerate malformed JSON
        equation = json.loads(equation_json)  # [wrapper-only] parse equation graph
        for key, elem in equation.items():  # [wrapper-only] scan every node for a declared min_max_range
            if not isinstance(elem, dict):  # [wrapper-only] skip non-dict entries
                continue  # [wrapper-only] next node
            mmr = elem.get("min_max_range")  # [wrapper-only] node's declared min/max range (if any)
            inst = elem.get("instance_name", "")  # [wrapper-only] the signal instance this node represents
            if inst and isinstance(mmr, dict):  # [wrapper-only] only when both a name and a range dict exist
                try:  # [wrapper-only] tolerate non-numeric range values
                    signal_ranges.setdefault(inst, {  # [wrapper-only] add range only if signalsheet didn't already provide one
                        "min": float(mmr.get("min", mmr.get("Min", 0))),
                        "max": float(mmr.get("max", mmr.get("Max", 0))),
                    })
                except (TypeError, ValueError):  # [wrapper-only] skip unparseable equation ranges
                    pass  # [wrapper-only] leave as-is
    except (json.JSONDecodeError, TypeError):  # [wrapper-only] malformed/absent equation_json
        pass  # [wrapper-only] proceed with signalsheet ranges only

    # Collect all numeric values per variable from canonical test cases
    test_values: Dict[str, set] = {}  # [wrapper-only] signal name -> set of numeric values used across test cases
    for tc in tcs:  # [wrapper-only] each test case
        for name, val_str in tc["precond"].items():  # [wrapper-only] precondition input values
            try:  # [wrapper-only] only numeric values count for boundary checks
                test_values.setdefault(name, set()).add(float(val_str))  # [wrapper-only] record numeric precond value
            except (ValueError, TypeError):  # [wrapper-only] skip non-numeric values (enums/strings)
                pass  # [wrapper-only] ignore
        for step in tc["steps"]:  # [wrapper-only] each step's action inputs
            for name, val_str in step["act"].items():  # [wrapper-only] action (input) values
                try:  # [wrapper-only] only numeric values count
                    test_values.setdefault(name, set()).add(float(val_str))  # [wrapper-only] record numeric action value
                except (ValueError, TypeError):  # [wrapper-only] skip non-numeric values
                    pass  # [wrapper-only] ignore

    logger.debug("M6: signal_ranges (min/max) known = %s", signal_ranges)  # [wrapper-only] trace
    logger.debug("M6: test_values per var = %s", test_values)  # [wrapper-only] trace

    # Score boundary value usage
    boundary_checks = 0  # [wrapper-only] denominator: number of boundary conditions expected
    boundary_hits = 0  # [wrapper-only] numerator: boundary conditions actually exercised

    for var, values in test_values.items():  # [wrapper-only] each signal that appears in the test cases
        if var in signal_ranges:  # [wrapper-only] signal has a known min/max range
            sr = signal_ranges[var]  # [wrapper-only] its {min,max}
            boundary_checks += 2  # [wrapper-only] expect both min and max to be exercised
            hit_min = sr["min"] in values or any(abs(v - sr["min"]) <= 1 for v in values)  # [wrapper-only] a test value at/within 1 of the min
            hit_max = sr["max"] in values or any(abs(v - sr["max"]) <= 1 for v in values)  # [wrapper-only] a test value at/within 1 of the max
            if hit_min:  # [wrapper-only] min boundary exercised
                boundary_hits += 1  # [wrapper-only] credit min hit
            if hit_max:  # [wrapper-only] max boundary exercised
                boundary_hits += 1  # [wrapper-only] credit max hit
            logger.debug("M6: var=%s range=[%s,%s] values=%s -> hit_min=%s hit_max=%s",  # [wrapper-only] trace
                          var, sr["min"], sr["max"], values, hit_min, hit_max)
        else:  # [wrapper-only] no known range -> fall back to a zero/negative edge check
            boundary_checks += 1  # [wrapper-only] expect one edge case (0 or negative)
            hit_neg_or_zero = 0 in values or any(v < 0 for v in values)  # [wrapper-only] any test value is zero or negative
            if hit_neg_or_zero:  # [wrapper-only] edge exercised
                boundary_hits += 1  # [wrapper-only] credit edge hit
            logger.debug("M6: var=%s no known range, values=%s -> hit_zero/neg=%s",  # [wrapper-only] trace
                          var, values, hit_neg_or_zero)

    score = boundary_hits / boundary_checks if boundary_checks > 0 else 0.5  # [wrapper-only] M6 = boundary hits / checks (0.5 neutral if nothing checkable)
    details = f"{boundary_hits}/{boundary_checks} boundary values present"  # [wrapper-only] human-readable detail
    logger.info("M6: RESULT score=%.4f (%s) [weight=0, does not affect composite]", min(1.0, score), details)  # [wrapper-only] trace (weight 0 -> ignored in composite)
    return min(1.0, score), details  # [wrapper-only] capped score + detail (DISABLED: weight 0)


# ═══════════════════════════════════════════════════════════════════
# M7: DLL PASS RATE (real implementation)
# ═══════════════════════════════════════════════════════════════════
def _m7_dll_pass_rate(   # [wrapper-only] M7 metric; its DLL core REUSES run.py _step_4a backbone (546-600) per-step (stateless)
    test_cases_str: str,
    equation_json: str,
    dll_path: str,
    signalsheet_data: str = "",
    model_interface_data: str = "",
    equation_results_json: str = "[]",
    weakspot_op_key: str = "",
) -> Tuple[float, str]:
    """Run test case inputs through the DLL and count equation-level passes.

    Mirrors run.py Phase 4 backbone EXACTLY — per equation in eq_results:
      1. eq_to_solve = assignment_chain.split('\\n')[::-1]  (from precomputed equation_results_json)
      2. simulated_response = step actions formatted as LLM would output
      3. attempt = _attempt_solve(simulated_response, eq_to_solve, fc)
      4. substitution_dict = _build_substitution_dict(attempt.final_assignments)
      5. expr = _substitute_ops(final_substituted_equation, substitution_dict)  (run.py regex)
      6. eq_list = generate_bodmas_chain(expr)
      7. final_check, _ = resolve_equation_block(eq_list, fc)
      8. actual = final_check[-1].split('=')[-1].strip()
      9. expected = equation's target — from exp_dict[sig] (output signal) or combined_act[sig]
         (mirrors run.py line 599: expected = merged_eq.equation_to_solve.split("=")[-1].strip())
    No special fc.functionCall path — same backbone for all weakspot types.
    """
    logger.info("M7: ===== DLL Pass Rate (post-hoc fallback path) =====")  # [wrapper-only] trace banner
    if not dll_path or not os.path.exists(dll_path):       # [wrapper-only] DLL required for M7 (run.py:816-826 always has a live FunctionCaller)
        logger.error("M7: DLL not available at %r -> raising (fail loud, no silent 0.5)", dll_path)  # [wrapper-only] trace
        raise RuntimeError(                                # [wrapper-only] fail loud — never fabricate a DLL pass rate
            f"M7: DLL required but not available at {dll_path!r}. "
            "DLL verification cannot be skipped — fix the 32-bit DLL setup before optimizing."
        )

    tcs, err = _normalize_test_cases(test_cases_str)       # [wrapper-only] parse Prompt-2 test cases (run.py holds them live in eq_ctx)
    if err or not tcs:                                     # [wrapper-only] unparseable → neutral 0.5 (nothing to score)
        logger.error("M7: parse error -> score=0.5 default (%s)", err)  # [wrapper-only] trace
        return 0.5, f"Cannot parse test cases: {err}"     # [wrapper-only] neutral score + detail

    # ── Get precomputed equation results (one entry per dependency equation) ──
    # run.py holds eq_ctx.equation_results live (run.py:430); wrapper JSON-serializes them into the Example.
    try:                                                   # [wrapper-only] tolerate malformed JSON
        eq_results = json.loads(equation_results_json)     # [wrapper-only] deserialize eq_ctx.equation_results (run.py:430 list[EquationResult])
    except (json.JSONDecodeError, TypeError):
        eq_results = []                                    # [wrapper-only] empty on parse failure

    logger.debug("M7: equation_results_json has %d entr(ies)", len(eq_results))  # [wrapper-only] trace
    if not eq_results:                                     # [wrapper-only] no equations to solve → neutral 0.5
        logger.warning("M7: no equation_results_json data -> score=0.5 default")  # [wrapper-only] trace
        return 0.5, "No equation data in equation_results_json — using default 0.5"  # [wrapper-only] neutral score + detail

    # ── Function-call weakspot detection (before any DLL spin-up) ─────────────
    # M7 measures Prompt 4 (per-equation, arithmetic) solving quality. For a
    # function-call weakspot (e.g. Srv_RampSwitch_out) Prompt 4 never runs — there
    # are no arithmetic equations to solve, only the weakspot's own DLL function.
    # The arithmetic backbone below cannot apply (eq_results hold the arg relations,
    # whose original_equation strings carry no "name = value" target). Functional
    # correctness for these weakspots is covered by M8 (stateful DLL execution),
    # so M7 returns a neutral, explicitly-N/A score rather than a misleading 0/0 —
    # and skips the (expensive) 32-bit FunctionCaller subprocess entirely.
    try:                                                   # [wrapper-only] tolerate malformed JSON
        _eq_json_m7 = json.loads(equation_json)            # [wrapper-only] parse equation KB (same source as run.py filtered_dict["equation"])
    except (json.JSONDecodeError, TypeError):
        _eq_json_m7 = {}                                   # [wrapper-only] empty on parse failure
    _ws_entry_m7 = _eq_json_m7.get(weakspot_op_key, {}) if weakspot_op_key else {}  # [wrapper-only] weakspot op entry ($opN) — same lookup as forward():388
    # resolve the op's fn name across the three KB schema variants (as forward():389 / M4:810)
    _ws_fn_m7 = (
        _ws_entry_m7.get("function_name")                  # [wrapper-only] primary schema key
        or _ws_entry_m7.get("class_method_name")           # [wrapper-only] alt schema key
        or _ws_entry_m7.get("method_name")                 # [wrapper-only] alt schema key
        or ""
    )
    if _ws_entry_m7.get("type") == "Function" and _ws_fn_m7 and _ws_entry_m7.get("args"):  # [wrapper-only] function-call weakspot: no arithmetic Prompt-4 to score
        logger.info("M7: weakspot %s is a function call (%s) — Prompt 4 arithmetic solving N/A; "  # [wrapper-only] trace
                     "functional correctness measured by M8. score=0.5 (neutral)",
                     weakspot_op_key, _ws_fn_m7)
        return 0.5, f"N/A — function-call weakspot ({_ws_fn_m7}); see M8 for functional correctness"  # [wrapper-only] neutral N/A (covered by M8), skips 32-bit subprocess

    # ── Ensure MBD pipeline modules are importable ───────────────────────────
    _MBD_ROOT = os.path.normpath(                          # [wrapper-only] locate the original MBD/weakspot_code_to_run_2 tree
        os.path.join(os.path.dirname(__file__), "..", "MBD", "weakspot_code_to_run_2")
    )
    if _MBD_ROOT not in sys.path:                          # [wrapper-only] make run.py's pipeline.* importable
        sys.path.insert(0, _MBD_ROOT)

    # [wrapper-only] one-time import of run.py's DLL modules; run.py imports these at module load
    if "pipeline.function_caller" not in sys.modules or "pipeline.equation_solver_dll" not in sys.modules:
        _repo_cfg = sys.modules.get("config")              # [wrapper-only] stash repo's own config module (name collides with MBD/config.py)
        _spec = importlib.util.spec_from_file_location(    # [wrapper-only] load MBD's config.py under the plain name pipeline.* expects
            "mbd_config_metric", os.path.join(_MBD_ROOT, "config.py")
        )
        _mod = importlib.util.module_from_spec(_spec)      # [wrapper-only] build the MBD config module
        sys.modules["config"] = _mod                       # [wrapper-only] temporarily shadow "config" so pipeline.* imports resolve
        _spec.loader.exec_module(_mod)                     # [wrapper-only] execute MBD/config.py
        try:
            import pipeline.function_caller  # noqa: F401   # [wrapper-only] import run.py's FunctionCaller module (run.py:821 uses it)
            import pipeline.equation_solver_dll  # noqa: F401  # [wrapper-only] import run.py's DLL solve helpers module (run.py:508-600 backbone)
        finally:
            if _repo_cfg is not None:                      # [wrapper-only] restore repo's config to avoid corrupting the wrapper
                sys.modules["config"] = _repo_cfg
            else:
                sys.modules.pop("config", None)            # [wrapper-only] remove the shadow if repo had no config module

    from pipeline.function_caller import FunctionCaller    # [wrapper-only] run.py:821 FunctionCaller class [imported, not reimplemented]
    from pipeline.equation_solver_dll import (             # [wrapper-only] run.py's Phase-4 DLL backbone helpers [imported]
        _attempt_solve, _build_substitution_dict,          # [wrapper-only] run.py:581-582 helpers
        generate_bodmas_chain, resolve_equation_block,     # [wrapper-only] run.py:594/597 helpers
    )

    # ── Initialize FunctionCaller ────────────────────────────────────────────
    with _dll_registry_lock:                               # [wrapper-only] process-wide lock (metric may run under thread pool; run.py is single-run)
        if dll_path not in _dll_caller_registry:           # [wrapper-only] cache one FunctionCaller per DLL (run.py builds it once per pipeline run)
            import csv                                     # [wrapper-only] parse serialized signal-sheet/model-interface CSV
            import io                                      # [wrapper-only] StringIO wrapper for csv.DictReader
            import pandas as pd                            # [wrapper-only] imported for parity with run.py DataFrame handling (unused here)

            # [wrapper-only] derive FCname/work_dir from dll_path layout (run.py:822-823 fcname/work_dir)
            env_libs_dir = os.path.dirname(dll_path)       # [wrapper-only] .../FCname/env_libs
            fc_dir       = os.path.dirname(env_libs_dir)   # [wrapper-only] .../FCname
            fcname       = os.path.basename(fc_dir)        # run.py:822 FCname=fcname
            work_dir     = os.path.dirname(fc_dir)         # run.py:823 work_dir (parent of FCname)

            signal_sheet_list = list(csv.DictReader(io.StringIO(signalsheet_data or "")))  # run.py:817 convert_signal_sheet_to_list(signal_sheet_df) → list-of-dicts
            # model_interface_data: pass load_model_interface()'s LIST-OF-DICTS (run.py:756/825) —
            # FunctionCaller iterates it as rows (row["Name"]/row["Value"], function_caller.py:494-496).
            # A DataFrame would iterate to column names and break CharTable calibration.
            mi_list = list(csv.DictReader(io.StringIO(model_interface_data or "")))  # run.py:825 model_interface_data list-of-dicts

            try:
                logger.info("M7: initializing FunctionCaller FCname=%s work_dir=%s", fcname, work_dir)  # [wrapper-only] trace
                _dll_caller_registry[dll_path] = FunctionCaller(  # run.py:821-826 FunctionCaller(FCname, work_dir, signal_sheet_data, model_interface_data)
                    FCname=fcname,                         # run.py:822 FCname=fcname
                    work_dir=work_dir,                     # run.py:823 work_dir=work_dir
                    signal_sheet_data=signal_sheet_list,   # run.py:824 signal_sheet_data=signal_sheet_data
                    model_interface_data=mi_list,          # run.py:825 model_interface_data=model_interface_data
                )
                logger.info("M7: FunctionCaller initialized successfully")  # [wrapper-only] trace
            except Exception as _fc_err:                   # [wrapper-only] init can fail if 32-bit Python missing → cache None
                logger.error("M7: FunctionCaller init failed: %s", _fc_err)  # [wrapper-only] trace
                _dll_caller_registry[dll_path] = None      # [wrapper-only] sentinel so we don't retry a broken init
        fc = _dll_caller_registry[dll_path]                # [wrapper-only] the cached FunctionCaller (run.py's `function_caller`)

    if fc is None:                                         # [wrapper-only] no working DLL caller → fail loud (never fabricate a pass rate)
        logger.error("M7: fc is None (32-bit Python missing) -> raising (fail loud, no silent 0.5)")  # [wrapper-only] trace
        raise RuntimeError(                                # [wrapper-only] fail loud — DLL verification cannot be skipped
            "M7: FunctionCaller unavailable (32-bit Python missing). "
            "DLL verification cannot be skipped — fix the 32-bit Python setup before optimizing."
        )

    try:                                                   # [wrapper-only] tolerate malformed JSON
        eq_json = json.loads(equation_json)                # [wrapper-only] parse equation KB (used for instance_name→type lookup below)
    except (json.JSONDecodeError, TypeError):
        eq_json = {}                                       # [wrapper-only] empty on parse failure

    # _substitute_ops — EXACT copy from run.py lines 584-588 (bug-for-bug identical)
    def _substitute_ops(equation, subs):                   # run.py:584 def _substitute_ops(equation, subs)
        return re.sub(                                     # run.py:585 return re.sub(
            r"\$op\d+",                                    # run.py:586 r"\$op\d+"
            lambda m: subs.get(m.group(0), m.group(0)),    # run.py:587 lambda m: subs.get(m.group(0), m.group(0))
            equation,                                      # run.py:588 equation
        )

    # Build instance_name → type lookup for formatting action values as LLM would
    # [wrapper-only] run.py never rebuilds this — its Prompt-4 LLM already emits typed lines;
    # here we synthesize that formatting to feed the same _attempt_solve backbone per step.
    inst_to_type: Dict[str, str] = {
        v["instance_name"]: v.get("type", "Variable")      # [wrapper-only] map instance name → its declared type (default Variable)
        for k, v in eq_json.items()                        # [wrapper-only] iterate KB entries
        if isinstance(v, dict) and v.get("instance_name")  # [wrapper-only] only dict entries carrying an instance_name
    }

    passes, total = 0, 0                                   # [wrapper-only] M7 numerator/denominator (run.py computes no score)
    for ti, tc in enumerate(tcs):                          # [wrapper-only] iterate each parsed test case (per-step stateless scoring)
        precond = tc.get("precond", {})                    # [wrapper-only] test-case pre-conditions (applied to every step)
        for si, step in enumerate(tc["steps"]):            # [wrapper-only] iterate steps within the test case
            # Merge pre-conditions + step actions (step actions override)
            combined_act = {**precond, **step["act"]}      # [wrapper-only] effective input state for this step
            exp_dict = step["exp"]                         # [wrapper-only] the step's Expected Results (M7 targets for output signals)

            # Format actions as simulated LLM response (same format as run.py expects)
            # e.g. "Variable(dT) = 100000" or "Parameter(delayTime) = 5000"
            # [wrapper-only] reconstructs what run.py's Prompt-4 LLM would output (run.py:564 simulated_response)
            simulated_response = [
                f"{inst_to_type.get(sig_name, 'Variable')}({sig_name}) = {val_str}"  # [wrapper-only] "Type(name) = value" line per signal
                for sig_name, val_str in combined_act.items()  # [wrapper-only] iterate this step's input assignments
                if val_str not in ("", None) and sig_name in inst_to_type  # [wrapper-only] skip blanks / unknown signals
            ]
            logger.debug("M7: TC%d.S%d simulated_response=%s", ti + 1, si + 1, simulated_response)  # [wrapper-only] trace

            if not simulated_response:                     # [wrapper-only] nothing to feed the DLL → skip step
                logger.debug("M7: TC%d.S%d skipped (empty simulated_response)", ti + 1, si + 1)  # [wrapper-only] trace
                continue                                   # [wrapper-only]

            # run.py: for each equation in merged_equation_map, verify against its target.
            # Target = the value assigned by the LLM (from exp_dict for output signals,
            # or from combined_act for intermediate/input arg signals).
            # Mirrors run.py line 599: expected = merged_eq.equation_to_solve.split("=")[-1].strip()
            for entry in eq_results:                       # run.py:545 for eq_id, merged_eq in merged_equation_map.items() (per-equation loop)
                orig_eq = entry.get("original_equation", "")  # run.py:534 original_equation (LHS = target signal name)
                if "=" not in orig_eq:                     # [wrapper-only] guard (need an "LHS = ..." form to name the target)
                    continue                               # [wrapper-only]
                sig_name = orig_eq.split("=")[0].strip()   # [wrapper-only] LHS signal name (run.py:528 lhs = full_eq.split("=")[0].strip())
                if not sig_name:                           # [wrapper-only] guard (empty LHS)
                    continue                               # [wrapper-only]

                # Find target: output signal → exp_dict; input/arg signal → combined_act
                # run.py:599 expected = merged_eq.equation_to_solve.split("=")[-1].strip() (the LLM-assigned value);
                # here that value lives in the test case's exp/act dicts instead of the equation RHS.
                if sig_name in exp_dict:                   # [wrapper-only] output signal → Expected Results value
                    target = exp_dict[sig_name]            # [wrapper-only] M7 target = expected output
                elif sig_name in combined_act:             # [wrapper-only] input/arg signal → its assigned action value
                    target = combined_act[sig_name]        # [wrapper-only] M7 target = assigned input
                else:
                    continue  # equation doesn't apply to this step  # [wrapper-only] this equation's signal not present in this step

                ac = entry.get("assignment_chain", "")     # run.py:535 assignment_chain (from EquationResult)
                fse = entry.get("final_substituted_equation", "")  # run.py:536 final_substituted_equation
                if not ac or not fse:                      # [wrapper-only] guard (need both to run the backbone)
                    continue                               # [wrapper-only]

                # run.py line 546: eq_to_solve = assignment_chain.split("\n")[::-1]
                eq_to_solve_entry = [ln for ln in ac.split("\n") if ln.strip()][::-1]  # run.py:546 eq_to_solve = assignment_chain.split("\n")[::-1]
                if not eq_to_solve_entry:               # [wrapper-only] guard (skip empty chain)
                    continue                            # [wrapper-only]

                total += 1                              # [wrapper-only] M7 denominator (run.py has no score)
                logger.debug("M7: TC%d.S%d eq sig=%s target=%s | final_substituted_equation=%s",  # [wrapper-only] trace
                              ti + 1, si + 1, sig_name, target, fse)  # [wrapper-only] trace args

                try:                                    # [wrapper-only] per-step guard (run.py runs once per equation, not per step)
                    # run.py lines 581-600: backbone (no special fc.functionCall path)
                    attempt = solve_with_dll_timeout(lambda: _attempt_solve(simulated_response, eq_to_solve_entry, fc), fc)  # [wrapper-only] watchdog: run.py:581 _attempt_solve (DLL) — hang → _DLLSolveTimeout (caught below → FAIL)
                    substitution_dict = _build_substitution_dict(attempt.final_assignments)  # run.py:582 substitution_dict = _build_substitution_dict(attempt.final_assignments)  [imported]
                    expr = _substitute_ops(fse, substitution_dict)  # run.py:591 expressions = [_substitute_ops(merged_eq.final_substituted_equation, substitution_dict)]
                    eq_list = generate_bodmas_chain(expr)  # run.py:594 eq_list = generate_bodmas_chain(expr)  [imported]
                    if not eq_list:                     # [wrapper-only] guard (empty chain → FAIL)
                        logger.debug("M7: TC%d.S%d sig=%s -> empty BODMAS chain, FAIL", ti + 1, si + 1, sig_name)  # [wrapper-only] trace
                        continue                        # [wrapper-only]
                    final_check, _ = solve_with_dll_timeout(lambda: resolve_equation_block(eq_list, fc), fc)  # [wrapper-only] watchdog: run.py:597 resolve_equation_block (DLL) — hang → _DLLSolveTimeout (caught below → FAIL)
                    if not final_check:                 # [wrapper-only] guard (no result → FAIL)
                        logger.debug("M7: TC%d.S%d sig=%s -> resolve_equation_block returned nothing, FAIL", ti + 1, si + 1, sig_name)  # [wrapper-only] trace
                        continue                        # [wrapper-only]
                    actual = final_check[-1].split("=")[-1].strip()  # run.py:600 actual = final_check[-1].split("=")[-1].strip()

                    # run.py line 609: if expected == actual: (tolerance ±1 for floats)
                    # [wrapper-only] LHS(DLL actual) == RHS(target) is the run.py DLL "pass"; float ±1 tolerance
                    try:
                        if abs(float(actual) - float(target)) <= 1:  # run.py:609 if expected == actual (numeric, ±1 tolerance)
                            passes += 1                    # [wrapper-only] M7 numerator: this equation's DLL check passed
                            logger.debug("M7: TC%d.S%d sig=%s actual=%s target=%s -> PASS", ti + 1, si + 1, sig_name, actual, target)  # [wrapper-only] trace
                        else:
                            logger.debug("M7: TC%d.S%d sig=%s actual=%s target=%s -> FAIL (diff>1)", ti + 1, si + 1, sig_name, actual, target)  # [wrapper-only] trace
                    except (ValueError, TypeError):        # [wrapper-only] non-numeric target/actual → fall back to string equality
                        if actual == str(target):          # run.py:609 if expected == actual (string form)
                            passes += 1                    # [wrapper-only] M7 numerator: string-equal DLL result passed
                            logger.debug("M7: TC%d.S%d sig=%s actual=%s target=%s -> PASS (string match)", ti + 1, si + 1, sig_name, actual, target)  # [wrapper-only] trace
                        else:
                            logger.debug("M7: TC%d.S%d sig=%s actual=%s target=%s -> FAIL (string mismatch)", ti + 1, si + 1, sig_name, actual, target)  # [wrapper-only] trace

                except Exception as e:                     # [wrapper-only] any DLL error for this step counts as FAIL (never crashes the metric)
                    logger.debug("M7: TC%d.S%d sig=%s -> DLL call exception: %s -> FAIL", ti + 1, si + 1, sig_name, e)  # [wrapper-only] trace

    score = passes / total if total > 0 else 0.5           # [wrapper-only] M7 = DLL pass rate; 0.5 neutral when nothing was checkable
    logger.info("M7: RESULT score=%.4f (DLL pass rate: %d/%d)", score, passes, total)  # [wrapper-only] trace
    return score, f"DLL pass rate: {passes}/{total}"       # [wrapper-only] final M7 score + human-readable detail


# ═══════════════════════════════════════════════════════════════════
# M8: FUNCTIONAL CORRECTNESS (sequential DLL execution, final test cases)
# ═══════════════════════════════════════════════════════════════════
def _m8_functional_correctness(   # [wrapper-only] M8 metric; DLL core REUSES run.py _step_4a backbone (546-600) SEQUENTIALLY (state carried)
    test_cases_str: str,
    equation_json: str,
    dll_path: str,
    signalsheet_data: str = "",
    model_interface_data: str = "",
    equation_results_json: str = "[]",
    raster_time: int = 100000,
    weakspot_op_key: str = "",
    const_values_json: str = "{}",
    relations_json: str = "{}",
    raw_test_cases_json: str = "",   # [wrapper-only] RAW Prompt-2 test (pre-post-processing); Parameter-arg fallback source (point 9)
    instance_names: str = "",        # [wrapper-only] declared weakspot OUTPUT signals (comma-sep) — for the wrong-I/O gate
) -> Tuple[float, str]:
    """Verify each final post-processed test case against the ASCET DLL.

    Reuses run.py _step_4a (508-600) DLL backbone via the SAME imported helpers as M7
    (_attempt_solve / _build_substitution_dict / generate_bodmas_chain / resolve_equation_block),
    but calls them SEQUENTIALLY so the FunctionCaller subprocess's internal state carries
    across steps (M7 is per-step stateless). The arithmetic path in _call_dll() below runs the
    exact run.py Phase 4 backbone per step (final input state → DLL output):
      1. eq_to_solve = assignment_chain.split('\\n')[::-1]            run.py:546
      2. simulated = running-input dict formatted as an LLM response  (state-carried)
      3. attempt = _attempt_solve(simulated, eq_to_solve, fc)         run.py:581  [imported]
      4. subs = _build_substitution_dict(attempt.final_assignments)   run.py:582  [imported]
      5. expr = _substitute_ops(final_substituted_equation, subs)     run.py:584-591 (regex copy)
      6. eq_list = generate_bodmas_chain(expr)                        run.py:594  [imported]
      7. final_check, _ = resolve_equation_block(eq_list, fc)         run.py:597  [imported]
      8. actual = final_check[-1].split('=')[-1].strip()              run.py:600
    Function-call weakspots (no arithmetic main equation) take the fc.functionCall path instead,
    resolving each arg via its own chain (still the same run.py backbone) then calling the
    weakspot's own DLL function. run.py itself has NO per-step expected-vs-actual comparison or
    warm-up/tick loop — that sequencing and the pass/fail scoring below are the wrapper-added layer.

    Unlike M7 (which solves each step independently to measure Prompt 4 quality),
    M8 executes steps SEQUENTIALLY so the DLL's internal state carries over:

      - Pre-conditions establish initial block state (warm-up call).
      - 'Wait N' advances the block N / raster_time DLL ticks before applying actions.
      - Each step's actions are merged into the running input dict; the DLL is called
        once and the result compared to 'Expected Results'.

    This catches post-processing corruption (flatten / dedup / MUX filter) that
    produces 'Frankenstein' input sets never verified together as a complete set.
    Also correctly handles stateful blocks (Srv_Debounce, LowpassT, Srv_Ramp, etc.)
    where each DLL call advances the block's internal state.

    Tolerance: ±(1.0 + 0.01 × |expected|)  — absolute ±1 for small values,
    ~1% relative for larger values.

    Fallback to 0.5 when DLL unavailable or no equation data.
    """
    logger.info("M8: ===== Functional Correctness (sequential DLL, final JSON) =====")  # [wrapper-only] trace banner
    if not dll_path or not os.path.exists(dll_path):       # run.py:800-807 dll_path presence check (run.py falls back to LLM-only; wrapper fails loud)
        logger.error("M8: DLL not available at %r -> raising (fail loud, no silent 0.5)", dll_path)  # [wrapper-only] trace
        raise RuntimeError(                                # [wrapper-only] fail loud — functional correctness cannot be skipped
            f"M8: DLL required but not available at {dll_path!r}. "
            "Functional correctness cannot be skipped — fix the 32-bit DLL setup before optimizing."
        )

    tcs, err = _normalize_test_cases(test_cases_str)       # [wrapper-only] parse final Prompt-2/post-processed test cases into TC dicts
    if err or not tcs:                                     # [wrapper-only] nothing parseable → neutral 0.5
        logger.error("M8: parse error -> score=0.5 default (%s)", err)  # [wrapper-only] trace
        return 0.5, f"Cannot parse test cases: {err}"     # [wrapper-only] neutral default when TCs unparseable

    logger.info("M8: %d final test case(s) to verify", len(tcs))  # [wrapper-only] trace

    # ── Wrong-I/O gate (M8_GATE_ON_WRONG_IO) ─────────────────────────────
    # A test that (c) verifies NO real measurable output, or (d) drives a weakspot OUTPUT as an INPUT, is
    # functionally invalid — the DLL would "verify" the wrong signal (e.g. internal `_flgDlydlnPostDrive`
    # instead of the measurable `InjPwrRly_r`). In that case M8 = 0 WITHOUT running the DLL. Mirrors M1's
    # (c)/(d) I/O-role checks but on the FINAL test cases M8 actually scores. Toggle off to disable.
    _m8_out_set = {n.strip().lower() for n in (instance_names or "").split(",") if n.strip()}  # [wrapper-only] measurable outputs (normalized)
    if M8_GATE_ON_WRONG_IO and _m8_out_set:                # [wrapper-only] gate only when we know the declared outputs
        _exp_all: set = set(); _in_all: set = set()        # [wrapper-only] all expected-output vs all input signal names
        for _tc in tcs:                                    # [wrapper-only] scan every final TC
            _in_all |= {k.strip().lower() for k in _tc.get("precond", {})}  # [wrapper-only] precond = inputs
            for _st in _tc.get("steps", []):               # [wrapper-only]
                _in_all |= {k.strip().lower() for k in _st.get("act", {})}  # [wrapper-only] actions = inputs
                _exp_all |= {k.strip().lower() for k in _st.get("exp", {})}  # [wrapper-only] expected = outputs checked
        _verifies_real = bool(_exp_all & _m8_out_set)      # [wrapper-only] (c) at least one expected var is a real measurable output
        _out_as_input = _in_all & _m8_out_set              # [wrapper-only] (d) a measurable output driven as an input
        if (not _verifies_real) or _out_as_input:          # [wrapper-only] wrong I/O → functionally invalid test
            _why = ("verifies no real measurable output" if not _verifies_real  # [wrapper-only]
                    else f"output signal driven as input: {sorted(_out_as_input)[:2]}")  # [wrapper-only]
            logger.info("M8: GATED to 0.0 (wrong I/O) — %s | measurable outputs=%s", _why, sorted(_m8_out_set))  # [wrapper-only] trace
            return 0.0, f"M8=0 (wrong I/O): {_why}; measurable output(s)={sorted(_m8_out_set)[:2]}"  # [wrapper-only] no DLL run

    # ── Detect a FUNCTION-CALL weakspot EARLY (e.g. getBit_unsigned_getBit,
    # Srv_RampSwitch_out). These have NO arithmetic "main equation" to solve — their
    # arg relations carry empty assignment_chains — so they must NOT bail at the
    # arithmetic main_entry gate below (which previously gave a misleading neutral 0.5).
    # They are verified by calling the weakspot's DLL function on its resolved args
    # (the _is_fn_call_ws / fc.functionCall path further down).
    try:                                                   # [wrapper-only] tolerate malformed equation KB JSON
        _eqj_early = json.loads(equation_json)             # [wrapper-only] parse KB to detect a fn-call weakspot before the arithmetic gate
    except (json.JSONDecodeError, TypeError):
        _eqj_early = {}                                    # [wrapper-only] empty on parse failure
    _wse = _eqj_early.get(weakspot_op_key, {}) if weakspot_op_key else {}  # [wrapper-only] the weakspot op's KB entry (as forward()/M4 do)
    _wsfn = (_wse.get("function_name") or _wse.get("class_method_name")  # [wrapper-only] resolve the weakspot's DLL fn name (multi-key fallback)
             or _wse.get("method_name") or "")

    # ── Pick main equation entry (arithmetic path) ──────────────────────────
    try:                                                   # [wrapper-only] tolerate malformed equation_results JSON
        eq_results = json.loads(equation_results_json)     # run.py:430 eq_ctx.equation_results (list[EquationResult]); here JSON-serialized in the Example
    except (json.JSONDecodeError, TypeError):
        eq_results = []                                    # [wrapper-only] empty on parse failure

    is_fn_call_ws = bool(_wse.get("type") == "Function" and _wsfn and _wse.get("args") and eq_results)  # [wrapper-only] fn-call weakspot iff KB type=Function w/ fn name + args + resolvable chains

    main_entry = None                                      # [wrapper-only] pick the arithmetic "main equation" (longest assignment_chain)
    for entry in eq_results:                               # [wrapper-only] scan every EquationResult
        ac = entry.get("assignment_chain", "")             # run.py:535 assignment_chain (from EquationResult)
        fse = entry.get("final_substituted_equation", "")  # run.py:536 final_substituted_equation
        if ac and fse and (main_entry is None or           # [wrapper-only] require both fields; keep the longest chain (the top-level equation)
                           len(ac) > len(main_entry.get("assignment_chain", ""))):
            main_entry = entry                             # [wrapper-only] this entry becomes the arithmetic main equation

    if main_entry is None and not is_fn_call_ws:           # [wrapper-only] neither arithmetic nor fn-call path available → neutral 0.5
        logger.warning("M8: no usable equation in equation_results_json -> score=0.5 default")  # [wrapper-only] trace
        return 0.5, "No usable equation in equation_results_json — using default 0.5"  # [wrapper-only] neutral default

    if main_entry is not None:                             # [wrapper-only] arithmetic path: seed the run.py backbone inputs
        assignment_chain = main_entry["assignment_chain"]  # run.py:535 orig.assignment_chain
        final_substituted_equation = main_entry["final_substituted_equation"]  # run.py:536 orig.final_substituted_equation
        eq_to_solve = [line for line in assignment_chain.split("\n") if line.strip()][::-1]  # run.py:546 eq_to_solve = assignment_chain.split("\n")[::-1] (reversed)
        # M8 arithmetic-path input trace
        logger.debug("M8: main_entry.original_equation=%s", main_entry.get("original_equation"))  # [wrapper-only] trace
        logger.debug("M8: main_entry.assignment_chain=\n%s", assignment_chain)  # [wrapper-only] trace
        logger.debug("M8: main_entry.final_substituted_equation=%s", final_substituted_equation)  # [wrapper-only] trace
    else:
        # Function-call weakspot with no arithmetic chain: verified via fc.functionCall.
        assignment_chain, final_substituted_equation, eq_to_solve = "", "", []  # [wrapper-only] no arithmetic main equation; fn-call path supplies the answer
        logger.info("M8: function-call weakspot %s (%s) — verifying via fc.functionCall "  # [wrapper-only] trace
                     "(no arithmetic main_entry)", weakspot_op_key, _wsfn)

    if not eq_to_solve and not is_fn_call_ws:              # [wrapper-only] empty chain and not a fn-call weakspot → neutral 0.5
        logger.warning("M8: assignment_chain is empty -> score=0.5 default")  # [wrapper-only] trace
        return 0.5, "assignment_chain is empty"           # [wrapper-only] neutral default

    # ── Ensure MBD pipeline modules are importable ───────────────────────────
    # [wrapper-only] locate MBD/weakspot_code_to_run_2 so run.py's pipeline.* helpers import
    _MBD_ROOT = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "MBD", "weakspot_code_to_run_2")
    )
    if _MBD_ROOT not in sys.path:                          # [wrapper-only] add MBD root to sys.path once
        sys.path.insert(0, _MBD_ROOT)                      # [wrapper-only] so `import pipeline.*` resolves to the original modules

    # [wrapper-only] load run.py's pipeline modules under MBD's own config.py, then restore the repo config
    if "pipeline.function_caller" not in sys.modules or "pipeline.equation_solver_dll" not in sys.modules:
        _repo_cfg = sys.modules.get("config")             # [wrapper-only] save the repo's config module to restore afterwards
        _spec = importlib.util.spec_from_file_location(   # [wrapper-only] build a spec for MBD's config.py (pipeline.* imports `config`)
            "mbd_config_metric", os.path.join(_MBD_ROOT, "config.py")
        )
        _mod = importlib.util.module_from_spec(_spec)      # [wrapper-only] instantiate MBD config module
        sys.modules["config"] = _mod                       # [wrapper-only] temporarily install MBD config as `config`
        _spec.loader.exec_module(_mod)                     # [wrapper-only] execute MBD config so pipeline.* sees its values
        try:
            import pipeline.function_caller  # noqa: F401  # run.py:814 from pipeline.function_caller import FunctionCaller [imported original]
            import pipeline.equation_solver_dll  # noqa: F401  # [wrapper-only] import the DLL solve helpers module (run.py imports these helpers at top)
        finally:
            if _repo_cfg is not None:                      # [wrapper-only] restore the repo's config after MBD imports resolve
                sys.modules["config"] = _repo_cfg          # [wrapper-only] put the original repo config back
            else:
                sys.modules.pop("config", None)            # [wrapper-only] no prior repo config → remove the temp one

    from pipeline.function_caller import FunctionCaller    # run.py:814 from pipeline.function_caller import FunctionCaller [imported original]
    from pipeline.equation_solver_dll import (             # [wrapper-only] import the SAME run.py _step_4a backbone helpers M7 uses
        _attempt_solve, _build_substitution_dict,          # run.py:581-582 attempt/substitution helpers [imported]
        generate_bodmas_chain, resolve_equation_block,     # run.py:594/597 BODMAS chain + DLL block resolver [imported]
    )

    # Constant defaults (dT + SystemConstants) — direct arg source + §2 DSM/struct context (parsed here so
    # _new_fc below can inject them).
    try:                                                   # [wrapper-only] tolerate malformed const-values JSON
        _const_values = json.loads(const_values_json) if const_values_json else {}  # run.py:795-796 const_values_dict (dT + SystemConstants)
    except (json.JSONDecodeError, TypeError):
        _const_values = {}                                 # [wrapper-only] empty on parse failure

    # ── Get/create shared FunctionCaller ────────────────────────────────────
    # [wrapper-only] cache one FunctionCaller per DLL across metric calls (run.py builds one per pipeline run)
    import csv                                             # [wrapper-only] parse CSV-serialized signalsheet/model-interface from the Example
    import io                                              # [wrapper-only] wrap CSV strings as file-like objects
    _env_libs_dir = os.path.dirname(dll_path)              # [wrapper-only] .../env_libs (dll parent; run.py:800 dll under env_libs)
    _fc_dir       = os.path.dirname(_env_libs_dir)         # run.py:728 fc_dir
    _fcname       = os.path.basename(_fc_dir)              # run.py:729 fcname
    _work_dir     = os.path.dirname(_fc_dir)               # run.py:730 work_dir
    _signal_sheet_list = list(csv.DictReader(io.StringIO(signalsheet_data or "")))  # run.py:817 signal_sheet list-of-dicts
    _mi_list = list(csv.DictReader(io.StringIO(model_interface_data or "")))        # run.py:756/825 model_interface list-of-dicts

    def _new_fc():                                         # [wrapper-only] §3: build a FRESH FunctionCaller (clean block state) — reused for per-TC reset
        _f = FunctionCaller(                               # run.py:821-826 FunctionCaller(FCname, work_dir, signal_sheet_data, model_interface_data)
            FCname=_fcname, work_dir=_work_dir,
            signal_sheet_data=_signal_sheet_list, model_interface_data=_mi_list,
        )
        # §2 DSM/param-struct context (best-effort; safe no-op when unloaded). This pipeline (like run.py:821)
        # does NOT load dfc_fid_data/array_matrix_data, so these stay None and DSM/struct-dependent arg chains
        # resolve with library defaults or fail -> None -> step FAIL (flag 0, per the locked decision). When a
        # future build DOES load them, inject the test's constants so DSM permission / ramp-param structs resolve.
        try:                                               # [wrapper-only] guarded: never crash the metric on context injection
            if isinstance(getattr(_f, "dfc_fid_data", None), dict):    # [wrapper-only] only when the build loaded FID keys
                _f.set_dfc_fid_data(_const_values)         # function_caller.py:654 (sets only existing keys)
            if isinstance(getattr(_f, "array_matrix_data", None), dict):  # [wrapper-only] only when the build loaded struct keys
                _f.set_array_matrix_data(_const_values)    # function_caller.py:663
        except Exception as _ce:                           # [wrapper-only] context injection is best-effort
            logger.debug("M8: DSM/array context injection skipped: %s", _ce)  # [wrapper-only] trace
        return _f                                          # [wrapper-only] the fresh caller

    with _dll_registry_lock:                               # [wrapper-only] serialize registry access (metric may run in a thread pool)
        if dll_path not in _dll_caller_registry:           # [wrapper-only] first time for this DLL → construct + cache
            try:
                logger.info("M8: initializing FunctionCaller FCname=%s work_dir=%s", _fcname, _work_dir)  # [wrapper-only] trace
                _dll_caller_registry[dll_path] = _new_fc()  # [wrapper-only] build + cache
                logger.info("M8: FunctionCaller initialized successfully")  # [wrapper-only] trace
            except Exception as _fc_err:                   # [wrapper-only] init can fail if 32-bit Python missing → cache None
                logger.error("M8: FunctionCaller init failed: %s", _fc_err)  # [wrapper-only] trace
                _dll_caller_registry[dll_path] = None      # [wrapper-only] sentinel so we don't retry a broken init

        fc = _dll_caller_registry[dll_path]                # [wrapper-only] the cached FunctionCaller (run.py's `function_caller`)

    if fc is None:                                         # [wrapper-only] no working DLL caller → fail loud (never fabricate a pass rate)
        logger.error("M8: fc is None (32-bit Python missing) -> raising (fail loud, no silent 0.5)")  # [wrapper-only] trace
        raise RuntimeError(                                # [wrapper-only] fail loud — functional correctness cannot be skipped
            "M8: FunctionCaller unavailable (32-bit Python missing). "
            "Functional correctness cannot be skipped — fix the 32-bit Python setup before optimizing."
        )

    try:                                                   # [wrapper-only] tolerate malformed equation KB JSON
        eq_json = json.loads(equation_json)                # [wrapper-only] parse equation KB (instance→type + fn-call weakspot arg lookup below)
    except (json.JSONDecodeError, TypeError):
        eq_json = {}                                       # [wrapper-only] empty on parse failure

    # _substitute_ops — EXACT copy from run.py lines 584-588 (bug-for-bug identical)
    def _substitute_ops(equation, subs):                   # run.py:584 def _substitute_ops(equation, subs)
        return re.sub(                                     # run.py:585 return re.sub(
            r"\$op\d+",                                    # run.py:586 r"\$op\d+"
            lambda m: subs.get(m.group(0), m.group(0)),    # run.py:587 lambda m: subs.get(m.group(0), m.group(0))
            equation,                                      # run.py:588 equation
        )

    # signal_name → type string ("Variable", "Parameter", etc.)
    # [wrapper-only] run.py never rebuilds this — its Prompt-4 LLM emits typed lines; here we synthesize them per step
    inst_to_type: Dict[str, str] = {
        v["instance_name"]: v.get("type", "Variable")      # [wrapper-only] map instance name → declared type (default Variable)
        for k, v in eq_json.items()                        # [wrapper-only] iterate KB entries
        if isinstance(v, dict) and v.get("instance_name")  # [wrapper-only] only dict entries carrying an instance_name
    }

    # ── FULL-CHAIN RESOLVE (DEFAULT ON; opt-out via MBD_M8_FULL_RESOLVE=0) — no run.py analog ──
    # The pruned weakspot eq_json above knows only the weakspot's LOCAL signals, so a test that drives
    # the UPSTREAM PHYSICAL inputs (e.g. EnvP_p, Epm_nEng) gets filtered out by _call_dll's
    # `name in inst_to_type` guard -> empty simulated -> None (M8 never runs the DLL). This mode widens
    # inst_to_type with the CORRECTLY-TYPED leaf signals referenced in the equation_results_json chains
    # (Variable(EnvP_p), Parameter(..._C), ...), so the physical inputs are recognised and the EXISTING
    # arg-resolution backbone (priority-5 chain solve) can compute each weakspot arg from them and call
    # the DLL. Any arg/leaf/function it still can't resolve -> None -> that step FAILs (honest 0), exactly
    # the "run the DLL if we have everything, else flag 0" behaviour. STRICT IMPROVEMENT (only turns some
    # 0s into real pass/fail where the weakspot fn is DLL-exported, e.g. Rail; unexported fns / gated /
    # disconnected weakspots stay 0 as before). Default ON; set MBD_M8_FULL_RESOLVE=0 to restore the old gate.
    _m8_full_resolve = os.getenv("MBD_M8_FULL_RESOLVE", "1") not in ("0", "false", "False")  # [wrapper-only] DEFAULT ON; opt-out via =0
    if _m8_full_resolve:                                    # [wrapper-only] widen inst_to_type with the chain leaves + their real types
        try:                                               # [wrapper-only] tolerate malformed equation_results JSON
            _er = json.loads(equation_results_json or "[]")  # [wrapper-only] the per-arg assignment chains (leaves live here)
        except (json.JSONDecodeError, TypeError):          # [wrapper-only] bad JSON -> no expansion
            _er = []                                        # [wrapper-only]
        _added = 0                                          # [wrapper-only] count of new leaf signals recognised
        for _e in (_er if isinstance(_er, list) else []):   # [wrapper-only] iterate chain entries
            if not isinstance(_e, dict):                    # [wrapper-only] skip malformed
                continue
            _blob = f"{_e.get('assignment_chain','') or ''}\n{_e.get('final_substituted_equation','') or ''}"  # [wrapper-only] chain text
            for _m in re.finditer(r"(Variable|Parameter|Local|SystemConstant|Literal)\(([A-Za-z_][A-Za-z0-9_]*)", _blob):  # [wrapper-only] Type(Name) tokens
                _typ, _nm = _m.group(1), _m.group(2)        # [wrapper-only] declared type + signal name
                if _nm not in inst_to_type:                 # [wrapper-only] eq_json types WIN (don't override the pruned-graph truth)
                    inst_to_type[_nm] = _typ                # [wrapper-only] recognise this leaf with its chain-declared type
                    _added += 1                             # [wrapper-only]
        logger.info("M8: FULL_RESOLVE ON — inst_to_type widened by %d chain-leaf signal(s) -> %d total (physical inputs now recognised)", _added, len(inst_to_type))  # [wrapper-only] trace

    # ── Detect function-call weakspot (e.g., Srv_RampSwitch_out) ────────────
    # When the weakspot op is a DLL function call (not arithmetic), we must:
    #   1. resolve each input arg via its own equation chain
    #   2. call fc.functionCall(fn, args, instance) to get the actual output
    # This replaces the single-chain BODMAS approach used for arithmetic ops.
    _ws_entry = eq_json.get(weakspot_op_key, {}) if weakspot_op_key else {}  # [wrapper-only] the weakspot op's KB entry (fn name/args live here)
    _ws_fn_name = (                                        # [wrapper-only] the DLL function to call (multi-key fallback, as forward()/M4)
        _ws_entry.get("function_name")
        or _ws_entry.get("class_method_name")
        or _ws_entry.get("method_name")
        or ""
    )
    _ws_instance = _ws_entry.get("instance_name", "")      # [wrapper-only] DLL instance name → fc.functionCall instance_name arg (function_caller.py:679)
    _ws_args = _ws_entry.get("args", [])          # ordered [$opN, ...] list  # [wrapper-only] arg op-keys in call order
    _ws_arg_names = _ws_entry.get("arg_names", [])  # ordered ["Swtch", "Xa", ...]  # [wrapper-only] arg names for fc.functionCall listOfArgNames (function_caller.py:678)
    _is_fn_call_ws = bool(_ws_fn_name and _ws_args and eq_results)  # [wrapper-only] fn-call path iff we have a fn name, args, and chains to resolve them

    # M8 fn-call weakspot detection trace
    logger.info(
        "M8: weakspot_op_key=%s ws_fn_name=%r ws_instance=%r ws_arg_names=%s ws_args=%s -> _is_fn_call_ws=%s",  # [wrapper-only] trace
        weakspot_op_key, _ws_fn_name, _ws_instance, _ws_arg_names, _ws_args, _is_fn_call_ws,
    )

    # ── §4 EXPORT PRE-CHECK — skip cleanly when the weakspot fn is not in the DLL/Python lib ──
    # is_function_available() returns "srv_lib"/"python" when callable, "NA" when the symbol is in NEITHER
    # the 32-bit DLL NOR the ASCET Python package (e.g. LowpassT_out, MoFSrv_Lib_RampEnhd_Out). Calling such a
    # fn raises inside fc.functionCall on EVERY step; pre-checking once gives an honest 0 with a clear reason
    # (per the locked decision: unverifiable -> flag 0) and avoids the per-step exception spam.
    if _is_fn_call_ws and _ws_fn_name:                     # [wrapper-only] only relevant on the fn-call path
        # [wrapper-only] GUARD (2026-07-15 fix): is_function_available() is a 32-bit DLL RPC that can HANG
        # (blocks the bridge _rpc/_read_exact, which has no read timeout) — a plain try/except catches
        # EXCEPTIONS, not a FREEZE. A hung probe here froze two concurrent 300-call runs on EGSSig 'LowpassT_out'.
        # Run it under the SAME Guard-2 watchdog as fc.functionCall; on hang, kill+drop the FunctionCaller and
        # flag the fn unverifiable (honest M8=0) so a single hung RPC can never freeze the run.
        try:                                               # [wrapper-only] availability probe must never crash the metric
            _pa_ok, _avail = _run_with_timeout(lambda: fc.is_function_available(_ws_fn_name), _M8_FNCALL_TIMEOUT_S)  # function_caller.py:448 -> "srv_lib" | "python" | "NA"  [watchdog-guarded]
        except Exception as _ae:                           # [wrapper-only] probe RAISED -> treat as unknown, let the run try
            _pa_ok, _avail = True, None                    # [wrapper-only]
            logger.debug("M8: is_function_available(%s) raised %s -> proceeding", _ws_fn_name, _ae)  # [wrapper-only] trace
        if not _pa_ok:                                     # [wrapper-only] probe HUNG (DLL bridge frozen) -> kill+poison, honest 0
            logger.warning("M8: is_function_available(%s) exceeded %.0fs -> DLL hang; killing helper + poisoning fn + rebuilding FunctionCaller, M8=0",  # [wrapper-only]
                            _ws_fn_name, _M8_FNCALL_TIMEOUT_S)  # [wrapper-only]
            try:
                if getattr(fc, "process", None) is not None:  # [wrapper-only] free the frozen 32-bit helper
                    fc.process.kill()
            except Exception:
                pass
            with _dll_registry_lock:                       # [wrapper-only] poison the fn + drop the fc so a later example rebuilds clean
                _M8_HUNG_FUNCTIONS.add(_ws_fn_name)
                _dll_caller_registry.pop(dll_path, None)
            return 0.0, f"M8=0: weakspot function '{_ws_fn_name}' DLL availability probe hung (watchdog killed the helper; unverifiable)"  # [wrapper-only]
        if _avail == "NA":                                 # [wrapper-only] genuinely not exported anywhere
            logger.info("M8: weakspot fn %r NOT exported in DLL/Python lib (is_function_available=NA) -> M8=0 (unverifiable)", _ws_fn_name)  # [wrapper-only] trace
            return 0.0, f"M8=0: weakspot function '{_ws_fn_name}' not exported in the DLL/Python library (unverifiable)"  # [wrapper-only] honest 0, no DLL run


    # ── Build $opN → resolution data for each weakspot function arg ──────────
    # Two sources, in priority order, are used at call time (see _call_dll):
    #   1. the arg's INSTANCE value taken directly from the test inputs / consts
    #      (honours the test case's explicit override, e.g. Rail_trqFlTempLimPCV=1000)
    #   2. the arg's own equation chain (eq_results entry) as a fallback when the
    #      test does not set the signal directly.
    #
    # eq_results entries carry `original_equation` == the relation string. The
    # relations dict (relations_json) is keyed by arg_name and aligned with the
    # weakspot's arg_names/args, so we map each arg op-key to its entry by EXACT
    # original_equation match (robust to ordering / filtered-empty relations).
    try:                                                   # [wrapper-only] tolerate malformed relations JSON
        _relations = json.loads(relations_json) if relations_json else {}  # run.py:429 eq_ctx.relations (arg_name → solvable relation); Example-serialized here
    except (json.JSONDecodeError, TypeError):
        _relations = {}                                    # [wrapper-only] empty on parse failure

    # [wrapper-only] index eq_results by their original_equation string so each arg's relation maps to its chain
    _orig_to_entry = {
        (e.get("original_equation") or "").strip(): e      # [wrapper-only] key = original_equation (run.py:522 keys original_eq_map the same way)
        for e in eq_results                                # [wrapper-only] iterate every EquationResult
        if (e.get("original_equation") or "").strip()      # [wrapper-only] skip entries with no original_equation
    }

    _op_to_entry: Dict[str, dict] = {}   # $opN → eq_results entry (chain fallback)  # [wrapper-only] per-arg chain source for _call_dll
    _arg_instance: Dict[str, str] = {}   # $opN → instance signal name (direct source)  # [wrapper-only] per-arg direct-input source
    if _is_fn_call_ws:                                     # [wrapper-only] only build arg maps for the fn-call path
        for _idx, _arg_op in enumerate(_ws_args):          # [wrapper-only] iterate weakspot fn args in call order
            _arg_name = _ws_arg_names[_idx] if _idx < len(_ws_arg_names) else ""  # [wrapper-only] this arg's name (may be missing)
            _arg_instance[_arg_op] = (eq_json.get(_arg_op, {}) or {}).get("instance_name", "")  # [wrapper-only] the arg op's signal name (direct-input source)
            # match the arg's relation/original_equation to its eq_results entry
            _rel = (_relations.get(_arg_name) or "").strip()  # [wrapper-only] this arg's relation string (looked up by arg name)
            _entry = _orig_to_entry.get(_rel)              # [wrapper-only] find the eq_results entry whose original_equation == this relation
            if _entry is None and len(eq_results) == len(_ws_args):  # [wrapper-only] no exact match but counts align → positional fallback
                _entry = eq_results[_idx]  # positional fallback (relations order == args order)  # [wrapper-only] map by index
            if _entry is not None:                         # [wrapper-only] record the arg's chain fallback if found
                _op_to_entry[_arg_op] = _entry             # [wrapper-only] $opN → its resolving eq_results entry
        # M8 fn-call arg-map trace
        logger.debug("M8: _arg_instance = %s", _arg_instance)  # [wrapper-only] trace
        logger.debug("M8: _op_to_entry keys = %s", list(_op_to_entry.keys()))  # [wrapper-only] trace
        for _idx, _arg_op in enumerate(_ws_args):          # [wrapper-only] warn about args with no resolution source
            _inst = _arg_instance.get(_arg_op, "")         # [wrapper-only] this arg's direct-input signal (if any)
            if not _inst and _arg_op not in _op_to_entry:  # [wrapper-only] neither direct input nor chain → will resolve to None
                logger.warning("M8: weakspot arg %s has neither instance name nor eq_results entry — may resolve to None", _arg_op)  # [wrapper-only] trace

    # [wrapper-only] Point-9 Parameter-arg fallback source: the RAW Prompt-2 test (pre-post-processing)
    # sets fn-call args BY ARG NAME (e.g. "initValue = 6", and in other draws "T = 5"). Post-processing can
    # drop these from the FINAL test cases, leaving a bare Parameter arg unresolvable (empty BODMAS chain →
    # None → the whole weakspot scores M8=0). Build per-final-TC {arg_name: value} maps from the raw test
    # (index-aligned when the TC counts match; else a global union, logged) so _call_dll can resolve those
    # args to the test author's intent instead of returning None.
    _raw_args_by_tc: List[Dict[str, str]] = []             # [wrapper-only] per-raw-TC {arg_name: value}
    _raw_args_global: Dict[str, str] = {}                  # [wrapper-only] union fallback (first value seen per arg name)
    try:
        _raw_tcs, _ = _normalize_test_cases(raw_test_cases_json) if (raw_test_cases_json or "").strip() else ([], "")  # [wrapper-only] parse raw Prompt-2 test (skip empty → no spurious parse-error)
        for _rtc in _raw_tcs:                              # [wrapper-only] one arg map per raw TC
            _rm: Dict[str, str] = dict(_rtc.get("precond", {}))   # [wrapper-only] seed with pre-condition assignments
            for _rst in _rtc.get("steps", []):             # [wrapper-only] merge every step's actions (later steps win)
                _rm.update(_rst.get("act", {}))            # [wrapper-only] arg_name → value (e.g. "initValue" → "6")
            _raw_args_by_tc.append(_rm)                    # [wrapper-only]
            for _k, _v in _rm.items():                     # [wrapper-only] accumulate the global union (first value per name)
                _raw_args_global.setdefault(_k, _v)        # [wrapper-only]
    except Exception as _re:                               # [wrapper-only] raw test unparseable → no fallback (behaviour unchanged)
        logger.debug("M8: raw-test Parameter fallback unavailable: %s", _re)  # [wrapper-only] trace
    _raw_aligned = (len(_raw_args_by_tc) == len(tcs))      # [wrapper-only] per-TC index-alignment valid only when counts match
    if _raw_args_by_tc and not _raw_aligned:               # [wrapper-only] counts differ → per-TC mapping unreliable; fall back to union
        logger.info("M8: raw-test fallback: raw TC count %d != final %d → using GLOBAL union (approximate)", len(_raw_args_by_tc), len(tcs))  # [wrapper-only]
    _cur_raw_args: Dict[str, str] = {}                     # [wrapper-only] current TC's raw arg values; set in the TC loop, read by _call_dll

    def _call_dll(inputs_dict: Dict[str, str]) -> Optional[str]:
        """Run the equation chain(s) through the DLL with the given inputs.

        For function-call weakspots: resolves each input arg via its own chain,
        then calls fc.functionCall(ws_fn_name, arg_values, ws_instance).
        For arithmetic weakspots: runs the main chain through BODMAS.

        The FunctionCaller subprocess (fc) is STATEFUL — each call advances the
        ASCET block's internal state (debounce timer, filter accumulator, etc.).
        """
        # [wrapper-only] reconstruct what run.py's Prompt-4 LLM would emit (run.py:564 simulated_response)
        simulated = [
            f"{inst_to_type.get(name, 'Variable')}({name}) = {val}"  # [wrapper-only] "Type(name) = value" line per running input
            for name, val in inputs_dict.items()           # [wrapper-only] iterate current running input state
            if val not in ("", None) and name in inst_to_type  # [wrapper-only] skip blanks / signals not in the KB
        ]
        if not simulated:                                  # [wrapper-only] nothing recognisable to feed the DLL → no output
            logger.debug("M8: _call_dll(inputs=%s) -> None (no recognised inputs; inst_to_type misses these names)", inputs_dict)  # [wrapper-only] trace
            return None                                    # [wrapper-only]

        # ── §1 helper: resolve ONE weakspot arg through its own equation chain via the DLL ──
        # run.py _step_4a backbone (581-600), using the per-call `simulated` (= THIS step's running inputs),
        # so the arg reflects the CURRENT scenario. Returns the value string, or None when unresolvable.
        def _resolve_arg_chain(_arg_op):                   # [wrapper-only] shared by the live-input path (§1) and the fallback
            _e = _op_to_entry.get(_arg_op)                 # [wrapper-only] this arg's eq_results entry (its chain)
            if _e is None:                                 # [wrapper-only] no chain → unresolvable
                return None
            _ec = _e.get("assignment_chain", "")           # run.py:535 assignment_chain
            _ef = _e.get("final_substituted_equation", "")  # run.py:536 final_substituted_equation
            _eqs = [ln for ln in _ec.split("\n") if ln.strip()][::-1]  # run.py:546 eq_to_solve (reversed)
            try:                                           # [wrapper-only] a DLL hang here → treat this arg as unresolvable (None), not a freeze
                _att = solve_with_dll_timeout(lambda: _attempt_solve(simulated, _eqs, fc), fc)  # [wrapper-only] watchdog around run.py:581 _attempt_solve
                _sub = _build_substitution_dict(_att.final_assignments)  # run.py:582 [imported]
                _xpr = _substitute_ops(_ef, _sub)              # run.py:591
                _eql = generate_bodmas_chain(_xpr)             # run.py:594 [imported]
                if not _eql:                                   # [wrapper-only] empty chain → unresolvable
                    return None
                _fc_res, _ = solve_with_dll_timeout(lambda: resolve_equation_block(_eql, fc), fc)  # [wrapper-only] watchdog around run.py:597 resolve_equation_block
            except _DLLSolveTimeout:                        # [wrapper-only] DLL froze on this arg chain → unresolvable (fc already killed + self-restarts)
                logger.warning("M8: arg-chain resolve for %s hit the DLL watchdog -> unresolvable", _arg_op)  # [wrapper-only] trace
                return None
            if not _fc_res:                                # [wrapper-only] no DLL result → unresolvable
                return None
            return _fc_res[-1].split("=")[-1].strip()      # run.py:600 actual value

        if _is_fn_call_ws:                                 # [wrapper-only] fn-call path: resolve each arg, then call the weakspot fn on the DLL
            # Guard 2 (skip): if this weakspot fn already hung the DLL earlier this run, never call
            # it again — return None (step FAIL) instantly instead of re-hanging for the full timeout.
            if _ws_fn_name in _M8_HUNG_FUNCTIONS:          # [wrapper-only] Guard 2: previously-hung fn → skip the DLL entirely
                logger.warning("M8: fn %s previously hung the DLL — skipping fc.functionCall (step FAIL, unverifiable)", _ws_fn_name)  # [wrapper-only] trace
                return None                                # [wrapper-only] unverifiable (same as the documented non-exported case)
            # Resolve each weakspot function arg, then call the function on the DLL.
            #
            # §1 arg value priority (fixed so the DLL RESPONDS to the per-step inputs):
            #   1. the arg's instance signal taken DIRECTLY from the test inputs,
            #   2. LIVE chain-resolve — when the arg's chain reaches a signal the test is CURRENTLY driving
            #      (so the arg tracks EnvP_p/Rail_stMon/... per step; was the bug: raw-test froze it),
            #   3. a constant default (dT etc.),
            #   4. raw-test by arg-name (fixed init/param the test set only by name, e.g. initValue/T),
            #   5. Literal,
            #   6. chain-resolve fallback (arg not driven, but has a chain).
            arg_values: List[Any] = []                     # [wrapper-only] ordered resolved arg values for fc.functionCall
            _driven = {n for n, v in inputs_dict.items() if v not in ("", None)}  # [wrapper-only] signals the test currently drives
            try:                                            # [wrapper-only] any resolution failure → None (step counts as FAIL)
                for _idx, _arg_op in enumerate(_ws_args):   # [wrapper-only] resolve each weakspot fn arg in call order
                    _arg_name = _ws_arg_names[_idx] if _idx < len(_ws_arg_names) else _arg_op  # [wrapper-only] arg name (falls back to op-key)
                    _inst = _arg_instance.get(_arg_op, "")  # [wrapper-only] this arg's direct-input signal name
                    _val: Optional[str] = None              # [wrapper-only] resolved value (str until float-cast below)
                    _src = ""                               # [wrapper-only] which source resolved it (trace only)

                    _arg_op_def = eq_json.get(_arg_op, {}) or {}  # [wrapper-only] the arg op's KB entry (for Literal detection)
                    # §1: does this arg's chain reach a signal the test CURRENTLY drives? -> resolve it LIVE
                    _e_chk = _op_to_entry.get(_arg_op)      # [wrapper-only] the arg's chain entry (if any)
                    _chain_driven = False                   # [wrapper-only] True → the per-step inputs affect this arg
                    if _e_chk is not None and _driven:      # [wrapper-only] only worth checking when there is a chain + driven inputs
                        _cblob = f"{_e_chk.get('assignment_chain','') or ''}\n{_e_chk.get('final_substituted_equation','') or ''}"  # [wrapper-only] chain text
                        _cleaves = {m.group(1) for m in re.finditer(r"(?:Variable|Parameter|Local)\(([A-Za-z_][A-Za-z0-9_]*)", _cblob)}  # [wrapper-only] chain leaf signals
                        _chain_driven = bool(_cleaves & _driven)  # [wrapper-only] the test drives something upstream of this arg

                    if _inst and inputs_dict.get(_inst) not in ("", None):  # [wrapper-only] priority 1: test case sets this signal directly
                        _val, _src = str(inputs_dict[_inst]), "input"  # [wrapper-only] use the explicit test input
                    elif _m8_full_resolve and _chain_driven:  # [wrapper-only] priority 2 (§1, gated by FULL_RESOLVE): LIVE chain-resolve from the current inputs
                        _val = _resolve_arg_chain(_arg_op)  # [wrapper-only] compute the arg from THIS step's inputs via the DLL
                        if _val is None:                    # [wrapper-only] driven arg that won't resolve (DSM/struct/unexported sub-block) → step FAIL
                            logger.debug("M8: _call_dll: arg %s (%s/%s) LIVE chain-resolve failed -> None", _arg_op, _arg_name, _inst)  # [wrapper-only] trace
                            return None                     # [wrapper-only]
                        _src = "chain-live"                 # [wrapper-only]
                    elif _inst and _inst in _const_values and _const_values[_inst] not in ("", None):  # [wrapper-only] priority 3: a constant default (dT etc.)
                        _val, _src = str(_const_values[_inst]), "const"  # [wrapper-only] use the const default
                    elif _arg_name in _cur_raw_args and _cur_raw_args.get(_arg_name) not in ("", None):  # [wrapper-only] priority 4 (point 9): RAW Prompt-2 test set this NON-DRIVEN arg BY NAME (e.g. initValue=6, T=5)
                        _val, _src = str(_cur_raw_args[_arg_name]), "raw-test"  # [wrapper-only] use the raw-test value (the test author's intent)
                    elif (_arg_op_def.get("type") == "Literal"  # [wrapper-only] priority 5: Literal arg — instance_name IS the numeric value
                          and re.fullmatch(r"-?\d+(?:\.\d+)?", str(_inst or ""))):  # [wrapper-only] confirm the literal is numeric
                        # Literal arg (e.g. bitPosition = Literal(0)): its instance_name
                        # IS the numeric value. generate_bodmas_chain() can't expand a bare
                        # literal, so resolve it directly instead of returning None.
                        _val, _src = str(_inst), "literal"  # [wrapper-only] the literal value itself
                    else:
                        # priority 6: resolve the arg through its own equation chain (not driven, but has a chain).
                        _val = _resolve_arg_chain(_arg_op)  # [wrapper-only] chain fallback via the DLL (run.py _step_4a backbone)
                        if _val is None:                    # [wrapper-only] no input/const/raw/literal/chain → unresolvable arg
                            logger.debug("M8: _call_dll: arg %s (%s/%s) unresolved (no input/const/raw/literal/chain) -> None", _arg_op, _arg_name, _inst)  # [wrapper-only] trace
                            return None                     # [wrapper-only] can't call the fn without every arg
                        _src = "chain"                      # [wrapper-only]

                    try:                                    # [wrapper-only] prefer numeric args (DLL fn expects numbers)
                        arg_values.append(float(_val))      # [wrapper-only] append as float when parseable
                    except (ValueError, TypeError):
                        arg_values.append(_val)             # [wrapper-only] otherwise pass the raw string
                    logger.debug("M8: _call_dll: arg[%d] %s=%s (inst=%s, src=%s) -> %s",  # [wrapper-only] trace
                                  _idx, _arg_name, _arg_op, _inst, _src, _val)

                # Real signature: functionCall(funcName, listOfArguments, listOfArgNames, instance_name)
                # NOTE: listOfArgNames MUST be supplied — the 32-bit helper marshals
                # args via `zip(str_args, str_arg_names)`, so a None/empty names list
                # silently drops ALL arguments (the DLL then reads uninitialised memory).
                # KNOWN LIMITATION: this returns 0.0 (via None → step FAIL) for library-function
                # weakspots the FC DLL does not EXPORT (e.g. Srv_TrnOnDly) — fc.functionCall
                # raises inside the DLL, so no actual output exists to compare. Such weakspots
                # need a DLL that exports the function; M8 cannot verify them otherwise.
                # Guard 2 (watchdog): some arg combos make a SINGLE fc.functionCall spin forever
                # inside the 32-bit DLL, and the bridge (function_caller.py:390) has no read timeout
                # -> the whole run freezes. Run the call under a timeout; on expiry, poison this fn
                # (skip it hereafter) and drop the shared FunctionCaller (its bridge lock is held by
                # the still-blocked worker), so the next example rebuilds a clean one. Step FAILs (None).
                _fc_ok, result = _run_with_timeout(        # [wrapper-only] Guard 2 watchdog around function_caller.py:674 functionCall(...)
                    lambda: fc.functionCall(               # function_caller.py:674 functionCall(funcName, listOfArguments, listOfArgNames, instance_name)
                        funcName=_ws_fn_name,              # function_caller.py:676 funcName
                        listOfArguments=arg_values,        # function_caller.py:677 listOfArguments
                        listOfArgNames=list(_ws_arg_names),  # function_caller.py:678 listOfArgNames (must be supplied; see note above)
                        instance_name=_ws_instance,        # function_caller.py:679 instance_name
                    ),
                    _M8_FNCALL_TIMEOUT_S,                  # [wrapper-only] Guard 2 timeout budget (healthy call = ms)
                )
                if not _fc_ok:                             # [wrapper-only] Guard 2: fn-call exceeded the timeout -> treat as a DLL hang
                    logger.warning("M8: fc.functionCall(%s) exceeded %.0fs -> DLL hang; poisoning fn + rebuilding FunctionCaller, step FAIL",  # [wrapper-only] trace
                                   _ws_fn_name, _M8_FNCALL_TIMEOUT_S)
                    with _dll_registry_lock:               # [wrapper-only] serialize poison-set + registry mutation
                        _M8_HUNG_FUNCTIONS.add(_ws_fn_name)  # [wrapper-only] never call this fn again this run
                        _dll_caller_registry.pop(dll_path, None)  # [wrapper-only] abandon the fc (bridge lock held by the blocked worker); next example builds fresh
                    return None                            # [wrapper-only] unverifiable this step (matches library-fn limitation)
                _out = result.get("result") if isinstance(result, dict) else result  # function_caller.py:706 return {"result": ...} → unwrap the result value
                logger.debug("M8: _call_dll: fc.functionCall(%s, %s, instance=%s) -> %s (unwrapped=%s)",  # [wrapper-only] trace
                              _ws_fn_name, arg_values, _ws_instance, result, _out)
                return str(_out) if _out is not None else None  # [wrapper-only] DLL output as string (None → step FAIL, incl. non-exported library fns)
            except Exception as e:                          # [wrapper-only] any DLL/marshalling error → None (step FAIL, never crashes the metric)
                logger.debug("M8: _call_dll: function-call path exception: %s -> None", e)  # [wrapper-only] trace
                return None                                 # [wrapper-only]
        else:                                          # arithmetic weakspot → same run.py _step_4a backbone (581-600) as M7, but called sequentially (DLL state carried)
            try:                                        # [wrapper-only] guard
                attempt = solve_with_dll_timeout(lambda: _attempt_solve(simulated, eq_to_solve, fc), fc)  # [wrapper-only] watchdog: run.py:581 _attempt_solve (DLL) — hang → _DLLSolveTimeout (caught below → None/FAIL)
                subs = _build_substitution_dict(attempt.final_assignments)  # run.py:582 substitution_dict = _build_substitution_dict(attempt.final_assignments)  [imported]
                expr = _substitute_ops(final_substituted_equation, subs)  # run.py:591 expressions = [_substitute_ops(final_substituted_equation, substitution_dict)]
                eq_list = generate_bodmas_chain(expr)   # run.py:594 eq_list = generate_bodmas_chain(expr)  [imported]
                if not eq_list:                         # [wrapper-only] guard (empty chain → None)
                    logger.debug("M8: _call_dll: empty BODMAS chain (expr=%s) -> None", expr)  # [wrapper-only] trace
                    return None                         # [wrapper-only]
                final_check, _ = solve_with_dll_timeout(lambda: resolve_equation_block(eq_list, fc), fc)  # [wrapper-only] watchdog: run.py:597 resolve_equation_block (DLL) — hang → _DLLSolveTimeout (caught below → None/FAIL)
                if not final_check:                     # [wrapper-only] guard (no result → None)
                    logger.debug("M8: _call_dll: resolve_equation_block empty -> None")  # [wrapper-only] trace
                    return None                         # [wrapper-only]
                result = final_check[-1].split("=")[-1].strip()  # run.py:600 actual = final_check[-1].split("=")[-1].strip()
                logger.debug("M8: _call_dll: arithmetic path -> %s", result)  # [wrapper-only] trace
                return result                           # [wrapper-only] return the DLL-resolved output for tolerance compare
            except Exception as e:                      # [wrapper-only] guard
                logger.debug("M8: _call_dll: arithmetic path exception: %s -> None", e)  # [wrapper-only] trace
                return None                             # [wrapper-only]

    total_steps = 0                                        # [wrapper-only] M8 denominator (steps with an Expected Results to check)
    total_passes = 0                                       # [wrapper-only] M8 numerator (steps whose DLL output matched)
    step_results: List[str] = []                           # [wrapper-only] per-step PASS/FAIL detail strings for the summary
    _step_actuals: List[float] = []                        # [wrapper-only] numeric DLL actual per CHECKED step (for the constant-output guard)

    # [wrapper-only] SEQUENTIAL execution loop — run.py has NO per-step compare/warm-up/tick loop; this whole loop is wrapper-added
    for ti, tc in enumerate(tcs):                          # [wrapper-only] iterate each final test case (state carried across its steps)
        _cur_raw_args.clear()                              # [wrapper-only] point-9: load this TC's raw-test arg values (Parameter fallback)
        if _raw_aligned and ti < len(_raw_args_by_tc):     # [wrapper-only] TC counts match → precise per-TC mapping
            _cur_raw_args.update(_raw_args_by_tc[ti])      # [wrapper-only] e.g. {"initValue":"6"} for _call_dll's arg resolution
        else:                                              # [wrapper-only] counts differ / no raw → best-effort union (logged above; empty if no raw)
            _cur_raw_args.update(_raw_args_global)         # [wrapper-only]
        logger.info("M8: ----- TC%d: precond=%s -----", ti + 1, tc.get("precond", {}))  # [wrapper-only] trace
        # ── §3 State reset between test cases (+ before TC1, to clear a PREVIOUS example's bleed) ──────
        # There is no `<FCname>_init` in these DLLs (the old attempt always failed), so stateful blocks
        # (ramp/filter/debounce accumulators) would carry state across TCs — and across M8 calls on the
        # SHARED cached FunctionCaller. Reset by rebuilding a FRESH caller (clean subprocess state) and
        # rebinding it (both the closed-over `fc` and the registry). Reassigning `fc` here propagates to the
        # nested _call_dll/_resolve_arg_chain via closure. On rebuild failure, keep the current caller.
        try:                                               # [wrapper-only] rebuild = the only reliable reset (no init fn)
            if _m8_full_resolve:                           # [wrapper-only] §3 gated: only rebuild-reset under corrected-M8 mode (off = old behaviour: no reset)
                _old_fc = fc                               # [wrapper-only] the caller to retire after the new one is up
                fc = _new_fc()                             # [wrapper-only] fresh clean-state FunctionCaller for this TC
                with _dll_registry_lock:                   # [wrapper-only] keep the shared registry consistent
                    _dll_caller_registry[dll_path] = fc    # [wrapper-only] later M7/M8 reuse the fresh caller
                try:                                       # [wrapper-only] stop the retired subprocess (frees its DLL/temp)
                    _old_fc.cleanup()                      # function_caller.py:124 stop old subprocess (state discarded)
                except Exception:                          # [wrapper-only] cleanup failure is non-fatal
                    pass
                logger.debug("M8: TC%d state reset via FunctionCaller rebuild (no <FCname>_init exists)", ti + 1)  # [wrapper-only] trace
        except Exception as e:                             # [wrapper-only] rebuild failed → keep current caller (state may bleed)
            logger.warning("M8: TC%d FunctionCaller rebuild failed (%s) — proceeding with existing state", ti + 1, e)  # [wrapper-only] trace

        current_inputs: Dict[str, str] = dict(tc.get("precond", {}))  # [wrapper-only] running input state seeded from pre-conditions

        # Warm-up: establish initial block state from pre-conditions.
        # For stateful blocks (LowpassT, Srv_Debounce, …) this seeds the
        # internal state before any step actions are applied.
        try:                                               # [wrapper-only] warm-up DLL call to seed stateful-block internal state
            warmup_result = _call_dll(current_inputs)      # [wrapper-only] one DLL call with just the pre-conditions
            logger.debug("M8: TC%d warm-up _call_dll(precond) -> %s", ti + 1, warmup_result)  # [wrapper-only] trace
        except Exception as e:                             # [wrapper-only] warm-up failure is non-fatal
            logger.debug("M8: TC%d warm-up exception: %s", ti + 1, e)  # [wrapper-only] trace

        for si, step in enumerate(tc.get("steps", [])):    # [wrapper-only] iterate steps within the test case (in order)
            wait_ms = step.get("wait")                     # [wrapper-only] the step's Wait directive (µs, or negative = raster count)
            exp_dict = step.get("exp", {})                 # [wrapper-only] the step's Expected Results (M8 targets)

            # Number of DLL ticks this step represents (minimum 1).
            # wait_ms < 0  → abs(wait_ms) rasters ("Wait @" = -1, "Wait N*@" = -N)
            # wait_ms > 0  → absolute µs, divide by raster_time
            if wait_ms is None or wait_ms == 0:            # [wrapper-only] no/zero wait → a single tick
                n_advances = 1                             # [wrapper-only] one DLL call
            elif wait_ms < 0:                              # [wrapper-only] negative encodes a raster count ("Wait N*@")
                n_advances = abs(wait_ms)  # -N = N rasters  # [wrapper-only] N ticks
            else:                                          # [wrapper-only] positive µs → divide by raster_time to get ticks
                n_advances = max(1, int(wait_ms) // max(1, int(raster_time)))  # [wrapper-only] at least 1 tick

            # ── Guard 1: cap DLL ticks per step (wrapper-only; no run.py analog) ──────────
            # A boundary test with a huge Wait (e.g. int32-max µs, or a large negative raster count)
            # otherwise expands to ~107k ticks -> 107k DLL calls for ONE step (observed: a 50 MB log).
            # A stateful block (delay/filter) reaches its final state within a handful of ticks, so
            # capping leaves the M8 verdict unchanged while bounding the loop.
            if n_advances > _M8_MAX_ADVANCE_TICKS:         # [wrapper-only] Guard 1: bound pathological Wait values
                logger.warning("M8: TC%d.S%d Wait=%s -> n_advances=%d exceeds cap %d; capping (block long since settled) to avoid a giant DLL-tick loop",
                               ti + 1, si + 1, wait_ms, n_advances, _M8_MAX_ADVANCE_TICKS)
                n_advances = _M8_MAX_ADVANCE_TICKS         # [wrapper-only] bounded tick count

            # M8 per-step trace
            logger.info("M8: TC%d.S%d: wait=%s -> n_advances=%d, act=%s, exp=%s",  # [wrapper-only] trace
                         ti + 1, si + 1, wait_ms, n_advances, step.get("act", {}), exp_dict)

            # Advance DLL state for (n_advances - 1) ticks WITHOUT applying
            # this step's actions yet.  This simulates the passage of time
            # before the action is triggered.
            for tick in range(n_advances - 1):             # [wrapper-only] advance internal state (n_advances-1) ticks before the action
                result = _call_dll(current_inputs)         # [wrapper-only] tick the DLL with the pre-action inputs
                logger.debug("M8: TC%d.S%d advance-tick %d/%d -> %s",  # [wrapper-only] trace
                              ti + 1, si + 1, tick + 1, n_advances - 1, result)
                if result is None:                         # [wrapper-only] DLL failing this tick → stop advancing
                    logger.debug("M8: TC%d.S%d advance-tick %d -> DLL returned None, stopping advance early",  # [wrapper-only] trace
                                  ti + 1, si + 1, tick + 1)
                    break  # DLL failing — stop advancing to avoid infinite retries  # [wrapper-only] bail out of the advance loop

            # Apply this step's actions to the running input state.
            current_inputs.update(step.get("act", {}))     # [wrapper-only] merge the step's actions into the running input dict (state carried)
            logger.debug("M8: TC%d.S%d current_inputs after applying actions = %s", ti + 1, si + 1, current_inputs)  # [wrapper-only] trace

            # Final DLL call: inputs now include the step's actions.
            actual = _call_dll(current_inputs)             # [wrapper-only] the step's actual DLL output (post-action) to compare vs Expected
            logger.info("M8: TC%d.S%d: DLL actual output = %s", ti + 1, si + 1, actual)  # [wrapper-only] trace

            if not exp_dict:                               # [wrapper-only] step declares no Expected Results → nothing to score
                logger.debug("M8: TC%d.S%d: no expected results -> skipped (not counted)", ti + 1, si + 1)  # [wrapper-only] trace
                continue  # No expected result to check — skip  # [wrapper-only] don't count this step

            total_steps += 1                               # [wrapper-only] this step is checkable → count it in the denominator
            try:                                           # [wrapper-only] record the numeric actual for the constant-output guard
                _step_actuals.append(float(actual))        # [wrapper-only] (None / non-numeric actuals are simply not recorded)
            except (TypeError, ValueError):
                pass                                       # [wrapper-only]
            step_pass = False                              # [wrapper-only] assume FAIL until a target matches
            cmp_detail = ""                                # [wrapper-only] human-readable compare detail for the summary

            # [wrapper-only] compare DLL output to each Expected Results target (run.py:609 uses exact ==; wrapper uses tolerance)
            for exp_name, exp_val_str in exp_dict.items():  # [wrapper-only] a step passes if ANY expected target matches
                try:                                       # [wrapper-only] numeric compare path
                    act_f = float(actual)                  # [wrapper-only] DLL actual output as float
                    exp_f = float(exp_val_str)             # [wrapper-only] expected value as float
                    # Tolerance: ~1% relative for continuous values, BUT a tight 0.5 for
                    # boolean/bit/status outputs (expected exactly 0 or 1) — otherwise the
                    # ±1 absolute slack lets a bit's 0 trivially "match" 1 (and vice-versa),
                    # giving function-call/bit weakspots a meaningless perfect M8.
                    if exp_f in (0.0, 1.0):                # [wrapper-only] boolean/bit output → tight tolerance
                        tol = 0.5                          # [wrapper-only] 0.5 so 0 can't "match" 1
                    else:                                  # [wrapper-only] continuous value → relative tolerance
                        tol = 1.0 + 0.01 * abs(exp_f)      # [wrapper-only] ±1 absolute for small, ~1% relative for large
                    diff = abs(act_f - exp_f)              # [wrapper-only] absolute difference
                    cmp_detail = f"{exp_name}: actual={act_f} expected={exp_f} diff={diff:.4f} tol={tol:.4f}"  # [wrapper-only] compare detail
                    if diff <= tol:                        # [wrapper-only] within tolerance → this target passes
                        step_pass = True                   # [wrapper-only] mark the step PASS
                        break                              # [wrapper-only] one matching target is enough
                except (TypeError, ValueError):            # [wrapper-only] non-numeric actual/expected → string compare
                    cmp_detail = f"{exp_name}: actual={actual!r} expected={exp_val_str!r} (string compare)"  # [wrapper-only] compare detail
                    if actual is not None and actual == str(exp_val_str):  # run.py:609 if expected == actual (string form)
                        step_pass = True                   # [wrapper-only] string-equal → step PASS
                        break                              # [wrapper-only]

            # M8 per-step PASS/FAIL trace + record
            if step_pass:                                  # [wrapper-only] step matched a target
                total_passes += 1                          # [wrapper-only] M8 numerator++
                logger.info("M8: TC%d.S%d -> PASS (%s)", ti + 1, si + 1, cmp_detail)  # [wrapper-only] trace
            else:
                logger.info("M8: TC%d.S%d -> FAIL (%s)", ti + 1, si + 1, cmp_detail)  # [wrapper-only] trace
            step_results.append(f"TC{ti+1}.S{si+1}={'PASS' if step_pass else 'FAIL'} ({cmp_detail})")  # [wrapper-only] record per-step result for the summary

    if total_steps == 0:                                   # [wrapper-only] no checkable steps → neutral 0.5
        logger.warning("M8: no steps with expected results -> score=0.5 default")  # [wrapper-only] trace
        return 0.5, "No steps with expected results to check — using default 0.5"  # [wrapper-only] neutral default

    # ── CONSTANT-OUTPUT GUARD (gated by FULL_RESOLVE) ────────────────────────────────────────────────
    # If the DLL produced the SAME numeric output on EVERY checked step (≥2 steps), the test never exercised
    # the weakspot's input→output response — so any "passes" are COINCIDENTAL (the LLM's expected happened to
    # equal that constant, e.g. GbxFan/GearLvr where the test under-stimulates the block). We cannot confirm
    # functional correctness from a frozen output, so flag M8=0 (unverifiable), per the flag-0 decision.
    if _m8_full_resolve and total_steps >= 2 and len(_step_actuals) >= 2:  # [wrapper-only] need ≥2 checked numeric steps to judge constancy
        _distinct = len({round(a, 6) for a in _step_actuals})  # [wrapper-only] distinct DLL outputs across checked steps
        if _distinct <= 1:                                 # [wrapper-only] output never changed → weakspot not exercised
            logger.info("M8: CONSTANT DLL output (%.6g) across all %d checked step(s) — test under-stimulates the weakspot; the %d/%d 'passes' are coincidental -> M8=0 (unverifiable)",
                         _step_actuals[0], len(_step_actuals), total_passes, total_steps)  # [wrapper-only] trace
            return 0.0, (f"M8=0 (constant DLL output {_step_actuals[0]:.6g} across {len(_step_actuals)} checked steps "
                         f"— test under-stimulates the weakspot; {total_passes}/{total_steps} passes coincidental)")  # [wrapper-only] honest 0

    score = total_passes / total_steps                     # [wrapper-only] M8 = sequential-DLL pass rate (run.py computes no score)
    logger.info("M8: RESULT score=%.4f (%d/%d steps pass)", score, total_passes, total_steps)  # [wrapper-only] trace
    for sr in step_results:                                # [wrapper-only] log each step's PASS/FAIL detail
        logger.info("M8: step result: %s", sr)             # [wrapper-only] trace
    # [wrapper-only] RICH feedback: fold the top failing steps' DLL-actual-vs-expected into the detail so GEPA's
    # reflection LLM sees the exact numeric discrepancy (the "why"), not just the pass ratio. Only the FAILing
    # steps (the actionable ones); capped so the feedback stays focused (not a raw-log dump).
    _fails = [sr for sr in step_results if "=FAIL" in sr][:4]  # [wrapper-only] top few failing steps (DLL actual vs expected)
    _detail = f"M8: {total_passes}/{total_steps} steps pass sequential DLL check"  # [wrapper-only]
    if _fails:                                              # [wrapper-only] append the concrete discrepancies
        _detail += " | failing steps (DLL ACTUAL vs your EXPECTED): " + "; ".join(_fails)  # [wrapper-only]
    return score, _detail                                  # [wrapper-only] final M8 score + rich, actionable detail


# ═══════════════════════════════════════════════════════════════════
# COMPOSITE METRIC
# ═══════════════════════════════════════════════════════════════════
# [wrapper-only] scoring layer — run.py GENERATES test cases and writes an XLSM but computes NO score.
# Runs the full M1..M8 suite over one (example, prediction) pair; M2/M7/M8 reuse run.py _step_4a via M7/M8.
def _compute_all_components(   # [wrapper-only] run all M1..M8 on (example, prediction) — no run.py analog
    example: dspy.Example,
    prediction: dspy.Prediction,
    use_pre_postprocess: bool = True,
) -> Dict[str, Any]:
    """Compute all metric components and return scores + details.

    use_pre_postprocess=True (default):
        M1/M3/M4/M6 evaluated on test_cases_pre_postprocess (raw Prompt 2 output).
        Falls back to test_cases when test_cases_pre_postprocess is absent (2-prompt pipeline).
        M2 always on final test_cases — signal names are populated by Prompt 4 solving.
        M7 always on final test_cases — DLL needs solved variable values from Prompt 4.

    use_pre_postprocess=False:
        All metrics on final test_cases (legacy / explicit override).
    """
    # Final output is always test_cases — used by M2 and M7 regardless of flag
    final_tc_str = getattr(prediction, "test_cases", "[]")  # [wrapper-only] final pipeline JSON from forward() (run.py writes an XLSM, not this string)

    # Pre-postprocess output (Prompt 2 raw) — used by M1/M3/M4/M6 when flag is set
    if use_pre_postprocess:                                # [wrapper-only] staged-metric mode: score early metrics on Prompt-2 raw
        pre_tc_str = getattr(prediction, "test_cases_pre_postprocess", final_tc_str)  # [wrapper-only] Prompt-2 raw output; fall back to final for 2-prompt pipelines
    else:                                                  # [wrapper-only] legacy override: everything on final output
        pre_tc_str = final_tc_str                          # [wrapper-only] use final JSON for all metrics

    # [wrapper-only] pull each metric input off the Example (run.py holds these live in eq_ctx; wrapper JSON-serializes them onto the Example)
    equation_json        = getattr(example, "equation_json", "{}")   # [wrapper-only] eq_ctx.filtered_dict["equation"] serialized
    signalsheet          = getattr(example, "signalsheet_data", "")  # [wrapper-only] signalsheet CSV/text (M1/M2/M6 input)
    model_interface      = getattr(example, "model_interface_data", "")  # [wrapper-only] model interface data (M2/M7 signal resolution)
    raster_time          = int(getattr(example, "raster_time", "100000"))  # [wrapper-only] dT raster in µs (M3/M8), default 100000
    dll_path             = getattr(example, "dll_path", "")          # [wrapper-only] path to the weakspot DLL (M7/M8 backtrace)
    equation_results_json = getattr(example, "equation_results_json", "[]")  # [wrapper-only] eq_ctx.equation_results serialized (M7/M8)
    weakspot_op_key      = getattr(example, "weakspot_op_key", "")   # [wrapper-only] "$opN" key selecting the weakspot op entry
    const_values         = getattr(example, "const_values", "{}")   # [wrapper-only] SystemConstant values (M8 substitution)
    relations_json       = getattr(example, "relations_json", "{}")  # [wrapper-only] eq_ctx.relations serialized (M8)
    instance_names       = getattr(example, "instance_names", "")   # [wrapper-only] eq_ctx.instance_names (M1 I/O classification)

    # [wrapper-only] Expert-curated Required-I/O KB (utils/required_io.json), keyed by fc#ws. None when
    # uncurated → M1 keeps instance_names, M2 skips completeness (no penalty). See utils/required_io.py.
    _req_io = _resolve_required_io(getattr(example, "fc_name", ""), getattr(example, "weakspot_id", ""), weakspot_op_key)  # [wrapper-only] op_key disambiguates duplicate fc#ws (a2mamg getBit $op4 vs putBit $op5)
    _req_outputs = (_req_io or {}).get("required_outputs") or None  # [wrapper-only] expert output override for M1 (None -> instance_names)
    if _req_io:  # [wrapper-only] trace which weakspot has a curated KB entry
        logger.info("_compute_all_components: required_io KB HIT for %s#%s -> %s",
                     getattr(example, "fc_name", "?"), getattr(example, "weakspot_id", "?"), _req_io)

    # [wrapper-only] trace banner + which inputs/mode this scoring pass uses (run.py logs per-ws, not per-metric)
    logger.info("=" * 70)
    logger.info("_compute_all_components: fc_name=%s weakspot_id=%s weakspot_op_key=%s raster_time=%s dll_path=%s",
                 getattr(example, "fc_name", "?"), getattr(example, "weakspot_id", "?"),
                 weakspot_op_key, raster_time, dll_path)
    logger.info("_compute_all_components: use_pre_postprocess=%s", use_pre_postprocess)
    logger.debug("_compute_all_components: pre_tc_str (Prompt 2 raw / M1,M3,M4,M6 input) = %s", pre_tc_str)
    logger.debug("_compute_all_components: final_tc_str (final JSON / M2,M7,M8 input) = %s", final_tc_str)

    # M1/M3/M4/M6 evaluated on Prompt 2 raw output (pre_tc_str)
    m3, m3_d = _m3_structural_validity(pre_tc_str, raster_time)  # [wrapper-only] M3 structural validity (rule-based) on Prompt-2 raw
    m1, m1_d = _m1_io_classification(pre_tc_str, equation_json, signalsheet, instance_names, required_outputs=_req_outputs)  # [wrapper-only] M1 on Prompt-2 raw (expert output override when curated)
    m4, m4_d = _m4_test_coverage(pre_tc_str, equation_json, weakspot_op_key)  # [wrapper-only] M4 KB-coverage (LLM judge) on Prompt-2 raw
    # M6 (boundary values) REMOVED 2026-07-09 — was weight 0 and unused; no longer computed/returned/scored.

    # M2 always on final output — checks signal names produced by Prompt 4 / post-processing
    m2, m2_d = _m2_signal_resolution(final_tc_str, signalsheet, model_interface, equation_json, required_io=_req_io)  # [wrapper-only] M2 name-resolution + input/precond completeness (final JSON)

    # M7: prefer inline 1/N score from forward() (captures run.py's actual retry count).
    # Falls back to post-hoc _m7_dll_pass_rate when inline score is absent
    # (e.g., DLL unavailable, or prediction from an older pipeline).
    _m7_inline = getattr(prediction, "m7_score", None)     # [wrapper-only] inline 1/N score forward() captured from run.py's actual Prompt-4 retry count
    if _m7_inline is not None:                             # [wrapper-only] prefer the inline score (reflects the real retry loop)
        m7 = float(_m7_inline)                             # [wrapper-only] use forward()'s inline M7 score
        m7_d = getattr(prediction, "m7_details", f"DLL inline: {m7:.3f}")  # [wrapper-only] inline detail string (default if absent)
        logger.info("M7: ===== DLL Pass Rate (using INLINE score from forward()) =====")  # [wrapper-only] trace
        logger.info("M7: RESULT score=%.4f (%s) [inline, from Prompt 4 retry loop]", m7, m7_d)  # [wrapper-only] trace
    else:                                                  # [wrapper-only] no inline score → recompute post-hoc
        m7, m7_d = _m7_dll_pass_rate(                      # [wrapper-only] M7 per-step DLL pass-rate (core reuses run.py _step_4a:546-600 stateless)
            final_tc_str, equation_json, dll_path, signalsheet, model_interface,
            equation_results_json=equation_results_json,
            weakspot_op_key=weakspot_op_key,
        )

    # M8 always on final output — verifies the complete post-processed test case
    # sequentially against the DLL, catching dedup/post-proc corruption.
    m8, m8_d = _m8_functional_correctness(                 # [wrapper-only] M8 sequential functional correctness (core reuses run.py _step_4a:546-600 with state carried)
        final_tc_str, equation_json, dll_path, signalsheet, model_interface,
        equation_results_json=equation_results_json,
        raster_time=raster_time,
        weakspot_op_key=weakspot_op_key,
        const_values_json=const_values,
        relations_json=relations_json,
        raw_test_cases_json=pre_tc_str,   # [wrapper-only] point-9: raw Prompt-2 test for the Parameter-arg fallback
        # [wrapper-only] §5 gate-align: when a curated KB steers the test to an INTERMEDIATE output, the M8
        # wrong-I/O gate must key on THAT output (else it always gates "no measurable output"). Use the KB
        # required_outputs override (same source M1 uses) as the measurable-output set; else physical instance_names.
        instance_names=(",".join(_req_outputs) if _req_outputs else instance_names),  # [wrapper-only] declared outputs for the M8 wrong-I/O gate
    )

    # [wrapper-only] trace: dump every M1..M8 score with its composite weight (weights defined at module top)
    logger.info("=" * 70)
    logger.info("_compute_all_components: SUMMARY OF ALL METRIC SCORES")
    logger.info("  M1 (I/O Classification,  20%%) = %.4f | %s", m1, m1_d)
    logger.info("  M2 (Signal Resolution,   15%%) = %.4f | %s", m2, m2_d)
    logger.info("  M3 (Structural Validity, 15%%) = %.4f | %s", m3, m3_d)
    logger.info("  M4 (Test Coverage,       20%%) = %.4f | %s", m4, m4_d)
    logger.info("  M7 (DLL Pass Rate,       15%%) = %.4f | %s", m7, m7_d)
    logger.info("  M8 (Functional Correct., 15%%) = %.4f | %s", m8, m8_d)
    logger.info("=" * 70)

    # [wrapper-only] return every component score + detail string (the composite metrics weight these).
    # required_outputs/instance_names are surfaced so the GEPA feedback can name the TARGET output signal
    # (esp. when a KB steers Prompt 2 to an INTERMEDIATE output) — see mbd_metric_with_feedback.
    return {
        "m1": m1, "m1_details": m1_d,
        "m2": m2, "m2_details": m2_d,
        "m3": m3, "m3_details": m3_d,
        "m4": m4, "m4_details": m4_d,
        "m7": m7, "m7_details": m7_d,
        "m8": m8, "m8_details": m8_d,
        "required_outputs": _req_outputs,   # [wrapper-only] KB output override (None → physical instance_names)
        "instance_names": instance_names,   # [wrapper-only] physical output signal(s) from the dataset
    }


# [wrapper-only] composite scoring — no run.py analog (run.py emits an XLSM, never a score).
def mbd_testcase_metric(   # [wrapper-only] composite metric for dspy.Evaluate (weighted M1..M8 total) — no run.py analog
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
) -> float:
    """Primary metric for DSPy optimizers. Returns float [0,1].

    M1/M3/M4/M6 evaluated on raw Prompt 2 output (test_cases_pre_postprocess).
    M2+M7 evaluated on final pipeline output (test_cases).
    Falls back to final for all if test_cases_pre_postprocess is absent.
    """
    # use_pre_postprocess=True (default): M1/M3/M4/M6 from Prompt 2 raw, M2+M7 from final
    scores = _compute_all_components(example, prediction)  # [wrapper-only] run the full M1..M8 suite on this (example, prediction) pair
    # [wrapper-only] weighted composite — weights sum to 1.0 (M1=.20 M2=.15 M3=.15 M4=.20 M6=0 M7=.15 M8=.15)
    total = (
        WEIGHT_M1_IO       * scores["m1"]                  # [wrapper-only] 0.20 * M1 I/O classification
        + WEIGHT_M2_SIGNAL     * scores["m2"]              # [wrapper-only] 0.15 * M2 signal resolution
        + WEIGHT_M3_STRUCTURAL * scores["m3"]              # [wrapper-only] 0.15 * M3 structural validity
        + WEIGHT_M4_COVERAGE   * scores["m4"]              # [wrapper-only] 0.20 * M4 KB coverage
        + WEIGHT_M7_DLL        * scores["m7"]              # [wrapper-only] 0.15 * M7 DLL pass rate
        + WEIGHT_M8_FUNCTIONAL * scores["m8"]              # [wrapper-only] 0.15 * M8 functional correctness
    )
    return total                                           # [wrapper-only] composite score in [0,1] for dspy.Evaluate


# [wrapper-only] Classify the M8 feedback line so GEPA is not misdirected. The rebuilt M8 is legitimately
# 0 for STRUCTURAL reasons the prompts cannot fix (wrong-I/O gate, weakspot fn not exported in the DLL,
# constant-output guard = test under-stimulates) — an unconditional "[WARN] steps inconsistent / post-proc
# corrupted" on those would push GEPA to rewrite a healthy prompt. So: [WARN] ONLY when the DLL actually RAN
# and failed (detail carries "steps pass"); otherwise [INFO] naming the structural reason (do-not-chase).
def _m8_feedback_line(m8_score: float, m8_details: str, solver: bool = False):  # [wrapper-only]
    _d = m8_details or ""
    if "steps pass" in _d:                                  # [wrapper-only] real DLL run → actionable warning on failure
        if m8_score < 0.5:
            return ("[WARN] Final test cases fail the sequential DLL check - post-processing "
                    "(dedup/MUX filter) may have corrupted input values") if solver else \
                   ("[WARN] Final test cases fail the sequential DLL check - generated steps may be "
                    "functionally inconsistent")
        return None                                        # [wrapper-only] DLL ran and mostly passed → no warning
    if _d.startswith("M8=0") or "not exported" in _d or "constant DLL output" in _d:  # [wrapper-only] structural 0
        _extra = " (the equation solver cannot affect this)" if solver else ""  # [wrapper-only]
        return f"[INFO] M8 not scorable here (structural, not a prompt defect){_extra}: {_d}"  # [wrapper-only] do-not-chase
    return None                                            # [wrapper-only] neutral/other → nothing


# [wrapper-only] Feedback text is written in the VOCABULARY of the prompt that reads it, so the GEPA reflection
# LLM improves the right prompt in its own language. Each metric block is ROLE (what the metric checks + what a
# perfect 1.0 needs) -> WHAT HAPPENED (the metric's own failure log) -> TO SCORE 1.0 (the concrete fix).
# All strings kept in sync with the finalized M1..M8 code. Continuation lines are indented to sit under "ROLE:".

# ── Prompt-2 language (test-case generator: Pre-Condition / Actions / Wait / Expected Results / Detailed
#    Explanation / instance_names). M1, M3, M4 are PRIMARILY driven here; M2 naming is partly driven here. ──
_ROLE_P2 = {
    "m1": ("M1 = OUTPUT-ROLE correctness. It checks 3 things on EVERY test case:\n"
           "       (1) the `Expected Results` CHECK the weakspot's real output signal(s) - the exact name(s) in your\n"
           "           `instance_names` input ('use this Expected Results') - not an invented name like 'out'/'bitValue', not an input;\n"
           "       (2) that output signal is NEVER placed in a `Pre-Condition` or an `Actions` line (you VERIFY it, you don't SET it);\n"
           "       (3) fixed constants (SystemConstant / Literal / dT) are NEVER set in `Actions` or checked in `Expected Results`\n"
           "           (they may only be stated in a `Pre-Condition` for reference).\n"
           "       Perfect 1.0 = all 3 hold for every test case (a wrong-output or output-as-input mistake is weighted extra)."),
    "m3": ("M3 = STRUCTURAL validity of every step/test case. It checks:\n"
           "       (1) each test case has a non-empty `Detailed Explanation`;\n"
           "       (2) every `Wait <time>` is a whole-number multiple of the raster dT (or '@' for one raster);\n"
           "       (3) every value in `Actions` and `Expected Results` is a plain number (no expressions, no variable names);\n"
           "       (4) a signal is NOT in both `Actions` and `Expected Results` of the SAME step (don't check a value you just set);\n"
           "       (5) a signal is NOT in both `Pre-Condition` and `Expected Results` (don't verify a value you pinned).\n"
           "       Setting a signal in `Pre-Condition` then driving it in `Actions` IS allowed (initialize-then-drive).\n"
           "       Checks (4) and (5) each cost 4x a normal check. Perfect 1.0 = all structural checks pass on all test cases."),
    "m4": ("M4 = test COVERAGE, graded by an LLM judge. It reads your FULL test cases (Pre-Conditions, Steps,\n"
           "       Expected Results, Detailed Explanation) and checks they COVER every required test-guideline for this\n"
           "       weakspot - each guideline needs a test whose `Pre-Condition`/`Actions` set up the scenario AND whose\n"
           "       `Expected Results` show the expected behaviour. Perfect 1.0 = every guideline has a covering test case."),
    "m2": ("M2 = signal-NAME validity (+ required-I/O completeness). The part YOU drive: every signal name you write in\n"
           "       `Pre-Condition`/`Actions`/`Expected Results` must be a REAL signal name (from the signal sheet / model\n"
           "       interface / equation) - never a placeholder like 'out'/'output'/'signal'/'input'."),
}
_FIX_P2 = {
    "m1": ("Put the exact output-signal name(s) from your `instance_names` input into EACH test case's `Expected Results`\n"
           "       as a value check ('<signal> = <value>'). Remove that output from any `Pre-Condition`/`Actions`. Keep\n"
           "       SystemConstants/Literals/dT out of `Actions` and `Expected Results`."),
    "m3": ("Make every `Wait` a multiple of dT (or '@'); write only plain numbers in `Actions`/`Expected Results`; give\n"
           "       every test case a `Detailed Explanation`; and never repeat a signal across Actions+Expected or\n"
           "       Pre-Condition+Expected of the same test case."),
    "m4": ("For each NOT-COVERED guideline below, add ONE dedicated test case: set up its scenario in `Pre-Condition`/`Actions`\n"
           "       and assert the expected outcome in `Expected Results`. The judge reads the actual steps, not just the explanation."),
    "m2": "Use EXACT signal-sheet names everywhere in your test cases - no placeholders or abstract names.",
}
# [wrapper-only] Why M2/M8/M7 are NOT primarily the test generator's to fix (shown last in the Prompt-2 feedback).
_P2_SECONDARY_NOTE = {
    "m2": ("Only PARTLY yours: the NAMES are yours (above), but any MISSING required physical inputs get resolved "
           "downstream (Prompt 4 back-trace + post-processing), so a low M2 here is only partly the test generator's fault."),
    "m8": ("Mostly NOT yours: M8 runs the FINAL test cases through the DLL, and the numeric expected values come mainly "
           "from the Prompt-4 solve. You influence M8 only by checking the right output signal and keeping a clean structure."),
    "m7": "NOT yours: M7 is the Prompt-4 equation solve. There is nothing for the test generator to change here.",
}

# ── Prompt-4 language (equation solver: equation_chain / target_value / const_values / variable_assignments,
#    integer values for Variable(...)/Parameter(...)/SystemConstant(...), LHS=RHS). M7, M8 primary; M2 naming partly. ──
_ROLE_P4 = {
    "m8": ("M8 = FUNCTIONAL correctness. It runs the final test cases through the REAL DLL step-by-step and checks the\n"
           "       DLL's actual output equals the `Expected Results` value at each step. YOUR assignments set the numeric\n"
           "       inputs the DLL runs with, so wrong values surface here as a mismatch. Perfect 1.0 = DLL agrees on every step."),
    "m7": ("M7 = SOLVE success. The pipeline BACKTRACES each intermediate/output signal in the test DOWN through the\n"
           "       `equation_chain` to the actual (physical) input signals. At EVERY backtrace step you must output\n"
           "       `variable_assignments` (integer values for each Variable(...)/Parameter(...)/SystemConstant(...)) that make\n"
           "       THAT equation evaluate to its `target_value` - i.e. LHS = RHS holds at every step of the backtrace -\n"
           "       ideally on the FIRST attempt (first-try = 1.0; a passing retry scores less; never passing = 0).\n"
           "       A function-call weakspot has no arithmetic to solve, so M7 is a fixed neutral 0.5."),
    "m2": ("M2 = signal-NAME validity + required-I/O completeness (computed on the FINAL test cases). name-validity: every\n"
           "       signal name used must be a REAL signal name (signal sheet / model interface / equation), not a placeholder\n"
           "       like 'out'/'output'/'signal'/'input'. completeness: every KB-required input must be driven and every required\n"
           "       precondition set. When a KB entry exists M2 = (name_validity + completeness) / 2, else just name_validity.\n"
           "       WHAT YOU DRIVE: the NAMES in your `variable_assignments` must be the EXACT Variable/Parameter/SystemConstant\n"
           "       names from the `equation_chain`. The MISSING required inputs/preconds are the test SETUP (Prompt 2), not yours."),
}
_FIX_P4 = {
    "m8": ("Recompute your assignments so the DLL's actual output matches the `Expected Results` - compare the ACTUAL-vs-EXPECTED\n"
           "       steps below and adjust the integer values feeding those steps."),
    "m7": ("Assign an integer to EVERY Variable/Parameter/SystemConstant in the `equation_chain` so it evaluates to `target_value`;\n"
           "       keep each Parameter the SAME across calls; never assign Local(...)/Operation(...)/Function(...)."),
    "m2": "Use the exact signal names from the `equation_chain` in your `variable_assignments`.",
}


def _impact_sorted(scores: dict, keys):  # [wrapper-only] order metric keys by weight*(1-score) desc = biggest fitness gap first
    _w = {"m1": WEIGHT_M1_IO, "m2": WEIGHT_M2_SIGNAL, "m3": WEIGHT_M3_STRUCTURAL,
          "m4": WEIGHT_M4_COVERAGE, "m7": WEIGHT_M7_DLL, "m8": WEIGHT_M8_FUNCTIONAL}  # [wrapper-only] live weights
    return sorted(keys, key=lambda k: _w[k] * (1.0 - float(scores.get(k) or 0.0)), reverse=True)  # [wrapper-only] largest gap first


# [wrapper-only] GEPA feedback metric — no run.py analog. INVARIANT: dspy GEPA asserts the returned
# fb.score EQUALS the module-level composite (gepa_utils.py: assert fb["score"] == module_score), so
# every predictor gets the SAME composite `total`; only the per-predictor FEEDBACK TEXT differs.
def mbd_metric_with_feedback(   # [wrapper-only] GEPA feedback metric (score==total + per-predictor text) — no run.py analog
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> Any:
    """GEPA-compatible metric with staged dispatch by predictor name.

    Signature follows GEPA's 5-parameter pattern:
      - pred_name is None → return float (used during baseline eval)
      - pred_name is set  → return dspy.Prediction(score=float, feedback=str)
        with predictor-specific GEPA signal (M2 + M8 shared across stages):
          context_extractor    → full composite score + all metric details
          testcase_generator   → M1/M3/M4/M6 (Prompt 2 raw) + M2/M8 (final)
          equation_solver_dll  → M7 (Prompt 4) + M2/M8 (final)
    """
    _ = pred_trace  # unused                               # [wrapper-only] accept GEPA's pred_trace arg but don't use it

    # Single call: M1/M3/M4/M6 from Prompt 2 raw, M2+M7 from final pipeline output.
    # This avoids a redundant second DLL call compared to calling _compute_all_components twice.
    scores = _compute_all_components(example, prediction, use_pre_postprocess=True)  # [wrapper-only] one M1..M8 pass (avoids a duplicate DLL call)
    # [wrapper-only] SAME weighted composite as mbd_testcase_metric — weights sum to 1.0
    total = (
        WEIGHT_M1_IO       * scores["m1"]                  # [wrapper-only] 0.20 * M1 I/O classification
        + WEIGHT_M2_SIGNAL     * scores["m2"]              # [wrapper-only] 0.15 * M2 signal resolution
        + WEIGHT_M3_STRUCTURAL * scores["m3"]              # [wrapper-only] 0.15 * M3 structural validity
        + WEIGHT_M4_COVERAGE   * scores["m4"]              # [wrapper-only] 0.20 * M4 KB coverage
        + WEIGHT_M7_DLL        * scores["m7"]              # [wrapper-only] 0.15 * M7 DLL pass rate
        + WEIGHT_M8_FUNCTIONAL * scores["m8"]              # [wrapper-only] 0.15 * M8 functional correctness
    )

    if pred_name is None:                                  # [wrapper-only] GEPA baseline eval passes no predictor name
        return total   # baseline eval — no GEPA feedback needed  # [wrapper-only] plain float, same value as mbd_testcase_metric

    # [wrapper-only] shared: the TARGET output signal Prompt 2 must verify (KB override when steered to an
    # INTERMEDIATE output, else the physical instance_names) — named explicitly so GEPA steers correctly.
    _target = ", ".join(scores.get("required_outputs") or []) or (scores.get("instance_names") or "(the weakspot output)")
    # [wrapper-only] INVARIANT (all branches): dspy GEPA asserts fb.score == module composite (gepa_utils.py
    # asserts fb["score"] == module_score); a rescaled per-predictor score raises AssertionError and the
    # predictor is never optimized. So gepa_score is ALWAYS `total`; only the FEEDBACK TEXT is tailored.

    def _num(k):                                           # [wrapper-only] metric score as float (0.0 if missing)
        return float(scores.get(k) or 0.0)

    _m7d = scores.get("m7_details") or ""                  # [wrapper-only] M7 detail (used to detect function-call weakspots)
    _m7_fncall = ("function-call" in _m7d or "function call" in _m7d or _m7d.startswith("N/A"))  # [wrapper-only] no arithmetic to solve

    def _block(k, role, fix, solver=False, note=None):     # [wrapper-only] ROLE -> WHAT HAPPENED -> TO SCORE 1.0 (M8 -> STATUS); +optional NOTE
        _tag = "M" + k[1]                                  # [wrapper-only] "M1".."M8"
        if _num(k) >= 0.999:                               # [wrapper-only] perfect → one-liner that still states the role (GEPA sees why it's 1.0)
            _lines = [f"  {_tag}={_num(k):.2f} OK  (ROLE: {role})"]
        else:
            _lines = [f"  {_tag}={_num(k):.2f}",
                      f"     ROLE: {role}",
                      f"     WHAT HAPPENED (metric log): {scores.get(k + '_details')}"]
            if k == "m8":                                  # [wrapper-only] M8: real-fail [WARN] vs structural [INFO] (do-not-chase)
                _fb = _m8_feedback_line(scores["m8"], scores["m8_details"], solver=solver)
                _lines.append(f"     STATUS: {_fb}" if _fb else f"     TO SCORE 1.0: {fix}")
            elif k == "m7" and _m7_fncall:                 # [wrapper-only] function-call weakspot → nothing to solve
                _lines.append("     TO SCORE 1.0: Function-call weakspot — no arithmetic to solve (M7 is a fixed neutral 0.5, not a defect). Correctness is measured by M8; nothing to change here.")
            else:
                _lines.append(f"     TO SCORE 1.0: {fix}")
        if note:                                           # [wrapper-only] "why this isn't (mainly) yours" for secondary metrics
            _lines.append(f"     NOTE: {note}")
        return _lines

    if "testcase_generator" in pred_name:                  # [wrapper-only] Prompt-2 predictor → it WRITES the test cases
        gepa_score = total                                 # [wrapper-only] INVARIANT: score == composite total
        parts = [
            f"PROMPT 2 (test-case generator) - composite this example = {total:.3f} (0=worst, 1=best).",  # [wrapper-only] magnitude
            "You WRITE the test cases: `Pre-Condition`, `Steps` (Actions / Wait / Expected Results) and the `Detailed Explanation`.",  # [wrapper-only] ownership, in Prompt-2 vocabulary
            f"TARGET output signal to CHECK in every test's `Expected Results`: {_target}",  # [wrapper-only] explicit target
            "PRIMARY - you drive these directly; fix weakest-first:",  # [wrapper-only]
        ]
        for _k in _impact_sorted(scores, ["m1", "m3", "m4"]):  # [wrapper-only] the 3 metrics Prompt 2 PRIMARILY drives, biggest gap first
            parts += _block(_k, _ROLE_P2[_k], _FIX_P2[_k])
        parts.append("NOT primarily driven by Prompt 2 (shown for the whole-chain picture - don't over-fit to these):")  # [wrapper-only]
        _m8_role_p2 = ("M8 runs the FINAL test cases through the real DLL step-by-step and checks the DLL's actual output equals the\n"
                       "       `Expected Results` value at each step. Perfect 1.0 = the DLL agrees on every step.")  # [wrapper-only] neutral M8 role
        parts += _block("m2", _ROLE_P2["m2"], _FIX_P2["m2"], note=_P2_SECONDARY_NOTE["m2"])  # [wrapper-only] naming yours, completeness downstream
        parts += _block("m8", _m8_role_p2, _FIX_P4["m8"], solver=False, note=_P2_SECONDARY_NOTE["m8"])  # [wrapper-only] mostly Prompt 4
        _m7_role_p2 = ("function-call weakspot - no arithmetic, fixed neutral 0.5" if _m7_fncall
                       else "the Prompt-4 equation solve - did the DLL reach target_value on the first try")  # [wrapper-only] brief M7 role for P2
        parts.append(f"  M7={_num('m7'):.2f}  ROLE: {_m7_role_p2}.")  # [wrapper-only] one line (P7 = Prompt 4's)
        parts.append(f"     NOTE: {_P2_SECONDARY_NOTE['m7']}")  # [wrapper-only]
        feedback = "\n".join(parts)                         # [wrapper-only]

    elif "equation_solver_dll" in pred_name or "equation_solver_retry" in pred_name:  # [wrapper-only] Prompt-4 solver/retry → it OWNS the numeric solve
        gepa_score = total                                 # [wrapper-only] INVARIANT: score == composite total
        _stage = "retry" if "equation_solver_retry" in pred_name else "first-attempt"  # [wrapper-only]
        parts = [
            f"PROMPT 4 equation solver ({_stage}) - composite this example = {total:.3f} (0=worst, 1=best).",  # [wrapper-only]
            "You OUTPUT `variable_assignments`: an integer for each Variable(...)/Parameter(...)/SystemConstant(...) in the",  # [wrapper-only] Prompt-4 vocabulary
            "`equation_chain` so it evaluates to `target_value` (LHS = RHS). You drive M7 and M8 (and the NAMES in M2); you do",  # [wrapper-only]
            "NOT choose which signals the test drives (that is Prompt 2). Fix weakest-first:",  # [wrapper-only]
        ]
        for _k in _impact_sorted(scores, ["m7", "m8"]):     # [wrapper-only] the two metrics Prompt 4 drives, biggest gap first
            parts += _block(_k, _ROLE_P4[_k], _FIX_P4[_k], solver=True)
        # [wrapper-only] M2 naming IS the solver's (variable_assignments); the MISSING-input part is Prompt 2's
        _m2d = (scores['m2_details'] or "").replace("MISSING required", "[Prompt-2 setup gap, not the solver] MISSING required")  # [wrapper-only]
        if _num("m2") >= 0.999:                            # [wrapper-only]
            parts.append(f"  M2={_num('m2'):.2f} OK  (ROLE: {_ROLE_P4['m2']})")
        else:
            parts += [f"  M2={_num('m2'):.2f}",
                      f"     ROLE: {_ROLE_P4['m2']}",
                      f"     WHAT HAPPENED (metric log): {_m2d}",
                      f"     TO SCORE 1.0: {_FIX_P4['m2']}  (the MISSING-input part is Prompt 2's job, not yours.)"]
        feedback = "\n".join(parts)                         # [wrapper-only]

    elif "context_extractor" in pred_name:                 # [wrapper-only] Prompt-1 predictor → context feeds BOTH Prompt 2 and Prompt 4
        gepa_score = total                                 # [wrapper-only] INVARIANT: score == composite total
        parts = [
            f"PROMPT 1 (context extractor) - composite this example = {total:.3f} (0=worst, 1=best).",  # [wrapper-only]
            "Your context is the input to BOTH Prompt 2 (test cases) and Prompt 4 (solve). Sharper `Overview`,",  # [wrapper-only] Prompt-1 vocabulary
            "`Inputs That Can Be Altered`, and `Outputs That Can Be Checked` -> fewer downstream errors. Full picture:",  # [wrapper-only]
            (f"  M1={_num('m1'):.2f} M2={_num('m2'):.2f} M3={_num('m3'):.2f} "
             f"M4={_num('m4'):.2f} M7={_num('m7'):.2f} M8={_num('m8'):.2f}"),  # [wrapper-only]
            f"TARGET output signal(s) your context must make unmistakable: {_target}",  # [wrapper-only]
        ]
        # [wrapper-only] Show what Prompt 2 is getting wrong (Prompt-2 language) so Prompt 1 can sharpen the context it reads.
        _p2_fail = [k for k in _impact_sorted(scores, ["m1", "m3", "m4", "m2"]) if _num(k) < 0.999]
        parts.append("")
        parts.append("What PROMPT 2 (test cases) is getting wrong - make the context it reads clearer on these:")
        if _p2_fail:
            for _k in _p2_fail:
                parts += [f"  M{_k[1]}={_num(_k):.2f}",
                          f"     ROLE: {_ROLE_P2[_k]}",
                          f"     WHAT HAPPENED (metric log): {scores.get(_k + '_details')}"]
        else:
            parts.append("  (nothing - Prompt 2 metrics are all clean.)")
        # [wrapper-only] Show what Prompt 4 is getting wrong (Prompt-4 language).
        _p4_fail = [k for k in _impact_sorted(scores, ["m7", "m8"]) if _num(k) < 0.999]
        parts.append("")
        parts.append("What PROMPT 4 (equation solve) is getting wrong - make the context it reads clearer on these:")
        if _p4_fail:
            for _k in _p4_fail:
                if _k == "m8":                             # [wrapper-only] M8 structural INFO vs real
                    _fb = _m8_feedback_line(scores["m8"], scores["m8_details"], solver=False)
                    parts += [f"  M8={_num('m8'):.2f}",
                              f"     ROLE: {_ROLE_P4['m8']}",
                              f"     WHAT HAPPENED: {(_fb or scores['m8_details'])}"]
                else:                                      # [wrapper-only] M7
                    parts += [f"  M7={_num('m7'):.2f}",
                              f"     ROLE: {_ROLE_P4['m7']}",
                              f"     WHAT HAPPENED (metric log): {scores.get('m7_details')}"]
        else:
            parts.append("  (nothing - Prompt 4 metrics are all clean.)")
        feedback = "\n".join(parts)                         # [wrapper-only]

    else:
        # Unknown predictor — fall back to full composite
        gepa_score = total                                 # [wrapper-only] INVARIANT: score == composite total
        feedback = f"Total: {total:.3f}"                   # [wrapper-only] minimal fallback feedback text

    return dspy.Prediction(score=gepa_score, feedback=feedback)  # [wrapper-only] GEPA expects Prediction(score, feedback); score==total invariant holds
