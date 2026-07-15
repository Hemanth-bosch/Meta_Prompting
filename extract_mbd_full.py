"""
MBD-UTA Full Pipeline DSPy Optimization
========================================

Full pipeline: wraps run.py's EXACT logic as a DSPy Module with 4 optimizable prompts.
  Prompt 1 (context_extractor):    context extraction from equation graph  — run.py _step_3a (438-458)
  Prompt 2 (testcase_generator):   JSON test case generation               — run.py _step_3b (465-501)
  Prompt 4 (equation_solver_dll):  DLL-verified equation solving (1st try)  — run.py _step_4a (561)
  Prompt 4 (equation_solver_retry):DLL-verified equation solving (retry)    — run.py _step_4a (614-620)

forward() reproduces run.py run_pipeline()'s per-weakspot body Phase 3a→3b→bridge→4a→post-process
(run.py:850-888 + 653-668); the dataset half (inputs + Phase 1/2) lives in utils/mbd_data.py, which
IMPORTS run.py's _locate_weakspot/_build_equation_context. All pipeline logic between prompts is the
SAME imported pipeline.* helpers run.py calls, in the same order. run.py bugs preserved intentionally
(retry `if retry_num < 1`, dedup heuristic, no final DLL check).

WRAPPER-ONLY ADDITIONS over run.py (run.py has NONE of these — it computes no score and ends by
writing an XLSM). Each is section-tagged [wrapper-only] at its site; consolidated here:
  (1) GEPA optimization of the 4 prompts               — optimization/opt.py; main() harness below.
  (2) M1..M8 metric suite (run.py computes no score)   — evaluation/mbd_metric.py; M7/M8 REUSE run.py's
      _step_4a DLL backbone (_attempt_solve/_build_substitution_dict/generate_bodmas_chain/
      resolve_equation_block). Inline M7 is computed in forward() (eq_m7_scores).
  (3) KB layering + OPT-IN Prompt-2 guideline injection — utils/kb_parser.py; forward()'s
      inject_kb_guidelines block (guidelines flow in as context DATA; the Prompt-2 seed is untouched).
  (4) train / val / test split                          — utils/mbd_data.stratified_split_mbd_dataset.
  (5) RawCompletionAdapter transport (run.py-faithful plain text so the ```json fence survives to
      extract_json_from_text_dll) — signatures/raw_completion_adapter.py; installed in main().
  (6) graded M7 + `--max-retries` knob (default 1 == run.py's single-retry bug EXACTLY) — forward().
  (7) solver-JSON-health counter (_solver_attempts / _solver_json_fails) — forward().
  (8) bare-array recovery for fence-less solver replies — _extract_dll_lenient() (below).
The pipeline LOGIC between the 4 prompts is otherwise run.py's own imported pipeline.* helpers, in
run.py's order, with run.py's bugs preserved intentionally.

END-TO-END ONE-TO-ONE FLOW MAP  (original MIDAS run.py  →  this DSPy/GEPA wrapper)
═════════════════════════════════════════════════════════════════════════════════════════
  run.py (original MIDAS)                          this wrapper (DSPy/GEPA)
─────────────────────────────────────────────────────────────────────────────────────────
  load_config() 200-258 / main() 934-944          build_parser() + main()         [CLI / budget]
  -- run_pipeline() inputs + graph prep -------    -- utils/mbd_data.load_mbd_dataset() -------
    load_equations/signal/interface  754-756         mbd_data: load_equations/... (raw text + df)
    extract_step_time(leaf_node)     784             mbd_data: extract_step_time -> raster_time field
    get_const_default_values + filt  795-796         mbd_data: const_values (Example.const_values)
    get_sequence_by_name/fix/global  829-831         mbd_data: same three graph helpers
  -- per-weakspot body --                          -- per dspy.Example --
    Phase 1 _locate_weakspot         843; 297-313    mbd_data._build_example_for_weakspot (IMPORTS it)
    Phase 2 _build_equation_context  848; 320-431    mbd_data._build_example_for_weakspot (IMPORTS it)
                                                       -> serialized to relations_json / equation_results_json /
                                                          op_map_json / weakspot_op_key  (deep-copy safe)
    Phase 3a _step_3a_get_context    851; 438-458    forward(): self.context_extractor      (Prompt 1)
    Phase 3b _step_3b_generate_tc    856; 465-501    forward(): self.testcase_generator     (Prompt 2)
      generate_and_parse_test_cases  486               forward(): bridge regex + newline-strip parse
      update_test_cases              489               forward(): update_test_cases()         [imported]
      extract_and_replace_equations  490               forward(): extract_and_replace_equations() [imported]
      filtered_const_values          493-499           forward(): filtered_const_values
    if len(equation_dict) != 0       876-888         forward(): same guard
    Phase 4a _step_4a_solve_dll      878; 508-669    forward(): per-equation solver loop
      merged_equation_map            521-537           forward(): merged_equation_map
      eq_to_solve / lhs_value        546-547           forward(): eq_to_solve / lhs_value
      generate_prompt + call_llm     550-564           forward(): self.equation_solver_dll    (Prompt 4)
      lock parameters first pass     567-579           forward(): locked_param_block
      _attempt_solve/_build_subst    581-582           forward(): same imported helpers
      _substitute_ops                584-589           forward(): _substitute_ops (inline copy)
      generate_bodmas_chain          591-594           forward(): generate_bodmas_chain       [imported]
      compute_unselected_mux  (#1)   596               forward(): compute_unselected_mux (2-arg)
      resolve_equation_block         597               forward(): resolve_equation_block      [imported]
      expected vs actual             599-600           forward(): expected / actual
      retry loop range(12)/if<1      608-637           forward(): self.equation_solver_retry  (Prompt 4 retry)
                                                         [bug preserved: only 1 real retry happens]
      post-process chain             653-668           forward(): update_equation_values -> map_equations_to_json
                                                         -> remove_variable_and_parameter_blocks -> flatten
                                                         -> deduplicate_actions -> compute_unselected_mux(#2, 3-arg)
                                                         -> clean_deduplicated_solution -> remove_unselected_mux_actions
    Phase 4b _step_4b (LLM-only)     676-720         NOT USED (wrapper is the use_dll=True path)
    final map (empty equation_dict)  888             forward(): map_equations_to_json
    WeakspotTestCaseOutput -> XLSM   890-927         NOT reproduced -> replaced by the M1..M8 metric
  (run.py computes no score)                        evaluation/mbd_metric.py (M1..M8; M7/M8 reuse run.py's DLL)
═════════════════════════════════════════════════════════════════════════════════════════
(Per-step inline `# run.py …` refs below repeat these anchors at each call site; this table is
the single end-to-end index. run.py line numbers are stable — it is the frozen original.)

FIDELITY — accepted/necessary deviations (data CONTENT is identical; see inline run.py refs):
  • DSPy renders the instruction as a system msg + each input field as a user-msg block, so the
    on-the-wire LAYOUT differs from run.py's single inline f-string. The instruction TEXT is the
    verbatim original prompt (signatures/mbd.py); data moves from f-string interpolation to fields.
  • equation_json / const_values are JSON (json.dumps) not Python-repr — REQUIRED because the metric
    parses them (json.loads); run.py rendered the dict repr. Same values.
  • equation_chain is "\n".join(eq_to_solve); run.py f-strings the list (repr). Same lines.
  • Prompt 3 (LLM-only Phase 4b) is NOT used — the wrapper is the use_dll=True path (DLL needed for
    M7/M8). MBDEquationSolvingLLM exists but is unreferenced.
  • The bridge runs ONE predictor call + graceful parse-failure handling instead of
    generate_and_parse_test_cases' 5× LLM retry (DSPy owns retries). Parse regex + newline-strip match.

run_pipeline() steps intentionally NOT reproduced (XLSM-OUTPUT / resource mgmt — NOT test-gen logic;
the wrapper's "output" is the metric, so these have no effect on the generated test cases):
  • extract_chartable_data → chartable_preconditions (run.py:832,895) — only feeds the XLSM.
  • WeakspotTestCaseOutput + get_weakspot_xlsm_dataframe + convertToTPTFormat_xlsm (run.py:890-927).
  • function_caller.cleanup() (run.py:907) — the wrapper CACHES FunctionCaller across eval/GEPA calls
    (_fc_cache) for speed and never tears it down; run.py cleans up at end of a one-shot run.
  • signal_sheet_data: run.py uses convert_signal_sheet_to_list(df) (run.py:817); the wrapper uses
    csv.DictReader on the raw signalsheet text — same list-of-dicts structure (see _get_function_caller).
  • leaf_node / weakspot_ids: both come from run_config.json fc_entries (run.py:240-241 == mbd_data).
The empty-equation_dict case now matches run.py:876-888 (post-process only when equations exist).

Compare with extract_mbd.py (2-prompt, no Phase 4, no post-processing).

Usage:
    py extract_mbd_full.py --eval-only
    py extract_mbd_full.py --optimizer gepa --gepa-auto light
    py extract_mbd_full.py --eval-test --load-extractor .cache/mbd_full_optimized.json

Environment Variables (CLI flags take priority):
    MODEL_NAME              Student LM (default: azure/gpt-4.1-mini)
    REFLECTION_LM_NAME      Reflection LM for GEPA (default: azure/gpt-5-mini)
"""

from __future__ import annotations

# ── stdlib (wrapper harness — run.py imports its own os/re/json/logging; the rest are wrapper-only) ──
import argparse          # [wrapper-only] build_parser() CLI (run.py used load_config()/argparse in main:934)
import csv               # [wrapper-only] csv.DictReader on raw signalsheet text (run.py:817 used convert_signal_sheet_to_list)
import importlib         # [wrapper-only] importlib.util to hot-swap MBD's config.py (see _mbd_config_spec below)
import io                # [wrapper-only] in-memory text buffers (log/history capture in the harness)
import json              # run.py:86 import json — JSON (de)serialize Example fields / solver replies
import logging           # run.py:87 import logging — step-by-step trace logger (see `logger` below)
import os                # run.py:88 import os — path joins for _MBD_ROOT / dll_path
import re                # run.py:89 import re — regex extraction (bridge JSON, _extract_dll_lenient)
import sys               # run.py:90 import sys — used here for sys.path.insert of the MBD pipeline root
import threading         # [wrapper-only] locks around cached FunctionCaller (run.py is single-shot, no cache)
import time              # [wrapper-only] wall-clock timing in the eval/GEPA harness (run.py has no timing)
from datetime import datetime  # run.py:85 from datetime import datetime — timestamped run artifacts here
from pathlib import Path       # [wrapper-only] path handling in main()/cache dirs

import dotenv             # [wrapper-only] load .env for MODEL_NAME/REFLECTION_LM_NAME (run.py:200-258 load_config read a JSON)
import dspy               # [wrapper-only] the DSPy framework this wrapper is built on (run.py has no DSPy)
# NOTE: dspy.utils.usage_tracker.track_usage is intentionally NOT used to wrap
# dspy.Evaluate — its ParallelExecutor deepcopies the UsageTracker (which holds a
# threading.Lock), failing every worker silently and zero-scoring the eval. Token
# usage is read from the LM history instead (see _token_usage_from_history below).

# Import project-level config BEFORE inserting MBD path into sys.path,
# so Python finds the project's config.py (not MBD's config.py).
from config import reconfigure_lm, reflectin_lm  # noqa: E402  # [wrapper-only] repo config.py — DSPy LM setup (run.py:97 `from config import cfg` loads MBD's own config)

# ── MBD pipeline path + config swap (same pattern as mbd_data.py) ───────────
_MBD_ROOT = os.path.normpath(                          # [wrapper-only] abs path to run.py's own root dir …
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "MBD", "weakspot_code_to_run_2")  # [wrapper-only] …so `from pipeline.* import` below resolves
)
if _MBD_ROOT not in sys.path:                          # [wrapper-only] add MBD root to sys.path once
    sys.path.insert(0, _MBD_ROOT)                      # [wrapper-only] prepend so pipeline.* wins over any same-named module

# [wrapper-only] temporarily install MBD's config.py under the name "config" so pipeline.* imports
# (which do `from config import cfg`, run.py:97) bind to MBD's config, not the repo's config.py.
_repo_config = sys.modules.get("config")               # [wrapper-only] stash the repo config module to restore after imports
_mbd_config_spec = importlib.util.spec_from_file_location(  # [wrapper-only] load MBD/config.py by explicit path
    "mbd_config", os.path.join(_MBD_ROOT, "config.py")
)
_mbd_config_mod = importlib.util.module_from_spec(_mbd_config_spec)  # [wrapper-only] instantiate the MBD config module
sys.modules["config"] = _mbd_config_mod                # [wrapper-only] shadow "config" with MBD's for the pipeline imports
_mbd_config_spec.loader.exec_module(_mbd_config_mod)   # [wrapper-only] execute MBD config so cfg is populated

# run.py pipeline imports — exact same as run.py lines 163-180
# ORIGINAL run.py:163-167 — bridge/test-case helpers (forward() calls both at run.py:489-490)
from pipeline.test_generator import update_test_cases, extract_and_replace_equations
# ORIGINAL run.py:168-174 — DLL-path solver helpers (subset; wrapper skips extract_json_from_text/clean_extracted_solution used only by LLM-only Phase 4b)
from pipeline.equation_solver import (
    extract_json_from_text_dll,                        # run.py:170 fence-only solver-reply extractor (wrapped by _extract_dll_lenient below)
    update_equation_values,                            # run.py:172 post-process: fold solved values back into test cases (run.py:653)
    map_equations_to_json,                             # run.py:173 post-process: serialize solved equations to JSON (run.py:659/888)
)
# ORIGINAL run.py:175-180 — the DLL-solving backbone forward()/M7/M8 all reuse verbatim
from pipeline.equation_solver_dll import (
    _attempt_solve,                                    # run.py:176 evaluate one substituted equation via the DLL
    _build_substitution_dict,                          # run.py:177 build var→value substitution map for a block
    resolve_equation_block,                            # run.py:178 resolve a full BODMAS block against the DLL
    generate_bodmas_chain,                             # run.py:179 emit the ordered BODMAS operation chain
)
# ORIGINAL run.py:142-150 — post-processing helpers (forward() runs them in run.py's 653-668 order)
from pipeline.postprocess import (
    remove_variable_and_parameter_blocks,              # run.py:149 strip Variable()/Parameter() scaffolding blocks
    flatten_precondition_and_actions,                  # run.py:147 flatten nested precondition/action structure
    deduplicate_actions,                               # run.py:145 dedup repeated actions (run.py dedup heuristic preserved)
    clean_deduplicated_solution,                       # run.py:143 tidy the deduplicated solution
    remove_unselected_mux_actions,                     # run.py:148 drop actions on unselected mux branches
    compute_unselected_mux_variables,                  # run.py:144 mark variables on unselected mux branches
)
from pipeline.types import MergedEquation             # run.py:138 MergedEquation dataclass (built in forward() merged_equation_map, run.py:531)
# NOTE: the original build_retry_instruction() wording now lives in the
# MBDEquationSolverRetry signature (GEPA-optimizable); its diagnostic data is passed
# as input fields, so build_retry_instruction is no longer imported here.
from pipeline.function_caller import FunctionCaller  # run.py:814 from pipeline.function_caller import FunctionCaller (run.py imports it lazily inside run_pipeline)

# [wrapper-only] undo the config shadow: put the repo's config.py back so the rest of THIS module
# (and later imports) sees the repo config, not MBD's.
if _repo_config is not None:                           # [wrapper-only] repo config existed before the swap → restore it
    sys.modules["config"] = _repo_config               # [wrapper-only] rebind "config" to the repo module
else:                                                  # [wrapper-only] nothing was there before the swap
    sys.modules.pop("config", None)                    # [wrapper-only] remove the MBD shadow entirely

# [wrapper-only] DSPy Signatures whose instruction TEXT is run.py's verbatim prompt wording
# (the 4 optimizable prompts): run.py rendered these as f-strings via prompts/*.py; DSPy makes them
# GEPA-optimizable Signatures. Prompt↔Signature map: Prompt1=context_prompt, Prompt2=testcase_prompt,
# Prompt4=equation_solution_prompt_dll, Prompt4-retry=build_retry_instruction wording.
from signatures.mbd import (
    MBDContextExtraction,                              # Prompt 1 seed == prompts/context_prompt.py wording (run.py:445)
    MBDTestCaseGeneration,                             # Prompt 2 seed == prompts/testcase_prompt.py wording (run.py:475)
    MBDEquationSolvingDLL,                             # Prompt 4 seed == prompts/equation_solution_prompt_dll.py wording (run.py:550)
    MBDEquationSolverRetry,                            # Prompt 4 retry seed == build_retry_instruction() wording (run.py:614-620)
)
# [wrapper-only] dataset layer — IMPORTS run.py's _locate_weakspot/_build_equation_context and
# serializes eq_ctx into dspy.Example fields (Phase 1/2 origin; see utils/mbd_data.py)
from utils.mbd_data import load_mbd_dataset, stratified_split_mbd_dataset, all_fc_entries
from utils.kb_parser import format_kb_guidelines_block   # [wrapper-only] KB-guideline injection into Prompt 2 (opt-in; NOT in run.py)
try:                                                     # [wrapper-only] expert Required-I/O KB — Prompt-2 output override (opt-in; NOT in run.py)
    from utils.required_io import resolve_required_outputs_str as _resolve_required_outputs_str
except ImportError:                                      # [wrapper-only] KB module missing -> override is a no-op (keeps instance_names)
    def _resolve_required_outputs_str(fc_name, ws_id): return None  # type: ignore[misc]
# [wrapper-only] the M1..M8 metric suite (run.py computes NO score; this is the GEPA objective).
# M7/M8 REUSE the run.py DLL helpers imported above.
from evaluation.mbd_metric import (
    mbd_testcase_metric,                              # [wrapper-only] plain composite metric (eval path)
    mbd_metric_with_feedback,                         # [wrapper-only] GEPA feedback metric (per-predictor text)
    _compute_all_components,                          # [wrapper-only] compute M1..M8 components dict
    _strip_json_comments,                             # [wrapper-only] tolerate // /* */ comments in Prompt-2 JSON (string-aware)
    solve_with_dll_timeout,                           # [wrapper-only] watchdog around a DLL solve (kills a hung 32-bit helper; self-restarts)
    _DLLSolveTimeout,                                 # [wrapper-only] raised when a DLL solve exceeds MBD_DLL_SOLVE_TIMEOUT_S
)
from optimization import OptimizationRunner, create_gepa_strategy  # [wrapper-only] GEPA harness (optimization/opt.py; run.py has no optimizer)

dotenv.load_dotenv(override=True)                     # [wrapper-only] load .env so MODEL_NAME/REFLECTION_LM_NAME are visible (run.py:200-258 read a JSON config)

# ── Debug logger for step-by-step pipeline execution ─────────────────────────
# Silent by default. A caller (e.g. an end-to-end runner script) configures
# logging (basicConfig / file handler) and sets this logger's level to DEBUG
# to get a full step-by-step trace of forward().
logger = logging.getLogger("extract_mbd_full")        # [wrapper-only] named trace logger (run.py used the root logger via basicConfig, run.py:188)


# ═══════════════════════════════════════════════════════════════════
# DATASET PREPARATION
# ═══════════════════════════════════════════════════════════════════

def prepare_full_examples(dataset):
    """Re-wrap examples from load_mbd_dataset() with full with_inputs() for MBDFullPipelineModule.

    mbd_data.py's with_inputs() stays at 5 fields to not break MBDTestGenerator.
    Here we extend it to include all 12 fields forward() needs.
    """
    # [wrapper-only] re-mark ALL 14 forward() fields as DSPy inputs (mbd_data's with_inputs stays at 5).
    # Field → run.py origin (values built in utils/mbd_data.py, serialized from eq_ctx / filtered_dict):
    return [                                            # [wrapper-only] rebuild the Example list with the wider input set
        ex.with_inputs(                                # [wrapper-only] DSPy input marking (run.py has no Example/with_inputs)
            "equation_json", "const_values", "weakspot_context",           # run.py filtered_dict["equation"] (355/377) / input_json["const_values"] (353) / get_weakspot_context render (445)
            "instance_names", "raster_time",                               # run.py:425 eq_ctx.instance_names / run.py:784 step_time (×1000)
            "weakspot_operation", "example_explanation_output",            # run.py:354 weakspot_op_key / run.py:472-474 get_weakspot_example_output
            "dll_path", "signalsheet_data", "model_interface_data",        # [wrapper-only] metric/DLL metadata (M7/M8 + M1/M2/M3/M6)
            "relations_json", "equation_results_json",                     # run.py:429 eq_ctx.relations / run.py:430 eq_ctx.equation_results (JSON-serialized)
            "op_map_json", "weakspot_op_key",                              # run.py:662 build_op_mapping(dependency_mapping) / run.py:419 eq_ctx.weakspot_op_key
            "fc_name", "weakspot_id",                                      # [wrapper-only] weakspot identity for the Required-I/O KB lookup in forward(); absorbed by forward(**kwargs) and passed to NO signature -> NEVER rendered into a prompt
        )
        for ex in dataset                              # [wrapper-only] one re-wrapped Example per input example
    ]


def _extract_dll_lenient(text: str):
    """run.py's `extract_json_from_text_dll` with BARE-ARRAY pre-normalization.

    run.py's extractor is FENCE-ONLY: it needs the assignment array inside a ```json ... ``` block.
    gpt-4.1-mini sometimes prints a valid array WITHOUT the fence, which the strict regex drops ->
    inflates the JSON-failure rate. We recover it by wrapping the bare array in a fence *BEFORE* calling
    the unchanged extractor — so the extractor finds it AND does NOT log a spurious "No valid JSON list
    found" (the warning then fires only for genuine failures, matching the health counter).

    Only fires when there is NO fence at all and the bare `[...]` is a non-empty list of STRINGS (the
    assignment shape), so prose like "[1, 2, 3]" is ignored and the fenced happy-path is byte-identical
    to run.py.
    """
    # [wrapper-only] bare-array recovery placed BEFORE run.py's FENCE-ONLY extractor
    # extract_json_from_text_dll (pipeline/equation_solver.py:65-92, imported above at run.py:170).
    if text and "```" not in text:                               # no fence -> maybe a fence-less array
        for _m in re.finditer(r"\[[^\[\]]*\]", text, re.DOTALL):  # flat [...] spans (no nested brackets)
            _span = _m.group(0)                                  # [wrapper-only] the candidate [...] span
            # [wrapper-only] try the span AS-IS, then with single->double quotes (some models emit
            # ['Variable(X) = 5'] instead of ["Variable(X) = 5"] — extract_json_from_text_dll fixes quotes
            # only INSIDE a fence, so a fence-less single-quoted array would otherwise be lost -> a retry).
            for _cand in (_span, _span.replace("'", '"')):       # [wrapper-only] as-is, then quote-normalised
                try:                                             # [wrapper-only] try to parse this candidate as JSON
                    _val = json.loads(_cand)                     # [wrapper-only] parse the [...] span
                except Exception:                                # [wrapper-only] not valid JSON -> try next form
                    continue                                     # [wrapper-only]
                if isinstance(_val, list) and _val and all(isinstance(_x, str) for _x in _val):  # [wrapper-only] assignment shape: non-empty list[str] (prose "[1,2,3]" is ignored)
                    text = f"```json\n{_cand}\n```"              # wrap the (normalised) assignment array in a fence
                    break                                        # [wrapper-only] recovered this span
            else:                                                # [wrapper-only] neither form parsed -> next [...] match
                continue                                         # [wrapper-only]
            break                                                # [wrapper-only] first assignment array wins -> stop scanning
    return extract_json_from_text_dll(text)                      # run.py:170/pipeline/equation_solver.py:65 unchanged fence-only extractor


# ═══════════════════════════════════════════════════════════════════
# DSPy MODULE
# ═══════════════════════════════════════════════════════════════════

class MBDFullPipelineModule(dspy.Module):
    """Full MBD-UTA pipeline: Prompt 1 + Prompt 2 + bridge + Prompt 4 + post-processing.

    Exact reproduction of run.py's pipeline with 3 call_llm() calls replaced
    by DSPy predictors so GEPA can optimize them. All other logic verbatim.
    """

    _fc_cache: dict = {}                                        # [wrapper-only] process-wide FunctionCaller cache (run.py rebuilds per run at 821-826; cached here to reuse across forward() calls)
    _fc_cache_lock = threading.Lock()                          # [wrapper-only] guards _fc_cache under GEPA's parallel eval threads (run.py is single-threaded async)

    def __init__(self, use_cot: bool = True):
        super().__init__()                                     # [wrapper-only] dspy.Module init (registers predictors as named parameters)
        Predictor = dspy.ChainOfThought if use_cot else dspy.Predict  # [wrapper-only] CoT vs plain Predict (both REPLACE run.py's raw call_llm; run.py has no CoT scaffold)
        self.context_extractor    = Predictor(MBDContextExtraction)    # run.py:454 `context = await call_llm(prompt)` (_step_3a Prompt 1) → DSPy predictor
        self.testcase_generator   = Predictor(MBDTestCaseGeneration)   # run.py:486 `await generate_and_parse_test_cases(prompt)` (_step_3b Prompt 2) → DSPy predictor
        self.equation_solver_dll  = Predictor(MBDEquationSolvingDLL)   # run.py:561 `solve_response = await call_llm(prompt)` (_step_4a Prompt 4, first call) → DSPy predictor
        self.equation_solver_retry = Predictor(MBDEquationSolverRetry)  # run.py:624 `solve_response = await call_llm(prompt)` (_step_4a retry loop) → DSPy predictor
        # OPT-IN: append resolved KB guidelines (new->old, via resolve_guidelines) into the Prompt-2
        # context so the LLM GENERATES coverage-aware test cases. Default from env so the GEPA optimizer
        # SUBPROCESS inherits it (compile runs out-of-process); callers may also set the attr directly.
        self.inject_kb_guidelines = os.getenv("MBD_INJECT_KB_GUIDELINES", "") not in ("", "0", "false", "False")  # [wrapper-only] opt-in Prompt-2 KB steering (no run.py analog); default from env so the compile subprocess inherits it
        # OPT-IN: override Prompt 2's `instance_names` ("use this Expected Results") with the expert
        # Required-I/O KB's required_outputs (utils/required_io.json, path via MBD_REQUIRED_IO_PATH). Lets
        # experts steer generation to a chosen output (e.g. a weakspot INTERMEDIATE output) — no code edit
        # per run. Missing KB entry / no outputs -> keep the dataset instance_names (byte-identical). Env
        # default so the GEPA compile SUBPROCESS inherits it; callers may also set the attr directly.
        self.required_io_in_prompt = os.getenv("MBD_REQUIRED_IO_IN_PROMPT", "") not in ("", "0", "false", "False")  # [wrapper-only] opt-in Prompt-2 output override (no run.py analog); default OFF -> instance_names unchanged
        # OPT-IN: DLL Prompt-4 retry budget. run.py hardcodes 1 real retry (the range(12)/`retry_num<1`
        # bug); default 1 preserves run.py's SOLVE behaviour EXACTLY. Raise it (e.g. 4/10) to give the
        # solver more backtrace attempts to reach LHS==RHS. Env default so the (in-process) GEPA generator
        # inherits it; callers may also set the attr directly. (M7's grading scale is independent of this
        # budget — see m7_reference_retries below: 0.2 per retry up to budget 4, then 1/(budget+1).)
        self.max_retries = int(os.getenv("MBD_MAX_RETRIES", "1") or "1")  # [wrapper-only] parameterizes run.py:612 hardcoded `retry_num < 1`; default 1 == run.py EXACTLY (env-sourced so compile subprocess inherits)
        # M7 grading reference: the graded score denominator is (m7_reference_retries + 1). None -> ref =
        # max(4, max_retries): for budgets <= 4 a FIXED 0.2 penalty PER RETRY (1.0/0.8/0.6/0.4/0.2/0.0), and
        # for budgets > 4 a 1/(budget+1) penalty per retry (e.g. budget 9 -> 1.0/0.9/.../0.1/0.0). So a
        # 1-retry solve = 0.8 for any budget <= 4. Override the attr only to pin a fixed penalty scale.
        self.m7_reference_retries = None                       # [wrapper-only] M7 grading scale (no run.py analog); None -> ref=max(4,max_retries) at forward()
        # Solver JSON health counters: total DLL Prompt-4 solve attempts and how many returned NO
        # parseable JSON list (extract_json_from_text_dll -> None == the "No valid JSON list found"
        # warning). Read after a run to see the solver's JSON-failure rate. Instance-level so one
        # generator accumulates across all its forward() calls (matrix / --eval-only baseline).
        self._solver_attempts = 0                              # [wrapper-only] solver JSON health counter (no run.py analog); incremented per DLL solve at forward():541/621
        self._solver_json_fails = 0                            # [wrapper-only] solver JSON health counter (no run.py analog); incremented when extract_json_from_text_dll -> None

    def forward(
        self,
        equation_json,
        const_values,
        weakspot_context,
        instance_names,
        raster_time,
        weakspot_operation="",
        example_explanation_output="",
        dll_path="",
        signalsheet_data="",
        model_interface_data="",
        relations_json="{}",
        equation_results_json="[]",
        op_map_json="{}",
        weakspot_op_key="",
        **kwargs,
    ):
        """forward() == run.py run_pipeline() per-weakspot body (run.py:850-888 + 653-668):
        Phase 3a → 3b → bridge → 4a (DLL solve + retry) → post-process chain. The ONLY structural
        change is that run.py's 3 `prompt = generate_*(); resp = await call_llm(prompt)` pairs are
        replaced by the 4 DSPy predictors (context / testcase / equation_solver_dll / _retry).

        VARIABLE-NAME MAP (wrapper ↔ run.py) — names match run.py, with these correspondences:
          function_caller  ≙ run.py function_caller     (FunctionCaller; run.py:821)
          test_cases       ≙ run.py test_cases          (parsed Prompt-2 list; _step_3b:486/490)
          test_cases_raw   ≙ run.py raw_response        (raw Prompt-2 text;  _step_3b:486)
          context_analysis ≙ run.py context             (Prompt-1 output;    run_pipeline:851)
          filtered_const_values ≙ run.py eq_ctx.filtered_const_values  (_step_3b:496)
          ctx / tc / solver_pred / retry_pred = DSPy predictor results that REPLACE run.py's
                             prompt+call_llm pairs (Prompt 1/2/4-first/4-retry).
        All other locals (eq_to_solve, lhs_value, merged_equation_map, original_eq_map, attempt,
        substitution_dict, eq_list, final_check, try_fc, expected, actual, is_first_pass,
        locked_param_block, all_unselected_vars, updated_dict, pre_final, cleaned, flattened,
        deduped, final_json, …) use run.py's EXACT names.
        """
        logger.info("=" * 70)                                                          # [wrapper-only] trace banner
        logger.info("forward(): START — weakspot_op_key=%s raster_time=%s dll_path=%s",  # [wrapper-only] trace (run.py logs per-ws at 838-840)
                     weakspot_op_key, raster_time, dll_path)                            # [wrapper-only] trace args
        logger.debug("forward(): equation_json (len=%d) = %s", len(equation_json or ""), equation_json)             # [wrapper-only] trace
        logger.debug("forward(): const_values = %s", const_values)                     # [wrapper-only] trace
        logger.debug("forward(): weakspot_context (len=%d) = %s", len(weakspot_context or ""), weakspot_context)    # [wrapper-only] trace
        logger.debug("forward(): instance_names = %s", instance_names)                 # [wrapper-only] trace

        # ── OPT-IN KB-guideline injection via a dedicated INPUT FIELD (wrapper-only; NOT in run.py) ──
        # Resolve the weakspot's KB guidelines with format_kb_guidelines_block(): NEW KB -> OLD KB
        # fallback -> "" (empty) if NEITHER KB has the block (the SAME resolver M4 uses). The resolved
        # text is passed via the dedicated `kb_guidelines` InputField on BOTH Prompt 1 and Prompt 2
        # (signatures/mbd.py); the field description frames it as MANDATORY BU requirements. When off
        # (or no KB match) -> _kb_guidelines == "" -> the field renders EMPTY on both prompts and is
        # ignored (empty field marker still appears, so inject-off is functionally — not byte —
        # identical to run.py). The instruction seeds (__doc__) are never touched.
        _kb_guidelines = ""                                    # [wrapper-only] NEW->OLD->"" ; empty field on both prompts when off / no match
        if self.inject_kb_guidelines:                          # [wrapper-only] steering, gated by flag/env
            _op = json.loads(equation_json).get(weakspot_op_key, {}) if equation_json else {}  # weakspot op entry (as M4:810)
            _fn = _op.get("function_name") or _op.get("class_method_name") or _op.get("method_name") or ""  # [wrapper-only] KB lookup key
            _kb_guidelines = format_kb_guidelines_block(_fn)   # [wrapper-only] NEW KB -> OLD KB fallback -> "" if neither has the block
            if _kb_guidelines:                                 # [wrapper-only] trace only when guidelines actually resolved
                logger.info("forward(): KB guidelines resolved (fn=%s, +%d chars) -> kb_guidelines FIELD on Prompt 1 AND Prompt 2", _fn, len(_kb_guidelines))  # [wrapper-only] trace

        # ── Prompt 1: Context Extraction == run.py _step_3a_get_context (438-458) ──
        logger.info("forward(): ----- Prompt 1 (context_extractor) — calling LM -----")  # [wrapper-only] trace
        ctx = self.context_extractor(                          # run.py:454 `context = await call_llm(prompt)` → DSPy predictor
            equation_json=equation_json,                       # run.py: filtered_dict["equation"]  (Prompt-1 input field)
            const_values=const_values,                         # run.py: filtered_dict["const_values"]
            weakspot_operation=weakspot_operation,             # run.py: filtered_dict["weakspot_operation"] ($opN)
            weakspot_context=weakspot_context,                 # run.py:445 Prompt-1 weakspot context (rendered via context_prompt.py:55/65 get_weakspot_context)
            kb_guidelines=_kb_guidelines,                      # [Q3] dedicated MANDATORY-BU-guidelines input field ("" when off/no match -> empty field, ignored)
        )                                                      # run.py:445 generate_context_prompt(eq_ctx.filtered_dict)
        context_analysis = ctx.context_analysis                # run.py:454 `context` (Prompt-1 free-text output)
        # FIDELITY FIX: run.py threads the ENTIRE Prompt-1 output into Prompt 2 (run_pipeline:851
        # context → 856 _step_3b). Under ChainOfThought the DETAILED analysis lands in ctx.reasoning
        # while context_analysis holds only a short summary — so without this, Prompt 2 would receive
        # a TRUNCATED context vs run.py. Prepend reasoning so Prompt 2 gets the SAME full context
        # run.py gives it. (dspy.Predict/--no-cot has no reasoning field → this is a no-op there.)
        _reasoning = getattr(ctx, "reasoning", None)           # [wrapper-only] CoT scratchpad = the detailed Prompt-1 analysis
        if _reasoning:                                         # [wrapper-only] only when CoT produced a reasoning field
            context_analysis = f"{_reasoning}\n\n{context_analysis}"  # restore run.py's full-context threading into Prompt 2
        logger.info("forward(): Prompt 1 DONE — context_analysis (len=%d)", len(context_analysis or ""))  # [wrapper-only] trace
        logger.debug("forward(): Prompt 1 context_analysis = %s", context_analysis)    # [wrapper-only] trace

        # ── OPT-IN: Required-I/O KB output override (wrapper-only; NOT in run.py) ──
        # When enabled, replace the `instance_names` fed to Prompt 2 ("use this Expected Results") with the
        # expert KB's required_outputs for THIS weakspot (fc_name#weakspot_id, arriving via **kwargs from
        # prepare_full_examples — never rendered, only used here). No KB entry / no outputs -> keep
        # instance_names (byte-identical to run.py). The METRIC resolves the SAME KB independently
        # (mbd_metric._compute_all_components), so prompt + M1 scoring stay aligned with no double-apply.
        if self.required_io_in_prompt:                         # [wrapper-only] opt-in flag (env/attr); default OFF
            _fc = kwargs.get("fc_name", "")                    # [wrapper-only] weakspot identity (kwarg; not a signature field)
            _ws = kwargs.get("weakspot_id", "")                # [wrapper-only] weakspot identity (kwarg; not a signature field)
            _ov = _resolve_required_outputs_str(_fc, _ws, weakspot_op_key)  # [wrapper-only] KB required_outputs (comma-joined) or None; op_key disambiguates duplicate fc#ws
            if _ov:                                            # [wrapper-only] only override when the KB lists outputs for this weakspot
                logger.info("forward(): Required-I/O KB PROMPT override (fc=%s ws=%s) — Expected Results instance_names %r -> %r", _fc, _ws, instance_names, _ov)  # [wrapper-only] trace
                instance_names = _ov                           # [wrapper-only] steer Prompt 2 to the expert-chosen output (e.g. intermediate)

        # ── Prompt 2: Test Case Generation == run.py _step_3b_generate_testcases (465-501) ─
        logger.info("forward(): ----- Prompt 2 (testcase_generator) — calling LM -----")  # [wrapper-only] trace
        # FIDELITY: run.py's Prompt 2 (generate_testcase_prompt) takes context, filtered_dict,
        # example_explanation_output, instance_names — NO raster_time. dT is carried in const_values.
        # So raster_time is NOT passed to the predictor (it stays Example metadata for the metric).
        tc = self.testcase_generator(                          # run.py:486 `await generate_and_parse_test_cases(prompt)` → DSPy predictor
            context_analysis=context_analysis,                 # run.py:476 `context` arg of generate_testcase_prompt
            equation_json=equation_json,                       # run.py:476 filtered_dict["equation"]
            const_values=const_values,                         # run.py: filtered_dict["const_values"]
            weakspot_operation=weakspot_operation,             # run.py: filtered_dict["weakspot_operation"]
            instance_names=instance_names,                     # run.py:476 eq_ctx.instance_names ("use this Expected Results")
            example_explanation_output=example_explanation_output,  # run.py:472-474 get_weakspot_example_output(...)
            kb_guidelines=_kb_guidelines,                      # [Q2] dedicated MANDATORY-BU-guidelines input field ("" when off/no match -> empty field, ignored)
        )                                                      # run.py:475 generate_testcase_prompt(context, filtered_dict, example, instance_names)
        test_cases_raw = tc.test_cases  # CAPTURE Prompt 2 raw output for staged metric  # run.py:486 `raw_response`
        logger.info("forward(): Prompt 2 DONE — test_cases_raw (len=%d)", len(test_cases_raw or ""))  # [wrapper-only] trace
        logger.debug("forward(): Prompt 2 test_cases_raw = %s", test_cases_raw)        # [wrapper-only] trace

        # ── Bridge: run.py lines 486-499 ════════════════════════════════════════
        # Parse Prompt 2 output (mirrors generate_and_parse_test_cases regex extraction)
        logger.info("forward(): ----- Bridge: parsing Prompt 2 JSON output -----")     # [wrapper-only] trace
        try:                                                   # run.py: generate_and_parse_test_cases wraps this in 5× retry; DSPy owns retries
            raw_str = _strip_json_comments(test_cases_raw.strip() if test_cases_raw else "")  # test_generator.py:60 raw text; [wrapper-only] strip // /* */ comments some LLMs add (string-aware)
            json_match = re.search(r"\[\s*{.*}\s*\]", raw_str, re.DOTALL)              # test_generator.py:64 regex extract JSON array
            if json_match:                                     # test_generator.py:65 if a JSON array was found
                # FIDELITY (test_generator.py:64-66): generate_and_parse_test_cases does
                # re.search(...).group().replace("\n", "") before json.loads — the newline strip
                # is functionally required so JSON with literal newlines inside string values
                # (e.g. "Detailed Explanation") parses (json.loads rejects raw control chars).
                test_cases = json.loads(json_match.group().replace("\n", ""))         # test_generator.py:66 json.loads(group().replace("\n",""))
                logger.debug("forward(): bridge — extracted JSON via regex match")     # [wrapper-only] trace
            else:                                              # no regex match → try raw (wrapper-added graceful path)
                test_cases = json.loads(raw_str)                                       # [wrapper-only] fallback parse (run.py would 5×-retry)
                logger.debug("forward(): bridge — parsed raw_str directly as JSON (no regex match)")  # [wrapper-only] trace
            if not isinstance(test_cases, list):               # run.py expects a JSON list of test cases
                raise ValueError("Not a list")                 # [wrapper-only] guard → graceful return below
            logger.info("forward(): bridge — parsed %d test case(s) from Prompt 2", len(test_cases))  # [wrapper-only] trace
        except (json.JSONDecodeError, ValueError, AttributeError) as _bridge_err:      # [wrapper-only] graceful parse-failure (run.py 5×-retries instead)
            logger.error("forward(): bridge — FAILED to parse Prompt 2 output: %s", _bridge_err)  # [wrapper-only] trace
            return dspy.Prediction(                            # [wrapper-only] early return so the metric can score a parse failure
                context_analysis=context_analysis,             # [wrapper-only] pass-through
                test_cases_pre_postprocess=test_cases_raw,     # [wrapper-only] Prompt-2 raw (metric M1/M3/M4/M6 input)
                test_cases=test_cases_raw,                     # [wrapper-only] final = raw (no pipeline ran)
                m7_score=None,                                 # [wrapper-only] no DLL solving happened
                m7_details="Prompt 2 parse failed — no equations processed",  # [wrapper-only] detail
            )

        # Deserialize precomputed Phase 2 data  (wrapper: run.py holds these live in eq_ctx; here JSON-serialized in the Example)
        relations = json.loads(relations_json) if relations_json else {}              # run.py:429 eq_ctx.relations  (var → solvable equation)
        equation_results_raw = json.loads(equation_results_json) if equation_results_json else []  # run.py:430 eq_ctx.equation_results (list[EquationResult])
        logger.debug("forward(): relations (count=%d) = %s", len(relations), relations)            # [wrapper-only] trace
        logger.debug("forward(): equation_results_raw — %d entr(ies)", len(equation_results_raw))  # [wrapper-only] trace

        # run.py line 489: update_test_cases(test_cases, eq_ctx.relations)
        updated_test_cases = update_test_cases(test_cases, relations)                 # run.py:489 update_test_cases(test_cases, eq_ctx.relations)  [imported helper]
        logger.debug("forward(): update_test_cases() -> %d test case(s)", len(updated_test_cases))  # [wrapper-only] trace

        # run.py line 490: updated_json, equation_dict = extract_and_replace_equations(updated_test_cases)
        # Signature: (list) -> (list, dict)  — 1 arg, returns (updated_json, equation_dict)
        updated_json, equation_dict = extract_and_replace_equations(updated_test_cases)  # run.py:490 extract_and_replace_equations(updated_test_cases)  [imported helper]
        logger.info("forward(): extract_and_replace_equations() -> %d equation(s) extracted", len(equation_dict))  # [wrapper-only] trace
        for _uid, _eq in equation_dict.items():                                       # [wrapper-only] trace loop
            logger.debug("forward(): equation_dict[%s] = %s", _uid, _eq)              # [wrapper-only] trace

        # run.py lines 493-499: compute filtered_const_values
        const_dict = json.loads(const_values) if const_values else {}                 # run.py:497 eq_ctx.filtered_dict["const_values"] (== full const_values; filter_equation passes it through)
        all_equations = " ".join(equation_dict.values())                             # run.py:493 all_equations = " ".join(equation_dict.values())
        matches = re.findall(r"SystemConstant\((.*?)\)", all_equations)              # run.py:494 re.findall(r"SystemConstant\((.*?)\)", all_equations)
        unique_constants = list(set(matches))                                         # run.py:495 unique_constants = list(set(matches))
        filtered_const_values = {k: v for k, v in const_dict.items() if k in unique_constants}  # run.py:496-499 eq_ctx.filtered_const_values (subset used in equations)
        logger.debug("forward(): unique_constants referenced = %s", unique_constants)  # [wrapper-only] trace
        logger.debug("forward(): filtered_const_values = %s", filtered_const_values)  # [wrapper-only] trace

        # ── Prompt 4: DLL Equation Solving == run.py _step_4a_solve_equations_dll (508-669) ─
        # run.py:821-826 builds FunctionCaller; here it is cached in _get_function_caller().
        function_caller = self._get_function_caller(dll_path, signalsheet_data, model_interface_data)  # run.py:821-826 FunctionCaller(FCname, work_dir, signal_sheet, model_interface)
        logger.info("forward(): function_caller=%s (None means DLL verification disabled)",  # [wrapper-only] trace
                     "available" if function_caller is not None else "None")          # [wrapper-only] trace arg

        # run.py lines 521-537: build merged_equation_map
        original_eq_map = {                                    # run.py:521-525 original_eq_map = {eq.original_equation: eq for eq in eq_ctx.equation_results if ...}
            r["original_equation"]: r                          # run.py:522 key = original_equation
            for r in equation_results_raw                      # run.py:523 iterate eq_ctx.equation_results (here: deserialized dicts)
            if r.get("original_equation", "").strip()          # run.py:524 if eq.original_equation.strip()
        }
        merged_equation_map = {}                               # run.py:526 merged_equation_map = {}
        for uid, full_eq in equation_dict.items():             # run.py:527 for uid, full_eq in equation_dict.items()
            lhs = full_eq.split("=")[0].strip()                # run.py:528 lhs = full_eq.split("=")[0].strip()
            if lhs in original_eq_map:                         # run.py:529 if lhs in original_eq_map
                orig = original_eq_map[lhs]                    # run.py:530 orig = original_eq_map[lhs]
                merged_equation_map[uid] = MergedEquation(     # run.py:531-537 merged_equation_map[uid] = MergedEquation(...)
                    uid=uid,                                   # run.py:532 uid=uid
                    equation_to_solve=full_eq,                 # run.py:533 equation_to_solve=full_eq
                    original_equation=lhs,                     # run.py:534 original_equation=lhs
                    assignment_chain=orig["assignment_chain"],                  # run.py:535 assignment_chain=orig.assignment_chain
                    final_substituted_equation=orig["final_substituted_equation"],  # run.py:536 final_substituted_equation=orig.final_substituted_equation
                )
            else:                                              # LHS not a known relation → skip (run.py implicitly skips; only matched UIDs enter the map)
                logger.debug("forward(): equation_dict[%s] LHS %r NOT found in original_eq_map (dropped from merged_equation_map)", uid, lhs)  # [wrapper-only] trace
        logger.info("forward(): merged_equation_map has %d entr(ies) (out of %d equation_dict entries)",  # [wrapper-only] trace
                     len(merged_equation_map), len(equation_dict))               # [wrapper-only] trace args

        # run.py lines 539-543
        updated_dict = {}                                      # run.py:539 updated_dict = {}
        is_first_pass = True                                   # run.py:540 is_first_pass = True
        locked_param_block = ""                                # run.py:541 locked_param_block = ""
        all_unselected_vars = set()                            # run.py:542 all_unselected_vars = set()
        substitution_dict = {}                                 # run.py:543 substitution_dict = {}

        # M7 inline tracking: record 1/N score per equation (N = attempts in retry loop)
        eq_m7_scores: list = []                                # [wrapper-only] M7 metric signal (run.py computes no score)

        # run.py lines 584-588: _substitute_ops defined INLINE (not importable from pipeline)
        def _substitute_ops(equation, subs):                  # run.py:584 def _substitute_ops(equation, subs)
            return re.sub(                                     # run.py:585 return re.sub(
                r"\$op\d+",                                    # run.py:586 r"\$op\d+"
                lambda m: subs.get(m.group(0), m.group(0)),    # run.py:587 lambda m: subs.get(m.group(0), m.group(0))
                equation,                                      # run.py:588 equation
            )

        for _eq_idx, (eq_id, merged_eq) in enumerate(merged_equation_map.items()):    # run.py:545 for eq_id, merged_eq in tqdm(merged_equation_map.items(), ...)
            logger.info("forward(): ===== Equation %d/%d (uid=%s) =====", _eq_idx + 1, len(merged_equation_map), eq_id)  # [wrapper-only] trace
            logger.debug("forward(): eq[%s] equation_to_solve = %s", eq_id, merged_eq.equation_to_solve)             # [wrapper-only] trace
            logger.debug("forward(): eq[%s] original_equation = %s", eq_id, merged_eq.original_equation)             # [wrapper-only] trace
            logger.debug("forward(): eq[%s] assignment_chain = %s", eq_id, merged_eq.assignment_chain)               # [wrapper-only] trace
            logger.debug("forward(): eq[%s] final_substituted_equation = %s", eq_id, merged_eq.final_substituted_equation)  # [wrapper-only] trace

            # run.py lines 546-547
            eq_to_solve = merged_eq.assignment_chain.split("\n")[::-1]                # run.py:546 eq_to_solve = merged_eq.assignment_chain.split("\n")[::-1]
            lhs_value = merged_eq.equation_to_solve.split("=")[-1].strip()            # run.py:547 lhs_value = merged_eq.equation_to_solve.split("=")[-1].strip()
            logger.debug("forward(): eq[%s] eq_to_solve (reversed assignment_chain, %d line(s)) = %s",  # [wrapper-only] trace
                          eq_id, len(eq_to_solve), eq_to_solve)                        # [wrapper-only] trace args
            logger.debug("forward(): eq[%s] lhs_value (target) = %s", eq_id, lhs_value)  # [wrapper-only] trace

            # run.py line 561 (call_llm) → DSPy predictor (Prompt 4, first call)
            logger.info("forward(): eq[%s] ----- Prompt 4 (equation_solver_dll) attempt #1 — calling LM -----", eq_id)  # [wrapper-only] trace
            solver_pred = self.equation_solver_dll(                # run.py:561 `solve_response = await call_llm(prompt)` → DSPy predictor
                equation_chain="\n".join(eq_to_solve),             # run.py:550-551 eq_to_solve arg ("\n".join vs run.py's list-repr; same lines)
                target_value=lhs_value,                            # run.py:551 lhs_value arg
                const_values=json.dumps(filtered_const_values),    # run.py:551 eq_ctx.filtered_const_values arg
                locked_param_block=locked_param_block,             # run.py:551 locked_param_block arg
            )                                                      # run.py:550 generate_equation_solution_prompt_dll(eq_to_solve, lhs_value, filtered_const_values, locked_param_block)
            logger.debug("forward(): eq[%s] Prompt 4 attempt#1 raw variable_assignments = %s", eq_id, solver_pred.variable_assignments)  # [wrapper-only] trace
            # _extract_dll_lenient = run.py's fence-only extractor + bare-array recovery (rescues valid
            # arrays the model printed WITHOUT a ```json fence). None when truly no array -> [] so
            # _attempt_solve doesn't crash (same as run.py intent).
            _raw_json = _extract_dll_lenient(solver_pred.variable_assignments)  # run.py:564 simulated_response = extract_json_from_text_dll(solve_response)
            self._solver_attempts += 1                             # [wrapper-only] solver JSON health: one solve attempt
            if _raw_json is None:                                  # [wrapper-only] None == the "No valid JSON list found" warning
                self._solver_json_fails += 1                       # [wrapper-only] count the JSON parse failure
            simulated_response = _raw_json or []                   # [wrapper-only] None/[] -> [] (behaviour unchanged)
            logger.debug("forward(): eq[%s] Prompt 4 attempt#1 simulated_response (%d item(s)) = %s",  # [wrapper-only] trace
                          eq_id, len(simulated_response), simulated_response)           # [wrapper-only] trace args

            # run.py lines 567-579: lock parameters on first pass.
            # NOTE: the ONLY deviation from run.py:571 is the added `and " = " in p` guard —
            # run.py uses `if p.startswith("Parameter(")` and would IndexError on a malformed
            # "Parameter(...)" line with no " = "; the guard skips such lines instead. Identical
            # behaviour for well-formed LLM output (every real Parameter line contains " = ").
            if is_first_pass and simulated_response:               # run.py:567 if is_first_pass and simulated_response
                locked = {                                         # run.py:568 locked = { ... }
                    p.split(" = ")[0]: p.split(" = ")[1]           # run.py:569 p.split(" = ")[0] : p.split(" = ")[1]
                    for p in simulated_response                    # run.py:570 for p in simulated_response
                    if p.startswith("Parameter(") and " = " in p   # run.py:571 if p.startswith("Parameter(")  [+ wrapper " = " guard]
                }
                if locked:                                         # run.py:573 if locked
                    locked_param_block = (                         # run.py:574-578 locked_param_block = ( ... )
                        "## Locked parameter values -- do not change on retry:\n"  # run.py:575 header line
                        + "\n".join(f"{k} = {v}" for k, v in locked.items())       # run.py:576 "\n".join(f"{k} = {v}" ...)
                        + "\n\n"                                   # run.py:577 trailing blank lines
                    )
                is_first_pass = False                              # run.py:579 is_first_pass = False
                logger.debug("forward(): eq[%s] locked_param_block (after first pass) = %r", eq_id, locked_param_block)  # [wrapper-only] trace

            if function_caller is not None:                       # run.py: DLL path always taken when use_dll (wrapper guards None when 32-bit Python missing)
                # run.py lines 581-596 (DLL available)
                try:                                              # [wrapper-only] DLL watchdog: a hung 32-bit solve → M7=0 for this eq, keep the run moving (2026-07-14 Rail freeze fix)
                    attempt = solve_with_dll_timeout(lambda: _attempt_solve(simulated_response, eq_to_solve, function_caller), function_caller)  # run.py:581 _attempt_solve  [imported, watchdog-guarded]
                except _DLLSolveTimeout as _to:                   # [wrapper-only] 32-bit DLL froze on these inputs (helper killed → self-restarts next eq)
                    logger.warning("forward(): eq[%s] DLL watchdog fired on attempt#1 (%s) -> M7=0, next equation", eq_id, _to)  # [wrapper-only] trace
                    eq_m7_scores.append(0.0)                      # [wrapper-only] count the frozen eq as a solve FAIL (not omitted from M7)
                    updated_dict[eq_id] = simulated_response      # [wrapper-only] keep the eq in updated_dict (run.py:639 always does this) so post-processing stays consistent
                    continue                                      # [wrapper-only] skip to the next equation
                substitution_dict = _build_substitution_dict(attempt.final_assignments)     # run.py:582 substitution_dict = _build_substitution_dict(attempt.final_assignments)  [imported]
                logger.debug("forward(): eq[%s] attempt#1 final_assignments = %s", eq_id, attempt.final_assignments)  # [wrapper-only] trace
                logger.debug("forward(): eq[%s] attempt#1 substitution_dict = %s", eq_id, substitution_dict)         # [wrapper-only] trace

                expressions = [_substitute_ops(merged_eq.final_substituted_equation, substitution_dict)]  # run.py:591 expressions = [_substitute_ops(merged_eq.final_substituted_equation, substitution_dict)]
                eq_list = []                                       # run.py:592 eq_list = []
                for expr in expressions:                           # run.py:593 for expr in expressions
                    eq_list = generate_bodmas_chain(expr)          # run.py:594 eq_list = generate_bodmas_chain(expr)  [imported]
                logger.debug("forward(): eq[%s] attempt#1 substituted expression = %s", eq_id, expressions)          # [wrapper-only] trace
                logger.debug("forward(): eq[%s] attempt#1 BODMAS chain (%d step(s)) = %s", eq_id, len(eq_list), eq_list)  # [wrapper-only] trace

                # run.py line 596: first compute_unselected_mux_variables uses eq_list (BODMAS output)
                all_unselected_vars |= compute_unselected_mux_variables(eq_list, substitution_dict)  # run.py:596 all_unselected_vars |= compute_unselected_mux_variables(eq_list, substitution_dict)  (2-arg)
                try:                                              # [wrapper-only] DLL watchdog around the resolve/verify call
                    final_check, try_fc = solve_with_dll_timeout(lambda: resolve_equation_block(eq_list, function_caller), function_caller)  # run.py:597 resolve_equation_block  [imported, watchdog-guarded]
                except _DLLSolveTimeout as _to:                   # [wrapper-only] DLL froze during verify → M7=0, next equation
                    logger.warning("forward(): eq[%s] DLL watchdog fired on resolve#1 (%s) -> M7=0, next equation", eq_id, _to)  # [wrapper-only] trace
                    eq_m7_scores.append(0.0)                      # [wrapper-only] frozen eq counts as FAIL
                    updated_dict[eq_id] = simulated_response      # [wrapper-only] keep the eq in updated_dict (run.py:639 always does this) so post-processing stays consistent
                    continue                                      # [wrapper-only] skip to the next equation
                logger.debug("forward(): eq[%s] attempt#1 final_check (DLL-resolved) = %s", eq_id, final_check)       # [wrapper-only] trace

                expected = merged_eq.equation_to_solve.split("=")[-1].strip()  # run.py:599 expected = merged_eq.equation_to_solve.split("=")[-1].strip()
                actual = final_check[-1].split("=")[-1].strip() if final_check else ""  # run.py:600 actual = final_check[-1].split("=")[-1].strip() if final_check else ""
                first_try_pass = (expected == actual)  # M7: capture before retry loop  # [wrapper-only] M7 signal (run.py has no score)
                n_retries_used = 0                     # [wrapper-only] M7 graded scoring: retries this eq needed
                logger.info("forward(): eq[%s] attempt#1: expected=%r actual=%r -> %s",  # [wrapper-only] trace
                             eq_id, expected, actual, "PASS" if first_try_pass else "MISMATCH")  # [wrapper-only] trace args

                # run.py lines 608-637: retry loop. run.py hardcodes `if retry_num < 1` (the intentional
                # 1-retry bug); here the guard is `retry_num < self.max_retries` (default 1 == run.py
                # EXACTLY, incl. the same 1.0/0.5/0.0 M7). max_retries>1 (e.g. 10) allows more backtrace
                # attempts to reach LHS==RHS; the range widens to fit the budget (>= max_retries+1 iters).
                for retry_num in range(max(12, self.max_retries + 2)):  # run.py:608 range(12); widened for larger budgets
                    if expected == actual:                         # run.py:609 if expected == actual
                        break                                      # run.py:611 break (DLL Verification PASS)
                    if retry_num < self.max_retries:               # run.py:612 `if retry_num < 1` — parameterized (default 1 == run.py)
                        n_retries_used += 1                        # [wrapper-only] count this retry for graded M7
                        # run.py builds the retry text via build_retry_instruction(); here the
                        # retry WORDING lives in the (GEPA-optimizable) MBDEquationSolverRetry
                        # signature, and the same diagnostic DATA is passed as input fields.
                        logger.info("forward(): eq[%s] ----- Prompt 4 RETRY #%d — calling retry LM (mismatch expected=%r actual=%r) -----",  # [wrapper-only] trace
                                     eq_id, retry_num + 1, expected, actual)            # [wrapper-only] trace args
                        retry_pred = self.equation_solver_retry(   # run.py:624 `solve_response = await call_llm(prompt)` (retry) → DSPy predictor
                            equation_chain="\n".join(eq_to_solve), # run.py:617-618 eq_to_solve arg
                            target_value=lhs_value,                # run.py:618 lhs_value arg
                            const_values=json.dumps(filtered_const_values),  # run.py:618 filtered_const_values arg
                            locked_param_block=locked_param_block, # run.py:619 locked_param_block (run.py concatenates +extra; here `extra` is the signature instruction)
                            expected=expected,                     # run.py:614 build_retry_instruction(expected, ...)
                            actual=actual,                         # run.py:614 build_retry_instruction(..., actual, ...)
                            dll_trace=str(try_fc),                 # run.py:614 build_retry_instruction(..., try_fc, ...)
                            eq_list=str(eq_list),                  # run.py:614 build_retry_instruction(..., eq_list, ...)
                            solved_equations=str(attempt.solved_equations),  # run.py:615 build_retry_instruction(..., attempt.solved_equations)
                        )                                          # run.py:617-620 generate_equation_solution_prompt_dll(..., locked_param_block + extra)
                        logger.debug("forward(): eq[%s] retry#%d raw variable_assignments = %s", eq_id, retry_num + 1, retry_pred.variable_assignments)  # [wrapper-only] trace
                        _raw_json = _extract_dll_lenient(retry_pred.variable_assignments)  # run.py:627 simulated_response = extract_json_from_text_dll(solve_response)
                        self._solver_attempts += 1                 # [wrapper-only] solver JSON health: retry solve attempt
                        if _raw_json is None:                      # [wrapper-only] None == "No valid JSON list found"
                            self._solver_json_fails += 1           # [wrapper-only] count the JSON parse failure
                        simulated_response = _raw_json or []       # [wrapper-only] None/[] -> [] (behaviour unchanged)
                        logger.debug("forward(): eq[%s] retry#%d simulated_response (%d item(s)) = %s",  # [wrapper-only] trace
                                      eq_id, retry_num + 1, len(simulated_response), simulated_response)  # [wrapper-only] trace args
                        try:                                       # [wrapper-only] DLL watchdog around the retry solve+verify
                            attempt = solve_with_dll_timeout(lambda: _attempt_solve(simulated_response, eq_to_solve, function_caller), function_caller)  # run.py:628 _attempt_solve  [guarded]
                            substitution_dict = _build_substitution_dict(attempt.final_assignments)     # run.py:629 substitution_dict = _build_substitution_dict(...)  [imported]
                            expressions = [_substitute_ops(merged_eq.final_substituted_equation, substitution_dict)]  # run.py:630 expressions = [_substitute_ops(...)]
                            for expr in expressions:                   # run.py:631 for expr in expressions
                                eq_list = generate_bodmas_chain(expr)  # run.py:632 eq_list = generate_bodmas_chain(expr)  [imported]
                            final_check, try_fc = solve_with_dll_timeout(lambda: resolve_equation_block(eq_list, function_caller), function_caller)  # run.py:633 resolve_equation_block  [guarded]
                        except _DLLSolveTimeout as _to:            # [wrapper-only] retry solve froze → stop retrying, force mismatch → M7=0
                            logger.warning("forward(): eq[%s] DLL watchdog fired on retry#%d (%s) -> stop retrying, M7=0", eq_id, retry_num + 1, _to)  # [wrapper-only] trace
                            actual = ""                            # [wrapper-only] force expected != actual at the post-loop M7 check
                            break                                  # [wrapper-only] exit the retry loop (eq scored 0)
                        actual = final_check[-1].split("=")[-1].strip() if final_check else ""  # run.py:634 actual = final_check[-1].split("=")[-1].strip() if final_check else ""
                        logger.info("forward(): eq[%s] retry#%d: expected=%r actual=%r -> %s",  # [wrapper-only] trace
                                     eq_id, retry_num + 1, expected, actual, "PASS" if expected == actual else "FAIL")  # [wrapper-only] trace args
                    else:                                          # run.py:635 else
                        break                                      # run.py:637 break (giving up after the single retry)

                # M7 inline (GRADED): a solved equation scores max(0, 1 - n_retries_used/(REF+1)), REF =
                # m7_reference_retries (DEFAULT max(4, max_retries)). Two regimes:
                #   • budget <= 4 -> REF=4 -> a FIXED 0.2 penalty PER RETRY: 1.0/0.8/0.6/0.4/0.2/0.0. So a
                #     1-retry solve scores 0.8 whether the budget is 1 or 4 -> M7 comparable across small budgets.
                #   • budget  > 4 -> REF=budget -> a 1/(budget+1) penalty per retry: e.g. budget 9 gives
                #     1.0/0.9/0.8/.../0.1, and a never-solved eq 0.0 -> the 0.1-gap ladder "1, 0.9, ... 0".
                # A never-solved equation always scores 0.0. The weakspot's M7 = mean over its equations.  # [wrapper-only] M7 metric (no run.py analog)
                if expected == actual:                             # [wrapper-only] solved (first try or after n_retries_used retries)
                    _m7_ref = self.m7_reference_retries if self.m7_reference_retries is not None else max(4, int(getattr(self, "max_retries", 0) or 0))  # [wrapper-only] ref=max(4,budget): 0.2/retry up to 4, then 1/(budget+1)/retry
                    _m7_eq = max(0.0, 1.0 - n_retries_used / (_m7_ref + 1))  # [wrapper-only] graded; 0.2 per retry at ref=4, 1/(ref+1) above
                    eq_m7_scores.append(_m7_eq)                    # [wrapper-only] M7 in (0,1]
                    logger.info("forward(): eq[%s] M7 inline score = %.3f (solved after %d retr%s, ref=%d)",  # [wrapper-only] trace
                                 eq_id, _m7_eq, n_retries_used, "y" if n_retries_used == 1 else "ies", _m7_ref)  # [wrapper-only] trace args
                else:                                              # [wrapper-only] never matched within the retry budget
                    eq_m7_scores.append(0.0)                       # [wrapper-only] M7 = 0.0
                    logger.info("forward(): eq[%s] M7 inline score = 0.0 (FAIL after %d retries — expected=%r actual=%r)",  # [wrapper-only] trace
                                 eq_id, n_retries_used, expected, actual)  # [wrapper-only] trace args
            else:                                                  # [wrapper-only] DLL unavailable branch (run.py always has DLL on this path)
                # function_caller=None (no 32-bit Python) — Prompt 4 ran, DLL verification skipped
                logger.warning("forward(): eq[%s] function_caller is None -> DLL verification SKIPPED, no M7 inline score for this eq", eq_id)  # [wrapper-only] trace

            # run.py line 639
            updated_dict[eq_id] = simulated_response               # run.py:639 updated_dict[eq_id] = simulated_response
            logger.debug("forward(): eq[%s] updated_dict[%s] = %s", eq_id, eq_id, simulated_response)  # [wrapper-only] trace

        # ── M7 inline score (must be computed before post-processing clears state) ─  # [wrapper-only] no run.py analog
        if eq_m7_scores:                                       # [wrapper-only] at least one equation was DLL-verified
            m7_inline_score = sum(eq_m7_scores) / len(eq_m7_scores)  # [wrapper-only] mean of per-eq GRADED scores
            n_first = sum(1 for s in eq_m7_scores if s == 1.0)      # [wrapper-only] solved first try (score 1.0)
            n_retry = sum(1 for s in eq_m7_scores if 0.0 < s < 1.0) # [wrapper-only] solved after >=1 retry (graded, budget-aware)
            n_fail  = sum(1 for s in eq_m7_scores if s == 0.0)      # [wrapper-only] never solved within the budget
            m7_inline_details = (                              # [wrapper-only] human-readable breakdown
                f"DLL inline: {n_first} first-try + {n_retry} retry-pass + {n_fail} fail"  # [wrapper-only]
                f" / {len(eq_m7_scores)} eqs"                  # [wrapper-only]
            )
        else:                                                  # [wrapper-only] no equations verified (DLL off / fn-call weakspot)
            m7_inline_score = None  # DLL unavailable → metric falls back to post-hoc  # [wrapper-only]
            m7_inline_details = "DLL unavailable — no inline M7"  # [wrapper-only]
        logger.info("forward(): M7 INLINE — score=%s (%s)", m7_inline_score, m7_inline_details)  # [wrapper-only] trace

        # ── Phase 4 branch — run.py lines 876-888 ───────────────────────────────
        # run.py only runs equation solving + the post-process chain when there ARE
        # equations (`if len(equation_dict) != 0`); otherwise it just maps the
        # (UUID-free) test cases straight through (run.py:888). Mirror that exactly.
        if len(equation_dict) != 0:                            # run.py:876 if len(equation_dict) != 0
            # ── Post-processing: run.py lines 653-668 VERBATIM ──────────────────────
            logger.info("forward(): ----- Post-processing chain -----")  # [wrapper-only] trace

            # run.py line 653: MUST call update_equation_values BEFORE map_equations_to_json
            # This mutates equation_dict: UUID strings → solved value lists
            update_equation_values(equation_dict, updated_dict)  # run.py:653 update_equation_values(equation_dict, updated_dict)  [imported, mutates equation_dict]
            logger.debug("forward(): post-process — update_equation_values() done; equation_dict now = %s", equation_dict)  # [wrapper-only] trace

            # run.py line 654
            pre_final = map_equations_to_json(updated_json, equation_dict)  # run.py:654 pre_final = map_equations_to_json(updated_json, equation_dict)  [imported]
            logger.debug("forward(): post-process — map_equations_to_json() -> pre_final = %s", pre_final)  # [wrapper-only] trace

            # run.py lines 655-657
            cleaned = remove_variable_and_parameter_blocks(pre_final)  # run.py:655 cleaned = remove_variable_and_parameter_blocks(pre_final)  [imported]
            logger.debug("forward(): post-process — remove_variable_and_parameter_blocks() -> %s", cleaned)  # [wrapper-only] trace
            flattened = flatten_precondition_and_actions(cleaned)  # run.py:656 flattened = flatten_precondition_and_actions(cleaned)  [imported]
            logger.debug("forward(): post-process — flatten_precondition_and_actions() -> %s", flattened)  # [wrapper-only] trace
            deduped = deduplicate_actions(flattened)  # dedup bug preserved  # run.py:657 deduped = deduplicate_actions(flattened)  [imported; Frankenstein-merge bug preserved]
            logger.debug("forward(): post-process — deduplicate_actions() -> %s", deduped)  # [wrapper-only] trace

            # run.py lines 659-665: second compute_unselected_mux_variables call
            # Uses precomputed op_map (from example) instead of rebuilding from dependency_mapping
            if merged_equation_map:                            # run.py:659 if merged_equation_map
                last_eq = list(merged_equation_map.values())[-1]  # run.py:660 last_eq = list(merged_equation_map.values())[-1]
                last_eq_to_solve = last_eq.assignment_chain.split("\n")[::-1]  # run.py:661 last_eq_to_solve = last_eq.assignment_chain.split("\n")[::-1]
                stored_op_map = json.loads(op_map_json) if op_map_json and op_map_json != "{}" else {}  # run.py:662 op_map = build_op_mapping(eq_ctx.dependency_mapping) (precomputed in mbd_data, JSON here)
                all_unselected_vars |= compute_unselected_mux_variables(  # run.py:663-665 all_unselected_vars |= compute_unselected_mux_variables(...)  (3-arg)
                    last_eq_to_solve, substitution_dict, stored_op_map  # run.py:664 args: last_eq_to_solve, substitution_dict, op_map
                )
                logger.debug("forward(): post-process — all_unselected_vars (final) = %s", all_unselected_vars)  # [wrapper-only] trace

            # run.py lines 667-668
            final_json = clean_deduplicated_solution(deduped)  # run.py:667 final_json = clean_deduplicated_solution(deduped)  [imported]
            logger.debug("forward(): post-process — clean_deduplicated_solution() -> %s", final_json)  # [wrapper-only] trace
            final_json = remove_unselected_mux_actions(final_json, all_unselected_vars)  # run.py:668 final_json = remove_unselected_mux_actions(final_json, all_unselected_vars)  [imported]
            logger.info("forward(): post-process — remove_unselected_mux_actions() -> final_json with %d test case(s)",  # [wrapper-only] trace
                         len(final_json) if isinstance(final_json, list) else -1)  # [wrapper-only] trace arg
            logger.debug("forward(): FINAL JSON = %s", json.dumps(final_json, indent=2))  # [wrapper-only] trace
        else:                                                  # run.py:887 else (equation_dict empty)
            # run.py lines 887-888: no equations to solve → just map the test cases through
            # (map_equations_to_json with an unchanged equation_dict strips type wrappers but
            # applies no solved values; the post-process chain is NOT run, exactly like run.py).
            logger.info("forward(): no equations to solve (equation_dict empty) — run.py:888 path")  # [wrapper-only] trace
            final_json = map_equations_to_json(updated_json, equation_dict)  # run.py:888 final_json = map_equations_to_json(updated_json, equation_dict)  [imported]
            logger.debug("forward(): final_json (no-equation path) = %s", final_json)  # [wrapper-only] trace

        logger.info("forward(): END — test_cases_pre_postprocess (Prompt2 raw, %d chars), "  # [wrapper-only] trace
                     "final test_cases (%d chars), m7_inline_score=%s",                       # [wrapper-only] trace
                     len(test_cases_raw or ""), len(json.dumps(final_json)), m7_inline_score)  # [wrapper-only] trace args
        logger.info("=" * 70)                                  # [wrapper-only] trace banner

        # run.py: run_pipeline wraps final_json in WeakspotTestCaseOutput → XLSM (890-927); the wrapper
        # instead returns a dspy.Prediction so GEPA/dspy.Evaluate can score it via the M1..M8 metric.
        return dspy.Prediction(                                # [wrapper-only] DSPy return (replaces run.py WeakspotTestCaseOutput/XLSM)
            context_analysis=context_analysis,                 # [wrapper-only] Prompt-1 output (carried for metric/feedback)
            test_cases_pre_postprocess=test_cases_raw,   # Prompt 2 raw — for staged metric  # [wrapper-only] M1/M3/M4/M6 input
            test_cases=json.dumps(final_json),            # final pipeline output  # run.py:893 final_json (the run.py-faithful test cases) → JSON string for the metric
            m7_score=m7_inline_score,     # inline 1/N score; None if DLL unavailable  # [wrapper-only] M7 metric signal
            m7_details=m7_inline_details, # breakdown string for logging  # [wrapper-only] M7 detail
        )

    @classmethod                                               # [wrapper-only] classmethod so the FunctionCaller can be CACHED on the class (cls._fc_cache); run.py builds it once per one-shot run (run.py:821-826), no caching
    def _get_function_caller(cls, dll_path, signalsheet_data, model_interface_data):  # run.py:813-826 FunctionCaller setup block (guarded by use_dll); wrapper adds per-dll_path caching

        """Initialize (and CACHE) the FunctionCaller — corresponds-to run.py's DLL setup block
        (run.py:798-826): DLL-presence/32-bit-Python check `use_dll` (run.py:801-806), then
        `function_caller = FunctionCaller(FCname, work_dir, signal_sheet_data, model_interface_data)`
        (run.py:821-826, guarded by `if use_dll and not args.dry_run`, run.py:813).

        [wrapper-only] CACHING is the ONE structural difference from run.py: run.py builds ONE
        FunctionCaller fresh per one-shot run (run.py:821-826) and tears it down at the end
        (function_caller.cleanup(), run.py:907). The wrapper is called once per forward() across many
        dspy.Evaluate/GEPA iterations, so it caches the FunctionCaller per dll_path in cls._fc_cache
        (guarded by cls._fc_cache_lock) and never calls cleanup(). The FunctionCaller CONSTRUCTOR ARGS
        are byte-for-byte run.py's (fcname/work_dir derived from dll_path exactly as run.py:728-730;
        signal_sheet_data/model_interface_data re-parsed to the SAME list-of-dicts run.py passes —
        see the inline anchors below).

        Returns None if DLL is unavailable (missing file — run.py's use_dll=False LLM-only path — or a
        missing 32-bit Python, cached as a negative result). When None, forward()'s solver loop still
        runs Prompt 4 but skips DLL verification (no inline M7).
        """
        if not dll_path or not os.path.exists(dll_path):       # run.py:800-806 use_dll requires os.path.isfile(dll_path) (+ 32-bit Python)
            logger.warning("_get_function_caller: dll_path %r missing or does not exist -> fc=None", dll_path)  # [wrapper-only] trace
            return None                                        # run.py: use_dll=False → LLM-only path; wrapper returns None (DLL verify skipped)
        with cls._fc_cache_lock:                               # [wrapper-only] caching not in run.py (run.py builds one FC per one-shot run)
            if dll_path in cls._fc_cache:                      # [wrapper-only] cache hit (reuse across eval/GEPA calls)
                logger.info("_get_function_caller: cache HIT for dll_path=%s -> %s", dll_path,  # [wrapper-only] trace
                             "fc available" if cls._fc_cache[dll_path] is not None else "None (cached failure)")  # [wrapper-only] trace arg
                return cls._fc_cache[dll_path]                 # [wrapper-only] return cached FunctionCaller
            # run.py lines 729-730: work_dir = dirname(fc_dir), fcname = basename(fc_dir)
            # dll_path = <data_root>/<fc_name>/env_libs/<dll_file>
            env_libs_dir = os.path.dirname(dll_path)      # <data_root>/<fc_name>/env_libs   # run.py:800 dll_path = join(fc_dir, "env_libs", f"{fcname}.dll") — inverted here
            fc_dir       = os.path.dirname(env_libs_dir)  # <data_root>/<fc_name>            # run.py:728 fc_dir = abspath(args.fc_dir)
            fcname       = os.path.basename(fc_dir)       # fc_name string                  # run.py:729 fcname = basename(fc_dir)
            work_dir     = os.path.dirname(fc_dir)        # <data_root>                      # run.py:730 work_dir = dirname(fc_dir)
            logger.info("_get_function_caller: cache MISS — dll_path=%s FCname=%s work_dir=%s", dll_path, fcname, work_dir)  # [wrapper-only] trace

            # run.py line 792: signal_sheet_data = convert_signal_sheet_to_list(signal_sheet_df)
            # convert_signal_sheet_to_list produces list-of-dicts from DataFrame.
            # Here we have raw CSV text — csv.DictReader produces the same structure.
            signal_sheet_list = list(csv.DictReader(io.StringIO(signalsheet_data or "")))  # run.py:817 signal_sheet_data = convert_signal_sheet_to_list(signal_sheet_df) (same list-of-dicts)
            logger.debug("_get_function_caller: signal_sheet_list has %d row(s)", len(signal_sheet_list))  # [wrapper-only] trace

            # run.py:756/825 — model_interface_data is load_model_interface()'s LIST-OF-DICTS
            # (csv.DictReader), which FunctionCaller iterates as rows (row["Name"]/row["Value"],
            # function_caller.py:494-496). mbd_data.py stores the raw CSV text; parse it to the SAME
            # list-of-dicts — NOT a DataFrame (`for row in DataFrame` yields column names, so
            # row["Name"] breaks and CharTable calibration is silently dropped).
            mi_list = list(csv.DictReader(io.StringIO(model_interface_data or "")))  # run.py:756 load_model_interface() list-of-dicts (re-parsed from stored CSV text)
            logger.debug("_get_function_caller: model_interface_data has %d row(s)", len(mi_list))  # [wrapper-only] trace

            # run.py lines 793-798; catch RuntimeError if 32-bit Python is missing
            try:                                               # run.py:813 if use_dll and not args.dry_run
                cls._fc_cache[dll_path] = FunctionCaller(      # run.py:821 function_caller = FunctionCaller(
                    FCname=fcname,                             # run.py:822 FCname=fcname
                    work_dir=work_dir,                         # run.py:823 work_dir=work_dir
                    signal_sheet_data=signal_sheet_list,       # run.py:824 signal_sheet_data=signal_sheet_data
                    model_interface_data=mi_list,              # run.py:825 model_interface_data=model_interface_data
                )
                logger.info("_get_function_caller: FunctionCaller initialized successfully (FCname=%s)", fcname)  # [wrapper-only] trace
            except Exception as _fc_err:                       # [wrapper-only] 32-bit Python missing → DLL verify off (run.py would set use_dll=False upstream)
                logging.getLogger("extract_mbd_full").warning(  # [wrapper-only] trace
                    "FunctionCaller unavailable (%s) — DLL verification disabled, M7=0.5", _fc_err  # [wrapper-only] trace
                )
                cls._fc_cache[dll_path] = None  # cache None so we don't retry every call  # [wrapper-only] negative-cache
            return cls._fc_cache[dll_path]                     # [wrapper-only] return the (possibly None) FunctionCaller


# ═══════════════════════════════════════════════════════════════════
# CLI  +  RUN HARNESS   —   [WRAPPER-ONLY] (no run.py counterpart)
# ═══════════════════════════════════════════════════════════════════
# Everything from here to the end of the file is GEPA-optimization / dspy.Evaluate /
# CLI / logging / run-summary scaffolding. run.py has NONE of this: run.py is driven by
# load_config()→run_config.json (200-258) + main() (934-944) and ends by writing an XLSM
# (890-927). It computes no score, runs no optimizer, and has no argparse. The closest run.py
# analogs are only loose: build_parser()/main() here ≙ run.py load_config()/main() (process
# entry + config), and dspy.Evaluate(metric=...) is the wrapper-added evaluation run.py lacks.
# To avoid fake one-to-one refs, the functions below carry a per-function [wrapper-only] header
# instead of per-line run.py:NNN anchors; the genuine run.py pipeline mapping lives in forward()
# and _get_function_caller above (and in mbd_data.py / mbd_metric.py / signatures/mbd.py).
def build_parser() -> argparse.ArgumentParser:   # [wrapper-only] CLI args (eval-only / optimizer / gepa budget) — no run.py analog
    parser = argparse.ArgumentParser(description="MBD-UTA Full Pipeline DSPy Optimization")  # [wrapper-only] argparse; run.py uses JSON config (load_config:200), no CLI flags
    # [wrapper-only] --model-name: which student LM to build the DSPy predictors with; run.py reads model from config.py/env, not a flag
    parser.add_argument(
        "--model-name", type=str, default=os.getenv("MODEL_NAME"),
        help="Student model name (default: from .env MODEL_NAME)",
    )
    # [wrapper-only] --no-cache: toggle DSPy LM cache; run.py has no DSPy cache layer
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable LLM response caching",
    )
    # [wrapper-only] --max-cost-usd: HARD cost cap; force-stops the run once cumulative LM spend exceeds it
    parser.add_argument(
        "--max-cost-usd", type=float, default=None,
        help="Hard USD cost cap. Tracks tokens across student+M4-judge+reflection via a litellm callback, "
             "logs a running total, and force-stops (os._exit) once cumulative cost exceeds this. "
             "e.g. --max-cost-usd 20",
    )
    # [wrapper-only] --no-cot: Predict vs ChainOfThought predictor class; run.py has no DSPy predictors
    parser.add_argument(
        "--no-cot", action="store_true",
        help="Use dspy.Predict instead of dspy.ChainOfThought",
    )
    # [wrapper-only] --eval-only: run dspy.Evaluate only (no GEPA); run.py never scores/evaluates
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only evaluate baseline, no optimization. Evaluates the VAL set (same split GEPA uses), "
             "so it is comparable to a GEPA baseline. Add --eval-all-data to score the whole dataset.",
    )
    # [wrapper-only] --eval-all-data: widen --eval-only scope to full dataset; eval harness knob, no run.py analog
    parser.add_argument(
        "--eval-all-data", action="store_true",
        help="With --eval-only: score the ENTIRE dataset instead of just the val set "
             "(the previous comprehensive-baseline behavior).",
    )
    # [wrapper-only] --eval-test: score held-out test split; eval harness knob, no run.py analog
    parser.add_argument(
        "--eval-test", action="store_true",
        help="Evaluate on test set (requires --load-extractor)",
    )
    # [wrapper-only] --inject-kb-guidelines: opt-in Prompt-2 KB steering (see forward():382); NOT in run.py
    parser.add_argument(
        "--inject-kb-guidelines", action="store_true",
        help="Append each weakspot's resolved KB guidelines (new KB -> old KB) into the Prompt-2 "
             "(test-case generator) context so generation is coverage-aware. Applies to eval AND GEPA "
             "(propagated to the optimizer subprocess via MBD_INJECT_KB_GUIDELINES). Default OFF -> "
             "prompt is byte-identical to the frozen MIDAS seed.",
    )
    # [wrapper-only] --required-io-outputs: opt-in Prompt-2 output override from the expert Required-I/O KB (see forward():425); NOT in run.py
    parser.add_argument(
        "--required-io-outputs", action="store_true",
        help="Override Prompt-2's Expected Results (instance_names) with the expert Required-I/O KB's "
             "required_outputs for each weakspot (utils/required_io.json; path via --required-io-path / "
             "MBD_REQUIRED_IO_PATH). Lets experts steer generation to a chosen (e.g. intermediate) output. "
             "Applies to eval AND GEPA (propagated via MBD_REQUIRED_IO_IN_PROMPT). No KB entry -> keep "
             "instance_names. Default OFF -> byte-identical to the frozen seed.",
    )
    # [wrapper-only] --required-io-path: point the Required-I/O KB (scoring M1/M2 + the Prompt-2 override) at a custom file
    parser.add_argument(
        "--required-io-path", type=str, default="",
        help="Path to the Required-I/O KB JSON (default utils/required_io.json). Drives BOTH scoring "
             "(M1 required_outputs / M2 input+precond completeness) AND the --required-io-outputs override.",
    )
    # [wrapper-only] --max-retries: Prompt-4 DLL backtrace retry budget; default 1 mirrors run.py's single-retry (run.py:653-668), higher values are a wrapper extension
    parser.add_argument(
        "--max-retries", type=int, default=1,
        help="DLL Prompt-4 backtrace retry budget per equation. Default 1 == run.py's single-retry "
             "behaviour AND M7's 1.0/0.5/0.0. Raise (e.g. 10) to give the solver more attempts to reach "
             "LHS==RHS; M7 then becomes graded (1 - retries/(max_retries+1)). Applies to eval AND GEPA "
             "(set on generator + MBD_MAX_RETRIES env).",
    )
    # [wrapper-only] --optimizer: which prompt optimizer; run.py has no optimizer at all
    parser.add_argument(
        "--optimizer", type=str, default="gepa",
        choices=["gepa"],
        help="Optimizer to use (default: gepa)",
    )
    # [wrapper-only] --gepa-auto: GEPA intensity preset (budget); GEPA-specific, no run.py analog
    parser.add_argument(
        "--gepa-auto", type=str, default="light",
        choices=["light", "medium", "heavy"],
        help="GEPA intensity preset (default: light). Ignored if --gepa-max-metric-calls is set.",
    )
    # [wrapper-only] --gepa-max-metric-calls: explicit GEPA metric-call budget override; GEPA-specific
    parser.add_argument(
        "--gepa-max-metric-calls", type=int, default=None,
        help="Explicit GEPA budget (total metric calls) for a 'lighter than light' run. "
             "When set, OVERRIDES --gepa-auto (light is ~1495). Try ~300-500 for a quick run.",
    )
    # [wrapper-only] --load-extractor: load a previously GEPA-optimized module from disk; run.py has no save/load of an optimized program
    parser.add_argument(
        "--load-extractor", type=str, default=None,
        help="Path to previously saved optimized extractor to load",
    )
    # [wrapper-only] --use-all-tasks: dataset selector (all weakspots from run_config.json); dataset shaping is a wrapper concern
    parser.add_argument(
        "--use-all-tasks", action="store_true",
        help="(DEFAULT now) Use all weakspots from run_config.json. Kept for back-compat/clarity; "
             "all-tasks is the default dataset. Use --dev-subset for the small hardcoded dev set.",
    )
    # [wrapper-only] --dev-subset: use small hardcoded dev set instead of run_config.json tasks; dev convenience, no run.py analog
    parser.add_argument(
        "--dev-subset", action="store_true",
        help="Use the small hardcoded dev subset (_default_fc_entries) instead of the default "
             "all-tasks set from run_config.json.",
    )
    # [wrapper-only] --num-threads: dspy.Evaluate/GEPA parallelism; run.py runs weakspots sequentially via asyncio (main:940-944)
    parser.add_argument(
        "--num-threads", type=int, default=2,
        help="Number of threads for parallel evaluation (default: 2)",
    )
    # [wrapper-only] --repeat: N re-runs per example for noise profiling (_measure_metric_noise); run.py runs each weakspot once
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="With --eval-only: run each example N times and log per-metric mean/std "
             "to MEASURE the LLM-induced run-to-run variability (no caching). Default 1.",
    )
    # [wrapper-only] --run-timestamp: reuse a caller-supplied run id; run.py always generates its own (run.py:938 datetime.now().strftime)
    parser.add_argument(
        "--run-timestamp", type=str, default=None,
        help="Pre-generated timestamp (YYYYmmddHHMMSS). Falls back to datetime.now().",
    )
    return parser  # [wrapper-only] hand the configured parser back to main()


# ═══════════════════════════════════════════════════════════════════
# TEE LOGGER  (mirrors extract_mbd.py _TeeLogger)
# ═══════════════════════════════════════════════════════════════════
class _TeeLogger:   # [wrapper-only] tee stdout/stderr to a shared log file — no run.py analog
    """Redirect a stream (stdout or stderr) to both terminal and a shared log file."""

    def __init__(self, original_stream, log_file):  # [wrapper-only] wrap the real stream + shared log file
        self._original = original_stream  # [wrapper-only] keep the terminal stream to still print live
        self._file = log_file  # [wrapper-only] shared log file to also capture output

    def write(self, msg: str):  # [wrapper-only] file-like write() so it can replace sys.stdout/stderr
        self._file.write(msg)  # [wrapper-only] persist to the log file first
        self._file.flush()  # [wrapper-only] flush so the log is complete even if the run crashes
        try:  # [wrapper-only] terminal may reject non-encodable chars
            self._original.write(msg)  # [wrapper-only] mirror to the terminal
        except UnicodeEncodeError:  # [wrapper-only] Windows consoles often can't encode UTF-8 glyphs
            # [wrapper-only] re-encode with replacement so a bad glyph never kills the run
            safe_msg = msg.encode(
                getattr(self._original, "encoding", "utf-8") or "utf-8",
                errors="replace",
            ).decode(
                getattr(self._original, "encoding", "utf-8") or "utf-8",
                errors="replace",
            )
            self._original.write(safe_msg)  # [wrapper-only] write the sanitized text

    def flush(self):  # [wrapper-only] file-like flush() for the tee
        self._file.flush()  # [wrapper-only] flush the log file
        self._original.flush()  # [wrapper-only] flush the terminal

    def fileno(self):  # [wrapper-only] delegate fileno() so libs treating this as a real fd work
        return self._original.fileno()  # [wrapper-only] use the underlying terminal fd

    def isatty(self):  # [wrapper-only] delegate isatty() so color/tty detection works
        return self._original.isatty()  # [wrapper-only] report the terminal's tty status


# ═══════════════════════════════════════════════════════════════════
# PER-EXAMPLE DIAGNOSTIC HELPER
# ═══════════════════════════════════════════════════════════════════
def _print_per_example_scores(generator: MBDFullPipelineModule, examples, logger=None):  # [wrapper-only] per-example M1..M8 breakdown — metric diag, no run.py analog
    """Run inference on each example ONCE, print the per-metric breakdown, and RETURN the list of
    per-example composite totals (0-1). The caller derives the baseline = mean(totals) from these
    SAME predictions, so --eval-only needs only ONE pipeline pass (no redundant second dspy.Evaluate)
    and the headline score is byte-consistent with the breakdown even with caching OFF."""
    _log = logger or logging.getLogger("extract_mbd_full")  # [wrapper-only] fall back to the module logger if none passed

    # [wrapper-only] print a human-readable section header (this diagnostic has no run.py counterpart)
    print("\n" + "-" * 70)
    print("PER-EXAMPLE METRIC BREAKDOWN")
    print("-" * 70)
    _log.info("=== PER-EXAMPLE METRIC BREAKDOWN ===")  # [wrapper-only] mirror the header into the log

    # Weights must match mbd_metric.py WEIGHT_M* constants exactly (M1=.20 M2=.15 M3=.15 M4=.20 M7=.15 M8=.15)
    # [wrapper-only] mirror so the printed total == mbd_testcase_metric composite
    WEIGHTS = [
        ("m1", 0.20), ("m2", 0.15), ("m3", 0.15), ("m4", 0.20),
        ("m7", 0.15), ("m8", 0.15),
    ]

    totals: list = []   # per-example composite totals (0-1); mean → the eval-only baseline
    for i, ex in enumerate(examples):  # [wrapper-only] iterate the eval examples once (single pipeline pass)
        fc_tag = f"{getattr(ex, 'fc_name', '?')} ws={getattr(ex, 'weakspot_id', '?')}"  # [wrapper-only] short id (fc + weakspot) for the printed row
        try:  # [wrapper-only] isolate per-example failures so one bad example doesn't abort the sweep
            pred = generator(  # [wrapper-only] run the full pipeline once == forward() (run.py run_pipeline body)
                # [wrapper-only] the args below unpack the DSPy Example fields into forward()'s signature (see forward():342+)
                equation_json=ex.equation_json,
                const_values=ex.const_values,
                weakspot_context=ex.weakspot_context,
                instance_names=ex.instance_names,
                fc_name=getattr(ex, "fc_name", ""), weakspot_id=getattr(ex, "weakspot_id", ""),  # [wrapper-only] weakspot identity for the Required-I/O KB output override (forward **kwargs; not rendered)
                raster_time=ex.raster_time,
                weakspot_operation=getattr(ex, "weakspot_operation", ""),
                example_explanation_output=getattr(ex, "example_explanation_output", ""),
                dll_path=getattr(ex, "dll_path", ""),
                signalsheet_data=getattr(ex, "signalsheet_data", ""),
                model_interface_data=getattr(ex, "model_interface_data", ""),
                relations_json=getattr(ex, "relations_json", "{}"),
                equation_results_json=getattr(ex, "equation_results_json", "[]"),
                op_map_json=getattr(ex, "op_map_json", "{}"),
                weakspot_op_key=getattr(ex, "weakspot_op_key", ""),
            )
            scores = _compute_all_components(ex, pred)  # [wrapper-only] run M1..M8 on this (example, pred) (mbd_metric.py:1783)
            total = sum(w * scores[k] for k, w in WEIGHTS)  # [wrapper-only] weighted composite == mbd_testcase_metric (mbd_metric.py:1900-1906)
            totals.append(total)   # collect for the single-pass baseline mean
            # [wrapper-only] build the one-line per-metric summary row
            msg = (
                f"[{i+1}] {fc_tag} → total={total:.3f} | "
                f"M1={scores['m1']:.2f} M2={scores['m2']:.2f} M3={scores['m3']:.2f} "
                f"M4={scores['m4']:.2f} "
                f"M7={scores['m7']:.2f} M8={scores['m8']:.2f}"
            )
            # [wrapper-only] print the summary row then each metric's human-readable detail string
            print(f"  {msg}")
            print(f"       M1: {scores['m1_details']}")
            print(f"       M2: {scores['m2_details']}")
            print(f"       M3: {scores['m3_details']}")
            print(f"       M4: {scores['m4_details']}")
            print(f"       M7: {scores['m7_details']}")
            print(f"       M8: {scores['m8_details']}")
            _log.info(msg)  # [wrapper-only] also log the summary row
        except Exception as e:  # [wrapper-only] one example blew up — record it and keep going
            err_msg = f"[{i+1}] {fc_tag} → ERROR: {e}"  # [wrapper-only] format the error row
            print(f"  {err_msg}")  # [wrapper-only] show the error
            _log.error(err_msg, exc_info=True)  # [wrapper-only] log with traceback

    return totals  # [wrapper-only] hand back per-example totals; caller means them into the baseline


def _measure_metric_noise(generator: MBDFullPipelineModule, examples, repeat: int, logger=None):  # [wrapper-only] LLM run-to-run noise profiler — no run.py analog
    """Run each example `repeat` times (cache stays OFF) and log per-metric mean ± std.

    This MEASURES the LLM-induced run-to-run variability directly — the same example,
    same prompts, repeated — so you can SEE how much each metric jitters (M2/M4 most,
    M3/M7/M8 least). Reports per-example and an aggregate "noise profile".
    """
    import statistics as _stat  # [wrapper-only] local import for mean/pstdev of the metric series
    _log = logger or logging.getLogger("extract_mbd_full")  # [wrapper-only] fall back to module logger
    metrics = ["total", "m1", "m2", "m3", "m4", "m7", "m8"]  # [wrapper-only] metric keys to profile (composite + M1..M8; M6 removed)
    # [wrapper-only] same composite weights as mbd_metric.py (M1=.20 M2=.15 M3=.15 M4=.20 M7=.15 M8=.15)
    WEIGHTS = [("m1", 0.20), ("m2", 0.15), ("m3", 0.15), ("m4", 0.20),
               ("m7", 0.15), ("m8", 0.15)]
    # [wrapper-only] print the noise-measurement header
    print("\n" + "=" * 70)
    print(f"METRIC NOISE MEASUREMENT — {repeat} repeats per example (cache OFF)")
    print("=" * 70)
    _log.info("=== METRIC NOISE MEASUREMENT (repeat=%d) ===", repeat)  # [wrapper-only] header into log

    agg_std = {m: [] for m in metrics}  # [wrapper-only] accumulate per-metric std across examples for the aggregate profile
    for i, ex in enumerate(examples):  # [wrapper-only] iterate examples
        fc_tag = f"{getattr(ex, 'fc_name', '?')} ws={getattr(ex, 'weakspot_id', '?')}"  # [wrapper-only] short id for the printed row
        series = {m: [] for m in metrics}  # [wrapper-only] this example's per-metric samples across repeats
        for r in range(repeat):  # [wrapper-only] re-run the same example `repeat` times (cache OFF) to measure jitter
            try:  # [wrapper-only] a failed repeat is logged and skipped, not fatal
                pred = generator(  # [wrapper-only] run the full pipeline (== forward()) for this repeat
                    # [wrapper-only] unpack the Example fields into forward()'s signature (see forward():342+)
                    equation_json=ex.equation_json, const_values=ex.const_values,
                    weakspot_context=ex.weakspot_context, instance_names=ex.instance_names,
                    fc_name=getattr(ex, "fc_name", ""), weakspot_id=getattr(ex, "weakspot_id", ""),  # [wrapper-only] weakspot identity for the Required-I/O KB output override (forward **kwargs; not rendered)
                    raster_time=ex.raster_time,
                    weakspot_operation=getattr(ex, "weakspot_operation", ""),
                    example_explanation_output=getattr(ex, "example_explanation_output", ""),
                    dll_path=getattr(ex, "dll_path", ""),
                    signalsheet_data=getattr(ex, "signalsheet_data", ""),
                    model_interface_data=getattr(ex, "model_interface_data", ""),
                    relations_json=getattr(ex, "relations_json", "{}"),
                    equation_results_json=getattr(ex, "equation_results_json", "[]"),
                    op_map_json=getattr(ex, "op_map_json", "{}"),
                    weakspot_op_key=getattr(ex, "weakspot_op_key", ""),
                )
                s = _compute_all_components(ex, pred)  # [wrapper-only] score M1..M8 for this repeat (mbd_metric.py:1783)
                s["total"] = sum(w * s[k] for k, w in WEIGHTS)  # [wrapper-only] weighted composite for this repeat
                for m in metrics:  # [wrapper-only] record each metric's sample for this repeat
                    series[m].append(s[m])  # [wrapper-only] append the sample
            except Exception as e:  # [wrapper-only] this repeat failed — log and skip it
                _log.error("[%d] %s repeat %d ERROR: %s", i + 1, fc_tag, r + 1, e)  # [wrapper-only] log the failed repeat
        if not series["total"]:  # [wrapper-only] every repeat failed → nothing to summarize, skip example
            continue  # [wrapper-only] move to next example
        line = f"[{i+1}] {fc_tag} (n={len(series['total'])}): "  # [wrapper-only] row prefix with successful-sample count
        parts = []  # [wrapper-only] collect per-metric "mean±std" cells
        for m in metrics:  # [wrapper-only] compute mean/std for each metric's samples
            vals = series[m]  # [wrapper-only] this metric's samples across repeats
            mean = _stat.mean(vals)  # [wrapper-only] mean across repeats
            std = _stat.pstdev(vals) if len(vals) > 1 else 0.0  # [wrapper-only] population std (0 if a single sample)
            agg_std[m].append(std)  # [wrapper-only] feed the aggregate noise profile
            parts.append(f"{m.upper()}={mean:.2f}±{std:.2f}")  # [wrapper-only] format the metric cell
        msg = line + "  ".join(parts)  # [wrapper-only] assemble the full per-example noise row
        print("  " + msg)  # [wrapper-only] show the row
        _log.info(msg)  # [wrapper-only] log the row

    print("\n  AGGREGATE NOISE PROFILE (avg std-dev across examples; higher = noisier):")  # [wrapper-only] aggregate-profile header
    # [wrapper-only] per-metric label (what KIND of check M1..M8 is) for the printed profile
    typ = {"total": "composite", "m1": "rule", "m2": "rule", "m3": "rule",
           "m4": "LLM-judge", "m7": "DLL", "m8": "DLL"}
    for m in metrics:  # [wrapper-only] one aggregate row per metric
        if agg_std[m]:  # [wrapper-only] only report metrics that had at least one example std sample
            avg = sum(agg_std[m]) / len(agg_std[m])  # [wrapper-only] mean std-dev across all examples for this metric
            row = f"    {m.upper():5s} [{typ[m]:10s}] avg_std={avg:.3f}"  # [wrapper-only] format the aggregate row
            print(row)  # [wrapper-only] show the aggregate row
            _log.info(row.strip())  # [wrapper-only] log the aggregate row
    print("=" * 70)  # [wrapper-only] trailing rule


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
# [wrapper-only] main() ≙ run.py main()/run_pipeline ONLY at the process-entry level (loose):
#   model config (reconfigure_lm) ≙ run.py LM setup; load_mbd_dataset() ≙ run.py inputs+Phase1/2
#   (utils/mbd_data.py); dspy.Evaluate(metric=mbd_testcase_metric) + GEPA are wrapper-ADDED (run.py
#   has no scoring/optimizer). The per-weakspot pipeline run.py runs inline is here inside forward().
def main():   # [wrapper-only] orchestrates config → data → baseline eval → GEPA → save (GEPA/eval harness)
    parser = build_parser()  # [wrapper-only] CLI arg parser (GEPA/eval harness; run.py reads run_config.json instead)
    args = parser.parse_args()  # [wrapper-only] parse the harness CLI flags

    # ── Timestamp ─────────────────────────────────────────────────
    # run.py:938 run_timestamp = datetime.now().strftime(...); here reusable via --run-timestamp so a subprocess shares it
    current_time = args.run_timestamp if args.run_timestamp else datetime.now().strftime("%Y%m%d%H%M%S")  # run.py:938 run timestamp (wrapper allows override)
    pipeline_started_at = datetime.now().isoformat(timespec="seconds")  # [wrapper-only] wall-clock start for the run summary
    pipeline_t0 = time.time()  # [wrapper-only] monotonic start for elapsed-time reporting

    # ── Build log/cache name prefix ────────────────────────────────
    # [wrapper-only] all tags below build a unique log/cache filename prefix (no run.py analog — run.py logs to a fixed path)
    model_tag = (args.model_name or os.getenv("MODEL_NAME", "Unknown")).replace("/", "_")  # [wrapper-only] model id as a filesystem-safe tag
    cot_tag = "NoCot" if args.no_cot else "CoT"  # [wrapper-only] tag whether predictors use ChainOfThought
    gepa_auto_tag = args.gepa_auto.capitalize()  # [wrapper-only] GEPA auto-budget tag (light/medium/heavy)
    if args.eval_only:  # [wrapper-only] eval-only run (no GEPA optimization)
        strategy_tag = "EvalOnly"  # [wrapper-only]
    elif args.eval_test:  # [wrapper-only] eval on the held-out test split
        strategy_tag = "EvalTest"  # [wrapper-only]
    else:  # [wrapper-only] default: full GEPA optimization run
        strategy_tag = f"GEPA_MBD_Full_{gepa_auto_tag}"  # [wrapper-only]
    log_prefix = f"{model_tag}_{strategy_tag}_{cot_tag}"  # [wrapper-only] assembled filename prefix

    # ── Log directory & TeeLogger ──────────────────────────────────
    # [wrapper-only] the whole tee block mirrors stdout/stderr to a run log (no run.py analog — run.py uses only the logging module)
    log_dir = "log"  # [wrapper-only] output log directory
    os.makedirs(log_dir, exist_ok=True)  # [wrapper-only] ensure it exists
    log_path = os.path.join(log_dir, f"{log_prefix}_mbd_full_run_{current_time}.log")  # [wrapper-only] per-run log file path
    _log_file = open(log_path, "a", encoding="utf-8")  # [wrapper-only] append-mode log file handle (shared by tee + truncation inspector)
    _tee_stdout = _TeeLogger(sys.stdout, _log_file)  # [wrapper-only] tee stdout → console + log file
    _tee_stderr = _TeeLogger(sys.stderr, _log_file)  # [wrapper-only] tee stderr → console + log file
    sys.stdout = _tee_stdout  # [wrapper-only] install the stdout tee
    sys.stderr = _tee_stderr  # [wrapper-only] install the stderr tee

    # ── Python logging setup ───────────────────────────────────────
    # [wrapper-only] the whole logging setup is harness scaffolding (run.py:110 just does logging.basicConfig once)
    _log_fmt = logging.Formatter(  # [wrapper-only] shared log line format
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _file_handler = logging.FileHandler(log_path, encoding="utf-8")  # [wrapper-only] file handler → the run log
    _file_handler.setFormatter(_log_fmt)  # [wrapper-only] apply the shared format
    _console_handler = logging.StreamHandler(_tee_stderr._original)  # [wrapper-only] console handler → the ORIGINAL stderr (avoids tee recursion)
    _console_handler.setFormatter(_log_fmt)  # [wrapper-only] apply the shared format
    logging.basicConfig(level=logging.WARNING, handlers=[_file_handler, _console_handler])  # [wrapper-only] root logger at WARNING with both handlers

    # NOTE: the metric module's logger is named "mbd_metric" (evaluation/mbd_metric.py),
    # NOT "mbd.metric" — elevating the dotted name was a no-op that hid the full per-component
    # M1..M8 breakdown ("SUMMARY OF ALL METRIC SCORES", per-check PASS/FAIL) from the run log.
    for _ns in ("extract_mbd_full", "mbd.data", "mbd_metric"):  # [wrapper-only] wrapper/metric loggers whose DEBUG output we want in the run log
        logging.getLogger(_ns).setLevel(logging.DEBUG)  # [wrapper-only] elevate each to DEBUG (root stays WARNING for third-party noise)

    # Suppress MIDAS function_caller "not available in the DLL" noise. The original MIDAS pipeline
    # (pipeline/function_caller.py:475,753) logs an ERROR every time an ASCET node-type
    # (Variable/Operation/Local/SystemConstant) isn't a DLL function and it falls back to Python —
    # benign, expected, and emitted thousands of times per run (it flooded a single GEPA run's log
    # with ~12k lines). DROP just those records via a logger filter (return False) so the log stays
    # readable; MIDAS code stays untouched. All other function_caller records pass through.
    class _DLLNoiseFilter(logging.Filter):  # [wrapper-only] filter for pipeline/function_caller.py:475/749/753 fallback ERRORs
        def filter(self, record: logging.LogRecord) -> bool:  # [wrapper-only] logging.Filter hook (True=keep, False=drop)
            if (record.name == "pipeline.function_caller"  # [wrapper-only] only touch the MIDAS function_caller logger
                    and "not available in the DLL" in record.getMessage()):  # pipeline/function_caller.py:475/749/753 benign fallback message
                return False  # drop benign DLL-fallback noise
            return True  # [wrapper-only] keep every other function_caller record
    logging.getLogger("pipeline.function_caller").addFilter(_DLLNoiseFilter())  # [wrapper-only] install the filter on MIDAS's function_caller logger

    # Suppress dspy's benign "Failed to use structured output format, falling back to JSON mode"
    # warning. On this Azure deployment the json_schema (structured-output) response_format is
    # rejected, so dspy's JSONAdapter falls back to plain json_object mode (which works). The
    # fallback is harmless but was logged on most LM calls (thousands of lines). Drop just that
    # record; every other adapter log passes through.
    class _StructuredOutputNoiseFilter(logging.Filter):  # [wrapper-only] filter for dspy JSONAdapter structured-output fallback warning
        def filter(self, record: logging.LogRecord) -> bool:  # [wrapper-only] logging.Filter hook
            return "Failed to use structured output format" not in record.getMessage()  # [wrapper-only] drop only the fallback warning, keep the rest
    logging.getLogger("dspy.adapters.json_adapter").addFilter(_StructuredOutputNoiseFilter())  # [wrapper-only] install on dspy's json_adapter logger

    # [wrapper-only] the whole truncation inspector is a debug aid: on a dspy max_tokens WARNING it dumps the last
    # LM interaction into the log so a truncated Prompt-2/4 reply can be diagnosed. No run.py analog.
    class _TruncationInspectorHandler(logging.Handler):  # [wrapper-only] custom log handler that inspects on truncation
        def emit(self, record: logging.LogRecord) -> None:  # [wrapper-only] called for every dspy.clients.lm record
            if record.levelno != logging.WARNING:  # [wrapper-only] only react to WARNING-level records
                return  # [wrapper-only] ignore other levels
            if "LM response was truncated" not in record.getMessage():  # [wrapper-only] only the truncation warning
                return  # [wrapper-only] ignore unrelated warnings
            try:  # [wrapper-only] inspect_history can fail if no history exists
                history_text = dspy.inspect_history(n=1)  # [wrapper-only] fetch the last LM prompt/response pair
            except Exception as exc:  # [wrapper-only] guard against inspect_history errors
                history_text = f"<could not retrieve history: {exc}>"  # [wrapper-only] fall back to an error note
            _log_file.write(  # [wrapper-only] append the delimited history dump to the run log
                "\n" + "=" * 80 + "\n"
                "[TRUNCATION INSPECTOR] Last LM interaction that hit max_tokens:\n"
                + "=" * 80 + "\n"
                + (history_text or "<empty history>")
                + "\n" + "=" * 80 + "\n\n"
            )
            _log_file.flush()  # [wrapper-only] flush so the dump survives a crash

    _trunc_inspector = _TruncationInspectorHandler()  # [wrapper-only] instantiate the inspector handler
    _trunc_inspector.setLevel(logging.WARNING)  # [wrapper-only] only WARNING+ reach emit()
    logging.getLogger("dspy.clients.lm").addHandler(_trunc_inspector)  # [wrapper-only] attach to dspy's LM client logger

    _run_logger = logging.getLogger("extract_mbd_full")  # [wrapper-only] this module's named logger for run-level messages
    _run_logger.info("MBD Full pipeline run started — log file: %s", log_path)  # [wrapper-only] trace: run started

    # ── Model configuration ────────────────────────────────────────
    model_to_use = args.model_name or os.getenv("MODEL_NAME", "azure/gpt-4.1-mini")  # (~) run.py uses cfg.AZURE_OPENAI_DEPLOYMENT; here a CLI/env-selectable litellm model id
    cache_enabled = not args.no_cache  # [wrapper-only] DSPy response-cache toggle (no run.py analog; --no-cache exposes run-to-run variance)

    # reconfigure_lm (config.py:391-407): builds a dspy.LM for `model_to_use` via create_lm()
    # — passing cache=, plus the default temperature/max_tokens — and installs it as the GLOBAL
    # DSPy LM through dspy.configure(lm=...). Every predictor (Prompt 1/2/4 + retry) then routes
    # its calls through this single LM. This is the wrapper-layer counterpart of run.py's LM setup
    # (run.py calls llm_client.call_llm, reading the same Azure config); here DSPy owns the client
    # so GEPA can drive and trace each predictor. cache=cache_enabled toggles DSPy's response cache
    # (disabled via --no-cache to expose true run-to-run variability of the pipeline).
    reconfigure_lm(model_to_use, cache=cache_enabled)  # config.py:449 build+install the global dspy.LM (wrapper counterpart of run.py's call_llm transport)

    # [wrapper-only] Cost guard: track tokens/cost across ALL models (student + M4 judge + reflection) via a
    # litellm callback, and force-stop the run once cumulative spend exceeds --max-cost-usd. Install EARLY so
    # the baseline + every GEPA rollout is counted. No cap given => still tracks (for the run-summary), no stop.
    if args.max_cost_usd is not None:
        from utils.cost_guard import install_cost_guard  # [wrapper-only] litellm success-callback cost tracker
        install_cost_guard(args.max_cost_usd)  # [wrapper-only] logs a running $ line every $0.50; os._exit(2) on exceed
        _run_logger.info("Cost guard installed: hard cap $%.2f (force-stop on exceed)", args.max_cost_usd)  # [wrapper-only]

    # ── Adapter: RawCompletionAdapter (run.py-faithful transport) ─────────────────
    # The DLL solver (Prompt 4) is DESIGNED to reply with "explanation + a ```json [...]``` fenced
    # array", which run.py parses via extract_json_from_text_dll (pipeline/equation_solver.py) — a
    # FENCE-ONLY regex search. run.py works because it calls the LLM in PLAIN TEXT (no response_format),
    # so the fence is emitted. dspy's JSONAdapter forces a structured JSON object -> the fence is
    # destroyed -> ~81% of solve attempts return no parseable JSON (measured by the solver-JSON-health
    # counter; M7/M8 then pin at the fn-call floor and retries are wasted). The DEFAULT ChatAdapter is
    # no better: on the marker-less solver reply its parse raises and ChatAdapter.__call__ re-routes to
    # JSONAdapter (chat_adapter.py) — same mangling.
    # RawCompletionAdapter reuses ChatAdapter's formatting but (1) KILLS the JSONAdapter fallback and
    # (2) hands the RAW completion to the output field when there are no markers — exactly run.py's
    # plain-text transport, so the ```json fence survives to the regex. Prompt-1 free text and
    # Prompt-2's fence-free [{...}] regex are unaffected; the verbatim prompt seeds are untouched.
    from signatures.raw_completion_adapter import RawCompletionAdapter  # [wrapper-only] plain-text adapter that preserves run.py's ```json fence transport
    dspy.configure(adapter=RawCompletionAdapter())  # [wrapper-only] install it globally (replicates run.py:561/624 call_llm PLAIN-TEXT solve replies)

    # ── Configuration banner ───────────────────────────────────────
    # [wrapper-only] the whole banner (all print()s) echoes the resolved harness config to console; no run.py analog
    print(f"\n{'='*70}")
    print(" MBD-UTA Full Pipeline Configuration:")
    print(f"  Model:           {model_to_use}")
    print(f"  Module:          {'Predict (no CoT)' if args.no_cot else 'ChainOfThought'}")
    print(f"  Cache:           {cache_enabled}")
    print(f"  Optimizer:       {args.optimizer} (auto={args.gepa_auto})")
    print(f"  Eval-only:       {args.eval_only}")
    print(f"  Eval-test:       {args.eval_test}")
    print(f"  Inject KB guides:{args.inject_kb_guidelines}")
    print(f"  Max retries:     {args.max_retries}")
    print(f"  Load extractor:  {args.load_extractor or '(none)'}")
    print(f"  Num threads:     {args.num_threads}")
    print(f"  Dataset:         {'dev-subset' if args.dev_subset else 'all-tasks (run_config.json)'}")
    print(f"  Log file:        {log_path}")
    print(f"{'='*70}\n")
    _run_logger.info(  # [wrapper-only] same config, logged as one structured line
        "Config — model=%s cache=%s cot=%s optimizer=%s gepa_auto=%s num_threads=%d",
        model_to_use, cache_enabled, not args.no_cot, args.optimizer, args.gepa_auto, args.num_threads,
    )

    # ── Data loading ───────────────────────────────────────────────
    # Q2: all-tasks (run_config.json, ~10 valid weakspots) is now the DEFAULT dataset — it maximizes
    # the usable set (the old hardcoded dev subset gave 9; --dev-subset restores it). --use-all-tasks
    # is retained as a no-op alias (already the default). fc_entries=None → load_mbd_dataset uses the
    # dev subset (_default_fc_entries).
    print("Loading MBD dataset...")  # [wrapper-only] trace
    _use_all = not args.dev_subset            # all-tasks unless the small dev subset is explicitly requested
    _run_logger.info("Loading MBD dataset (all_tasks=%s, dev_subset=%s)", _use_all, args.dev_subset)  # [wrapper-only] trace
    fc_entries = None if args.dev_subset else all_fc_entries()  # run.py:230 raw.get("fc_entries"); all_fc_entries() reads the SAME run_config.json (None → dev subset)
    dataset = load_mbd_dataset(fc_entries=fc_entries)  # run.py Phase 1/2 per FC: builds one dspy.Example per weakspot (utils/mbd_data.py:157 — filter_equation/relations precomputed there)

    if not dataset:  # [wrapper-only] no examples loaded → cannot run
        msg = "ERROR: No examples loaded. Check data directory and FC entries."  # [wrapper-only] error text
        print(msg)  # [wrapper-only] trace
        _run_logger.error(msg)  # [wrapper-only] trace
        _restore_streams(sys, _tee_stdout, _tee_stderr, _log_file)  # [wrapper-only] undo the stdout/stderr tee before exit
        sys.exit(1)  # [wrapper-only] abort the run

    # Re-wrap examples with full 12-field with_inputs() for MBDFullPipelineModule
    dataset_full = prepare_full_examples(dataset)  # extract_mbd_full.py:235 re-declare all 12 forward() inputs via with_inputs (wrapper: run.py passes data positionally)
    # GEPA split: 60% train / 40% val / 0% test (global split — see stratified_split docstring).
    train_set, val_set, test_set = stratified_split_mbd_dataset(  # [wrapper-only] GEPA train/val/test split (utils/mbd_data.py:492 — run.py runs every FC once, no split)
        dataset_full, train_ratio=0.6, val_ratio=0.4, test_ratio=0.0
    )
    print(f"Split: train={len(train_set)} | val={len(val_set)} | test={len(test_set)}")  # [wrapper-only] trace the split sizes
    _run_logger.info(  # [wrapper-only] log the split sizes
        "Dataset split: total=%d train=%d val=%d test=%d",
        len(dataset_full), len(train_set), len(val_set), len(test_set),
    )

    # ── DLL → single-thread (match run.py) ─────────────────────────
    # run.py runs the pipeline SINGLE-THREADED with one FunctionCaller reused sequentially. The
    # wrapper caches one FunctionCaller per dll_path and shares it across dspy.Evaluate/GEPA threads;
    # its per-call lock does NOT protect M8's stateful init→advance→call sequence, so concurrent
    # same-FC examples could corrupt each other's DLL state. When a DLL is present, force num_threads=1
    # (exactly matching run.py). When no DLL is usable there is no shared-DLL risk, so keep the request.
    _dll_present = any(  # [wrapper-only] does ANY example carry a resolvable 32-bit DLL path?
        getattr(ex, "dll_path", "") and os.path.exists(getattr(ex, "dll_path", ""))  # [wrapper-only] dll_path set AND the file exists on disk
        for ex in dataset_full  # [wrapper-only] scan every example
    )
    effective_num_threads = 1 if _dll_present else args.num_threads  # [wrapper-only] force 1 thread when a DLL is shared (matches run.py single-threading)
    if _dll_present and args.num_threads != 1:  # [wrapper-only] only warn when we actually override a >1 request
        print(f"DLL present -> forcing num_threads=1 (was {args.num_threads}); run.py is single-threaded "  # [wrapper-only] trace the override reason
              "and the shared 32-bit DLL subprocess is not safe for M8's stateful sequence under parallelism.")
        _run_logger.warning(  # [wrapper-only] log the override
            "DLL present -> overriding num_threads %d -> 1 (single-threaded DLL eval, matches run.py)",
            args.num_threads,
        )

    if len(test_set) == 0:  # [wrapper-only] empty test split → nothing to evaluate on test
        print("WARNING: test_set is empty — test evaluation will be skipped.")  # [wrapper-only] trace
        _run_logger.warning("test_set is empty — test evaluation will be skipped")  # [wrapper-only] trace

    # ── Create module ──────────────────────────────────────────────
    # Set the env var BEFORE constructing the module (its __init__ reads it) and BEFORE GEPA compile,
    # so the optimizer subprocess reconstructs the module with injection ON too.
    if args.inject_kb_guidelines:  # [wrapper-only] opt-in KB-guideline injection (forward():387 gate; NOT in run.py)
        os.environ["MBD_INJECT_KB_GUIDELINES"] = "1"  # [wrapper-only] env so the optimizer subprocess rebuilds the module with injection ON
    if args.max_retries and int(args.max_retries) != 1:  # [wrapper-only] only export a non-default DLL retry budget (1 == run.py's `if retry_num < 1`)
        os.environ["MBD_MAX_RETRIES"] = str(int(args.max_retries))     # so any subprocess reconstruction inherits it
    if getattr(args, "required_io_path", ""):  # [wrapper-only] point the KB (scoring + prompt override) at a custom file
        os.environ["MBD_REQUIRED_IO_PATH"] = args.required_io_path      # env BEFORE loader use + so the subprocess inherits it
        from utils.required_io import set_kb_path as _set_kb_path       # [wrapper-only] reload the in-process KB now (loader cached at import)
        _n_io = _set_kb_path(args.required_io_path)                     # [wrapper-only] rebind cache for scoring + prompt
        _run_logger.info("Required-I/O KB re-pointed to %s (%d entries)", args.required_io_path, _n_io)  # [wrapper-only] trace
    if getattr(args, "required_io_outputs", False):  # [wrapper-only] opt-in Prompt-2 output override (forward():425 gate; NOT in run.py)
        os.environ["MBD_REQUIRED_IO_IN_PROMPT"] = "1"  # [wrapper-only] env so the optimizer subprocess rebuilds the module with the override ON
    generator = MBDFullPipelineModule(use_cot=not args.no_cot)  # [wrapper-only] the DSPy module whose forward() runs run.py's per-weakspot pipeline
    generator.inject_kb_guidelines = bool(args.inject_kb_guidelines)   # explicit in-process override
    generator.max_retries = int(args.max_retries)                      # explicit in-process override (DLL retry budget)
    generator.required_io_in_prompt = bool(getattr(args, "required_io_outputs", False))  # [wrapper-only] explicit in-process override (Prompt-2 output steering)
    _run_logger.info("Created MBDFullPipelineModule(use_cot=%s, inject_kb_guidelines=%s, max_retries=%s, required_io_in_prompt=%s)",  # [wrapper-only] trace the module config
                     not args.no_cot, generator.inject_kb_guidelines, generator.max_retries, generator.required_io_in_prompt)

    # ── Load cached extractor (if requested) ───────────────────────
    if args.load_extractor:  # [wrapper-only] optionally warm-start from a saved GEPA-optimized state (no run.py analog)
        extractor_path = args.load_extractor  # [wrapper-only] the requested state file
        if os.path.exists(extractor_path):  # [wrapper-only] only load if the path exists
            generator.load(extractor_path)  # [wrapper-only] restore the module's optimized prompts/demos
            print(f"Loaded extractor from: {extractor_path}")  # [wrapper-only] trace
            _run_logger.info("Loaded extractor from: %s", extractor_path)  # [wrapper-only] trace
        else:  # [wrapper-only] path missing → warn and keep the seed module
            print(f"WARNING: --load-extractor path not found: {extractor_path}")  # [wrapper-only] trace
            _run_logger.warning("load-extractor path not found: %s", extractor_path)  # [wrapper-only] trace

    # ── Verify seed signature ──────────────────────────────────────
    _print_signature_banner(generator, _run_logger)  # [wrapper-only] print the seed predictor instructions so GEPA's starting point is auditable

    # ── Save seed state ────────────────────────────────────────────
    seed_state_path = os.path.join(log_dir, f"{log_prefix}_mbd_full_run_{current_time}_seed_state.json")  # [wrapper-only] path for the pre-optimization module snapshot
    with open(seed_state_path, "w", encoding="utf-8") as _fp:  # [wrapper-only] write the seed snapshot (run.py never persists module state — it has no optimization)
        json.dump(generator.dump_state(), _fp, indent=2, ensure_ascii=False)  # [wrapper-only] dump seed prompts/demos as JSON so a run can be reproduced
    print(f"Seed state saved -> {seed_state_path}")  # [wrapper-only] trace
    _run_logger.info("Seed state saved: %s", seed_state_path)  # [wrapper-only] trace

    # ── Evaluation setup ───────────────────────────────────────────
    # --eval-only evaluates the VAL set (the SAME split GEPA uses as its baseline), so an
    # eval-only baseline is directly comparable to a GEPA run's baseline / optimized val score
    # (e.g. old-KB vs new-KB on the identical val examples). Pass --eval-all-data to instead score
    # the ENTIRE dataset (the previous comprehensive-baseline behavior). GEPA mode always uses val_set.
    baseline_devset = dataset_full if (args.eval_only and args.eval_all_data) else val_set  # [wrapper-only] choose which split to score (run.py has no scoring split — it just processes every weakspot)
    _devset_label = "ALL data" if (args.eval_only and args.eval_all_data) else "val set"  # [wrapper-only] human label for the chosen split
    # [wrapper-only] dspy.Evaluate batches forward() over the devset; each forward() == run.py:727 run_pipeline()'s
    # per-weakspot body (run once per FunctionCaller in run.py's loop) — here run in parallel for a mean score.
    evaluate = dspy.Evaluate(
        devset=baseline_devset,                            # [wrapper-only] the split to iterate forward() over
        metric=mbd_testcase_metric,                        # [wrapper-only] composite M1..M8 scorer (run.py never scores — it emits output JSON)
        num_threads=effective_num_threads,                 # [wrapper-only] parallel example execution (run.py is a sequential per-FC loop)
        display_progress=True,                             # [wrapper-only] progress bar
        display_table=5,                                   # [wrapper-only] show first 5 rows of results
    )

    # ── Baseline evaluation ────────────────────────────────────────
    print("\n" + "=" * 70)                                # [wrapper-only] banner
    print(f"BASELINE EVALUATION (Human Prompt Instructions — {_devset_label})")  # [wrapper-only] banner
    print("=" * 70)                                        # [wrapper-only] banner
    _run_logger.info("Starting baseline evaluation on %s (%d examples)", _devset_label, len(baseline_devset))  # [wrapper-only] trace
    t0 = time.time()                                       # [wrapper-only] start timer for baseline eval duration
    # NOTE: do NOT wrap dspy.Evaluate in track_usage() — its ParallelExecutor worker
    # deepcopies the UsageTracker, which holds a threading.Lock that cannot be deepcopied,
    # so EVERY example fails silently and the eval scores 0.0. Read usage from LM history.
    _hist0 = _lm_history_len()                             # [wrapper-only] snapshot LM history length so token usage can be diffed post-eval
    if args.eval_only:                                     # [wrapper-only] eval-only path scores via a single-pass breakdown (no dspy.Evaluate double-run)
        # SINGLE PASS: run the pipeline ONCE per example; the baseline is the mean of the SAME
        # per-example composites shown in the breakdown below. Previously --eval-only ran the
        # pipeline TWICE — dspy.Evaluate for the headline score + _print_per_example_scores for the
        # breakdown — which, with caching OFF (Azure), produced two different LLM generations and a
        # headline that disagreed with the breakdown. One pass halves the cost AND makes the headline
        # byte-consistent with the breakdown (important for repeatability experiments).
        print(f"\n--eval-only: evaluating {len(baseline_devset)} weakspots ({_devset_label}) — single pass:")  # [wrapper-only] trace
        _run_logger.info("eval-only mode (single pass) — per-example scores (n=%d)", len(baseline_devset))  # [wrapper-only] trace
        _per_totals = _print_per_example_scores(generator, baseline_devset, logger=_run_logger)  # [wrapper-only] run forward() once per weakspot + print each composite; returns per-example totals
        baseline_score = 100.0 * sum(_per_totals) / len(_per_totals) if _per_totals else 0.0  # [wrapper-only] baseline = mean composite (percent); guard empty devset
    elif args.eval_test:                                   # [wrapper-only] eval-test still needs a val baseline pass
        baseline_result = evaluate(generator, devset=baseline_devset)   # eval-test: dspy.Evaluate
        baseline_score = baseline_result.score             # [wrapper-only] mean composite score from dspy.Evaluate
    else:                                                  # [wrapper-only] pure GEPA path: DON'T run a redundant baseline pass —
        # GEPA measures its own iteration-0 seed baseline under identical conditions; adopt that after compile().
        print("(GEPA run: baseline deferred to GEPA's own iteration-0 seed eval — single baseline, no redundant wrapper pass.)")  # [wrapper-only]
        _run_logger.info("Baseline deferred to GEPA internal iteration-0 eval (no wrapper baseline pass)")  # [wrapper-only]
        baseline_score = None                              # [wrapper-only] filled from the runner (GEPA) after optimization
    baseline_duration = time.time() - t0                   # [wrapper-only] elapsed wall time for the baseline eval
    baseline_token_usage = _token_usage_from_history(_hist0)  # [wrapper-only] diff LM history since _hist0 → prompt/completion token totals
    if baseline_score is not None:                         # [wrapper-only] (GEPA defers its baseline — nothing to print yet)
        print(f"Baseline score: {baseline_score:.1f}%  ({baseline_duration:.1f}s, {len(baseline_devset)} examples)")  # [wrapper-only] trace
        _log_token_usage("Baseline", baseline_token_usage, _run_logger)  # [wrapper-only] trace token usage
        _run_logger.info(  # [wrapper-only] trace: baseline result summary
            "Baseline evaluation complete: score=%.2f%% duration=%.1fs n=%d",
            baseline_score, baseline_duration, len(baseline_devset),
        )

    # ── Solver JSON health (Prompt-4 DLL solving) ──────────────────
    # Fraction of DLL solve attempts whose LLM output had no parseable JSON list (the "No valid JSON
    # list found" path). HIGH here => the solver is starved: equations rarely reach LHS==RHS, retries
    # are wasted, and M7/M8 stay low. Use this on a quick --eval-only before spending GEPA budget.
    _sj_att = getattr(generator, "_solver_attempts", 0)    # [wrapper-only] total DLL solve attempts counted during forward() calls (diagnostic counter, not in run.py)
    _sj_fail = getattr(generator, "_solver_json_fails", 0)  # [wrapper-only] attempts whose LM output had no parseable JSON list ("No valid JSON list found" path)
    if _sj_att:                                            # [wrapper-only] only meaningful after an eval pass ran (GEPA defers its baseline → counters still 0 here)
        _sj_rate = 100.0 * _sj_fail / _sj_att              # [wrapper-only] fail percentage
        print(f"Solver JSON health: {_sj_fail}/{_sj_att} solve attempts returned no parseable JSON "  # [wrapper-only] trace
              f"({_sj_rate:.1f}% fail)")                    # [wrapper-only] trace (continuation)
        _run_logger.info("Solver JSON health: %d/%d attempts failed to parse (%.1f%%)", _sj_fail, _sj_att, _sj_rate)  # [wrapper-only] trace

    # ── eval-only mode (breakdown already printed above in the single pass) ──
    if args.eval_only:                                     # [wrapper-only] eval-only stops here — no optimization stage exists (run.py likewise never optimizes)
        if args.repeat and args.repeat > 1:                # [wrapper-only] optional: repeat the eval to quantify LM/metric nondeterminism
            _measure_metric_noise(generator, baseline_devset, args.repeat, logger=_run_logger)  # [wrapper-only] run N passes and report per-metric variance
        # [wrapper-only] persist a JSON run summary (baseline == best; no optimized state) — run.py has no such summary
        _save_run_summary(
            log_dir=log_dir,
            log_prefix=log_prefix,
            current_time=current_time,
            pipeline_started_at=pipeline_started_at,
            pipeline_t0=pipeline_t0,
            model_to_use=model_to_use,
            args=args,
            baseline_score=baseline_score,
            best_score=baseline_score,
            best_strategy="Baseline",
            saved_model_path=None,
            seed_state_path=seed_state_path,
            optimized_state_path=None,
            run_logger=_run_logger,
            baseline_token_usage=baseline_token_usage,
        )
        _restore_streams(sys, _tee_stdout, _tee_stderr, _log_file)  # [wrapper-only] undo the stdout/stderr tee before exiting (log-file teeing is wrapper infra)
        return                                             # [wrapper-only] eval-only run is complete — no GEPA compile follows

    # ── eval-test mode ─────────────────────────────────────────────
    if args.eval_test:                                     # [wrapper-only] eval-test path scores the held-out test split only, then exits (no optimization)
        print("\n" + "=" * 70)                            # [wrapper-only] banner
        print("TEST SET EVALUATION (held-out)")            # [wrapper-only] banner
        print("=" * 70)                                    # [wrapper-only] banner
        if len(test_set) == 0:                             # [wrapper-only] guard: cannot score an empty held-out split
            print("ERROR: test_set is empty — cannot evaluate. Re-run with more tasks (--use-all-tasks).")  # [wrapper-only] trace
            _run_logger.error("test_set is empty — cannot run --eval-test")  # [wrapper-only] trace
            _restore_streams(sys, _tee_stdout, _tee_stderr, _log_file)  # [wrapper-only] undo stdout/stderr tee before exiting
            sys.exit(1)                                    # [wrapper-only] hard-fail: no data to evaluate
        _run_logger.info("Starting test evaluation (%d examples)", len(test_set))  # [wrapper-only] trace
        test_result = evaluate(generator, devset=test_set)  # [wrapper-only] batch forward() over the test split (== run.py run_pipeline per weakspot, here scored)
        test_score = test_result.score                     # [wrapper-only] mean composite over test split
        print(f"Test score: {test_score:.1f}%")            # [wrapper-only] trace
        print(f"Generalization gap vs baseline: {abs(baseline_score - test_score):.1f}%")  # [wrapper-only] trace: val-vs-test gap
        _run_logger.info("Test evaluation: score=%.2f%%", test_score)  # [wrapper-only] trace
        _restore_streams(sys, _tee_stdout, _tee_stderr, _log_file)  # [wrapper-only] undo stdout/stderr tee before exiting
        return                                             # [wrapper-only] eval-test run is complete — no GEPA compile follows

    # ── GEPA optimization ──────────────────────────────────────────
    # [wrapper-only] ENTIRE GEPA stage below has NO analog in run.py: run.py runs the fixed human
    # prompts once per weakspot and emits output JSON; it never evolves prompts. GEPA reflectively
    # rewrites the 4 predictors' instructions against the train split to maximize the val composite.
    print("\n" + "=" * 70)                                 # [wrapper-only] banner
    print(f"GEPA OPTIMIZATION (auto={args.gepa_auto}, num_threads={effective_num_threads})")  # [wrapper-only] banner
    print("=" * 70)                                        # [wrapper-only] banner
    # [wrapper-only] trace GEPA config (budget preset, threads, split sizes)
    _run_logger.info(
        "Starting GEPA optimization: auto=%s num_threads=%d train=%d val=%d",
        args.gepa_auto, effective_num_threads, len(train_set), len(val_set),
    )

    cache_dir = ".cache"                                   # [wrapper-only] where GEPA logs + the optimized model .json are written
    os.makedirs(cache_dir, exist_ok=True)                  # [wrapper-only] ensure the cache dir exists
    gepa_log_dir = os.path.join(cache_dir, f"gepa_mbd_full_{current_time}")  # [wrapper-only] per-run GEPA log/checkpoint dir

    # reflection_minibatch_size = full train set: GEPA's RoundRobinReflectionComponentSelector
    # picks ONE predictor per iteration (context→testcase→solver→retry→…). With the default
    # minibatch of 3, the solver's/retry's turn can land on a minibatch with no equation-solving
    # examples → empty traces → "No valid predictions found for any module" → wasted iteration.
    # Using the whole train set every reflection guarantees the minibatch always exercises every
    # predictor that any train example invokes (Prompt 2 always; solver/retry on the arithmetic
    # weakspots), so no turn is wasted and each predictor reflects on the full signal.
    _refl_mb = max(3, len(train_set))                      # [wrapper-only] reflection minibatch = whole train set (min 3) so every predictor's turn has examples
    # Budget: an explicit --gepa-max-metric-calls gives a "lighter than light" run
    # (light ≈ 1495 metric calls). When set, it overrides --gepa-auto.
    _budget_tag = (f"{args.gepa_max_metric_calls} calls"   # [wrapper-only] human-readable budget label (explicit call cap ...
                   if args.gepa_max_metric_calls else f"auto={args.gepa_auto}")  # [wrapper-only] ... else the auto preset name)
    # [wrapper-only] build the GEPA strategy config; create_gepa_strategy is defined in optimization/opt.py
    strategy = create_gepa_strategy(
        name=f"GEPA_MBD_Full_{args.gepa_auto.capitalize()}",  # [wrapper-only] strategy display name (tags the saved model + summary)
        description=f"GEPA ({_budget_tag}) — reflective prompt optimization for MBD-UTA full pipeline",  # [wrapper-only] free-text description
        auto=(None if args.gepa_max_metric_calls else args.gepa_auto),  # [wrapper-only] explicit call cap disables the auto preset
        num_threads=effective_num_threads,                 # [wrapper-only] parallelism for GEPA's internal evals
        track_stats=True,                                  # [wrapper-only] keep per-iteration stats for the summary
        use_merge=True,                                    # [wrapper-only] enable GEPA candidate merging
        log_dir=gepa_log_dir,                              # [wrapper-only] GEPA checkpoint/log dir chosen above
        reflection_minibatch_size=_refl_mb,                # [wrapper-only] full-train-set minibatch (see note above)
        max_metric_calls=args.gepa_max_metric_calls,  # None → auto preset is used
    )
    # [wrapper-only] trace the resolved strategy config
    _run_logger.info(
        "GEPA strategy: name=%s budget=%s reflection_minibatch_size=%d log_dir=%s",
        strategy.name, _budget_tag, _refl_mb, gepa_log_dir,
    )

    # [wrapper-only] wire up the optimization runner (optimization/opt.py) — the seed module + splits + metrics
    runner = OptimizationRunner(
        metric=mbd_testcase_metric,                        # [wrapper-only] scalar composite metric used to rank candidates
        student_model=generator,                           # [wrapper-only] the seed module GEPA evolves
        train_set=train_set,                               # [wrapper-only] examples GEPA reflects on
        val_set=val_set,                                   # [wrapper-only] examples GEPA scores candidates against
        evaluate_fn=evaluate,                              # [wrapper-only] reuse the dspy.Evaluate built above
        metric_with_feedback=mbd_metric_with_feedback,     # [wrapper-only] feedback-emitting metric GEPA uses for reflection text
        reflection_lm=reflectin_lm,                        # [wrapper-only] the LM that rewrites predictor instructions
    )

    optimized_generator = None                             # [wrapper-only] default: no optimized module yet (falls back to seed if nothing wins)
    best_score = baseline_score                            # [wrapper-only] default best == baseline until a candidate beats it
    try:                                                   # [wrapper-only] guard the whole compile so a crash still persists a failure summary
        optimized_generator, best_score = runner.run_optimization([strategy])  # [wrapper-only] run GEPA compile() → best module + its val score
    except Exception as exc:                               # [wrapper-only] any optimization failure → write an error summary and exit 1
        failed_at = datetime.now().isoformat(timespec="seconds")  # [wrapper-only] failure timestamp for the error summary
        # [wrapper-only] assemble the failure record (run.py has no run-summary/error-report concept)
        failure_summary = {
            "run_type": "extract_mbd_full",
            "status": "failed",
            "started_at": pipeline_started_at,
            "failed_at": failed_at,
            "total_duration_seconds": round(time.time() - pipeline_t0, 1),
            "model_name": model_to_use,
            "optimizer_requested": args.optimizer,
            "gepa_auto": args.gepa_auto,
            "baseline_score": baseline_score,
            "seed_state_path": seed_state_path,
            "log_path": log_path,
            "error": str(exc),
            "runner_summary": getattr(runner, "run_summary", {}),
        }
        _restore_streams(sys, _tee_stdout, _tee_stderr, _log_file)  # [wrapper-only] undo stdout/stderr tee before printing to the real console
        err_path = os.path.join(log_dir, f"{log_prefix}_mbd_full_run_{current_time}_error_summary.json")  # [wrapper-only] path for the error summary JSON
        with open(err_path, "w", encoding="utf-8") as _fp:  # [wrapper-only] write the failure record to disk
            json.dump(failure_summary, _fp, indent=2, ensure_ascii=False)  # [wrapper-only] serialize the failure summary
        print(f"\nOptimization FAILED. Error summary: {err_path}", file=sys.__stdout__)  # [wrapper-only] report to the real stdout (streams restored)
        print(f"Seed state preserved at: {seed_state_path}", file=sys.__stdout__)  # [wrapper-only] point the user at the recoverable seed state
        raise SystemExit(1) from exc                       # [wrapper-only] non-zero exit so callers/CI see the failure

    if optimized_generator is None:                        # [wrapper-only] compile returned nothing better than baseline
        print("No strategy beat the baseline — using unoptimized generator.")  # [wrapper-only] trace
        _run_logger.info("No strategy beat baseline — using unoptimized generator")  # [wrapper-only] trace
        optimized_generator = generator                    # [wrapper-only] fall back to the seed module so downstream save/eval still runs

    # [wrapper-only] Adopt GEPA's OWN iteration-0 seed eval as THE (single) baseline. No separate wrapper baseline
    # pass runs for GEPA, so `improvement` below is apples-to-apples — and exactly 0 when GEPA retained the seed.
    if baseline_score is None:                             # [wrapper-only] GEPA path deferred its baseline
        baseline_score = getattr(runner, "gepa_baseline_score", best_score)  # [wrapper-only] GEPA's iteration-0 seed score
    _seed_retained = getattr(runner, "gepa_seed_retained", False)  # [wrapper-only] True => GEPA found no candidate that beat the seed

    # ── Print optimized instruction ────────────────────────────────
    _print_signature_banner(optimized_generator, _run_logger, label="OPTIMIZED INSTRUCTION")  # [wrapper-only] show GEPA's rewritten predictor instructions for audit

    # ── Save optimized state ───────────────────────────────────────
    # [wrapper-only] path for the post-optimization module snapshot (run.py never persists module state)
    optimized_state_path = os.path.join(
        log_dir, f"{log_prefix}_mbd_full_run_{current_time}_optimized_state.json"
    )
    with open(optimized_state_path, "w", encoding="utf-8") as _fp:  # [wrapper-only] write the optimized snapshot
        json.dump(optimized_generator.dump_state(), _fp, indent=2, ensure_ascii=False)  # [wrapper-only] dump optimized prompts/demos as JSON
    print(f"Optimized state saved -> {optimized_state_path}")  # [wrapper-only] trace
    _run_logger.info("Optimized state saved: %s", optimized_state_path)  # [wrapper-only] trace

    # ── Save model .json ───────────────────────────────────────────
    best_strategy_name = getattr(runner, "best_strategy_name", strategy.name)  # [wrapper-only] winning strategy's name (default to the only strategy) for the model filename
    # [wrapper-only] build the optimized-model filename (model/CoT/strategy/timestamp tags)
    model_path = os.path.join(
        cache_dir,
        f"{model_tag}_{best_strategy_name.replace(' ', '_')}_{cot_tag}_mbd_full_optimized_{current_time}.json",
    )
    optimized_generator.save(model_path)                   # [wrapper-only] persist the optimized DSPy module (reloadable via --load-extractor)
    print(f"\nSaved optimized model: {model_path}")        # [wrapper-only] trace
    _run_logger.info("Saved optimized model: %s", model_path)  # [wrapper-only] trace

    # ── Comparison ────────────────────────────────────────────────
    improvement = best_score - baseline_score              # [wrapper-only] delta of optimized val score over baseline val score
    print("\n" + "=" * 70)                                 # [wrapper-only] banner
    print("RESULTS COMPARISON")                            # [wrapper-only] banner
    print("=" * 70)                                        # [wrapper-only] banner
    print(f"  Baseline:   {baseline_score:.1f}%   (GEPA's own iteration-0 seed eval — single baseline)")  # [wrapper-only] trace baseline
    print(f"  Optimized:  {best_score:.1f}%")              # [wrapper-only] trace optimized
    print(f"  Improvement:{improvement:+.1f}%")            # [wrapper-only] trace delta
    if _seed_retained:                                     # [wrapper-only] GEPA kept the seed → no prompt changed; improvement is 0 by construction
        print("  NOTE: GEPA RETAINED THE SEED — no predictor was improved (all 4 prompts unchanged). "
              "Any non-zero % above would only be val-set noise; here it is honest 0.")  # [wrapper-only]
    # [wrapper-only] trace the baseline/optimized/improvement triple
    _run_logger.info(
        "RESULTS — baseline=%.2f%% optimized=%.2f%% improvement=%+.2f%% seed_retained=%s",
        baseline_score, best_score, improvement, _seed_retained,
    )

    # ── Test set evaluation ────────────────────────────────────────
    test_token_usage: dict = {}                            # [wrapper-only] default empty usage (populated only if a test split exists)
    if len(test_set) > 0:                                  # [wrapper-only] score the optimized module on the held-out test split when one exists
        print("\n" + "=" * 70)                             # [wrapper-only] banner
        print("TEST SET EVALUATION (held-out)")            # [wrapper-only] banner
        print("=" * 70)                                    # [wrapper-only] banner
        _run_logger.info("Starting test evaluation (%d examples)", len(test_set))  # [wrapper-only] trace
        # (see baseline note) track_usage() breaks dspy.Evaluate; read usage from LM history.
        _hist0 = _lm_history_len()                         # [wrapper-only] snapshot LM-history length before eval (token-delta baseline)
        test_result = evaluate(optimized_generator, devset=test_set)  # [wrapper-only] dspy.Evaluate the optimized module on held-out split (no run.py analog — run.py has no train/test split)
        test_token_usage = _token_usage_from_history(_hist0)  # [wrapper-only] sum tokens consumed since _hist0 snapshot
        test_score = test_result.score                     # [wrapper-only] extract aggregate score from Evaluate result
        print(f"  Test score:           {test_score:.1f}%")  # [wrapper-only] trace
        print(f"  Generalization gap:   {abs(best_score - test_score):.1f}%")  # [wrapper-only] trace dev-vs-test gap
        _log_token_usage("TestSet", test_token_usage, _run_logger)  # [wrapper-only] pretty-print token usage
        # [wrapper-only] trace the test score + generalization gap
        _run_logger.info(
            "Test evaluation: score=%.2f%% gap=%.2f%%",
            test_score, abs(best_score - test_score),
        )
    else:
        print("\nTest set is empty — skipping test evaluation.")  # [wrapper-only] trace
        test_score = None                                  # [wrapper-only] sentinel so _save_run_summary records no test score
        _run_logger.warning("Skipping test evaluation — test_set is empty")  # [wrapper-only] trace

    # ── Run summary JSON ───────────────────────────────────────────
    # [wrapper-only] persist scores/tokens/paths as JSON — no run.py analog (run.py writes only pipeline outputs)
    _save_run_summary(
        log_dir=log_dir,
        log_prefix=log_prefix,
        current_time=current_time,
        pipeline_started_at=pipeline_started_at,
        pipeline_t0=pipeline_t0,
        model_to_use=model_to_use,
        args=args,
        baseline_score=baseline_score,
        best_score=best_score,
        best_strategy=best_strategy_name,
        saved_model_path=model_path,
        seed_state_path=seed_state_path,
        optimized_state_path=optimized_state_path,
        test_score=test_score,
        run_logger=_run_logger,
        runner_summary=getattr(runner, "run_summary", {}),
        baseline_token_usage=baseline_token_usage,
        test_token_usage=test_token_usage,
        seed_retained=_seed_retained,   # [wrapper-only] record whether GEPA kept the seed (honest 0-improvement)
    )

    # [wrapper-only] final completion banner (model path + log path)
    print(f"\n{'='*70}")
    print("MBD Full Pipeline complete!")
    print(f"  Model saved:    {model_path}")
    print(f"  Log file:       {log_path}")
    print(f"{'='*70}\n")
    _run_logger.info("MBD full pipeline complete.")        # [wrapper-only] trace

    _restore_streams(sys, _tee_stdout, _tee_stderr, _log_file)  # [wrapper-only] undo the _TeeLogger stdout/stderr redirect installed at run start


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════
def _print_signature_banner(generator: MBDFullPipelineModule, run_logger, label: str = "SEED INSTRUCTION"):  # [wrapper-only] dump predictor instructions — no run.py analog
    """Print current signature instructions from all predictors."""
    # [wrapper-only] banner around the instruction dump
    print(f"\n{'='*70}")
    print(f"[{label}] — instructions from all predictors:")
    print(f"{'='*70}")
    for name, pred in generator.named_predictors():        # [wrapper-only] iterate the 4 DSPy predictors (context/testcase/solver/retry)
        instr = pred.signature.instructions if hasattr(pred, "signature") else "(no signature)"  # [wrapper-only] read predictor's current instruction seed (what GEPA/optimizer mutates)
        print(f"\n  [{name}]")                              # [wrapper-only] predictor name
        print(f"  {instr[:500]}")                           # [wrapper-only] first 500 chars of the instruction
        run_logger.info("[%s] %s — %d chars", label, name, len(instr or ""))  # [wrapper-only] trace instruction length
    print(f"{'='*70}\n")                                    # [wrapper-only] banner


def _cost_summary_safe():  # [wrapper-only] cost-guard totals for the summary; {} if the guard was never installed
    try:
        from utils.cost_guard import cost_summary
        return cost_summary()
    except Exception:
        return {}


def _save_run_summary(   # [wrapper-only] write run-summary JSON (scores/tokens/paths) — no run.py analog
    *,
    log_dir: str,
    log_prefix: str,
    current_time: str,
    pipeline_started_at: str,
    pipeline_t0: float,
    model_to_use: str,
    args,
    baseline_score: float,
    best_score: float,
    best_strategy: str,
    saved_model_path,
    seed_state_path: str,
    optimized_state_path,
    run_logger,
    test_score=None,
    runner_summary=None,
    baseline_token_usage=None,
    test_token_usage=None,
    seed_retained: bool = False,   # [wrapper-only] True => GEPA kept the seed (no predictor improved); improvement is 0 by construction
):
    # [wrapper-only] assemble the run-summary record (metadata + scores + token usage + artifact paths)
    summary = {
        "run_type": "extract_mbd_full",
        "status": "completed",
        "started_at": pipeline_started_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "total_duration_seconds": round(time.time() - pipeline_t0, 1),
        "model_name": model_to_use,
        "optimizer_requested": args.optimizer,
        "gepa_auto": args.gepa_auto,
        "eval_only": args.eval_only,
        "inject_kb_guidelines": args.inject_kb_guidelines,
        "max_retries": args.max_retries,
        "cot": not args.no_cot,
        "num_threads": args.num_threads,
        "baseline_score": baseline_score,          # [wrapper-only] GEPA's OWN iteration-0 seed eval (single baseline; no redundant wrapper pass)
        "baseline_source": ("gepa_iteration0_seed" if args.optimizer == "gepa" and not args.eval_only else "wrapper_eval"),  # [wrapper-only] provenance
        "best_score": best_score,
        "best_strategy": best_strategy,
        "improvement": round(best_score - baseline_score, 4),  # [wrapper-only] apples-to-apples vs GEPA's own baseline (0 when seed retained)
        "seed_retained": seed_retained,            # [wrapper-only] True => no predictor beat the seed (all 4 prompts unchanged)
        "test_score": test_score,
        "saved_model_path": saved_model_path,
        "seed_state_path": seed_state_path,
        "optimized_state_path": optimized_state_path,
        "cost": _cost_summary_safe(),   # [wrapper-only] token/$ totals across all models (empty {} if the guard wasn't installed)
        "runner_summary": runner_summary or {},
        "token_usage": {
            "baseline_eval": baseline_token_usage or {},
            "test_eval": test_token_usage or {},
        },
    }
    summary_path = os.path.join(log_dir, f"{log_prefix}_mbd_full_run_{current_time}_run_summary.json")  # [wrapper-only] primary summary path (in log dir)
    cache_summary_path = os.path.join(".cache", f"{log_prefix}_mbd_full_run_{current_time}_run_summary.json")  # [wrapper-only] duplicate into .cache for downstream tooling
    for path in (summary_path, cache_summary_path):        # [wrapper-only] write the same JSON to both locations
        try:
            with open(path, "w", encoding="utf-8") as _fp:  # [wrapper-only] open target file for writing
                json.dump(summary, _fp, indent=2, ensure_ascii=False)  # [wrapper-only] serialize summary dict as pretty JSON
            run_logger.info("Run summary saved: %s", path)  # [wrapper-only] trace success
        except OSError as e:                               # [wrapper-only] non-fatal — one path may be unwritable (e.g. .cache missing)
            run_logger.warning("Could not save run summary %s: %s", path, e)  # [wrapper-only] trace failure, continue
    print(f"Run summary saved -> {summary_path}")          # [wrapper-only] echo primary path to console


def _lm_history_len() -> int:   # [wrapper-only] LM-history length for token deltas — no run.py analog
    """Current length of the active LM's call history (for usage deltas)."""
    try:
        return len(getattr(dspy.settings.lm, "history", []) or [])  # [wrapper-only] count entries in dspy LM history (start index for a later delta)
    except Exception:                                      # [wrapper-only] LM may be unset / lack history — treat as 0
        return 0                                           # [wrapper-only] safe default


def _token_usage_from_history(start_idx: int) -> dict:   # [wrapper-only] sum prompt/completion tokens — no run.py analog
    """Sum prompt/completion tokens from LM history since start_idx, keyed by model.

    Replaces track_usage() (which corrupts dspy.Evaluate — see import note). Best-effort:
    history is appended by worker threads under the GIL, so this is a close count.
    """
    usage: dict = {}                                       # [wrapper-only] per-model {prompt_tokens, completion_tokens} accumulator
    try:
        hist = (getattr(dspy.settings.lm, "history", []) or [])[start_idx:]  # [wrapper-only] slice history to only calls made since the snapshot
        for entry in hist:                                 # [wrapper-only] one dict per LM call
            u = (entry or {}).get("usage") or {}           # [wrapper-only] the call's token-usage sub-dict
            model = (entry or {}).get("model") or getattr(dspy.settings.lm, "model", "?")  # [wrapper-only] key by model name (fallback to active LM's model)
            d = usage.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0})  # [wrapper-only] get/create this model's tally
            d["prompt_tokens"] += int(u.get("prompt_tokens", 0) or 0)  # [wrapper-only] accumulate prompt tokens
            d["completion_tokens"] += int(u.get("completion_tokens", 0) or 0)  # [wrapper-only] accumulate completion tokens
    except Exception:                                      # [wrapper-only] best-effort — never let token counting break the run
        pass
    return usage                                           # [wrapper-only] return per-model usage map


def _log_token_usage(label: str, usage: dict, run_logger) -> None:   # [wrapper-only] pretty-print token usage — no run.py analog
    if not usage:                                          # [wrapper-only] nothing to print when usage map is empty
        return
    total_prompt = 0                                       # [wrapper-only] cross-model prompt-token running total
    total_completion = 0                                   # [wrapper-only] cross-model completion-token running total
    for model, stats in usage.items():                     # [wrapper-only] one line per model
        pt = stats.get("prompt_tokens", 0) or 0            # [wrapper-only] this model's prompt tokens
        ct = stats.get("completion_tokens", 0) or 0        # [wrapper-only] this model's completion tokens
        total_prompt += pt                                 # [wrapper-only] add to grand total
        total_completion += ct                             # [wrapper-only] add to grand total
        msg = f"  [{label}] {model}: prompt={pt:,} completion={ct:,} total={pt+ct:,}"  # [wrapper-only] per-model formatted line
        print(msg)                                         # [wrapper-only] echo to console
        run_logger.info(msg.strip())                       # [wrapper-only] and to the run log
    if len(usage) > 1:                                     # [wrapper-only] only emit a TOTAL row when >1 model was used
        msg = f"  [{label}] TOTAL: prompt={total_prompt:,} completion={total_completion:,} total={total_prompt+total_completion:,}"  # [wrapper-only] grand-total line
        print(msg)                                         # [wrapper-only] echo to console
        run_logger.info(msg.strip())                       # [wrapper-only] and to the run log


def _restore_streams(sys_module, tee_stdout, tee_stderr, log_file):   # [wrapper-only] undo _TeeLogger redirect — no run.py analog
    sys_module.stdout = tee_stdout._original               # [wrapper-only] restore the original stdout captured by the tee
    sys_module.stderr = tee_stderr._original               # [wrapper-only] restore the original stderr captured by the tee
    log_file.close()                                       # [wrapper-only] close the log-file handle the tee was writing to


if __name__ == "__main__":
    main()
