"""
MBD-UTA Full Pipeline DSPy Optimization
========================================

Full pipeline: wraps run.py's EXACT logic as a DSPy Module with 3 optimizable prompts.
  Prompt 1 (context_extractor):   context extraction from equation graph
  Prompt 2 (testcase_generator):  JSON test case generation
  Prompt 4 (equation_solver_dll): DLL-verified equation solving

All pipeline logic between prompts is copied verbatim from run.py.
No new abstractions. Bugs in run.py preserved intentionally.

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

import argparse
import csv
import importlib
import io
import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import dotenv
import dspy
from dspy.utils.usage_tracker import track_usage

# Import project-level config BEFORE inserting MBD path into sys.path,
# so Python finds the project's config.py (not MBD's config.py).
from config import reconfigure_lm, reflectin_lm  # noqa: E402

# ── MBD pipeline path + config swap (same pattern as mbd_data.py) ───────────
_MBD_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "MBD", "weakspot_code_to_run_2")
)
if _MBD_ROOT not in sys.path:
    sys.path.insert(0, _MBD_ROOT)

_repo_config = sys.modules.get("config")
_mbd_config_spec = importlib.util.spec_from_file_location(
    "mbd_config", os.path.join(_MBD_ROOT, "config.py")
)
_mbd_config_mod = importlib.util.module_from_spec(_mbd_config_spec)
sys.modules["config"] = _mbd_config_mod
_mbd_config_spec.loader.exec_module(_mbd_config_mod)

# run.py pipeline imports — exact same as run.py lines 163-180
from pipeline.test_generator import update_test_cases, extract_and_replace_equations
from pipeline.equation_solver import (
    extract_json_from_text_dll,
    update_equation_values,
    map_equations_to_json,
)
from pipeline.equation_solver_dll import (
    _attempt_solve,
    _build_substitution_dict,
    resolve_equation_block,
    generate_bodmas_chain,
)
from pipeline.postprocess import (
    remove_variable_and_parameter_blocks,
    flatten_precondition_and_actions,
    deduplicate_actions,
    clean_deduplicated_solution,
    remove_unselected_mux_actions,
    compute_unselected_mux_variables,
)
from pipeline.types import MergedEquation
# build_retry_instruction is in prompts/, NOT in pipeline/equation_solver_dll
from prompts.equation_solution_prompt_dll import build_retry_instruction
from pipeline.function_caller import FunctionCaller

# Restore repo-level config
if _repo_config is not None:
    sys.modules["config"] = _repo_config
else:
    sys.modules.pop("config", None)

from signatures.mbd import MBDContextExtraction, MBDTestCaseGeneration, MBDEquationSolvingDLL
from utils.mbd_data import load_mbd_dataset, stratified_split_mbd_dataset, all_fc_entries
from evaluation.mbd_metric import (
    mbd_testcase_metric,
    mbd_metric_with_feedback,
    _compute_all_components,
)
from optimization import OptimizationRunner, create_gepa_strategy

dotenv.load_dotenv(override=True)

# ── Debug logger for step-by-step pipeline execution ─────────────────────────
# Silent by default. A caller (e.g. an end-to-end runner script) configures
# logging (basicConfig / file handler) and sets this logger's level to DEBUG
# to get a full step-by-step trace of forward().
logger = logging.getLogger("extract_mbd_full")


# ═══════════════════════════════════════════════════════════════════
# DATASET PREPARATION
# ═══════════════════════════════════════════════════════════════════

def prepare_full_examples(dataset):
    """Re-wrap examples from load_mbd_dataset() with full with_inputs() for MBDFullPipelineModule.

    mbd_data.py's with_inputs() stays at 5 fields to not break MBDTestGenerator.
    Here we extend it to include all 12 fields forward() needs.
    """
    return [
        ex.with_inputs(
            "equation_json", "const_values", "weakspot_context",
            "instance_names", "raster_time",
            "dll_path", "signalsheet_data", "model_interface_data",
            "relations_json", "equation_results_json",
            "op_map_json", "weakspot_op_key",
        )
        for ex in dataset
    ]


# ═══════════════════════════════════════════════════════════════════
# DSPy MODULE
# ═══════════════════════════════════════════════════════════════════

class MBDFullPipelineModule(dspy.Module):
    """Full MBD-UTA pipeline: Prompt 1 + Prompt 2 + bridge + Prompt 4 + post-processing.

    Exact reproduction of run.py's pipeline with 3 call_llm() calls replaced
    by DSPy predictors so GEPA can optimize them. All other logic verbatim.
    """

    _fc_cache: dict = {}
    _fc_cache_lock = threading.Lock()

    def __init__(self, use_cot: bool = True):
        super().__init__()
        Predictor = dspy.ChainOfThought if use_cot else dspy.Predict
        self.context_extractor   = Predictor(MBDContextExtraction)   # Prompt 1
        self.testcase_generator  = Predictor(MBDTestCaseGeneration)  # Prompt 2
        self.equation_solver_dll = Predictor(MBDEquationSolvingDLL)  # Prompt 4

    def forward(
        self,
        equation_json,
        const_values,
        weakspot_context,
        instance_names,
        raster_time,
        dll_path="",
        signalsheet_data="",
        model_interface_data="",
        relations_json="{}",
        equation_results_json="[]",
        op_map_json="{}",
        weakspot_op_key="",
        **kwargs,
    ):
        logger.info("=" * 70)
        logger.info("forward(): START — fc/weakspot_op_key=%s raster_time=%s dll_path=%s",
                     weakspot_op_key, raster_time, dll_path)
        logger.debug("forward(): equation_json (len=%d) = %s", len(equation_json or ""), equation_json)
        logger.debug("forward(): const_values = %s", const_values)
        logger.debug("forward(): weakspot_context (len=%d) = %s", len(weakspot_context or ""), weakspot_context)
        logger.debug("forward(): instance_names = %s", instance_names)

        # ── Prompt 1: Context Extraction (run.py _step_3a_get_context) ──────────
        logger.info("forward(): ----- Prompt 1 (context_extractor) — calling LM -----")
        ctx = self.context_extractor(
            equation_json=equation_json,
            const_values=const_values,
            weakspot_context=weakspot_context,
        )
        context_analysis = ctx.context_analysis
        logger.info("forward(): Prompt 1 DONE — context_analysis (len=%d)", len(context_analysis or ""))
        logger.debug("forward(): Prompt 1 context_analysis = %s", context_analysis)

        # ── Prompt 2: Test Case Generation (run.py _step_3b_generate_testcases) ─
        logger.info("forward(): ----- Prompt 2 (testcase_generator) — calling LM -----")
        tc = self.testcase_generator(
            context_analysis=context_analysis,
            equation_json=equation_json,
            const_values=const_values,
            instance_names=instance_names,
            raster_time=raster_time,
        )
        test_cases_raw = tc.test_cases  # CAPTURE Prompt 2 raw output for staged metric
        logger.info("forward(): Prompt 2 DONE — test_cases_raw (len=%d)", len(test_cases_raw or ""))
        logger.debug("forward(): Prompt 2 test_cases_raw = %s", test_cases_raw)

        # ── Bridge: run.py lines 486-499 ────────────────────────────────────────
        # Parse Prompt 2 output (mirrors generate_and_parse_test_cases regex extraction)
        logger.info("forward(): ----- Bridge: parsing Prompt 2 JSON output -----")
        try:
            raw_str = test_cases_raw.strip() if test_cases_raw else ""
            json_match = re.search(r"\[\s*{.*}\s*\]", raw_str, re.DOTALL)
            if json_match:
                tc_list = json.loads(json_match.group())
                logger.debug("forward(): bridge — extracted JSON via regex match")
            else:
                tc_list = json.loads(raw_str)
                logger.debug("forward(): bridge — parsed raw_str directly as JSON (no regex match)")
            if not isinstance(tc_list, list):
                raise ValueError("Not a list")
            logger.info("forward(): bridge — parsed %d test case(s) from Prompt 2", len(tc_list))
        except (json.JSONDecodeError, ValueError, AttributeError) as _bridge_err:
            logger.error("forward(): bridge — FAILED to parse Prompt 2 output: %s", _bridge_err)
            return dspy.Prediction(
                context_analysis=context_analysis,
                test_cases_pre_postprocess=test_cases_raw,
                test_cases=test_cases_raw,
                m7_score=None,
                m7_details="Prompt 2 parse failed — no equations processed",
            )

        # Deserialize precomputed Phase 2 data
        relations = json.loads(relations_json) if relations_json else {}
        equation_results_raw = json.loads(equation_results_json) if equation_results_json else []
        logger.debug("forward(): relations (count=%d) = %s", len(relations), relations)
        logger.debug("forward(): equation_results_raw — %d entr(ies)", len(equation_results_raw))

        # run.py line 489: update_test_cases(test_cases, eq_ctx.relations)
        updated_test_cases = update_test_cases(tc_list, relations)
        logger.debug("forward(): update_test_cases() -> %d test case(s)", len(updated_test_cases))

        # run.py line 490: updated_json, equation_dict = extract_and_replace_equations(updated_test_cases)
        # Signature: (list) -> (list, dict)  — 1 arg, returns (updated_json, equation_dict)
        updated_json, equation_dict = extract_and_replace_equations(updated_test_cases)
        logger.info("forward(): extract_and_replace_equations() -> %d equation(s) extracted", len(equation_dict))
        for _uid, _eq in equation_dict.items():
            logger.debug("forward(): equation_dict[%s] = %s", _uid, _eq)

        # run.py lines 493-499: compute filtered_const_values
        const_dict = json.loads(const_values) if const_values else {}
        all_equations = " ".join(equation_dict.values())
        matches = re.findall(r"SystemConstant\((.*?)\)", all_equations)
        unique_constants = list(set(matches))
        filtered_const_values = {k: v for k, v in const_dict.items() if k in unique_constants}
        logger.debug("forward(): unique_constants referenced = %s", unique_constants)
        logger.debug("forward(): filtered_const_values = %s", filtered_const_values)

        # ── Prompt 4: DLL Equation Solving (run.py _step_4a_solve_equations_dll) ─
        fc = self._get_function_caller(dll_path, signalsheet_data, model_interface_data)
        logger.info("forward(): FunctionCaller fc=%s (None means DLL verification disabled)",
                     "available" if fc is not None else "None")

        # run.py lines 521-537: build merged_equation_map
        original_eq_map = {
            r["original_equation"]: r
            for r in equation_results_raw
            if r.get("original_equation", "").strip()
        }
        merged_equation_map = {}
        for uid, full_eq in equation_dict.items():
            lhs = full_eq.split("=")[0].strip()
            if lhs in original_eq_map:
                orig = original_eq_map[lhs]
                merged_equation_map[uid] = MergedEquation(
                    uid=uid,
                    equation_to_solve=full_eq,
                    original_equation=lhs,
                    assignment_chain=orig["assignment_chain"],
                    final_substituted_equation=orig["final_substituted_equation"],
                )
            else:
                logger.debug("forward(): equation_dict[%s] LHS %r NOT found in original_eq_map (dropped from merged_equation_map)", uid, lhs)
        logger.info("forward(): merged_equation_map has %d entr(ies) (out of %d equation_dict entries)",
                     len(merged_equation_map), len(equation_dict))

        # run.py lines 539-543
        updated_dict = {}
        is_first_pass = True
        locked_param_block = ""
        all_unselected_vars = set()
        substitution_dict = {}

        # M7 inline tracking: record 1/N score per equation (N = attempts in retry loop)
        eq_m7_scores: list = []

        # run.py lines 584-588: _substitute_ops defined INLINE (not importable from pipeline)
        def _substitute_ops(equation, subs):
            return re.sub(
                r"\$op\d+",
                lambda m: subs.get(m.group(0), m.group(0)),
                equation,
            )

        for _eq_idx, (eq_id, merged_eq) in enumerate(merged_equation_map.items()):
            logger.info("forward(): ===== Equation %d/%d (uid=%s) =====", _eq_idx + 1, len(merged_equation_map), eq_id)
            logger.debug("forward(): eq[%s] equation_to_solve = %s", eq_id, merged_eq.equation_to_solve)
            logger.debug("forward(): eq[%s] original_equation = %s", eq_id, merged_eq.original_equation)
            logger.debug("forward(): eq[%s] assignment_chain = %s", eq_id, merged_eq.assignment_chain)
            logger.debug("forward(): eq[%s] final_substituted_equation = %s", eq_id, merged_eq.final_substituted_equation)

            # run.py lines 546-547
            eq_to_solve = merged_eq.assignment_chain.split("\n")[::-1]
            lhs_value = merged_eq.equation_to_solve.split("=")[-1].strip()
            logger.debug("forward(): eq[%s] eq_to_solve (reversed assignment_chain, %d line(s)) = %s",
                          eq_id, len(eq_to_solve), eq_to_solve)
            logger.debug("forward(): eq[%s] lhs_value (target) = %s", eq_id, lhs_value)

            # run.py line 561 (call_llm) → DSPy predictor (Prompt 4, first call)
            logger.info("forward(): eq[%s] ----- Prompt 4 (equation_solver_dll) attempt #1 — calling LM -----", eq_id)
            solver_pred = self.equation_solver_dll(
                equation_chain="\n".join(eq_to_solve),
                target_value=lhs_value,
                const_values=json.dumps(filtered_const_values),
                locked_param_block=locked_param_block,
                retry_instruction="",
            )
            logger.debug("forward(): eq[%s] Prompt 4 attempt#1 raw variable_assignments = %s", eq_id, solver_pred.variable_assignments)
            # extract_json_from_text_dll returns None when LLM output has no valid JSON list.
            # Treat None as [] so _attempt_solve doesn't crash — same as run.py intent.
            simulated_response = extract_json_from_text_dll(solver_pred.variable_assignments) or []
            logger.debug("forward(): eq[%s] Prompt 4 attempt#1 simulated_response (%d item(s)) = %s",
                          eq_id, len(simulated_response), simulated_response)

            # run.py lines 567-579: lock parameters on first pass (exact copy)
            if is_first_pass and simulated_response:
                locked = {
                    p.split(" = ")[0]: p.split(" = ")[1]
                    for p in simulated_response
                    if p.startswith("Parameter(") and " = " in p
                }
                if locked:
                    locked_param_block = (
                        "## Locked parameter values -- do not change on retry:\n"
                        + "\n".join(f"{k} = {v}" for k, v in locked.items())
                        + "\n\n"
                    )
                is_first_pass = False
                logger.debug("forward(): eq[%s] locked_param_block (after first pass) = %r", eq_id, locked_param_block)

            if fc is not None:
                # run.py lines 581-596 (DLL available)
                attempt = _attempt_solve(simulated_response, eq_to_solve, fc)
                substitution_dict = _build_substitution_dict(attempt.final_assignments)
                logger.debug("forward(): eq[%s] attempt#1 final_assignments = %s", eq_id, attempt.final_assignments)
                logger.debug("forward(): eq[%s] attempt#1 substitution_dict = %s", eq_id, substitution_dict)

                expressions = [_substitute_ops(merged_eq.final_substituted_equation, substitution_dict)]
                eq_list = []
                for expr in expressions:
                    eq_list = generate_bodmas_chain(expr)
                logger.debug("forward(): eq[%s] attempt#1 substituted expression = %s", eq_id, expressions)
                logger.debug("forward(): eq[%s] attempt#1 BODMAS chain (%d step(s)) = %s", eq_id, len(eq_list), eq_list)

                # run.py line 596: first compute_unselected_mux_variables uses eq_list (BODMAS output)
                all_unselected_vars |= compute_unselected_mux_variables(eq_list, substitution_dict)
                final_check, try_fc = resolve_equation_block(eq_list, fc)
                logger.debug("forward(): eq[%s] attempt#1 final_check (DLL-resolved) = %s", eq_id, final_check)

                expected = merged_eq.equation_to_solve.split("=")[-1].strip()
                actual = final_check[-1].split("=")[-1].strip() if final_check else ""
                first_try_pass = (expected == actual)  # M7: capture before retry loop
                logger.info("forward(): eq[%s] attempt#1: expected=%r actual=%r -> %s",
                             eq_id, expected, actual, "PASS" if first_try_pass else "MISMATCH")

                # run.py lines 608-637: EXACT retry loop — bug preserved intentionally
                # (only 1 real retry ever happens despite range(12), due to if retry_num < 1)
                for retry_num in range(12):
                    if expected == actual:
                        break
                    if retry_num < 1:
                        extra = build_retry_instruction(
                            expected, actual, try_fc, eq_list, attempt.solved_equations
                        )
                        logger.info("forward(): eq[%s] ----- Prompt 4 RETRY #%d — calling LM (mismatch expected=%r actual=%r) -----",
                                     eq_id, retry_num + 1, expected, actual)
                        logger.debug("forward(): eq[%s] retry_instruction = %s", eq_id, extra)
                        # run.py line 624 (retry call_llm) → DSPy predictor again
                        retry_pred = self.equation_solver_dll(
                            equation_chain="\n".join(eq_to_solve),
                            target_value=lhs_value,
                            const_values=json.dumps(filtered_const_values),
                            locked_param_block=locked_param_block + extra,
                            retry_instruction=extra,
                        )
                        logger.debug("forward(): eq[%s] retry#%d raw variable_assignments = %s", eq_id, retry_num + 1, retry_pred.variable_assignments)
                        simulated_response = extract_json_from_text_dll(retry_pred.variable_assignments) or []
                        logger.debug("forward(): eq[%s] retry#%d simulated_response (%d item(s)) = %s",
                                      eq_id, retry_num + 1, len(simulated_response), simulated_response)
                        attempt = _attempt_solve(simulated_response, eq_to_solve, fc)
                        substitution_dict = _build_substitution_dict(attempt.final_assignments)
                        expressions = [_substitute_ops(merged_eq.final_substituted_equation, substitution_dict)]
                        for expr in expressions:
                            eq_list = generate_bodmas_chain(expr)
                        final_check, try_fc = resolve_equation_block(eq_list, fc)
                        actual = final_check[-1].split("=")[-1].strip() if final_check else ""
                        logger.info("forward(): eq[%s] retry#%d: expected=%r actual=%r -> %s",
                                     eq_id, retry_num + 1, expected, actual, "PASS" if expected == actual else "FAIL")
                    else:
                        break

                # M7 inline: 1/N scoring — 1.0 (first try), 0.5 (needed retry), 0.0 (failed)
                if first_try_pass:
                    eq_m7_scores.append(1.0)
                    logger.info("forward(): eq[%s] M7 inline score = 1.0 (first-try pass)", eq_id)
                elif expected == actual:
                    eq_m7_scores.append(0.5)
                    logger.info("forward(): eq[%s] M7 inline score = 0.5 (retry pass)", eq_id)
                else:
                    eq_m7_scores.append(0.0)
                    logger.info("forward(): eq[%s] M7 inline score = 0.0 (FAIL — expected=%r actual=%r)", eq_id, expected, actual)
            else:
                # fc=None (no 32-bit Python) — Prompt 4 ran, DLL verification skipped
                logger.warning("forward(): eq[%s] fc is None -> DLL verification SKIPPED, no M7 inline score for this eq", eq_id)

            # run.py line 639
            updated_dict[eq_id] = simulated_response
            logger.debug("forward(): eq[%s] updated_dict[%s] = %s", eq_id, eq_id, simulated_response)

        # ── M7 inline score (must be computed before post-processing clears state) ─
        if eq_m7_scores:
            m7_inline_score = sum(eq_m7_scores) / len(eq_m7_scores)
            n_first = sum(1 for s in eq_m7_scores if s == 1.0)
            n_retry = sum(1 for s in eq_m7_scores if s == 0.5)
            n_fail  = sum(1 for s in eq_m7_scores if s == 0.0)
            m7_inline_details = (
                f"DLL inline: {n_first} first-try + {n_retry} retry-pass + {n_fail} fail"
                f" / {len(eq_m7_scores)} eqs"
            )
        else:
            m7_inline_score = None  # DLL unavailable → metric falls back to post-hoc
            m7_inline_details = "DLL unavailable — no inline M7"
        logger.info("forward(): M7 INLINE — score=%s (%s)", m7_inline_score, m7_inline_details)

        # ── Post-processing: run.py lines 653-668 VERBATIM ──────────────────────
        logger.info("forward(): ----- Post-processing chain -----")

        # run.py line 653: MUST call update_equation_values BEFORE map_equations_to_json
        # This mutates equation_dict: UUID strings → solved value lists
        update_equation_values(equation_dict, updated_dict)
        logger.debug("forward(): post-process — update_equation_values() done; equation_dict now = %s", equation_dict)

        # run.py line 654
        pre_final = map_equations_to_json(updated_json, equation_dict)
        logger.debug("forward(): post-process — map_equations_to_json() -> pre_final = %s", pre_final)

        # run.py lines 655-657
        cleaned = remove_variable_and_parameter_blocks(pre_final)
        logger.debug("forward(): post-process — remove_variable_and_parameter_blocks() -> %s", cleaned)
        flattened = flatten_precondition_and_actions(cleaned)
        logger.debug("forward(): post-process — flatten_precondition_and_actions() -> %s", flattened)
        deduped = deduplicate_actions(flattened)  # dedup bug preserved
        logger.debug("forward(): post-process — deduplicate_actions() -> %s", deduped)

        # run.py lines 659-665: second compute_unselected_mux_variables call
        # Uses precomputed op_map (from example) instead of rebuilding from dependency_mapping
        if merged_equation_map:
            last_eq = list(merged_equation_map.values())[-1]
            last_eq_to_solve = last_eq.assignment_chain.split("\n")[::-1]
            stored_op_map = json.loads(op_map_json) if op_map_json and op_map_json != "{}" else {}
            all_unselected_vars |= compute_unselected_mux_variables(
                last_eq_to_solve, substitution_dict, stored_op_map
            )
            logger.debug("forward(): post-process — all_unselected_vars (final) = %s", all_unselected_vars)

        # run.py lines 667-668
        final_json = clean_deduplicated_solution(deduped)
        logger.debug("forward(): post-process — clean_deduplicated_solution() -> %s", final_json)
        final_json = remove_unselected_mux_actions(final_json, all_unselected_vars)
        logger.info("forward(): post-process — remove_unselected_mux_actions() -> final_json with %d test case(s)",
                     len(final_json) if isinstance(final_json, list) else -1)
        logger.debug("forward(): FINAL JSON = %s", json.dumps(final_json, indent=2))

        logger.info("forward(): END — test_cases_pre_postprocess (Prompt2 raw, %d chars), "
                     "final test_cases (%d chars), m7_inline_score=%s",
                     len(test_cases_raw or ""), len(json.dumps(final_json)), m7_inline_score)
        logger.info("=" * 70)

        return dspy.Prediction(
            context_analysis=context_analysis,
            test_cases_pre_postprocess=test_cases_raw,   # Prompt 2 raw — for staged metric
            test_cases=json.dumps(final_json),            # final pipeline output
            m7_score=m7_inline_score,     # inline 1/N score; None if DLL unavailable
            m7_details=m7_inline_details, # breakdown string for logging
        )

    @classmethod
    def _get_function_caller(cls, dll_path, signalsheet_data, model_interface_data):
        """Initialize and cache FunctionCaller — same logic as run.py lines 789-798.

        Returns None if DLL is unavailable (missing file or missing 32-bit Python).
        When None, the solver loop runs Prompt 4 but skips DLL verification.
        """
        if not dll_path or not os.path.exists(dll_path):
            logger.warning("_get_function_caller: dll_path %r missing or does not exist -> fc=None", dll_path)
            return None
        with cls._fc_cache_lock:
            if dll_path in cls._fc_cache:
                logger.info("_get_function_caller: cache HIT for dll_path=%s -> %s", dll_path,
                             "fc available" if cls._fc_cache[dll_path] is not None else "None (cached failure)")
                return cls._fc_cache[dll_path]
            # run.py lines 729-730: work_dir = dirname(fc_dir), fcname = basename(fc_dir)
            # dll_path = <data_root>/<fc_name>/env_libs/<dll_file>
            env_libs_dir = os.path.dirname(dll_path)      # <data_root>/<fc_name>/env_libs
            fc_dir       = os.path.dirname(env_libs_dir)  # <data_root>/<fc_name>
            fcname       = os.path.basename(fc_dir)       # fc_name string
            work_dir     = os.path.dirname(fc_dir)        # <data_root>
            logger.info("_get_function_caller: cache MISS — dll_path=%s FCname=%s work_dir=%s", dll_path, fcname, work_dir)

            # run.py line 792: signal_sheet_data = convert_signal_sheet_to_list(signal_sheet_df)
            # convert_signal_sheet_to_list produces list-of-dicts from DataFrame.
            # Here we have raw CSV text — csv.DictReader produces the same structure.
            signal_sheet_list = list(csv.DictReader(io.StringIO(signalsheet_data or "")))
            logger.debug("_get_function_caller: signal_sheet_list has %d row(s)", len(signal_sheet_list))

            # run.py line 797: model_interface_data is a DataFrame in run.py.
            # mbd_data.py stores it as raw CSV text — convert to DataFrame to match.
            import pandas as pd
            try:
                mi_df = pd.read_csv(io.StringIO(model_interface_data or ""))
                logger.debug("_get_function_caller: model_interface_data parsed as DataFrame, shape=%s", mi_df.shape)
            except Exception as _mi_err:
                mi_df = model_interface_data  # fallback: pass as-is
                logger.warning("_get_function_caller: model_interface_data CSV parse failed (%s) -> using raw text", _mi_err)

            # run.py lines 793-798; catch RuntimeError if 32-bit Python is missing
            try:
                cls._fc_cache[dll_path] = FunctionCaller(
                    FCname=fcname,
                    work_dir=work_dir,
                    signal_sheet_data=signal_sheet_list,
                    model_interface_data=mi_df,
                )
                logger.info("_get_function_caller: FunctionCaller initialized successfully (FCname=%s)", fcname)
            except Exception as _fc_err:
                logging.getLogger("extract_mbd_full").warning(
                    "FunctionCaller unavailable (%s) — DLL verification disabled, M7=0.5", _fc_err
                )
                cls._fc_cache[dll_path] = None  # cache None so we don't retry every call
            return cls._fc_cache[dll_path]


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MBD-UTA Full Pipeline DSPy Optimization")
    parser.add_argument(
        "--model-name", type=str, default=os.getenv("MODEL_NAME"),
        help="Student model name (default: from .env MODEL_NAME)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable LLM response caching",
    )
    parser.add_argument(
        "--no-cot", action="store_true",
        help="Use dspy.Predict instead of dspy.ChainOfThought",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only evaluate baseline, no optimization",
    )
    parser.add_argument(
        "--eval-test", action="store_true",
        help="Evaluate on test set (requires --load-extractor)",
    )
    parser.add_argument(
        "--optimizer", type=str, default="gepa",
        choices=["gepa"],
        help="Optimizer to use (default: gepa)",
    )
    parser.add_argument(
        "--gepa-auto", type=str, default="light",
        choices=["light", "medium", "heavy"],
        help="GEPA intensity preset (default: light)",
    )
    parser.add_argument(
        "--load-extractor", type=str, default=None,
        help="Path to previously saved optimized extractor to load",
    )
    parser.add_argument(
        "--use-all-tasks", action="store_true",
        help="Use all 37 tasks instead of default 10-15 subset",
    )
    parser.add_argument(
        "--num-threads", type=int, default=2,
        help="Number of threads for parallel evaluation (default: 2)",
    )
    parser.add_argument(
        "--run-timestamp", type=str, default=None,
        help="Pre-generated timestamp (YYYYmmddHHMMSS). Falls back to datetime.now().",
    )
    return parser


# ═══════════════════════════════════════════════════════════════════
# TEE LOGGER  (mirrors extract_mbd.py _TeeLogger)
# ═══════════════════════════════════════════════════════════════════
class _TeeLogger:
    """Redirect a stream (stdout or stderr) to both terminal and a shared log file."""

    def __init__(self, original_stream, log_file):
        self._original = original_stream
        self._file = log_file

    def write(self, msg: str):
        self._file.write(msg)
        self._file.flush()
        try:
            self._original.write(msg)
        except UnicodeEncodeError:
            safe_msg = msg.encode(
                getattr(self._original, "encoding", "utf-8") or "utf-8",
                errors="replace",
            ).decode(
                getattr(self._original, "encoding", "utf-8") or "utf-8",
                errors="replace",
            )
            self._original.write(safe_msg)

    def flush(self):
        self._file.flush()
        self._original.flush()

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return self._original.isatty()


# ═══════════════════════════════════════════════════════════════════
# PER-EXAMPLE DIAGNOSTIC HELPER
# ═══════════════════════════════════════════════════════════════════
def _print_per_example_scores(generator: MBDFullPipelineModule, examples, logger=None):
    """Run inference on each example and print per-metric breakdown."""
    _log = logger or logging.getLogger("extract_mbd_full")

    print("\n" + "-" * 70)
    print("PER-EXAMPLE METRIC BREAKDOWN")
    print("-" * 70)
    _log.info("=== PER-EXAMPLE METRIC BREAKDOWN ===")

    # Weights must match mbd_metric.py constants exactly
    WEIGHTS = [
        ("m1", 0.15), ("m2", 0.15), ("m3", 0.05), ("m4", 0.30),
        ("m6", 0.0), ("m7", 0.10), ("m8", 0.25),
    ]

    for i, ex in enumerate(examples):
        fc_tag = f"{getattr(ex, 'fc_name', '?')} ws={getattr(ex, 'weakspot_id', '?')}"
        try:
            pred = generator(
                equation_json=ex.equation_json,
                const_values=ex.const_values,
                weakspot_context=ex.weakspot_context,
                instance_names=ex.instance_names,
                raster_time=ex.raster_time,
                dll_path=getattr(ex, "dll_path", ""),
                signalsheet_data=getattr(ex, "signalsheet_data", ""),
                model_interface_data=getattr(ex, "model_interface_data", ""),
                relations_json=getattr(ex, "relations_json", "{}"),
                equation_results_json=getattr(ex, "equation_results_json", "[]"),
                op_map_json=getattr(ex, "op_map_json", "{}"),
                weakspot_op_key=getattr(ex, "weakspot_op_key", ""),
            )
            scores = _compute_all_components(ex, pred)
            total = sum(w * scores[k] for k, w in WEIGHTS)
            msg = (
                f"[{i+1}] {fc_tag} → total={total:.3f} | "
                f"M1={scores['m1']:.2f} M2={scores['m2']:.2f} M3={scores['m3']:.2f} "
                f"M4={scores['m4']:.2f} M6={scores['m6']:.2f} "
                f"M7={scores['m7']:.2f} M8={scores['m8']:.2f}"
            )
            print(f"  {msg}")
            print(f"       M1: {scores['m1_details']}")
            print(f"       M2: {scores['m2_details']}")
            print(f"       M3: {scores['m3_details']}")
            print(f"       M4: {scores['m4_details']}")
            print(f"       M6: {scores['m6_details']}")
            print(f"       M7: {scores['m7_details']}")
            print(f"       M8: {scores['m8_details']}")
            _log.info(msg)
        except Exception as e:
            err_msg = f"[{i+1}] {fc_tag} → ERROR: {e}"
            print(f"  {err_msg}")
            _log.error(err_msg, exc_info=True)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Timestamp ─────────────────────────────────────────────────
    current_time = args.run_timestamp if args.run_timestamp else datetime.now().strftime("%Y%m%d%H%M%S")
    pipeline_started_at = datetime.now().isoformat(timespec="seconds")
    pipeline_t0 = time.time()

    # ── Build log/cache name prefix ────────────────────────────────
    model_tag = (args.model_name or os.getenv("MODEL_NAME", "Unknown")).replace("/", "_")
    cot_tag = "NoCot" if args.no_cot else "CoT"
    gepa_auto_tag = args.gepa_auto.capitalize()
    if args.eval_only:
        strategy_tag = "EvalOnly"
    elif args.eval_test:
        strategy_tag = "EvalTest"
    else:
        strategy_tag = f"GEPA_MBD_Full_{gepa_auto_tag}"
    log_prefix = f"{model_tag}_{strategy_tag}_{cot_tag}"

    # ── Log directory & TeeLogger ──────────────────────────────────
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_prefix}_mbd_full_run_{current_time}.log")
    _log_file = open(log_path, "a", encoding="utf-8")
    _tee_stdout = _TeeLogger(sys.stdout, _log_file)
    _tee_stderr = _TeeLogger(sys.stderr, _log_file)
    sys.stdout = _tee_stdout
    sys.stderr = _tee_stderr

    # ── Python logging setup ───────────────────────────────────────
    _log_fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _file_handler = logging.FileHandler(log_path, encoding="utf-8")
    _file_handler.setFormatter(_log_fmt)
    _console_handler = logging.StreamHandler(_tee_stderr._original)
    _console_handler.setFormatter(_log_fmt)
    logging.basicConfig(level=logging.WARNING, handlers=[_file_handler, _console_handler])

    for _ns in ("extract_mbd_full", "mbd.data", "mbd.metric"):
        logging.getLogger(_ns).setLevel(logging.DEBUG)

    class _TruncationInspectorHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno != logging.WARNING:
                return
            if "LM response was truncated" not in record.getMessage():
                return
            try:
                history_text = dspy.inspect_history(n=1)
            except Exception as exc:
                history_text = f"<could not retrieve history: {exc}>"
            _log_file.write(
                "\n" + "=" * 80 + "\n"
                "[TRUNCATION INSPECTOR] Last LM interaction that hit max_tokens:\n"
                + "=" * 80 + "\n"
                + (history_text or "<empty history>")
                + "\n" + "=" * 80 + "\n\n"
            )
            _log_file.flush()

    _trunc_inspector = _TruncationInspectorHandler()
    _trunc_inspector.setLevel(logging.WARNING)
    logging.getLogger("dspy.clients.lm").addHandler(_trunc_inspector)

    _run_logger = logging.getLogger("extract_mbd_full")
    _run_logger.info("MBD Full pipeline run started — log file: %s", log_path)

    # ── Model configuration ────────────────────────────────────────
    model_to_use = args.model_name or os.getenv("MODEL_NAME", "azure/gpt-4.1-mini")
    cache_enabled = not args.no_cache
    reconfigure_lm(model_to_use, cache=cache_enabled)

    # ── Configuration banner ───────────────────────────────────────
    print(f"\n{'='*70}")
    print(" MBD-UTA Full Pipeline Configuration:")
    print(f"  Model:           {model_to_use}")
    print(f"  Module:          {'Predict (no CoT)' if args.no_cot else 'ChainOfThought'}")
    print(f"  Cache:           {cache_enabled}")
    print(f"  Optimizer:       {args.optimizer} (auto={args.gepa_auto})")
    print(f"  Eval-only:       {args.eval_only}")
    print(f"  Eval-test:       {args.eval_test}")
    print(f"  Load extractor:  {args.load_extractor or '(none)'}")
    print(f"  Num threads:     {args.num_threads}")
    print(f"  All tasks:       {args.use_all_tasks}")
    print(f"  Log file:        {log_path}")
    print(f"{'='*70}\n")
    _run_logger.info(
        "Config — model=%s cache=%s cot=%s optimizer=%s gepa_auto=%s num_threads=%d",
        model_to_use, cache_enabled, not args.no_cot, args.optimizer, args.gepa_auto, args.num_threads,
    )

    # ── Data loading ───────────────────────────────────────────────
    print("Loading MBD dataset...")
    _run_logger.info("Loading MBD dataset (use_all_tasks=%s)", args.use_all_tasks)
    fc_entries = all_fc_entries() if args.use_all_tasks else None
    dataset = load_mbd_dataset(fc_entries=fc_entries)

    if not dataset:
        msg = "ERROR: No examples loaded. Check data directory and FC entries."
        print(msg)
        _run_logger.error(msg)
        _restore_streams(sys, _tee_stdout, _tee_stderr, _log_file)
        sys.exit(1)

    # Re-wrap examples with full 12-field with_inputs() for MBDFullPipelineModule
    dataset_full = prepare_full_examples(dataset)
    train_set, val_set, test_set = stratified_split_mbd_dataset(dataset_full)
    print(f"Split: train={len(train_set)} | val={len(val_set)} | test={len(test_set)}")
    _run_logger.info(
        "Dataset split: total=%d train=%d val=%d test=%d",
        len(dataset_full), len(train_set), len(val_set), len(test_set),
    )

    if len(test_set) == 0:
        print("WARNING: test_set is empty — test evaluation will be skipped.")
        _run_logger.warning("test_set is empty — test evaluation will be skipped")

    # ── Create module ──────────────────────────────────────────────
    generator = MBDFullPipelineModule(use_cot=not args.no_cot)
    _run_logger.info("Created MBDFullPipelineModule(use_cot=%s)", not args.no_cot)

    # ── Load cached extractor (if requested) ───────────────────────
    if args.load_extractor:
        extractor_path = args.load_extractor
        if os.path.exists(extractor_path):
            generator.load(extractor_path)
            print(f"Loaded extractor from: {extractor_path}")
            _run_logger.info("Loaded extractor from: %s", extractor_path)
        else:
            print(f"WARNING: --load-extractor path not found: {extractor_path}")
            _run_logger.warning("load-extractor path not found: %s", extractor_path)

    # ── Verify seed signature ──────────────────────────────────────
    _print_signature_banner(generator, _run_logger)

    # ── Save seed state ────────────────────────────────────────────
    seed_state_path = os.path.join(log_dir, f"{log_prefix}_mbd_full_run_{current_time}_seed_state.json")
    with open(seed_state_path, "w", encoding="utf-8") as _fp:
        json.dump(generator.dump_state(), _fp, indent=2, ensure_ascii=False)
    print(f"Seed state saved -> {seed_state_path}")
    _run_logger.info("Seed state saved: %s", seed_state_path)

    # ── Evaluation setup ───────────────────────────────────────────
    evaluate = dspy.Evaluate(
        devset=val_set,
        metric=mbd_testcase_metric,
        num_threads=args.num_threads,
        display_progress=True,
        display_table=5,
    )

    # ── Baseline evaluation ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION (Human Prompt Instructions — val set)")
    print("=" * 70)
    _run_logger.info("Starting baseline evaluation on val set (%d examples)", len(val_set))
    t0 = time.time()
    with track_usage() as _baseline_usage:
        baseline_result = evaluate(generator, devset=val_set)
    baseline_duration = time.time() - t0
    baseline_score = baseline_result.score
    baseline_token_usage = _baseline_usage.get_total_tokens()
    print(f"Baseline score: {baseline_score:.1f}%  ({baseline_duration:.1f}s)")
    _log_token_usage("Baseline", baseline_token_usage, _run_logger)
    _run_logger.info(
        "Baseline evaluation complete: score=%.2f%% duration=%.1fs",
        baseline_score, baseline_duration,
    )

    # ── eval-only mode ─────────────────────────────────────────────
    if args.eval_only:
        print("\n--eval-only: stopping after baseline.")
        _run_logger.info("eval-only mode — printing per-example scores then exiting")
        _print_per_example_scores(generator, val_set, logger=_run_logger)
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
        _restore_streams(sys, _tee_stdout, _tee_stderr, _log_file)
        return

    # ── eval-test mode ─────────────────────────────────────────────
    if args.eval_test:
        print("\n" + "=" * 70)
        print("TEST SET EVALUATION (held-out)")
        print("=" * 70)
        if len(test_set) == 0:
            print("ERROR: test_set is empty — cannot evaluate. Re-run with more tasks (--use-all-tasks).")
            _run_logger.error("test_set is empty — cannot run --eval-test")
            _restore_streams(sys, _tee_stdout, _tee_stderr, _log_file)
            sys.exit(1)
        _run_logger.info("Starting test evaluation (%d examples)", len(test_set))
        test_result = evaluate(generator, devset=test_set)
        test_score = test_result.score
        print(f"Test score: {test_score:.1f}%")
        print(f"Generalization gap vs baseline: {abs(baseline_score - test_score):.1f}%")
        _run_logger.info("Test evaluation: score=%.2f%%", test_score)
        _restore_streams(sys, _tee_stdout, _tee_stderr, _log_file)
        return

    # ── GEPA optimization ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"GEPA OPTIMIZATION (auto={args.gepa_auto}, num_threads={args.num_threads})")
    print("=" * 70)
    _run_logger.info(
        "Starting GEPA optimization: auto=%s num_threads=%d train=%d val=%d",
        args.gepa_auto, args.num_threads, len(train_set), len(val_set),
    )

    cache_dir = ".cache"
    os.makedirs(cache_dir, exist_ok=True)
    gepa_log_dir = os.path.join(cache_dir, f"gepa_mbd_full_{current_time}")

    strategy = create_gepa_strategy(
        name=f"GEPA_MBD_Full_{args.gepa_auto.capitalize()}",
        description=f"GEPA ({args.gepa_auto}) — reflective prompt optimization for MBD-UTA full pipeline",
        auto=args.gepa_auto,
        num_threads=args.num_threads,
        track_stats=True,
        use_merge=True,
        log_dir=gepa_log_dir,
    )
    _run_logger.info(
        "GEPA strategy: name=%s auto=%s log_dir=%s",
        strategy.name, args.gepa_auto, gepa_log_dir,
    )

    runner = OptimizationRunner(
        metric=mbd_testcase_metric,
        student_model=generator,
        train_set=train_set,
        val_set=val_set,
        evaluate_fn=evaluate,
        metric_with_feedback=mbd_metric_with_feedback,
        reflection_lm=reflectin_lm,
    )

    optimized_generator = None
    best_score = baseline_score
    try:
        optimized_generator, best_score = runner.run_optimization([strategy])
    except Exception as exc:
        failed_at = datetime.now().isoformat(timespec="seconds")
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
        _restore_streams(sys, _tee_stdout, _tee_stderr, _log_file)
        err_path = os.path.join(log_dir, f"{log_prefix}_mbd_full_run_{current_time}_error_summary.json")
        with open(err_path, "w", encoding="utf-8") as _fp:
            json.dump(failure_summary, _fp, indent=2, ensure_ascii=False)
        print(f"\nOptimization FAILED. Error summary: {err_path}", file=sys.__stdout__)
        print(f"Seed state preserved at: {seed_state_path}", file=sys.__stdout__)
        raise SystemExit(1) from exc

    if optimized_generator is None:
        print("No strategy beat the baseline — using unoptimized generator.")
        _run_logger.info("No strategy beat baseline — using unoptimized generator")
        optimized_generator = generator

    # ── Print optimized instruction ────────────────────────────────
    _print_signature_banner(optimized_generator, _run_logger, label="OPTIMIZED INSTRUCTION")

    # ── Save optimized state ───────────────────────────────────────
    optimized_state_path = os.path.join(
        log_dir, f"{log_prefix}_mbd_full_run_{current_time}_optimized_state.json"
    )
    with open(optimized_state_path, "w", encoding="utf-8") as _fp:
        json.dump(optimized_generator.dump_state(), _fp, indent=2, ensure_ascii=False)
    print(f"Optimized state saved -> {optimized_state_path}")
    _run_logger.info("Optimized state saved: %s", optimized_state_path)

    # ── Save model .json ───────────────────────────────────────────
    best_strategy_name = getattr(runner, "best_strategy_name", strategy.name)
    model_path = os.path.join(
        cache_dir,
        f"{model_tag}_{best_strategy_name.replace(' ', '_')}_{cot_tag}_mbd_full_optimized_{current_time}.json",
    )
    optimized_generator.save(model_path)
    print(f"\nSaved optimized model: {model_path}")
    _run_logger.info("Saved optimized model: %s", model_path)

    # ── Comparison ────────────────────────────────────────────────
    improvement = best_score - baseline_score
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"  Baseline:   {baseline_score:.1f}%")
    print(f"  Optimized:  {best_score:.1f}%")
    print(f"  Improvement:{improvement:+.1f}%")
    _run_logger.info(
        "RESULTS — baseline=%.2f%% optimized=%.2f%% improvement=%+.2f%%",
        baseline_score, best_score, improvement,
    )

    # ── Test set evaluation ────────────────────────────────────────
    test_token_usage: dict = {}
    if len(test_set) > 0:
        print("\n" + "=" * 70)
        print("TEST SET EVALUATION (held-out)")
        print("=" * 70)
        _run_logger.info("Starting test evaluation (%d examples)", len(test_set))
        with track_usage() as _test_usage:
            test_result = evaluate(optimized_generator, devset=test_set)
        test_token_usage = _test_usage.get_total_tokens()
        test_score = test_result.score
        print(f"  Test score:           {test_score:.1f}%")
        print(f"  Generalization gap:   {abs(best_score - test_score):.1f}%")
        _log_token_usage("TestSet", test_token_usage, _run_logger)
        _run_logger.info(
            "Test evaluation: score=%.2f%% gap=%.2f%%",
            test_score, abs(best_score - test_score),
        )
    else:
        print("\nTest set is empty — skipping test evaluation.")
        test_score = None
        _run_logger.warning("Skipping test evaluation — test_set is empty")

    # ── Run summary JSON ───────────────────────────────────────────
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
    )

    print(f"\n{'='*70}")
    print("MBD Full Pipeline complete!")
    print(f"  Model saved:    {model_path}")
    print(f"  Log file:       {log_path}")
    print(f"{'='*70}\n")
    _run_logger.info("MBD full pipeline complete.")

    _restore_streams(sys, _tee_stdout, _tee_stderr, _log_file)


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════
def _print_signature_banner(generator: MBDFullPipelineModule, run_logger, label: str = "SEED INSTRUCTION"):
    """Print current signature instructions from all predictors."""
    print(f"\n{'='*70}")
    print(f"[{label}] — instructions from all predictors:")
    print(f"{'='*70}")
    for name, pred in generator.named_predictors():
        instr = pred.signature.instructions if hasattr(pred, "signature") else "(no signature)"
        print(f"\n  [{name}]")
        print(f"  {instr[:500]}")
        run_logger.info("[%s] %s — %d chars", label, name, len(instr or ""))
    print(f"{'='*70}\n")


def _save_run_summary(
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
):
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
        "cot": not args.no_cot,
        "num_threads": args.num_threads,
        "baseline_score": baseline_score,
        "best_score": best_score,
        "best_strategy": best_strategy,
        "improvement": round(best_score - baseline_score, 4),
        "test_score": test_score,
        "saved_model_path": saved_model_path,
        "seed_state_path": seed_state_path,
        "optimized_state_path": optimized_state_path,
        "runner_summary": runner_summary or {},
        "token_usage": {
            "baseline_eval": baseline_token_usage or {},
            "test_eval": test_token_usage or {},
        },
    }
    summary_path = os.path.join(log_dir, f"{log_prefix}_mbd_full_run_{current_time}_run_summary.json")
    cache_summary_path = os.path.join(".cache", f"{log_prefix}_mbd_full_run_{current_time}_run_summary.json")
    for path in (summary_path, cache_summary_path):
        try:
            with open(path, "w", encoding="utf-8") as _fp:
                json.dump(summary, _fp, indent=2, ensure_ascii=False)
            run_logger.info("Run summary saved: %s", path)
        except OSError as e:
            run_logger.warning("Could not save run summary %s: %s", path, e)
    print(f"Run summary saved -> {summary_path}")


def _log_token_usage(label: str, usage: dict, run_logger) -> None:
    if not usage:
        return
    total_prompt = 0
    total_completion = 0
    for model, stats in usage.items():
        pt = stats.get("prompt_tokens", 0) or 0
        ct = stats.get("completion_tokens", 0) or 0
        total_prompt += pt
        total_completion += ct
        msg = f"  [{label}] {model}: prompt={pt:,} completion={ct:,} total={pt+ct:,}"
        print(msg)
        run_logger.info(msg.strip())
    if len(usage) > 1:
        msg = f"  [{label}] TOTAL: prompt={total_prompt:,} completion={total_completion:,} total={total_prompt+total_completion:,}"
        print(msg)
        run_logger.info(msg.strip())


def _restore_streams(sys_module, tee_stdout, tee_stderr, log_file):
    sys_module.stdout = tee_stdout._original
    sys_module.stderr = tee_stderr._original
    log_file.close()


if __name__ == "__main__":
    main()
