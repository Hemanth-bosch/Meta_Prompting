r"""[wrapper-only] TRUE MIDAS (run.py) M1-M8 baseline for the GEPA val set — NO run.py modification.

WHY: the GEPA/apples numbers come from DSPy predictors, whose adapter reformats every prompt (system
preamble + [[## field ##]] markers). This script instead generates test cases with run.py's OWN prompt
builders + transport (temp-0 `call_llm`, SYSTEM_PROMPT_TESTCASE), then scores with the SAME wrapper M1-M8
metric — so we learn what MIDAS's original prompts actually score, comparable to the GEPA-optimized number.

It IMPORTS and CALLS run.py's functions only; it never edits anything under MBD/weakspot_code_to_run_2/.
It mirrors extract_mbd_full.MBDFullPipelineModule.forward()'s bridge/solve/post-process body VERBATIM,
swapping ONLY the 3 DSPy predictor calls for run.py's builder + call_llm + run.py's parser.

Run:  .meta-prompting\Scripts\python.exe _true_midas_baseline.py --limit 1 --repeats 1   (S1 smoke)
      .meta-prompting\Scripts\python.exe _true_midas_baseline.py --repeats 3 --judge-cache-off  (full val)
"""
import os
# [wrapper-only] MUST set the generation model BEFORE any MIDAS import — run.py's llm_client reads
# cfg.AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT","gpt-4o-mini") AT IMPORT. This env-only
# override (process-local; does not touch .env or MIDAS) makes the baseline generate with gpt-4.1-mini so
# the comparison isolates the PROMPT effect vs the GEPA run (same model).
os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
os.environ.setdefault("MBD_M8_FULL_RESOLVE", "1")   # [wrapper-only] M8 full-chain resolve ON (match apples/GEPA)

import sys
import re
import json
import asyncio
import argparse
import logging
import statistics
import traceback
from datetime import datetime

import dotenv; dotenv.load_dotenv(override=True)

logging.basicConfig(level=logging.WARNING)
for _n in ("pipeline.function_caller", "LiteLLM", "litellm", "httpx", "dspy", "llm_client", "openai"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

import config  # noqa: F401  [wrapper-only] pin the REPO config before mbd_data swaps `config` to the MBD one
import dspy
from config import reconfigure_lm  # [wrapper-only] repo LM builder (for the M4 judge) — grabbed BEFORE the swap

# [wrapper-only] mbd_data import triggers the MBD sys.path/config swap; it also re-exports run.py Phase 1/2.
from utils.mbd_data import (
    load_mbd_dataset, all_fc_entries, stratified_split_mbd_dataset,
    _locate_weakspot, _build_equation_context,   # run.py Phase 1/2 (imported by mbd_data from run.py)
)
from extract_mbd_full import MBDFullPipelineModule, prepare_full_examples, _extract_dll_lenient

# ── run.py transport + builders + parsers (MBD now on sys.path) ──────────────────────────────────
from llm_client import call_llm  # run.py:454/561 async Azure call (system=SYSTEM_PROMPT_TESTCASE, temp 0)
from prompts.context_prompt import generate_context_prompt        # run.py:445 Prompt 1 builder
from prompts.testcase_prompt import generate_testcase_prompt      # run.py:475 Prompt 2 builder
from prompts.equation_solution_prompt_dll import (
    generate_equation_solution_prompt_dll, build_retry_instruction,  # run.py:550/614 Prompt 4 builder + retry
)
from pipeline.context_helpers import get_weakspot_example_output  # run.py:472 Prompt-2 example block
from pipeline.test_generator import (
    generate_and_parse_test_cases, update_test_cases, extract_and_replace_equations,  # run.py:486/489/490
)
from pipeline.equation_solver import (
    extract_json_from_text_dll, update_equation_values, map_equations_to_json,  # run.py:564/653/654
)
from pipeline.equation_solver_dll import (
    _attempt_solve, _build_substitution_dict, resolve_equation_block, generate_bodmas_chain,  # run.py:581/582/594/597
)
from pipeline.postprocess import (
    clean_deduplicated_solution, compute_unselected_mux_variables, deduplicate_actions,
    flatten_precondition_and_actions, remove_unselected_mux_actions, remove_variable_and_parameter_blocks,
)
from pipeline.tree_to_string import build_op_mapping  # run.py:662 op_map from dependency_mapping
from pipeline.types import MergedEquation  # run.py:531 MergedEquation
# raw-data loaders for Strategy-B eq_ctx rebuild (same calls load_mbd_dataset uses)
from pipeline.input_loader import (
    load_equations, load_signal_sheet, load_model_interface, get_filtered_dfs,
    get_const_default_values, extract_step_time,
)
from pipeline.graph import build_global_ops, fix_operation_instance_names, get_sequence_by_name

# ── the SAME metric the wrapper/GEPA use ─────────────────────────────────────────────────────────
import evaluation.mbd_metric as mbd_metric
from evaluation.mbd_metric import (
    _compute_all_components, solve_with_dll_timeout, _DLLSolveTimeout,
    WEIGHT_M1_IO, WEIGHT_M2_SIGNAL, WEIGHT_M3_STRUCTURAL, WEIGHT_M4_COVERAGE,
    WEIGHT_M7_DLL, WEIGHT_M8_FUNCTIONAL, M1_IO_CHECK_WEIGHT, M3_OVERLAP_WEIGHT, M8_GATE_ON_WRONG_IO,
)
import utils.kb_parser as kb
_REAL_NEW_KB = dict(kb.NEW_WEAKSPOT_GUIDELINES)  # [wrapper-only] the BU new KB (for the M4 judge)
from utils.required_io import set_kb_path as _set_kb_path

W = [("m1", WEIGHT_M1_IO), ("m2", WEIGHT_M2_SIGNAL), ("m3", WEIGHT_M3_STRUCTURAL),
     ("m4", WEIGHT_M4_COVERAGE), ("m7", WEIGHT_M7_DLL), ("m8", WEIGHT_M8_FUNCTIONAL)]
COMPONENTS = [k for k, _ in W]

# [wrapper-only] MBD data root (where load_mbd_dataset reads the per-FC folders)
_MBD_DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "MBD", "weakspot_code_to_run_2", "data")

_eq_ctx_raw_cache = {}   # [wrapper-only] cache (eqn_dict, const_values, data, global_ops) per (fc, leaf)


def _composite(comp):  # [wrapper-only] 0..1 weighted composite
    return sum(w * float(comp.get(k) or 0.0) for k, w in W)


def build_eq_ctx(fc_name, leaf_node, weakspot_id):
    """[wrapper-only] Strategy B: re-run run.py Phase 1/2 LIVE for one weakspot (same calls as
    load_mbd_dataset), returning the real eq_ctx run.py's builders consume. No MIDAS edit."""
    key = (fc_name, leaf_node)
    if key not in _eq_ctx_raw_cache:
        fc_dir = os.path.join(_MBD_DATA_ROOT, fc_name)
        eqn_dict = load_equations(fc_dir)                                   # run.py:754
        signal_df = load_signal_sheet(fc_dir)                              # run.py:755
        model_interface = load_model_interface(fc_dir)                     # run.py:756
        _, _, const_df, _ = get_filtered_dfs(signal_df)                    # run.py:781
        step_time = extract_step_time(leaf_node)                          # run.py:784
        const_values = {k: v for k, v in                                  # run.py:795-796 (drop None)
                        get_const_default_values(const_df, step_time, model_interface).items()
                        if v is not None}
        data = get_sequence_by_name(eqn_dict, leaf_node)                  # run.py:829
        fix_operation_instance_names(data)                               # run.py:830
        global_ops = build_global_ops(data)                             # run.py:831
        _eq_ctx_raw_cache[key] = (data, global_ops, const_values)
    data, global_ops, const_values = _eq_ctx_raw_cache[key]
    selection = _locate_weakspot(data, weakspot_id)                       # run.py:843
    if selection is None:
        return None
    return _build_equation_context(data, global_ops, selection, const_values)  # run.py:848


async def generate_one(ex, eq_ctx, max_retries, m7_ref, lenient):
    """[wrapper-only] MIDAS-faithful pipeline for one val weakspot. Mirrors forward()'s body
    (extract_mbd_full.py:461-794) EXACTLY, swapping ONLY the 3 DSPy predictor calls for run.py's
    builder + call_llm + run.py parser. Returns a metric-compatible dspy.Prediction."""
    # metric inputs off the Example (same fields forward reads)
    dll_path = getattr(ex, "dll_path", "")
    signalsheet_data = getattr(ex, "signalsheet_data", "")
    model_interface_data = getattr(ex, "model_interface_data", "")
    op_map_json = getattr(ex, "op_map_json", "{}")

    _extract = _extract_dll_lenient if lenient else extract_json_from_text_dll  # run.py-exact by default

    # ── Prompt 1 (run.py:445,454) ── raw call_llm string, no CoT split (stricter than forward) ──
    context_analysis = await call_llm(generate_context_prompt(eq_ctx.filtered_dict))

    # ── Prompt 2 (run.py:472-486) ── run.py builder + generate_and_parse_test_cases (captures raw) ──
    example_output = get_weakspot_example_output(eq_ctx.filtered_dict, eq_ctx.weakspot_op_key)
    tc_prompt = generate_testcase_prompt(context_analysis, eq_ctx.filtered_dict,
                                         example_output, eq_ctx.instance_names)
    try:
        test_cases, test_cases_raw = await generate_and_parse_test_cases(tc_prompt)  # run.py:486
    except Exception as _p2_err:  # run.py's parser re-raises after 5 internal retries (run.py would crash)
        return dspy.Prediction(
            context_analysis=context_analysis,
            test_cases_pre_postprocess="",
            test_cases="[]",
            m7_score=None,
            m7_details=f"Prompt 2 parse failed after run.py retries: {_p2_err}",
            _parse_failed=True,
        )

    # ── Bridge + post-process + Prompt-4 solve — copy of forward() with the Prompt-4 transport swap ──
    relations = eq_ctx.relations                                          # run.py:429 (LIVE object)
    equation_results_raw = eq_ctx.equation_results or []                  # run.py:430 (LIVE list of EquationResult)
    updated_test_cases = update_test_cases(test_cases, relations)         # run.py:489
    updated_json, equation_dict = extract_and_replace_equations(updated_test_cases)  # run.py:490

    # run.py:493-499 filtered_const_values (AFTER extract_and_replace_equations)
    const_dict = eq_ctx.filtered_dict.get("const_values", {}) or {}       # run.py:497 eq_ctx.filtered_dict["const_values"]
    all_equations = " ".join(equation_dict.values())                     # run.py:493
    _uc = re.findall(r"SystemConstant\((.*?)\)", all_equations)          # run.py:494
    unique_constants = list(set(_uc))                                    # run.py:495
    filtered_const_values = {k: v for k, v in const_dict.items() if k in unique_constants}  # run.py:496-499
    eq_ctx.filtered_const_values = filtered_const_values                 # run.py:496 (mutate eq_ctx)

    function_caller = MBDFullPipelineModule._get_function_caller(dll_path, signalsheet_data, model_interface_data)  # run.py:821-826

    # run.py:521-537 merged_equation_map
    original_eq_map = {r["original_equation"]: r for r in equation_results_raw
                       if isinstance(r, dict) and r.get("original_equation", "").strip()}
    # equation_results are EquationResult objects (LIVE) → normalize to dicts for uniform access
    def _er(r, k):  # [wrapper-only] read attr or dict key
        return getattr(r, k, None) if not isinstance(r, dict) else r.get(k)
    original_eq_map = {}
    for r in equation_results_raw:
        _oe = _er(r, "original_equation")
        if _oe and str(_oe).strip():
            original_eq_map[_oe] = r
    merged_equation_map = {}
    for uid, full_eq in equation_dict.items():
        lhs = full_eq.split("=")[0].strip()
        if lhs in original_eq_map:
            orig = original_eq_map[lhs]
            merged_equation_map[uid] = MergedEquation(
                uid=uid, equation_to_solve=full_eq, original_equation=lhs,
                assignment_chain=_er(orig, "assignment_chain"),
                final_substituted_equation=_er(orig, "final_substituted_equation"),
            )

    updated_dict = {}
    is_first_pass = True
    locked_param_block = ""
    all_unselected_vars = set()
    substitution_dict = {}
    eq_m7_scores = []

    def _substitute_ops(equation, subs):  # run.py:584-588
        return re.sub(r"\$op\d+", lambda m: subs.get(m.group(0), m.group(0)), equation)

    for eq_id, merged_eq in merged_equation_map.items():
        eq_to_solve = merged_eq.assignment_chain.split("\n")[::-1]        # run.py:546
        lhs_value = merged_eq.equation_to_solve.split("=")[-1].strip()   # run.py:547

        # ── Prompt 4 first attempt (run.py:550-564) — run.py builder + call_llm + run.py parser ──
        solve_prompt = generate_equation_solution_prompt_dll(eq_to_solve, lhs_value, filtered_const_values, locked_param_block)
        solve_response = await call_llm(solve_prompt)
        _raw_json = _extract(solve_response)
        simulated_response = _raw_json or []

        if is_first_pass and simulated_response:                          # run.py:567-579 locked params
            locked = {p.split(" = ")[0]: p.split(" = ")[1] for p in simulated_response
                      if p.startswith("Parameter(") and " = " in p}
            if locked:
                locked_param_block = ("## Locked parameter values -- do not change on retry:\n"
                                      + "\n".join(f"{k} = {v}" for k, v in locked.items()) + "\n\n")
            is_first_pass = False

        if function_caller is not None:
            try:  # DLL watchdog (wrapper safety; run.py has none) — a hang → M7=0 for this eq, keep going
                attempt = solve_with_dll_timeout(lambda: _attempt_solve(simulated_response, eq_to_solve, function_caller), function_caller)  # run.py:581
            except _DLLSolveTimeout:
                eq_m7_scores.append(0.0); updated_dict[eq_id] = simulated_response; continue
            substitution_dict = _build_substitution_dict(attempt.final_assignments)  # run.py:582
            eq_list = generate_bodmas_chain(_substitute_ops(merged_eq.final_substituted_equation, substitution_dict))  # run.py:591-594
            all_unselected_vars |= compute_unselected_mux_variables(eq_list, substitution_dict)  # run.py:596
            try:
                final_check, try_fc = solve_with_dll_timeout(lambda: resolve_equation_block(eq_list, function_caller), function_caller)  # run.py:597
            except _DLLSolveTimeout:
                eq_m7_scores.append(0.0); updated_dict[eq_id] = simulated_response; continue
            expected = merged_eq.equation_to_solve.split("=")[-1].strip()  # run.py:599
            actual = final_check[-1].split("=")[-1].strip() if final_check else ""  # run.py:600
            n_retries_used = 0

            for retry_num in range(max(12, max_retries + 2)):            # run.py:608 range(12)
                if expected == actual:
                    break                                               # run.py:611 PASS
                if retry_num < max_retries:                             # run.py:612 (parameterized)
                    n_retries_used += 1
                    # ── Prompt 4 RETRY (run.py:614-627) — run.py's exact retry: build_retry_instruction
                    #    injected into locked_param_block slot, then call_llm + run.py parser ──
                    extra = build_retry_instruction(expected, actual, try_fc, eq_list, attempt.solved_equations)  # run.py:614
                    retry_prompt = generate_equation_solution_prompt_dll(eq_to_solve, lhs_value, filtered_const_values, locked_param_block + extra)  # run.py:617-620
                    solve_response = await call_llm(retry_prompt)        # run.py:624
                    _raw_json = _extract(solve_response)                 # run.py:627
                    simulated_response = _raw_json or []
                    try:
                        attempt = solve_with_dll_timeout(lambda: _attempt_solve(simulated_response, eq_to_solve, function_caller), function_caller)  # run.py:628
                        substitution_dict = _build_substitution_dict(attempt.final_assignments)  # run.py:629
                        eq_list = generate_bodmas_chain(_substitute_ops(merged_eq.final_substituted_equation, substitution_dict))  # run.py:630-632
                        final_check, try_fc = solve_with_dll_timeout(lambda: resolve_equation_block(eq_list, function_caller), function_caller)  # run.py:633
                    except _DLLSolveTimeout:
                        actual = ""; break
                    actual = final_check[-1].split("=")[-1].strip() if final_check else ""  # run.py:634
                else:
                    break                                               # run.py:637 give up

            # M7 inline graded score (wrapper metric signal; run.py computes none)
            if expected == actual:
                _ref = m7_ref if m7_ref is not None else max(4, int(max_retries or 0))
                eq_m7_scores.append(max(0.0, 1.0 - n_retries_used / (_ref + 1)))
            else:
                eq_m7_scores.append(0.0)

        updated_dict[eq_id] = simulated_response                        # run.py:639

    # M7 inline summary (must be before post-process)
    if eq_m7_scores:
        m7_inline_score = sum(eq_m7_scores) / len(eq_m7_scores)
        n_first = sum(1 for s in eq_m7_scores if s == 1.0)
        n_retry = sum(1 for s in eq_m7_scores if 0.0 < s < 1.0)
        n_fail = sum(1 for s in eq_m7_scores if s == 0.0)
        m7_inline_details = f"DLL inline: {n_first} first-try + {n_retry} retry-pass + {n_fail} fail / {len(eq_m7_scores)} eqs"
    else:
        m7_inline_score = None
        m7_inline_details = "DLL unavailable — no inline M7"

    # ── Phase 4 branch (run.py:876-888) ── post-process chain or straight-through ──
    if len(equation_dict) != 0:
        update_equation_values(equation_dict, updated_dict)              # run.py:653
        pre_final = map_equations_to_json(updated_json, equation_dict)   # run.py:654
        cleaned = remove_variable_and_parameter_blocks(pre_final)        # run.py:655
        flattened = flatten_precondition_and_actions(cleaned)           # run.py:656
        deduped = deduplicate_actions(flattened)                        # run.py:657
        if merged_equation_map:                                        # run.py:659-665 2nd unselected-mux
            last_eq = list(merged_equation_map.values())[-1]
            last_eq_to_solve = last_eq.assignment_chain.split("\n")[::-1]
            op_map = build_op_mapping(eq_ctx.dependency_mapping)        # run.py:662 (LIVE, not serialized)
            all_unselected_vars |= compute_unselected_mux_variables(last_eq_to_solve, substitution_dict, op_map)
        final_json = clean_deduplicated_solution(deduped)              # run.py:667
        final_json = remove_unselected_mux_actions(final_json, all_unselected_vars)  # run.py:668
    else:
        final_json = map_equations_to_json(updated_json, equation_dict)  # run.py:888

    return dspy.Prediction(
        context_analysis=context_analysis,
        test_cases_pre_postprocess=test_cases_raw,      # Prompt-2 raw → M1/M3/M4
        test_cases=json.dumps(final_json),               # final → M2/M7/M8
        m7_score=m7_inline_score,
        m7_details=m7_inline_details,
    )


def _set_m4_kb(use_new: bool):  # [wrapper-only] toggle the M4 judge's KB (apples V1=old, V2/V4=new)
    kb.NEW_WEAKSPOT_GUIDELINES = _REAL_NEW_KB if use_new else {}


def score_one(ex, prediction):  # [wrapper-only] SAME metric as GEPA/apples
    comp = _compute_all_components(ex, prediction, use_pre_postprocess=True)
    comp["composite"] = 100.0 * _composite(comp)
    return comp


def _exkey(ex):
    return f"{getattr(ex,'fc_name','?')}#{getattr(ex,'weakspot_id','?')}"


def _ms(vals):
    return (statistics.mean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser(description="TRUE MIDAS (run.py) M1-M8 baseline for the GEPA val set.")
    ap.add_argument("--judge-model", default=os.getenv("MODEL_NAME", "azure/gpt-4.1-mini"),
                    help="Global DSPy LM (needed so the M4 judge predictor can build; the judge itself pins azure/gpt-5.2). Does NOT affect generation.")
    ap.add_argument("--gen-model", default="gpt-4.1-mini",
                    help="Azure DEPLOYMENT name run.py's call_llm generates with (via AZURE_OPENAI_DEPLOYMENT).")
    ap.add_argument("--max-retries", type=int, default=2, help="Prompt-4 DLL retry budget (2 matches GEPA; 1 = run.py native).")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--judge-cache-off", action="store_true", help="Clear the M4 judge cache each repeat -> M4-noise band (generation stays deterministic).")
    ap.add_argument("--split", choices=["val", "train", "all"], default="val")
    ap.add_argument("--limit", type=int, default=0, help="0 = all examples in the split.")
    ap.add_argument("--required-io-path", default="utils/required_io.json")
    ap.add_argument("--m7-reference-retries", type=int, default=4)
    ap.add_argument("--lenient-solver-json", action="store_true", help="Use the wrapper's lenient DLL JSON extractor instead of run.py's fence-only one.")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    # generation model (env override read by run.py's cfg — already set at import; re-affirm for clarity)
    os.environ["AZURE_OPENAI_DEPLOYMENT"] = args.gen_model

    # M4 judge LM + KB (scoring parity with apples/GEPA)
    reconfigure_lm(args.judge_model, cache=False)      # global DSPy LM (judge builds its own gpt-5.2)
    if args.required_io_path:
        _set_kb_path(args.required_io_path); os.environ["MBD_REQUIRED_IO_PATH"] = args.required_io_path
    _set_m4_kb(True)                                   # new BU KB for the M4 judge

    # transport sanity (S0)
    import llm_client as _llmc
    if not getattr(_llmc.cfg, "AZURE_OPENAI_ENDPOINT", ""):
        print("FATAL: AZURE_OPENAI_ENDPOINT not set — run `az login` + set the endpoint in .env."); sys.exit(2)

    RULES = {
        "composite_weights": {k: w for k, w in W}, "M1_IO_CHECK_WEIGHT": M1_IO_CHECK_WEIGHT,
        "M3_OVERLAP_WEIGHT": M3_OVERLAP_WEIGHT, "M8_GATE_ON_WRONG_IO": M8_GATE_ON_WRONG_IO,
        "m7_reference_retries": args.m7_reference_retries, "max_retries": args.max_retries,
        "gen_model": args.gen_model, "M4_judge_model": os.getenv("M4_JUDGE_MODEL_NAME", "azure/gpt-5.2"),
        "M8_FULL_RESOLVE": os.getenv("MBD_M8_FULL_RESOLVE", "1") not in ("0", "false", "False"),
        "required_io_path": args.required_io_path, "temperature": "0.0 (run.py hardcoded)",
        "transport": "run.py llm_client.call_llm (system=SYSTEM_PROMPT_TESTCASE)",
        "repeats": args.repeats, "judge_cache_off": args.judge_cache_off,
        "inject_kb": False, "required_io_output_override": False,
        "solver_json": "lenient" if args.lenient_solver_json else "run.py fence-only",
    }
    print("=" * 80)
    print("TRUE MIDAS (run.py) BASELINE — M1-M8 with run.py's own prompts + transport")
    print(f"  gen_model={args.gen_model}  M4_judge={RULES['M4_judge_model']}  max_retries={args.max_retries}  "
          f"m7_ref={args.m7_reference_retries}  repeats={args.repeats}  judge_cache_off={args.judge_cache_off}")
    print(f"  KB={args.required_io_path}  M8_FULL_RESOLVE={RULES['M8_FULL_RESOLVE']}  "
          f"composite weights sum={sum(w for _, w in W):.2f}")
    print("=" * 80)

    # build the SAME dataset + val split GEPA uses
    ds = prepare_full_examples(load_mbd_dataset(fc_entries=all_fc_entries()))
    tr, val, _ = stratified_split_mbd_dataset(ds, train_ratio=0.6, val_ratio=0.4, test_ratio=0.0)
    pool = {"val": list(val), "train": list(tr), "all": list(tr) + list(val)}[args.split]
    examples = pool[:args.limit] if args.limit else pool
    print(f"{args.split} examples: {len(examples)}  ({', '.join(_exkey(e) for e in examples)})\n")

    # precompute eq_ctx per example (deterministic; not an LLM call)
    eq_ctxs = {}
    for ex in examples:
        try:
            eq_ctxs[_exkey(ex)] = build_eq_ctx(ex.fc_name, ex.leaf_node, int(ex.weakspot_id))
        except Exception as e:
            print(f"  [WARN] eq_ctx build failed for {_exkey(ex)}: {e}"); eq_ctxs[_exkey(ex)] = None

    per_rep = []          # list of {metric: mean-over-examples} per repeat
    pex = {}              # exkey -> list of per-repeat composite %
    fails = 0
    for r in range(args.repeats):
        if args.judge_cache_off:  # clear judge memo so M4 re-judges (M4-noise band; generation deterministic)
            mbd_metric._judge_result_cache.clear(); mbd_metric._judge_reasoning_cache.clear()
        print(f"-- repeat {r+1}/{args.repeats} --", flush=True)
        rep_comps = []
        for ei, ex in enumerate(examples, 1):
            k = _exkey(ex)
            eq_ctx = eq_ctxs.get(k)
            if eq_ctx is None:
                print(f"  [{ei}/{len(examples)}] {k}  SKIP (no eq_ctx)", flush=True); continue
            print(f"  [{ei}/{len(examples)}] {k}  running (Prompts 1/2/4)...", flush=True)
            try:
                pred = asyncio.run(generate_one(ex, eq_ctx, args.max_retries, args.m7_reference_retries, args.lenient_solver_json))
                comp = score_one(ex, pred)
                rep_comps.append(comp)
                pex.setdefault(k, []).append(round(comp["composite"], 2))
                print(f"      composite={comp['composite']:.1f}%  " +
                      "  ".join(f"{m}={float(comp.get(m) or 0):.2f}" for m in COMPONENTS), flush=True)
            except Exception as e:
                fails += 1
                print(f"      FAILED: {e}", flush=True); traceback.print_exc()
        if rep_comps:
            rm = {m: statistics.mean([float(c.get(m) or 0.0) for c in rep_comps]) for m in COMPONENTS}
            rm["composite"] = 100.0 * sum(w * rm[m] for m, w in W)
            per_rep.append(rm)

    # aggregate over repeats
    report = {}
    for m in COMPONENTS + ["composite"]:
        vals = [rr[m] for rr in per_rep if m in rr]
        mean, std = _ms(vals) if vals else (0.0, 0.0)
        report[m] = {"mean": mean, "std": std}

    print("\n" + "=" * 80)
    print("RESULT — TRUE MIDAS baseline (mean +/- std over repeats):")
    for m in COMPONENTS:
        print(f"  {m.upper()}: {report[m]['mean']:.3f} +/- {report[m]['std']:.3f}")
    print(f"  COMPOSITE: {report['composite']['mean']:.2f}% +/- {report['composite']['std']:.2f}")
    print("\nPer-example composite% (mean over repeats):")
    for k, lst in pex.items():
        mn, sd = _ms(lst)
        print(f"  {k:34} {mn:6.2f} +/- {sd:4.2f}   (n={len(lst)}, vals={lst})")
    if fails:
        print(f"\n[WARN] {fails} example-repeat(s) FAILED.")

    # write JSON + CSV (mirror apples output)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lbl = f"_{args.label}" if args.label else ""
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apples_compare", "true_midas_baseline")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"true_midas_baseline{lbl}_{ts}")
    out = {
        "config": {"repeats": args.repeats, "gen_model": args.gen_model, "split": args.split,
                   "limit": args.limit, "max_retries": args.max_retries, "rules": RULES},
        "report": report,
        "per_example_composite": pex,
        "raw_per_repeat": per_rep,
        "val_examples": [_exkey(e) for e in examples],
        "fail_count": fails,
    }
    json.dump(out, open(base + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    with open(base + ".csv", "w", encoding="utf-8") as f:
        f.write("metric,mean,std\n")
        for m in COMPONENTS + ["composite"]:
            f.write(f"{m},{report[m]['mean']:.4f},{report[m]['std']:.4f}\n")
    print(f"\nWrote {base}.json and .csv")


if __name__ == "__main__":
    main()
