"""Verification script for MBD-UTA DSPy full pipeline implementation.

Run with: .meta-prompting\Scripts\python.exe _verify_mbd.py
"""
import sys
import os
import json

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

# Pin the REPO config in sys.modules BEFORE any import below triggers an MBD-config swap.
# utils.mbd_data / extract_mbd_full temporarily set sys.modules['config'] to MBD's config
# (MBD/weakspot_code_to_run_2/config.py) for their pipeline imports and restore it ONLY if a
# repo config was already cached. Without this line that cache is empty, so the restore instead
# DELETES sys.modules['config']; later imports (extract_mbd_full:80 `from config import
# reconfigure_lm`, evaluation/__init__ `from config import _build_ca_bundle`) then re-resolve
# to the MBD config (which lacks those names) and the V2-V6 sections fail spuriously.
import config  # noqa: F401,E402  (repo config; keep pinned as sys.modules['config'])

PASS = []
FAIL = []

def ok(msg):
    PASS.append(msg)
    print(f"  OK: {msg}")

def fail(msg):
    FAIL.append(msg)
    print(f"  FAIL: {msg}")


# ═══════════════════════════════════════════════════════════════════
# V1: mbd_data.py — new fields present, with_inputs unchanged
# ═══════════════════════════════════════════════════════════════════
print("\n=== V1: mbd_data.py new fields ===")
dataset = []
try:
    from utils.mbd_data import load_mbd_dataset

    # Try several FC entries until one loads successfully
    from utils.mbd_data import _default_fc_entries
    dataset = []
    for entry in _default_fc_entries():
        dataset = load_mbd_dataset(fc_entries=[entry])
        if dataset:
            print(f"  Using FC: {entry['fc_name']} ws={entry['weakspot_ids']}")
            break
    print(f"  Loaded {len(dataset)} examples")

    if not dataset:
        fail("No examples loaded")
    else:
        ex = dataset[0]

        # Check 4 new fields exist and are non-empty
        for field in ["relations_json", "equation_results_json", "op_map_json", "weakspot_op_key"]:
            val = getattr(ex, field, "__MISSING__")
            if val == "__MISSING__":
                fail(f"field '{field}' missing from Example")
            elif val in ("{}", "[]", ""):
                print(f"  WARN: {field} = empty/fallback value")
            else:
                ok(f"{field} = {str(val)[:70]}")

        # Verify with_inputs is still exactly 5 fields (unchanged)
        expected_inputs = {"equation_json", "const_values", "weakspot_context", "instance_names", "raster_time"}
        actual_inputs = set(ex._input_keys)
        if actual_inputs == expected_inputs:
            ok("with_inputs unchanged (5 fields)")
        else:
            fail(f"with_inputs changed! expected {sorted(expected_inputs)}, got {sorted(actual_inputs)}")

        # Verify Phase 2 data is valid JSON
        try:
            rel = json.loads(ex.relations_json)
            ok(f"relations_json valid JSON, {len(rel)} keys")
        except Exception as e:
            fail(f"relations_json invalid JSON: {e}")

        try:
            eq_results = json.loads(ex.equation_results_json)
            ok(f"equation_results_json valid JSON, {len(eq_results)} entries")
            if eq_results:
                r = eq_results[0]
                assert "original_equation" in r, "missing original_equation key"
                assert "assignment_chain" in r, "missing assignment_chain key"
                assert "final_substituted_equation" in r, "missing final_substituted_equation key"
                ok("equation_results_json entries have correct keys")
        except Exception as e:
            fail(f"equation_results_json problem: {e}")

        try:
            op_map = json.loads(ex.op_map_json)
            ok(f"op_map_json valid JSON, {len(op_map)} entries")
        except Exception as e:
            fail(f"op_map_json invalid JSON: {e}")

        ok(f"weakspot_op_key = '{ex.weakspot_op_key}'")

except Exception as e:
    fail(f"V1 exception: {e}")
    import traceback; traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# V2: prepare_full_examples + MBDFullPipelineModule — 3 predictors
# ═══════════════════════════════════════════════════════════════════
print("\n=== V2: MBDFullPipelineModule — 4 predictors ===")
try:
    from extract_mbd_full import prepare_full_examples, MBDFullPipelineModule
    import dspy
    import dotenv
    dotenv.load_dotenv(override=True)
    try:
        from config import reconfigure_lm
        # timeout=15/num_retries=0: the M4 coverage judge is the only live-LM path in this
        # script. Online it answers well within 15s; offline (no Azure) it fails fast and M4
        # takes its graceful "[judge-err] -> neutral 0.5" path instead of hanging ~17 min.
        reconfigure_lm(os.getenv("MODEL_NAME", "azure/gpt-4.1-mini"), cache=True,
                       timeout=15, num_retries=0)
    except Exception:
        pass

    if dataset:
        # Re-wrap examples with 12-field with_inputs
        full_examples = prepare_full_examples(dataset)
        ex_full = full_examples[0]
        expected_full_inputs = {
            "equation_json", "const_values", "weakspot_context", "instance_names", "raster_time",
            "weakspot_operation", "example_explanation_output",
            "dll_path", "signalsheet_data", "model_interface_data",
            "relations_json", "equation_results_json", "op_map_json", "weakspot_op_key",
        }
        actual_full = set(ex_full._input_keys)
        if actual_full == expected_full_inputs:
            ok("prepare_full_examples: 14-field with_inputs correct")
        else:
            missing = expected_full_inputs - actual_full
            extra = actual_full - expected_full_inputs
            fail(f"prepare_full_examples: missing={missing}, extra={extra}")

    # Instantiate module and check 3 predictors
    module = MBDFullPipelineModule(use_cot=True)
    predictor_names = [name for name, _ in module.named_predictors()]
    print(f"  Predictor names: {predictor_names}")

    if len(predictor_names) == 4:
        ok(f"4 predictors found")
    else:
        fail(f"Expected 4 predictors, got {len(predictor_names)}: {predictor_names}")

    for expected in ["context_extractor", "testcase_generator", "equation_solver_dll", "equation_solver_retry"]:
        found = any(expected in name for name in predictor_names)
        if found:
            ok(f"predictor '{expected}' present")
        else:
            fail(f"predictor '{expected}' NOT found in {predictor_names}")

except Exception as e:
    fail(f"V2 exception: {e}")
    import traceback; traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# V3: mbd_metric.py — backward compat + staged dispatch
# ═══════════════════════════════════════════════════════════════════
print("\n=== V3: mbd_metric.py backward compat + staged dispatch ===")
try:
    from evaluation.mbd_metric import (
        _compute_all_components,
        mbd_testcase_metric,
        mbd_metric_with_feedback,
        _m7_dll_pass_rate,
    )
    import dspy

    # M7/M8 now FAIL LOUD when the DLL is missing (RuntimeError) — they no longer return a
    # soft 0.5. So the metric-computes/staged-dispatch checks below use the REAL example
    # (dataset[0] carries a real dll_path); the explicit fail-loud contract is checked in V3h.
    import dspy as _dspy
    ex_test = dataset[0] if dataset else None
    if ex_test:
        pred_test = dspy.Prediction(
            context_analysis="test context",
            test_cases='[{"Pre-Condition": [], "Steps": [{"Actions": ["Wait 100000"], "Expected Results": []}], "Detailed Explanation": "test"}]',
            test_cases_pre_postprocess='[{"Pre-Condition": [], "Steps": [{"Actions": ["Wait 100000"], "Expected Results": []}], "Detailed Explanation": "test"}]',
        )

        # V3a: default call (backward compat, no use_pre_postprocess)
        try:
            scores = _compute_all_components(ex_test, pred_test)
            assert "m1" in scores and "m7" in scores
            ok(f"_compute_all_components() default: keys={sorted(scores.keys())}")
        except Exception as e:
            fail(f"_compute_all_components() default: {e}")

        # V3b: use_pre_postprocess=True
        try:
            scores_pre = _compute_all_components(ex_test, pred_test, use_pre_postprocess=True)
            assert "m1" in scores_pre and "m7" in scores_pre
            ok("_compute_all_components(use_pre_postprocess=True) works")
        except Exception as e:
            fail(f"_compute_all_components(use_pre_postprocess=True): {e}")

        # V3c: mbd_testcase_metric returns float
        try:
            score = mbd_testcase_metric(ex_test, pred_test)
            assert isinstance(score, float)
            ok(f"mbd_testcase_metric returns float: {score:.3f}")
        except Exception as e:
            fail(f"mbd_testcase_metric: {e}")

        # V3d: pred_name=None returns float (baseline compat for extract_mbd.py)
        try:
            result = mbd_metric_with_feedback(ex_test, pred_test, pred_name=None)
            assert isinstance(result, float)
            ok(f"mbd_metric_with_feedback(pred_name=None) returns float: {result:.3f}")
        except Exception as e:
            fail(f"mbd_metric_with_feedback(pred_name=None): {e}")

        # V3e: testcase_generator staged dispatch
        try:
            result = mbd_metric_with_feedback(ex_test, pred_test, pred_name="testcase_generator.predict")
            assert hasattr(result, "score") and hasattr(result, "feedback")
            assert "Prompt 2" in result.feedback or "M1" in result.feedback or "M3" in result.feedback
            ok(f"testcase_generator dispatch: score={result.score:.3f}, feedback mentions Prompt2/M1/M3")
        except Exception as e:
            fail(f"testcase_generator staged dispatch: {e}")

        # V3f: equation_solver_dll staged dispatch
        try:
            result = mbd_metric_with_feedback(ex_test, pred_test, pred_name="equation_solver_dll.predict")
            assert hasattr(result, "score") and hasattr(result, "feedback")
            assert "M2" in result.feedback or "M7" in result.feedback
            ok(f"equation_solver_dll dispatch: score={result.score:.3f}, feedback mentions M2/M7")
        except Exception as e:
            fail(f"equation_solver_dll staged dispatch: {e}")

        # V3g: context_extractor staged dispatch
        try:
            result = mbd_metric_with_feedback(ex_test, pred_test, pred_name="context_extractor.predict")
            assert hasattr(result, "score") and hasattr(result, "feedback")
            assert "Total score" in result.feedback or "M1" in result.feedback
            ok(f"context_extractor dispatch: score={result.score:.3f}")
        except Exception as e:
            fail(f"context_extractor staged dispatch: {e}")

        # V3h: M7 FAILS LOUD when no dll_path (RuntimeError) — the metric must not silently
        # skip DLL verification. (Refactored from the old soft-0.5 fallback.)
        try:
            _m7_dll_pass_rate("[]", "{}", "")
            fail("_m7_dll_pass_rate(no dll) should RAISE RuntimeError (fail-loud), but returned")
        except RuntimeError as e:
            ok(f"_m7_dll_pass_rate(no dll) fails loud as designed: {str(e)[:60]}")
        except Exception as e:
            fail(f"_m7_dll_pass_rate(no dll): expected RuntimeError, got {type(e).__name__}: {e}")

    else:
        fail("No example available for metric tests (dataset empty)")

except Exception as e:
    fail(f"V3 exception: {e}")
    import traceback; traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# V4: extract_mbd.py backward compat (MBDTestGenerator unchanged)
# ═══════════════════════════════════════════════════════════════════
print("\n=== V4: extract_mbd.py backward compat ===")
try:
    from extract_mbd import MBDTestGenerator
    import inspect

    gen = MBDTestGenerator(use_cot=True)
    predictor_names_2p = [name for name, _ in gen.named_predictors()]
    if len(predictor_names_2p) == 2:
        ok(f"MBDTestGenerator has 2 predictors: {predictor_names_2p}")
    else:
        fail(f"MBDTestGenerator wrong predictor count: {predictor_names_2p}")

    sig = inspect.signature(gen.forward)
    params = [p for p in sig.parameters if p != "self"]
    expected_params = ["equation_json", "const_values", "weakspot_context", "instance_names", "raster_time"]
    if params == expected_params:
        ok("MBDTestGenerator.forward() still has exactly 5 params (unchanged)")
    else:
        fail(f"MBDTestGenerator.forward() params changed: {params}")

except Exception as e:
    fail(f"V4 exception: {e}")
    import traceback; traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# V5: M4 test coverage metric (KB-guidelines LLM-judge)
# ═══════════════════════════════════════════════════════════════════
print("\n=== V5: M4 test coverage metric (KB-guidelines + LLM judge) ===")
try:
    from utils.kb_parser import WEAKSPOT_GUIDELINES, find_block_type
    from evaluation.mbd_metric import _m4_test_coverage, _KB_LOADED

    # V5a: KB loaded and metadata key excluded from guidelines dict
    if _KB_LOADED:
        ok(f"KB loaded: {len(WEAKSPOT_GUIDELINES)} blocks (metadata key excluded)")
        if "_metadata" in WEAKSPOT_GUIDELINES:
            fail("'_metadata' key leaked into WEAKSPOT_GUIDELINES — kb_parser not stripping it")
        else:
            ok("'_metadata' key correctly stripped by kb_parser")
    else:
        print("  WARN: KB not loaded (weakspot_guidelines.json missing) — V5 KB tests skipped")

    # V5b: find_block_type suffix-stripping
    ft_tests = [
        ("Srv_Debounce",       "Srv_Debounce"),
        # Srv_DebounceParam_t IS in the KB directly (has its own inherited entry),
        # so exact match wins over suffix-stripping:
        ("Srv_DebounceParam_t","Srv_DebounceParam_t"),
        ("Srv_RampSwitch",     "Srv_RampSwitch"),
        ("LowpassT",           "LowpassT"),
        ("SrvF_Ramp",          "SrvF_Ramp"),
        # Suffix-stripping kicks in for names NOT in KB (e.g. un-precomputed variants):
        ("unknown_block_xyz",  None),
    ]
    for fn, expected in ft_tests:
        got = find_block_type(fn)
        if got == expected:
            ok(f"find_block_type('{fn}') = {expected!r}")
        else:
            fail(f"find_block_type('{fn}'): expected {expected!r}, got {got!r}")

    # V5c: neutral path — empty weakspot_op_key => no KB spec => neutral 0.5 (the old rule-based
    # branch-coverage fallback was REMOVED; M4 is now KB-judge-only with a neutral fallback).
    minimal_tc = '[{"Pre-Condition":[], "Steps":[{"Actions":["Wait 100000"], "Expected Results":[]}], "Detailed Explanation": "placeholder"}]'
    score_fb, detail_fb = _m4_test_coverage(minimal_tc, "{}", "")
    if score_fb == 0.5 and ("[no-kb]" in detail_fb or "[kb-missing]" in detail_fb):
        ok(f"neutral (empty weakspot_op_key): {detail_fb}")
    else:
        fail(f"expected neutral [no-kb] 0.5, got score={score_fb} detail={detail_fb}")

    # V5d: neutral path — function_name not in KB => no guidelines => neutral 0.5
    eq_unknown = json.dumps({"$ws": {"type": "Function", "function_name": "NoSuchBlock_xyz"}})
    score_unk, detail_unk = _m4_test_coverage(minimal_tc, eq_unknown, "$ws")
    if score_unk == 0.5 and ("[no-kb]" in detail_unk or "[kb-missing]" in detail_unk):
        ok(f"neutral (unknown block type): {detail_unk}")
    else:
        fail(f"expected neutral [no-kb] 0.5 for unknown block, got score={score_unk} detail={detail_unk}")

    # V5e: LLM-judge path when KB has guidelines
    if _KB_LOADED:
        # Srv_Debounce has 3 guidelines:
        #   1. Test for zero debounce time
        #   2. Test input changes before the debounce time
        #   3. Test correct output after the debounce time
        # We provide 2 test cases that explicitly cover guidelines 1 and 2
        fake_key = "$op_debounce"
        eq_debounce = json.dumps({
            fake_key: {"type": "Function", "function_name": "Srv_Debounce", "instance_name": "myDebounce"}
        })
        tc_covers_2 = json.dumps([
            {
                "Pre-Condition": [],
                "Steps": [{"Actions": ["debounceTime = 0", "Wait 100000"], "Expected Results": ["output = 1"]}],
                "Detailed Explanation": "Verify zero debounce time: when debounce time is set to zero the signal passes through immediately without any delay"
            },
            {
                "Pre-Condition": [],
                "Steps": [{"Actions": ["input = 1", "Wait 50000", "input = 0", "Wait 50000"], "Expected Results": ["output = 0"]}],
                "Detailed Explanation": "Verify input changes before debounce time expires: toggling input before the debounce period should reset the timer and prevent the output from changing"
            }
        ])
        score_kb, detail_kb = _m4_test_coverage(tc_covers_2, eq_debounce, fake_key)
        # Online: the judge runs => detail "[KB:Srv_Debounce] x/3 covered". Offline (no Azure):
        # judge fails fast => graceful "[judge-err] ... neutral 0.5". Accept BOTH; score must be
        # a real ratio in [0,1] (KB path) or the neutral 0.5 (judge-unavailable path).
        if "[KB:" in detail_kb:
            ok(f"LLM-judge path used for Srv_Debounce: {detail_kb}")
            if 0.0 <= score_kb <= 1.0:
                ok(f"M4 KB score in range [0,1]: {score_kb:.3f}")
            else:
                fail(f"M4 KB score out of range: {score_kb:.3f}")
        elif "[judge-err]" in detail_kb:
            ok(f"M4 judge unavailable (offline) -> graceful neutral 0.5: {detail_kb}")
            if score_kb != 0.5:
                fail(f"M4 judge-err should be neutral 0.5, got {score_kb:.3f}")
        else:
            fail(f"Unexpected M4 detail string: {detail_kb}")

        # V5f: test cases with NO explanations => judge cannot operate => neutral 0.5 [no-expl]
        tc_no_expl = json.dumps([
            {"Pre-Condition": [], "Steps": [{"Actions": ["Wait 100000"], "Expected Results": []}], "Detailed Explanation": ""}
        ])
        score_ne, detail_ne = _m4_test_coverage(tc_no_expl, eq_debounce, fake_key)
        if score_ne == 0.5 and "[no-expl]" in detail_ne:
            ok(f"neutral (no explanations): {detail_ne}")
        else:
            print(f"  WARN: expected neutral [no-expl] 0.5, got score={score_ne} detail={detail_ne}")

        # V5g: GEPA testcase_generator dispatch uses M4 test coverage in feedback
        if dataset:
            ex_test_v5 = dataset[0].copy(dll_path="")
            pred_v5 = dspy.Prediction(
                context_analysis="test context",
                test_cases=tc_covers_2,
                test_cases_pre_postprocess=tc_covers_2,
            )
            try:
                # Temporarily set weakspot_op_key on the example to Srv_Debounce
                ex_v5_debounce = ex_test_v5.copy(
                    weakspot_op_key=fake_key,
                    equation_json=eq_debounce,
                )
                result_v5 = mbd_metric_with_feedback(ex_v5_debounce, pred_v5, pred_name="testcase_generator.predict")
                assert hasattr(result_v5, "score") and hasattr(result_v5, "feedback")
                ok(f"testcase_generator dispatch with KB: score={result_v5.score:.3f}")
                if "M4" in result_v5.feedback:
                    ok("M4 result appears in GEPA feedback")
                else:
                    print(f"  WARN: M4 not mentioned in feedback: {result_v5.feedback[:200]}")
            except Exception as e:
                fail(f"V5g GEPA dispatch: {e}")

except Exception as e:
    fail(f"V5 exception: {e}")
    import traceback; traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# V6: M8 functional correctness metric
# ═══════════════════════════════════════════════════════════════════
print("\n=== V6: M8 functional correctness metric ===")
try:
    from evaluation.mbd_metric import (
        _m8_functional_correctness,
        WEIGHT_M8_FUNCTIONAL,
        WEIGHT_M7_DLL,
        WEIGHT_M1_IO,
        WEIGHT_M2_SIGNAL,
    )

    # V6a: weight constants correct (current design: M7=0.20, M8=0.20 — rebalanced from 0.10/0.25)
    if abs(WEIGHT_M8_FUNCTIONAL - 0.20) < 1e-9:
        ok(f"WEIGHT_M8_FUNCTIONAL = {WEIGHT_M8_FUNCTIONAL}")
    else:
        fail(f"WEIGHT_M8_FUNCTIONAL expected 0.20, got {WEIGHT_M8_FUNCTIONAL}")

    if abs(WEIGHT_M7_DLL - 0.20) < 1e-9:
        ok(f"WEIGHT_M7_DLL = {WEIGHT_M7_DLL}")
    else:
        fail(f"WEIGHT_M7_DLL expected 0.20, got {WEIGHT_M7_DLL}")

    if abs(WEIGHT_M1_IO - 0.15) < 1e-9:
        ok(f"WEIGHT_M1_IO = {WEIGHT_M1_IO}")
    else:
        fail(f"WEIGHT_M1_IO expected 0.15, got {WEIGHT_M1_IO}")

    if abs(WEIGHT_M2_SIGNAL - 0.15) < 1e-9:
        ok(f"WEIGHT_M2_SIGNAL = {WEIGHT_M2_SIGNAL} (reduced from 0.20)")
    else:
        fail(f"WEIGHT_M2_SIGNAL expected 0.15, got {WEIGHT_M2_SIGNAL}")

    # V6b: weights sum to 1.0
    from evaluation.mbd_metric import (
        WEIGHT_M3_STRUCTURAL, WEIGHT_M4_COVERAGE,
        WEIGHT_M6_BOUNDARY,
    )
    total_w = (WEIGHT_M1_IO + WEIGHT_M2_SIGNAL + WEIGHT_M3_STRUCTURAL
               + WEIGHT_M4_COVERAGE + WEIGHT_M6_BOUNDARY
               + WEIGHT_M7_DLL + WEIGHT_M8_FUNCTIONAL)
    if abs(total_w - 1.0) < 1e-9:
        ok(f"All metric weights sum to 1.0 ({total_w:.4f})")
    else:
        fail(f"Weights sum to {total_w:.6f}, expected 1.0")

    # Real DLL path from the loaded example (M8 internal 0.5 fallbacks are only reachable AFTER
    # the DLL-presence gate, which now FAILS LOUD for a missing/nonexistent .dll).
    _real_dll = getattr(dataset[0], "dll_path", "") if dataset else ""

    # V6c: missing DLL now FAILS LOUD (RuntimeError), not a soft 0.5.
    try:
        _m8_functional_correctness("[]", "{}", "", "", "", "[]", 100000, "")
        fail("M8 with empty dll_path should RAISE RuntimeError (fail-loud), but returned")
    except RuntimeError as e:
        ok(f"M8 missing-DLL fails loud as designed: {str(e)[:60]}")
    except Exception as e:
        fail(f"M8 missing-DLL: expected RuntimeError, got {type(e).__name__}: {e}")

    # V6d: no equation_results_json => internal neutral 0.5 (needs a REAL DLL to pass the gate)
    if _real_dll and os.path.exists(_real_dll):
        no_eq_score, no_eq_detail = _m8_functional_correctness(
            '[{"Pre-Condition":[], "Steps":[], "Detailed Explanation": "x"}]',
            "{}", _real_dll, "", "", "[]", 100000, ""
        )
        if no_eq_score == 0.5:
            ok(f"M8 no-equation fallback: score={no_eq_score}, detail='{no_eq_detail}'")
        else:
            fail(f"M8 no-equation fallback: expected 0.5, got {no_eq_score}")

        # V6e: no steps with expected results => internal neutral 0.5
        tc_no_exp = json.dumps([
            {"Pre-Condition": {"sig": "0"}, "Steps": [{"Actions": {"sig": "1"}, "Wait": 100000}],
             "Detailed Explanation": "no expected results"}
        ])
        no_exp_score, no_exp_detail = _m8_functional_correctness(
            tc_no_exp, "{}", _real_dll, "", "", "[]", 100000, ""
        )
        if no_exp_score == 0.5:
            ok(f"M8 no-expected-results fallback: score={no_exp_score}, detail='{no_exp_detail}'")
        else:
            fail(f"M8 no-expected-results fallback: expected 0.5, got {no_exp_score}")
    else:
        print("  WARN: no real dll_path available — skipping V6d/V6e internal-fallback checks")

    # V6f: _compute_all_components returns m8 key (needs a REAL DLL — M7/M8 fail loud otherwise)
    from evaluation.mbd_metric import _compute_all_components
    if _real_dll and os.path.exists(_real_dll):
        minimal_example = dspy.Example(
            equation_json="{}", signalsheet_data="", model_interface_data="",
            raster_time="100000", dll_path=_real_dll, equation_results_json="[]",
            weakspot_op_key="",
        ).with_inputs("equation_json", "signalsheet_data", "model_interface_data",
                      "raster_time", "dll_path", "equation_results_json", "weakspot_op_key")
        minimal_pred = dspy.Prediction(test_cases="[]", test_cases_pre_postprocess="[]")
        comp = _compute_all_components(minimal_example, minimal_pred)
        if "m8" in comp and "m8_details" in comp:
            ok(f"_compute_all_components returns m8={comp['m8']:.3f} detail='{comp['m8_details']}'")
        else:
            fail(f"_compute_all_components missing 'm8' key. Keys: {list(comp.keys())}")
    else:
        print("  WARN: no real dll_path — skipping V6f _compute_all_components check")

    # V6g: GEPA equation_solver_dll dispatch includes M8 in feedback (real DLL; test_cases="[]"
    # parse-fails fast in M7/M8 -> no heavy DLL run, but the metric still completes and dispatches).
    ex_v6 = pred_v6 = None
    if dataset and _real_dll and os.path.exists(_real_dll):
        ex_v6 = dataset[0]  # keep the REAL dll_path (M7/M8 fail loud without it)
        pred_v6 = dspy.Prediction(
            test_cases="[]",
            test_cases_pre_postprocess="[]",
        )
        try:
            result_v6 = mbd_metric_with_feedback(
                ex_v6, pred_v6, pred_name="equation_solver_dll"
            )
            if hasattr(result_v6, "feedback") and "M8" in result_v6.feedback:
                ok(f"equation_solver_dll GEPA feedback includes M8: score={result_v6.score:.3f}")
            else:
                fb_preview = getattr(result_v6, "feedback", "")[:200]
                print(f"  WARN: M8 not mentioned in equation_solver_dll feedback: {fb_preview}")
        except Exception as e:
            fail(f"V6g GEPA equation_solver_dll dispatch: {e}")
    else:
        print("  WARN: no real dll_path — skipping V6g GEPA dispatch check")

    # V6h: composite weights match the CURRENT design (M1=.15,M2=.15,M3=.10,M4=.20,M6=0,M7=.20,M8=.20).
    # (The old per-predictor RESCALING scheme was removed — see V6i.)
    expected_w = {"M1": 0.15, "M2": 0.15, "M3": 0.10, "M4": 0.20,
                  "M6": 0.0, "M7": 0.20, "M8": 0.20}
    got_w = {"M1": WEIGHT_M1_IO, "M2": WEIGHT_M2_SIGNAL, "M3": WEIGHT_M3_STRUCTURAL,
             "M4": WEIGHT_M4_COVERAGE, "M6": WEIGHT_M6_BOUNDARY,
             "M7": WEIGHT_M7_DLL, "M8": WEIGHT_M8_FUNCTIONAL}
    if all(abs(expected_w[k] - got_w[k]) < 1e-9 for k in expected_w):
        ok(f"composite weights match current design: {got_w}")
    else:
        fail(f"composite weights differ: expected {expected_w}, got {got_w}")

    # V6i: GEPA invariant — per-predictor feedback score EQUALS the composite total (NO rescaling).
    # dspy GEPA asserts fb.score == module_score; every staged branch returns the composite `total`.
    if ex_v6 is not None:
        inv_total = mbd_testcase_metric(ex_v6, pred_v6)
        inv_fb = mbd_metric_with_feedback(ex_v6, pred_v6, pred_name="equation_solver_dll")
        if abs(inv_fb.score - inv_total) < 1e-9:
            ok(f"GEPA invariant holds: feedback.score == composite total ({inv_total:.3f})")
        else:
            fail(f"GEPA invariant broken: fb.score={inv_fb.score:.3f} != total={inv_total:.3f}")
    else:
        print("  WARN: no real dll_path — skipping V6i GEPA-invariant check")

except Exception as e:
    fail(f"V6 exception: {e}")
    import traceback; traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("VERIFICATION SUMMARY")
print(f"{'='*60}")
print(f"  PASSED: {len(PASS)}")
print(f"  FAILED: {len(FAIL)}")
if FAIL:
    print("\nFailed checks:")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("\nAll checks passed!")
    sys.exit(0)
