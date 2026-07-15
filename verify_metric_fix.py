"""
Metric-only verification: replay the last good run's FINAL test cases through
_compute_all_components to verify the M7/M8 fixes — WITHOUT any LLM call.

Loads the real example (equation_json, const_values, relations_json,
equation_results_json, dll_path, ...) and feeds a dspy.Prediction whose
test_cases == the final post-processed JSON produced by the
2026-06-12 11:44 run.  Only the local 32-bit DLL subprocess is exercised
(M2/M7/M8); no network is required.

Usage (project venv):
    .\.meta-prompting\Scripts\python.exe verify_metric_fix.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime

import dotenv
import dspy

from utils.mbd_data import load_mbd_dataset
from extract_mbd_full import prepare_full_examples
from evaluation.mbd_metric import _compute_all_components

dotenv.load_dotenv(override=True)

# Final post-processed test cases from log/e2e_Rail_TempLim_SrvRampSwitch_20260612114419.log
FINAL_TEST_CASES = [
    {"Pre-Condition": ["Rail_stTrqLim = 0", "Rail_trqFlTempLimPCV = 1000", "Rail_trqTempLimMeUn = 2000", "Rail_dtrqLimRmp = 100"],
     "Steps": [{"Actions": ["Wait 100000"], "Expected Results": ["Rail_trqFlTempLim = 1000"]}],
     "Detailed Explanation": "Initial condition with switch=0, output should be at Target A immediately."},
    {"Pre-Condition": ["Rail_stTrqLim = 0", "Rail_trqFlTempLimPCV = 1000", "Rail_trqTempLimMeUn = 2000", "Rail_dtrqLimRmp = 100"],
     "Steps": [{"Actions": ["Rail_stTrqLim = 1", "Wait 100000"], "Expected Results": ["Rail_trqFlTempLim > 1000", "Rail_trqFlTempLim < 2000"]},
               {"Actions": ["Wait 100000"], "Expected Results": ["Rail_trqFlTempLim > 1000", "Rail_trqFlTempLim < 2000"]}],
     "Detailed Explanation": "1000 to Target B (2000) over time."},
    {"Pre-Condition": ["Rail_stTrqLim = 1", "Rail_trqFlTempLimPCV = 1000", "Rail_trqTempLimMeUn = 2000", "Rail_dtrqLimRmp = 100"],
     "Steps": [{"Actions": ["Rail_stTrqLim = 0", "Wait 100000"], "Expected Results": ["Rail_trqFlTempLim < 2000", "Rail_trqFlTempLim > 1000"]},
               {"Actions": ["Wait 100000"], "Expected Results": ["Rail_trqFlTempLim < 2000", "Rail_trqFlTempLim > 1000"]}],
     "Detailed Explanation": "2000 to Target A (1000) over time."},
    {"Pre-Condition": ["Rail_stTrqLim = 0", "Rail_trqFlTempLimPCV = 1500", "Rail_trqTempLimMeUn = 2500", "Rail_dtrqLimRmp = 100"],
     "Steps": [{"Actions": ["Wait 200000"], "Expected Results": ["Rail_trqFlTempLim = 1500"]}],
     "Detailed Explanation": "Holding switch steady at 0 after ramp completion, output remains at Target A."},
    {"Pre-Condition": ["Rail_stTrqLim = 1", "Rail_trqFlTempLimPCV = 1500", "Rail_trqTempLimMeUn = 2500", "Rail_dtrqLimRmp = 100"],
     "Steps": [{"Actions": ["Wait 200000"], "Expected Results": ["Rail_trqFlTempLim = 2500"]}],
     "Detailed Explanation": "Holding switch steady at 1 after ramp completion, output remains at Target B."},
    {"Pre-Condition": ["Rail_stTrqLim = 0", "Rail_trqFlTempLimPCV = 1000", "Rail_trqTempLimMeUn = 2000", "Rail_dtrqLimRmp = 0"],
     "Steps": [{"Actions": ["Rail_stTrqLim = 1", "Wait 100000"], "Expected Results": ["Rail_trqFlTempLim = 2000"]}],
     "Detailed Explanation": "Ramp parameter zero means instant ramp; output jumps immediately to Target B on switch=1."},
    {"Pre-Condition": ["Rail_stTrqLim = 0", "Rail_trqFlTempLimPCV = 1000", "Rail_trqTempLimMeUn = 2000", "Rail_dtrqLimRmp = 1000000"],
     "Steps": [{"Actions": ["Rail_stTrqLim = 1", "Wait 100000"], "Expected Results": ["Rail_trqFlTempLim = 2000"]}],
     "Detailed Explanation": "Very high ramp parameter causes instant ramp; output jumps immediately to Target B on switch=1."},
    {"Pre-Condition": ["Rail_stTrqLim = 0", "Rail_trqFlTempLimPCV = -3276", "Rail_trqTempLimMeUn = 3276", "Rail_dtrqLimRmp = 100"],
     "Steps": [{"Actions": ["Rail_stTrqLim = 1", "Wait 100000"], "Expected Results": ["Rail_trqFlTempLim > -3276", "Rail_trqFlTempLim < 3276"]},
               {"Actions": ["Wait 100000"], "Expected Results": ["Rail_trqFlTempLim > -3276", "Rail_trqFlTempLim < 3276"]}],
     "Detailed Explanation": "Target values at min and max limits test boundary conditions during ramp from min to max."},
    {"Pre-Condition": ["Rail_stTrqLim = 0", "Rail_trqFlTempLimPCV = 1000", "Rail_trqTempLimMeUn = 2000", "Rail_dtrqLimRmp = 50"],
     "Steps": [{"Actions": ["Rail_stTrqLim = 1", "Wait 100000"], "Expected Results": ["Rail_trqFlTempLim > 1000", "Rail_trqFlTempLim < 2000"]},
               {"Actions": ["Wait 100000"], "Expected Results": ["Rail_trqFlTempLim > 1000", "Rail_trqFlTempLim < 2000"]},
               {"Actions": ["Wait 100000"], "Expected Results": ["Rail_trqFlTempLim > 1000", "Rail_trqFlTempLim < 2000"]}],
     "Detailed Explanation": "Multiple sequential calls simulate time progression to verify ramp increments towards Target B."},
    {"Pre-Condition": ["Rail_stTrqLim = 2", "Rail_trqFlTempLimPCV = 1000", "Rail_trqTempLimMeUn = 2000", "Rail_dtrqLimRmp = 100"],
     "Steps": [{"Actions": ["Wait 100000"], "Expected Results": ["Rail_trqFlTempLim = 1000"]}],
     "Detailed Explanation": "2 tests robustness; output expected to remain at Target A or default safe value."},
]


def main():
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs("log", exist_ok=True)
    log_path = os.path.join("log", f"verify_metric_fix_{ts}.log")

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8"); fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt)
    root = logging.getLogger(); root.setLevel(logging.WARNING); root.addHandler(fh); root.addHandler(ch)
    logging.getLogger("mbd_metric").setLevel(logging.DEBUG)
    run_log = logging.getLogger("verify"); run_log.setLevel(logging.DEBUG)

    run_log.info("=" * 70)
    run_log.info("METRIC-ONLY VERIFICATION (no LLM) — Rail_TempLim_50.8.0 / weakspot_id=2")
    run_log.info("Log file: %s", log_path)
    run_log.info("=" * 70)

    fc_entries = [{"fc_name": "Rail_TempLim_50.8.0", "leaf_node": "_Proc", "weakspot_ids": [2], "raster_time": 100}]
    dataset = load_mbd_dataset(fc_entries=fc_entries)
    example = prepare_full_examples(dataset)[0]
    run_log.info("Loaded example: weakspot_op_key=%s dll_path=%s",
                 getattr(example, "weakspot_op_key", "?"), getattr(example, "dll_path", "?"))

    final_str = json.dumps(FINAL_TEST_CASES)
    prediction = dspy.Prediction(
        context_analysis="",
        test_cases_pre_postprocess=final_str,
        test_cases=final_str,
        m7_score=None,           # force M7 fallback path (exercises the fix)
        m7_details="forced fallback for verification",
    )

    scores = _compute_all_components(example, prediction)

    WEIGHTS = [("m1", 0.15), ("m2", 0.15), ("m3", 0.05), ("m4", 0.30), ("m6", 0.0), ("m7", 0.10), ("m8", 0.25)]
    total = sum(w * scores[k] for k, w in WEIGHTS)
    run_log.info("=" * 70)
    run_log.info("FINAL COMPOSITE SCORE = %.4f", total)
    for k, w in WEIGHTS:
        run_log.info("  %s (weight=%.2f) = %.4f | %s", k.upper(), w, scores[k], scores[f"{k}_details"])
    run_log.info("=" * 70)

    print(f"\nDone. Log: {log_path}")
    print(f"Composite: {total:.4f} | M7={scores['m7']:.3f} M8={scores['m8']:.3f}")


if __name__ == "__main__":
    main()
