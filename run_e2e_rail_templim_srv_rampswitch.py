"""
End-to-end debug run: Rail_TempLim_50.8.0 / weakspot_id=2 (Srv_RampSwitch / Rail_trqRampSwitch)
=================================================================================================

Runs ONE example through MBDFullPipelineModule.forward() (Prompt 1 -> Prompt 2 ->
bridge -> Prompt 4 retry loop -> post-processing -> final JSON), then runs the
full M1-M8 metric evaluation (_compute_all_components) on the resulting prediction.

NO GEPA optimization. Single inference + evaluation. Everything is logged in
detail (DEBUG level) to a log file for step-by-step review.

Usage:
    py run_e2e_rail_templim_srv_rampswitch.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime

import dotenv

from config import reconfigure_lm  # noqa: E402

from extract_mbd_full import MBDFullPipelineModule, prepare_full_examples
from utils.mbd_data import load_mbd_dataset
from evaluation.mbd_metric import _compute_all_components

dotenv.load_dotenv(override=True)


def main():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"e2e_Rail_TempLim_SrvRampSwitch_{timestamp}.log")

    # ── Logging setup ────────────────────────────────────────────────
    log_fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(log_fmt)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    for ns in ("extract_mbd_full", "mbd_metric"):
        lg = logging.getLogger(ns)
        lg.setLevel(logging.DEBUG)

    run_logger = logging.getLogger("e2e_runner")
    run_logger.setLevel(logging.DEBUG)
    run_logger.info("=" * 70)
    run_logger.info("END-TO-END DEBUG RUN: Rail_TempLim_50.8.0 / weakspot_id=2 (Srv_RampSwitch)")
    run_logger.info("Log file: %s", log_path)
    run_logger.info("=" * 70)

    # ── Model configuration ─────────────────────────────────────────
    model_to_use = os.getenv("MODEL_NAME", "azure/gpt-4.1-mini")
    run_logger.info("Configuring LM: %s", model_to_use)
    reconfigure_lm(model_to_use, cache=True)

    # ── Load the target example ─────────────────────────────────────
    fc_entries = [
        {"fc_name": "Rail_TempLim_50.8.0", "leaf_node": "_Proc", "weakspot_ids": [2], "raster_time": 100},
    ]
    run_logger.info("Loading dataset: fc_entries=%s", fc_entries)
    dataset = load_mbd_dataset(fc_entries=fc_entries)
    if not dataset:
        run_logger.error("No examples loaded — aborting")
        sys.exit(1)

    examples = prepare_full_examples(dataset)
    example = examples[0]
    run_logger.info(
        "Loaded example: fc_name=%s weakspot_id=%s weakspot_op_key=%s raster_time=%s dll_path=%s",
        getattr(example, "fc_name", "?"),
        getattr(example, "weakspot_id", "?"),
        getattr(example, "weakspot_op_key", "?"),
        getattr(example, "raster_time", "?"),
        getattr(example, "dll_path", "?"),
    )
    run_logger.debug("Example inputs() keys = %s", list(example.inputs().keys()))

    # ── Run forward() — Prompt1 -> Prompt2 -> bridge -> Prompt4 -> postproc ──
    run_logger.info("Instantiating MBDFullPipelineModule(use_cot=True)")
    generator = MBDFullPipelineModule(use_cot=True)

    run_logger.info("Calling generator.forward(**example.inputs()) ...")
    prediction = generator(**example.inputs())
    run_logger.info("forward() returned. m7_score=%s m7_details=%s",
                     getattr(prediction, "m7_score", None), getattr(prediction, "m7_details", None))

    # ── Run full M1-M8 metric evaluation ─────────────────────────────
    run_logger.info("Calling _compute_all_components(example, prediction) ...")
    scores = _compute_all_components(example, prediction)

    WEIGHTS = [
        ("m1", 0.15), ("m2", 0.15), ("m3", 0.05), ("m4", 0.30),
        ("m6", 0.0), ("m7", 0.10), ("m8", 0.25),
    ]
    total = sum(w * scores[k] for k, w in WEIGHTS)

    run_logger.info("=" * 70)
    run_logger.info("FINAL COMPOSITE SCORE = %.4f", total)
    for k, w in WEIGHTS:
        run_logger.info("  %s (weight=%.2f) = %.4f | %s", k.upper(), w, scores[k], scores[f"{k}_details"])
    run_logger.info("=" * 70)

    # Dump final test_cases JSON for reference
    try:
        final_tc = json.loads(prediction.test_cases)
        run_logger.info("FINAL test_cases (%d entries):\n%s", len(final_tc), json.dumps(final_tc, indent=2))
    except Exception as e:
        run_logger.warning("Could not pretty-print final test_cases: %s", e)

    print(f"\nDone. Log written to: {log_path}")
    print(f"Final composite score: {total:.4f}")


if __name__ == "__main__":
    main()
