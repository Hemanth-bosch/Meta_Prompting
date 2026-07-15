"""
MBD-UTA (Unit Test Assist) DSPy Optimization Pipeline
=====================================================

Transforms MBD weakspot test generation into a DSPy Module and applies
GEPA reflective optimization to improve prompt quality.

Pipeline:
  1. Load equation data for selected weakspot tasks
  2. Create DSPy Module (ContextExtractor + TestCaseGenerator)
  3. Evaluate baseline (human prompts = initial Signature instruction)
  4. Run GEPA optimization
  5. Compare baseline vs optimized on val/test sets

Usage:
    py extract_mbd.py --eval-only
    py extract_mbd.py --optimizer gepa --gepa-auto light
    py extract_mbd.py --eval-test --load-extractor .cache/mbd_optimized.json

Environment Variables (CLI flags take priority):
    MODEL_NAME              Student LM (default: azure/gpt-4.1-mini)
    REFLECTION_LM_NAME      Reflection LM for GEPA (default: azure/gpt-5-mini)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import dotenv
import dspy
from dspy.utils.usage_tracker import track_usage

from config import reconfigure_lm, reflectin_lm
from signatures.mbd import MBDContextExtraction, MBDTestCaseGeneration
from utils.mbd_data import load_mbd_dataset, stratified_split_mbd_dataset
from evaluation.mbd_metric import (
    mbd_testcase_metric,
    mbd_metric_with_feedback,
    _compute_all_components,
)
from optimization import (
    OptimizationRunner,
    create_gepa_strategy,
)

dotenv.load_dotenv(override=True)


# ═══════════════════════════════════════════════════════════════════
# DSPy MODULE
# ═══════════════════════════════════════════════════════════════════
class MBDTestGenerator(dspy.Module):
    """Two-stage test case generator: context extraction → test case generation.

    Stage 1: Analyze the weakspot equation graph (produces structured context)
    Stage 2: Generate JSON test cases from the context
    """

    def __init__(self, use_cot: bool = True):
        super().__init__()
        if use_cot:
            self.context_extractor = dspy.ChainOfThought(MBDContextExtraction)
            self.testcase_generator = dspy.ChainOfThought(MBDTestCaseGeneration)
        else:
            self.context_extractor = dspy.Predict(MBDContextExtraction)
            self.testcase_generator = dspy.Predict(MBDTestCaseGeneration)

    def forward(self, equation_json, const_values, weakspot_context, instance_names, raster_time):
        _log = logging.getLogger("extract_mbd.module")
        _log.debug(
            "MBDTestGenerator.forward called | instance_names=%s | raster_time=%s",
            str(instance_names)[:80],
            raster_time,
        )

        # Stage 1: Context extraction
        context_result = self.context_extractor(
            equation_json=equation_json,
            const_values=const_values,
            weakspot_context=weakspot_context,
        )
        _log.debug(
            "Stage 1 done — context_analysis length=%d",
            len(getattr(context_result, "context_analysis", "") or ""),
        )

        # Stage 2: Test case generation
        testcase_result = self.testcase_generator(
            context_analysis=context_result.context_analysis,
            equation_json=equation_json,
            const_values=const_values,
            instance_names=instance_names,
            raster_time=raster_time,
        )
        _log.debug(
            "Stage 2 done — test_cases length=%d",
            len(getattr(testcase_result, "test_cases", "") or ""),
        )

        return dspy.Prediction(
            context_analysis=context_result.context_analysis,
            test_cases=testcase_result.test_cases,
        )


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MBD-UTA DSPy Optimization Pipeline")
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
# TEE LOGGER  (mirrors extract_dca.py _TeeLogger)
# ═══════════════════════════════════════════════════════════════════
class _TeeLogger:
    """Redirect a stream (stdout or stderr) to both terminal and a shared log file."""

    def __init__(self, original_stream, log_file):
        self._original = original_stream
        self._file = log_file

    def write(self, msg: str):
        self._file.write(msg)
        self._file.flush()
        # Guard against UnicodeEncodeError on Windows cp1252 terminals.
        # Replace un-encodable characters so DSPy table display never crashes.
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
def _print_per_example_scores(generator: MBDTestGenerator, examples, logger=None):
    """Run inference on each example and print per-metric breakdown."""
    _log = logger or logging.getLogger("extract_mbd")

    print("\n" + "-" * 70)
    print("PER-EXAMPLE METRIC BREAKDOWN")
    print("-" * 70)
    _log.info("=== PER-EXAMPLE METRIC BREAKDOWN ===")

    WEIGHTS = [("m1", 0.25), ("m2", 0.20), ("m3", 0.10), ("m4", 0.25), ("m6", 0.05), ("m7", 0.15)]

    for i, ex in enumerate(examples):
        fc_tag = f"{getattr(ex, 'fc_name', '?')} ws={getattr(ex, 'weakspot_id', '?')}"
        try:
            pred = generator(
                equation_json=ex.equation_json,
                const_values=ex.const_values,
                weakspot_context=ex.weakspot_context,
                instance_names=ex.instance_names,
                raster_time=ex.raster_time,
            )
            scores = _compute_all_components(ex, pred)
            total = sum(w * scores[k] for k, w in WEIGHTS)
            msg = (
                f"[{i+1}] {fc_tag} → total={total:.3f} | "
                f"M1={scores['m1']:.2f} M2={scores['m2']:.2f} M3={scores['m3']:.2f} "
                f"M4={scores['m4']:.2f} M6={scores['m6']:.2f} M7={scores['m7']:.2f}"
            )
            print(f"  {msg}")
            print(f"       M1: {scores['m1_details']}")
            print(f"       M2: {scores['m2_details']}")
            print(f"       M3: {scores['m3_details']}")
            print(f"       M4: {scores['m4_details']}")
            print(f"       M6: {scores['m6_details']}")
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
        strategy_tag = f"GEPA_MBD_{gepa_auto_tag}"
    log_prefix = f"{model_tag}_{strategy_tag}_{cot_tag}"

    # ── Log directory & TeeLogger ──────────────────────────────────
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_prefix}_mbd_run_{current_time}.log")
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

    # Project namespaces: DEBUG
    # NOTE: metric logger is "mbd_metric" (evaluation/mbd_metric.py), not "mbd.metric".
    for _ns in ("extract_mbd", "extract_mbd.module", "mbd.data", "mbd_metric"):
        logging.getLogger(_ns).setLevel(logging.DEBUG)

    # Truncation inspector — logs full LM interaction when response is truncated
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

    _run_logger = logging.getLogger("extract_mbd")
    _run_logger.info("MBD optimization run started — log file: %s", log_path)

    # ── Model configuration ────────────────────────────────────────
    model_to_use = args.model_name or os.getenv("MODEL_NAME", "azure/gpt-4.1-mini")
    cache_enabled = not args.no_cache
    reconfigure_lm(model_to_use, cache=cache_enabled)

    # ── Configuration banner ───────────────────────────────────────
    print(f"\n{'='*70}")
    print(" MBD-UTA Pipeline Configuration:")
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
    from utils.mbd_data import all_fc_entries
    fc_entries = all_fc_entries() if args.use_all_tasks else None
    dataset = load_mbd_dataset(fc_entries=fc_entries)

    if not dataset:
        msg = "ERROR: No examples loaded. Check data directory and FC entries."
        print(msg)
        _run_logger.error(msg)
        _restore_streams(sys, _tee_stdout, _tee_stderr, _log_file)
        sys.exit(1)

    train_set, val_set, test_set = stratified_split_mbd_dataset(dataset)
    print(f"Split: train={len(train_set)} | val={len(val_set)} | test={len(test_set)}")
    _run_logger.info(
        "Dataset split: total=%d train=%d val=%d test=%d",
        len(dataset), len(train_set), len(val_set), len(test_set),
    )

    # Warn if test set is empty (small dataset)
    if len(test_set) == 0:
        print("WARNING: test_set is empty — test evaluation will be skipped.")
        _run_logger.warning("test_set is empty — test evaluation will be skipped")

    # ── Create module ──────────────────────────────────────────────
    generator = MBDTestGenerator(use_cot=not args.no_cot)
    _run_logger.info("Created MBDTestGenerator(use_cot=%s)", not args.no_cot)

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

    # ── Verify seed signature (mirrors extract_dca.py pattern) ─────
    _print_signature_banner(generator, _run_logger)

    # ── Save seed state (dump_state before any optimization) ───────
    seed_state_path = os.path.join(log_dir, f"{log_prefix}_mbd_run_{current_time}_seed_state.json")
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
    baseline_score = baseline_result.score  # EvaluationResult.score is 0-100 scale
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

    # GEPA log_dir for checkpointing
    cache_dir = ".cache"
    os.makedirs(cache_dir, exist_ok=True)
    gepa_log_dir = os.path.join(cache_dir, f"gepa_mbd_{current_time}")

    strategy = create_gepa_strategy(
        name=f"GEPA_MBD_{args.gepa_auto.capitalize()}",
        description=f"GEPA ({args.gepa_auto}) — reflective prompt optimization for MBD-UTA",
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
            "run_type": "extract_mbd",
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
        err_path = os.path.join(log_dir, f"{log_prefix}_mbd_run_{current_time}_error_summary.json")
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
        log_dir, f"{log_prefix}_mbd_run_{current_time}_optimized_state.json"
    )
    with open(optimized_state_path, "w", encoding="utf-8") as _fp:
        json.dump(optimized_generator.dump_state(), _fp, indent=2, ensure_ascii=False)
    print(f"Optimized state saved -> {optimized_state_path}")
    _run_logger.info("Optimized state saved: %s", optimized_state_path)

    # ── Save model .json ───────────────────────────────────────────
    best_strategy_name = getattr(runner, "best_strategy_name", strategy.name)
    model_path = os.path.join(
        cache_dir,
        f"{model_tag}_{best_strategy_name.replace(' ', '_')}_{cot_tag}_mbd_optimized_{current_time}.json",
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
    print("MBD Pipeline complete!")
    print(f"  Model saved:    {model_path}")
    print(f"  Log file:       {log_path}")
    print(f"{'='*70}\n")
    _run_logger.info("MBD pipeline complete.")

    _restore_streams(sys, _tee_stdout, _tee_stderr, _log_file)


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════
def _print_signature_banner(generator: MBDTestGenerator, run_logger, label: str = "SEED INSTRUCTION"):
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
        "run_type": "extract_mbd",
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
    # Save to log dir
    summary_path = os.path.join(log_dir, f"{log_prefix}_mbd_run_{current_time}_run_summary.json")
    # Save to cache dir (also, for easy discovery)
    cache_summary_path = os.path.join(".cache", f"{log_prefix}_mbd_run_{current_time}_run_summary.json")
    for path in (summary_path, cache_summary_path):
        try:
            with open(path, "w", encoding="utf-8") as _fp:
                json.dump(summary, _fp, indent=2, ensure_ascii=False)
            run_logger.info("Run summary saved: %s", path)
        except OSError as e:
            run_logger.warning("Could not save run summary %s: %s", path, e)
    print(f"Run summary saved -> {summary_path}")


def _log_token_usage(label: str, usage: dict, run_logger) -> None:
    """Print and log token usage summary from a track_usage() context."""
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
    """Restore original stdout/stderr and close the log file."""
    sys_module.stdout = tee_stdout._original
    sys_module.stderr = tee_stderr._original
    log_file.close()


if __name__ == "__main__":
    main()
