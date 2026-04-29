#!/usr/bin/env python3
"""run_tournament.py — CLI entrypoint for the Sequential Cascaded DCA Optimization.

Flow
----
1. Parse args → build TournamentConfig
2. Setup logging (TeeLogger for the entire run)
3. Load DCA dataset & stratified split
4. Initialize leaderboard
5. Stage 0: Seed prompt generation (or load seed from file)
6. Stages 1-3: Run each stage via StageRunner, seeding from top-K of
   the previous stage
7. Final tournament re-evaluation at eval_temperature
8. Optional test set evaluation via subprocess
9. Generate & save summary to output_dir AND log_dir

Usage examples
--------------
    # Default: seed → bootstrap → miprov2
    uv run python -m dca_tournaments.run_tournament

    # Quick smoke test (Stage 0 only, 1 candidate)
    uv run python -m dca_tournaments.run_tournament --stages seed --n-seed-candidates 1

    # Full pipeline with GEPA
    uv run python -m dca_tournaments.run_tournament --stages seed,bootstrap,miprov2,gepa

    # Skip Stage 0 — use a saved program as the sole seed
    uv run python -m dca_tournaments.run_tournament --seed-from-file .cache/my_best.json --stages bootstrap,miprov2

    # Dry run — print config and estimated runs, then exit
    uv run python -m dca_tournaments.run_tournament --dry-run

    # Change models
    uv run python -m dca_tournaments.run_tournament --student-model azure/gpt-4.1 --seed-model azure/gpt-4.1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import dotenv
import dspy

from config import create_lm
from utils import load_dca_dataset, stratified_split_dca_dataset
from evaluation.dca_metric import dca_comparison_metric, dca_comparison_metric_with_feedback
from dca_tournaments.tour_config import TournamentConfig, VALID_STAGES
from dca_tournaments.leaderboard import OptimizationLeaderboard
from dca_tournaments.seed_prompt_generator import SeedPromptGenerator
from dca_tournaments.stage_runner import StageRunner
from dca_tournaments.tour_utils import (
    TeeLogger,
    evaluate_candidate_at_temperature,
    load_candidate,
    save_candidate,
    generate_summary,
    save_summary,
    print_final_summary,
    get_program_info,
    create_mini_dataset,
    plot_pareto_candidates,
)


def _stage_dir_name(stage: str) -> str:
    return StageRunner.get_stage_dir_name(stage)

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _save_stage_snapshot(
    stage_name: str,
    all_candidates: list,
    top_k_selected: list,
    config: TournamentConfig,
) -> None:
    """Save all candidates and top-K selected entries as JSON to the log dir.

    Creates two files per stage:
      - ``stage_{name}_all_candidates.json``  — every entry produced
      - ``stage_{name}_top_k_selected.json``  — the entries forwarded to the next stage
    """
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    def _entry_to_dict(e, rank: int | None = None) -> dict:
        d: dict = {}
        if rank is not None:
            d["rank"] = rank
        d.update({
            "stage": e.stage,
            "config_name": e.config_name,
            "val_total_score": round(e.val_total_score, 4),
            "val_cls_score": round(e.val_cls_score, 4),
            "val_change_score": round(e.val_change_score, 4),
            "n_demos": e.n_demos,
            "instruction_length": e.instruction_length,
            "val_avg_total_tokens": e.val_avg_total_tokens,
            "val_avg_input_tokens": e.val_avg_input_tokens,
            "val_avg_output_tokens": e.val_avg_output_tokens,
            "val_avg_reasoning_tokens": e.val_avg_reasoning_tokens,
            "per_category_scores": e.per_category_scores,
            "parent_path": e.parent_path,
            "program_path": e.program_path,
            "timestamp": e.timestamp,
        })
        return d

    try:
        all_path = log_dir / f"stage_{stage_name}_all_candidates.json"
        with open(all_path, "w", encoding="utf-8") as f:
            json.dump({
                "stage": stage_name,
                "n_candidates": len(all_candidates),
                "selection_criteria": config.selection_criteria,
                "cross_stage_selection": config.cross_stage_selection,
                "candidates": [_entry_to_dict(e, i + 1) for i, e in enumerate(all_candidates)],
            }, f, indent=2, ensure_ascii=False, default=str)
        print(f"  [{_ts()}] All candidates saved → {all_path}")

        top_path = log_dir / f"stage_{stage_name}_top_k_selected.json"
        with open(top_path, "w", encoding="utf-8") as f:
            json.dump({
                "stage": stage_name,
                "top_k": config.top_k,
                "n_selected": len(top_k_selected),
                "selection_criteria": config.selection_criteria,
                "selected": [_entry_to_dict(e, i + 1) for i, e in enumerate(top_k_selected)],
            }, f, indent=2, ensure_ascii=False, default=str)
        print(f"  [{_ts()}] Top-K selected saved → {top_path}")
    except Exception as exc:
        print(f"  [WARN] Failed to save stage snapshot for '{stage_name}': {exc}")

dotenv.load_dotenv(override=True)


# ═══════════════════════════════════════════════════════════════════
# Argument parser
# ═══════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sequential Cascaded DCA Prompt Optimization (Tournament Pipeline V2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Stage control
    g = p.add_argument_group("Stage control")
    g.add_argument("--stages", type=str, default=None,
                   help=f"Comma-separated list of stages to run. "
                        f"Valid: {','.join(VALID_STAGES)}  (default: seed,bootstrap,miprov2)")
    g.add_argument("--top-k", type=int, default=None,
                   help="Number of top candidates to propagate between stages (default: 2)")
    g.add_argument("--skip-final-tournament", action="store_true",
                   help="Skip the final re-evaluation at eval_temperature")
    g.add_argument("--skip-test-eval", action="store_true",
                   help="Skip test set evaluation at the end")
    g.add_argument("--dry-run", action="store_true",
                   help="Print config and estimated runs, then exit")

    # Models
    g = p.add_argument_group("Models")
    g.add_argument("--student-model", type=str, default=None,
                   help="Student LM (default: azure/gpt-4.1-mini)")
    g.add_argument("--seed-model", type=str, default=None,
                   help="Seed LM for Stage 0 meta-prompt (default: azure/gpt-5.2)")
    g.add_argument("--reflection-model", type=str, default=None,
                   help="Reflection LM for GEPA (default: from .env REFLECTION_MODEL_NAME)")
    g.add_argument("--cot", action="store_true", default=False,
                   help="Use ChainOfThought instead of Predict")
    g.add_argument("--no-cot", action="store_true", default=False,
                   help="Use Predict (default)")

    # Per-stage model overrides
    g = p.add_argument_group("Per-stage model overrides")
    g.add_argument("--bootstrap-model", type=str, default=None,
                   help="Model override for bootstrap stage (default: student model)")
    g.add_argument("--miprov2-model", type=str, default=None,
                   help="Model override for MIPROv2 stage (default: student model)")
    g.add_argument("--gepa-model", type=str, default=None,
                   help="Model override for GEPA stage (default: student model)")

    # Temperature
    g = p.add_argument_group("Temperature")
    g.add_argument("--optimization-temperature", type=float, default=None,
                   help="Student LM temp during optimization (default: 0.0 via dspy)")
    g.add_argument("--eval-temperature", type=float, default=None,
                   help="Temp for final tournament re-eval (default: 0.1)")
    g.add_argument("--seed-eval-temperature", type=float, default=None,
                   help="Temp for Stage 0 baseline eval (default: 0.1)")
    g.add_argument("--seed-generation-temperature", type=float, default=None,
                   help="Temp for Stage 0 seed LLM meta-prompt call (default: 1.0)")

    # Mini datasets
    g = p.add_argument_group("Mini datasets")
    g.add_argument("--mini-train-size", type=int, default=None,
                   help="Mini train dataset size (0=full, default: 120)")
    g.add_argument("--mini-val-size", type=int, default=None,
                   help="Mini val dataset size (0=full, default: 40)")

    # Instructions mode
    g = p.add_argument_group("Instructions mode")
    g.add_argument("--instructions-only", action="store_true", default=False,
                   help="Only optimize instructions (no field descriptions)")
    g.add_argument("--with-field-descs", action="store_true", default=False,
                   help="Also generate field descriptions in Stage 0")

    # Stage 0
    g = p.add_argument_group("Stage 0 — Seed generation")
    g.add_argument("--n-seed-candidates", type=int, default=None,
                   help="Number of seed prompt candidates (default: 3)")
    g.add_argument("--max-context-examples", type=int, default=None,
                   help="Number of train examples shown to seed LLM per class (default: 6)")
    g.add_argument("--seed-context-mode", type=str, default=None,
                   choices=["worst", "best"],
                   help="Which examples to show the seed LLM: 'worst' (failures, default) "
                        "or 'best' (gold examples that the baseline already got right)")
    g.add_argument("--seed-from-file", type=str, default=None,
                   help="Skip Stage 0 — use this saved .json program as the sole seed")
    g.add_argument("--seed-reasoning-effort", type=str, default=None,
                   choices=["low", "medium", "high"],
                   help="Reasoning effort for seed LLM (default: low)")
    g.add_argument("--seed-add-demos", action="store_true", default=False,
                   help="Also evaluate seed candidates with demos added")
    g.add_argument("--no-seed-add-demos", action="store_true", default=False,
                   help="Don't add demos variant in Stage 0")
    g.add_argument("--seed-parallel-evaluations", type=int, default=None,
                   help="Max concurrent Stage-0 seed evaluation variants (default: 4)")

    # Stage 1
    g = p.add_argument_group("Stage 1 — Bootstrap")
    g.add_argument("--bootstrap-configs", type=str, default=None,
                   help='JSON array of bootstrap configs')
    g.add_argument("--bootstrap-stop-at-score", type=float, default=None,
                   help="Early stopping threshold for Bootstrap")
    g.add_argument("--bootstrap-num-threads", type=int, default=None,
                   help="Number of threads for Bootstrap optimizer (default: 4)")
    g.add_argument("--bootstrap-teacher-model", type=str, default=None,
                   help="Teacher model for Bootstrap demo generation (default: azure/gpt-5.2)")

    # Stage 2
    g = p.add_argument_group("Stage 2 — MIPROv2")
    g.add_argument("--miprov2-auto", type=str, default=None,
                   help="Comma-separated intensities, e.g. 'light,medium'")
    g.add_argument("--miprov2-models", type=str, default=None,
                   help="Comma-separated model names for MIPROv2 grid")
    g.add_argument("--miprov2-max-labeled", type=int, default=None,
                   help="Max labeled demos for MIPROv2")
    g.add_argument("--miprov2-max-bootstrapped", type=int, default=None,
                   help="Max bootstrapped demos for MIPROv2")
    g.add_argument("--miprov2-clear-descs", action="store_true",
                   help="Clear field descriptions before MIPROv2 optimization")
    g.add_argument("--miprov2-num-threads", type=int, default=None,
                   help="Number of threads for MIPROv2 optimizer (default: 4)")
    g.add_argument("--miprov2-prompt-model", type=str, default=None,
                   help="Proposer model for MIPROv2 instruction generation (default: azure/gpt-5.2)")
    g.add_argument("--miprov2-prompt-reasoning-effort", type=str, default=None,
                   choices=["low", "medium", "high"],
                   help="Reasoning effort for MIPROv2 prompt model (default: low)")
    g.add_argument("--miprov2-minibatch-size", type=int, default=None,
                   help="MIPROv2 minibatch size per trial (None=auto, default: auto)")
    g.add_argument("--miprov2-metric-threshold", type=float, default=None,
                   help="Only keep bootstrap traces with metric >= threshold")
    g.add_argument("--miprov2-teacher-model", type=str, default=None,
                   help="Teacher model for MIPROv2 demo generation (default: azure/gpt-5.2)")

    # Stage 3 — GEPA
    g = p.add_argument_group("Stage 3 — GEPA")
    g.add_argument("--gepa-auto", type=str, default=None,
                   choices=["light", "medium", "heavy"],
                   help="GEPA intensity (default: medium)")
    g.add_argument("--gepa-reflection-images", action="store_true", default=False,
                   help="Send images to reflection LM during GEPA")
    g.add_argument("--gepa-reflection-detail", type=str, default=None,
                   choices=["low", "high", "auto"],
                   help="Image detail level for reflection (default: low)")
    g.add_argument("--gepa-reflection-max-images", type=int, default=None,
                   help="Max images per GEPA reflection call (default: 6)")
    g.add_argument("--gepa-reflection-model-name", type=str, default=None,
                   help="Model for GEPA reflection LM")
    g.add_argument("--gepa-minibatch-size", type=int, default=None,
                   help="GEPA reflection_minibatch_size (default: 5)")
    g.add_argument("--gepa-train-size", type=int, default=None,
                   help="Per-GEPA train subset size (default: use global mini_train)")
    g.add_argument("--gepa-val-size", type=int, default=None,
                   help="Per-GEPA val subset size (default: use global mini_val)")
    g.add_argument("--gepa-num-threads", type=int, default=None,
                   help="Number of threads for GEPA optimizer (default: 4)")
    g.add_argument("--gepa-teacher-model", type=str, default=None,
                   help="Teacher model for GEPA demo generation (default: azure/gpt-5.2)")
    g.add_argument("--gepa-add-format-failure-feedback", action="store_true", default=None,
                   help="Include format failures in GEPA reflection (default: True)")
    g.add_argument("--gepa-no-format-failure-feedback", action="store_true", default=False,
                   help="Disable format failure feedback in GEPA")
    g.add_argument("--gepa-perfect-score", type=float, default=None,
                   help="Perfect score for GEPA gain calibration (default: 1.0)")
    g.add_argument("--gepa-lowest-score", type=float, default=None,
                   help="Lowest score for GEPA gain calibration (default: 0.0)")
    g.add_argument("--gepa-candidate-selection", type=str, default=None,
                   choices=["pareto", "greedy"],
                   help="GEPA internal candidate selection strategy (default: pareto)")

    # Selection criteria
    g = p.add_argument_group("Selection criteria")
    g.add_argument("--selection-criteria", type=str, default=None,
                   choices=["score_only", "pareto", "tokens_only"],
                   help="How to select top-K candidates (default: score_only)")
    g.add_argument("--cross-stage-selection", action="store_true", default=False,
                   help="Select top-K from ALL stages combined")
    g.add_argument("--no-cross-stage-selection", action="store_true", default=False,
                   help="Select top-K from current stage only")

    # Token tracking
    g = p.add_argument_group("Token tracking")
    g.add_argument("--track-tokens", action="store_true", default=False,
                   help="Track token usage via dspy.track_usage()")
    g.add_argument("--no-track-tokens", action="store_true", default=False,
                   help="Disable token tracking")

    # Hybrid metric
    g = p.add_argument_group("Hybrid metric")
    g.add_argument("--hybrid-metric", action="store_true", default=False,
                   help="Use hybrid metric: score * weight + length_penalty * weight "
                        "(penalises long prompts during optimisation; "
                        "plain eval always uses true DCA score)")
    g.add_argument("--hybrid-score-weight", type=float, default=None,
                   help="Weight for DCA score in hybrid metric (default: 0.5)")
    g.add_argument("--hybrid-length-weight", type=float, default=None,
                   help="Weight for prompt-length efficiency term (default: 0.5)")
    g.add_argument("--hybrid-max-prompt-tokens", type=int, default=None,
                   help="Input token count at which length_score reaches 0 (default: 4000)")

    # Paths & data
    g = p.add_argument_group("Paths & data")
    g.add_argument("--output-dir", type=str, default=None,
                   help="Base output directory (default: .cache/tournament)")
    g.add_argument("--log-dir", type=str, default=None,
                   help="Base log directory (default: log/tournament)")
    g.add_argument("--pairs-dir", type=str, default=None,
                   help="DCA image pairs directory (default: data/dca_data/image_pairs)")
    g.add_argument("--gt-dir", type=str, default=None,
                   help="DCA ground truth directory (default: data/dca_data/image_pairs_llm_output_exp1)")
    g.add_argument("--split-seed", type=int, default=None,
                   help="Random seed for train/val/test split (default: 42)")
    g.add_argument("--train-ratio", type=float, default=None,
                   help="Train split ratio (default: 0.6)")
    g.add_argument("--val-ratio", type=float, default=None,
                   help="Val split ratio (default: 0.2)")
    g.add_argument("--test-ratio", type=float, default=None,
                   help="Test split ratio (default: 0.2)")
    g.add_argument("--resume-from", type=str, default=None,
                   help="Resume from existing tournament dir (reuses timestamp + leaderboard)")

    # Parallelism
    g = p.add_argument_group("Parallelism")
    g.add_argument("--max-parallel", type=int, default=None,
                   help="Max concurrent optimizer runs per stage (default: 4)")
    g.add_argument("--eval-num-threads", type=int, default=None,
                   help="Threads for candidate evaluation (default: 4)")

    return p


# ═══════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Build config ─────────────────────────────────────────────
    config = TournamentConfig.from_cli_args(args)

    # ── Dry-run mode ─────────────────────────────────────────────
    if args.dry_run:
        config.print_summary()
        print("Dry run — exiting.\n")
        return

    # ── Create directories ───────────────────────────────────────
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    # Save config early (before any work) so we can inspect on crash
    config.save()
    config.save(Path(config.log_dir) / "config.json")

    # ── TeeLogger for entire run ─────────────────────────────────
    tee = TeeLogger(config.log_dir, "tournament_main")
    tee.__enter__()

    started_at = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    stages_run: list[str] = []
    # Maps stage → top-K CandidateEntry objects passed to the next stage.
    # Used for the cascade table in the final summary.
    stage_top_k_history: dict[str, list] = {}
    # Maps stage → global best val_total_score immediately before that stage ran.
    # Gives the true cross-stage improvement delta.
    stage_pre_best: dict[str, float] = {}

    try:
        config.print_summary()

        # ── Configure LM ────────────────────────────────────────
        print(f"[{_ts()}] Configuring student LM...")
        student_lm_kwargs = {}
        if config.optimization_temperature is not None:
            student_lm_kwargs["temperature"] = config.optimization_temperature

        student_lm = create_lm(
            config.student_model_name,
            cache=False,
            **student_lm_kwargs,
        )
        dspy.configure(lm=student_lm)
        print(f"  Student LM: {config.student_model_name}")

        # ── Load data ───────────────────────────────────────────
        print(f"\n[{_ts()}] Loading DCA dataset...")
        dataset = load_dca_dataset(config.pairs_dir, config.gt_dir)
        if not dataset:
            print("ERROR: No valid DCA image pairs found. Check data paths.")
            return

        print(f"\n[{_ts()}] Splitting dataset (seed={config.split_seed})...")

        # Stratification uses ["expected_discard_image", "expected_human_label_category", "doubt"]
        # by default inside stratified_split_dca_dataset (see utils/splitting.py).
        train_set, val_set, test_set = stratified_split_dca_dataset(
            dataset,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            random_seed=config.split_seed,
            verbose=True,
        )

        # Save split manifest
        val_names = [str(getattr(ex, "pair_name", "")) for ex in val_set]
        test_names = [str(getattr(ex, "pair_name", "")) for ex in test_set]
        split_manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "seed": config.split_seed,
            "ratios": {"train": config.train_ratio, "val": config.val_ratio, "test": config.test_ratio},
            "total_examples": len(dataset),
            "val": {"count": len(val_names), "pair_names": val_names},
            "test": {"count": len(test_names), "pair_names": test_names},
        }
        split_path = Path(config.output_dir) / "split_manifest.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_manifest, f, indent=2)
        # Also save to standard cache location for test eval script
        cache_split_path = Path(".cache") / f"dca_split_seed{config.split_seed}.json"
        cache_split_path.parent.mkdir(exist_ok=True)
        with open(cache_split_path, "w", encoding="utf-8") as f:
            json.dump(split_manifest, f, indent=2)

        print(f"\n  train: {len(train_set)}  |  val: {len(val_set)}  |  test: {len(test_set)}")

        # ── Create mini datasets ────────────────────────────────
        # create_mini_dataset stratifies by the same 3 fields as
        # stratified_split_dca_dataset: (expected_discard_image,
        # expected_human_label_category, doubt) to preserve binary
        # classification balance, category proportions, and doubt
        # distribution in the mini sets.
        mini_train = create_mini_dataset(train_set, config.mini_train_size, seed=config.split_seed)
        mini_val = create_mini_dataset(val_set, config.mini_val_size, seed=config.split_seed)

        # Log mini dataset category distributions for debugging
        from collections import Counter as _Counter
        _mini_train_cats = _Counter(str(getattr(ex, "expected_human_label_category", "?")) for ex in mini_train)
        _mini_val_cats = _Counter(str(getattr(ex, "expected_human_label_category", "?")) for ex in mini_val)
        print(f"  Mini train categories: {dict(sorted(_mini_train_cats.items()))}")
        print(f"  Mini val   categories: {dict(sorted(_mini_val_cats.items()))}")

        # ── Initialize leaderboard ──────────────────────────────
        lb_path = Path(config.output_dir) / "leaderboard.json"
        if args.resume_from and lb_path.exists():
            print(f"\nResuming from existing leaderboard: {lb_path}")
            leaderboard = OptimizationLeaderboard.from_file(lb_path)
        else:
            leaderboard = OptimizationLeaderboard(save_path=str(lb_path))

        is_resume_mode = bool(args.resume_from and lb_path.exists())
        if is_resume_mode:
            print("\nExisting stage entries in resumed leaderboard:")
            for stage_name in config.stages:
                print(f"  {stage_name:12s}: {len(leaderboard.stage_entries(stage_name))}")

        metric = dca_comparison_metric
        metric_with_feedback = dca_comparison_metric_with_feedback
        if config.use_hybrid_metric:
            from evaluation.dca_metric import make_hybrid_metric
            metric, metric_with_feedback = make_hybrid_metric(
                dca_comparison_metric,
                dca_comparison_metric_with_feedback,
                score_weight=config.hybrid_score_weight,
                length_weight=config.hybrid_length_weight,
                max_prompt_tokens=config.hybrid_max_prompt_tokens,
            )
            print(
                f"\n  Hybrid metric enabled: "
                f"score*{config.hybrid_score_weight} + "
                f"length_efficiency*{config.hybrid_length_weight} "
                f"(max_prompt_tokens={config.hybrid_max_prompt_tokens})"
            )

        # ── Stage 0: Seed generation ────────────────────────────
        seed_entries = []

        if "seed" in config.stages:
            if is_resume_mode and leaderboard.stage_entries("seed") and not config.seed_from_file:
                print("\nSkipping Stage 0 (seed): already completed in resumed leaderboard.")
                seed_entries = leaderboard.top_k(config.top_k, stage="seed")
            else:
                stages_run.append("seed")

                if config.seed_from_file:
                    # Load external seed instead of generating
                    print(f"\n  Loading seed from file: {config.seed_from_file}")
                    program = load_candidate(config.seed_from_file, use_cot=config.use_cot)
                    info = get_program_info(program)

                    from dca_tournaments.tour_utils import evaluate_candidate
                    eval_result = evaluate_candidate(
                        program, mini_val, metric, config.eval_num_threads,
                    )

                    # Save a copy into our tournament dir
                    seed_dir = Path(config.output_dir) / _stage_dir_name("seed")
                    seed_dir.mkdir(parents=True, exist_ok=True)
                    seed_path = seed_dir / "seed_from_file.json"
                    save_candidate(program, seed_path, metadata={
                        "stage": "seed",
                        "config_name": "seed_from_file",
                        "model": config.student_model_name,
                        "student_model": config.student_model_name,
                        "original_path": config.seed_from_file,
                        "eval_result": eval_result,
                    })

                    from dca_tournaments.leaderboard import CandidateEntry
                    entry = CandidateEntry(
                        program_path=str(seed_path),
                        stage="seed",
                        config_name="seed_from_file",
                        parent_path=None,
                        val_cls_score=eval_result["cls_score"],
                        val_change_score=eval_result["change_score"],
                        val_total_score=eval_result["total_score"],
                        per_category_scores=eval_result["per_category_scores"],
                        n_demos=info["n_demos"],
                        instruction_length=info["instruction_length"],
                        timestamp=datetime.now().isoformat(timespec="seconds"),
                        val_avg_total_tokens=eval_result.get("avg_total_tokens"),
                        val_avg_input_tokens=eval_result.get("avg_input_tokens"),
                        val_avg_output_tokens=eval_result.get("avg_output_tokens"),
                        val_avg_reasoning_tokens=eval_result.get("avg_reasoning_tokens"),
                        val_eval_time_seconds=eval_result.get("eval_time_seconds"),
                        metadata={
                            "stage": "seed",
                            "config_name": "seed_from_file",
                            "model": config.student_model_name,
                            "student_model": config.student_model_name,
                            "original_path": config.seed_from_file,
                            "eval_result": eval_result,
                        },
                    )
                    leaderboard.add(entry)
                    seed_entries = [entry]

                else:
                    # LLM seed generation
                    generator = SeedPromptGenerator(
                        seed_model_name=config.seed_model_name,
                        student_model_name=config.student_model_name,
                        n_candidates=config.n_seed_candidates,
                        max_context_examples=config.max_context_examples,
                        use_cot=config.use_cot,
                        seed_temperature=config.seed_generation_temperature,
                        seed_eval_temperature=config.seed_eval_temperature,
                        eval_num_threads=config.eval_num_threads,
                        instructions_only=config.instructions_only,
                        seed_reasoning_effort=config.seed_reasoning_effort,
                        seed_add_demos_variant=config.seed_add_demos_variant,
                        seed_parallel_evaluations=config.seed_parallel_evaluations,
                        track_tokens=config.track_tokens,
                        seed_context_mode=config.seed_context_mode,
                    )
                    # Baseline safety-net seeds (bare + with_demos) are now
                    # added inside SeedPromptGenerator.generate() itself, so
                    # they compete with LLM-generated seeds in the top-K
                    # selection.  This guarantees we never regress below the
                    # original signature even if the seed LLM underperforms.
                    seed_entries = generator.generate(
                        train_set=mini_train,
                        val_set=mini_val,
                        metric=metric,
                        leaderboard=leaderboard,
                        output_dir=config.output_dir,
                        log_dir=config.log_dir,
                    )

                    # Restore student LM after seed generation (it may have changed)
                    dspy.configure(lm=student_lm)

        if not seed_entries and is_resume_mode:
            for fallback_stage in ("gepa", "miprov2", "bootstrap", "seed"):
                existing = leaderboard.top_k(config.top_k, stage=fallback_stage)
                if existing:
                    seed_entries = existing
                    break

        # If no seed stage, require --seed-from-file for subsequent stages
        if not seed_entries and config.stages and config.stages[0] != "seed":
            if config.seed_from_file:
                # Load the seed even without running Stage 0
                program = load_candidate(config.seed_from_file, use_cot=config.use_cot)
                info = get_program_info(program)
                from dca_tournaments.tour_utils import evaluate_candidate
                eval_result = evaluate_candidate(
                    program, mini_val, metric, config.eval_num_threads,
                )
                seed_dir = Path(config.output_dir) / "external_seed"
                seed_dir.mkdir(parents=True, exist_ok=True)
                seed_path = seed_dir / "external_seed.json"
                save_candidate(program, seed_path, metadata={
                    "stage": "external",
                    "config_name": "external_seed",
                    "model": config.student_model_name,
                    "student_model": config.student_model_name,
                    "original_path": config.seed_from_file,
                    "eval_result": eval_result,
                })

                from dca_tournaments.leaderboard import CandidateEntry
                entry = CandidateEntry(
                    program_path=str(seed_path),
                    stage="external",
                    config_name="external_seed",
                    parent_path=config.seed_from_file,
                    val_cls_score=eval_result["cls_score"],
                    val_change_score=eval_result["change_score"],
                    val_total_score=eval_result["total_score"],
                    per_category_scores=eval_result["per_category_scores"],
                    n_demos=info["n_demos"],
                    instruction_length=info["instruction_length"],
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    val_avg_total_tokens=eval_result.get("avg_total_tokens"),
                    val_avg_input_tokens=eval_result.get("avg_input_tokens"),
                    val_avg_output_tokens=eval_result.get("avg_output_tokens"),
                    val_avg_reasoning_tokens=eval_result.get("avg_reasoning_tokens"),
                    val_eval_time_seconds=eval_result.get("eval_time_seconds"),
                    metadata={
                        "stage": "external",
                        "config_name": "external_seed",
                        "model": config.student_model_name,
                        "student_model": config.student_model_name,
                        "original_path": config.seed_from_file,
                        "eval_result": eval_result,
                    },
                )
                leaderboard.add(entry)
                seed_entries = [entry]
            else:
                print("\nERROR: No seed entries and no --seed-from-file specified.")
                print("Either include 'seed' in --stages or provide --seed-from-file.")
                return

        # ── Stages 1-3: optimization stages ─────────────────────
        runner = StageRunner(
            config=config,
            train_set=mini_train,
            val_set=mini_val,
            metric=metric,
            metric_with_feedback=metric_with_feedback,
            leaderboard=leaderboard,
        )

        # Track top-K entries flowing from one stage to the next
        current_top = seed_entries
        # Cross-stage improvement tracking
        stage_best_scores: dict[str, float] = {}
        if seed_entries:
            stage_best_scores["seed"] = max(e.val_total_score for e in seed_entries)

        # ── Apply top-K selection on seed entries before stages 1+ ─────
        # seed_entries may contain more than top_k candidates (LLM seeds +
        # baseline safety nets).  Trim to top_k using selection criteria.
        if seed_entries and len(seed_entries) > config.top_k:
            if config.cross_stage_selection:
                # exclude_stages removes seed_baseline BEFORE _require_token_data
                # runs, so the diagnostic train-set baseline (which has no meaningful
                # token data) never causes pareto/tokens_only selection to crash.
                current_top = leaderboard.top_k(
                    config.top_k,
                    criteria=config.selection_criteria,
                    exclude_stages=["seed_baseline"],
                )
                # Refill if pool was smaller than top_k after exclusion
                if len(current_top) < config.top_k:
                    extras = leaderboard.top_k(
                        config.top_k + 5,
                        criteria=config.selection_criteria,
                        exclude_stages=["seed_baseline"],
                    )
                    for e in extras:
                        if e not in current_top:
                            current_top.append(e)
                            if len(current_top) >= config.top_k:
                                break
            else:
                current_top = leaderboard.top_k(
                    config.top_k, stage="seed",
                    criteria=config.selection_criteria,
                )
            print(f"\n  [{_ts()}] Top-{config.top_k} selected from {len(seed_entries)} seed candidates:")
            for i, e in enumerate(current_top):
                tok_str = f"  avg_tok/smp={e.val_avg_total_tokens:.0f}" if e.val_avg_total_tokens is not None else ""
                print(f"    {i+1}. [{e.stage}] {e.config_name}  "
                      f"total={e.val_total_score:.4f}  {tok_str}")
        else:
            current_top = seed_entries

        # Record which seed candidates flow to the first optimization stage.
        stage_top_k_history["seed"] = list(current_top)

        # Save seed stage output to log dir — include all seed + seed_baseline entries
        _all_seed_candidates = (
            leaderboard.stage_entries("seed")
            + leaderboard.stage_entries("seed_baseline")
        )
        _save_stage_snapshot(
            stage_name="seed",
            all_candidates=_all_seed_candidates,
            top_k_selected=current_top,
            config=config,
        )

        for stage in config.stages:
            if stage == "seed":
                continue  # already handled above

            if is_resume_mode and leaderboard.stage_entries(stage):
                print(f"\nSkipping stage '{stage}': already completed in resumed leaderboard.")
                # Record stage-specific best score (always stage-scoped)
                _stage_entries = leaderboard.stage_entries(stage)
                stage_best_scores[stage] = max(
                    e.val_total_score for e in _stage_entries
                )
                # Select top-K for next stage (respect cross_stage_selection)
                if config.cross_stage_selection:
                    current_top = leaderboard.top_k(
                        config.top_k + 5,
                        criteria=config.selection_criteria,
                        exclude_stages=["seed_baseline"],
                    )[:config.top_k]
                else:
                    current_top = leaderboard.top_k(
                        config.top_k, stage=stage,
                        criteria=config.selection_criteria,
                    )
                stage_top_k_history[stage] = list(current_top)
                continue

            stages_run.append(stage)

            print(f"\n  Seeding stage '{stage}' with {len(current_top)} candidates:")
            for i, e in enumerate(current_top):
                tok_str = f"  avg_tok/smp={e.val_avg_total_tokens:.0f}" if e.val_avg_total_tokens is not None else ""
                print(f"    {i+1}. [{e.stage}] {e.config_name}  "
                      f"total={e.val_total_score:.4f}  {tok_str}")

            # Exclude seed_baseline from "best before" — its score is from
            # train_set (diagnostic), not val_set, so not comparable.
            _comparable_entries = [e for e in leaderboard.entries if e.stage != "seed_baseline"]
            pre_stage_best_score = max(
                (e.val_total_score for e in _comparable_entries),
                default=0.0,
            )
            pre_stage_best_tokens = min(
                (e.val_avg_total_tokens for e in _comparable_entries if e.val_avg_total_tokens is not None and e.val_avg_total_tokens > 0),
                default=None,
            )
            # Record for the cascade summary: what was the global best before this stage.
            stage_pre_best[stage] = pre_stage_best_score

            new_entries = runner.run_stage(stage, current_top)

            # ── Cross-stage improvement check ───────────────────
            # Use ALL entries from this stage (not just the selection-
            # filtered top-K returned by run_stage) so that "best this
            # stage" reflects the true best, regardless of selection mode.
            all_stage_new = leaderboard.stage_entries(stage)
            if all_stage_new:
                stage_best = max(e.val_total_score for e in all_stage_new)
                stage_best_entry = max(all_stage_new, key=lambda e: e.val_total_score)
                stage_min_tokens_entry = min(
                    (e for e in all_stage_new if e.val_avg_total_tokens is not None and e.val_avg_total_tokens > 0),
                    key=lambda e: e.val_avg_total_tokens, default=None,
                )
                stage_best_scores[stage] = stage_best
                improvement = stage_best - pre_stage_best_score
                prev_stages = [s for s in stage_best_scores if s != stage]
                prev_best_name = max(prev_stages,
                                     key=lambda s: stage_best_scores[s]) if prev_stages else "N/A"
                prev_best_val = stage_best_scores.get(prev_best_name, 0.0) if prev_stages else 0.0

                print(f"\n  ┌─ Stage '{stage}' improvement report ─────────")
                print(f"  │")
                print(f"  │ BY SCORE (val_total_score):")
                print(f"  │   Best this stage     : {stage_best:.4f}  ({stage_best_entry.config_name})")
                print(f"  │   Best before stage   : {pre_stage_best_score:.4f}")
                print(f"  │   Δ (score)           : {improvement:+.4f}")
                if improvement > 0:
                    print(f"  │   ✓ Stage '{stage}' improved over previous best")
                elif improvement == 0:
                    print(f"  │   ○ Stage '{stage}' matched but did not beat previous best")
                else:
                    print(f"  │   ✗ Stage '{stage}' did NOT improve")
                print(f"  │")
                print(f"  │ BY TOKENS (avg per val sample, lower=cheaper inference):")
                if stage_min_tokens_entry:
                    print(f"  │   Min avg tok/smp stg  : {stage_min_tokens_entry.val_avg_total_tokens:.0f}  "
                          f"({stage_min_tokens_entry.config_name}, score={stage_min_tokens_entry.val_total_score:.4f})")
                else:
                    print(f"  │   Min avg tok/smp stg  : N/A (no token data)")
                if pre_stage_best_tokens is not None:
                    print(f"  │   Min avg tok/smp prev : {pre_stage_best_tokens:.0f}")
                else:
                    print(f"  │   Min avg tok/smp prev : N/A")
                print(f"  │")
                print(f"  │ Best previous stage   : {prev_best_name} ({prev_best_val:.4f})")
                print(f"  └{'─'*55}")

                # Cumulative stage progress summary
                print(f"\n  Stage progress so far:")
                for s_name, s_score in stage_best_scores.items():
                    marker = " ◀ current" if s_name == stage else ""
                    print(f"    {s_name:12s}: {s_score:.4f}{marker}")
            else:
                print(f"\n  [WARN] Stage '{stage}' produced no entries.")

            # Select top-K using configured criteria
            if config.cross_stage_selection:
                # Select from ALL entries across all stages (exclude seed_baseline)
                current_top = leaderboard.top_k(
                    config.top_k + 5,
                    criteria=config.selection_criteria,
                    exclude_stages=["seed_baseline"],
                )[:config.top_k]
            else:
                # Select from current stage only
                current_top = leaderboard.top_k(
                    config.top_k, stage=stage,
                    criteria=config.selection_criteria,
                )

            if not current_top and new_entries:
                current_top = new_entries

            # Record which candidates flow to the next stage.
            stage_top_k_history[stage] = list(current_top)

            # Save stage output snapshot to log dir — use ALL entries
            # from this stage (not just top-K returned by run_stage)
            all_stage_entries = leaderboard.stage_entries(stage)
            _save_stage_snapshot(
                stage_name=stage,
                all_candidates=all_stage_entries,
                top_k_selected=current_top,
                config=config,
            )

        # ── Final tournament re-evaluation ──────────────────────
        test_eval_result = None

        if not config.skip_final_tournament and len(leaderboard.entries) >= 1:
            print(f"\n[{_ts()}] {'='*70}")
            print(f" Final Tournament Re-evaluation on FULL val set "
                  f"(temperature={config.eval_temperature})")
            print(f"{'='*70}")

            # Re-evaluate ALL entries on full val set for a fair apples-to-apples
            # comparison.  During optimization stages scores came from mini_val;
            # this pass makes every entry comparable on the same data.
            all_entries = list(leaderboard.entries)
            print(f"  Re-evaluating {len(all_entries)} candidates on full val set "
                  f"({len(val_set)} examples)...")

            for idx, entry in enumerate(all_entries, 1):
                print(f"\n  [{_ts()}] ({idx}/{len(all_entries)}) Re-evaluating: "
                      f"{entry.config_name}")
                try:
                    result = evaluate_candidate_at_temperature(
                        program_path=entry.program_path,
                        val_set=val_set,
                        metric=metric,
                        temperature=config.eval_temperature,
                        model_name=entry.metadata.get("model", config.student_model_name) if entry.metadata else config.student_model_name,
                        use_cot=config.use_cot,
                        num_threads=config.eval_num_threads,
                    )
                    print(f"    @temp={config.eval_temperature}: "
                          f"total={result['total_score']:.4f}  "
                          f"cls={result['cls_score']:.4f}  "
                          f"chg={result['change_score']:.4f}")

                    # Update entry scores with eval-temperature results
                    entry.val_total_score = result["total_score"]
                    entry.val_cls_score = result["cls_score"]
                    entry.val_change_score = result["change_score"]
                    entry.per_category_scores = result["per_category_scores"]
                    entry.val_avg_total_tokens     = result.get("avg_total_tokens")
                    entry.val_avg_input_tokens     = result.get("avg_input_tokens")
                    entry.val_avg_output_tokens    = result.get("avg_output_tokens")
                    entry.val_avg_reasoning_tokens = result.get("avg_reasoning_tokens")
                    entry.val_eval_time_seconds    = result.get("eval_time_seconds")
                    if entry.metadata is None:
                        entry.metadata = {}
                    entry.metadata["final_eval_temperature"] = config.eval_temperature
                    entry.metadata["final_eval_dataset"] = "full_val"

                except Exception as exc:
                    print(f"    [ERROR] Re-evaluation failed: {exc}")

            # Re-sort leaderboard after updating scores
            leaderboard.entries.sort(
                key=lambda e: e.val_total_score, reverse=True,
            )
            leaderboard.save()

            # ── Present final Pareto front ──────────────────────
            print(f"\n[{_ts()}] {'='*70}")
            print(f" Final Pareto Front (score vs avg tokens/sample)")
            print(f"{'='*70}")
            try:
                pareto_front = leaderboard.pareto_front()
                print(f"  Pareto-optimal candidates ({len(pareto_front)}):")
                for i, pe in enumerate(pareto_front, 1):
                    tok_str = (f"avg_tok/smp={pe.val_avg_total_tokens:.0f}"
                               if pe.val_avg_total_tokens is not None else "avg_tok/smp=?")
                    print(f"    {i}. [{pe.stage}] {pe.config_name}  "
                          f"total={pe.val_total_score:.4f}  {tok_str}")
            except Exception as pex:
                print(f"  [INFO] Pareto front skipped: {pex}")

            # Overall best by score
            overall_best = leaderboard.best()
            if overall_best:
                print(f"\n  Overall best by SCORE: [{overall_best.stage}] {overall_best.config_name}  "
                      f"total={overall_best.val_total_score:.4f}")

            # Overall cheapest by avg tokens/sample (inference cost)
            valid_token_entries = [e for e in leaderboard.entries
                                  if e.val_avg_total_tokens is not None and e.val_avg_total_tokens > 0]
            if valid_token_entries:
                cheapest = min(valid_token_entries, key=lambda e: e.val_avg_total_tokens)
                print(f"  Overall CHEAPEST inference: [{cheapest.stage}] {cheapest.config_name}  "
                      f"avg_tok/smp={cheapest.val_avg_total_tokens:.0f}  total={cheapest.val_total_score:.4f}")
            print(f"{'='*70}")

        # ── Test set evaluation ─────────────────────────────────
        if config.run_test_eval:
            winner = leaderboard.best()
            if winner:
                print(f"\n[{_ts()}] {'='*70}")
                print(f" Test Set Evaluation")
                print(f"{'='*70}")
                print(f"  Winner: {winner.config_name} (total={winner.val_total_score:.4f})")

                try:
                    test_eval_result = _run_test_eval(
                        config=config,
                        winner=winner,
                        split_path=str(cache_split_path),
                    )
                except Exception as exc:
                    print(f"  [ERROR] Test eval failed: {exc}")
                    test_eval_result = {"error": str(exc)}

        # ── Generate & save summary ─────────────────────────────
        total_time = time.time() - t0
        summary = generate_summary(
            config=config,
            leaderboard=leaderboard,
            stages_run=stages_run,
            total_time_seconds=total_time,
            test_eval_result=test_eval_result,
            started_at=started_at,
            stage_top_k_history=stage_top_k_history,
            stage_pre_best=stage_pre_best,
        )

        save_summary(
            summary,
            Path(config.output_dir) / "summary.json",
            Path(config.log_dir) / "summary.json",
        )

        print_final_summary(summary)

        # Print leaderboard table
        leaderboard.print_table(max_rows=20)

        # Export leaderboard to Excel for offline analysis
        try:
            excel_path = Path(config.output_dir) / "leaderboard.xlsx"
            leaderboard.export_to_excel(excel_path)
        except Exception as _xl_exc:
            print(f"  [WARN] Excel export failed: {_xl_exc}")

        # Save leaderboard table as text too (useful for log-based debugging)
        try:
            _table_path = Path(config.log_dir) / "leaderboard_table.txt"
            _table_path.parent.mkdir(parents=True, exist_ok=True)
            with open(_table_path, "w", encoding="utf-8") as _f:
                leaderboard.print_table(file=_f)
            print(f"  Leaderboard text → {_table_path}")
        except Exception:
            pass

        # Save per-stage summaries to log dir
        try:
            for _stage_name in leaderboard.get_stages():
                _stage_sum = leaderboard.get_stage_summary(_stage_name)
                _stage_sum_path = Path(config.log_dir) / f"stage_{_stage_name}_summary.json"
                with open(_stage_sum_path, "w", encoding="utf-8") as _f:
                    json.dump(_stage_sum, _f, indent=2, ensure_ascii=False, default=str)
        except Exception:
            pass

        # Generate Pareto scatter plot (accuracy vs tokens) — always
        if len(leaderboard.entries) > 0:
            plot_path = Path(config.output_dir) / "pareto_plot.png"
            plot_pareto_candidates(leaderboard, plot_path)
            # Also save to log dir
            try:
                import shutil
                log_plot_path = Path(config.log_dir) / "pareto_plot.png"
                shutil.copy2(str(plot_path), str(log_plot_path))
            except Exception:
                pass

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving partial results...")
        total_time = time.time() - t0
        summary = generate_summary(
            config=config,
            leaderboard=leaderboard,
            stages_run=stages_run,
            total_time_seconds=total_time,
            started_at=started_at,
            stage_top_k_history=stage_top_k_history,
            stage_pre_best=stage_pre_best,
        )
        save_summary(
            summary,
            Path(config.output_dir) / "summary_partial.json",
            Path(config.log_dir) / "summary_partial.json",
        )
        print("Partial summary saved.")

    except Exception as exc:
        print(f"\n\nFATAL ERROR: {exc}")
        logging.getLogger("tournament").exception("Tournament failed")
        # Save whatever we have
        try:
            total_time = time.time() - t0
            summary = generate_summary(
                config=config,
                leaderboard=leaderboard,
                stages_run=stages_run,
                total_time_seconds=total_time,
                started_at=started_at,
                stage_top_k_history=stage_top_k_history,
                stage_pre_best=stage_pre_best,
            )
            summary["error"] = str(exc)
            save_summary(
                summary,
                Path(config.output_dir) / "summary_error.json",
                Path(config.log_dir) / "summary_error.json",
            )
        except Exception:
            pass
        raise

    finally:
        tee.teardown()


# ═══════════════════════════════════════════════════════════════════
# Test set evaluation helper
# ═══════════════════════════════════════════════════════════════════

def _run_test_eval(
    config: TournamentConfig,
    winner,
    split_path: str,
) -> dict:
    """Run the existing test eval script as a subprocess.

    Uses ``evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py``
    which already handles per-pair scoring with rule-based + LLM judge metrics.
    """
    eval_script = Path("evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py")
    if not eval_script.exists():
        print(f"  Test eval script not found: {eval_script}")
        return {"error": f"Script not found: {eval_script}"}

    output_dir = Path(config.output_dir) / "test_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(eval_script),
        "--model-path", winner.program_path,
        "--model-name", winner.metadata.get("model", config.student_model_name) if winner.metadata else config.student_model_name,
        "--split-file", split_path,
        "--temperature", str(config.eval_temperature),
        "--which-set", "test",
        "--output-dir", str(output_dir),
    ]

    if not config.use_cot:
        cmd.append("--no-cot")

    print(f"  Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 minute timeout
    )

    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"  [WARN] Test eval exited with code {result.returncode}")
        if result.stderr:
            print(f"  stderr: {result.stderr[-500:]}")

    # Try to parse results from output dir
    test_results_files = list(output_dir.glob("eval_results_*_test.json"))
    if test_results_files:
        latest = max(
            test_results_files,
            key=lambda p: (p.stat().st_mtime, p.stat().st_size),
        )
        with open(latest, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        if isinstance(test_data, dict) and test_data.get("results"):
            best_result = max(
                test_data["results"],
                key=lambda r: r.get("avg_total_score", float("-inf")),
            )
            return {
                "raw": test_data,
                "avg_total_score": best_result.get("avg_total_score"),
                "avg_cls_score": best_result.get("avg_cls_score"),
                "avg_change_score": best_result.get("avg_change_score"),
                "n_evaluated": best_result.get("n_evaluated"),
                "label": best_result.get("label"),
                "model_name": best_result.get("model_name"),
            }
        return test_data
    else:
        return {
            "note": "Test eval completed but no result file found",
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-500:],
        }


if __name__ == "__main__":
    main()
