"""Isolated compile() worker for the tournament pipeline.

Invoked as a subprocess by ``stage_runner._spawn_optimizer()``.
Runs in a fresh Python interpreter with no inherited DSPy ContextVar state,
no nested thread pools from the parent, and no Azure client race conditions.

Exit codes:
    0 — compiled program saved to ``output_path`` successfully
    1 — any failure (traceback printed to stderr)

Config JSON (written by stage_runner, path passed via --config):
    stage                  : "bootstrap" | "miprov2" | "gepa"
    seed_path              : str  — path to saved seed .json program
    output_path            : str  — where to save the compiled program
    use_cot                : bool
    instructions_only      : bool

    # Data (must match parent's split exactly)
    pairs_dir              : str
    gt_dir                 : str
    split_seed             : int
    train_ratio            : float
    val_ratio              : float
    test_ratio             : float
    mini_train_size        : int | null
    mini_val_size          : int | null

    # LM
    student_model_name     : str
    optimization_temperature: float | null

    # Stage-specific blocks (only the one matching ``stage`` is used)
    bootstrap_cfg          : dict | null
        n, max_rounds, max_labeled_demos, max_bootstrapped_demos
    bootstrap_teacher_model: str | null
    bootstrap_stop_at_score: float | null
    bootstrap_num_threads  : int

    miprov2_cfg            : dict | null
        auto, model_name, max_labeled_demos, max_bootstrapped_demos, clear_field_descs
    miprov2_prompt_model   : str | null
    miprov2_prompt_model_reasoning_effort: str | null
    miprov2_minibatch_size : int | null
    miprov2_metric_threshold: float | null
    miprov2_teacher_model  : str | null
    miprov2_num_threads    : int

    gepa_cfg               : dict | null
        auto, use_merge, track_stats, reflection_minibatch_size, train_size, val_size
    gepa_reflection_model  : str | null
    gepa_reflection_images : bool
    gepa_reflection_image_detail: str
    gepa_reflection_max_images: int
    gepa_add_format_failure_as_feedback: bool
    gepa_perfect_score     : float | null
    gepa_lowest_score      : float | null
    gepa_candidate_selection_strategy: str | null
    gepa_num_threads       : int
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import dotenv

dotenv.load_dotenv()

import dspy

from config import create_lm
from utils import load_dca_dataset, stratified_split_dca_dataset
from evaluation.dca_metric import dca_comparison_metric, dca_comparison_metric_with_feedback
from dca_tournaments.tour_utils import create_mini_dataset, save_candidate, load_candidate

logger = logging.getLogger(__name__)


# ── Timestamp helper ──────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ── P1 enforcement (mirrors stage_runner._load_student) ──────────

def _load_student(seed_path: str, use_cot: bool, instructions_only: bool):
    """Load + deep-copy seed program, enforcing P1 if instructions_only."""
    program = load_candidate(seed_path, use_cot=use_cot)
    student = copy.deepcopy(program)

    if instructions_only:
        for predictor in student.predictors():
            sig = predictor.signature
            desc_parts = []
            for fname, finfo in sig.fields.items():
                desc = ""
                if finfo.json_schema_extra and isinstance(finfo.json_schema_extra, dict):
                    desc = finfo.json_schema_extra.get("desc", "")
                if desc:
                    desc_parts.append(f"- {fname}: {desc}")
            if desc_parts:
                merged = "\n".join(desc_parts)
                current_doc = sig.__doc__ or ""
                new_doc = (
                    f"{current_doc}\n\nField guidance:\n{merged}"
                    if current_doc
                    else f"Field guidance:\n{merged}"
                )
                predictor.signature = sig.with_instructions(new_doc)
            for fname in list(predictor.signature.fields.keys()):
                predictor.signature = predictor.signature.with_updated_fields(fname, desc="")

    return student


# ── Field desc snapshot / normalize (mirrors stage_runner) ────────

def _snapshot_field_descs(program) -> list[dict[str, str]]:
    def _descs(sig):
        return {
            fname: ((finfo.json_schema_extra or {}).get("desc", ""))
            for fname, finfo in sig.fields.items()
        }
    return [_descs(p.signature) for p in program.predictors()]


def _normalize_compiled_field_descs(compiled_program, original_field_descs, stage_name: str,
                                     instructions_only: bool) -> None:
    predictors = list(compiled_program.predictors())

    if instructions_only:
        for predictor in predictors:
            sig = predictor.signature
            field_desc_parts = []
            for fname, finfo in sig.fields.items():
                desc = finfo.json_schema_extra.get("desc", "") if finfo.json_schema_extra else ""
                if desc:
                    field_desc_parts.append(f"- {fname}: {desc}")
            if field_desc_parts:
                merged = "\n".join(field_desc_parts)
                current_doc = sig.__doc__ or ""
                new_doc = (
                    f"{current_doc}\n\nField guidance (from {stage_name.upper()}):\n{merged}"
                    if current_doc
                    else f"Field guidance (from {stage_name.upper()}):\n{merged}"
                )
                predictor.signature = sig.with_instructions(new_doc)
            for fname in list(predictor.signature.fields.keys()):
                predictor.signature = predictor.signature.with_updated_fields(fname, desc="")
        return

    # Non-instructions_only: restore original descs
    restored = 0
    for predictor, snapshot in zip(predictors, original_field_descs):
        current = {
            fname: ((finfo.json_schema_extra or {}).get("desc", ""))
            for fname, finfo in predictor.signature.fields.items()
        }
        for fname, orig_desc in snapshot.items():
            if current.get(fname, "") != orig_desc:
                predictor.signature = predictor.signature.with_updated_fields(fname, desc=orig_desc)
                restored += 1
    if restored:
        print(f"  [DESC] Restored {restored} original field desc(s) after {stage_name.upper()}")


# ── Stage runners ─────────────────────────────────────────────────

def _run_bootstrap(student, train_set, val_set, cfg: dict) -> object:
    from dspy.teleprompt import BootstrapFewShotWithRandomSearch

    bc = cfg["bootstrap_cfg"]
    n_candidates = bc.get("n", 16)
    max_rounds = bc.get("max_rounds", 2)
    num_threads = cfg.get("bootstrap_num_threads", 4)

    bs_kwargs = dict(
        metric=dca_comparison_metric,
        max_labeled_demos=bc["max_labeled_demos"],
        max_bootstrapped_demos=bc["max_bootstrapped_demos"],
        num_candidate_programs=n_candidates,
        max_rounds=max_rounds,
        num_threads=num_threads,
    )
    if cfg.get("bootstrap_stop_at_score") is not None:
        bs_kwargs["stop_at_score"] = cfg["bootstrap_stop_at_score"]
    if cfg.get("bootstrap_teacher_model"):
        bs_kwargs["teacher_settings"] = {"lm": create_lm(cfg["bootstrap_teacher_model"], cache=False)}

    optimizer = BootstrapFewShotWithRandomSearch(**bs_kwargs)
    print(f"  [{_ts()}] Starting Bootstrap.compile() "
          f"(n={n_candidates}, rounds={max_rounds}, "
          f"d{bc['max_labeled_demos']}b{bc['max_bootstrapped_demos']}, "
          f"threads={num_threads})...")
    compiled = optimizer.compile(student=student, trainset=train_set, valset=val_set)
    return compiled


def _run_miprov2(student, train_set, val_set, cfg: dict) -> object:
    from dspy.teleprompt import MIPROv2

    mc = cfg["miprov2_cfg"]
    intensity = mc["auto"]
    max_labeled = mc["max_labeled_demos"]
    max_bootstrapped = mc["max_bootstrapped_demos"]
    num_threads = cfg.get("miprov2_num_threads", 4)
    model_name = mc.get("model_name") or cfg["student_model_name"]

    # Clear field descs if requested
    if mc.get("clear_field_descs"):
        for fname in student.signature.fields:
            student.signature = student.signature.with_updated_fields(fname, desc="")

    mipro_kwargs = dict(
        metric=dca_comparison_metric,
        auto=intensity,
        max_labeled_demos=max_labeled,
        max_bootstrapped_demos=max_bootstrapped,
        num_threads=num_threads,
    )
    if cfg.get("miprov2_prompt_model"):
        pm_kwargs = {"cache": False}
        if cfg.get("miprov2_prompt_model_reasoning_effort"):
            pm_kwargs["reasoning_effort"] = cfg["miprov2_prompt_model_reasoning_effort"]
        mipro_kwargs["prompt_model"] = create_lm(cfg["miprov2_prompt_model"], **pm_kwargs)
    if cfg.get("miprov2_minibatch_size") is not None:
        mipro_kwargs["minibatch_size"] = cfg["miprov2_minibatch_size"]
    if cfg.get("miprov2_metric_threshold") is not None:
        mipro_kwargs["metric_threshold"] = cfg["miprov2_metric_threshold"]
    if cfg.get("miprov2_teacher_model"):
        mipro_kwargs["teacher_settings"] = {"lm": create_lm(cfg["miprov2_teacher_model"], cache=False)}

    optimizer = MIPROv2(**mipro_kwargs)
    print(f"  [{_ts()}] Starting MIPROv2.compile() "
          f"(auto={intensity}, d{max_labeled}b{max_bootstrapped}, threads={num_threads})...")
    compiled = optimizer.compile(student=student, trainset=train_set, valset=val_set)
    return compiled


def _run_gepa(student, train_set, val_set, cfg: dict) -> object:
    try:
        from dspy import GEPA
    except ImportError:
        from dspy.teleprompt import GEPA

    from dca_tournaments.gepa_utils import (
        MultimodalReflectionLM,
        gepa_image_store,
        make_image_storing_metric,
        patch_gepa_reflective_dataset,
    )

    gc = cfg["gepa_cfg"]
    num_threads = cfg.get("gepa_num_threads", 4)
    gepa_reflection_images = cfg.get("gepa_reflection_images", True)

    # Apply the reflective-dataset patch
    patch_gepa_reflective_dataset(send_images=gepa_reflection_images)

    # Choose metric
    if gepa_reflection_images:
        gepa_metric = make_image_storing_metric(dca_comparison_metric_with_feedback)
    else:
        gepa_metric = dca_comparison_metric_with_feedback

    gepa_kwargs: dict = {
        "metric": gepa_metric,
        "auto": gc["auto"],
        "use_merge": gc.get("use_merge", True),
        "track_stats": gc.get("track_stats", True),
        "num_threads": num_threads,
        "reflection_minibatch_size": gc.get("reflection_minibatch_size",
                                            cfg.get("gepa_reflection_minibatch_size", 5)),
        "add_format_failure_as_feedback": cfg.get("gepa_add_format_failure_as_feedback", True),
    }
    if cfg.get("gepa_perfect_score") is not None:
        gepa_kwargs["perfect_score"] = cfg["gepa_perfect_score"]
    if cfg.get("gepa_lowest_score") is not None:
        gepa_kwargs["failure_score"] = cfg["gepa_lowest_score"]
    if cfg.get("gepa_candidate_selection_strategy") is not None:
        strat = cfg["gepa_candidate_selection_strategy"]
        if strat == "greedy":
            strat = "current_best"
        gepa_kwargs["candidate_selection_strategy"] = strat

    # Build reflection LM
    refl_model = cfg.get("gepa_reflection_model") or cfg["student_model_name"]
    refl_kwargs: dict = {"cache": False}
    if cfg.get("optimization_temperature") is not None:
        refl_kwargs["temperature"] = cfg["optimization_temperature"]
    reflection_lm = create_lm(refl_model, **refl_kwargs)
    if gepa_reflection_images:
        reflection_lm = MultimodalReflectionLM(
            base_lm=reflection_lm,
            detail=cfg.get("gepa_reflection_image_detail", "auto"),
            max_images=cfg.get("gepa_reflection_max_images", 6),
        )
    gepa_kwargs["reflection_lm"] = reflection_lm

    # Per-run dataset subsetting
    _gepa_train_sz = gc.get("train_size") or cfg.get("gepa_train_size")
    _gepa_val_sz = gc.get("val_size") or cfg.get("gepa_val_size")
    split_seed = cfg.get("split_seed", 42)
    if _gepa_train_sz and _gepa_train_sz < len(train_set):
        train_set = create_mini_dataset(train_set, _gepa_train_sz, seed=split_seed)
        print(f"  GEPA train subset: {len(train_set)}")
    if _gepa_val_sz and _gepa_val_sz < len(val_set):
        val_set = create_mini_dataset(val_set, _gepa_val_sz, seed=split_seed)
        print(f"  GEPA val subset: {len(val_set)}")

    optimizer = GEPA(**gepa_kwargs)
    print(f"  [{_ts()}] Starting GEPA.compile() (auto={gc['auto']}, threads={num_threads})...")
    compiled = optimizer.compile(student=student, trainset=train_set, valset=val_set)
    return compiled


# ── Main ──────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Tournament optimizer subprocess worker")
    parser.add_argument("--config", required=True, help="Path to worker config JSON")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Config file not found: {cfg_path}", file=sys.stderr)
        return 1

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    stage = cfg["stage"]
    seed_path = cfg["seed_path"]
    output_path = cfg["output_path"]
    use_cot = cfg.get("use_cot", False)
    instructions_only = cfg.get("instructions_only", True)

    print(f"\n{'='*70}")
    print(f"  [{_ts()}] Subprocess worker: stage={stage}  pid={__import__('os').getpid()}")
    print(f"  Seed       : {seed_path}")
    print(f"  Output     : {output_path}")
    print(f"  Model      : {cfg.get('student_model_name', '?')}")
    print(f"{'='*70}")

    t0 = time.time()

    # ── Load & split dataset ───────────────────────────────────────
    print(f"  [{_ts()}] Loading dataset from {cfg['pairs_dir']} ...")
    dataset = load_dca_dataset(cfg["pairs_dir"], cfg["gt_dir"])
    train_set, val_set, _ = stratified_split_dca_dataset(
        dataset,
        train_ratio=cfg.get("train_ratio", 0.6),
        val_ratio=cfg.get("val_ratio", 0.2),
        test_ratio=cfg.get("test_ratio", 0.2),
        random_seed=cfg.get("split_seed", 42),
        verbose=False,
    )
    mini_train = create_mini_dataset(train_set, cfg.get("mini_train_size"), seed=cfg.get("split_seed", 42))
    mini_val = create_mini_dataset(val_set, cfg.get("mini_val_size"), seed=cfg.get("split_seed", 42))
    print(f"  [{_ts()}] Data: full train={len(train_set)}, val={len(val_set)}")
    print(f"  [{_ts()}] Mini: train={len(mini_train)}, val={len(mini_val)}")

    # ── Configure DSPy ────────────────────────────────────────────
    lm_kwargs: dict = {"cache": False}
    if cfg.get("optimization_temperature") is not None:
        lm_kwargs["temperature"] = cfg["optimization_temperature"]
    lm = create_lm(cfg["student_model_name"], **lm_kwargs)
    dspy.configure(lm=lm)
    print(f"  [{_ts()}] DSPy configured with {cfg['student_model_name']}")

    # ── Load seed + P1 enforcement ────────────────────────────────
    student = _load_student(seed_path, use_cot=use_cot, instructions_only=instructions_only)
    original_field_descs = _snapshot_field_descs(student)
    print(f"  [{_ts()}] Seed loaded: {Path(seed_path).name}")

    # ── Run compile() ─────────────────────────────────────────────
    try:
        if stage == "bootstrap":
            compiled = _run_bootstrap(student, mini_train, mini_val, cfg)
        elif stage == "miprov2":
            # MIPROv2 may use a different model per combo
            mc = cfg.get("miprov2_cfg", {})
            model_name = mc.get("model_name") or cfg["student_model_name"]
            if model_name != cfg["student_model_name"]:
                combo_lm_kwargs: dict = {"cache": False}
                if cfg.get("optimization_temperature") is not None:
                    combo_lm_kwargs["temperature"] = cfg["optimization_temperature"]
                dspy.configure(lm=create_lm(model_name, **combo_lm_kwargs))
                print(f"  [{_ts()}] MIPROv2 reconfigured to {model_name}")
            compiled = _run_miprov2(student, mini_train, mini_val, cfg)
        elif stage == "gepa":
            compiled = _run_gepa(student, mini_train, mini_val, cfg)
        else:
            print(f"[ERROR] Unknown stage: {stage}", file=sys.stderr)
            return 1
    except Exception as exc:
        import traceback
        print(f"\n[ERROR] compile() failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    elapsed_compile = time.time() - t0
    print(f"  [{_ts()}] compile() done in {elapsed_compile:.1f}s")

    # ── Normalize field descs (P1 post-compile) ───────────────────
    _normalize_compiled_field_descs(compiled, original_field_descs, stage, instructions_only)

    # ── Save compiled program ─────────────────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(out_path))
    print(f"  [{_ts()}] Saved compiled program → {out_path}")

    total = time.time() - t0
    print(f"  [{_ts()}] Subprocess done in {total:.1f}s")
    print(f"{'='*70}\n")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    sys.exit(main())
