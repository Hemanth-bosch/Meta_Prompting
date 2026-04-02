"""
DCA (Document Change Analysis) Prompt Optimization Pipeline
============================================================

Compares original document images against re-imaged versions and classifies
each pair as:
  discard_image = "true"  -- images are identical or differences are trivial
  discard_image = "false" -- meaningful changes exist

Also provides a change summary describing what changed between the two images.
1. If there's only one image and : it is informational: "NoMatchFound" - discard - false
2. If there's only one image and : it is non informational: "Only one image present and it is non informational - watermark or logo" - discard - true

3. If there are two images and they are both non informational: "Both images are non informational"
4. They are visually identical/trivial changes: "Images are identical - no changes"
In these cases - discard - true

5. If there are meaningful differences, describe the specific changes - discard - false

Other cases:
1. When one of the images is completely crossed out - "Image 1/2 is crossed out"
2. One of the images is tiny/corrupted/unreadable - "Could not infer"


This script uses DSPy prompt optimization to improve the accuracy of both the
classification decision and the change summary.  Four optimization strategies
are supported via --optimizer (or run all four with "all"):

  bootstrap   BootstrapFewShotWithRandomSearch
  miprov2     MIPROv2 (Bayesian instruction + demo search)
  gepa        GEPA (reflection-LM guided prompt improvement)
  simba       SIMBA (iterative prompt refinement on train set only)

Dataset
-------
~376 human-labelled image pairs from data/dca_data/image_pairs/.
Ground truth in data/dca_data/image_pairs_llm_output_exp1/.

Split (seed=42, stratified on discard_image + category + doubt):
  train 60%  |  val 20%  |  test 20%

Usage
-----
    uv run python extract_dca.py
    uv run python extract_dca.py --optimizer miprov2 --miprov2-auto heavy
    uv run python extract_dca.py --optimizer all --save-model
    uv run python extract_dca.py --model-name azure/gpt-4o --optimizer gepa --save-model

On Windows, set PYTHONIOENCODING first to avoid Unicode issues:
    $env:PYTHONIOENCODING="utf-8"; uv run python extract_dca.py

Environment Variables (CLI flags take priority)
-----------------------------------------------
  MODEL_NAME              Student LM (e.g. azure/gpt-4o-mini)
  REFLECTION_LM_NAME      Reflection LM for GEPA (e.g. azure/gpt-5)
  DCA_IMAGE_PAIRS_DIR     Image pairs directory
  DCA_GT_DIR              Ground truth JSON directory
  CACHED_EXTRACTOR_PATH   Previously saved .json extractor to load
  CACHED_EXTRACTOR_DIR    Directory for saving new optimized extractors
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import List

import dotenv
import dspy

from config import reconfigure_lm, reflectin_lm
from signatures.dca import DCAImageComparison
from utils import load_dca_dataset, stratified_split_dca_dataset
from evaluation.dca_metric import dca_comparison_metric, dca_comparison_metric_with_feedback
from optimization import (
    OptimizationRunner,
    create_bootstrap_strategy,
    create_miprov2_strategy,
    create_simba_strategy,
    create_gepa_strategy,
)

dotenv.load_dotenv(override=True)

# ── Argument Parsing ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="DCA Image Comparison Optimization")

parser.add_argument("--pairs-dir", type=str, default=os.getenv("DCA_IMAGE_PAIRS_DIR", "data/dca_data/image_pairs"),
                    help='Image pairs directory (default: env DCA_IMAGE_PAIRS_DIR or "data/dca_data/image_pairs")')
parser.add_argument("--gt-dir", type=str, default=os.getenv("DCA_GT_DIR", "data/dca_data/image_pairs_llm_output_exp1"),
                    help='Ground truth JSON directory (default: env DCA_GT_DIR or "data/dca_data/image_pairs_llm_output_exp1")')
parser.add_argument("--model-name", type=str, default=os.getenv("MODEL_NAME"),
                    help="Student model name (default: from .env MODEL_NAME)")
parser.add_argument("--no-cache", action="store_true",
                    help="Disable LLM response caching")
parser.add_argument("--load-extractor", type=str, default=None,
                    help="Path to previously saved extractor to load")
parser.add_argument("--skip-load", action="store_true",
                    help="Skip loading cached extractor even if it exists")
parser.add_argument("--save-model", action="store_true", default=True,
                    help="Save the optimized model after training")
parser.add_argument("--optimizer", type=str, default="miprov2",
                    choices=["miprov2", "bootstrap", "gepa", "simba", "all"],
                    help="Optimizer to use (default: miprov2)")
parser.add_argument("--miprov2-auto", type=str, default="medium",
                    choices=["light", "medium", "heavy"],
                    help="MIPROv2 intensity (default: medium)")
parser.add_argument("--simba-max-steps", type=int, default=8,
                    help="SimBA max optimization steps (default: 8)")
parser.add_argument("--gepa-auto", type=str, default="light",
                    choices=["light", "medium", "heavy"],
                    help="GEPA intensity preset (default: light)")
parser.add_argument("--reasoning-effort", type=str, default=None,
                    choices=["low", "medium", "high"],
                    help="Reasoning effort for reasoning models (e.g. gpt-5-mini). "
                         "Passed to the LM as reasoning_effort kwarg. (default: None)")
parser.add_argument("--no-cot", action="store_true", default=False,
                    help="Use dspy.Predict instead of dspy.ChainOfThought (no reasoning scratchpad)")

args = parser.parse_args()

# ── Timestamp ────────────────────────────────────────────────────
# Set once here so the log file and saved model share the same timestamp.
current_time = datetime.now().strftime("%Y%m%d%H%M%S")

# ── Log filename prefix ───────────────────────────────────────────
# Compute the same prefix that the saved model will use, so the log file
# is immediately identifiable even if the run crashes before saving.
# For "all" we know all strategies will run; for a single optimizer we know
# the exact strategy name upfront.
_model_display = (args.model_name or os.getenv("MODEL_NAME", "Unknown")).replace("/", "_")
_cot_tag       = "NoCot" if args.no_cot else "CoT"
_re_tag        = f"_RE{args.reasoning_effort}" if args.reasoning_effort else ""
if args.optimizer == "miprov2":
    _strategy_display = f"MIPROv2_DCA_{args.miprov2_auto.capitalize()}"
elif args.optimizer == "bootstrap":
    _strategy_display = "Bootstrap_DCA"
elif args.optimizer == "gepa":
    _strategy_display = f"GEPA_DCA_{args.gepa_auto.capitalize()}"
elif args.optimizer == "simba":
    _strategy_display = f"SimBA_DCA_Steps{args.simba_max_steps}"
else:  # "all"
    _strategy_display = "All"
_log_prefix = f"{_model_display}_{_strategy_display}_{_cot_tag}{_re_tag}"

# ── TeeLogger ────────────────────────────────────────────────────
# Captures all stdout AND stderr (including DSPy's logging output and tqdm
# progress bars) to a log file while still printing to the terminal.
_log_dir = "log"
os.makedirs(_log_dir, exist_ok=True)
_stdout_log_path = os.path.join(_log_dir, f"{_log_prefix}_dca_run_{current_time}.log")


class _TeeLogger:
    """Redirect a stream (stdout or stderr) to both the terminal and a shared
    log file simultaneously.

    Parameters
    ----------
    original_stream:
        The original stream object (``sys.stdout`` or ``sys.stderr``) to
        forward writes to in addition to the file.
    log_file:
        An already-opened, writable text file object shared by both the
        stdout- and stderr-tee instances (avoids opening the same path twice).
    """

    def __init__(self, original_stream, log_file):
        self._original = original_stream
        self._file = log_file

    def write(self, msg: str):
        self._file.write(msg)
        self._file.flush()
        self._original.write(msg)

    def flush(self):
        self._file.flush()
        self._original.flush()

    # Some libraries (e.g. subprocess, tqdm) check for ``fileno``.
    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return self._original.isatty()


# Open the log file once; share the handle between both tee instances.
_log_file = open(_stdout_log_path, "a", encoding="utf-8")
_tee_stdout = _TeeLogger(sys.stdout, _log_file)
_tee_stderr = _TeeLogger(sys.stderr, _log_file)
sys.stdout = _tee_stdout
sys.stderr = _tee_stderr

# ── Logging Setup ────────────────────────────────────────────────
_log_fmt = logging.Formatter(
    "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_file_handler = logging.FileHandler(_stdout_log_path, encoding="utf-8")
_file_handler.setFormatter(_log_fmt)
# Use the original stderr (before the tee replaced it) so log records go to
# the terminal only via the console handler.  The file handler handles writing
# to the log file directly, avoiding double-writes.
_console_handler = logging.StreamHandler(_tee_stderr._original)
_console_handler.setFormatter(_log_fmt)

# Root logger: WARNING only (suppresses LiteLLM / httpcore / httpx noise)
logging.basicConfig(level=logging.WARNING, handlers=[_file_handler, _console_handler])

# Project namespaces: DEBUG + INFO
for _ns in ("extract_dca", "dca.data_loading", "dca"):
    logging.getLogger(_ns).setLevel(logging.DEBUG)

# DSPy creates its StreamHandler lazily (on first optimizer call), at which
# point sys.stderr is already our tee — so DSPy's own handler already writes
# to the log file via the tee.  Do NOT add _file_handler to the dspy logger;
# doing so would cause every DSPy log record to appear twice in the file.

_run_logger = logging.getLogger("extract_dca")
_run_logger.info("DCA optimization run started -- log file: %s", _stdout_log_path)


# ── Model Configuration ──────────────────────────────────────────
cache_enabled = not args.no_cache
if args.model_name or args.no_cache:
    model_to_use = args.model_name or os.getenv("MODEL_NAME")
    lm_kwargs = {}
    if args.reasoning_effort is not None:
        lm_kwargs["reasoning_effort"] = args.reasoning_effort
    re_info = f", reasoning_effort={args.reasoning_effort}" if args.reasoning_effort else ""
    print(f"\nReconfiguring DSPy with model: {model_to_use}, cache: {cache_enabled}{re_info}")
    reconfigure_lm(model_to_use, cache=cache_enabled, **lm_kwargs)
    print(f"Model configured.\n")


# ── Data Paths ───────────────────────────────────────────────────
# Defaults already resolved from env vars in argparse above.
pairs_dir = args.pairs_dir
gt_dir    = args.gt_dir


# ── Display Configuration ────────────────────────────────────────
_re_display  = f" | reasoning_effort={args.reasoning_effort}" if args.reasoning_effort else ""
_cot_display = "Predict (no CoT)" if args.no_cot else "ChainOfThought"
print(f"\n{'='*80}")
print(f" DCA Pipeline Configuration:")
print(f"  Model: {args.model_name or os.getenv('MODEL_NAME', 'Unknown')}{_re_display}")
print(f"  Module: {_cot_display}")
print(f"  Cache: {cache_enabled}")
_opt_detail = {
    "miprov2":   f"auto={args.miprov2_auto}",
    "bootstrap": "BootstrapFewShotWithRandomSearch",
    "gepa":      f"auto={args.gepa_auto}",
    "simba":     f"max_steps={args.simba_max_steps}",
    "all":       f"MIPROv2 auto={args.miprov2_auto}, Bootstrap, GEPA auto={args.gepa_auto}, SimBA steps={args.simba_max_steps}",
}.get(args.optimizer, args.optimizer)
print(f"  Optimizer: {args.optimizer} ({_opt_detail})")
print(f"  Load extractor: {args.load_extractor if not args.skip_load else 'No'}")
print(f"  Save model: {args.save_model}")
print(f"\n Data Paths:")
print(f"  Image pairs: {pairs_dir}")
print(f"  Ground truth: {gt_dir}")
print(f"{'='*80}\n")


# ── Data Loading ─────────────────────────────────────────────────
dataset = load_dca_dataset(pairs_dir, gt_dir)
if len(dataset) == 0:
    print("ERROR: No valid DCA image pairs found. Check your data paths.")
    sys.stdout = _tee_stdout._original
    sys.stderr = _tee_stderr._original
    _log_file.close()
    exit(1)

split_seed = 42

def _pair_names(examples: List[dspy.Example]) -> list[str]:
    return [
        str(getattr(ex, "pair_name"))
        for ex in examples
        if getattr(ex, "pair_name", None) is not None
    ]

split_dir = ".cache"
os.makedirs(split_dir, exist_ok=True)
split_latest_path = os.path.join(split_dir, f"dca_split_seed{split_seed}.json")

# Stratified split: train 60%, val 20%, test 20%
# Delete .cache/dca_split_seed42.json to force a fresh split.
train_set, val_set, test_set = stratified_split_dca_dataset(
    dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
    random_seed=split_seed, verbose=True
)

val_names  = _pair_names(val_set)
test_names = _pair_names(test_set)

split_manifest = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "seed": split_seed,
    "stratify_by": ["expected_discard_image", "expected_human_label_category", "doubt"],
    "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
    "total_examples": len(dataset),
    "val":  {"count": len(val_names),  "pair_names": val_names},
    "test": {"count": len(test_names), "pair_names": test_names},
}

with open(split_latest_path, "w", encoding="utf-8") as f:
    json.dump(split_manifest, f, indent=2)

print(f"Saved split manifest: {split_latest_path}")
print(f"  train: {len(train_set)}  |  val: {len(val_set)}  |  test: {len(test_set)}")


# ── Create Extractor ─────────────────────────────────────────────
# ChainOfThought: LM reasons step-by-step before answering (more accurate,
#                 more tokens).
# Predict (--no-cot): LM answers directly (faster, cheaper; test whether
#                     reasoning is actually needed for this task).
# optimizer.compile() mutates this object; the result can be serialized
# with comparator.save(<path>).
if args.no_cot:
    comparator = dspy.Predict(DCAImageComparison)
else:
    comparator = dspy.ChainOfThought(DCAImageComparison)

if not args.skip_load:
    cached_extractor_path = args.load_extractor or dotenv.get_key(dotenv.find_dotenv(), "CACHED_EXTRACTOR_PATH")
    if cached_extractor_path and os.path.exists(cached_extractor_path) and os.path.isfile(cached_extractor_path):
        print(f"Loading cached extractor from: {cached_extractor_path}")
        comparator.load(cached_extractor_path)
        print("Extractor loaded successfully\n")
    else:
        print("No cached extractor found, starting from scratch\n")
else:
    print("Skipping extractor loading (--skip-load)\n")


# ── Evaluation Setup ─────────────────────────────────────────────
# dca_comparison_metric returns a weighted float 0-1:
#   0.6 * classification_score + 0.4 * change_score
# dspy.Evaluate reports scores on a 0-100 scale.
evaluate = dspy.Evaluate(
    devset=None,
    metric=dca_comparison_metric,
    num_threads=4,
    display_progress=True,
    display_table=True,
)


# ── Optimization Strategies ──────────────────────────────────────
cache_dir = dotenv.get_key(dotenv.find_dotenv(), "CACHED_EXTRACTOR_DIR") or ".cache"

custom_strategies = []

if args.optimizer in ("bootstrap", "all"):
    custom_strategies.extend([
        create_bootstrap_strategy(
            name="Balanced_DCA_Bootstrap",
            description="Balanced approach with equal labeled and bootstrapped demos for DCA classification",
            max_labeled_demos=4,        # Image context overhead; each demo adds tokens
            max_bootstrapped_demos=4,   # Teacher-generated demos drawn from correct train predictions
            num_candidate_programs=32,  # Random candidates to evaluate; more = better coverage
            num_candidate_traces=(2,5), # Generate 2-5 candidate traces per example for bootstrapping
            num_threads=4,              # Parallel LLM calls during candidate evaluation
            max_rounds=2,               # Second round refines demo quality; >2 risks overfitting
        ),
        create_bootstrap_strategy(
            name="Label_Heavy_DCA_Bootstrap",
            description="Label-heavy approach focusing on ground truth examples for DCA classification",
            max_labeled_demos=6,        # One extra human-verified example vs balanced variant
            max_bootstrapped_demos=4,   # Fewer teacher demos to reduce noise risk
            num_candidate_programs=32,  # Slightly more candidates for broader coverage
            num_candidate_traces=(2,5), # Generate 2-5 candidate traces per example for bootstrapping
            num_threads=4,
            max_rounds=2,
        ),
    ])

if args.optimizer in ("miprov2", "all"):
    custom_strategies.append(
        create_miprov2_strategy(
            name=f"MIPROv2_DCA_{args.miprov2_auto.capitalize()}",
            description=f"MIPROv2 ({args.miprov2_auto}) -- Bayesian instruction + demo search",
            auto=args.miprov2_auto,
            max_labeled_demos=6,
            max_bootstrapped_demos=6,
            num_threads=4,
        ),
    )

if args.optimizer in ("gepa", "all"):
    custom_strategies.append(
        create_gepa_strategy(
            name=f"GEPA_DCA_{args.gepa_auto.capitalize()}",
            description=f"GEPA ({args.gepa_auto}) -- reflection-LM guided prompt improvement",
            auto=args.gepa_auto,
            num_threads=4,
            track_stats=True,
            use_merge=True,
        ),
    )

if args.optimizer in ("simba", "all"):
    custom_strategies.append(
        create_simba_strategy(
            name=f"SimBA_DCA_Steps{args.simba_max_steps}",
            description=f"SimBA ({args.simba_max_steps} steps) -- iterative prompt refinement",
            max_steps=args.simba_max_steps,
            max_labeled_demos=4,
            num_threads=4,
            bsize=32,
            num_candidates=6,
        ),
    )

print(f"\nOptimization strategies ({len(custom_strategies)}):")
for i, s in enumerate(custom_strategies, 1):
    print(f"  {i}. {s.name} ({s.optimizer_type}): {s.description}")

runner = OptimizationRunner(
    metric=dca_comparison_metric,
    student_model=comparator,
    train_set=train_set,
    val_set=val_set,
    evaluate_fn=evaluate,
    metric_with_feedback=dca_comparison_metric_with_feedback,
    reflection_lm=reflectin_lm,
)

print(f"\n{'='*80}")
print(f"Starting Optimization...")
print(f"{'='*80}\n")
optimized_comparator, best_score = runner.run_optimization(custom_strategies)

if optimized_comparator is None:
    print("No strategy beat the baseline -- using unoptimized comparator.")
    optimized_comparator = comparator

print(f"\n{'='*80}")
print(f"Optimization Results Summary")
print(f"{'='*80}")
print(f"Best Strategy Val Score: {best_score:.1f}%")
print(f"{'='*80}\n")


# ── Save Model ───────────────────────────────────────────────────
# File naming: {model}_{strategy}_{cot}[_RE{effort}]_dca_optimized_{YYYYMMDDHHmmss}.json
# The CoT/NoCot and RE suffix ensure runs with different --no-cot / --reasoning-effort
# flags never overwrite each other's saved files.
model_name = (args.model_name or os.getenv("MODEL_NAME", "Unknown_Model")).replace("/", "_")
best_strategy_name = runner.best_strategy_name.replace(" ", "_") if hasattr(runner, "best_strategy_name") else "Unknown"
cot_tag = "NoCot" if args.no_cot else "CoT"
re_tag  = f"_RE{args.reasoning_effort}" if args.reasoning_effort else ""

model_path = None
if args.save_model:
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(
        cache_dir,
        f"{model_name}_{best_strategy_name}_{cot_tag}{re_tag}_dca_optimized_max_labelled_6_bootstrapped_6_{current_time}.json",
    )
    optimized_comparator.save(model_path)
    print(f"\nSaved optimized model to: {model_path}")

print(f"\n{'='*80}")
print(f"DCA Pipeline complete!")
print(f"{'='*80}\n")


# ── Close TeeLogger and rename log to match saved model ──────────
# For runs where a model was saved, rename the log file to share the same
# base name as the model JSON (easy to find the log for a given model file).
sys.stdout = _tee_stdout._original
sys.stderr = _tee_stderr._original
_log_file.close()
if model_path is not None:
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    final_log_path = os.path.join(_log_dir, f"{model_stem}.log")
    try:
        os.rename(_stdout_log_path, final_log_path)
        print(f"Log renamed to: {final_log_path}", file=sys.__stdout__)
    except OSError as e:
        print(f"Could not rename log file: {e}", file=sys.__stdout__)
