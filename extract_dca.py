"""
DCA (Document Change Analysis) Prompt Optimization Pipeline
============================================================

Compares original document images against re-imaged versions and classifies
each pair as:
  discard_image = "true"   — images are identical or differences are trivial
                             (resolution change, watermark added, minor border)
  discard_image = "false"  — meaningful changes exist (text edits, removed
                             content, diagram modifications, cross-out marks)

Also, provides a change summary between the original and re-imaged versions in an image pair, based on the following rules:



This script uses DSPy prompt optimization to improve the accuracy of that
binary classification decision and change summary generation.  Four optimization strategies are supported
and can be selected via --optimizer (or run all four at once with "all"):

  bootstrap   BootstrapFewShotWithRandomSearch — explores random candidate
              programs that vary the few-shot demonstration mix; effective
              baseline for any classification task and fast to run.

  miprov2     MIPROv2 — jointly optimizes both the instruction text (system
              prompt) and in-context few-shot demos using Bayesian search over
              the combined candidate space.  More powerful than Bootstrap for
              finding novel instructions, but slower due to more LLM calls.

  gepa        GEPA (Gradient-guided Evolutionary Prompt Augmentation) — uses a
              stronger "reflection" LM to analyze prediction failures and
              generate improved instruction candidates.  Particularly effective
              for binary tasks where false positives and false negatives carry
              different costs: for DCA, missing a real change (false negative) is
              generally more costly than over-flagging an identical pair (false
              positive).  Requires metric_with_feedback (already implemented as
              dca_comparison_metric_with_feedback in evaluation/dca_metric.py) and
              a reflection LM set via REFLECTION_LM_NAME in .env.

  simba       SIMBA (Stochastic Iterative Matching-Based Augmentation) —
              iteratively proposes and selects candidate program modifications
              working entirely on the training set.  Does NOT require a separate
              validation set during compile() — useful when validation data is
              scarce or you want to maximise training signal.

Dataset
-------
~376 human-labelled image pairs from data/dca_data/image_pairs/.
Ground truth annotations stored in data/dca_data/image_pairs_llm_output_exp1/.

Split (random_seed=42, stratified on expected_discard_image to preserve class ratio):
  train  60% (~226 examples) — used by optimizers during compilation
  val    20% (~75 examples)  — used to score each optimizer candidate and pick the best
  test   20% (~75 examples)  — held out; used only for final post-hoc analysis

Hyperparameter Rationale
------------------------
These defaults were chosen for the DCA dataset size (~226 training pairs) and
the multimodal (image) inference context.  Images are large and consume significant
context window tokens, so demo counts are kept lower than for text-only tasks.

Bootstrap (BootstrapFewShotWithRandomSearch):
  max_labeled_demos=3–4       Keep low: each image demo adds substantial tokens
  max_bootstrapped_demos=2–3  Teacher-generated examples; a few well-chosen ones suffice
  num_candidate_programs=12–16  Enough to cover the random search space without
                                excess API calls
  max_rounds=2                One extra refinement round; more risks overfitting the
                              small train set

MIPROv2:
  auto="medium"               10 Bayesian trials; appropriate for ~226 training image
                              pairs and typical Azure OpenAI token budgets.
                              "light" = 5 trials (quick sanity check),
                              "heavy" = 25 trials (best quality, expensive)
  max_labeled_demos=3         Balanced for image context overhead
  max_bootstrapped_demos=3    Symmetric with labeled demos

GEPA:
  auto="light"                Default iteration budget; bump to "medium" if budget allows
  use_merge=True              Merges the best elements of multiple prompt candidates
                              rather than picking a single winner — reduces variance
  track_stats=True            Logs per-iteration optimization stats to stdout; set False
                              to reduce verbosity in production runs
  reflection_lm               Must be a stronger model than the student; set via
                              REFLECTION_LM_NAME in .env (e.g. azure/gpt-5).
                              Exposed as reflectin_lm in config.py — the spelling
                              is an intentional project-wide convention; do not rename.

SIMBA:
  max_steps=8                 Iterative optimization steps over the training set;
                              increase to 12–16 for deeper (slower) search
  bsize=32                    ~14% of the 226-example train set per evaluation batch —
                              large enough for a reliable signal, small enough to be fast
  num_candidates=6            Candidate modifications generated per step; sweet spot
                              between exploration depth and API cost for image inference
  max_labeled_demos=4         Maximum in-context demos included in the compiled program

Usage Examples
--------------
Run with default settings (MIPROv2 medium, test-set analysis only):
    uv run python extract_dca.py

Specify optimizer and intensity:
    uv run python extract_dca.py --optimizer miprov2 --miprov2-auto heavy

Run all four optimizers and analyze both val and test sets:
    uv run python extract_dca.py --optimizer all --analyze-sets both

Run with a different student model and save everything:
    uv run python extract_dca.py \\
        --model-name azure/gpt-4o \\
        --optimizer gepa --gepa-auto medium \\
        --save-model --save-results

Override data paths explicitly (useful when running from a different CWD):
    uv run python extract_dca.py \\
        --pairs-dir data/dca_data/image_pairs \\
        --gt-dir    data/dca_data/image_pairs_llm_output_exp1

On Windows, set PYTHONIOENCODING first to avoid Unicode issues in the terminal:
    $env:PYTHONIOENCODING="utf-8"; uv run python extract_dca.py

Environment Variables (all optional — CLI flags take priority over env vars)
-----------------------------------------------------------------------------
  MODEL_NAME              Student LM used for inference (e.g. azure/gpt-4o-mini)
  REFLECTION_LM_NAME      Reflection LM for GEPA optimizer (e.g. azure/gpt-5)
  DCA_IMAGE_PAIRS_DIR     Path to image pairs directory
  DCA_GT_DIR              Path to ground truth JSON directory
  CACHED_EXTRACTOR_PATH   Path to a previously saved .json extractor to load
  CACHED_EXTRACTOR_DIR    Directory where new optimized extractors are saved
"""

import dspy
from signatures.dca import DCAImageComparison
from utils import load_dca_dataset, stratified_split_yanase_dataset
from config import *
from config import reconfigure_lm
from evaluation.dca_metric import dca_comparison_metric, dca_comparison_metric_with_feedback
from optimization import (
    OptimizationRunner,
    create_bootstrap_strategy,
    create_miprov2_strategy,
    create_simba_strategy,
    create_gepa_strategy,
)
from evaluation.posthoc_analysis import analyze_dca_performance

from datetime import datetime
from typing import List, Dict, Any
import json
import logging
import os
import dotenv
import argparse

dotenv.load_dotenv(override=True)

# ── Argument Parsing ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="DCA Image Comparison Optimization with MIPROv2")

# Data path arguments
parser.add_argument("--pairs-dir", type=str, default=None,
                    help='Path to DCA image pairs directory '
                         '(default: from env DCA_IMAGE_PAIRS_DIR or '
                         '"data/dca_data/image_pairs")')
parser.add_argument("--gt-dir", type=str, default=None,
                    help='Path to DCA ground truth JSON directory '
                         '(default: from env DCA_GT_DIR or '
                         '"data/dca_data/image_pairs_llm_output_exp1")')

# Model and caching arguments
parser.add_argument("--model-name", type=str, default=None,
                    help="Model name to use (default: from .env MODEL_NAME)")
parser.add_argument("--no-cache", action="store_true",
                    help="Disable LLM response caching (default: caching enabled)")
parser.add_argument("--load-extractor", type=str, default=None,
                    help="Path to previously saved extractor to load")
parser.add_argument("--skip-load", action="store_true",
                    help="Skip loading cached extractor even if it exists")

# Analysis arguments
parser.add_argument("--analyze-sets", type=str, default="test", choices=["val", "test", "both"],
                    help="Which dataset to analyze: val, test, or both (default: test)")
parser.add_argument("--save-model", action="store_true", default=False,
                    help="Save the optimized model after training")
parser.add_argument("--save-results", action="store_true", default=False,
                    help="Save performance analysis results to JSON")

# Optimizer arguments
parser.add_argument("--optimizer", type=str, default="miprov2",
                    choices=["miprov2", "bootstrap", "gepa", "simba", "all"],
                    help="Which optimizer to use: miprov2, bootstrap, gepa, simba, or all (default: miprov2)")
parser.add_argument("--miprov2-auto", type=str, default="medium",
                    choices=["light", "medium", "heavy"],
                    help="MIPROv2 intensity: light, medium, or heavy (default: medium)")
parser.add_argument("--simba-max-steps", type=int, default=8,
                    help="SimBA max optimization steps (default: 8)")
parser.add_argument("--gepa-auto", type=str, default="light",
                    choices=["light", "medium", "heavy"],
                    help="GEPA intensity preset (default: light)")

args = parser.parse_args()

# ── Logging Setup ────────────────────────────────────────────────
# Writes structured logs to log/dca_run_<timestamp>.log (and echoes to console).
# Key events captured:
#   INFO  — dataset loading summary, AbsentImage one-sided pairs, optimizer phases
#   DEBUG — AbsentImage.format() calls (every LLM invocation for absent slots)
_log_run_time = datetime.now().strftime("%Y%m%d%H%M")
_log_dir = "log"
os.makedirs(_log_dir, exist_ok=True)
_log_path = os.path.join(_log_dir, f"dca_run_{_log_run_time}.log")
_log_fmt = logging.Formatter(
    "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_file_handler = logging.FileHandler(_log_path, encoding="utf-8")
_file_handler.setFormatter(_log_fmt)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_fmt)

# Root logger: WARNING only (suppresses LiteLLM / httpcore / httpx noise)
logging.basicConfig(level=logging.WARNING, handlers=[_file_handler, _console_handler])

# Our own namespaces: DEBUG (AbsentImage format() calls) and INFO
for _ns in ("extract_dca", "dca.data_loading", "dca"):
    logging.getLogger(_ns).setLevel(logging.DEBUG)

_run_logger = logging.getLogger("extract_dca")
_run_logger.info("DCA optimization run started — log file: %s", _log_path)


# ── Model Configuration ──────────────────────────────────────────
# Reconfigures DSPy's global LM only when the user explicitly provides
# --model-name or --no-cache.  Otherwise the default LM that was already
# initialized in config.py (from MODEL_NAME in .env) is used as-is.
# Note: reflectin_lm (intentional project-wide typo — missing one 't') is the
# stronger reflection model used exclusively by the GEPA optimizer.  Both
# config.py and this file reference it by the same misspelled name; do not rename.
cache_enabled = not args.no_cache
if args.model_name or args.no_cache:
    model_to_use = args.model_name or os.getenv("MODEL_NAME")
    print(f"\n🔄 Reconfiguring DSPy with model: {model_to_use}, cache: {cache_enabled}")
    reconfigure_lm(model_to_use, cache=cache_enabled)
    print(f"✅ Model configured: {model_to_use}, cache: {cache_enabled}\n")


# ── Data Paths ───────────────────────────────────────────────────
# Resolution order: CLI arg > environment variable > hard-coded default.
# The hard-coded defaults point to data/dca_data/ at the root of this repository.
pairs_dir = args.pairs_dir or os.getenv("DCA_IMAGE_PAIRS_DIR", "data/dca_data/image_pairs")
gt_dir    = args.gt_dir    or os.getenv("DCA_GT_DIR",           "data/dca_data/image_pairs_llm_output_exp1")


# ── Display Configuration ────────────────────────────────────────
print(f"\n{'='*80}")
print(f" DCA Pipeline Configuration:")
print(f"  Model: {args.model_name or os.getenv('MODEL_NAME', 'Unknown')}")
print(f"  Cache: {cache_enabled}")
print(f"  Optimizer: {args.optimizer} (MIPROv2 auto={args.miprov2_auto}, SimBA steps={args.simba_max_steps}, GEPA auto={args.gepa_auto})")
print(f"  Load extractor: {args.load_extractor if not args.skip_load else 'No'}")
print(f"  Analyze sets: {args.analyze_sets}")
print(f"  Save model: {args.save_model}")
print(f"  Save results: {args.save_results}")
print(f"\n Data Paths:")
print(f"  Image pairs: {pairs_dir}")
print(f"  Ground truth: {gt_dir}")
print(f"{'='*80}\n")


# ── Data Loading ─────────────────────────────────────────────────
dataset = load_dca_dataset(pairs_dir, gt_dir)
if len(dataset) == 0:
    print("ERROR: No valid DCA image pairs found. Check your data paths.")
    exit(1)

split_seed = 42

# TODO we are already using seed, we need not to save and load test val everytime
def _pair_names(examples: List[dspy.Example]) -> list[str]:
    return [
        str(getattr(ex, "pair_name"))
        for ex in examples
        if getattr(ex, "pair_name", None) is not None
    ]


split_dir = ".cache"
if not os.path.exists(split_dir):
    os.makedirs(split_dir)

split_latest_path = os.path.join(split_dir, f"dca_split_seed{split_seed}.json")

# ── Load or compute split ────────────────────────────────────────
# The manifest stores only val and test pair names (+ seed).
# Train = all dataset pairs not in val or test — any new pairs added to the
# dataset are automatically absorbed into train.
# Delete .cache/dca_split_seed42.json to force a fresh stratified split.
dataset_by_name = {str(getattr(ex, "pair_name", "")): ex for ex in dataset}

# TODO Make a dca config -> move dca related configs such as stratify to a new dca config file and import here
train_set, val_set, test_set = stratified_split_yanase_dataset(
    dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
    random_seed=split_seed, stratify_by=["expected_discard_image","expected_human_label_category","doubt"], verbose=True
)

val_names  = _pair_names(val_set)
test_names = _pair_names(test_set)

split_ts = datetime.now().strftime("%Y%m%d%H%M%S")

split_manifest = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "seed": split_seed,
    "stratify_by": ["expected_discard_image","expected_human_label_category","doubt"],
    "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
    "total_examples": len(dataset),
    "val":  {"count": len(val_names),  "pair_names": val_names},
    "test": {"count": len(test_names), "pair_names": test_names},
}

for split_path in [split_latest_path]:
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

print(f"🧩 Saved deterministic split manifest: {split_latest_path}")
print(f"   Val pair count:  {len(val_names)}")
print(f"   Test pair count: {len(test_names)}")

print(f"train set: {len(train_set)}")
print(f"val set:   {len(val_set)}")
print(f"test set:  {len(test_set)}")


# ── Create Extractor ─────────────────────────────────────────────
# dspy.ChainOfThought wraps the signature so the LM produces an internal
# "reasoning" scratchpad before committing to discard_image and changes.
# Chain-of-thought prompting improves accuracy on visually ambiguous pairs
# where the model needs to reason through subtle differences before deciding.
# optimizer.compile() mutates this object in-place; the result is the
# optimized comparator that can be serialized with comparator.save(<path>).
comparator = dspy.ChainOfThought(DCAImageComparison)

# Load cached extractor if available
if not args.skip_load:
    cached_extractor_path = args.load_extractor or dotenv.get_key(dotenv.find_dotenv(), "CACHED_EXTRACTOR_PATH")
    if cached_extractor_path and os.path.exists(cached_extractor_path) and os.path.isfile(cached_extractor_path):
        print(f"📂 Loading cached extractor from: {cached_extractor_path}")
        comparator.load(cached_extractor_path)
        print("✅ Extractor loaded successfully\n")
    else:
        print(f"ℹ️  No cached extractor found, starting from scratch\n")
else:
    print(f"⏭️  Skipping extractor loading (--skip-load)\n")


# ── Evaluation Setup ─────────────────────────────────────────────
# dspy.Evaluate is a reusable evaluator object.  devset=None here because we
# reassign the devset per call (evaluate(comparator, devset=val_set)).
# num_threads=4 parallelises LLM inference across examples; increase to 8 on
# machines with more cores, but watch Azure OpenAI rate limits (TPM/RPM).
# dca_comparison_metric returns a weighted float 0–1:
#   0.6 × classification_score (1.0 if discard_image correct, else 0.0)
#   0.4 × change_score         (precision-60 / recall-40 F-beta via judge_summary_metric)
# dspy.Evaluate reports scores on a 0–100 scale (× 100).
# display_table=True prints a sample of predictions after each evaluation,
# which helps debug label format issues (e.g. model outputting "True" vs "true").
evaluate = dspy.Evaluate(
    devset=None,
    metric=dca_comparison_metric,
    num_threads=4,
    display_progress=True,
    display_table=True,
)

# TODO Why we need this?
# # ── Baseline Evaluation ──────────────────────────────────────────
# print(f"\n{'='*80}")
# print(f"📊 Baseline Evaluation (Before Optimization)")
# print(f"{'='*80}\n")

# print("Evaluating baseline on validation set...")
# baseline_val_result = evaluate(comparator, devset=val_set)
# baseline_val_score = baseline_val_result.score  # EvaluationResult.score is 0-100 scale
# print(f"Baseline Val Score: {baseline_val_score:.1f}%\n")

# print("Evaluating baseline on test set...")
# baseline_test_result = evaluate(comparator, devset=test_set)
# baseline_test_score = baseline_test_result.score  # EvaluationResult.score is 0-100 scale
# print(f"Baseline Test Score: {baseline_test_score:.1f}%")


# ── Optimization Strategies ──────────────────────────────────────
# Each create_*_strategy() call returns an OptimizationConfig dataclass
# (defined in optimization/opt.py).  OptimizationRunner.run_optimization()
# iterates over the list, compiles the student (comparator) with each optimizer,
# evaluates the result on val_set, and tracks whichever candidate scores highest.
# The best compiled program and its val score are returned at the end.
custom_strategies = []

# TODO Jahnavi to look into
if args.optimizer in ("bootstrap", "all"):
    # Two Bootstrap variants with different labeled/bootstrapped demo ratios.
    # Running both lets us learn whether human-verified ground truth examples
    # or teacher-generated (bootstrapped) examples are more useful for DCA.

    # Balanced variant: equal mix — good starting point to understand the
    # baseline few-shot capability of the model on this task.
    custom_strategies.extend([
        create_bootstrap_strategy(
            name="Balanced_DCA_Bootstrap",
            description="Balanced approach with equal labeled and bootstrapped demos for DCA classification",
            max_labeled_demos=4,        # Image context overhead; each demo adds tokens
            max_bootstrapped_demos=4,   # Teacher-generated demos drawn from correct train predictions
            num_candidate_programs=16,  # Random candidates to evaluate; more = better coverage
            num_threads=4,              # Parallel LLM calls during candidate evaluation
            max_rounds=2,               # Second round refines demo quality; >2 risks overfitting
        ),
        # Label-heavy variant: more human ground truth, fewer bootstrapped demos.
        # Useful if the student model makes systematic bootstrapping errors that
        # would propagate bad examples into the few-shot context.
        create_bootstrap_strategy(
            name="Label_Heavy_DCA_Bootstrap",
            description="Label-heavy approach focusing on ground truth examples for DCA classification",
            max_labeled_demos=6,        # One extra human-verified example vs balanced variant
            max_bootstrapped_demos=4,   # Fewer teacher demos to reduce noise risk
            num_candidate_programs=16,  # Slightly more candidates for broader coverage
            num_threads=4,
            max_rounds=2,
        ),
    ])

if args.optimizer in ("miprov2", "all"):
    # MIPROv2 jointly searches over instruction text AND demo selection using
    # Bayesian optimization (unlike Bootstrap which only varies the demos).
    # The auto preset controls the number of Bayesian trials:
    #   light  = 5 trials  — quick sanity check, limited instruction diversity
    #   medium = 10 trials — recommended for ~226 training pairs and image inference budgets
    #   heavy  = 25 trials — best quality; expensive due to many LLM calls per trial
    custom_strategies.append(
        create_miprov2_strategy(
            name=f"MIPROv2_DCA_{args.miprov2_auto.capitalize()}",
            description=(
                f"MIPROv2 optimizer ({args.miprov2_auto} preset) for DCA image comparison — "
                f"optimizes both instructions and demos via Bayesian search"
            ),
            auto=args.miprov2_auto,     # Intensity preset passed in via --miprov2-auto
            max_labeled_demos=3,        # Balanced for image context overhead
            max_bootstrapped_demos=3,   # Symmetric with labeled demos
            num_threads=4,
        ),
    )

# TODO Jahnavi to look into
if args.optimizer in ("gepa", "all"):
    # GEPA uses a stronger reflection_lm (set via REFLECTION_LM_NAME in .env)
    # to analyze failure cases from the training set and generate improved
    # instruction candidates — a form of automatic prompt debugging.
    # DCA is a good fit for GEPA because false negatives (model misses a real
    # change) and false positives (model flags identical images) have different
    # downstream costs; GEPA's feedback mechanism can surface these patterns.
    # use_merge=True: GEPA merges the best elements of multiple candidate prompts
    # rather than picking a single winner — reduces output variance.
    # track_stats=True: logs per-iteration statistics to stdout; set to False
    # in production runs to reduce verbosity.
    # dca_comparison_metric_with_feedback (used as metric_with_feedback in the
    # OptimizationRunner) provides structured error explanations (FALSE POSITIVE /
    # FALSE NEGATIVE) that GEPA uses to target instruction improvements.
    custom_strategies.append(
        create_gepa_strategy(
            name=f"GEPA_DCA_{args.gepa_auto.capitalize()}",
            description=(
                f"GEPA optimizer for DCA image comparison — "
                f"uses reflection LM to generate feedback and refine prompts"
            ),
            auto=args.gepa_auto,    # Intensity preset passed in via --gepa-auto
            num_threads=4,
            track_stats=True,       # Log per-step stats for debugging
            use_merge=True,         # Merge best aspects of candidate prompts
        ),
    )

if args.optimizer in ("simba", "all"):
    # SIMBA is unique among the four optimizers: it does NOT require a validation
    # set during compile() — it works entirely on the training set.  This is
    # handled automatically in OptimizationRunner._run_strategy() which skips
    # passing valset= when the optimizer_type is "simba".
    # After compilation, SIMBA's result is still evaluated on val_set for a fair
    # comparison with the other strategies.
    custom_strategies.append(
        create_simba_strategy(
            name=f"SimBA_DCA_Steps{args.simba_max_steps}",
            description=(
                f"SimBA optimizer ({args.simba_max_steps} steps) for DCA image comparison — "
                f"iterative prompt refinement with candidate selection"
            ),
            max_steps=args.simba_max_steps,  # CLI-configurable; default 8; increase for deeper search
            max_labeled_demos=4,             # Max in-context demos in the compiled program
            num_threads=4,
            bsize=32,           # Eval batch size; ~14% of 226-example train set — stable signal
            num_candidates=6,   # Candidate modifications generated per step; 6 is a sweet spot
        ),
    )

print(f"\n📋 Optimization strategies ({len(custom_strategies)}):")
for i, s in enumerate(custom_strategies, 1):
    print(f"  {i}. {s.name} ({s.optimizer_type}): {s.description}")

# TODO Jahnavi to look into reflection and metric with feedback
runner = OptimizationRunner(
    metric=dca_comparison_metric,
    student_model=comparator,
    train_set=train_set,
    val_set=val_set,
    evaluate_fn=evaluate,
    # metric_with_feedback=dca_comparison_metric_with_feedback,
    # reflection_lm=reflectin_lm,
)

print(f"\n{'='*80}")
print(f"🚀 Starting Optimization...")
print(f"{'='*80}\n")
optimized_comparator, best_score = runner.run_optimization(custom_strategies)

# Fall back to unoptimized comparator if no strategy improved over baseline
if optimized_comparator is None:
    print("⚠️  No strategy beat the baseline — using unoptimized comparator for analysis/save.")
    optimized_comparator = comparator

# Calculate and display improvement
print(f"\n{'='*80}")
print(f"📈 Optimization Results Summary")
print(f"{'='*80}")
print(f"Best Strategy Val Score: {best_score:.1f}%")
print(f"{'='*80}\n")


# TODO Review, we should use same metrics as DCA metric - No extra fancy
# TODO Not needed
# ── Post-hoc Analysis ────────────────────────────────────────────
# analyze_dca_performance() runs the optimized comparator on the specified
# split and returns a detailed dict covering two evaluation dimensions:
#
#   Classification (discard_image field):
#     accuracy, precision, recall, F1, full confusion matrix (TP/TN/FP/FN),
#     per-actual-class breakdown (how many TP/FN for discard, TN/FP for keep).
#
#   Change summary quality (changes field):
#     avg_change_score  — mean weighted F-beta (precision 60%, recall 40%)
#     change_quality    — per-label counts: Exact match / Good / Partial / No match
#     Co-failure analysis — pairs that failed both classification AND summary
#     FP/FN drill-down  — change quality broken out by error type
#
# This is separate from dspy.Evaluate (which only reports an aggregate score)
# and provides the per-class + per-dimension breakdown needed to diagnose
# error patterns and guide further prompt optimization.
# Results are printed to stdout and optionally serialized to JSON below.
# print(f"\n{'='*80}")
# print(f"📊 Performance Analysis")
# print(f"{'='*80}\n")

# all_results = {}

# if args.analyze_sets in ["val", "both"]:
#     print(f"Analyzing validation set ({len(val_set)} examples)...")
#     val_results = analyze_dca_performance(optimized_comparator, val_set)
#     chq = val_results.get("change_quality", {})
#     print(f"\n✅ Validation — accuracy: {val_results['accuracy']:.3f} "
#           f"| avg change score: {val_results.get('avg_change_score') or 0:.3f} "
#           f"| change quality: Exact={chq.get('Exact match',0)}, "
#           f"Good={chq.get('Good',0)}, Partial={chq.get('Partial',0)}, "
#           f"No match={chq.get('No match',0)}\n")
#     all_results["val"] = val_results

# if args.analyze_sets in ["test", "both"]:
#     print(f"Analyzing test set ({len(test_set)} examples)...")
#     test_results = analyze_dca_performance(optimized_comparator, test_set)
#     chq = test_results.get("change_quality", {})
#     print(f"\n✅ Test — accuracy: {test_results['accuracy']:.3f} "
#           f"| avg change score: {test_results.get('avg_change_score') or 0:.3f} "
#           f"| change quality: Exact={chq.get('Exact match',0)}, "
#           f"Good={chq.get('Good',0)}, Partial={chq.get('Partial',0)}, "
#           f"No match={chq.get('No match',0)}\n")
#     all_results["test"] = test_results


# ── Save Results ──────────────────────────────────────────────────
# File naming convention: {model}_{strategy}_{pipeline}_{YYYYMMDDHH}.{ext}
# The hour suffix (HH) lets you run the pipeline multiple times on the same
# day without overwriting previous results, while keeping filenames sortable.
# optimized_comparator.save(<path>) serializes the DSPy program state — the
# optimized instruction text, any few-shot demos, and module metadata — to a
# JSON file.  It can be reloaded on a fresh run via --load-extractor <path>.
# misclassified and results (per-example list) are excluded from the serialized
# JSON to keep file sizes manageable; their counts are stored instead.
model_name = (args.model_name or os.getenv("MODEL_NAME", "Unknown_Model")).replace("/", "_")
current_time = datetime.now().strftime("%Y%m%d%H")
best_strategy_name = runner.best_strategy_name.replace(" ", "_") if hasattr(runner, "best_strategy_name") else "Unknown"

if args.save_model or args.save_results:
    # Create directory if not exists
    cache_dir = dotenv.get_key(dotenv.find_dotenv(), "CACHED_EXTRACTOR_DIR") or ".cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Save optimized model
    if args.save_model:
        model_path = os.path.join(
            cache_dir,
            f"{model_name}_{best_strategy_name}_dca_optimized_{current_time}.json",
        )
        optimized_comparator.save(model_path)
        print(f"\n💾 Saved optimized model to: {model_path}")

    # TODO upadte
    # # Save performance results
    # if args.save_results:
    #     results_path = os.path.join(
    #         cache_dir,
    #         f"{model_name}_{best_strategy_name}_dca_analysis_{current_time}.json",
    #     )
    #     serializable_results = {}
    #     for split_name, result_dict in all_results.items():
    #         serializable_results[split_name] = {
    #             k: v for k, v in result_dict.items()
    #             if k not in ("misclassified", "results")
    #         }
    #         serializable_results[split_name]["misclassified_count"] = len(
    #             result_dict.get("misclassified", [])
    #         )
    #         serializable_results[split_name]["results_count"] = len(
    #             result_dict.get("results", [])
    #         )

    #     with open(results_path, "w") as f:
    #         json.dump(
    #             {
    #                 "model": model_name,
    #                 "best_strategy": best_strategy_name,
    #                 "best_score": best_score,
    #                 "optimizer": args.optimizer,
    #                 "miprov2_auto": args.miprov2_auto,
    #                 "gepa_auto": args.gepa_auto,
    #                 "simba_max_steps": args.simba_max_steps,
    #                 "analyze_sets": args.analyze_sets,
    #                 "train_size": len(train_set),
    #                 "val_size": len(val_set),
    #                 "test_size": len(test_set),
    #                 "results": serializable_results,
    #             },
    #             f,
    #             indent=2,
    #         )
    #     print(f"💾 Saved performance results to: {results_path}")

print(f"\n{'='*80}")
print(f"✅ DCA Pipeline complete!")
print(f"{'='*80}\n")


# ── JSON Logging ──────────────────────────────────────────────────
# Dumps the complete LLM call history (prompts sent, responses received,
# token usage) to a timestamped JSON file in the log/ directory.
# Useful for post-run debugging of unexpected model outputs and for auditing
# total token consumption after a long optimization run.
# The `default=` lambda handles DSPy internal objects that are not natively
# JSON-serializable: it calls .to_dict() if the object has that method,
# otherwise falls back to str().  This prevents serialization errors without
# silently dropping data.
log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

history_path = os.path.join(log_dir, f"llm_history_dca_{current_time}.json")
try:
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            dspy.settings.lm.history,
            f,
            indent=2,
            ensure_ascii=False,
            default=lambda x: getattr(x, "to_dict", lambda: str(x))(),
        )
    print(f"💾 Saved LLM history to: {history_path}")
except Exception as e:
    print(f"⚠️ Failed to save LLM history: {e}")
