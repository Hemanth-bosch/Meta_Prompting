"""
DCA (Document Change Analysis) Prompt Optimization Pipeline
============================================================

Compares original document images against re-imaged versions and classifies
each pair as:
  discard_image = "true"  -- images are identical or differences are trivial
  discard_image = "false" -- meaningful changes exist

Also provides a change summary describing what changed between the two images.
1. If there's only one image and : it is informational: "NoMatchFound" - discard - false
2. If there's only one image and : it is non informational: "Only one image and it is non informational" - discard - true

3. If there are two images and they are both non informational: "Both images are non informational and contain only logo/watermark etc. no meaningful content"
4. They are visually identical/trivial changes: "Images are identical - no changes"
In these cases - discard - true

5. If there are meaningful differences, describe the specific changes - discard - false

Other cases:
1. When one of the images is completely crossed out - "Image has been crossed out"
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
    uv run python extract_dca.py --optimizer all
    uv run python extract_dca.py --model-name azure/gpt-4o --optimizer gepa

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
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import dotenv
import dspy

from config import reconfigure_lm, reflectin_lm, create_lm
from performance_class_adaptive_dca import PerformanceClassAdaptiveSampler
from signatures.dca import DCAImageComparison
from utils import load_dca_dataset, stratified_split_dca_dataset
from evaluation.dca_metric import dca_comparison_metric, dca_comparison_metric_with_feedback, select_hard_examples
from dca_diverse_sampling import (
    select_diverse_boundary_examples,select_diverse_boundary_informative_examples
)
from dca_true_balanced_sampling import (
    select_true_balanced_train_and_dev,
)
from performance_train_dev_sampling import (
    PerformanceTrainDevSelector
)
from sess_lc_dca import (
    select_sess_lc_dev_and_get_remaining,
)
from sess_rep_dca import (
    select_sess_rep_examples,
    select_ipomp_difficulty_train_examples,
    _example_id,
)
from wrep_dca import select_sess_wrep_dev_and_difficulty_train
from wrep_dca_wrep_train import select_sess_wrep_dev_and_wrep_train
from ipomp_dca import IPOMPDCASelector
from ipomp_dca_difficulty import IPOMPDCASelector as IPOMPDifficultySelector
from apex_dca import APEXController
from performance_train_sampling import PerformanceTrainSampler
from performance_train_sampling import PerformanceTrainSampler
from performance_train_sess_lc import (
    PerformanceTrainSessLCSampler,
)
from optimization import (
    OptimizationRunner,
    create_bootstrap_strategy,
    create_miprov2_strategy,
    create_simba_strategy,
    create_gepa_strategy,
)



def _quote_cli_arg(value: str) -> str:
    text = str(value)
    if not text:
        return '""'
    if any(ch in text for ch in ' \t"'):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def _build_rerun_from_seed_command(seed_state_path: str) -> str:
    forwarded_args: list[str] = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "--load-extractor":
            skip_next = True
            continue
        if arg == "--skip-load":
            continue
        forwarded_args.append(_quote_cli_arg(arg))
    forwarded_args.extend(["--load-extractor", _quote_cli_arg(seed_state_path)])
    return "uv run python extract_dca.py " + " ".join(forwarded_args)

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
parser.add_argument("--auto-load-compatible-cache", action="store_true",
                    help="Allow automatic loading of a compatible cached extractor when no explicit path is provided")
parser.add_argument("--skip-load", action="store_true",
                    help="Skip loading cached extractor even if it exists")
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
parser.add_argument("--max-labeled-demos", type=int, default=6,
                    help="Max labeled (human) demos passed to MIPROv2 (default: 6)")
parser.add_argument("--max-bootstrapped-demos", type=int, default=6,
                    help="Max bootstrapped demos passed to MIPROv2 (default: 6)")
parser.add_argument("--initial-instruction-file", type=str, default=None,
                    help="Path to a .md/.txt file whose content replaces the default signature "
                         "instruction before optimization starts (e.g. human_prompts/v1_prompt_midas.md)")
parser.add_argument("--clear-field-descs", action="store_true", default=False,
                    help="Clear all field desc strings before optimization so MIPROv2 "
                         "sees only the instruction block (works with or without --initial-instruction-file)")
parser.add_argument("--run-timestamp", type=str, default=None,
                    help="Pre-generated timestamp (YYYYmmddHHMMSS) supplied by the grid runner to "
                         "guarantee uniqueness across parallel runs. Falls back to datetime.now() "
                         "when running extract_dca.py directly.")

# ── Un-hardcoded optimizer knobs (all previously fixed in code) ──────────────
# Split + evaluation
parser.add_argument("--split-seed", type=int, default=42,
                    help="Seed for the stratified train/val/test split (default: 42)")
parser.add_argument("--train-ratio", type=float, default=0.6, help="Train split ratio (default: 0.6)")
parser.add_argument("--val-ratio", type=float, default=0.2, help="Val split ratio (default: 0.2)")
parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio (default: 0.2)")
parser.add_argument("--eval-num-threads", type=int, default=4,
                    help="Threads for dspy.Evaluate scoring (default: 4)")
parser.add_argument("--gepa-train-size", type=int, default=None,
                    help="Number of training examples to use for GEPA optimization")
parser.add_argument(
    "--gepa-sampling",
    type=str,
    default="random",
    choices=["random", "hard", "balanced", "adaptive", "diverse_boundary","diverse_boundary_informative","apex", "true_balanced","ipomp","performance_train","performance_class_adaptive","performance_train_dev","ipomp_difficulty","sess_rep","sess_lc","wrep","wrep_train","performance_train_sess_lc"],
    help="GEPA training sampling strategy: random, hard, balanced, adaptive, diverse_boundary, diverse_boundary_informative, apex, true_balanced, ipomp, performance_train, performance_class_adaptive, performance_train_dev, ipomp_difficulty, sess_rep, sess_lc, wrep, wrep_train, or performance_train_sess_lc sampling"
)
parser.add_argument(
    "--wrep-lambda",
    type=float,
    default=0.5,
    help=(
        "SESS-WREP difficulty weighting. "
        "0.0 = pure representativeness, "
        "1.0 = maximum difficulty weighting."
    ),
)
parser.add_argument(
    "--diverse-clusters",
    type=int,
    default=5,
    help="Number of K-means clusters for diverse/boundary sampling",
)

parser.add_argument(
    "--ipomp-hard-pool-size",
    type=int,
    default=100,
    help=(
        "Number of difficult candidates retained before "
        "CLIP diversity selection (default: 100)"
    ),
)
# ------------------------------------------------------------
# PERFORMANCE-BASED TRAIN SAMPLING
# ------------------------------------------------------------

parser.add_argument(
    "--performance-correlation-threshold",
    type=float,
    default=0.90,
    help=(
        "Correlation threshold used to identify redundant "
        "performance patterns (default: 0.90)"
    ),
)

parser.add_argument(
    "--performance-hard-pool-fraction",
    type=float,
    default=0.50,
    help=(
        "Fraction of candidate training examples considered "
        "for performance-based selection (default: 0.50)"
    ),
)

parser.add_argument(
    "--performance-rounds",
    type=int,
    default=3,
    help=(
        "Number of outer performance-based training "
        "selection rounds (default: 3)"
    ),
)

parser.add_argument(
    "--performance-seed",
    type=int,
    default=42,
    help=(
        "Random seed for performance-based train sampling "
        "(default: 42)"
    ),
)
# ------------------------------------------------------------
# DIVERSE + BOUNDARY + INFORMATIVE SAMPLING
# ------------------------------------------------------------
parser.add_argument(
    "--ipomp-rounds",
    type=int,
    default=2,
    help=(
        "Number of GEPA candidate-prompt rounds used "
        "to build the IPOMP Stage 2 performance matrix "
        "(default: 2)"
    )
)

parser.add_argument(
    "--ipomp-cluster-fraction",
    type=float,
    default=0.5,
    help="Fraction of IPOMP examples selected from clustering",
)

parser.add_argument(
    "--ipomp-correlation-threshold",
    type=float,
    default=0.90,
    help="Correlation threshold for redundant DCA examples",
)

parser.add_argument(
    "--ipomp-clusters",
    type=int,
    default=None,
    help="Number of K-means clusters",
)

parser.add_argument(
    "--apex-lookback",
    type=int,
    default=4,
    help="Number of recent evaluations used to classify examples.",
)

parser.add_argument(
    "--apex-seed",
    type=int,
    default=42,
    help="Random seed used by APEX.",
)

parser.add_argument(
    "--diverse-representative-fraction",
    type=float,
    default=0.4,
    help="Fraction of samples selected as representative/diverse.",
)

parser.add_argument(
    "--diverse-boundary-fraction",
    type=float,
    default=0.3,
    help="Fraction of samples selected as boundary examples.",
)

parser.add_argument(
    "--diverse-informative-fraction",
    type=float,
    default=0.3,
    help="Fraction of samples selected as informative examples.",
)
parser.add_argument(
    "--gepa-dev-size",
    type=int,
    default=30,
    help="Number of development examples used for GEPA iteration evaluation"
)
parser.add_argument(
    "--adaptive-rounds",
    type=int,
    default=3,
    help="Number of outer adaptive sampling rounds (default: 3)"
)

parser.add_argument(
    "--adaptive-exploration",
    type=float,
    default=0.20,
    help="Fraction of training samples selected randomly for exploration (default: 0.20)"
)

parser.add_argument(
    "--adaptive-seed",
    type=int,
    default=42,
    help="Random seed used by adaptive sampling (default: 42)"
)
parser.add_argument(
    "--gepa-reflection-minibatch-size",
    type=int,
    default=3,
    help="GEPA: number of examples used in each reflection minibatch (default: 3)"
)
# Bootstrap knobs
parser.add_argument("--bootstrap-labeled-demos", type=int, default=1,
                    help="Bootstrap: max labeled demos (default: 1)")
parser.add_argument("--bootstrap-bootstrapped-demos", type=int, default=1,
                    help="Bootstrap: max bootstrapped demos (default: 1)")
parser.add_argument("--bootstrap-num-candidate-programs", type=int, default=1,
                    help="Bootstrap: random-search candidate programs (default: 1)")
parser.add_argument("--bootstrap-max-rounds", type=int, default=1,
                    help="Bootstrap: bootstrapping rounds (default: 1)")
parser.add_argument("--bootstrap-num-threads", type=int, default=4,
                    help="Bootstrap: parallel eval threads (default: 4)")
# MIPROv2 knobs
parser.add_argument("--miprov2-num-threads", type=int, default=4,
                    help="MIPROv2: parallel eval threads (default: 4)")
# GEPA knobs
parser.add_argument("--gepa-num-threads", type=int, default=4,
                    help="GEPA: parallel eval threads (default: 4)")
parser.add_argument("--gepa-track-stats", action=argparse.BooleanOptionalAction, default=True,
                    help="GEPA: track detailed_results for in-pass best-score (default: True)")
parser.add_argument("--gepa-use-merge", action=argparse.BooleanOptionalAction, default=True,
                    help="GEPA: merge candidate programs (default: True)")
parser.add_argument("--gepa-reflection-images", action=argparse.BooleanOptionalAction, default=True,
                    help="GEPA: send document image-pairs to the reflection LM (multimodal) (default: True)")
parser.add_argument("--gepa-reflection-detail", type=str, default="auto",
                    choices=["low", "high", "auto"],
                    help="GEPA: image detail level for multimodal reflection (default: auto)")
parser.add_argument("--gepa-reflection-max-images", type=int, default=6,
                    help="GEPA: max images per reflection call (default: 6)")
# SimBA knobs
parser.add_argument("--simba-labeled-dem    os", type=int, default=4,
                    help="SimBA: max labeled demos (default: 4)")
parser.add_argument("--simba-num-threads", type=int, default=4,
                    help="SimBA: parallel threads (default: 4)")
parser.add_argument("--simba-bsize", type=int, default=32,
                    help="SimBA: minibatch size (default: 32)")
parser.add_argument("--simba-num-candidates", type=int, default=6,
                    help="SimBA: candidates per step (default: 6)")
# Teacher / prompt / reflection model overrides (default None -> use student LM;
# GEPA reflection defaults to the .env REFLECTION_LM_NAME, typically gpt-5.2)
parser.add_argument("--bootstrap-teacher-model", type=str, default=None,
                    help="Teacher LM for Bootstrap demo generation (default: student model)")
parser.add_argument("--miprov2-teacher-model", type=str, default=None,
                    help="Teacher LM for MIPROv2 demo generation (default: student model)")
parser.add_argument("--miprov2-prompt-model", type=str, default=None,
                    help="Proposer LM for MIPROv2 instruction generation (default: student model)")
parser.add_argument("--gepa-reflection-model", type=str, default=None,
                    help="Reflection LM for GEPA (default: env REFLECTION_LM_NAME, e.g. azure/gpt-5.2)")
parser.add_argument("--teacher-reasoning-effort", type=str, default=None,
                    choices=["low", "medium", "high"],
                    help="reasoning_effort applied to teacher/prompt/reflection LMs (default: None)")

args = parser.parse_args()

# Central flag — purely driven by the explicit switch
_should_clear_descs: bool = args.clear_field_descs

pairs_dir = args.pairs_dir
gt_dir = args.gt_dir
split_seed = args.split_seed
_split_ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
_split_stratify_by = ["expected_discard_image", "expected_human_label_category", "doubt"]


def _slug_token(value: str | None, default: str = "na") -> str:
    text = str(value or "").strip().lower()
    cleaned = [ch if ch.isalnum() else "_" for ch in text]
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or default


def _short_hash(*parts: object, length: int = 10) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def _build_split_cache_path() -> str:
    pairs_tag = _slug_token(Path(os.path.abspath(pairs_dir)).name)[:24]
    gt_tag = _slug_token(Path(os.path.abspath(gt_dir)).name)[:24]
    ratios_tag = (
        f"tr{int(_split_ratios['train'] * 100)}"
        f"va{int(_split_ratios['val'] * 100)}"
        f"te{int(_split_ratios['test'] * 100)}"
    )
    split_fingerprint = _short_hash(
        os.path.abspath(pairs_dir),
        os.path.abspath(gt_dir),
        split_seed,
        _split_ratios,
        _split_stratify_by,
    )
    return os.path.join(
        ".cache",
        "splits",
        f"dca_split_seed{split_seed}_{pairs_tag}_{gt_tag}_{ratios_tag}_{split_fingerprint}.json",
    )


# Create the split manifest path once.
split_latest_path = _build_split_cache_path()

# Make sure the split directory exists.
os.makedirs(
    os.path.dirname(split_latest_path),
    exist_ok=True
)

def _build_dataset_cache_tag() -> str:
    dataset_name = _slug_token(Path(os.path.abspath(pairs_dir)).name)[:18]
    return f"{dataset_name}_{_short_hash(os.path.abspath(pairs_dir), os.path.abspath(gt_dir))}"


def _format_demo_tag(max_labeled_demos, max_bootstrapped_demos) -> str:
    labeled = "na" if max_labeled_demos is None else str(max_labeled_demos)
    boot = "na" if max_bootstrapped_demos is None else str(max_bootstrapped_demos)
    return f"d{labeled}b{boot}"


def _primary_predictor(program):
    """Return the single predictor used by the DCA comparator program."""
    predictors = list(program.predictors()) if hasattr(program, "predictors") else []
    if not predictors:
        raise AttributeError("Comparator program does not expose any predictors")
    return predictors[0]


def _get_program_signature(program):
    """Read the active signature from the comparator's primary predictor."""
    return _primary_predictor(program).signature


def _set_program_signature(program, signature) -> None:
    """Replace the active signature on the comparator's primary predictor."""
    _primary_predictor(program).signature = signature


def _find_matching_cached_extractor(
    cache_root: str,
    strategy_tag: str,
    demo_tag: str,
    model_tag: str,
    cot_tag: str,
    re_tag: str,
    instr_tag: str,
    cfd_tag: str,
    dataset_tag: str,
    split_seed_value: int,
) -> str | None:
    root = Path(cache_root)
    if not root.exists() or not root.is_dir():
        return None

    pattern = (
        f"{model_tag}_{strategy_tag}_{cot_tag}{re_tag}_dca_optimized_{demo_tag}"
        f"{instr_tag}{cfd_tag}_ds{dataset_tag}_sp{split_seed_value}_*.json"
    )
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return str(matches[0]) if matches else None

# ── Timestamp ────────────────────────────────────────────────────
# Set once here so the log file and saved model share the same timestamp.
# When launched from run_dca_grid.py each run receives a pre-generated unique
# timestamp via --run-timestamp so parallel runs don't collide.
current_time = args.run_timestamp if args.run_timestamp else datetime.now().strftime("%Y%m%d%H%M%S")
pipeline_started_at = datetime.now().isoformat(timespec="seconds")
pipeline_t0 = time.time()
# ── Unique experiment directory ─────────────────────────────────
# Every execution gets its own directory so repeated experiments
# never overwrite or mix their outputs.

experiment_root = Path("experiments")

experiment_name = (
    f"exp_{current_time}_"
    f"{_slug_token(args.gepa_sampling)}_"
    f"train{args.gepa_train_size}_"
    f"dev{args.gepa_dev_size}_"
    f"seed{args.split_seed}"
)

experiment_dir = experiment_root / experiment_name
experiment_dir.mkdir(parents=True, exist_ok=True)

print(f"\nExperiment directory: {experiment_dir}")

# ── Log filename prefix ───────────────────────────────────────────
# Compute the same prefix that the saved model will use, so the log file
# is immediately identifiable even if the run crashes before saving.
# For "all" we know all strategies will run; for a single optimizer we know
# the exact strategy name upfront.
_model_display = (args.model_name or os.getenv("MODEL_NAME", "Unknown")).replace("/", "_")
_cot_tag       = "NoCot" if args.no_cot else "CoT"
_re_tag        = f"_RE{args.reasoning_effort}" if args.reasoning_effort else ""
_dataset_tag   = _build_dataset_cache_tag()
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
_demo_tag   = f"_d{args.max_labeled_demos}b{args.max_bootstrapped_demos}"
_instr_tag  = ("_" + os.path.splitext(os.path.basename(args.initial_instruction_file))[0]
               if args.initial_instruction_file else "")
_cfd_tag    = "_cfd" if args.clear_field_descs else ""
_split_tag  = f"_ds{_dataset_tag}_sp{split_seed}"
_log_prefix = f"{_model_display}_{_strategy_display}_{_cot_tag}{_re_tag}{_demo_tag}{_instr_tag}{_cfd_tag}{_split_tag}"
# ── Unique experiment directory ─────────────────────────────────
# Create a completely separate directory for every experiment.
# This keeps the model, prompt/state, log, summary, and split
# information together and prevents repeated runs from mixing files.

experiment_root = Path("experiments")
experiment_root.mkdir(parents=True, exist_ok=True)

experiment_dir = (
    experiment_root
    / f"exp_{current_time}_{_slug_token(args.gepa_sampling)}"
    f"_train{args.gepa_train_size}"
    f"_dev{args.gepa_dev_size}"
    f"_seed{split_seed}"
)

experiment_dir.mkdir(parents=True, exist_ok=True)

print(f"Experiment directory: {experiment_dir}")

# ── TeeLogger ────────────────────────────────────────────────────
# Captures all stdout AND stderr (including DSPy's logging output and tqdm
# progress bars) to a log file while still printing to the terminal.
_log_dir = str(experiment_dir)
os.makedirs(_log_dir, exist_ok=True)

_stdout_log_path = os.path.join(
    _log_dir,
    "run.log"
)


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

# ── Truncation Inspector ─────────────────────────────────────────
# Whenever DSPy emits a "LM response was truncated" WARNING, immediately
# dump the last LM interaction (prompt + full truncated response) to the log
# so we can see exactly what the model was generating and why it hit the limit.

class _TruncationInspectorHandler(logging.Handler):
    """On every truncation WARNING from dspy.clients.lm, log the LM history."""

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
print(f"  Save model: always")
print(f"  Initial instruction: {args.initial_instruction_file or '(signature docstring)'}")
print(f"  Clear field descs: {args.clear_field_descs}")
print(f"  Max labeled demos: {args.max_labeled_demos}  |  Max bootstrapped demos: {args.max_bootstrapped_demos}")
print(f"  Dataset cache tag: {_dataset_tag}  |  Split seed: {split_seed}")
print(f"\n Data Paths:")
print(f"  Image pairs: {pairs_dir}")
print(f"  Ground truth: {gt_dir}")
print(f"{'='*80}\n")


# ────────────────────────────────────────────────────────────────
# INITIAL DCA COMPARATOR
# ────────────────────────────────────────────────────────────────
# This comparator is the baseline/unoptimized DCA program.
# It is used only for calculating informativeness during
# Diverse + Boundary + Informative sample selection.
# ────────────────────────────────────────────────────────────────

if args.no_cot:
    comparator = dspy.Predict(DCAImageComparison)
else:
    comparator = dspy.ChainOfThought(DCAImageComparison)

print("\n" + "=" * 80)
print("INITIAL DCA COMPARATOR CREATED")
print("=" * 80)
print(f"Comparator : {type(comparator).__name__}")
print("Purpose    : Informativeness scoring for sample selection")
print("=" * 80 + "\n")

# ── Data Loading ─────────────────────────────────────────────────
dataset = load_dca_dataset(pairs_dir, gt_dir)
if len(dataset) == 0:
    print("ERROR: No valid DCA image pairs found. Check your data paths.")
    sys.stdout = _tee_stdout._original
    sys.stderr = _tee_stderr._original
    _log_file.close()
    exit(1)

def _pair_names(examples: List[dspy.Example]) -> list[str]:
    return [
        str(getattr(ex, "pair_name"))
        for ex in examples
        if getattr(ex, "pair_name", None) is not None
    ]

def _score_dca_examples(examples, comparator):
    """
    Score every example in the candidate training pool using
    the current DCA comparator.

    Lower score = harder example.
    """

    scored_examples = []

    print("\n" + "=" * 80)
    print(" - SCORING CANDIDATE POOL")
    print("=" * 80)

    for idx, example in enumerate(examples):
        try:
            prediction = comparator(
                image_original=example.image_original,
                image_reimage=example.image_reimage,
            )

            score = float(
                dca_comparison_metric(
                    example,
                    prediction,
                )
            )

            error = 1.0 - score

            scored_examples.append(
                {
                    "example": example,
                    "score": score,
                    "error": error,
                    "index": idx,
                }
            )

        except Exception as exc:
            print(
                f"[WARNING] Could not score adaptive example "
                f"{idx}: {exc}"
            )

    if not scored_examples:
        raise ValueError(
            "No candidate training examples could be scored."
        )

    print(
        f"Successfully scored "
        f"{len(scored_examples)}/{len(examples)} examples."
    )

    return scored_examples



def _select_adaptive_examples(
    scored_examples,
    sample_size,
    exploration_fraction=0.20,
    seed=42,
):
    """
    Select training examples using adaptive difficulty weighting.

    Hard examples receive higher probability, while a fixed
    exploration fraction is sampled randomly to preserve diversity.
    """

    if sample_size <= 0:
        return [], {
        "training_samples": 0,
        "exploitation": 0,
        "exploration": 0,
        "exploration_rate": exploration_fraction,
        "seed": seed,
        "candidate_pool_size": len(scored_examples),
        "samples": [],
    }
        
    if len(scored_examples) < sample_size:
        raise ValueError(
            f"Only {len(scored_examples)} examples are available, "
            f"but {sample_size} are required."
        )

    rng = random.Random(seed)

    # ---------------------------------------------------------
    # Calculate adaptive weights
    # ---------------------------------------------------------
    for item in scored_examples:
        error = item["error"]

        # Higher error -> higher sampling weight
        item["weight"] = max(
            0.05,
            (error + 0.01) ** 0.5,
        )

    # ---------------------------------------------------------
    # Exploration samples
    # ---------------------------------------------------------
    exploration_size = int(
        sample_size * exploration_fraction
    )

    exploitation_size = sample_size - exploration_size

    # ---------------------------------------------------------
    # Exploitation:
    # select examples with highest adaptive weights
    # ---------------------------------------------------------
    ranked = sorted(
        scored_examples,
        key=lambda item: item["weight"],
        reverse=True,
    )

    exploitation_items = ranked[:exploitation_size]

    selected_ids = {
        id(item["example"])
        for item in exploitation_items
    }

    # ---------------------------------------------------------
    # Exploration:
    # randomly select from remaining examples
    # ---------------------------------------------------------
    remaining = [
        item
        for item in scored_examples
        if id(item["example"]) not in selected_ids
    ]

    if exploration_size > 0:
        exploration_items = rng.sample(
            remaining,
            min(exploration_size, len(remaining)),
        )
    else:
        exploration_items = []

    # ---------------------------------------------------------
    # Combine exploitation + exploration
    # ---------------------------------------------------------
    selected_items = (
        exploitation_items +
        exploration_items
    )

    rng.shuffle(selected_items)

    selected_examples = [
        item["example"]
        for item in selected_items[:sample_size]
    ]

    # ---------------------------------------------------------
    # LOG ADAPTIVE SAMPLING INFORMATION
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("ADAPTIVE SAMPLING SELECTION")
    print("=" * 80)

    print(f"Candidate pool    : {len(scored_examples)}")
    print(f"Training samples  : {len(selected_examples)}")
    print(f"Exploitation      : {len(exploitation_items)}")
    print(f"Exploration       : {len(exploration_items)}")
    print(
        f"Exploration rate  : "
        f"{exploration_fraction:.2f}"
    )
    print(f"Random seed       : {seed}")

    # ---------------------------------------------------------
    # LOG ALL SELECTED EXAMPLES
    # ---------------------------------------------------------
    print("\nALL SELECTED TRAINING EXAMPLES")
    print("-" * 80)

    for rank, item in enumerate(
        selected_items[:sample_size],
        start=1,
    ):
        example = item["example"]

        pair_name = getattr(
            example,
            "pair_name",
            "unknown",
        )

        selection_type = (
            "EXPLOITATION"
            if id(example) in selected_ids
            else "EXPLORATION"
        )

        print(
            f"{rank:3d}. "
            f"pair={pair_name} | "
            f"type={selection_type} | "
            f"score={item['score']:.4f} | "
            f"error={item['error']:.4f} | "
            f"weight={item['weight']:.4f}"
        )

    print("-" * 80)

    # ---------------------------------------------------------
    # SUMMARY OF SELECTED DIFFICULTY
    # ---------------------------------------------------------
    selected_scores = [
        item["score"]
        for item in selected_items[:sample_size]
    ]

    selected_errors = [
        item["error"]
        for item in selected_items[:sample_size]
    ]

    if selected_scores:
        print(
            f"Selected score range : "
            f"{min(selected_scores):.4f} - "
            f"{max(selected_scores):.4f}"
        )

        print(
            f"Selected error range : "
            f"{min(selected_errors):.4f} - "
            f"{max(selected_errors):.4f}"
        )

    print("=" * 80)

    # ---------------------------------------------------------
    # RETURN SELECTED TRAINING SET + SELECTION INFORMATION
    # ---------------------------------------------------------
    selection_info = {
        "training_samples": len(selected_examples),
        "exploitation": len(exploitation_items),
        "exploration": len(exploration_items),
        "exploration_rate": exploration_fraction,
        "seed": seed,
        "candidate_pool_size": len(scored_examples),
        "samples": selected_examples,
    }

    return selected_examples, selection_info
# ────────────────────────────────────────────────────────────────
# Stratified split: train 60%, val 20%, test 20%
# ────────────────────────────────────────────────────────────────
# The split manifest path is fingerprinted by dataset path + split settings.

train_pool, val_set, test_set = stratified_split_dca_dataset(
    dataset,
    train_ratio=_split_ratios["train"],
    val_ratio=_split_ratios["val"],
    test_ratio=_split_ratios["test"],
    random_seed=split_seed,
    verbose=True
)

# ────────────────────────────────────────────────────────────────
# GEPA Train / Development split
# ────────────────────────────────────────────────────────────────
# The training pool is further divided into:
#
#   30 samples -> FIXED GEPA development set
#   Remaining  -> Candidate pool for training sampling
#
# The development set remains the same across experiments so that
# Random Sampling vs Hard Sampling can be compared fairly.
# Validation and test sets are NOT touched.
# ────────────────────────────────────────────────────────────────

gepa_train_size = args.gepa_train_size or 50
gepa_dev_size = args.gepa_dev_size or 30

if len(train_pool) < gepa_train_size + gepa_dev_size:
    raise ValueError(
        f"Not enough training samples for GEPA train/dev split. "
        f"Available: {len(train_pool)}, "
        f"required: {gepa_train_size + gepa_dev_size}"
    )

# ────────────────────────────────────────────────────────────────
# CREATE GEPA DEV SET + CANDIDATE TRAINING POOL
# ────────────────────────────────────────────────────────────────
#
# IMPORTANT:
#   - The original 254-example train_pool is never modified.
#   - The official 82-example validation set is untouched.
#   - The official 93-example test set is untouched.
#   - Existing sampling strategies keep their original behavior.
#
# For true_balanced:
#   1. Identify categories inside the 254-example train pool.
#   2. Select a category-balanced GEPA Train set.
#   3. Select a category-balanced GEPA Dev set.
#   4. Ensure Train and Dev have no overlap.
#
# ────────────────────────────────────────────────────────────────


if args.gepa_sampling == "diverse_boundary_informative":

    # ------------------------------------------------------------
    # EXISTING DIVERSE + BOUNDARY + INFORMATIVE DEV SELECTION
    # ------------------------------------------------------------

    dev_set, dev_selection_info = (
        select_diverse_boundary_informative_examples(
            examples=train_pool,
            sample_size=args.gepa_dev_size,
            n_clusters=args.diverse_clusters,
            representative_fraction=args.diverse_representative_fraction,
            boundary_fraction=args.diverse_boundary_fraction,
            informative_fraction=args.diverse_informative_fraction,
            seed=split_seed,
            cache_dir=".cache/dca_embeddings",
            comparator=comparator,
            metric=dca_comparison_metric,
        )
    )

    # Remove selected Dev examples from the training pool.
    dev_ids = {
        getattr(example, "pair_name", None)
        for example in dev_set
    }

    candidate_train_pool = [
        example
        for example in train_pool
        if getattr(example, "pair_name", None) not in dev_ids
    ]



elif args.gepa_sampling == "true_balanced":

    # ------------------------------------------------------------
    # TRUE CATEGORY-BALANCED TRAIN + DEV
    # ------------------------------------------------------------
    #
    # Both GEPA Train and GEPA Dev are selected from the
    # ORIGINAL 254-example training pool.
    #
    # Validation (82) and Test (93) are NOT involved.
    #
    # Train and Dev are guaranteed to have no overlap.
    # ------------------------------------------------------------

    train_set, dev_set = (
        select_true_balanced_train_and_dev(
            train_pool=train_pool,
            train_size=args.gepa_train_size,
            dev_size=args.gepa_dev_size,
            seed=split_seed,
        )
    )

    # IDs selected for Train and Dev.
    train_ids = {
        getattr(example, "pair_name", None)
        for example in train_set
    }

    dev_ids = {
        getattr(example, "pair_name", None)
        for example in dev_set
    }

    # Remaining examples from the original 254 training pool.
    #
    # These are NOT used for the current GEPA run.
    # They are kept available as the unused candidate pool.
    candidate_train_pool = [
        example
        for example in train_pool
        if getattr(example, "pair_name", None)
        not in train_ids
        and getattr(example, "pair_name", None)
        not in dev_ids
    ]

    print("\n" + "=" * 80)
    print("TRUE CATEGORY-BALANCED SAMPLING")
    print("=" * 80)
    print(f"Original training pool : {len(train_pool)}")
    print(f"GEPA Train             : {len(train_set)}")
    print(f"GEPA Dev               : {len(dev_set)}")
    print(f"Unused training pool   : {len(candidate_train_pool)}")
    print("Validation             : UNTOUCHED")
    print("Test                   : UNTOUCHED")
    print("=" * 80)
elif args.gepa_sampling == "performance_train":

    # ------------------------------------------------------------
    # PERFORMANCE-BASED DYNAMIC TRAINING
    # ------------------------------------------------------------
    #
    # The 254-example train pool is split ONCE:
    #
    #   30 examples  -> fixed Dev set
    #   224 examples -> candidate training pool
    #
    # The Dev set never changes during performance rounds.
    # The candidate pool is scored repeatedly and new Train
    # subsets are selected dynamically.
    #
    # Validation and Test are untouched.
    # ------------------------------------------------------------

    import random

    rng = random.Random(split_seed)

    performance_pool = list(train_pool)
    rng.shuffle(performance_pool)

    # Fixed Dev
    dev_set = performance_pool[:args.gepa_dev_size]

    # Candidate pool
    candidate_train_pool = performance_pool[args.gepa_dev_size:]

    print("\n" + "=" * 80)
    print("PERFORMANCE-BASED DYNAMIC TRAINING SAMPLING")
    print("=" * 80)

    print(
        f"Original training pool : {len(train_pool)}"
    )

    print(
        f"Fixed GEPA Dev         : {len(dev_set)}"
    )

    print(
        f"Candidate train pool   : {len(candidate_train_pool)}"
    )

    print(
        f"Dynamic GEPA Train     : {args.gepa_train_size}"
    )

    print(
        f"Performance rounds     : "
        f"{args.performance_rounds}"
    )

    print(
        f"Correlation threshold  : "
        f"{args.performance_correlation_threshold}"
    )

    print(
        f"Hard pool fraction     : "
        f"{args.performance_hard_pool_fraction}"
    )

    print("Validation             : UNTOUCHED")
    print("Test                   : UNTOUCHED")

    print("=" * 80)

    if len(candidate_train_pool) < args.gepa_train_size:

        raise ValueError(
            f"Candidate training pool "
            f"({len(candidate_train_pool)}) is smaller "
            f"than requested training size "
            f"({args.gepa_train_size})."
        )

    performance_sampler = PerformanceTrainSampler(
        examples=candidate_train_pool,
        sample_size=args.gepa_train_size,
        correlation_threshold=(
            args.performance_correlation_threshold
        ),
        hard_pool_fraction=(
            args.performance_hard_pool_fraction
        ),
        seed=args.performance_seed,
    )

    # Initial training set.
    train_set, selection_info = (
        performance_sampler.select_training_examples()
    )

    print("\nInitial performance-selected Train:")
    print(
        f"  Train : {len(train_set)}"
    )

    print(
        f"  Dev   : {len(dev_set)}"
    )

    print(
        f"  Pool  : {len(candidate_train_pool)}"
    )

    print("=" * 80)
elif args.gepa_sampling == "performance_train_sess_lc":

    print("\n" + "=" * 80)
    print("PERFORMANCE-TRAIN + SESS-LC")
    print("=" * 80)

    # ============================================================
    # STEP 1: SESS-LC selects the fixed Dev set
    # ============================================================

    (
        dev_set,
        candidate_train_pool,
    ) = select_sess_lc_dev_and_get_remaining(
        examples=train_pool,
        dev_size=args.gepa_dev_size,
        comparator=comparator,
        metric=dca_comparison_metric,
    )

    # ============================================================
    # STEP 2: Create Performance-Train sampler
    #
    # IMPORTANT:
    # dev_size is NOT passed here.
    # SESS-LC has already selected Dev.
    # ============================================================

    performance_sampler = PerformanceTrainSessLCSampler(
        examples=candidate_train_pool,
        train_size=args.gepa_train_size,
        correlation_threshold=(
            args.performance_correlation_threshold
        ),
        hard_pool_fraction=(
            args.performance_hard_pool_fraction
        ),
        seed=args.performance_seed,
    )

    # ============================================================
    # STEP 3: Initial Train selection
    # ============================================================

    (
        train_set,
        selection_info,
    ) = performance_sampler.select_training_examples()

    # ============================================================
    # SAFETY CHECKS
    # ============================================================

    if len(train_set) != args.gepa_train_size:
        raise RuntimeError(
            "Performance-Train + SESS-LC produced "
            "the wrong training-set size. "
            f"Expected {args.gepa_train_size}, "
            f"got {len(train_set)}."
        )

    if len(dev_set) != args.gepa_dev_size:
        raise RuntimeError(
            "SESS-LC produced the wrong development-set size. "
            f"Expected {args.gepa_dev_size}, "
            f"got {len(dev_set)}."
        )

    train_ids = {
        _example_id(example)
        for example in train_set
    }

    dev_ids = {
        _example_id(example)
        for example in dev_set
    }

    val_ids = {
        _example_id(example)
        for example in val_set
    }

    test_ids = {
        _example_id(example)
        for example in test_set
    }

    # ------------------------------------------------------------
    # Train / Dev
    # ------------------------------------------------------------

    train_dev_overlap = train_ids & dev_ids

    if train_dev_overlap:
        raise RuntimeError(
            "Performance-Train + SESS-LC created "
            "Train/Dev overlap: "
            f"{sorted(train_dev_overlap)}"
        )

    # ------------------------------------------------------------
    # Train / Validation
    # ------------------------------------------------------------

    if train_ids & val_ids:
        raise RuntimeError(
            "Performance-Train + SESS-LC created "
            "Train/Validation overlap."
        )

    # ------------------------------------------------------------
    # Train / Test
    # ------------------------------------------------------------

    if train_ids & test_ids:
        raise RuntimeError(
            "Performance-Train + SESS-LC created "
            "Train/Test overlap."
        )

    # ------------------------------------------------------------
    # Dev / Validation
    # ------------------------------------------------------------

    if dev_ids & val_ids:
        raise RuntimeError(
            "Performance-Train + SESS-LC created "
            "Dev/Validation overlap."
        )

    # ------------------------------------------------------------
    # Dev / Test
    # ------------------------------------------------------------

    if dev_ids & test_ids:
        raise RuntimeError(
            "Performance-Train + SESS-LC created "
            "Dev/Test overlap."
        )

    print("\n" + "=" * 80)
    print("PERFORMANCE-TRAIN + SESS-LC SAMPLING COMPLETE")
    print("=" * 80)

    print(
        f"Original Train pool : {len(train_pool)}"
    )

    print(
        f"SESS-LC Dev         : {len(dev_set)}"
    )

    print(
        f"Remaining candidates: "
        f"{len(candidate_train_pool)}"
    )

    print(
        f"Performance Train   : {len(train_set)}"
    )

    print(
        f"Validation           : {len(val_set)} [UNTOUCHED]"
    )

    print(
        f"Test                 : {len(test_set)} [UNTOUCHED]"
    )

    print("=" * 80)
elif args.gepa_sampling == "performance_train_dev":

    # ------------------------------------------------------------
    # PERFORMANCE-BASED TRAIN + DEV SAMPLING
    # ------------------------------------------------------------
    #
    # The original train_pool is the only source of examples.
    #
    # Validation and Test are NEVER used for sampling.
    #
    # Initial selection:
    #
    #   train_pool
    #       |
    #       +----> initial Train
    #       |
    #       +----> initial Dev
    #
    # After GEPA candidate prompts are produced:
    #
    #   prompt performance
    #          |
    #          v
    #   performance matrix
    #          |
    #          v
    #   correlation / redundancy
    #          |
    #          v
    #   contrasting replacement
    #          |
    #          +----> Train refinement
    #          |
    #          +----> Dev refinement
    #
    # Train and Dev are refined independently while preventing
    # Train/Dev overlap.
    # ------------------------------------------------------------

    if args.optimizer != "gepa":
        raise ValueError(
            "performance_train_dev sampling is currently "
            "supported only with GEPA."
        )

    print("\n" + "=" * 80)
    print("PERFORMANCE-BASED TRAIN + DEV SAMPLING")
    print("=" * 80)

    print(
        f"Original training pool : "
        f"{len(train_pool)}"
    )

    print(
        f"Requested GEPA train   : "
        f"{args.gepa_train_size}"
    )

    print(
        f"Requested GEPA dev     : "
        f"{args.gepa_dev_size}"
    )

    print(
        f"Correlation threshold  : "
        f"{args.performance_correlation_threshold}"
    )

    print(
        f"Performance rounds     : "
        f"{args.performance_rounds}"
    )

    print(
        "Validation             : UNTOUCHED"
    )

    print(
        "Test                   : UNTOUCHED"
    )

    print("=" * 80)

    # ------------------------------------------------------------
    # Create the performance-based Train + Dev selector
    # ------------------------------------------------------------

    performance_train_dev_selector = PerformanceTrainDevSelector(
        examples=train_pool,
        train_size=args.gepa_train_size,
        dev_size=args.gepa_dev_size,
        correlation_threshold=(
            args.performance_correlation_threshold
        ),
        seed=args.performance_seed,
    )

    # ------------------------------------------------------------
    # INITIAL TRAIN + DEV SELECTION
    # ------------------------------------------------------------
    #
    # This creates the initial Train and Dev sets.
    # The original train_pool is NOT modified.
    # ------------------------------------------------------------
    import random
    rng = random.Random(
        args.performance_seed
    )

    performance_pool = list(
        train_pool
    )

    rng.shuffle(
        performance_pool
    )

    # ------------------------------------------------------------
    # Initial Train
    # ------------------------------------------------------------

    train_set = performance_pool[
        :args.gepa_train_size
    ]

    # ------------------------------------------------------------
    # Initial Dev
    #
    # Dev is selected immediately after Train.
    # Therefore Train and Dev cannot overlap.
    # ------------------------------------------------------------

    dev_start = args.gepa_train_size

    dev_end = (
        args.gepa_train_size
        + args.gepa_dev_size
    )

    dev_set = performance_pool[
        dev_start:dev_end
    ]

    # ------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------

    if len(train_set) != args.gepa_train_size:
        raise RuntimeError(
            "Performance Train+Dev sampler produced "
            "the wrong training-set size. "
            f"Expected={args.gepa_train_size}, "
            f"Got={len(train_set)}"
        )

    if len(dev_set) != args.gepa_dev_size:
        raise RuntimeError(
            "Performance Train+Dev sampler produced "
            "the wrong development-set size. "
            f"Expected={args.gepa_dev_size}, "
            f"Got={len(dev_set)}"
        )

    # ------------------------------------------------------------
    # Verify Train / Dev do not overlap
    # ------------------------------------------------------------

    train_ids = {
        getattr(example, "pair_name", None)
        for example in train_set
    }

    dev_ids = {
        getattr(example, "pair_name", None)
        for example in dev_set
    }

    overlap = train_ids & dev_ids

    if overlap:
        raise RuntimeError(
            "Performance Train+Dev sampling produced "
            f"Train/Dev overlap: {overlap}"
        )

    # ------------------------------------------------------------
    # Candidate pool
    #
    # Only examples from the original train_pool are allowed.
    # Validation and Test are not included.
    # ------------------------------------------------------------

    performance_train_dev_candidate_pool = list(
    train_pool
)

    print("\n" + "=" * 80)
    print("PERFORMANCE TRAIN + DEV INITIAL SELECTION COMPLETE")
    print("=" * 80)

    print(
    f"Original train pool : "
    f"{len(train_pool)}"
    )

    print(
    f"GEPA Train          : "
    f"{len(train_set)}"
    )

    print(
    f"GEPA Dev            : "
    f"{len(dev_set)}"
    )

    print(
    f"Candidate pool      : "
    f"{len(performance_train_dev_candidate_pool)}"
    )

    print(
    f"Validation          : "
    f"{len(val_set)} [UNTOUCHED]"
    )

    print(
    f"Test                : "
    f"{len(test_set)} [UNTOUCHED]"
    )

    print("=" * 80)
elif args.gepa_sampling == "ipomp":

    # ------------------------------------------------------------
    # IPOMP-DCA
    # ------------------------------------------------------------
    #
    # Stage 1:
    #   CLIP image-pair embeddings
    #   -> diverse GEPA Dev
    #   -> GEPA Train
    #
    # Stage 2 happens later, after GEPA has produced
    # multiple candidate prompts.
    #
    # IMPORTANT:
    #   Original train_pool remains unchanged.
    #   Validation and test remain untouched.
    # ------------------------------------------------------------

    if args.optimizer != "gepa":
        raise ValueError(
            "IPOMP sampling is currently "
            "supported only with GEPA."
        )

    print("\n" + "=" * 80)
    print("IPOMP-DCA STAGE 1")
    print("=" * 80)

    print(
        f"Available training pool : "
        f"{len(train_pool)}"
    )

    print(
        f"Requested GEPA train    : "
        f"{args.gepa_train_size}"
    )

    print(
        f"Requested GEPA dev      : "
        f"{args.gepa_dev_size}"
    )

    # ------------------------------------------------------------
    # Create IPOMP selector
    # ------------------------------------------------------------

    ipomp_selector = IPOMPDCASelector(
        examples=train_pool,
        train_size=args.gepa_train_size,
        dev_size=args.gepa_dev_size,
        cluster_fraction=args.ipomp_cluster_fraction,
        num_clusters=args.ipomp_clusters,
        correlation_threshold=args.ipomp_correlation_threshold,
        seed=split_seed,
        hard_pool_size=args.ipomp_hard_pool_size,
    )

    # ------------------------------------------------------------
    # STAGE 1
    # ------------------------------------------------------------

    train_set, dev_set = (
    ipomp_selector.initial_selection(
        comparator=comparator,
        metric=dca_comparison_metric,
    )
)
    # ------------------------------------------------------------
    # Verify sizes
    # ------------------------------------------------------------

    if len(train_set) != args.gepa_train_size:
        raise RuntimeError(
            "IPOMP produced the wrong training-set size. "
            f"Expected {args.gepa_train_size}, "
            f"got {len(train_set)}."
        )

    if len(dev_set) != args.gepa_dev_size:
        raise RuntimeError(
            "IPOMP produced the wrong development-set size. "
            f"Expected {args.gepa_dev_size}, "
            f"got {len(dev_set)}."
        )

    # ------------------------------------------------------------
    # Verify NO Train/Dev overlap
    # ------------------------------------------------------------

    train_ids = {
        getattr(example, "pair_name", None)
        for example in train_set
    }

    dev_ids = {
        getattr(example, "pair_name", None)
        for example in dev_set
    }

    overlap = train_ids & dev_ids

    if overlap:
        raise RuntimeError(
            "IPOMP produced overlapping "
            f"train/dev examples: {overlap}"
        )

    # ------------------------------------------------------------
    # Create UNUSED candidate pool.
    #
    # These examples can later be used by Stage 2 as
    # replacement evaluation examples.
    # ------------------------------------------------------------

    candidate_train_pool = [
        example
        for example in train_pool
        if getattr(example, "pair_name", None)
        not in train_ids
        and getattr(example, "pair_name", None)
        not in dev_ids
    ]

    # ------------------------------------------------------------
    # Print Stage 1 result
    # ------------------------------------------------------------

    ipomp_selector.print_selection(
        train_set,
        dev_set,
    )

    print("\n" + "=" * 80)
    print("IPOMP STAGE 1 COMPLETE")
    print("=" * 80)

    print(
        f"Original training pool : "
        f"{len(train_pool)}"
    )

    print(
        f"GEPA Train             : "
        f"{len(train_set)}"
    )

    print(
        f"GEPA Dev               : "
        f"{len(dev_set)}"
    )

    print(
        f"Unused candidate pool  : "
        f"{len(candidate_train_pool)}"
    )

    print(
        f"Validation             : "
        f"{len(val_set)} [UNTOUCHED]"
    )

    print(
        f"Test                   : "
        f"{len(test_set)} [UNTOUCHED]"
    )

    print("=" * 80)
elif args.gepa_sampling == "ipomp_difficulty":

    # ------------------------------------------------------------
    # IPOMP-DCA DIFFICULTY-AWARE
    # ------------------------------------------------------------
    #
    # Stage 1:
    #   CLIP image-pair embeddings
    #   -> diverse GEPA Dev
    #   -> difficulty-aware/diverse GEPA Train
    #
    # Stage 2:
    #   Existing IPOMP refinement
    #   -> CLIP diversity
    #   + DCA difficulty
    #   -> replacement evaluation example
    #
    # IMPORTANT:
    #   Original train_pool remains unchanged.
    #   Validation and test remain untouched.
    # ------------------------------------------------------------

    if args.optimizer != "gepa":
        raise ValueError(
            "IPOMP difficulty-aware sampling is currently "
            "supported only with GEPA."
        )

    print("\n" + "=" * 80)
    print("IPOMP-DCA DIFFICULTY-AWARE STAGE 1")
    print("=" * 80)

    print(
        f"Available training pool : "
        f"{len(train_pool)}"
    )

    print(
        f"Requested GEPA train    : "
        f"{args.gepa_train_size}"
    )

    print(
        f"Requested GEPA dev      : "
        f"{args.gepa_dev_size}"
    )

    # ------------------------------------------------------------
    # Create IPOMP DIFFICULTY selector
    # ------------------------------------------------------------

    ipomp_difficulty_selector = IPOMPDifficultySelector(
        examples=train_pool,
        train_size=args.gepa_train_size,
        dev_size=args.gepa_dev_size,
        cluster_fraction=args.ipomp_cluster_fraction,
        num_clusters=args.ipomp_clusters,
        correlation_threshold=args.ipomp_correlation_threshold,
        seed=split_seed,
        hard_pool_size=args.ipomp_hard_pool_size,
    )

    # ------------------------------------------------------------
    # STAGE 1
    # ------------------------------------------------------------

    train_set, dev_set = (
        ipomp_difficulty_selector.initial_selection(
            comparator=comparator,
            metric=dca_comparison_metric,
        )
    )

    # ------------------------------------------------------------
    # Verify sizes
    # ------------------------------------------------------------

    if len(train_set) != args.gepa_train_size:
        raise RuntimeError(
            "IPOMP difficulty-aware sampling produced "
            "the wrong training-set size. "
            f"Expected {args.gepa_train_size}, "
            f"got {len(train_set)}."
        )

    if len(dev_set) != args.gepa_dev_size:
        raise RuntimeError(
            "IPOMP difficulty-aware sampling produced "
            "the wrong development-set size. "
            f"Expected {args.gepa_dev_size}, "
            f"got {len(dev_set)}."
        )

    # ------------------------------------------------------------
    # Verify NO Train/Dev overlap
    # ------------------------------------------------------------

    train_ids = {
        getattr(example, "pair_name", None)
        for example in train_set
    }

    dev_ids = {
        getattr(example, "pair_name", None)
        for example in dev_set
    }

    overlap = train_ids & dev_ids

    if overlap:
        raise RuntimeError(
            "IPOMP difficulty-aware sampling produced "
            f"overlapping train/dev examples: {overlap}"
        )

    # ------------------------------------------------------------
    # Create UNUSED candidate pool.
    #
    # These examples can later be used by Stage 2
    # as replacement evaluation examples.
    # ------------------------------------------------------------

    candidate_train_pool = [
        example
        for example in train_pool
        if getattr(example, "pair_name", None)
        not in train_ids
        and getattr(example, "pair_name", None)
        not in dev_ids
    ]

    # ------------------------------------------------------------
    # Print Stage 1 result
    # ------------------------------------------------------------

    ipomp_difficulty_selector.print_selection(
        train_set,
        dev_set,
    )

    print("\n" + "=" * 80)
    print("IPOMP-DCA DIFFICULTY-AWARE STAGE 1 COMPLETE")
    print("=" * 80)

    print(
        f"Original training pool : "
        f"{len(train_pool)}"
    )

    print(
        f"GEPA Train             : "
        f"{len(train_set)}"
    )

    print(
        f"GEPA Dev               : "
        f"{len(dev_set)}"
    )

    print(
        f"Unused candidate pool  : "
        f"{len(candidate_train_pool)}"
    )

    print(
        f"Validation             : "
        f"{len(val_set)} [UNTOUCHED]"
    )

    print(
        f"Test                   : "
        f"{len(test_set)} [UNTOUCHED]"
    )

    print("=" * 80)
        

elif args.gepa_sampling == "performance_class_adaptive":

    # ------------------------------------------------------------
    # PERFORMANCE-BASED CLASS-ADAPTIVE TRAINING
    # ------------------------------------------------------------
    #
    # 254-example official train pool
    #       ↓
    # 30 examples -> FIXED Dev
    # 224 examples -> candidate training pool
    #
    # Only the 50-example GEPA Train set changes dynamically.
    #
    # Dev / Validation / Test remain fixed.
    # ------------------------------------------------------------

    import random

    rng = random.Random(split_seed)

    performance_class_pool = list(train_pool)
    rng.shuffle(performance_class_pool)

    # Fixed Dev set
    dev_set = performance_class_pool[
        :args.gepa_dev_size
    ]

    # Remaining examples are available for dynamic training selection
    candidate_train_pool = performance_class_pool[
        args.gepa_dev_size:
    ]

    print("\n" + "=" * 80)
    print("PERFORMANCE-BASED CLASS-ADAPTIVE SAMPLING")
    print("=" * 80)

    print(
        f"Original training pool : {len(train_pool)}"
    )

    print(
        f"Fixed GEPA Dev         : {len(dev_set)}"
    )

    print(
        f"Candidate train pool   : {len(candidate_train_pool)}"
    )

    print(
        f"Dynamic GEPA Train     : "
        f"{args.gepa_train_size}"
    )

    print(
        "Validation             : "
        "UNTOUCHED"
    )

    print(
        "Test                   : "
        "UNTOUCHED"
    )

    print("=" * 80)
elif args.gepa_sampling == "sess_rep":

    print("\n" + "=" * 80)
    print("SESS-REP DEV + IPOMP-DIFFICULTY TRAIN")
    print("=" * 80)

    # ------------------------------------------------------------
    # 1. SESS-REP selects Dev
    # ------------------------------------------------------------
    dev_set = select_sess_rep_examples(
        examples=train_pool,
        sample_size=args.gepa_dev_size,
        model_name="openai/clip-vit-base-patch32",
        cache_dir=".cache/dca_embeddings",
    )

    dev_ids = {
    _example_id(example)
    for example in dev_set
}

    print(f"\nSESS-REP Dev selected : {len(dev_set)}")

    # ------------------------------------------------------------
    # 2. Remove SESS-REP Dev examples
    # ------------------------------------------------------------
    train_candidates = [
        example
        for example in train_pool
        if _example_id(example) not in dev_ids
    ]
    
    candidate_train_pool = train_candidates

    print(
        f"Remaining Train candidates : "
        f"{len(train_candidates)}"
    )

    # ------------------------------------------------------------
    # 3. ONLY the existing IPOMP-Difficulty Train logic
    # ------------------------------------------------------------
    train_set, train_info = select_ipomp_difficulty_train_examples(
        examples=train_candidates,
        sample_size=args.gepa_train_size,
        comparator=comparator,
        metric=dca_comparison_metric,
        model_name="openai/clip-vit-base-patch32",
        cache_dir=".cache/dca_embeddings",
        hard_pool_size=args.ipomp_hard_pool_size,
        seed=args.split_seed,
    )

    # ------------------------------------------------------------
    # 4. Safety check: Train/Dev must not overlap
    # ------------------------------------------------------------
    train_ids = {
    _example_id(example)
    for example in train_set
}

    overlap = train_ids & dev_ids

    if overlap:
        raise RuntimeError(
            "SESS-REP + IPOMP-Difficulty produced "
            f"Train/Dev overlap: {sorted(overlap)}"
        )

    print("\n" + "=" * 80)
    print("SESS-REP + IPOMP-DIFFICULTY SELECTION COMPLETE")
    print("=" * 80)
    print(f"Original pool : {len(candidate_train_pool)}")
    print(f"Train         : {len(train_set)}")
    print(f"Dev           : {len(dev_set)}")
    print(f"Train/Dev overlap : {len(overlap)}")
    print(f"Validation    : {len(val_set)} [UNTOUCHED]")
    print(f"Test          : {len(test_set)} [UNTOUCHED]")
    print("=" * 80)
elif args.gepa_sampling == "sess_lc":

    print("\n" + "=" * 80)
    print("SESS-LC DEV + IPOMP-DIFFICULTY TRAIN")
    print("=" * 80)

    # ------------------------------------------------------------
    # 1. SESS-LC selects Dev
    # ------------------------------------------------------------

    dev_set, candidate_train_pool = (
        select_sess_lc_dev_and_get_remaining(
            examples=train_pool,
            dev_size=args.gepa_dev_size,
            comparator=comparator,
            metric=dca_comparison_metric,
        )
    )

    print(
        f"\nSESS-LC Dev selected : {len(dev_set)}"
    )

    print(
        f"Remaining train candidates : "
        f"{len(candidate_train_pool)}"
    )

    # ------------------------------------------------------------
    # 2. Existing IPOMP-Difficulty / Hard + Diverse
    #    Train selection
    # ------------------------------------------------------------

    train_set, train_info = (
        select_ipomp_difficulty_train_examples(
            examples=candidate_train_pool,
            sample_size=args.gepa_train_size,
            comparator=comparator,
            metric=dca_comparison_metric,
            model_name="openai/clip-vit-base-patch32",
            cache_dir=".cache/dca_embeddings",
            hard_pool_size=args.ipomp_hard_pool_size,
            seed=args.split_seed,
        )
    )

    # ------------------------------------------------------------
    # 3. Verify Train / Dev sizes
    # ------------------------------------------------------------

    if len(train_set) != args.gepa_train_size:
        raise RuntimeError(
            "SESS-LC/IPOMP-Difficulty produced "
            "the wrong training-set size. "
            f"Expected {args.gepa_train_size}, "
            f"got {len(train_set)}."
        )

    if len(dev_set) != args.gepa_dev_size:
        raise RuntimeError(
            "SESS-LC produced the wrong "
            "development-set size. "
            f"Expected {args.gepa_dev_size}, "
            f"got {len(dev_set)}."
        )

    # ------------------------------------------------------------
    # 4. Verify NO Train / Dev overlap
    # ------------------------------------------------------------

    train_ids = {
        _example_id(example)
        for example in train_set
    }

    dev_ids = {
        _example_id(example)
        for example in dev_set
    }

    overlap = train_ids & dev_ids

    if overlap:
        raise RuntimeError(
            "SESS-LC/IPOMP-Difficulty produced "
            f"Train/Dev overlap: {sorted(overlap)}"
        )

    # ------------------------------------------------------------
    # 5. Verify Validation / Test remain untouched
    # ------------------------------------------------------------

    val_ids = {
        _example_id(example)
        for example in val_set
    }

    test_ids = {
        _example_id(example)
        for example in test_set
    }

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    dev_val_overlap = dev_ids & val_ids
    dev_test_overlap = dev_ids & test_ids

    if train_val_overlap:
        raise RuntimeError(
            "SESS-LC Train overlaps Validation set: "
            f"{sorted(train_val_overlap)}"
        )

    if train_test_overlap:
        raise RuntimeError(
            "SESS-LC Train overlaps Test set: "
            f"{sorted(train_test_overlap)}"
        )

    if dev_val_overlap:
        raise RuntimeError(
            "SESS-LC Dev overlaps Validation set: "
            f"{sorted(dev_val_overlap)}"
        )

    if dev_test_overlap:
        raise RuntimeError(
            "SESS-LC Dev overlaps Test set: "
            f"{sorted(dev_test_overlap)}"
        )

    print("\n" + "=" * 80)
    print("SESS-LC SAMPLING COMPLETE")
    print("=" * 80)

    print(
        f"Original training pool : {len(train_pool)}"
    )

    print(
        f"SESS-LC Dev            : {len(dev_set)}"
    )

    print(
        f"GEPA Train             : {len(train_set)}"
    )

    print(
        f"Unused candidates      : "
        f"{len(candidate_train_pool) - len(train_set)}"
    )

    print(
        f"Validation             : "
        f"{len(val_set)} [UNTOUCHED]"
    )

    print(
        f"Test                   : "
        f"{len(test_set)} [UNTOUCHED]"
    )

    print("=" * 80)
elif args.gepa_sampling == "wrep":

    print("\n" + "=" * 80)
    print("SESS-WREP DEV + IPOMP-DIFFICULTY TRAIN")
    print("=" * 80)

    # ------------------------------------------------------------
    # WREP performs the complete Train + Dev selection:
    #
    #   Original Train pool (254)
    #          |
    #          +----> WREP selects Dev (30)
    #          |
    #          +----> remove Dev
    #                    |
    #                    +----> difficulty / hard pool
    #                              |
    #                              +----> CLIP diversity
    #                                       |
    #                                       +----> Train (50)
    #
    # Validation and Test are never used for sampling.
    # ------------------------------------------------------------

    (
        train_set,
        dev_set,
        train_candidates,
        train_info,
        dev_scored_examples,
    ) = select_sess_wrep_dev_and_difficulty_train(
        examples=train_pool,
        train_size=args.gepa_train_size,
        dev_size=args.gepa_dev_size,
        comparator=comparator,
        metric=dca_comparison_metric,
        model_name="openai/clip-vit-base-patch32",
        cache_dir=".cache/dca_embeddings/wrep",
        hard_pool_size=args.ipomp_hard_pool_size,
        wrep_lambda=args.wrep_lambda,
        seed=args.split_seed,
    )

    # The remaining candidates are used only for bookkeeping.
    candidate_train_pool = train_candidates

    # ------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------

    train_ids = {
        _example_id(example)
        for example in train_set
    }

    dev_ids = {
        _example_id(example)
        for example in dev_set
    }

    val_ids = {
        _example_id(example)
        for example in val_set
    }

    test_ids = {
        _example_id(example)
        for example in test_set
    }

    train_dev_overlap = train_ids & dev_ids
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    dev_val_overlap = dev_ids & val_ids
    dev_test_overlap = dev_ids & test_ids

    if train_dev_overlap:
        raise RuntimeError(
            "SESS-WREP Train/Dev overlap detected: "
            f"{sorted(train_dev_overlap)}"
        )

    if train_val_overlap:
        raise RuntimeError(
            "SESS-WREP Train/Validation overlap detected: "
            f"{sorted(train_val_overlap)}"
        )

    if train_test_overlap:
        raise RuntimeError(
            "SESS-WREP Train/Test overlap detected: "
            f"{sorted(train_test_overlap)}"
        )

    if dev_val_overlap:
        raise RuntimeError(
            "SESS-WREP Dev/Validation overlap detected: "
            f"{sorted(dev_val_overlap)}"
        )

    if dev_test_overlap:
        raise RuntimeError(
            "SESS-WREP Dev/Test overlap detected: "
            f"{sorted(dev_test_overlap)}"
        )

    # ------------------------------------------------------------
    # Final WREP selection summary
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("SESS-WREP SAMPLING COMPLETE")
    print("=" * 80)

    print(
        f"Original Train pool : {len(train_pool)}"
    )

    print(
        f"WREP Dev            : {len(dev_set)}"
    )

    print(
        f"Remaining candidates: {len(candidate_train_pool)}"
    )

    print(
        f"WREP Train          : {len(train_set)}"
    )

    print(
        f"Train/Dev overlap   : {len(train_dev_overlap)}"
    )

    print(
        f"Validation          : {len(val_set)} [UNTOUCHED]"
    )

    print(
        f"Test                : {len(test_set)} [UNTOUCHED]"
    )

    print("=" * 80)
elif args.gepa_sampling == "wrep_train":

    print("\n" + "=" * 80)
    print("SESS-WREP DEV + WREP TRAIN")
    print("=" * 80)

    (
        train_set,
        dev_set,
        train_candidates,
        train_info,
        dev_scored_examples,
    ) = select_sess_wrep_dev_and_wrep_train(
        examples=train_pool,
        train_size=args.gepa_train_size,
        dev_size=args.gepa_dev_size,
        comparator=comparator,
        metric=dca_comparison_metric,
        model_name="openai/clip-vit-base-patch32",
        cache_dir=".cache/dca_embeddings/wrep_train",
        wrep_lambda=args.wrep_lambda,
        seed=args.split_seed,
    )

    candidate_train_pool = train_candidates

    # ------------------------------------------------------------
    # Explicit overlap checks
    # ------------------------------------------------------------

    train_ids = {
        _example_id(example)
        for example in train_set
    }

    dev_ids = {
        _example_id(example)
        for example in dev_set
    }

    val_ids = {
        _example_id(example)
        for example in val_set
    }

    test_ids = {
        _example_id(example)
        for example in test_set
    }

    train_dev_overlap = train_ids & dev_ids
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    dev_val_overlap = dev_ids & val_ids
    dev_test_overlap = dev_ids & test_ids

    if train_dev_overlap:
        raise RuntimeError(
            "WREP-Train Train/Dev overlap detected: "
            f"{sorted(train_dev_overlap)}"
        )

    if train_val_overlap:
        raise RuntimeError(
            "WREP-Train Train/Validation overlap detected: "
            f"{sorted(train_val_overlap)}"
        )

    if train_test_overlap:
        raise RuntimeError(
            "WREP-Train Train/Test overlap detected: "
            f"{sorted(train_test_overlap)}"
        )

    if dev_val_overlap:
        raise RuntimeError(
            "WREP-Train Dev/Validation overlap detected: "
            f"{sorted(dev_val_overlap)}"
        )

    if dev_test_overlap:
        raise RuntimeError(
            "WREP-Train Dev/Test overlap detected: "
            f"{sorted(dev_test_overlap)}"
        )

    print("\n" + "=" * 80)
    print("SESS-WREP DEV + WREP TRAIN COMPLETE")
    print("=" * 80)

    print(f"Original Train pool : {len(train_pool)}")
    print(f"WREP Dev            : {len(dev_set)}")
    print(f"Remaining candidates: {len(train_candidates)}")
    print(f"WREP Train          : {len(train_set)}")
    print(f"Train/Dev overlap   : {len(train_dev_overlap)}")
    print(f"Validation          : {len(val_set)} [UNTOUCHED]")
    print(f"Test                : {len(test_set)} [UNTOUCHED]")
    print("=" * 80)
else:

    # ------------------------------------------------------------
    # EXISTING BEHAVIOR
    #
    # KEEP THIS EXACTLY AS IT WAS.
    #
    # Applies to:
    #   random
    #   hard
    #   balanced
    #   adaptive
    #   diverse_boundary
    #   apex
    # ------------------------------------------------------------
    import random
    rng = random.Random(split_seed)

    shuffled_train_pool = list(train_pool)
    rng.shuffle(shuffled_train_pool)

    dev_set = shuffled_train_pool[
        :args.gepa_dev_size
    ]

    candidate_train_pool = shuffled_train_pool[
        args.gepa_dev_size:
    ]
# ────────────────────────────────────────────────────────────────
# Validation and test sets remain unchanged
# ────────────────────────────────────────────────────────────────

val_names = _pair_names(val_set)
test_names = _pair_names(test_set)

split_manifest = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "cache_path": split_latest_path,
    "dataset_cache_tag": _dataset_tag,
    "seed": split_seed,
    "stratify_by": _split_stratify_by,
    "ratios": _split_ratios,
    "pairs_dir": os.path.abspath(pairs_dir),
    "gt_dir": os.path.abspath(gt_dir),
    "total_examples": len(dataset),

    "gepa_dev": {
        "count": len(dev_set),
        "pair_names": _pair_names(dev_set),
    },

    "val": {
        "count": len(val_names),
        "pair_names": val_names,
    },

    "test": {
        "count": len(test_names),
        "pair_names": test_names,
    },
}

with open(split_latest_path, "w", encoding="utf-8") as f:
    json.dump(split_manifest, f, indent=2)

print(f"Saved split manifest: {split_latest_path}")

if args.gepa_sampling == "performance_train_dev":
    _candidate_train_pool_for_log = performance_train_dev_candidate_pool
else:
    _candidate_train_pool_for_log = candidate_train_pool

print(
    f"  | candidate train pool: {len(_candidate_train_pool_for_log)}"
    f"  | dev: {len(dev_set)}"
    f"  | val: {len(val_set)}"
    f"  | test: {len(test_set)}"
)

print(
    f"  GEPA requested train size: {args.gepa_train_size}"
    f"  | GEPA dev size: {args.gepa_dev_size}"
)

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

cache_dir = dotenv.get_key(dotenv.find_dotenv(), "CACHED_EXTRACTOR_DIR") or ".cache"

if not args.skip_load:
    explicit_cached_extractor = args.load_extractor or dotenv.get_key(dotenv.find_dotenv(), "CACHED_EXTRACTOR_PATH")
    auto_cached_extractor = None
    if explicit_cached_extractor is None and args.auto_load_compatible_cache and args.optimizer != "all":
        auto_cached_extractor = _find_matching_cached_extractor(
            cache_root=cache_dir,
            strategy_tag=_strategy_display,
            demo_tag=_demo_tag.lstrip("_"),
            model_tag=_model_display,
            cot_tag=_cot_tag,
            re_tag=_re_tag,
            instr_tag=_instr_tag,
            cfd_tag=_cfd_tag,
            dataset_tag=_dataset_tag,
            split_seed_value=split_seed,
        )
    cached_extractor_path = explicit_cached_extractor or auto_cached_extractor
    if cached_extractor_path and os.path.exists(cached_extractor_path) and os.path.isfile(cached_extractor_path):
        load_source = "explicit path" if explicit_cached_extractor else "auto-matched cache"
        print(f"Loading cached extractor from ({load_source}): {cached_extractor_path}")
        comparator.load(cached_extractor_path)
        print("Extractor loaded successfully\n")
    else:
        if explicit_cached_extractor is None and not args.auto_load_compatible_cache:
            print("No explicit cached extractor provided; automatic cache matching is disabled by default. Starting from scratch\n")
        elif explicit_cached_extractor is None and args.optimizer == "all":
            print("Automatic cache matching is disabled for --optimizer all. Starting from scratch\n")
        else:
            print("No cached extractor found, starting from scratch\n")
else:
    print("Skipping extractor loading (--skip-load)\n")

# ── Inject initial instruction from file (e.g. human-written prompt) ─────────
# Replaces the default signature docstring instruction so MIPROv2 optimizes
# starting from this human prompt instead of the auto-generated one.
if args.initial_instruction_file:
    _instr_path = os.path.abspath(args.initial_instruction_file)
    if not os.path.isfile(_instr_path):
        print(f"[ERROR] --initial-instruction-file not found: {_instr_path}")
        sys.exit(1)
    with open(_instr_path, encoding="utf-8") as _fh:
        _initial_instruction = _fh.read().strip()
    _signature = _get_program_signature(comparator)
    _set_program_signature(comparator, _signature.with_instructions(_initial_instruction))
    print(f"Initial instruction loaded from: {_instr_path}\n")

if _should_clear_descs:
    _signature = _get_program_signature(comparator)
    for _fname in list(_signature.fields):
        _signature = _signature.with_updated_fields(_fname, desc="")
    _set_program_signature(comparator, _signature)
    print("Field descriptions cleared (--clear-field-descs).\n")

# ── Verify seed instruction on the live signature object ─────────────────
# Reads back from comparator.signature.instructions so we prove the object
# MIPROv2 will receive already contains the human prompt, not the class docstring.
# Also prints field descriptions — those are fixed (MIPROv2 doesn't touch them)
# but are always part of every LLM call.
_live_signature = _get_program_signature(comparator)
_live_instruction = _live_signature.instructions
print(f"\n{'='*80}")
print("[SEED INSTRUCTION VERIFICATION] — full text read back from the active predictor signature:")
print(f"{'='*80}")
print(_live_instruction)
print(f"\n--- Field descriptions (cleared={_should_clear_descs}) ---")
for fname, finfo in _live_signature.fields.items():
    _desc = getattr(finfo, "json_schema_extra", {}) or {}
    _desc_str = _desc.get("desc", "") if isinstance(_desc, dict) else ""
    _desc_display = _desc_str[:200] if _desc_str else "(empty)"
    print(f"  [{fname}]  {_desc_display}")
print(f"{'='*80}\n")
if args.initial_instruction_file:
    # Hard check: the live instruction must exactly match what we loaded from file
    if _live_instruction.strip() != _initial_instruction.strip():
        print("[ERROR] Seed instruction mismatch — live signature does not match the file content!")
        sys.exit(1)
    else:
        print("[OK] Seed instruction verified: live signature matches the instruction file exactly.\n")

# ── Save seed state (DSPy's own dump_state — same format as .save()) ─────────
_seed_prompt_path = os.path.join(_log_dir, f"{_log_prefix}_dca_run_{current_time}_seed_state.json")
with open(_seed_prompt_path, "w", encoding="utf-8") as _fp:
    json.dump(comparator.dump_state(), _fp, indent=2, ensure_ascii=False)
print(f"Seed state saved   -> {_seed_prompt_path}\n")

# ────────────────────────────────────────────────────────────────
# GEPA TRAINING SAMPLE SELECTION
# ────────────────────────────────────────────────────────────────
# The 30-sample dev_set remains fixed.
# The remaining candidate_train_pool is used to construct the
# GEPA training set.
#
# random -> randomly selects 50 examples
# hard   -> selects 50 examples with the lowest baseline scores
# ────────────────────────────────────────────────────────────────
apex_controller = None

if args.gepa_sampling == "apex":
    apex_controller = APEXController(
        examples=candidate_train_pool,
        sample_size=args.gepa_train_size,
        evaluation_size=args.gepa_dev_size,
        lookback=args.apex_lookback,
        seed=args.apex_seed,
    )

    print("\n" + "=" * 70)
    print("APEX SAMPLING ENABLED")
    print("=" * 70)
    print(f"Training size  : {args.gepa_train_size}")
    print(f"APEX lookback      : {args.apex_lookback}")
    print(f"APEX seed          : {args.apex_seed}")


elif args.gepa_sampling == "random":

    rng = random.Random(split_seed)

    train_set = rng.sample(
        candidate_train_pool,
        args.gepa_train_size
    )

    print("\n" + "=" * 80)
    print("GEPA TRAINING SAMPLE SELECTION")
    print("=" * 80)
    print("Sampling strategy : RANDOM")
    print(f"Training samples  : {len(train_set)}")
    print(f"Development set   : {len(dev_set)}")
    print("=" * 80 + "\n")



elif args.gepa_sampling == "hard":

    train_set = select_hard_examples(
        examples=candidate_train_pool,
        comparator=comparator,
        metric=dca_comparison_metric,
        sample_size=args.gepa_train_size,
    )

    print("\n" + "=" * 80)
    print("GEPA TRAINING SAMPLE SELECTION")
    print("=" * 80)
    print("Sampling strategy : HARD")
    print(f"Training samples  : {len(train_set)}")
    print(f"Development set   : {len(dev_set)}")
    print("=" * 80 + "\n")

elif args.gepa_sampling == "balanced":

    rng = random.Random(split_seed)

    hard_size = args.gepa_train_size // 2
    random_size = args.gepa_train_size - hard_size

    # Select hard examples
    hard_examples = select_hard_examples(
        examples=candidate_train_pool,
        comparator=comparator,
        metric=dca_comparison_metric,
        sample_size=hard_size,
    )

    # Remove selected hard examples from the pool
    hard_ids = {id(example) for example in hard_examples}

    remaining_examples = [
        example
        for example in candidate_train_pool
        if id(example) not in hard_ids
    ]

    # Select remaining examples randomly
    random_examples = rng.sample(
        remaining_examples,
        random_size
    )

    # Combine hard + random examples
    train_set = hard_examples + random_examples

    # Shuffle final training set
    rng.shuffle(train_set)

    print("\n" + "=" * 80)
    print("GEPA TRAINING SAMPLE SELECTION")
    print("=" * 80)
    print("Sampling strategy : BALANCED")
    print(f"Hard examples     : {len(hard_examples)}")
    print(f"Random examples   : {len(random_examples)}")
    print(f"Training samples  : {len(train_set)}")
    print(f"Development set   : {len(dev_set)}")
    print("=" * 80 + "\n")
elif args.gepa_sampling == "diverse_boundary_informative":

    train_set, selection_info = (
        select_diverse_boundary_informative_examples(
            examples=candidate_train_pool,
            sample_size=args.gepa_train_size,
            n_clusters=args.diverse_clusters,
            representative_fraction=args.diverse_representative_fraction,
            boundary_fraction=args.diverse_boundary_fraction,
            informative_fraction=args.diverse_informative_fraction,
            seed=args.split_seed,
            cache_dir=".cache/dca_embeddings",
            comparator=comparator,
            metric=dca_comparison_metric,
        )
    )
    print("\n" + "=" * 80)
    print("GEPA TRAINING SAMPLE SELECTION")
    print("=" * 80)

    print(
        "Sampling strategy : DIVERSE + BOUNDARY + INFORMATIVE"
    )

    print(
        f"Candidate pool    : "
        f"{len(candidate_train_pool)}"
    )

    print(
        f"Training samples  : "
        f"{len(train_set)}"
    )

    print(
        f"K-means clusters  : "
        f"{args.diverse_clusters}"
    )

    print(
        f"Representative    : "
        f"{selection_info['representative_count']}"
    )

    print(
        f"Boundary          : "
        f"{selection_info['boundary_count']}"
    )

    print(
        f"Development set   : "
        f"{len(dev_set)}"
    )

    print("=" * 80 + "\n")    
        
elif args.gepa_sampling == "adaptive":

    # Initial adaptive training set.
    # Round 1 starts with a random sample.
    rng = random.Random(args.adaptive_seed)

    train_set = rng.sample(
        candidate_train_pool,
        args.gepa_train_size
    )

    # ---------------------------------------------------------
    # LOG ROUND 1 SELECTED EXAMPLES
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("ADAPTIVE SAMPLING - ROUND 1 INITIAL TRAINING SET")
    print("=" * 80)
    print("Sampling strategy : ADAPTIVE")
    print("Initial selection : RANDOM")
    print(f"Training samples  : {len(train_set)}")
    print(f"Development set   : {len(dev_set)}")
    print(f"Adaptive rounds   : {args.adaptive_rounds}")
    print(f"Exploration       : {args.adaptive_exploration:.0%}")

    print("\nROUND 1 SELECTED EXAMPLES")
    print("-" * 80)

    for rank, example in enumerate(train_set, start=1):

        pair_name = getattr(
            example,
            "pair_name",
            f"example_{rank}"
        )

        print(
            f"{rank:3d}. "
            f"pair={pair_name} | "
            f"type=INITIAL_RANDOM"
        )

    print("-" * 80)
    print("=" * 80 + "\n")
elif args.gepa_sampling == "true_balanced":

    print("\n" + "=" * 80)
    print("TRUE CATEGORY-BALANCED SAMPLING")
    print("=" * 80)

    print(f"Training samples     : {len(train_set)}")
    print(f"Development set      : {len(dev_set)}")
    print(f"Validation set       : {len(val_set)}")
    print(f"Test set             : {len(test_set)}")

    print("=" * 80 + "\n")
    
elif args.gepa_sampling == "performance_class_adaptive":

    # ------------------------------------------------------------
    # PERFORMANCE-BASED CLASS-ADAPTIVE SAMPLER
    # ------------------------------------------------------------

    if len(candidate_train_pool) < args.gepa_train_size:
        raise ValueError(
            f"Candidate training pool "
            f"({len(candidate_train_pool)}) is smaller "
            f"than requested training size "
            f"({args.gepa_train_size})."
        )

    performance_class_sampler = (
        PerformanceClassAdaptiveSampler(
            examples=candidate_train_pool,
            sample_size=args.gepa_train_size,
            seed=split_seed
            if hasattr(args, "split_seed")
            else split_seed,
            min_class_fraction=0.05,
            max_class_fraction=0.50,
        )
    )

    # Round 1:
    # Start with a random training set.
    train_set, selection_info = (
        performance_class_sampler.select_initial_training_set()
    )

    print("\n" + "=" * 80)
    print("PERFORMANCE CLASS-ADAPTIVE TRAINING")
    print("=" * 80)

    print(
        f"Initial training samples : "
        f"{len(train_set)}"
    )

    print(
        f"Fixed Dev                : "
        f"{len(dev_set)}"
    )

    print(
        f"Candidate pool           : "
        f"{len(candidate_train_pool)}"
    )

    print("\nInitial class distribution:")

    for category, count in sorted(
        selection_info["class_counts"].items()
    ):
        print(
            f"  {category}: "
            f"{count} "
            f"({count / len(train_set):.1%})"
        )

    print("=" * 80)
# ── Evaluation Setup ─────────────────────────────────────────────
# dca_comparison_metric returns a weighted float 0-1:
#   0.6 * classification_score + 0.4 * change_score
# dspy.Evaluate reports scores on a 0-100 scale.
evaluate = dspy.Evaluate(
    devset=dev_set,
    metric=dca_comparison_metric,
    num_threads=args.eval_num_threads,
    display_progress=True,
    display_table=True,
)


# ── Teacher / prompt / reflection LMs + optional multimodal GEPA reflection ───
# All optimizer models are CLI-driven (no hard-coding). Teacher/prompt overrides
# default to None -> the student LM is used (unchanged behavior). GEPA reflection
# defaults to the .env REFLECTION_LM_NAME (reflectin_lm, typically gpt-5.2).
def _build_aux_lm(model_name):
    """Build an auxiliary LM (teacher/prompt/reflection) from a name, or None."""
    if not model_name:
        return None
    _kw = {"cache": cache_enabled}
    if args.teacher_reasoning_effort:
        _kw["reasoning_effort"] = args.teacher_reasoning_effort
    return create_lm(model_name, **_kw)

_bootstrap_teacher_lm = _build_aux_lm(args.bootstrap_teacher_model)
_miprov2_teacher_lm   = _build_aux_lm(args.miprov2_teacher_model)
_miprov2_prompt_lm    = _build_aux_lm(args.miprov2_prompt_model)

# GEPA reflection LM: explicit override, else the .env reflection LM (gpt-5.2).
_gepa_reflection_lm   = _build_aux_lm(args.gepa_reflection_model) or reflectin_lm
_metric_with_feedback = dca_comparison_metric_with_feedback

# Multimodal GEPA reflection (parity with the tournament): send the document
# image-pairs to the reflection LM so it can SEE what changed. Reuses the same
# machinery as dca_tournaments (image store + reflective-dataset patch).
_gepa_multimodal = bool(args.gepa_reflection_images) and args.optimizer in ("gepa", "all")
if _gepa_multimodal:
    from dca_tournaments.gepa_utils import (
        MultimodalReflectionLM,
        make_image_storing_metric,
        patch_gepa_reflective_dataset,
    )
    patch_gepa_reflective_dataset(send_images=True)
    _gepa_reflection_lm = MultimodalReflectionLM(
        base_lm=_gepa_reflection_lm,
        detail=args.gepa_reflection_detail,
        max_images=args.gepa_reflection_max_images,
    )
    _metric_with_feedback = make_image_storing_metric(dca_comparison_metric_with_feedback)
    print(f"GEPA multimodal reflection ENABLED (detail={args.gepa_reflection_detail}, "
          f"max_images={args.gepa_reflection_max_images}).\n")


# ── Optimization Strategies ──────────────────────────────────────
custom_strategies = []
strategy_by_name = {}


def _register_strategy(strategy):
    custom_strategies.append(strategy)
    strategy_by_name[strategy.name] = strategy

if args.optimizer in ("bootstrap", "all"):
    _register_strategy(
        create_bootstrap_strategy(
            name="Balanced_DCA_Bootstrap",
            description="Balanced approach with equal labeled and bootstrapped demos for DCA classification",
            max_labeled_demos=args.bootstrap_labeled_demos,
            max_bootstrapped_demos=args.bootstrap_bootstrapped_demos,
            num_candidate_programs=args.bootstrap_num_candidate_programs,
            num_threads=args.bootstrap_num_threads,
            max_rounds=args.bootstrap_max_rounds,
            teacher_lm=_bootstrap_teacher_lm,   # None -> student LM (unchanged default)
        )
    )
    # _register_strategy(
    #     create_bootstrap_strategy(
    #         name="Label_Heavy_DCA_Bootstrap",
    #         description="Label-heavy approach focusing on ground truth examples for DCA classification",
    #         max_labeled_demos=1,        # One extra human-verified example vs balanced variant
    #         max_bootstrapped_demos=1,   # Fewer teacher demos to reduce noise risk
    #         num_candidate_programs=1,  # Slightly more candidates for broader coverage
    #         num_candidate_traces=(2,5), # Generate 2-5 candidate traces per example for bootstrapping
    #         num_threads=4,
    #         max_rounds=1,
    #     )
    # )

if args.optimizer in ("miprov2", "all"):
    _register_strategy(
        create_miprov2_strategy(
            name=f"MIPROv2_DCA_{args.miprov2_auto.capitalize()}",
            description=f"MIPROv2 ({args.miprov2_auto}) -- Bayesian instruction + demo search",
            auto=args.miprov2_auto,
            max_labeled_demos=args.max_labeled_demos,
            max_bootstrapped_demos=args.max_bootstrapped_demos,
            num_threads=args.miprov2_num_threads,
            teacher_lm=_miprov2_teacher_lm,   # None -> student LM
            prompt_lm=_miprov2_prompt_lm,     # None -> student LM (proposer)
        )
    )

if args.optimizer in ("gepa", "all"):
    _register_strategy(
        create_gepa_strategy(
            name=f"GEPA_DCA_{args.gepa_auto.capitalize()}",
            description=f"GEPA ({args.gepa_auto}) -- reflection-LM guided prompt improvement",
            auto=args.gepa_auto,
            num_threads=args.gepa_num_threads,
            track_stats=args.gepa_track_stats,
            use_merge=args.gepa_use_merge,
            reflection_lm=_gepa_reflection_lm,
            reflection_minibatch_size=args.gepa_reflection_minibatch_size, # plain gpt-5.2 or multimodal wrapper
        )
    )

if args.optimizer in ("simba", "all"):
    _register_strategy(
        create_simba_strategy(
            name=f"SimBA_DCA_Steps{args.simba_max_steps}",
            description=f"SimBA ({args.simba_max_steps} steps) -- iterative prompt refinement",
            max_steps=args.simba_max_steps,
            max_labeled_demos=args.simba_labeled_demos,
            num_threads=args.simba_num_threads,
            bsize=args.simba_bsize,
            num_candidates=args.simba_num_candidates,
        )
    )

print(f"\nOptimization strategies ({len(custom_strategies)}):")
for i, s in enumerate(custom_strategies, 1):
    print(f"  {i}. {s.name} ({s.optimizer_type}): {s.description}")

# ============================================================
# OPTIMIZATION
# ============================================================
# ============================================================
# OPTIMIZATION
# ============================================================

optimized_comparator = comparator

# existing optimization branches below
if args.gepa_sampling == "adaptive":
    ...
elif args.gepa_sampling == "apex":
    ...
elif args.gepa_sampling == "performance_train":
    ...
elif args.gepa_sampling == "performance_class_adaptive":
    ...   
elif args.gepa_sampling == "performance_train_dev":
    ...
elif args.gepa_sampling == "ipomp_difficulty":
    ...    
elif args.gepa_sampling == "performance_train_sess_lc":
    ...
adaptive_history = []

if args.gepa_sampling not in (
    "adaptive",
    "apex",
    "ipomp",
    "performance_train",
    "performance_class_adaptive",
    "performance_train_dev",
    "ipomp_difficulty",
    "performance_train_sess_lc",
    ):

    # --------------------------------------------------------
    # EXISTING OPTIMIZATION PATH
    # --------------------------------------------------------

    runner = OptimizationRunner(
        metric=dca_comparison_metric,
        student_model=comparator,
        train_set=train_set,
        dev_set=dev_set,
        val_set=val_set,
        evaluate_fn=evaluate,
        metric_with_feedback=_metric_with_feedback,
        reflection_lm=_gepa_reflection_lm,
        apex_controller=apex_controller,
    )

    print(f"\n{'='*80}")
    print(f"Starting Optimization...")
    print(f"{'='*80}\n")

    optimized_comparator = None
    best_score = None

    try:
        optimized_comparator, best_score = runner.run_optimization(
            custom_strategies
        )

    except Exception as exc:
        failed_at = datetime.now().isoformat(timespec="seconds")
        rerun_command = _build_rerun_from_seed_command(_seed_prompt_path)

        failure_summary = {
            "run_type": "extract_dca",
            "status": "failed",
            "started_at": pipeline_started_at,
            "failed_at": failed_at,
            "total_duration_seconds": round(
                time.time() - pipeline_t0, 1
            ),
            "model_name": args.model_name or os.getenv(
                "MODEL_NAME",
                "Unknown_Model"
            ),
            "optimizer_requested": args.optimizer,
            "pairs_dir": os.path.abspath(pairs_dir),
            "gt_dir": os.path.abspath(gt_dir),
            "split_seed": split_seed,
            "split_manifest_path": split_latest_path,
            "seed_state_path": _seed_prompt_path,
            "log_path": _stdout_log_path,
            "error": str(exc),
            "runner_summary": getattr(
                runner,
                "run_summary",
                {}
            ),
            "rerun_from_seed_command": rerun_command,
        }

        sys.stdout = _tee_stdout._original
        sys.stderr = _tee_stderr._original
        _log_file.close()

        failure_summary_log_path = os.path.join(
            _log_dir,
            f"{_log_prefix}_dca_run_{current_time}_error_summary.json"
        )

        failure_summary_cache_path = os.path.join(
            cache_dir,
            f"{_log_prefix}_dca_run_{current_time}_error_summary.json"
        )

        for _summary_path in (
            failure_summary_log_path,
            failure_summary_cache_path
        ):
            try:
                with open(
                    _summary_path,
                    "w",
                    encoding="utf-8"
                ) as _fp:
                    json.dump(
                        failure_summary,
                        _fp,
                        indent=2,
                        ensure_ascii=False
                    )

                print(
                    f"Failure summary saved to: {_summary_path}",
                    file=sys.__stdout__
                )

            except OSError as summary_exc:
                print(
                    f"Could not save failure summary "
                    f"{_summary_path}: {summary_exc}",
                    file=sys.__stdout__
                )

        print(
            "\nOptimization aborted after an unrecoverable "
            "optimizer failure.",
            file=sys.__stdout__
        )

        print(
            f"Seed state is preserved at: {_seed_prompt_path}",
            file=sys.__stdout__
        )

        print(
            "Rerun from the saved seed state with:",
            file=sys.__stdout__
        )

        print(
            f"  {rerun_command}",
            file=sys.__stdout__
        )

        raise SystemExit(1) from exc
elif args.gepa_sampling == "performance_class_adaptive":

    if args.optimizer != "gepa":
        raise ValueError(
            "Performance class-adaptive sampling "
            "is currently supported only with GEPA."
        )

    print("\n" + "=" * 80)
    print("STARTING PERFORMANCE CLASS-ADAPTIVE GEPA")
    print("=" * 80)

    optimized_comparator = comparator
    best_score = None
    runner = None

    class_adaptive_history = []

    for adaptive_round in range(
        1,
        args.adaptive_rounds + 1
    ):

        print("\n" + "#" * 80)
        print(
            f"CLASS-ADAPTIVE ROUND "
            f"{adaptive_round}/"
            f"{args.adaptive_rounds}"
        )
        print("#" * 80)

        # --------------------------------------------------------
        # GEPA optimization using CURRENT training distribution
        # --------------------------------------------------------

        runner = OptimizationRunner(
            metric=dca_comparison_metric,
            student_model=optimized_comparator,
            train_set=train_set,
            dev_set=dev_set,
            val_set=val_set,
            evaluate_fn=evaluate,
            metric_with_feedback=_metric_with_feedback,
            reflection_lm=_gepa_reflection_lm,
            apex_controller=apex_controller,
        )

        round_comparator = None
        round_score = None

        try:

            round_comparator, round_score = (
                runner.run_optimization(
                    custom_strategies
                )
            )

        except Exception as exc:

            print(
                f"[ERROR] Class-adaptive round "
                f"{adaptive_round} failed: {exc}"
            )

            raise

        # Carry optimized comparator into next round
        if round_comparator is not None:
            optimized_comparator = round_comparator

        if round_score is not None:

            if (
                best_score is None
                or round_score > best_score
            ):
                best_score = round_score

        print(
            f"\nRound score : {round_score}"
        )

        print(
            f"Best score  : {best_score}"
        )

        # --------------------------------------------------------
        # Stop after final round
        # --------------------------------------------------------

        if adaptive_round == args.adaptive_rounds:
            break

        # --------------------------------------------------------
        # PERFORMANCE-BASED CLASS ADAPTATION
        # --------------------------------------------------------
        #
        # Evaluate every candidate example using the CURRENT
        # optimized comparator.
        #
        # Then:
        #
        #   example scores
        #        ↓
        #   group by category
        #        ↓
        #   average score per category
        #        ↓
        #   class error = 1 - class score
        #        ↓
        #   class error determines next allocation
        # --------------------------------------------------------

        train_set, selection_info = (
            performance_class_sampler.select_next_training_set(
                comparator=optimized_comparator,
                metric_fn=dca_comparison_metric,
                round_number=adaptive_round + 1,
            )
        )

        class_adaptive_history.append(
            selection_info
        )

        print(
            f"\nNext training set prepared: "
            f"{len(train_set)} examples"
        )

    print("\n" + "=" * 80)
    print("PERFORMANCE CLASS-ADAPTIVE GEPA COMPLETE")
    print("=" * 80)

    print(
        f"Final Train size : {len(train_set)}"
    )

    print(
        f"Fixed Dev size   : {len(dev_set)}"
    )

    print(
        f"Validation size  : "
        f"{len(val_set)} [UNTOUCHED]"
    )

    print(
        f"Test size        : "
        f"{len(test_set)} [UNTOUCHED]"
    )

    print(
        f"Best score       : {best_score}"
    )

    print("=" * 80)

elif args.gepa_sampling == "adaptive":

    # ========================================================
    # ADAPTIVE GEPA OPTIMIZATION
    # ========================================================

    if args.optimizer != "gepa":
        raise ValueError(
            "Adaptive sampling is currently supported only with GEPA."
        )

    print(f"\n{'='*80}")
    print("Starting ADAPTIVE GEPA Optimization...")
    print(f"{'='*80}\n")

    optimized_comparator = comparator
    best_score = None
    runner = None

    for adaptive_round in range(
        1,
        args.adaptive_rounds + 1
    ):

        print(f"\n{'#'*80}")
        print(
            f"ADAPTIVE ROUND "
            f"{adaptive_round}/{args.adaptive_rounds}"
        )
        print(f"{'#'*80}\n")

        # ----------------------------------------------------
        # Create a new runner using the CURRENT training set
        # ----------------------------------------------------

        runner = OptimizationRunner(
            metric=dca_comparison_metric,
            student_model=optimized_comparator,
            train_set=train_set,
            dev_set=dev_set,
            val_set=val_set,
            evaluate_fn=evaluate,
            metric_with_feedback=_metric_with_feedback,
            reflection_lm=_gepa_reflection_lm,
            apex_controller=apex_controller,
        )

        round_comparator = None
        round_score = None

        try:

            round_comparator, round_score = (
                runner.run_optimization(
                    custom_strategies
                )
            )

        except Exception as exc:

            print(
                f"\n[ERROR] Adaptive round "
                f"{adaptive_round} failed: {exc}"
            )

            raise

        # ----------------------------------------------------
        # Keep the optimized comparator for the next round
        # ----------------------------------------------------

        if round_comparator is not None:
            optimized_comparator = round_comparator

        # ----------------------------------------------------
        # Keep the best score
        # ----------------------------------------------------

        if round_score is not None:

            if (
                best_score is None
                or round_score > best_score
            ):
                best_score = round_score

        print(f"\n{'='*80}")
        print(
            f"ADAPTIVE ROUND {adaptive_round} COMPLETE"
        )
        print(f"Round Score : {round_score}")
        print(f"Best Score  : {best_score}")
        print(f"{'='*80}\n")

        # ----------------------------------------------------
        # Stop after the final round
        # ----------------------------------------------------

        if adaptive_round == args.adaptive_rounds:
            break

        # ----------------------------------------------------
        # Score candidate pool using current optimized prompt
        # ----------------------------------------------------

        scored_examples = _score_dca_examples(
            candidate_train_pool,
            optimized_comparator,
        )

        # ----------------------------------------------------
        # Select NEXT training set adaptively
        # ----------------------------------------------------

               # ----------------------------------------------------

        train_set, selection_info = _select_adaptive_examples(
            scored_examples=scored_examples,
            sample_size=gepa_train_size,
            exploration_fraction=args.adaptive_exploration,
            seed=args.adaptive_seed + adaptive_round,
        )

        selection_info["round"] = adaptive_round + 1

        adaptive_history.append(selection_info)

        print(
            f"Next adaptive training set prepared: "
            f"{len(train_set)} examples"
        )

elif args.gepa_sampling == "apex":

    # ========================================================
    # APEX STAGE 1 GEPA OPTIMIZATION
    # ========================================================

    if args.optimizer != "gepa":
        raise ValueError(
            "APEX sampling is currently supported only with GEPA."
        )

    print(f"\n{'='*80}")
    print("Starting APEX Stage 1 GEPA Optimization...")
    print(f"{'='*80}\n")

    optimized_comparator = comparator
    best_score = None
    runner = None

    apex_history = []

    for apex_round in range(
        1,
        args.adaptive_rounds + 1
    ):

        print(f"\n{'#'*80}")
        print(
            f"APEX ROUND "
            f"{apex_round}/{args.adaptive_rounds}"
        )
        print(f"{'#'*80}\n")

        # ----------------------------------------------------
        # Select training examples for this round
        # ----------------------------------------------------

        train_set = apex_controller.select_training_examples()

        print(
            f"APEX selected "
            f"{len(train_set)} training examples."
        )

        apex_controller.print_status()
        # ----------------------------------------------------
# Select dynamic evaluation examples for this round
# ----------------------------------------------------

        evaluation_set = apex_controller.select_evaluation_examples(
            candidate_train_pool
)

        print("\n" + "=" * 70)
        print("APEX EVALUATION SET")
        print("=" * 70)
        print(f"Evaluation pool : {len(candidate_train_pool)}")
        print(f"Evaluation size : {len(evaluation_set)}")
        print(
            "Evaluation pairs:",
        [example.pair_name for example in evaluation_set]
)
        print("=" * 70)

        apex_controller.print_status()

        # ----------------------------------------------------
        # Create GEPA runner
        # ----------------------------------------------------

        runner = OptimizationRunner(
            metric=dca_comparison_metric,
            student_model=optimized_comparator,
            train_set=train_set,
            dev_set=dev_set,
            val_set=val_set,
            evaluate_fn=evaluate,
            metric_with_feedback=_metric_with_feedback,
            reflection_lm=_gepa_reflection_lm,
            apex_controller=apex_controller,
        )

        round_comparator = None
        round_score = None

        # ----------------------------------------------------
        # Run GEPA
        # ----------------------------------------------------

        try:

            round_comparator, round_score = (
                runner.run_optimization(
                    custom_strategies
                )
            )

        except Exception as exc:

            print(
                f"\n[ERROR] APEX round "
                f"{apex_round} failed: {exc}"
            )

            raise

        # ----------------------------------------------------
        # Keep optimized prompt
        # ----------------------------------------------------

        if round_comparator is not None:
            optimized_comparator = round_comparator

        # ----------------------------------------------------
        # Keep best score
        # ----------------------------------------------------

        if round_score is not None:

            if (
                best_score is None
                or round_score > best_score
            ):
                best_score = round_score

        print(f"\n{'='*80}")
        print(
            f"APEX ROUND {apex_round} COMPLETE"
        )
        print(f"Round Score : {round_score}")
        print(f"Best Score  : {best_score}")
        print(f"{'='*80}\n")

        # ----------------------------------------------------
        # Build APEX history
        # ----------------------------------------------------

        if (
    optimized_comparator is not None
    and apex_round < args.adaptive_rounds
):

            scored_examples = _score_dca_examples(
                evaluation_set,
                optimized_comparator,
            )

            for item in scored_examples:

                example = item["example"]
                score = item["score"]

                passed = score >= 1.0

                apex_controller.record_result(
                    example=example,
                    score=score,
                    passed=passed,
                )

        apex_controller.print_status()

        apex_history.append({
            "round": apex_round,
            "train_size": len(train_set),
            "round_score": round_score,
            "best_score": best_score,
        })

elif args.gepa_sampling == "ipomp":
    # ========================================================
    # IPOMP GEPA OPTIMIZATION
    # ========================================================
    #
    # Stage 1 has already selected:
    #
    #   train_set
    #   dev_set
    #
    # from the original 254-example training pool.
    #
    # Now GEPA generates multiple candidate prompts.
    #
    # Their per-example Dev performance is recorded.
    #
    # Then IPOMP Stage 2:
    #
    #   performance matrix
    #        ↓
    #   correlation
    #        ↓
    #   redundancy
    #        ↓
    #   contrasting replacement
    #
    # Finally GEPA is run again on the refined Dev set.
    #
    # Validation and test remain untouched.
    # ========================================================

    if args.optimizer != "gepa":
        raise ValueError(
            "IPOMP sampling is currently supported "
            "only with GEPA."
        )

    print(f"\n{'=' * 80}")
    print("STARTING IPOMP GEPA OPTIMIZATION")
    print(f"{'=' * 80}\n")

    optimized_comparator = comparator
    best_score = None
    runner = None

    # --------------------------------------------------------
    # IPOMP PERFORMANCE HISTORY
    # --------------------------------------------------------

    ipomp_history = []

    print(
        f"IPOMP candidate-prompt rounds : "
        f"{args.ipomp_rounds}"
    )

    print(
        f"Fixed GEPA train size         : "
        f"{len(train_set)}"
    )

    print(
        f"Initial GEPA dev size         : "
        f"{len(dev_set)}"
    )

    print(
        f"Unused replacement pool       : "
        f"{len(candidate_train_pool)}"
    )

    # --------------------------------------------------------
    # Candidate Prompt Generation
    # --------------------------------------------------------
    #
    # Each GEPA round produces a candidate optimized program.
    #
    # We evaluate that candidate separately on every current
    # Dev example and store the score in:
    #
    #     ipomp_selector.prompt_scores
    #
    # This produces the Stage 2 performance matrix.
    # --------------------------------------------------------

    for ipomp_round in range(
        1,
        args.ipomp_rounds + 1,
    ):

        print(f"\n{'#' * 80}")
        print(
            f"IPOMP CANDIDATE ROUND "
            f"{ipomp_round}/{args.ipomp_rounds}"
        )
        print(f"{'#' * 80}\n")

        # ----------------------------------------------------
        # Build a fresh Evaluate object for the CURRENT Dev.
        # ----------------------------------------------------

        round_evaluate = dspy.Evaluate(
            devset=dev_set,
            metric=dca_comparison_metric,
            num_threads=args.eval_num_threads,
            display_progress=True,
            display_table=True,
        )

        # ----------------------------------------------------
        # Create runner using the CURRENT optimized comparator.
        # ----------------------------------------------------

        runner = OptimizationRunner(
            metric=dca_comparison_metric,
            student_model=optimized_comparator,
            train_set=train_set,
            dev_set=dev_set,
            val_set=val_set,
            evaluate_fn=round_evaluate,
            metric_with_feedback=_metric_with_feedback,
            reflection_lm=_gepa_reflection_lm,
            apex_controller=None,
        )

        # ----------------------------------------------------
        # Run GEPA
        # ----------------------------------------------------

        try:

            round_comparator, round_score = (
                runner.run_optimization(
                    custom_strategies
                )
            )

        except Exception as exc:

            print(
                f"\n[ERROR] IPOMP candidate round "
                f"{ipomp_round} failed: {exc}"
            )

            raise

        # ----------------------------------------------------
        # Keep optimized comparator
        # ----------------------------------------------------

        if round_comparator is not None:
            optimized_comparator = (
                round_comparator
            )

        # ----------------------------------------------------
        # Keep best score
        # ----------------------------------------------------

        if round_score is not None:

            if (
                best_score is None
                or round_score > best_score
            ):
                best_score = round_score

        print(
            f"\nIPOMP round score : "
            f"{round_score}"
        )

        print(
            f"IPOMP best score  : "
            f"{best_score}"
        )

        # ----------------------------------------------------
        # SCORE EVERY DEV EXAMPLE WITH THIS CANDIDATE PROMPT
        # ----------------------------------------------------

        print("\n" + "-" * 80)
        print(
            f"IPOMP PERFORMANCE SCORING - "
            f"PROMPT {ipomp_round}"
        )
        print("-" * 80)

        scored_dev_examples = _score_dca_examples(
            dev_set,
            optimized_comparator,
        )

        # ----------------------------------------------------
        # Store this prompt's per-example performance.
        # ----------------------------------------------------

        ipomp_selector.record_prompt_scores(
            prompt_id=f"gepa_round_{ipomp_round}",
            scored_examples=scored_dev_examples,
        )

        # ----------------------------------------------------
        # Calculate summary statistics
        # ----------------------------------------------------

        prompt_scores = [
            float(item["score"])
            for item in scored_dev_examples
        ]

        prompt_passes = [
            score >= 1.0
            for score in prompt_scores
        ]

        prompt_accuracy = (
            sum(prompt_passes)
            / len(prompt_passes)
            if prompt_passes
            else 0.0
        )

        print(
            f"\nPrompt {ipomp_round} Dev results:"
        )

        print(
            f"  Examples evaluated : "
            f"{len(scored_dev_examples)}"
        )

        print(
            f"  Pass rate          : "
            f"{prompt_accuracy * 100:.2f}%"
        )

        print(
            f"  Mean DCA score     : "
            f"{sum(prompt_scores) / len(prompt_scores):.4f}"
            if prompt_scores
            else
            "  Mean DCA score     : N/A"
        )

        ipomp_history.append(
            {
                "round": ipomp_round,
                "round_score": round_score,
                "best_score": best_score,
                "dev_size": len(dev_set),
                "prompt_pass_rate": prompt_accuracy,
                "mean_prompt_score": (
                    sum(prompt_scores)
                    / len(prompt_scores)
                    if prompt_scores
                    else 0.0
                ),
            }
        )

    # ========================================================
    # IPOMP STAGE 2
    # ========================================================

    print(f"\n{'=' * 80}")
    print("STARTING IPOMP STAGE 2")
    print(f"{'=' * 80}\n")

    print(
        "The GEPA candidate prompts have now produced "
        "the performance matrix."
    )

    print(
        f"Prompt columns : "
        f"{len(ipomp_selector.prompt_scores)}"
    )

    print(
        f"Dev examples   : "
        f"{len(dev_set)}"
    )

    # --------------------------------------------------------
    # Refine Dev using model-performance redundancy
    # --------------------------------------------------------

    refined_dev_set, refinement_info = (
        ipomp_selector.run_stage2_refinement(
            evaluation_examples=dev_set,
            candidate_pool=candidate_train_pool,
        )
    )

    # --------------------------------------------------------
    # Keep Dev size fixed
    # --------------------------------------------------------

    if len(refined_dev_set) != len(dev_set):

        raise RuntimeError(
            "IPOMP Stage 2 changed the Dev set size. "
            f"Before={len(dev_set)}, "
            f"After={len(refined_dev_set)}"
        )

    # --------------------------------------------------------
    # Check no Train/Dev overlap after replacement
    # --------------------------------------------------------

    refined_dev_ids = {
        getattr(example, "pair_name", None)
        for example in refined_dev_set
    }

    train_ids = {
        getattr(example, "pair_name", None)
        for example in train_set
    }

    overlap = (
        refined_dev_ids
        & train_ids
    )

    if overlap:

        raise RuntimeError(
            "IPOMP Stage 2 created "
            f"Train/Dev overlap: {overlap}"
        )

    # --------------------------------------------------------
    # Replace active Dev set
    # --------------------------------------------------------

    dev_set = refined_dev_set

    print("\n" + "=" * 80)
    print("IPOMP STAGE 2 RESULT")
    print("=" * 80)

    print(
        f"Original Dev size : "
        f"{len(refined_dev_set)}"
    )

    if refinement_info is None:

        print(
            "No Dev replacement was performed."
        )

    else:

        print(
            f"Removed          : "
            f"{refinement_info['removed']}"
        )

        print(
            f"Replacement      : "
            f"{refinement_info['replacement']}"
        )

        print(
            f"Correlation      : "
            f"{refinement_info['correlation']:.4f}"
        )

    print(
        f"Final Dev size    : "
        f"{len(dev_set)}"
    )

    print(
        f"Validation        : "
        f"{len(val_set)} [UNTOUCHED]"
    )

    print(
        f"Test              : "
        f"{len(test_set)} [UNTOUCHED]"
    )

    print("=" * 80)

    # ========================================================
    # FINAL GEPA RUN ON REFINED DEV
    # ========================================================

    print(f"\n{'=' * 80}")
    print("FINAL GEPA RUN USING IPOMP-REFINED DEV SET")
    print(f"{'=' * 80}\n")

    final_evaluate = dspy.Evaluate(
        devset=dev_set,
        metric=dca_comparison_metric,
        num_threads=args.eval_num_threads,
        display_progress=True,
        display_table=True,
    )

    runner = OptimizationRunner(
        metric=dca_comparison_metric,
        student_model=optimized_comparator,
        train_set=train_set,
        dev_set=dev_set,
        val_set=val_set,
        evaluate_fn=final_evaluate,
        metric_with_feedback=_metric_with_feedback,
        reflection_lm=_gepa_reflection_lm,
        apex_controller=None,
    )

    try:

        final_comparator, final_score = (
            runner.run_optimization(
                custom_strategies
            )
        )

    except Exception as exc:

        print(
            f"\n[ERROR] Final IPOMP GEPA run failed: "
            f"{exc}"
        )

        raise

    if final_comparator is not None:

        optimized_comparator = (
            final_comparator
        )

    if final_score is not None:

        if (
            best_score is None
            or final_score > best_score
        ):
            best_score = final_score

    print("\n" + "=" * 80)
    print("IPOMP COMPLETE")
    print("=" * 80)

    print(
        f"GEPA Train size : "
        f"{len(train_set)}"
    )

    print(
        f"Final Dev size  : "
        f"{len(dev_set)}"
    )

    print(
        f"Validation size : "
        f"{len(val_set)} [UNTOUCHED]"
    )

    print(
        f"Test size       : "
        f"{len(test_set)} [UNTOUCHED]"
    )

    print(
        f"Best GEPA score : "
        f"{best_score}"
    )

    print("=" * 80 + "\n")    

elif args.gepa_sampling == "ipomp_difficulty":

    # ========================================================
    # IPOMP-DCA DIFFICULTY-AWARE GEPA OPTIMIZATION
    # ========================================================
    #
    # Stage 1 has already selected:
    #
    #   train_set
    #   dev_set
    #
    # from the original training pool.
    #
    # GEPA generates multiple candidate prompts.
    #
    # Their per-example Dev performance is recorded.
    #
    # Stage 2:
    #
    #   performance matrix
    #          ↓
    #      correlation
    #          ↓
    #       redundancy
    #          ↓
    #   contrasting replacement
    #          ↓
    #   CLIP diversity + DCA difficulty
    #
    # The only difference from normal IPOMP is that the
    # replacement example is selected using BOTH:
    #
    #   1. CLIP distance
    #   2. DCA difficulty
    #
    # Validation and test remain untouched.
    # ========================================================

    if args.optimizer != "gepa":
        raise ValueError(
            "IPOMP difficulty-aware sampling is currently "
            "supported only with GEPA."
        )

    print(f"\n{'=' * 80}")
    print("STARTING IPOMP-DIFFICULTY GEPA OPTIMIZATION")
    print(f"{'=' * 80}\n")

    optimized_comparator = comparator
    best_score = None
    runner = None

    # --------------------------------------------------------
    # IPOMP DIFFICULTY PERFORMANCE HISTORY
    # --------------------------------------------------------

    ipomp_difficulty_history = []

    print(
        f"IPOMP-Difficulty candidate-prompt rounds : "
        f"{args.ipomp_rounds}"
    )

    print(
        f"Fixed GEPA train size                    : "
        f"{len(train_set)}"
    )

    print(
        f"Initial GEPA dev size                    : "
        f"{len(dev_set)}"
    )

    print(
        f"Unused replacement pool                  : "
        f"{len(candidate_train_pool)}"
    )

    # ========================================================
    # CANDIDATE PROMPT GENERATION
    # ========================================================

    for ipomp_round in range(
        1,
        args.ipomp_rounds + 1,
    ):

        print(f"\n{'#' * 80}")
        print(
            f"IPOMP-DIFFICULTY CANDIDATE ROUND "
            f"{ipomp_round}/{args.ipomp_rounds}"
        )
        print(f"{'#' * 80}\n")

        # ----------------------------------------------------
        # Build a fresh Evaluate object for CURRENT Dev
        # ----------------------------------------------------

        round_evaluate = dspy.Evaluate(
            devset=dev_set,
            metric=dca_comparison_metric,
            num_threads=args.eval_num_threads,
            display_progress=True,
            display_table=True,
        )

        # ----------------------------------------------------
        # Create runner using CURRENT optimized comparator
        # ----------------------------------------------------

        runner = OptimizationRunner(
            metric=dca_comparison_metric,
            student_model=optimized_comparator,
            train_set=train_set,
            dev_set=dev_set,
            val_set=val_set,
            evaluate_fn=round_evaluate,
            metric_with_feedback=_metric_with_feedback,
            reflection_lm=_gepa_reflection_lm,
            apex_controller=None,
        )

        # ----------------------------------------------------
        # Run GEPA
        # ----------------------------------------------------

        try:

            round_comparator, round_score = (
                runner.run_optimization(
                    custom_strategies
                )
            )

        except Exception as exc:

            print(
                f"\n[ERROR] IPOMP-Difficulty candidate "
                f"round {ipomp_round} failed: {exc}"
            )

            raise

        # ----------------------------------------------------
        # Keep optimized comparator
        # ----------------------------------------------------

        if round_comparator is not None:
            optimized_comparator = (
                round_comparator
            )

        # ----------------------------------------------------
        # Keep best score
        # ----------------------------------------------------

        if round_score is not None:

            if (
                best_score is None
                or round_score > best_score
            ):
                best_score = round_score

        print(
            f"\nIPOMP-Difficulty round score : "
            f"{round_score}"
        )

        print(
            f"IPOMP-Difficulty best score  : "
            f"{best_score}"
        )

        # ====================================================
        # SCORE EVERY DEV EXAMPLE
        # ====================================================

        print("\n" + "-" * 80)
        print(
            f"IPOMP-DIFFICULTY PERFORMANCE SCORING - "
            f"PROMPT {ipomp_round}"
        )
        print("-" * 80)

        scored_dev_examples = _score_dca_examples(
            dev_set,
            optimized_comparator,
        )

        # ----------------------------------------------------
        # Store prompt's per-example performance
        # ----------------------------------------------------

        ipomp_difficulty_selector.record_prompt_scores(
            prompt_id=f"gepa_round_{ipomp_round}",
            scored_examples=scored_dev_examples,
        )

        # ----------------------------------------------------
        # Calculate summary statistics
        # ----------------------------------------------------

        prompt_scores = [
            float(item["score"])
            for item in scored_dev_examples
        ]

        prompt_passes = [
            score >= 1.0
            for score in prompt_scores
        ]

        prompt_accuracy = (
            sum(prompt_passes)
            / len(prompt_passes)
            if prompt_passes
            else 0.0
        )

        print(
            f"\nPrompt {ipomp_round} Dev results:"
        )

        print(
            f"  Examples evaluated : "
            f"{len(scored_dev_examples)}"
        )

        print(
            f"  Pass rate          : "
            f"{prompt_accuracy * 100:.2f}%"
        )

        print(
            f"  Mean DCA score     : "
            f"{sum(prompt_scores) / len(prompt_scores):.4f}"
            if prompt_scores
            else
            "  Mean DCA score     : N/A"
        )

        ipomp_difficulty_history.append(
            {
                "round": ipomp_round,
                "round_score": round_score,
                "best_score": best_score,
                "dev_size": len(dev_set),
                "prompt_pass_rate": prompt_accuracy,
                "mean_prompt_score": (
                    sum(prompt_scores)
                    / len(prompt_scores)
                    if prompt_scores
                    else 0.0
                ),
            }
        )

    # ========================================================
    # IPOMP-DIFFICULTY STAGE 2
    # ========================================================

    print(f"\n{'=' * 80}")
    print("STARTING IPOMP-DIFFICULTY STAGE 2")
    print(f"{'=' * 80}\n")

    print(
        "The GEPA candidate prompts have now produced "
        "the performance matrix."
    )

    print(
        f"Prompt columns : "
        f"{len(ipomp_difficulty_selector.prompt_scores)}"
    )

    print(
        f"Dev examples   : "
        f"{len(dev_set)}"
    )

    # --------------------------------------------------------
    # Refine Dev using model-performance redundancy
    #
    # The NEW selector performs replacement using:
    #
    #   normalized CLIP distance
    #          +
    #       DCA difficulty
    # --------------------------------------------------------

    refined_dev_set, refinement_info = (
        ipomp_difficulty_selector.run_stage2_refinement(
            evaluation_examples=dev_set,
            candidate_pool=candidate_train_pool,
        )
    )

    # --------------------------------------------------------
    # Keep Dev size fixed
    # --------------------------------------------------------

    if len(refined_dev_set) != len(dev_set):

        raise RuntimeError(
            "IPOMP-Difficulty Stage 2 changed the Dev set size. "
            f"Before={len(dev_set)}, "
            f"After={len(refined_dev_set)}"
        )

    # --------------------------------------------------------
    # Check no Train/Dev overlap
    # --------------------------------------------------------

    refined_dev_ids = {
        getattr(example, "pair_name", None)
        for example in refined_dev_set
    }

    train_ids = {
        getattr(example, "pair_name", None)
        for example in train_set
    }

    overlap = (
        refined_dev_ids
        & train_ids
    )

    if overlap:

        raise RuntimeError(
            "IPOMP-Difficulty Stage 2 created "
            f"Train/Dev overlap: {overlap}"
        )

    # --------------------------------------------------------
    # Replace active Dev set
    # --------------------------------------------------------

    dev_set = refined_dev_set

    print("\n" + "=" * 80)
    print("IPOMP-DIFFICULTY STAGE 2 RESULT")
    print("=" * 80)

    print(
        f"Original Dev size : "
        f"{len(refined_dev_set)}"
    )

    if refinement_info is None:

        print(
            "No Dev replacement was performed."
        )

    else:

        print(
            f"Removed          : "
            f"{refinement_info['removed']}"
        )

        print(
            f"Replacement      : "
            f"{refinement_info['replacement']}"
        )

        print(
            f"Correlation      : "
            f"{refinement_info['correlation']:.4f}"
        )

        # ----------------------------------------------------
        # Additional information from the new
        # difficulty-aware replacement logic
        # ----------------------------------------------------

        if "clip_distance" in refinement_info:
            print(
                f"CLIP distance    : "
                f"{refinement_info['clip_distance']:.4f}"
            )

        if "normalized_diversity" in refinement_info:
            print(
                f"Normalized CLIP  : "
                f"{refinement_info['normalized_diversity']:.4f}"
            )

        if "dca_difficulty" in refinement_info:
            print(
                f"DCA difficulty   : "
                f"{refinement_info['dca_difficulty']:.4f}"
            )

        if "replacement_score" in refinement_info:
            print(
                f"Combined score    : "
                f"{refinement_info['replacement_score']:.4f}"
            )

    print(
        f"Final Dev size    : "
        f"{len(dev_set)}"
    )

    print(
        f"Validation        : "
        f"{len(val_set)} [UNTOUCHED]"
    )

    print(
        f"Test              : "
        f"{len(test_set)} [UNTOUCHED]"
    )

    print("=" * 80)

    # ========================================================
    # FINAL GEPA RUN ON REFINED DEV
    # ========================================================

    print(f"\n{'=' * 80}")
    print(
        "FINAL GEPA RUN USING "
        "IPOMP-DIFFICULTY-REFINED DEV SET"
    )
    print(f"{'=' * 80}\n")

    final_evaluate = dspy.Evaluate(
        devset=dev_set,
        metric=dca_comparison_metric,
        num_threads=args.eval_num_threads,
        display_progress=True,
        display_table=True,
    )

    runner = OptimizationRunner(
        metric=dca_comparison_metric,
        student_model=optimized_comparator,
        train_set=train_set,
        dev_set=dev_set,
        val_set=val_set,
        evaluate_fn=final_evaluate,
        metric_with_feedback=_metric_with_feedback,
        reflection_lm=_gepa_reflection_lm,
        apex_controller=None,
    )

    try:

        final_comparator, final_score = (
            runner.run_optimization(
                custom_strategies
            )
        )

    except Exception as exc:

        print(
            f"\n[ERROR] Final IPOMP-Difficulty "
            f"GEPA run failed: {exc}"
        )

        raise

    # --------------------------------------------------------
    # Keep final optimized comparator
    # --------------------------------------------------------

    if final_comparator is not None:

        optimized_comparator = (
            final_comparator
        )

    # --------------------------------------------------------
    # Keep best score
    # --------------------------------------------------------

    if final_score is not None:

        if (
            best_score is None
            or final_score > best_score
        ):
            best_score = final_score

    print("\n" + "=" * 80)
    print("IPOMP-DIFFICULTY COMPLETE")
    print("=" * 80)

    print(
        f"GEPA Train size : "
        f"{len(train_set)}"
    )

    print(
        f"Final Dev size  : "
        f"{len(dev_set)}"
    )

    print(
        f"Validation size : "
        f"{len(val_set)} [UNTOUCHED]"
    )

    print(
        f"Test size       : "
        f"{len(test_set)} [UNTOUCHED]"
    )

    print(
        f"Best GEPA score : "
        f"{best_score}"
    )

    print("=" * 80 + "\n")    
elif args.gepa_sampling == "performance_train":
    

    # ========================================================
    # PERFORMANCE-BASED DYNAMIC GEPA OPTIMIZATION
    # ========================================================
    #
    # Workflow:
    #
    #   Round 1
    #       ↓
    #   Select training examples
    #       ↓
    #   GEPA optimization
    #       ↓
    #   Score ALL candidate training examples
    #       ↓
    #   Update performance history
    #       ↓
    #   Round 2 selects a NEW training set
    #       ↓
    #   GEPA optimization
    #       ↓
    #   Repeat
    #
    # IMPORTANT:
    #
    #   Dev set       = FIXED
    #   Validation    = UNTOUCHED
    #   Test          = UNTOUCHED
    #
    # Only the GEPA training set changes dynamically.
    # ========================================================

    if args.optimizer != "gepa":
        raise ValueError(
            "Performance sampling is currently supported only with GEPA."
        )

    print(f"\n{'=' * 80}")
    print("STARTING PERFORMANCE-BASED DYNAMIC GEPA OPTIMIZATION")
    print(f"{'=' * 80}\n")

    print(
        f"Candidate training pool : "
        f"{len(candidate_train_pool)}"
    )

    print(
        f"Training size            : "
        f"{len(train_set)}"
    )

    print(
        f"Fixed Dev size           : "
        f"{len(dev_set)}"
    )

    print(
        f"Validation size          : "
        f"{len(val_set)} [UNTOUCHED]"
    )

    print(
        f"Test size                : "
        f"{len(test_set)} [UNTOUCHED]"
    )

    print(
        f"Performance rounds       : "
        f"{args.performance_rounds}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Initial comparator
    # --------------------------------------------------------

    optimized_comparator = comparator

    best_score = None
    runner = None

    performance_history = []

    # --------------------------------------------------------
    # Dynamic performance rounds
    # --------------------------------------------------------

    for performance_round in range(
        1,
        args.performance_rounds + 1
    ):

        print(f"\n{'#' * 80}")
        print(
            f"PERFORMANCE ROUND "
            f"{performance_round}/{args.performance_rounds}"
        )
        print(f"{'#' * 80}\n")

        print(
            f"Training examples this round : "
            f"{len(train_set)}"
        )

        print(
            f"Fixed Dev examples           : "
            f"{len(dev_set)}"
        )

        # ----------------------------------------------------
        # Create GEPA runner using CURRENT training set
        # ----------------------------------------------------

        runner = OptimizationRunner(
            metric=dca_comparison_metric,
            student_model=optimized_comparator,
            train_set=train_set,
            dev_set=dev_set,
            val_set=val_set,
            evaluate_fn=evaluate,
            metric_with_feedback=_metric_with_feedback,
            reflection_lm=_gepa_reflection_lm,
            apex_controller=apex_controller,
        )

        round_comparator = None
        round_score = None

        # ----------------------------------------------------
        # Run GEPA
        # ----------------------------------------------------

        try:

            round_comparator, round_score = (
                runner.run_optimization(
                    custom_strategies
                )
            )

        except Exception as exc:

            print(
                "\nPerformance-based GEPA round failed."
            )

            print(
                f"Round : {performance_round}"
            )

            print(
                f"Error : {exc}"
            )

            raise

        # ----------------------------------------------------
        # Update optimized comparator
        # ----------------------------------------------------

        if round_comparator is not None:

            optimized_comparator = round_comparator

        if round_score is not None:

            best_score = round_score

        # ----------------------------------------------------
        # Print round result
        # ----------------------------------------------------

        print(f"\n{'=' * 80}")
        print(
            f"PERFORMANCE ROUND "
            f"{performance_round} COMPLETE"
        )
        print(f"{'=' * 80}")

        print(
            f"Round Score : {round_score}"
        )

        print(
            f"Best Score  : {best_score}"
        )

        print(
            f"Train Size  : {len(train_set)}"
        )

        print(
            f"Dev Size    : {len(dev_set)} [FIXED]"
        )

        print(f"{'=' * 80}\n")

        # ----------------------------------------------------
        # Final round:
        # No need to select another training set.
        # ----------------------------------------------------

        if performance_round == args.performance_rounds:

            print(
                "Final performance round completed."
            )

            break

        # ====================================================
        # SCORE ENTIRE CANDIDATE TRAINING POOL
        # ====================================================
        #
        # IMPORTANT:
        #
        # We score ALL candidate training examples,
        # not just the examples used in this GEPA round.
        #
        # Example:
        #
        #   Candidate pool = 224
        #   Current train  = 50
        #
        # We score:
        #
        #   224 examples
        #
        # This lets the next round choose from the entire
        # candidate pool.
        # ====================================================

        print(f"\n{'=' * 80}")
        print(
            "SCORING ENTIRE CANDIDATE TRAINING POOL"
        )
        print(f"{'=' * 80}")

        print(
            f"Candidate pool size : "
            f"{len(candidate_train_pool)}"
        )

        print(
            "Current comparator  : "
            f"{type(optimized_comparator).__name__}"
        )

        scored_candidate_pool = _score_dca_examples(
            candidate_train_pool,
            optimized_comparator,
        )

        print(
            f"Successfully scored : "
            f"{len(scored_candidate_pool)}"
        )

        print(f"{'=' * 80}\n")

        # ====================================================
        # RECORD PERFORMANCE VECTOR
        # ====================================================
        #
        # Each GEPA round becomes one performance column.
        #
        # Example:
        #
        #             Round1 Round2
        # pair_1        0.0    1.0
        # pair_2        1.0    1.0
        # pair_3        0.4    0.8
        #
        # After multiple rounds, examples have performance
        # vectors that can be compared for redundancy.
        # ====================================================

        prompt_id = (
            f"gepa_round_{performance_round}"
        )

        performance_sampler.record_prompt_scores(
            prompt_id=prompt_id,
            scored_examples=scored_candidate_pool,
        )

        # ----------------------------------------------------
        # Print performance matrix
        # ----------------------------------------------------

        performance_sampler.print_performance_matrix()

        # ====================================================
        # SELECT NEXT TRAINING SET
        # ====================================================
        #
        # The sampler now uses the accumulated performance
        # history to select the next training examples.
        #
        # This is where the training set changes dynamically.
        # ====================================================

        next_train_set, selection_info = (
            performance_sampler.select_training_examples()
        )

        train_set = next_train_set

        # ----------------------------------------------------
        # Store round information
        # ----------------------------------------------------

        selection_info["round"] = (
            performance_round + 1
        )

        performance_history.append(
            selection_info
        )

        # ----------------------------------------------------
        # Verify training size
        # ----------------------------------------------------

        if len(train_set) != args.gepa_train_size:

            raise RuntimeError(
                "Performance sampler produced the wrong "
                "training-set size. "
                f"Expected {args.gepa_train_size}, "
                f"got {len(train_set)}."
            )

        # ----------------------------------------------------
        # Verify Train / Dev separation
        # ----------------------------------------------------

        train_ids = {
            getattr(example, "pair_name", None)
            for example in train_set
        }

        dev_ids = {
            getattr(example, "pair_name", None)
            for example in dev_set
        }

        overlap = train_ids & dev_ids

        if overlap:

            raise RuntimeError(
                "Performance sampling created "
                f"Train/Dev overlap: {overlap}"
            )

        # ----------------------------------------------------
        # Log next training set
        # ----------------------------------------------------

        print(f"\n{'=' * 80}")
        print(
            "NEXT PERFORMANCE-BASED TRAINING SET"
        )
        print(f"{'=' * 80}")

        print(
            f"Round                    : "
            f"{performance_round + 1}"
        )

        print(
            f"Training samples         : "
            f"{len(train_set)}"
        )

        print(
            f"Fixed Dev samples        : "
            f"{len(dev_set)}"
        )

        print(
            f"Candidate pool           : "
            f"{len(candidate_train_pool)}"
        )

        print(
            f"Validation               : "
            f"{len(val_set)} [UNTOUCHED]"
        )

        print(
            f"Test                     : "
            f"{len(test_set)} [UNTOUCHED]"
        )

        print("\nSelected examples:")

        for rank, example in enumerate(
            train_set,
            start=1
        ):

            pair_name = getattr(
                example,
                "pair_name",
                f"example_{rank}"
            )

            print(
                f"{rank:3d}. {pair_name}"
            )

        print(f"{'=' * 80}\n")

    # --------------------------------------------------------
    # Final performance sampling summary
    # --------------------------------------------------------

    print(f"\n{'=' * 80}")
    print(
        "PERFORMANCE-BASED GEPA OPTIMIZATION COMPLETE"
    )
    print(f"{'=' * 80}")

    print(
        f"Total performance rounds : "
        f"{args.performance_rounds}"
    )

    print(
        f"Final training size      : "
        f"{len(train_set)}"
    )

    print(
        f"Fixed Dev size           : "
        f"{len(dev_set)}"
    )

    print(
        f"Validation size          : "
        f"{len(val_set)} [UNTOUCHED]"
    )

    print(
        f"Test size                : "
        f"{len(test_set)} [UNTOUCHED]"
    )

    print(
        f"Final score              : "
        f"{best_score}"
    )

    print(f"{'=' * 80}\n")

elif args.gepa_sampling == "performance_train_sess_lc":

    # ============================================================
    # PERFORMANCE-BASED TRAIN + SESS-LC FIXED DEV
    # ============================================================
    #
    # Workflow:
    #
    #   Original Train pool
    #          |
    #          v
    #      SESS-LC
    #          |
    #          +----> Fixed Dev
    #
    #   Remaining pool
    #          |
    #          v
    #   Initial random Train
    #          |
    #          v
    #        GEPA
    #          |
    #          v
    #   Candidate prompt performance
    #          |
    #          v
    #   Performance matrix
    #          |
    #          v
    #   Correlation / redundancy
    #          |
    #          v
    #   CLIP contrasting replacement
    #          |
    #          v
    #   New Train
    #          |
    #          v
    #        GEPA
    #
    # IMPORTANT:
    #   - Dev is selected only once.
    #   - Dev never changes after SESS-LC selection.
    #   - Train size remains fixed.
    #   - Validation is untouched.
    #   - Test is untouched.
    #   - Train/Dev overlap is prevented.
    # ============================================================

    if args.optimizer != "gepa":
        raise ValueError(
            "Performance Train + SESS-LC sampling "
            "is currently supported only with GEPA."
        )

    # ------------------------------------------------------------
    # STEP 1: SESS-LC SELECTS DEV
    # ------------------------------------------------------------
    #
    # SESS-LC:
    #   - scores the original training pool
    #   - calculates confidence/difficulty
    #   - selects the low-confidence examples as Dev
    #   - removes Dev from the candidate Train pool
    #
    # Validation and Test are NOT involved.
    # ------------------------------------------------------------

    dev_set, candidate_train_pool = (
        select_sess_lc_dev_and_get_remaining(
            examples=train_pool,
            dev_size=args.gepa_dev_size,
            comparator=comparator,
            metric=dca_comparison_metric,
        )
    )

    print("\n" + "=" * 80)
    print("PERFORMANCE TRAIN + SESS-LC DEV SAMPLING")
    print("=" * 80)

    print(
        f"Original training pool : "
        f"{len(train_pool)}"
    )

    print(
        f"Fixed SESS-LC Dev      : "
        f"{len(dev_set)}"
    )

    print(
        f"Candidate train pool   : "
        f"{len(candidate_train_pool)}"
    )

    print(
        f"Initial Train size     : "
        f"{args.gepa_train_size}"
    )

    print(
        f"Performance rounds     : "
        f"{args.performance_rounds}"
    )

    print(
        f"Correlation threshold  : "
        f"{args.performance_correlation_threshold}"
    )

    print(
        "Train refinement       : "
        "Performance redundancy + CLIP contrast"
    )

    print(
        "Dev strategy           : "
        "SESS-LC fixed"
    )

    print(
        f"Validation             : "
        f"{len(val_set)} [UNTOUCHED]"
    )

    print(
        f"Test                   : "
        f"{len(test_set)} [UNTOUCHED]"
    )

    print("=" * 80)

    # ------------------------------------------------------------
    # STEP 2: CHECK TRAIN POOL SIZE
    # ------------------------------------------------------------

    if len(candidate_train_pool) < args.gepa_train_size:

        raise ValueError(
            f"Candidate training pool "
            f"({len(candidate_train_pool)}) is smaller "
            f"than requested training size "
            f"({args.gepa_train_size})."
        )

    # ------------------------------------------------------------
    # STEP 3: CREATE PERFORMANCE REFINEMENT SELECTOR
    # ------------------------------------------------------------
    #
    # This is the NEW selector.
    #
    # It does NOT select Dev.
    #
    # Dev has already been fixed by SESS-LC.
    #
    # It stores GEPA candidate-prompt performance and later
    # performs:
    #
    #   performance correlation
    #          +
    #   redundancy detection
    #          +
    #   CLIP contrasting replacement
    #
    # to refine Train.
    # ------------------------------------------------------------

    performance_sampler = PerformanceTrainSessLCSampler(
        examples=candidate_train_pool,
        train_size=args.gepa_train_size,
        correlation_threshold=(
            args.performance_correlation_threshold
        ),
        seed=args.performance_seed,
    )

    # ------------------------------------------------------------
    # STEP 4: INITIAL TRAIN SELECTION
    # ------------------------------------------------------------
    #
    # The new selector performs refinement only after GEPA
    # candidate-prompt performance has been collected.
    #
    # Therefore we need an initial Train set.
    #
    # We use a seeded random selection so the experiment is
    # reproducible.
    # ------------------------------------------------------------
    import random
    rng = random.Random(
        args.performance_seed
    )

    initial_pool = list(
        candidate_train_pool
    )

    rng.shuffle(
        initial_pool
    )

    train_set = initial_pool[
        :args.gepa_train_size
    ]

    # ------------------------------------------------------------
    # STEP 5: INITIAL SELECTION INFORMATION
    # ------------------------------------------------------------

    selection_info = {
        "method":
            "performance_train_sess_lc",

        "round":
            0,

        "train_size":
            len(train_set),

        "dev_size":
            len(dev_set),

        "candidate_pool_size":
            len(candidate_train_pool),

        "initial_selection":
            True,
    }

    print("\n" + "=" * 80)
    print("INITIAL TRAIN SELECTION")
    print("=" * 80)

    print(
        f"Train : "
        f"{len(train_set)}"
    )

    print(
        f"Dev   : "
        f"{len(dev_set)}"
    )

    print(
        f"Pool  : "
        f"{len(candidate_train_pool)}"
    )

    print(
        "Selection : "
        "Seeded random initialization"
    )

    print("=" * 80)

    # ------------------------------------------------------------
    # STEP 6: SAFETY CHECK
    # ------------------------------------------------------------

    train_ids = {
        getattr(
            example,
            "pair_name",
            None,
        )
        for example in train_set
    }

    dev_ids = {
        getattr(
            example,
            "pair_name",
            None,
        )
        for example in dev_set
    }

    overlap = (
        train_ids
        & dev_ids
    )

    if overlap:

        raise RuntimeError(
            "Initial Performance Train + SESS-LC "
            f"created Train/Dev overlap: {overlap}"
        )

    if len(train_set) != args.gepa_train_size:

        raise RuntimeError(
            "Initial performance Train set has "
            "the wrong size. "
            f"Expected={args.gepa_train_size}, "
            f"got={len(train_set)}"
        )

    # ============================================================
    # PERFORMANCE ROUNDS
    # ============================================================
    #
    # IMPORTANT:
    #
    # The SESS-LC Dev remains fixed throughout these rounds.
    #
    # Only Train can change.
    # ============================================================

    optimized_comparator = comparator

    best_score = None

    runner = None

    performance_history = []

    for performance_round in range(
        1,
        args.performance_rounds + 1
    ):

        print(
            f"\n{'#' * 80}"
        )

        print(
            f"PERFORMANCE TRAIN + "
            f"SESS-LC ROUND "
            f"{performance_round}/"
            f"{args.performance_rounds}"
        )

        print(
            f"{'#' * 80}\n"
        )

        print(
            f"Current Train size : "
            f"{len(train_set)}"
        )

        print(
            f"Fixed Dev size     : "
            f"{len(dev_set)}"
        )

        print(
            f"Candidate pool     : "
            f"{len(candidate_train_pool)}"
        )

        # --------------------------------------------------------
        # CREATE GEPA EVALUATOR
        # --------------------------------------------------------

        round_evaluate = dspy.Evaluate(
            devset=dev_set,
            metric=dca_comparison_metric,
            num_threads=args.eval_num_threads,
            display_progress=True,
            display_table=True,
        )

        # --------------------------------------------------------
        # CREATE GEPA RUNNER
        # --------------------------------------------------------

        runner = OptimizationRunner(
            metric=dca_comparison_metric,
            student_model=optimized_comparator,
            train_set=train_set,
            dev_set=dev_set,
            val_set=val_set,
            evaluate_fn=round_evaluate,
            metric_with_feedback=_metric_with_feedback,
            reflection_lm=_gepa_reflection_lm,
            apex_controller=None,
        )

        round_comparator = None

        round_score = None

        # --------------------------------------------------------
        # RUN GEPA
        # --------------------------------------------------------

        try:

            round_comparator, round_score = (
                runner.run_optimization(
                    custom_strategies
                )
            )

        except Exception as exc:

            print(
                "\nPerformance Train + "
                "SESS-LC GEPA round failed."
            )

            print(
                f"Round : "
                f"{performance_round}"
            )

            print(
                f"Error : "
                f"{exc}"
            )

            raise

        # --------------------------------------------------------
        # UPDATE OPTIMIZED COMPARATOR
        # --------------------------------------------------------

        if round_comparator is not None:

            optimized_comparator = (
                round_comparator
            )

        # --------------------------------------------------------
        # UPDATE BEST SCORE
        # --------------------------------------------------------

        if round_score is not None:

            if (
                best_score is None
                or round_score > best_score
            ):

                best_score = round_score

        print(
            f"\n{'=' * 80}"
        )

        print(
            "GEPA ROUND COMPLETE"
        )

        print(
            f"Round score : "
            f"{round_score}"
        )

        print(
            f"Best score  : "
            f"{best_score}"
        )

        print(
            f"{'=' * 80}\n"
        )

        # ========================================================
        # SCORE CANDIDATE TRAINING POOL
        # ========================================================
        #
        # The NEW Train selector needs performance of every
        # candidate example for this GEPA prompt.
        #
        # The candidate pool is:
        #
        #     candidate_train_pool
        #
        # It excludes the fixed SESS-LC Dev set.
        #
        # Validation/Test are not scored here.
        # ========================================================

        if (
            optimized_comparator is not None
        ):

            scored_candidate_pool = (
                _score_dca_examples(
                    candidate_train_pool,
                    optimized_comparator,
                )
            )

            # ----------------------------------------------------
            # Give this GEPA prompt a unique ID.
            # ----------------------------------------------------

            prompt_id = (
                f"performance_train_sess_lc_"
                f"round_{performance_round}"
            )

            # ----------------------------------------------------
            # Record performance matrix.
            # ----------------------------------------------------

            performance_sampler.record_prompt_scores(
                prompt_id=prompt_id,
                scored_examples=scored_candidate_pool,
            )

            print(
                "\nPerformance matrix updated:"
            )

            print(
                f"  Prompt ID : "
                f"{prompt_id}"
            )

            print(
                f"  Examples  : "
                f"{len(scored_candidate_pool)}"
            )

        # ========================================================
        # TRAIN REFINEMENT
        # ========================================================
        #
        # Refinement requires at least TWO prompt-performance
        # vectors.
        #
        # Therefore:
        #
        #   Round 1:
        #       collect performance
        #       no refinement yet
        #
        #   Round 2:
        #       performance matrix has >= 2 prompts
        #       find redundant Train examples
        #       remove one
        #       replace with CLIP-contrasting candidate
        #
        #   Round 3:
        #       repeat the process
        #
        # Dev remains fixed.
        # ========================================================

        if (
            len(
                list(performance_sampler.prompt_order)
            ) >= 2
        ):

            print(
                "\n" + "-" * 80
            )

            print(
                "REFINING TRAIN SET"
            )

            print(
                "-" * 80
            )

            train_set, refinement_info = (
    performance_sampler.select_training_examples()
)

            # ----------------------------------------------------
            # Replace ONLY Train.
            #
            # Dev is intentionally NOT changed.
            # ----------------------------------------------------

            # train_set is already updated by select_training_examples()

            print(
                "\nTrain refinement complete."
            )

            print(
                f"Train size : "
                f"{len(train_set)}"
            )

            if refinement_info:

                print(
                    f"Refinement : "
                    f"{refinement_info}"
                )

        else:

            print(
                "\nTrain refinement skipped."
            )

            print(
                "Reason: fewer than two "
                "GEPA performance vectors."
            )

        # ========================================================
        # SAFETY CHECKS AFTER REFINEMENT
        # ========================================================

        # --------------------------------------------------------
        # Train size
        # --------------------------------------------------------

        if (
            len(train_set)
            != args.gepa_train_size
        ):

            raise RuntimeError(
                "Performance Train + SESS-LC "
                "produced the wrong training-set size. "
                f"Expected "
                f"{args.gepa_train_size}, "
                f"got {len(train_set)}."
            )

        # --------------------------------------------------------
        # Dev size
        # --------------------------------------------------------

        if (
            len(dev_set)
            != args.gepa_dev_size
        ):

            raise RuntimeError(
                "SESS-LC Dev set changed size. "
                f"Expected "
                f"{args.gepa_dev_size}, "
                f"got {len(dev_set)}."
            )

        # --------------------------------------------------------
        # Train / Dev overlap
        # --------------------------------------------------------

        train_ids = {
            getattr(
                example,
                "pair_name",
                None,
            )
            for example in train_set
        }

        dev_ids = {
            getattr(
                example,
                "pair_name",
                None,
            )
            for example in dev_set
        }

        overlap = (
            train_ids
            & dev_ids
        )

        if overlap:

            raise RuntimeError(
                "Performance Train + SESS-LC "
                f"created Train/Dev overlap: {overlap}"
            )

        # --------------------------------------------------------
        # Validation / Test safety
        # --------------------------------------------------------

        val_ids = {
            getattr(
                example,
                "pair_name",
                None,
            )
            for example in val_set
        }

        test_ids = {
            getattr(
                example,
                "pair_name",
                None,
            )
            for example in test_set
        }

        train_val_overlap = (
            train_ids
            & val_ids
        )

        train_test_overlap = (
            train_ids
            & test_ids
        )

        dev_val_overlap = (
            dev_ids
            & val_ids
        )

        dev_test_overlap = (
            dev_ids
            & test_ids
        )

        if train_val_overlap:

            raise RuntimeError(
                "Train overlaps with Validation: "
                f"{train_val_overlap}"
            )

        if train_test_overlap:

            raise RuntimeError(
                "Train overlaps with Test: "
                f"{train_test_overlap}"
            )

        if dev_val_overlap:

            raise RuntimeError(
                "Dev overlaps with Validation: "
                f"{dev_val_overlap}"
            )

        if dev_test_overlap:

            raise RuntimeError(
                "Dev overlaps with Test: "
                f"{dev_test_overlap}"
            )

        # --------------------------------------------------------
        # Store performance history
        # --------------------------------------------------------

        performance_history.append(
            {
                "round":
                    performance_round,

                "train_size":
                    len(train_set),

                "dev_size":
                    len(dev_set),

                "round_score":
                    round_score,

                "best_score":
                    best_score,

                "prompt_count":
                    list(performance_sampler.prompt_order),

                "train_refined":
                    (
                        len(
                            performance_sampler.prompt_order
                        ) >= 2
                    ),
            }
        )

        # --------------------------------------------------------
        # Print current Train set
        # --------------------------------------------------------

        print(
            f"\n{'=' * 80}"
        )

        print(
            "CURRENT TRAIN SET"
        )

        print(
            f"{'=' * 80}"
        )

        print(
            f"Round : "
            f"{performance_round}"
        )

        print(
            f"Train : "
            f"{len(train_set)}"
        )

        print(
            f"Fixed Dev : "
            f"{len(dev_set)}"
        )

        print(
            f"Val : "
            f"{len(val_set)} [UNTOUCHED]"
        )

        print(
            f"Test : "
            f"{len(test_set)} [UNTOUCHED]"
        )

        print(
            "\nSelected Train examples:"
        )

        for rank, example in enumerate(
            train_set,
            start=1,
        ):

            pair_name = getattr(
                example,
                "pair_name",
                f"example_{rank}",
            )

            print(
                f"{rank:3d}. "
                f"{pair_name}"
            )

        print(
            f"{'=' * 80}\n"
        )

    # ============================================================
    # FINAL PERFORMANCE SUMMARY
    # ============================================================

    print(
        f"\n{'=' * 80}"
    )

    print(
        "PERFORMANCE TRAIN + "
        "SESS-LC OPTIMIZATION COMPLETE"
    )

    print(
        f"{'=' * 80}"
    )

    print(
        f"Total performance rounds : "
        f"{args.performance_rounds}"
    )

    print(
        f"Final training size      : "
        f"{len(train_set)}"
    )

    print(
        f"Fixed SESS-LC Dev size   : "
        f"{len(dev_set)}"
    )

    print(
        f"Validation size          : "
        f"{len(val_set)} [UNTOUCHED]"
    )

    print(
        f"Test size                : "
        f"{len(test_set)} [UNTOUCHED]"
    )

    print(
    f"Prompt order : "
    f"{list(performance_sampler.prompt_order)}"
)

    print(
        f"Final score              : "
        f"{best_score}"
    )

    print(
        f"{'=' * 80}\n"
    )
elif args.gepa_sampling == "performance_train_dev":

    # ========================================================
    # PERFORMANCE-BASED DYNAMIC TRAIN + DEV GEPA OPTIMIZATION
    # ========================================================
    #
    # Workflow:
    #
    #   Initial Train + Dev
    #          ↓
    #   GEPA optimization
    #          ↓
    #   Score candidate Train examples
    #   Score candidate Dev examples
    #          ↓
    #   Build Train performance matrix
    #   Build Dev performance matrix
    #          ↓
    #   Performance-based selection
    #          ↓
    #   NEW Train + NEW Dev
    #          ↓
    #   Next GEPA round
    #
    # Important:
    #
    #   Train = dynamically selected
    #   Dev   = dynamically selected
    #   Val   = untouched
    #   Test  = untouched
    #
    # Train and Dev must never overlap.
    # ========================================================

    if args.optimizer != "gepa":
        raise ValueError(
            "Performance Train+Dev sampling is currently "
            "supported only with GEPA."
        )

    print(f"\n{'=' * 80}")
    print(
        "STARTING PERFORMANCE-BASED "
        "DYNAMIC TRAIN + DEV GEPA OPTIMIZATION"
    )
    print(f"{'=' * 80}\n")

    print(
        f"Original training pool : "
        f"{len(train_pool)}"
    )

    print(
        f"Initial Train size     : "
        f"{len(train_set)}"
    )

    print(
        f"Initial Dev size       : "
        f"{len(dev_set)}"
    )

    print(
        f"Performance rounds     : "
        f"{args.performance_rounds}"
    )

    print(
        f"Correlation threshold  : "
        f"{args.performance_correlation_threshold}"
    )

    print(
        f"Validation size        : "
        f"{len(val_set)} [UNTOUCHED]"
    )

    print(
        f"Test size              : "
        f"{len(test_set)} [UNTOUCHED]"
    )

    print(f"{'=' * 80}\n")

    # --------------------------------------------------------
    # Initial comparator
    # --------------------------------------------------------

    optimized_comparator = comparator

    best_score = None
    runner = None

    performance_history = []

    # ========================================================
    # PERFORMANCE ROUNDS
    # ========================================================

    for performance_round in range(
        1,
        args.performance_rounds + 1
    ):

        print(f"\n{'#' * 80}")
        print(
            f"PERFORMANCE TRAIN + DEV ROUND "
            f"{performance_round}/"
            f"{args.performance_rounds}"
        )
        print(f"{'#' * 80}\n")

        print(
            f"Current Train size : "
            f"{len(train_set)}"
        )

        print(
            f"Current Dev size   : "
            f"{len(dev_set)}"
        )

        # ----------------------------------------------------
        # Create GEPA Evaluate object using CURRENT Dev
        # ----------------------------------------------------

        round_evaluate = dspy.Evaluate(
            devset=dev_set,
            metric=dca_comparison_metric,
            num_threads=args.eval_num_threads,
            display_progress=True,
            display_table=True,
        )

        # ----------------------------------------------------
        # Create GEPA runner using CURRENT Train + Dev
        # ----------------------------------------------------

        runner = OptimizationRunner(
            metric=dca_comparison_metric,
            student_model=optimized_comparator,
            train_set=train_set,
            dev_set=dev_set,
            val_set=val_set,
            evaluate_fn=round_evaluate,
            metric_with_feedback=_metric_with_feedback,
            reflection_lm=_gepa_reflection_lm,
            apex_controller=None,
        )

        round_comparator = None
        round_score = None

        # ----------------------------------------------------
        # Run GEPA
        # ----------------------------------------------------

        try:

            round_comparator, round_score = (
                runner.run_optimization(
                    custom_strategies
                )
            )

        except Exception as exc:

            print(
                "\nPerformance Train+Dev GEPA "
                "round failed."
            )

            print(
                f"Round : "
                f"{performance_round}"
            )

            print(
                f"Error : "
                f"{exc}"
            )

            raise

        # ----------------------------------------------------
        # Update optimized comparator
        # ----------------------------------------------------

        if round_comparator is not None:

            optimized_comparator = (
                round_comparator
            )

        # ----------------------------------------------------
        # Update best score
        # ----------------------------------------------------

        if round_score is not None:

            if (
                best_score is None
                or round_score > best_score
            ):
                best_score = round_score

        # ----------------------------------------------------
        # Print GEPA round result
        # ----------------------------------------------------

        print(f"\n{'=' * 80}")
        print(
            f"PERFORMANCE TRAIN + DEV ROUND "
            f"{performance_round} COMPLETE"
        )
        print(f"{'=' * 80}")

        print(
            f"Round Score : "
            f"{round_score}"
        )

        print(
            f"Best Score  : "
            f"{best_score}"
        )

        print(
            f"Train Size  : "
            f"{len(train_set)}"
        )

        print(
            f"Dev Size    : "
            f"{len(dev_set)}"
        )

        print(f"{'=' * 80}\n")

        # ----------------------------------------------------
        # Final round
        #
        # No need to select another Train/Dev set.
        # ----------------------------------------------------

        if (
            performance_round
            == args.performance_rounds
        ):

            print(
                "Final performance Train+Dev "
                "round completed."
            )

            break

        # ====================================================
        # SCORE CANDIDATE POOL
        # ====================================================
        #
        # The important difference from performance_train:
        #
        # We now collect performance information for
        # BOTH Train and Dev.
        #
        # The selector is responsible for applying the
        # same performance-based selection logic to both.
        # ====================================================

        print(f"\n{'=' * 80}")
        print(
            "SCORING CANDIDATE EXAMPLES "
            "FOR TRAIN + DEV"
        )
        print(f"{'=' * 80}\n")

        # ----------------------------------------------------
        # Score the complete candidate pool.
        #
        # The candidate pool contains examples that are
        # available for future Train/Dev selection.
        # ----------------------------------------------------

        scored_candidate_pool = (
            _score_dca_examples(
                performance_train_dev_candidate_pool,
                optimized_comparator,
            )
        )

        print(
            f"Candidate pool size : "
            f"{len(performance_train_dev_candidate_pool)}"
        )

        print(
            f"Successfully scored : "
            f"{len(scored_candidate_pool)}"
        )

        # ====================================================
        # RECORD PERFORMANCE
        # ====================================================
        #
        # One GEPA round becomes one performance column.
        #
        # Example:
        #
        #             Round1 Round2 Round3
        # example_1     0.0    1.0    1.0
        # example_2     1.0    1.0    0.6
        # example_3     0.4    0.0    0.8
        #
        # The same performance history is used for
        # selecting both Train and Dev.
        # ====================================================

        prompt_id = (
            f"gepa_round_{performance_round}"
        )

        performance_train_dev_selector.record_prompt_scores(
            prompt_id=prompt_id,
            scored_examples=scored_candidate_pool,
        )

        # ----------------------------------------------------
        # Print performance matrix
        # ----------------------------------------------------

        performance_train_dev_selector.print_performance_matrix()

        # ====================================================
        # SELECT NEXT TRAIN + DEV
        # ====================================================
        #
        # This is the important part.
        #
        # The selector performs performance-based selection
        # for BOTH sets.
        #
        # Train:
        #     performance-based selection
        #
        # Dev:
        #     performance-based selection
        #
        # The selector must guarantee:
        #
        #     len(Train) == args.gepa_train_size
        #     len(Dev)   == args.gepa_dev_size
        #     Train ∩ Dev = empty
        # ====================================================

        (
    next_train_set,
    next_dev_set,
    selection_info,
) = (
    performance_train_dev_selector.refine_train_dev(
        train_set=train_set,
        dev_set=dev_set,
        train_candidate_pool=(
            performance_train_dev_candidate_pool
        ),
        dev_candidate_pool=(
            performance_train_dev_candidate_pool
        ),
        prevent_overlap=True,
    )
)
        # ----------------------------------------------------
        # Replace current Train + Dev
        # ----------------------------------------------------

        train_set = next_train_set
        dev_set = next_dev_set

        # ----------------------------------------------------
        # Store round information
        # ----------------------------------------------------

        selection_info["round"] = (
            performance_round + 1
        )

        performance_history.append(
            selection_info
        )

        # ====================================================
        # SAFETY CHECKS
        # ====================================================

        # ----------------------------------------------------
        # Train size
        # ----------------------------------------------------

        if (
            len(train_set)
            != args.gepa_train_size
        ):

            raise RuntimeError(
                "Performance Train+Dev sampler "
                "produced the wrong training-set size. "
                f"Expected "
                f"{args.gepa_train_size}, "
                f"got {len(train_set)}."
            )

        # ----------------------------------------------------
        # Dev size
        # ----------------------------------------------------

        if (
            len(dev_set)
            != args.gepa_dev_size
        ):

            raise RuntimeError(
                "Performance Train+Dev sampler "
                "produced the wrong development-set size. "
                f"Expected "
                f"{args.gepa_dev_size}, "
                f"got {len(dev_set)}."
            )

        # ----------------------------------------------------
        # Train / Dev overlap check
        # ----------------------------------------------------

        train_ids = {
            getattr(
                example,
                "pair_name",
                None,
            )
            for example in train_set
        }

        dev_ids = {
            getattr(
                example,
                "pair_name",
                None,
            )
            for example in dev_set
        }

        overlap = (
            train_ids
            & dev_ids
        )

        if overlap:

            raise RuntimeError(
                "Performance Train+Dev sampling "
                "created Train/Dev overlap: "
                f"{overlap}"
            )

        # ====================================================
        # VALIDATION / TEST SAFETY
        # ====================================================

        val_ids = {
            getattr(
                example,
                "pair_name",
                None,
            )
            for example in val_set
        }

        test_ids = {
            getattr(
                example,
                "pair_name",
                None,
            )
            for example in test_set
        }

        train_val_overlap = (
            train_ids
            & val_ids
        )

        train_test_overlap = (
            train_ids
            & test_ids
        )

        dev_val_overlap = (
            dev_ids
            & val_ids
        )

        dev_test_overlap = (
            dev_ids
            & test_ids
        )

        if train_val_overlap:
            raise RuntimeError(
                "Train/Validation overlap detected: "
                f"{train_val_overlap}"
            )

        if train_test_overlap:
            raise RuntimeError(
                "Train/Test overlap detected: "
                f"{train_test_overlap}"
            )

        if dev_val_overlap:
            raise RuntimeError(
                "Dev/Validation overlap detected: "
                f"{dev_val_overlap}"
            )

        if dev_test_overlap:
            raise RuntimeError(
                "Dev/Test overlap detected: "
                f"{dev_test_overlap}"
            )

        # ====================================================
        # PRINT NEXT SET
        # ====================================================

        print(f"\n{'=' * 80}")
        print(
            "NEXT PERFORMANCE-BASED "
            "TRAIN + DEV SET"
        )
        print(f"{'=' * 80}")

        print(
            f"Next Train size : "
            f"{len(train_set)}"
        )

        print(
            f"Next Dev size   : "
            f"{len(dev_set)}"
        )

        print(
            "Train/Dev overlap : "
            f"{len(overlap)}"
        )

        print(
            f"Validation size : "
            f"{len(val_set)} [UNTOUCHED]"
        )

        print(
            f"Test size       : "
            f"{len(test_set)} [UNTOUCHED]"
        )

        print(f"{'=' * 80}\n")

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print(f"\n{'=' * 80}")
    print(
        "PERFORMANCE TRAIN + DEV "
        "OPTIMIZATION COMPLETE"
    )
    print(f"{'=' * 80}")

    print(
        f"Final Train size : "
        f"{len(train_set)}"
    )

    print(
        f"Final Dev size   : "
        f"{len(dev_set)}"
    )

    print(
        f"Validation size  : "
        f"{len(val_set)} [UNTOUCHED]"
    )

    print(
        f"Test size        : "
        f"{len(test_set)} [UNTOUCHED]"
    )

    print(
        f"Best GEPA score  : "
        f"{best_score}"
    )

    print(f"{'=' * 80}\n")
# ============================================================
# FALLBACK
# ============================================================

if optimized_comparator is None:

    print(
        "No strategy beat the baseline -- "
        "using unoptimized comparator."
    )

    optimized_comparator = comparator


print(f"\n{'='*80}")
print(f"Optimization Results Summary")
print(f"{'='*80}")

if best_score is not None:
    print(
        f"Best Strategy Val Score: {best_score:.1f}%"
    )
else:
    print(
        "Best Strategy Val Score: N/A"
    )

print(f"{'='*80}\n")

# ── Print final optimized instruction so you can compare with the seed ─────
_opt_instruction = getattr(optimized_comparator, "signature", None)
if _opt_instruction is not None:
    _opt_instruction = _opt_instruction.instructions
print(f"{'='*80}")
print("[OPTIMIZED INSTRUCTION] — final instruction produced by the optimizer:")
print(f"{'='*80}")
print(_opt_instruction or "<could not retrieve>")
print(f"{'='*80}\n")

# ── Save optimized state (DSPy's own dump_state — same format as .save()) ────
_opt_prompt_path = os.path.join(
    experiment_dir,
    "optimized_state.json"
)
with open(_opt_prompt_path, "w", encoding="utf-8") as _fp:
    json.dump(optimized_comparator.dump_state(), _fp, indent=2, ensure_ascii=False)
print(f"Optimized state saved -> {_opt_prompt_path}\n")


# ── Save Model ───────────────────────────────────────────────────
# File naming: {model}_{strategy}_{cot}[_RE{effort}]_dca_optimized_{YYYYMMDDHHmmss}.json
# The CoT/NoCot and RE suffix ensure runs with different --no-cot / --reasoning-effort
# flags never overwrite each other's saved files.
model_name = (args.model_name or os.getenv("MODEL_NAME", "Unknown_Model")).replace("/", "_")
best_strategy_name = getattr(runner, "best_strategy_name", "Baseline")
best_strategy_slug = best_strategy_name.replace(" ", "_")
best_strategy_config = strategy_by_name.get(best_strategy_name)
cot_tag = "NoCot" if args.no_cot else "CoT"
re_tag  = f"_RE{args.reasoning_effort}" if args.reasoning_effort else ""

model_path = None
os.makedirs(cache_dir, exist_ok=True)
demo_tag = _format_demo_tag(
    getattr(best_strategy_config, "max_labeled_demos", None),
    getattr(best_strategy_config, "max_bootstrapped_demos", None),
)
instr_tag = ("_" + os.path.splitext(os.path.basename(args.initial_instruction_file))[0]
             if args.initial_instruction_file else "")
cfd_tag = "_cfd" if args.clear_field_descs else ""
model_path = os.path.join(
    experiment_dir,
    "optimized_model.json"
)
optimized_comparator.save(model_path)
print(f"\nSaved optimized model to: {model_path}")


# ── Final Test Evaluation ───────────────────────────────────────
print(f"\n{'='*80}")
print("FINAL TEST EVALUATION")
print(f"{'='*80}")

test_accuracy = None

try:
    test_accuracy = evaluate(
        optimized_comparator,
        test_set,
    )

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Accuracy (%): {test_accuracy * 100:.2f}%")

except Exception as e:
    print(f"Could not evaluate test set: {e}")


print(f"\n{'='*80}")
print(f"DCA Pipeline complete!")
print(f"{'='*80}\n")

experiment_summary = {
    "experiment_directory": str(experiment_dir),
    "timestamp": current_time,

    "optimizer": args.optimizer,
    "sampling": args.gepa_sampling,

    "model": args.model_name,
    "gepa_auto": args.gepa_auto,

    "train_size": len(train_set),
    "requested_train_size": args.gepa_train_size,

    "dev_size": len(dev_set),
    "requested_dev_size": args.gepa_dev_size,

    "validation_size": len(val_set),
    "test_size": len(test_set),

    "split_seed": split_seed,

    "validation_accuracy": best_score,
    "test_accuracy": test_accuracy,

    "optimized_model": model_path,
    "optimized_state": _opt_prompt_path,
    "log_file": _stdout_log_path,
    "split_manifest": split_latest_path,
}

summary_path = experiment_dir / "experiment_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(
        experiment_summary,
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"Experiment summary saved -> {summary_path}")
# ── Close TeeLogger and rename log to match saved model ──────────
# For runs where a model was saved, rename the log file to share the same
# base name as the model JSON (easy to find the log for a given model file).
sys.stdout = _tee_stdout._original

sys.stderr = _tee_stderr._original

_log_file.close()

if model_path is not None:

    model_stem = os.path.splitext(
        os.path.basename(model_path)
    )[0]

    final_run_summary = {

        "run_type": "extract_dca",

        "started_at": pipeline_started_at,

        "completed_at": datetime.now().isoformat(
            timespec="seconds"
        ),

        "total_duration_seconds": round(
            time.time() - pipeline_t0,
            1
        ),

        "model_name": args.model_name or os.getenv(
            "MODEL_NAME",
            "Unknown_Model"
        ),

        "optimizer_requested": args.optimizer,

        "best_strategy_name": best_strategy_name,

        "best_score": best_score,
        
        "test_accuracy": test_accuracy,

        "pairs_dir": os.path.abspath(pairs_dir),

        "gt_dir": os.path.abspath(gt_dir),

        "split_seed": split_seed,

        "split_manifest_path": split_latest_path,

        "saved_model_path": model_path,

        "seed_state_path": _seed_prompt_path,

        "optimized_state_path": _opt_prompt_path,

        "runner_summary": getattr(
            runner,
            "run_summary",
            {}
        ),

        # -----------------------------------------------------
        # ADAPTIVE SAMPLING INFORMATION
        # -----------------------------------------------------
        "adaptive_sampling": {

            "enabled": (
                args.gepa_sampling == "adaptive"
            ),

            "rounds": (
                args.adaptive_rounds
                if args.gepa_sampling == "adaptive"
                else 0
            ),

            "train_size": gepa_train_size,

            "dev_size": gepa_dev_size,

            "exploration_fraction": (
                args.adaptive_exploration
            ),

            "adaptive_seed": args.adaptive_seed,

            "candidate_pool_size": (
                len(candidate_train_pool)
            ),

            "history": adaptive_history,
        },
    }
    final_log_path = os.path.join(_log_dir, f"{model_stem}.log")
    final_seed_state_path = os.path.join(_log_dir, f"{model_stem}_seed_state.json")
    final_optimized_state_path = os.path.join(_log_dir, f"{model_stem}_optimized_state.json")
    final_run_summary_log_path = os.path.join(_log_dir, f"{model_stem}_run_summary.json")
    final_run_summary_cache_path = os.path.join(cache_dir, f"{model_stem}_run_summary.json")
    try:
        os.rename(_stdout_log_path, final_log_path)
        print(f"Log renamed to: {final_log_path}", file=sys.__stdout__)
    except OSError as e:
        print(f"Could not rename log file: {e}", file=sys.__stdout__)
    try:
        os.replace(_seed_prompt_path, final_seed_state_path)
        print(f"Seed state renamed to: {final_seed_state_path}", file=sys.__stdout__)
    except OSError as e:
        print(f"Could not rename seed state file: {e}", file=sys.__stdout__)
    try:
        os.replace(_opt_prompt_path, final_optimized_state_path)
        print(f"Optimized state renamed to: {final_optimized_state_path}", file=sys.__stdout__)
    except OSError as e:
        print(f"Could not rename optimized state file: {e}", file=sys.__stdout__)
    final_run_summary["log_path"] = final_log_path
    final_run_summary["seed_state_path"] = final_seed_state_path
    final_run_summary["optimized_state_path"] = final_optimized_state_path
    for _summary_path in (final_run_summary_log_path, final_run_summary_cache_path):
        try:
            with open(_summary_path, "w", encoding="utf-8") as _fp:
                json.dump(final_run_summary, _fp, indent=2, ensure_ascii=False)
            print(f"Run summary saved to: {_summary_path}", file=sys.__stdout__)
        except OSError as e:
            print(f"Could not save run summary file {_summary_path}: {e}", file=sys.__stdout__)
