"""
extract_dca.py — Single-run DCA Prompt Optimization
====================================================

PIPELINE OVERVIEW (what this script does end-to-end)
-----------------------------------------------------
  1. Parse CLI args → read env vars
  2. Set up TeeLogger (all output → terminal + log file simultaneously)
  3. Load dataset (~376 labelled image pairs from disk)
  4. Stratified train/val/test split (60/20/20, seed=42)
  5. Build the DSPy comparator module (Predict or ChainOfThought)
  6. Optionally load a previously saved extractor (warm-start optimization)
  7. Optionally inject a human-written instruction from a .md/.txt file
  8. Save the "seed state" (prompt before optimization) to log/
  9. Run the selected optimizer(s) via OptimizationRunner
  10. Save the "optimized state" + final model .json to .cache/
  11. Rename log + state files to share the model's base name for traceability

TASK DESCRIPTION (what the comparator does)
-------------------------------------------
Given two document images (original vs. re-imaged version), classify:
  discard_image = "true"   → images are identical / differences are trivial
  discard_image = "false"  → meaningful changes exist

And produce a change summary.  Special-case rules:
  A1: single non-informational image → discard=true, changes="Only one image and it is non informational"
  A2: single informational image     → discard=false, changes="NoMatchFound"
  B3: both non-informational         → discard=true,  changes="Both images are non informational..."
  B5a: trivially identical           → discard=true,  changes="Images are identical - no changes"
  B5b: meaningfully different        → discard=false,  changes=<detailed description>
  Special: crossed-out               → discard=false, changes="Image has been crossed out"
  Special: corrupted/unreadable      → discard=false, changes="Could not infer"

OPTIMIZERS (select via --optimizer)
-----------------------------------
  bootstrap   BootstrapFewShotWithRandomSearch  — demo selection
  miprov2     MIPROv2  — Bayesian instruction + demo search (recommended)
  gepa        GEPA     — reflection-LM guided prompt improvement
  simba       SimBA    — iterative prompt refinement on train set only
  all         Run all four in sequence

DATASET
-------
  ~376 human-labelled image pairs: data/dca_data/image_pairs/
  Ground truth JSONs:               data/dca_data/image_pairs_llm_output_exp1/
  Split (seed=42, stratified by discard_image + category + doubt):
    train 60%  |  val 20%  |  test 20%

USAGE EXAMPLES
--------------
    # Quick run with default MIPROv2 medium
    $env:PYTHONIOENCODING="utf-8"; uv run python extract_dca.py

    # Heavy MIPROv2 optimization
    uv run python extract_dca.py --optimizer miprov2 --miprov2-auto heavy

    # Run all four optimizers
    uv run python extract_dca.py --optimizer all

    # Use a different model
    uv run python extract_dca.py --model-name azure/gpt-4o --optimizer gepa

    # Start from a human-written prompt file
    uv run python extract_dca.py --initial-instruction-file human_prompts/v1_prompt_midas.md

OUTPUT FILES (all in .cache/ and log/)
---------------------------------------
  {prefix}_seed_state.json       — DSPy state before optimization
  {prefix}_optimized_state.json  — DSPy state after optimization
  {prefix}.json                  — Final saved model (same as optimized_state, renamed)
  {prefix}.log                   — Full console output for this run
  {prefix}_run_summary.json      — Timing, scores, paths, best strategy

ENVIRONMENT VARIABLES (.env, overridden by CLI flags)
-----------------------------------------------------
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


def _quote_cli_arg(value: str) -> str:
    """Shell-quote a CLI argument value so it is safe to paste into a terminal.

    Example: _quote_cli_arg("hello world") -> '"hello world"'
    """
    text = str(value)
    if not text:
        return '""'
    if any(ch in text for ch in ' \t"'):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def _build_rerun_from_seed_command(seed_state_path: str) -> str:
    """Build the CLI command needed to re-run this exact job from the saved seed state.

    Useful when optimization fails mid-run: the seed (pre-optimization) state
    is already on disk and this command lets you resume from that checkpoint
    rather than regenerating the seed from scratch.

    Strips ``--load-extractor`` and ``--skip-load`` from the original argv,
    then appends ``--load-extractor <seed_state_path>`` so the next run
    picks up from the saved seed prompt.
    """
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

args = parser.parse_args()

# Central flag — purely driven by the explicit switch
_should_clear_descs: bool = args.clear_field_descs

pairs_dir = args.pairs_dir
gt_dir = args.gt_dir
split_seed = 42
_split_ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
_split_stratify_by = ["expected_discard_image", "expected_human_label_category", "doubt"]


def _slug_token(value: str | None, default: str = "na") -> str:
    """Convert any string to a safe filesystem/filename token.

    Replaces non-alphanumeric chars with underscores and collapses doubles.
    Example: _slug_token("azure/gpt-4.1 mini") -> "azure_gpt_4_1_mini"
    """
    text = str(value or "").strip().lower()
    cleaned = [ch if ch.isalnum() else "_" for ch in text]
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or default


def _short_hash(*parts: object, length: int = 10) -> str:
    """Return a short deterministic hex hash of the given parts (JSON-serialised).

    Used to fingerprint split configurations so that re-running with the same
    dataset paths and ratios always maps to the same cached split manifest.
    """
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def _build_split_cache_path() -> str:
    """Return a fingerprinted path for the split manifest JSON.

    The path encodes the dataset name, ratios, and a content hash so that
    different dataset/split configurations never collide on disk.
    Example output: .cache/splits/dca_split_seed42_image_pairs_..._tr60va20te20_<hash>.json
    """
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


def _build_dataset_cache_tag() -> str:
    """Short tag that uniquely identifies the dataset (folder name + content hash).

    Embedded in model filenames so saved models are traceable to their dataset.
    """
    dataset_name = _slug_token(Path(os.path.abspath(pairs_dir)).name)[:18]
    return f"{dataset_name}_{_short_hash(os.path.abspath(pairs_dir), os.path.abspath(gt_dir))}"


def _format_demo_tag(max_labeled_demos, max_bootstrapped_demos) -> str:
    """Build a compact demo-count tag for filenames, e.g. 'd3b5' = 3 labeled, 5 bootstrapped."""
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
    """Find the most-recently modified extractor in cache_root matching all tags.

    Used for ``--auto-load-compatible-cache``: avoids re-running a slow
    optimization if a compatible cached model already exists on disk.
    Returns None when nothing matches.
    """
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

split_latest_path = _build_split_cache_path()
os.makedirs(os.path.dirname(split_latest_path), exist_ok=True)

# Stratified split: train 60%, val 20%, test 20%
# The split manifest path is fingerprinted by dataset path + split settings.
train_set, val_set, test_set = stratified_split_dca_dataset(
    dataset,
    train_ratio=_split_ratios["train"],
    val_ratio=_split_ratios["val"],
    test_ratio=_split_ratios["test"],
    random_seed=split_seed, verbose=True
)

val_names  = _pair_names(val_set)
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
    "val":  {"count": len(val_names),  "pair_names": val_names},
    "test": {"count": len(test_names), "pair_names": test_names},
}

with open(split_latest_path, "w", encoding="utf-8") as f:
    json.dump(split_manifest, f, indent=2)

print(f"Saved split manifest: {split_latest_path}")
print(f"  train: {len(train_set)}  |  val: {len(val_set)}  |  test: {len(test_set)}")


# ── Create Extractor ─────────────────────────────────────────────
# ChainOfThought: LM reasons step-by-step before answering (more accurate,
#                 more tokens).  Produced by dspy.ChainOfThought.
# Predict (--no-cot): LM answers directly (faster, cheaper).  Use this to
#                     test whether chain-of-thought reasoning is actually
#                     needed for the task.
#
# The comparator wraps the DCAImageComparison signature (defined in
# signatures/dca.py) which lists the input fields (images, metadata) and
# output fields (discard_image, changes).  optimizer.compile() mutates
# this object in-place; the result can be re-loaded with comparator.load().
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


# ── Evaluation Setup ─────────────────────────────────────────────
# dca_comparison_metric returns a weighted float in [0, 1]:
#   0.6 * classification_score + 0.4 * change_summary_score
# dspy.Evaluate wraps this and reports on a 0-100 scale in its table.
# num_threads=4 evaluates 4 examples concurrently to speed up scoring.
# devset=None here because the runner passes val_set at compile time.
evaluate = dspy.Evaluate(
    devset=None,
    metric=dca_comparison_metric,
    num_threads=4,
    display_progress=True,
    display_table=True,
)


# ── Optimization Strategies ──────────────────────────────────────
# Each strategy is a named config object consumed by OptimizationRunner.
# _register_strategy() adds it to both the ordered list (custom_strategies)
# and the by-name lookup dict (strategy_by_name) used when saving the model.
#
# Strategy selection logic:
#   --optimizer bootstrap   → only Bootstrap
#   --optimizer miprov2     → only MIPROv2
#   --optimizer gepa        → only GEPA
#   --optimizer simba       → only SimBA
#   --optimizer all         → all four run sequentially
#
# Each optimizer.compile() call starts from the SAME seed comparator and
# competes on the val set.  The one with the highest val score is saved.
custom_strategies: list = []
strategy_by_name: dict = {}

def _register_strategy(strategy) -> None:
    custom_strategies.append(strategy)
    strategy_by_name[strategy.name] = strategy

if args.optimizer in ("bootstrap", "all"):
    _register_strategy(
        create_bootstrap_strategy(
            name="Balanced_DCA_Bootstrap",
            description="Balanced approach with equal labeled and bootstrapped demos for DCA classification",
            max_labeled_demos=1,        # Image context overhead; each demo adds tokens
            max_bootstrapped_demos=1,   # Teacher-generated demos drawn from correct train predictions
            num_candidate_programs=1,  # Random candidates to evaluate; more = better coverage
            num_candidate_traces=(2,5), # Generate 2-5 candidate traces per example for bootstrapping
            num_threads=4,              # Parallel LLM calls during candidate evaluation
            max_rounds=1,               # Second round refines demo quality; >2 risks overfitting
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
            num_threads=4,
        )
    )

if args.optimizer in ("gepa", "all"):
    _register_strategy(
        create_gepa_strategy(
            name=f"GEPA_DCA_{args.gepa_auto.capitalize()}",
            description=f"GEPA ({args.gepa_auto}) -- reflection-LM guided prompt improvement",
            auto=args.gepa_auto,
            num_threads=4,
            track_stats=True,
            use_merge=True,
        )
    )

if args.optimizer in ("simba", "all"):
    _register_strategy(
        create_simba_strategy(
            name=f"SimBA_DCA_Steps{args.simba_max_steps}",
            description=f"SimBA ({args.simba_max_steps} steps) -- iterative prompt refinement",
            max_steps=args.simba_max_steps,
            max_labeled_demos=4,
            num_threads=4,
            bsize=32,
            num_candidates=6,
        )
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
optimized_comparator = None
best_score = None
try:
    optimized_comparator, best_score = runner.run_optimization(custom_strategies)
except Exception as exc:
    failed_at = datetime.now().isoformat(timespec="seconds")
    rerun_command = _build_rerun_from_seed_command(_seed_prompt_path)
    failure_summary = {
        "run_type": "extract_dca",
        "status": "failed",
        "started_at": pipeline_started_at,
        "failed_at": failed_at,
        "total_duration_seconds": round(time.time() - pipeline_t0, 1),
        "model_name": args.model_name or os.getenv("MODEL_NAME", "Unknown_Model"),
        "optimizer_requested": args.optimizer,
        "pairs_dir": os.path.abspath(pairs_dir),
        "gt_dir": os.path.abspath(gt_dir),
        "split_seed": split_seed,
        "split_manifest_path": split_latest_path,
        "seed_state_path": _seed_prompt_path,
        "log_path": _stdout_log_path,
        "error": str(exc),
        "runner_summary": getattr(runner, "run_summary", {}),
        "rerun_from_seed_command": rerun_command,
    }

    sys.stdout = _tee_stdout._original
    sys.stderr = _tee_stderr._original
    _log_file.close()

    failure_summary_log_path = os.path.join(_log_dir, f"{_log_prefix}_dca_run_{current_time}_error_summary.json")
    failure_summary_cache_path = os.path.join(cache_dir, f"{_log_prefix}_dca_run_{current_time}_error_summary.json")
    for _summary_path in (failure_summary_log_path, failure_summary_cache_path):
        try:
            with open(_summary_path, "w", encoding="utf-8") as _fp:
                json.dump(failure_summary, _fp, indent=2, ensure_ascii=False)
            print(f"Failure summary saved to: {_summary_path}", file=sys.__stdout__)
        except OSError as summary_exc:
            print(f"Could not save failure summary {_summary_path}: {summary_exc}", file=sys.__stdout__)

    print("\nOptimization aborted after an unrecoverable optimizer failure.", file=sys.__stdout__)
    print(f"Seed state is preserved at: {_seed_prompt_path}", file=sys.__stdout__)
    print("Rerun from the saved seed state with:", file=sys.__stdout__)
    print(f"  {rerun_command}", file=sys.__stdout__)
    raise SystemExit(1) from exc

if optimized_comparator is None:
    print("No strategy beat the baseline -- using unoptimized comparator.")
    optimized_comparator = comparator

print(f"\n{'='*80}")
print(f"Optimization Results Summary")
print(f"{'='*80}")
print(f"Best Strategy Val Score: {best_score:.1f}%")
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
_opt_prompt_path = os.path.join(_log_dir, f"{_log_prefix}_dca_run_{current_time}_optimized_state.json")
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
    cache_dir,
    f"{model_name}_{best_strategy_slug}_{cot_tag}{re_tag}_dca_optimized_{demo_tag}{instr_tag}{cfd_tag}_ds{_dataset_tag}_sp{split_seed}_{current_time}.json",
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
    final_run_summary = {
        "run_type": "extract_dca",
        "started_at": pipeline_started_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "total_duration_seconds": round(time.time() - pipeline_t0, 1),
        "model_name": args.model_name or os.getenv("MODEL_NAME", "Unknown_Model"),
        "optimizer_requested": args.optimizer,
        "best_strategy_name": best_strategy_name,
        "best_score": best_score,
        "pairs_dir": os.path.abspath(pairs_dir),
        "gt_dir": os.path.abspath(gt_dir),
        "split_seed": split_seed,
        "split_manifest_path": split_latest_path,
        "saved_model_path": model_path,
        "seed_state_path": _seed_prompt_path,
        "optimized_state_path": _opt_prompt_path,
        "runner_summary": getattr(runner, "run_summary", {}),
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
