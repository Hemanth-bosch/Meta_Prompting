"""Tournament utilities \u2014 logging, evaluation, save/load, category sampling.

Shared helpers used by every stage of the cascaded optimization pipeline.

MODULE CONTENTS
---------------
TeeLogger
    Context manager that duplicates stdout+stderr to a log file.  Every
    print() and logger call is captured without touching the calling code.

    Example::

        with TeeLogger(\"my_run.log\"):
            run_all_stages()   # stdout + stderr also written to my_run.log

evaluate_candidate(program, val_set, metric, \u2026)
    Run \`program\` on each example in \`val_set\` and collect per-sample token
    usage from \`lm.history\` (the LM\u2019s own API-reported token counts).
    Returns avg_total_score, per_sample_token_stats dict, and timing info.
    Token stats power the Pareto front analysis and hybrid metric penalty.

evaluate_candidate_at_temperature(program, \u2026, temperature)
    Same as evaluate_candidate but temporarily overrides the student LM
    temperature, used by seed_prompt_generator for diverse score sampling.

save_candidate(program, path) / load_candidate(path, \u2026)
    Thin wrappers around DSPy\u2019s save()/load() that also store/restore the
    custom instruction text injected into the signature.

get_program_info(program)
    Extract the instruction text and number of demos from a compiled program.
    Used to populate CandidateEntry.metadata.

create_mini_dataset(full_set, n, stratify_by, \u2026)
    Stratified random subsample of the dataset.  Preserves the category
    distribution of the full set so mini metrics stay meaningful.

    Example::

        mini_train = create_mini_dataset(train_set, n=32, stratify_by=\"category\")

generate_summary(leaderboard, config, \u2026)
    Build the final summary dict (best score, Pareto front, per-stage stats)
    written to \`tournament_summary.json\` and printed to console.

plot_pareto_candidates(entries, output_path, \u2026)
    Draw a scatter plot of score vs avg_tokens with the Pareto front
    highlighted.  Saved as \`pareto_plot.png\` in the tournament directory.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

import dspy

from config import create_lm
from signatures.dca import DCAImageComparison
from evaluation.dca_metric import (
    dca_comparison_metric,
    compute_classification_score,
)
from dca_tournaments.leaderboard import CandidateEntry, OptimizationLeaderboard


logger = logging.getLogger("dca_tournaments.utils")

# ── Categories ──────────────────────────────────────────────────
DCA_CATEGORIES = ["A1", "A2", "B1", "B2", "B3", "B4", "B5a", "B5b"]


def _ts() -> str:
    """Return a short timestamp for log lines."""
    return datetime.now().strftime("%H:%M:%S")


# ═══════════════════════════════════════════════════════════════════
# TeeLogger — duplicates stdout + stderr to a log file
# ═══════════════════════════════════════════════════════════════════

class _TeeStream:
    """Redirect a stream to both the terminal and a shared log file."""

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

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return self._original.isatty()


class TeeLogger:
    """Context-manager that tees stdout + stderr to a log file.

    Usage::

        with TeeLogger(log_dir, "stage_1_bootstrap") as tee:
            print("this goes to both terminal and log file")
        # streams restored automatically
    """

    def __init__(self, log_dir: str | Path, name: str):
        self.log_dir = Path(log_dir)
        self.name = name
        self.log_path = self.log_dir / f"{name}.log"
        self._file = None
        self._tee_stdout = None
        self._tee_stderr = None
        self._orig_stdout = None
        self._orig_stderr = None

    def __enter__(self) -> "TeeLogger":
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._file = open(self.log_path, "a", encoding="utf-8")
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._tee_stdout = _TeeStream(self._orig_stdout, self._file)
        self._tee_stderr = _TeeStream(self._orig_stderr, self._file)
        sys.stdout = self._tee_stdout
        sys.stderr = self._tee_stderr
        return self

    def __exit__(self, *exc):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        if self._file:
            self._file.close()
            self._file = None

    def teardown(self):
        """Manual teardown (for cases where context manager is inconvenient)."""
        if self._orig_stdout:
            sys.stdout = self._orig_stdout
        if self._orig_stderr:
            sys.stderr = self._orig_stderr
        if self._file:
            self._file.close()
            self._file = None


def setup_tournament_logging(log_dir: str | Path, name: str) -> tuple[TeeLogger, Path]:
    """Create and enter a `TeeLogger`, returning it and the resolved log path."""
    tee = TeeLogger(log_dir, name)
    tee.__enter__()
    return tee, tee.log_path


def teardown_logging(tee: TeeLogger | None):
    """Restore stdout/stderr and close a previously created tee logger."""
    if tee is not None:
        tee.teardown()


# ═══════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════

def _extract_token_stats_from_history(lm, history_before: int, n_samples: int) -> dict:
    """Extract per-sample average token stats from new LM history entries.

    Returns dict with avg_total_tokens, avg_input_tokens, avg_output_tokens,
    avg_reasoning_tokens (all float or None).
    """
    def _get(obj, key):
        if obj is None:
            return None
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    try:
        if not hasattr(lm, "history") or n_samples <= 0:
            return {}
        new_entries = lm.history[history_before:]
        if not new_entries:
            return {}
        total_t = input_t = output_t = reasoning_t = 0
        for entry in new_entries:
            usage = entry.get("usage") or _get(entry.get("response"), "usage")
            pt = int(_get(usage, "prompt_tokens") or _get(usage, "input_tokens") or 0)
            ct = int(_get(usage, "completion_tokens") or _get(usage, "output_tokens") or 0)
            tt = _get(usage, "total_tokens")
            details = _get(usage, "completion_tokens_details")
            rt = int(_get(details, "reasoning_tokens") or 0)
            input_t     += pt
            output_t    += ct
            reasoning_t += rt
            total_t     += int(tt) if tt is not None else (pt + ct)
        n = n_samples
        return {
            "avg_total_tokens":     total_t     / n if total_t     > 0 else None,
            "avg_input_tokens":     input_t     / n if input_t     > 0 else None,
            "avg_output_tokens":    output_t    / n if output_t    > 0 else None,
            "avg_reasoning_tokens": reasoning_t / n if reasoning_t > 0 else None,
        }
    except Exception:
        return {}


def evaluate_candidate(
    program,
    val_set: list,
    metric=None,
    num_threads: int = 4,
    lm=None,
) -> dict:
    """Evaluate a program on val_set and return aggregate + per-category scores.

    Parameters
    ----------
    lm : optional
        Explicit LM instance to use for inference **and** token tracking.
        When provided, every worker thread is wrapped with
        ``dspy.settings.context(lm=lm)`` so that Python's ContextVar
        propagation (or lack thereof in ThreadPoolExecutor < 3.12) never
        causes history to be recorded on the wrong LM object.

    Returns dict with keys:
        total_score, cls_score, change_score, per_category_scores, n_evaluated,
        eval_time_seconds, avg_total_tokens, avg_input_tokens, avg_output_tokens,
        avg_reasoning_tokens
    """
    if metric is None:
        metric = dca_comparison_metric

    n_total = len(val_set)
    t0 = time.time()

    # Snapshot LM history length for per-sample token tracking.
    # If caller supplied an explicit ``lm``, use it; otherwise fall back to
    # whatever ``dspy.settings.lm`` resolves to in the calling thread.
    _lm = lm or dspy.settings.lm
    _history_before = len(_lm.history) if hasattr(_lm, "history") else 0

    print(f"  [{_ts()}] Evaluating {n_total} examples ({num_threads} threads)...")

    # Single-pass evaluation with per-category breakdown.
    cat_scores: dict[str, list[float]] = defaultdict(list)
    cat_cls: dict[str, list[float]] = defaultdict(list)
    progress_lock = Lock()
    done_count = [0]
    errors_count = [0]

    def _eval_one(idx_example):
        idx, example = idx_example
        # Explicitly set the LM context in every worker thread so that
        # (a) the program uses ``_lm`` for inference and
        # (b) ``_lm.history`` grows — making token extraction correct
        # even when ThreadPoolExecutor doesn't propagate ContextVars.
        with dspy.settings.context(lm=_lm):
            try:
                pred = program(
                    image_original=getattr(example, "image_original", None),
                    image_reimage=getattr(example, "image_reimage", None),
                )
                score = metric(example, pred)
                cat = str(getattr(example, "expected_human_label_category", "Unknown"))

                exp_discard = str(getattr(example, "expected_discard_image", "")).strip().lower()
                pred_discard = str(getattr(pred, "discard_image", "")).strip().lower()
                true_vals = {"true", "yes", "1"}
                cls = compute_classification_score(
                    exp_discard in true_vals, pred_discard in true_vals
                )
            except Exception as exc:
                cat = str(getattr(example, "expected_human_label_category", "Unknown"))
                score = 0.0
                cls = 0.0
                logger.debug("Eval error on example %d: %s", idx, exc)
                with progress_lock:
                    errors_count[0] += 1

        with progress_lock:
            done_count[0] += 1
            if done_count[0] % 20 == 0 or done_count[0] == n_total:
                elapsed = time.time() - t0
                rate = done_count[0] / elapsed if elapsed > 0 else 0
                eta = (n_total - done_count[0]) / rate if rate > 0 else 0
                print(f"\r  [{_ts()}] Eval: {done_count[0]}/{n_total}  "
                      f"({rate:.1f}/s, ETA {eta:.0f}s, {errors_count[0]} errors)",
                      end="", flush=True)

        return cat, score, cls

    if num_threads <= 1:
        results = [_eval_one((i, ex)) for i, ex in enumerate(val_set)]
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            results = list(pool.map(_eval_one, enumerate(val_set)))

    for cat, score, cls in results:
        cat_scores[cat].append(score)
        cat_cls[cat].append(cls)

    elapsed = time.time() - t0
    print(f"\n  [{_ts()}] Evaluation done in {elapsed:.1f}s ({errors_count[0]} errors)")

    per_category = {}
    for cat in DCA_CATEGORIES:
        if cat in cat_scores and cat_scores[cat]:
            per_category[cat] = sum(cat_scores[cat]) / len(cat_scores[cat])

    # Compute overall cls and change from per-example scores
    all_total = []
    all_cls = []
    for cat in cat_scores:
        all_total.extend(cat_scores[cat])
        all_cls.extend(cat_cls[cat])

    avg_cls = sum(all_cls) / len(all_cls) if all_cls else 0.0
    avg_total = sum(all_total) / len(all_total) if all_total else 0.0
    # Derive change score from: total = 0.6*cls + 0.4*chg  →  chg = (total - 0.6*cls) / 0.4
    avg_change = (avg_total - 0.6 * avg_cls) / 0.4 if avg_total > 0 else 0.0
    avg_change = max(0.0, min(1.0, avg_change))

    token_stats = _extract_token_stats_from_history(_lm, _history_before, n_total)

    result = {
        "total_score": avg_total,
        "cls_score": avg_cls,
        "change_score": avg_change,
        "per_category_scores": per_category,
        "n_evaluated": len(val_set),
        "n_errors": errors_count[0],
        "eval_time_seconds": elapsed,
        **token_stats,
    }

    # Log per-category breakdown
    cat_parts = [f"{c}={per_category[c]:.3f}" for c in DCA_CATEGORIES if c in per_category]
    print(f"  Per-category: {' | '.join(cat_parts)}")

    # Log token stats if available
    if token_stats.get("avg_total_tokens") is not None:
        inp = f"{token_stats['avg_input_tokens']:.0f}"     if token_stats.get("avg_input_tokens")     is not None else "N/A"
        out = f"{token_stats['avg_output_tokens']:.0f}"    if token_stats.get("avg_output_tokens")    is not None else "N/A"
        rsn = f"{token_stats['avg_reasoning_tokens']:.0f}" if token_stats.get("avg_reasoning_tokens") is not None else "N/A"
        tot = f"{token_stats['avg_total_tokens']:.0f}"
        print(f"  Avg tokens/sample: total={tot}  input={inp}  output={out}  reasoning={rsn}")

    return result


def evaluate_candidate_at_temperature(
    program_path: str,
    val_set: list,
    metric,
    temperature: float,
    model_name: str,
    use_cot: bool = False,
    num_threads: int = 4,
) -> dict:
    """Re-evaluate a saved program at a specific temperature.

    Creates a fresh LM, loads the program, evaluates, and restores original LM.
    """
    original_lm = dspy.settings.lm
    print(f"  [{_ts()}] Re-eval at temp={temperature} with {model_name}")
    print(f"           Program: {Path(program_path).name}")

    try:
        eval_lm = create_lm(model_name, temperature=temperature, cache=False)
        dspy.configure(lm=eval_lm)
        program = load_candidate(program_path, use_cot=use_cot)
        program.lm = eval_lm
        result = evaluate_candidate(program, val_set, metric, num_threads, lm=eval_lm)
    finally:
        dspy.configure(lm=original_lm)

    return result


# ═══════════════════════════════════════════════════════════════════
# Eval result validation (Layer 3)
# ═══════════════════════════════════════════════════════════════════

def is_eval_result_trustworthy(
    result: dict,
    *,
    max_error_rate: float = 0.20,
    min_tokens: float = 500,
) -> bool:
    """Check whether an evaluation result is reliable enough to persist.

    Returns False (= discard / keep previous values) when:
      - All examples failed (n_evaluated == 0)
      - Error rate exceeds ``max_error_rate`` (default 20%)
      - Token data is null/missing despite a supposedly successful run
      - Token count is implausibly low (below ``min_tokens`` per sample)

    Use at every boundary where eval results are written to the leaderboard
    or candidate entries, to prevent Azure outages or transient errors from
    poisoning the persisted state.
    """
    n_evaluated = result.get("n_evaluated", 0)
    if n_evaluated == 0:
        return False

    n_errors = result.get("n_errors", 0)
    error_rate = n_errors / n_evaluated
    if error_rate > max_error_rate:
        return False

    avg_tok = result.get("avg_total_tokens")
    if avg_tok is None:
        return False
    if avg_tok < min_tokens:
        # Sanity floor: a real DCA inference always uses >500 tokens
        # (prompt alone is ~2.5k). A low value indicates demos were
        # silently dropped or the eval didn't actually run properly.
        return False

    return True


# ═══════════════════════════════════════════════════════════════════
# Demo key normalization
# ═══════════════════════════════════════════════════════════════════

def normalize_demos(program) -> int:
    """Fix ``expected_*`` prefixed demo keys so they survive save/load.

    DSPy's labeled demos (from training data, non-augmented) store output
    fields as ``expected_<field>``.  In memory, ``dspy.Example.__getattr__``
    silently maps ``example.<field>`` → ``example.expected_<field>``, so
    everything works.  But after ``program.save()`` + ``program.load()``
    the demos become **plain dicts** — the ``expected_*`` prefix is now a
    dead literal key that ``ChatAdapter`` cannot match to the signature's
    output fields, so the demo is silently dropped from the prompt.

    This function:
      1. Renames ``expected_<output_field>`` → ``<output_field>``
      2. Drops non-signature metadata keys (``pair_name``, ``doubt``, etc.)
      3. Sets ``augmented = True`` on fixed demos (signals bare-key format)

    Call this **before** ``program.save()`` and **after** ``program.load()``.

    Returns the number of demos that were fixed.
    """
    demos = getattr(program, "demos", None)
    if not demos:
        return 0

    sig = getattr(program, "signature", None)
    if sig is None:
        return 0

    output_fields = set(sig.output_fields.keys())
    allowed_keys = set(sig.fields.keys()) | {"augmented"}

    fixed = 0
    normalized = []
    for demo in demos:
        items = demo.items() if isinstance(demo, dict) else list(demo._store.items())
        needs_fix = any(
            k.startswith("expected_") and k[len("expected_"):] in output_fields
            for k, _ in items
        )
        if not needs_fix:
            normalized.append(demo if isinstance(demo, dict) else dict(items))
            continue

        new_demo: dict = {}
        for k, v in items:
            if k.startswith("expected_") and k[len("expected_"):] in output_fields:
                new_demo[k[len("expected_"):]] = v
            elif k in allowed_keys:
                new_demo[k] = v
            # else: drop non-signature metadata (pair_name, doubt, etc.)
        new_demo["augmented"] = True
        normalized.append(new_demo)
        fixed += 1

    program.demos = normalized
    return fixed


# ═══════════════════════════════════════════════════════════════════
# Save / Load candidates
# ═══════════════════════════════════════════════════════════════════

def save_candidate(
    program,
    path: str | Path,
    metadata: dict | None = None,
) -> Path:
    """Save a DSPy program and its metadata sidecar.

    Performs post-save verification: reloads the file and checks that demo
    keys round-trip correctly (no lingering ``expected_*`` prefixes that
    would cause silent demo dropout at inference time).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize expected_* demo keys before saving
    n_fixed = normalize_demos(program)
    if n_fixed:
        logger.info("save_candidate: normalized %d demo(s) with expected_* keys → %s", n_fixed, path.name)

    # Save DSPy state
    program.save(str(path))

    # ── Layer 1: Post-save verification ───────────────────────────
    _verify_saved_program(path, program)

    # Save metadata sidecar for every candidate
    meta = dict(metadata or {})
    meta.setdefault("saved_at", datetime.now().isoformat(timespec="seconds"))
    meta_path = path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return path


def _verify_saved_program(path: Path, original_program) -> None:
    """Verify that a saved program file has correct demo keys.

    Raises RuntimeError if:
      - Demo count in saved file differs from in-memory program
      - Any demo still has ``expected_*`` keys that would be silently dropped
    """
    sig = getattr(original_program, "signature", None)
    if sig is None:
        return  # can't verify without a signature

    output_field_names = set(sig.output_fields.keys())
    expected_demo_count = len(getattr(original_program, "demos", []))

    # Read the raw JSON to inspect demo keys at the serialization level
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return  # if we can't read, skip (don't crash on verification)

    # Find demos in the saved state (DSPy stores them at various locations)
    saved_demos = raw.get("demos", [])
    if not saved_demos:
        # Check nested structure (Module wrapping)
        for key in raw:
            if isinstance(raw[key], dict) and "demos" in raw[key]:
                saved_demos = raw[key]["demos"]
                break

    # Check 1: demo count
    if expected_demo_count > 0 and len(saved_demos) != expected_demo_count:
        logger.warning(
            "VERIFY FAILED: demo count mismatch in %s (memory=%d, file=%d)",
            path.name, expected_demo_count, len(saved_demos),
        )

    # Check 2: no lingering expected_* output keys
    for i, demo in enumerate(saved_demos):
        if not isinstance(demo, dict):
            continue
        bad_keys = [
            k for k in demo
            if k.startswith("expected_") and k[len("expected_"):] in output_field_names
        ]
        if bad_keys:
            raise RuntimeError(
                f"VERIFY FAILED: demo {i} in {path.name} has unrenamed expected_* keys: "
                f"{bad_keys}. normalize_demos() did not fix these before save. "
                f"This would cause silent demo dropout at inference time."
            )


def load_candidate(path: str | Path, use_cot: bool = False):
    """Load a saved DSPy program (Predict or ChainOfThought).

    After loading, normalizes any lingering ``expected_*`` demo keys and
    verifies the loaded demo count matches what the file contains.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Program file not found: {path}")

    if use_cot:
        program = dspy.ChainOfThought(DCAImageComparison)
    else:
        program = dspy.Predict(DCAImageComparison)

    program.load(str(path))

    # Normalize expected_* demo keys from old/buggy saved files
    n_fixed = normalize_demos(program)
    if n_fixed:
        logger.info("load_candidate: normalized %d demo(s) with expected_* keys ← %s", n_fixed, path.name)

    # Layer 1: verify loaded demos are usable
    _verify_loaded_demos(program, path)

    return program


def _verify_loaded_demos(program, path: Path) -> None:
    """Warn if a loaded program has fewer usable demos than expected.

    This catches the case where ``expected_*`` keys survived normalization
    (shouldn't happen after fix) or where DSPy's load silently dropped demos.
    """
    sig = getattr(program, "signature", None)
    if sig is None:
        return

    demos = getattr(program, "demos", [])
    if not demos:
        return

    output_fields = set(sig.output_fields.keys())
    for i, demo in enumerate(demos):
        items = demo.items() if isinstance(demo, dict) else list(demo._store.items())
        has_output = any(k in output_fields for k, _ in items)
        has_bad_expected = any(
            k.startswith("expected_") and k[len("expected_"):] in output_fields
            for k, _ in items
        )
        if has_bad_expected and not has_output:
            logger.error(
                "LOAD VERIFY: demo %d in %s still has dead expected_* keys after normalize. "
                "This demo will be silently dropped from the prompt.",
                i, path.name,
            )


def get_program_info(program) -> dict:
    """Extract metadata from a loaded program (demo count, instruction length)."""
    n_demos = len(getattr(program, "demos", []))
    instruction = ""
    sig = getattr(program, "signature", None)
    if sig:
        instruction = getattr(sig, "instructions", "") or ""
    return {
        "n_demos": n_demos,
        "instruction_length": len(instruction),
        "instruction_preview": instruction[:200] if instruction else "",
    }


# ═══════════════════════════════════════════════════════════════════
# Category-balanced sampling
# ═══════════════════════════════════════════════════════════════════

def category_balanced_sample(
    dataset: list,
    max_per_category: int = 1,
    seed: int = 42,
) -> list:
    """Select up to ``max_per_category`` examples per human_label_category.

    Returns a flat list (up to 8 × max_per_category examples for DCA).
    """
    import random
    rng = random.Random(seed)

    groups: dict[str, list] = defaultdict(list)
    for ex in dataset:
        cat = str(getattr(ex, "expected_human_label_category", "Unknown"))
        groups[cat].append(ex)

    selected = []
    for cat in DCA_CATEGORIES:
        pool = groups.get(cat, [])
        if pool:
            n = min(max_per_category, len(pool))
            selected.extend(rng.sample(pool, n))

    return selected


def create_mini_dataset(
    dataset: list,
    target_size: int | None,
    seed: int = 42,
    stratify_by: list[str] | None = None,
) -> list:
    """Create a stratified mini-dataset using ``stratified_split_dca_dataset``.

    Delegates to the *same* stratified splitting function used for the main
    train/val/test split, so the mini set preserves the exact same
    distribution characteristics (binary classification balance, category
    proportions, doubt distribution).

    Parameters
    ----------
    dataset : list
        Full DSPy dataset (list of dspy.Example).
    target_size : int or None
        Desired number of examples.  None means use full dataset.
    seed : int
        RNG seed for reproducibility.
    stratify_by : list[str] or None
        Field name(s) to stratify by.  Passed through to
        ``stratified_split_dca_dataset``.  Uses its default (3-field
        composite key) when None.
    """
    if target_size is None or target_size >= len(dataset):
        return dataset

    from utils import stratified_split_dca_dataset

    mini_ratio = target_size / len(dataset)
    kwargs: dict = {
        "train_ratio": mini_ratio,
        "val_ratio": 1.0 - mini_ratio,
        "test_ratio": 0.0,
        "random_seed": seed,
        "verbose": False,
    }
    if stratify_by is not None:
        kwargs["stratify_by"] = stratify_by

    # Take the "train" portion — that's our mini subset
    mini, _, _ = stratified_split_dca_dataset(dataset, **kwargs)

    # Log per-category breakdown for readability
    from collections import Counter as _Counter
    cat_counts = _Counter(
        str(getattr(ex, "expected_human_label_category", "?")) for ex in mini
    )
    cat_str = ", ".join(
        f"{c}={cat_counts.get(c, 0)}" for c in DCA_CATEGORIES if c in cat_counts
    )
    print(f"  [{_ts()}] Mini dataset: {len(mini)}/{len(dataset)} examples [{cat_str}]")
    return mini


def find_worst_performing_examples(
    program,
    dataset: list,
    metric,
    n: int,
    per_category_limit: int = 2,
    num_threads: int = 4,
    return_eval_result: bool = False,
    select_best: bool = False,
) -> "list[dict] | tuple[list[dict], dict]":
    """Run inference on dataset and return the ``n`` worst- (or best-) scoring examples.

    Balances across categories (max ``per_category_limit`` per category).
    Each returned item is a dict with: example, prediction, score, category.

    Args:
        return_eval_result: If True, also return an aggregate eval-result dict
            (same format as ``evaluate_candidate``) computed from the **same**
            single inference pass — eliminating a redundant second evaluation.
            In that case the function returns ``(worst_examples, eval_result)``.
        select_best: If True, return the *best*-scoring examples instead of
            the worst.  Useful for providing gold-standard demonstrations to
            the seed LLM rather than failure examples.
    """
    n_total = len(dataset)
    t0 = time.time()
    print(f"  [{_ts()}] Scoring {n_total} examples ({num_threads} threads)...")
    progress_lock = Lock()
    done_count = [0]
    errors_count = [0]

    def _score_one(example):
        try:
            pred = program(
                image_original=getattr(example, "image_original", None),
                image_reimage=getattr(example, "image_reimage", None),
            )
            score = metric(example, pred)
            # Compute binary classification score for optional eval_result
            exp_discard = str(getattr(example, "expected_discard_image", "")).strip().lower()
            pred_discard = str(getattr(pred, "discard_image", "")).strip().lower()
            true_vals = {"true", "yes", "1"}
            cls = compute_classification_score(
                exp_discard in true_vals, pred_discard in true_vals
            )
        except Exception as exc:
            pred = None
            score = 0.0
            cls = 0.0
            logger.debug("Score error: %s", exc)
            with progress_lock:
                errors_count[0] += 1

        cat = str(getattr(example, "expected_human_label_category", "Unknown"))

        with progress_lock:
            done_count[0] += 1
            if done_count[0] % 20 == 0 or done_count[0] == n_total:
                elapsed = time.time() - t0
                rate = done_count[0] / elapsed if elapsed > 0 else 0
                eta = (n_total - done_count[0]) / rate if rate > 0 else 0
                print(f"\r  [{_ts()}] Scoring: {done_count[0]}/{n_total}  "
                      f"({rate:.1f}/s, ETA {eta:.0f}s, {errors_count[0]} errors)",
                      end="", flush=True)

        return {
            "example": example,
            "prediction": pred,
            "score": score,
            "cls": cls,
            "category": cat,
        }

    if num_threads <= 1:
        scored = [_score_one(ex) for ex in dataset]
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            scored = list(pool.map(_score_one, dataset))

    elapsed = time.time() - t0
    print(f"\n  [{_ts()}] Scoring done in {elapsed:.1f}s ({errors_count[0]} errors)")

    # Sort: ascending = worst first; descending = best first
    scored.sort(key=lambda x: x["score"], reverse=select_best)

    # Pick worst/best N, balanced across categories
    selected = []
    cat_counts: dict[str, int] = defaultdict(int)
    for item in scored:
        if len(selected) >= n:
            break
        cat = item["category"]
        if cat_counts[cat] < per_category_limit:
            selected.append(item)
            cat_counts[cat] += 1

    if not return_eval_result:
        return selected

    # ── Build aggregate eval_result from the same inference pass ──────
    cat_scores_agg: dict[str, list[float]] = defaultdict(list)
    cat_cls_agg: dict[str, list[float]] = defaultdict(list)
    for item in scored:
        cat_scores_agg[item["category"]].append(item["score"])
        cat_cls_agg[item["category"]].append(item["cls"])

    per_category: dict[str, float] = {}
    for cat in DCA_CATEGORIES:
        if cat in cat_scores_agg and cat_scores_agg[cat]:
            per_category[cat] = sum(cat_scores_agg[cat]) / len(cat_scores_agg[cat])

    all_total = [s for vals in cat_scores_agg.values() for s in vals]
    all_cls   = [s for vals in cat_cls_agg.values() for s in vals]
    avg_cls    = sum(all_cls)   / len(all_cls)   if all_cls   else 0.0
    avg_total  = sum(all_total) / len(all_total) if all_total else 0.0
    avg_change = (avg_total - 0.6 * avg_cls) / 0.4 if avg_total > 0 else 0.0
    avg_change = max(0.0, min(1.0, avg_change))

    eval_result = {
        "total_score":         avg_total,
        "cls_score":           avg_cls,
        "change_score":        avg_change,
        "per_category_scores": per_category,
        "n_evaluated":         len(scored),
    }

    cat_parts = [f"{c}={per_category[c]:.3f}" for c in DCA_CATEGORIES if c in per_category]
    print(f"  Per-category: {' | '.join(cat_parts)}")

    return selected, eval_result


# ═══════════════════════════════════════════════════════════════════
# Summary generation
# ═══════════════════════════════════════════════════════════════════

def generate_summary(
    config,
    leaderboard: OptimizationLeaderboard,
    stages_run: list[str],
    total_time_seconds: float,
    test_eval_result: dict | None = None,
    started_at: str = "",
    stage_top_k_history: dict | None = None,
    stage_pre_best: dict | None = None,
    stage_execution_history: dict | None = None,
) -> dict:
    """Build a complete end-of-run summary dict.

    Parameters
    ----------
    stage_top_k_history : dict[str, list[dict]], optional
        Maps stage name → frozen snapshots of the top-K entries that were
        passed to the next stage. Recorded in run_tournament.py at selection
        time so the audit trail is not mutated by later full-val re-evaluation.
    stage_pre_best : dict[str, float], optional
        Maps stage name → the global best val_total_score immediately before
        that stage ran.  Used to compute true cross-stage improvement.
    """
    final_selection_exclude_stages = ["seed_baseline"]

    summary = {
        "tournament_id": f"tournament_{config.timestamp}",
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "total_duration_seconds": round(total_time_seconds, 1),
        "config": config.to_dict(),
        "stages_run": stages_run,        # stages freshly executed this session
        "per_stage": {},
    }

    # Include any stage that materially participated in the run, even if it
    # failed before producing leaderboard entries. This keeps summary_error.json
    # honest for zero-candidate failures such as a bootstrap subprocess crash.
    # "seed_baseline" is diagnostic (scored on train, not val) so exclude it.
    leaderboard_stages = [s for s in leaderboard.get_stages() if s != "seed_baseline"]
    execution_stages = [
        s for s in (stage_execution_history or {}).keys()
        if s != "seed_baseline"
    ]
    participated_stages = set(leaderboard_stages) | set(execution_stages) | {
        s for s in stages_run if s != "seed_baseline"
    }

    all_stages = [
        stage for stage in config.stages
        if stage in participated_stages
    ]
    for stage in leaderboard_stages + execution_stages + stages_run:
        if stage != "seed_baseline" and stage in participated_stages and stage not in all_stages:
            all_stages.append(stage)
    summary["stages_all"] = all_stages

    for stage in all_stages:
        stage_sum = leaderboard.get_stage_summary(stage)

        # Override improvement with "vs global best before this stage ran"
        # if we have that info (more accurate than vs-previous-stage-best).
        if stage_pre_best and stage in stage_pre_best:
            stage_sum["pre_stage_best"] = stage_pre_best[stage]
            stage_sum["improvement_over_alltime_prev"] = round(
                stage_sum["best_score"] - stage_pre_best[stage], 4
            )

        # Record which top-K entries were passed to the next stage.
        if stage_top_k_history and stage in stage_top_k_history:
            tk = stage_top_k_history[stage]
            stage_sum["top_k_passed_to_next"] = len(tk)
            stage_sum["top_k_entries"] = [
                {
                    "stage": e["stage"],
                    "config": e["config_name"],
                    "score": round(e["val_total_score"], 4),
                    "program_path": e["program_path"],
                    "optimization_started_at": e.get("optimization_started_at"),
                    "optimization_completed_at": e.get("optimization_completed_at"),
                    "optimization_duration_seconds": e.get("optimization_duration_seconds"),
                }
                for e in tk
            ]

        if stage_execution_history and stage in stage_execution_history:
            stage_sum["execution"] = dict(stage_execution_history[stage])

        summary["per_stage"][stage] = stage_sum

    # Final leaders / selection guidance
    full = leaderboard.get_full_summary(
        selection_criteria=config.selection_criteria,
        exclude_stages=final_selection_exclude_stages,
    )
    summary["selection_criteria"] = config.selection_criteria
    summary["score_leader"] = full.get("score_leader")
    if config.selection_criteria == "pareto":
        summary["final_selection_mode"] = "manual_pareto_choice"
        try:
            summary["pareto_front"] = [
                {
                    "stage": e.stage,
                    "config_name": e.config_name,
                    "val_total_score": round(e.val_total_score, 4),
                    "val_avg_total_tokens": e.val_avg_total_tokens,
                    "program_path": e.program_path,
                }
                for e in leaderboard.pareto_front([
                    e for e in leaderboard.entries
                    if e.stage not in final_selection_exclude_stages
                ])
            ]
        except Exception:
            summary["pareto_front"] = []
    else:
        summary["selection_leader"] = full.get("selection_leader")
    summary["n_total_candidates"] = full["n_total_candidates"]

    # Final tournament / test eval
    if test_eval_result:
        summary["test_eval"] = test_eval_result

    # Ranked list (compact)
    summary["all_candidates_ranked"] = [
        {
            "rank": i + 1,
            "stage": e.stage,
            "config": e.config_name,
            "val_total": round(e.val_total_score, 4),
            "val_cls": round(e.val_cls_score, 4),
            "val_change": round(e.val_change_score, 4),
            "val_avg_total_tokens": e.val_avg_total_tokens,
            "val_avg_input_tokens": e.val_avg_input_tokens,
            "val_avg_output_tokens": e.val_avg_output_tokens,
            "val_avg_reasoning_tokens": e.val_avg_reasoning_tokens,
            "opt_prompt_tokens": e.prompt_tokens,
            "opt_completion_tokens": e.completion_tokens,
            "optimization_started_at": (e.metadata or {}).get("optimization_started_at"),
            "optimization_completed_at": (e.metadata or {}).get("optimization_completed_at"),
            "optimization_duration_seconds": (e.metadata or {}).get("optimization_duration_seconds"),
            "compile_duration_seconds": (e.metadata or {}).get("compile_duration_seconds"),
            "evaluation_duration_seconds": (e.metadata or {}).get("evaluation_duration_seconds", e.val_eval_time_seconds),
            "parent": Path(e.parent_path).stem if e.parent_path else None,
            "program_path": e.program_path,
        }
        for i, e in enumerate(
            [
                entry for entry in leaderboard.entries
                if entry.stage not in final_selection_exclude_stages
            ]
        )
    ]

    return summary


def save_summary(summary: dict, *paths: str | Path):
    """Save summary JSON to one or more locations."""
    for p in paths:
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  Summary saved → {p}")


def print_final_summary(summary: dict):
    """Print the end-of-tournament summary with a full cascade table."""
    W = 82
    print(f"\n{'#'*W}")
    print(f" TOURNAMENT COMPLETE — {summary['tournament_id']}")
    print(f"{'#'*W}")
    print(f"  Duration   : {summary['total_duration_seconds']:.0f}s")

    # Use stages_all (includes resumed stages) if present, else fall back
    stages_all = summary.get("stages_all", summary.get("stages_run", []))
    stages_run = summary.get("stages_run", [])
    print(f"  Stages     : {' → '.join(stages_all)}")
    if stages_run != stages_all:
        print(f"  Run this session: {' → '.join(stages_run) or '(none — all resumed)'}")
    print(f"  Candidates : {summary['n_total_candidates']}")

    # ── Cascade table ────────────────────────────────────────────────────────
    per_stage = summary.get("per_stage", {})
    if stages_all and per_stage:
        print(f"\n{'─'*W}")
        print(f"  STAGE CASCADE  (cross-stage top-K selection: best-so-far flows forward)")
        print(f"{'─'*W}")
        # Header
        print(f"  {'Stage':<12} {'#Cands':>7} {'BestScore':>10} {'Δ vs prev':>11}"
              f"  {'Best config':<32} {'→ Next stage'}")
        print(f"  {'─'*12} {'─'*7} {'─'*10} {'─'*11}"
              f"  {'─'*32} {'─'*16}")

        n_stages = len(stages_all)
        for i, stage in enumerate(stages_all):
            ps = per_stage.get(stage, {})
            n = ps.get("n_candidates", 0)
            best = ps.get("best_score", 0.0)
            # Prefer the more accurate "improvement_over_alltime_prev" if available
            imp = ps.get("improvement_over_alltime_prev", ps.get("improvement_over_prev"))
            imp_str = f"{imp:+.4f}" if imp is not None else "—"
            bname = (ps.get("best_candidate") or "—")[:32]

            topk = ps.get("top_k_passed_to_next")
            if topk is not None and i < n_stages - 1:
                next_stage = stages_all[i + 1]
                next_str = f"{topk} → {next_stage}"
            elif i == n_stages - 1:
                next_str = "(final stage)"
            else:
                next_str = "?"

            resumed = "" if stage in stages_run else " [resumed]"
            print(f"  {stage:<12} {n:>7} {best:>10.4f} {imp_str:>11}"
                  f"  {bname:<32} {next_str}{resumed}")

        print(f"  {'─'*12} {'─'*7} {'─'*10} {'─'*11}"
              f"  {'─'*32} {'─'*16}")

    # ── Top-K entries passed to each next stage ──────────────────────────────
    any_topk = any(
        per_stage.get(s, {}).get("top_k_entries")
        for s in stages_all[:-1]  # last stage has nothing after it
    )
    if any_topk:
        print(f"\n  TOP-K FLOW (candidates that seeded each subsequent stage):")
        for i, stage in enumerate(stages_all[:-1]):
            ps = per_stage.get(stage, {})
            tk_entries = ps.get("top_k_entries")
            if tk_entries:
                next_stage = stages_all[i + 1]
                print(f"\n    [{stage}] → [{next_stage}]  ({len(tk_entries)} candidates):")
                for j, te in enumerate(tk_entries, 1):
                    src = te.get("stage", stage)
                    # Show source stage if it differs (cross-stage selection)
                    src_tag = f"[{src}] " if src != stage else ""
                    print(f"      {j}. {src_tag}{te['config']}  score={te['score']:.4f}")

    # ── Final leaders / selection guidance ──────────────────────────────────
    def _print_candidate_block(label: str, candidate: dict) -> None:
        print(f"\n{'─'*W}")
        print(f"  {label}: [{candidate['stage']}] {candidate['config_name']}")
        print(f"{'─'*W}")
        print(f"    Val total  = {candidate['val_total_score']:.4f}")
        print(f"    Val cls    = {candidate['val_cls_score']:.4f}")
        print(f"    Val change = {candidate['val_change_score']:.4f}")
        if candidate.get("val_avg_total_tokens") is not None:
            print(f"    Avg tok/smp = {candidate['val_avg_total_tokens']:.0f}"
                  f"  (in={candidate.get('val_avg_input_tokens') or 0:.0f}"
                  f" out={candidate.get('val_avg_output_tokens') or 0:.0f})")
        if candidate.get("per_category_scores"):
            cats = sorted(candidate["per_category_scores"].keys())
            cat_str = "  ".join(
                f"{c}={candidate['per_category_scores'][c]:.3f}" for c in cats
            )
            print(f"    Per-cat    : {cat_str}")

    score_leader = summary.get("score_leader")
    selection_leader = summary.get("selection_leader")
    selection_criteria = summary.get("selection_criteria", "score_only")

    if selection_criteria == "pareto":
        print(f"\n{'─'*W}")
        print("  FINAL SELECTION: manual from Pareto front / plot")
        print(f"{'─'*W}")
        pareto_entries = summary.get("pareto_front") or []
        print(f"    Selection criteria = pareto")
        print(f"    Pareto candidates  = {len(pareto_entries)}")
    elif selection_leader:
        _print_candidate_block(f"SELECTION LEADER ({selection_criteria})", selection_leader)

    if score_leader and score_leader != selection_leader:
        _print_candidate_block("SCORE LEADER", score_leader)

    # ── Test set evaluation ───────────────────────────────────────────────────
    te = summary.get("test_eval")
    if te:
        print(f"\n{'─'*W}")
        print(f"  TEST SET EVALUATION (winner on held-out test set):")
        print(f"{'─'*W}")
        print(f"    Total  = {te.get('avg_total_score', te.get('total_score', 'N/A'))}")
        print(f"    Cls    = {te.get('avg_cls_score', te.get('cls_score', 'N/A'))}")
        print(f"    Change = {te.get('avg_change_score', te.get('change_score', 'N/A'))}")
        if te.get("n_evaluated"):
            print(f"    N eval = {te['n_evaluated']}")

    print(f"{'#'*W}\n")


# ═══════════════════════════════════════════════════════════════════
# Token-usage helpers
# ═══════════════════════════════════════════════════════════════════

def token_usage_from_tracker(tracker) -> dict:
    """Extract aggregate token counts from a ``dspy.track_usage()`` tracker.

    Parameters
    ----------
    tracker : object
        The object returned by ``dspy.track_usage().__enter__()``.
        Expected to have ``get_total_tokens()`` → dict[model_name → {prompt_tokens, completion_tokens}].

    Returns
    -------
    dict with keys prompt_tokens, completion_tokens, total_tokens (all int, summed across models).
    """
    prompt = 0
    completion = 0
    try:
        totals = tracker.get_total_tokens()
        if isinstance(totals, dict):
            for model_name, usage in totals.items():
                if isinstance(usage, dict):
                    prompt += usage.get("prompt_tokens", 0) or 0
                    completion += usage.get("completion_tokens", 0) or 0
    except Exception:
        pass
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


# ═══════════════════════════════════════════════════════════════════
# Pareto scatter plot
# ═══════════════════════════════════════════════════════════════════

def plot_pareto_candidates(
    leaderboard: OptimizationLeaderboard,
    output_path: str | Path,
    title: str = "Tournament Candidates — Score vs Avg Tokens/Sample",
    exclude_stages: list[str] | None = None,
):
    """Generate a 2D scatter plot: val_total_score (X, %) vs val_avg_total_tokens (Y)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [WARN] matplotlib not available — skipping scatter plot")
        return

    excluded = set(exclude_stages or [])
    entries = [e for e in leaderboard.entries if e.stage not in excluded]
    if not entries:
        print(f"  [WARN] No entries in leaderboard — skipping scatter plot")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_token_entries = [e for e in entries if e.val_avg_total_tokens is not None and e.val_avg_total_tokens > 0]
    if not valid_token_entries:
        print("  [WARN] No val_avg_total_tokens available — skipping Pareto plot")
        return

    stage_colours = {
        "seed": "#2196F3",
        "seed_baseline": "#90CAF9",
        "bootstrap": "#4CAF50",
        "miprov2": "#FF9800",
        "gepa": "#9C27B0",
        "external": "#607D8B",
    }
    top_markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.scatter(
        [e.val_total_score * 100.0 for e in valid_token_entries],
        [e.val_avg_total_tokens for e in valid_token_entries],
        c="#BDBDBD",
        label="All candidates",
        alpha=0.75,
        s=45,
        edgecolors="white",
        linewidths=0.4,
        zorder=1,
    )

    pareto_entries = leaderboard.pareto_front(valid_token_entries)
    if pareto_entries:
        pareto_sorted = sorted(pareto_entries, key=lambda e: (e.val_total_score, e.val_avg_total_tokens))
        px = [e.val_total_score * 100.0 for e in pareto_sorted]
        py = [e.val_avg_total_tokens for e in pareto_sorted]
        ax.step(px, py, where="post", color="#D32F2F", linewidth=2.0, label="Pareto front", zorder=3)
        ax.scatter(px, py, c="#D32F2F", marker="D", s=42, zorder=4)

    top_labeled = sorted(valid_token_entries, key=lambda e: e.val_total_score, reverse=True)[:4]
    for rank, e in enumerate(top_labeled, 1):
        x = e.val_total_score * 100.0
        y = e.val_avg_total_tokens
        ax.scatter(
            [x], [y],
            c=stage_colours.get(e.stage, "#455A64"),
            marker=top_markers[(rank - 1) % len(top_markers)],
            s=95,
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
        )
        label = f"#{rank} {e.config_name}\n({e.stage})"
        ax.annotate(
            label, (x, y),
            textcoords="offset points", xytext=(8, 6),
            fontsize=7, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="grey", alpha=0.8),
        )

    ax.set_xlabel("Val Total Score (%)", fontsize=12)
    ax.set_ylabel("Avg Tokens / Sample (val set)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  [{_ts()}] Scatter plot saved → {output_path}")
