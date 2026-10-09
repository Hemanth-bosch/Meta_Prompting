"""
SESS-LC subset selection for DCA.

DCA adaptation:
- Dev selection: SESS-LC Low-Confidence Selection.
- Train selection: existing IPOMP-Difficulty Hard + Diverse logic.

SESS-LC idea:
    Candidate DCA examples
            |
            v
       Baseline model
            |
            v
      DCA performance
            |
            v
      Difficulty/Error
            |
            v
   Low-confidence examples
            |
            v
        Dev set

For DCA, the existing DCA metric is used as the confidence signal:

    confidence(example) = DCA score
    difficulty(example) = 1 - DCA score

Therefore:
    high score  -> high confidence -> low priority
    low score   -> low confidence  -> high priority

The Dev set is selected FIRST.
The caller must remove Dev examples before selecting Train.

Validation and Test are never passed to this module.
"""

from __future__ import annotations

from typing import Any


def _example_id(example: Any) -> str:
    """
    Return a stable identifier for a DCA example.
    """
    pair_name = getattr(example, "pair_name", None)

    if pair_name is not None:
        return str(pair_name)

    example_id = getattr(example, "id", None)

    if example_id is not None:
        return str(example_id)

    return str(id(example))


def _safe_score(
    example: Any,
    prediction: Any,
    metric,
) -> float:
    """
    Compute the DCA score safely.

    The supplied metric is expected to return a value in [0, 1].
    """
    score = float(metric(example, prediction))

    # Protect against numerical / implementation issues.
    score = max(0.0, min(1.0, score))

    return score


def score_dca_candidates(
    examples,
    comparator,
    metric,
):
    """
    Run the baseline comparator on every candidate and calculate:

        confidence = DCA metric score
        difficulty = 1 - confidence

    Returns a list of dictionaries containing:
        example
        score
        confidence
        difficulty
        prediction

    IMPORTANT:
    This function evaluates ONLY the examples passed to it.
    """

    examples = list(examples)

    scored_examples = []

    print("\n" + "=" * 80)
    print("SESS-LC BASELINE CONFIDENCE SCORING")
    print("=" * 80)

    print(f"Candidates to score : {len(examples)}")

    for index, example in enumerate(examples, start=1):

        try:
            prediction = comparator(
                image_original=example.image_original,
                image_reimage=example.image_reimage,
            )

            score = _safe_score(
                example=example,
                prediction=prediction,
                metric=metric,
            )

            confidence = score
            difficulty = 1.0 - confidence

            scored_examples.append(
                {
                    "example": example,
                    "prediction": prediction,
                    "score": score,
                    "confidence": confidence,
                    "difficulty": difficulty,
                }
            )

            print(
                f"\rScoring {index}/{len(examples)}",
                end="",
                flush=True,
            )

        except Exception as exc:

            print(
                f"\n[WARNING] Could not score "
                f"{_example_id(example)}: {exc}"
            )

    print()

    if not scored_examples:
        raise RuntimeError(
            "SESS-LC could not score any candidate examples."
        )

    print("\nBaseline scoring complete.")
    print(f"Successfully scored : {len(scored_examples)}")

    return scored_examples


def select_sess_lc_examples(
    examples,
    sample_size,
    comparator,
    metric,
):
    """
    Select a Dev set using SESS-LC.

    Selection principle:

        Low confidence / high difficulty
                    >
        High confidence / low difficulty

    For DCA:

        confidence = dca_comparison_metric(...)
        difficulty = 1 - confidence

    The lowest-scoring DCA examples are therefore selected first.

    This is deliberately different from SESS-REP:

        SESS-REP -> representative / coverage based
        SESS-LC  -> low-confidence / difficulty based
    """

    examples = list(examples)

    if sample_size <= 0:
        raise ValueError(
            f"sample_size must be positive, got {sample_size}"
        )

    if sample_size > len(examples):
        raise ValueError(
            f"Requested Dev size={sample_size}, "
            f"but only {len(examples)} candidates are available."
        )

    print("\n" + "=" * 80)
    print("SESS-LC DEV SELECTION")
    print("=" * 80)

    print(
        f"Candidate pool : {len(examples)}"
    )

    print(
        f"Requested Dev  : {sample_size}"
    )

    # ------------------------------------------------------------
    # 1. Baseline evaluation
    # ------------------------------------------------------------

    scored_examples = score_dca_candidates(
        examples=examples,
        comparator=comparator,
        metric=metric,
    )

    # ------------------------------------------------------------
    # 2. Rank by low confidence
    # ------------------------------------------------------------
    #
    # Lower DCA score = lower confidence.
    #
    # Therefore:
    #
    #     score 0.00 -> highest priority
    #     score 0.20 -> high priority
    #     score 0.50 -> medium priority
    #     score 0.80 -> low priority
    #     score 1.00 -> lowest priority
    #
    # Difficulty = 1 - score gives exactly the same ordering.
    # ------------------------------------------------------------

    scored_examples.sort(
        key=lambda item: (
            -item["difficulty"],
            _example_id(item["example"]),
        )
    )

    selected = scored_examples[:sample_size]

    # ------------------------------------------------------------
    # 3. Print selected Dev examples
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("SESS-LC SELECTED DEV EXAMPLES")
    print("=" * 80)

    for rank, item in enumerate(selected, start=1):

        print(
            f"{rank:3d}. "
            f"{_example_id(item['example'])} | "
            f"score={item['score']:.4f} | "
            f"confidence={item['confidence']:.4f} | "
            f"difficulty={item['difficulty']:.4f}"
        )

    print("=" * 80)

    print(
        f"SESS-LC Dev selected : {len(selected)}"
    )

    print(
        f"Highest difficulty   : "
        f"{selected[0]['difficulty']:.4f}"
    )

    print(
        f"Lowest difficulty    : "
        f"{selected[-1]['difficulty']:.4f}"
    )

    print("=" * 80)

    return [
        item["example"]
        for item in selected
    ]


def select_sess_lc_dev_and_get_remaining(
    examples,
    dev_size,
    comparator,
    metric,
):
    """
    Convenience function for the DCA pipeline.

    1. Select Dev using SESS-LC.
    2. Remove Dev from the candidate pool.
    3. Return:
         dev_set
         remaining_train_candidates

    The existing Hard + Diverse Train selector can then operate
    on remaining_train_candidates exactly as before.
    """

    examples = list(examples)

    if dev_size >= len(examples):
        raise ValueError(
            f"Dev size={dev_size} leaves no candidates for Train. "
            f"Available={len(examples)}."
        )

    # ------------------------------------------------------------
    # SESS-LC -> Dev
    # ------------------------------------------------------------

    dev_set = select_sess_lc_examples(
        examples=examples,
        sample_size=dev_size,
        comparator=comparator,
        metric=metric,
    )

    dev_ids = {
        _example_id(example)
        for example in dev_set
    }

    # ------------------------------------------------------------
    # Remove Dev from candidate pool
    # ------------------------------------------------------------

    remaining_train_candidates = [
        example
        for example in examples
        if _example_id(example) not in dev_ids
    ]

    # ------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------

    if len(dev_set) != dev_size:
        raise RuntimeError(
            "SESS-LC produced the wrong Dev size. "
            f"Expected={dev_size}, got={len(dev_set)}"
        )

    remaining_ids = {
        _example_id(example)
        for example in remaining_train_candidates
    }

    overlap = dev_ids & remaining_ids

    if overlap:
        raise RuntimeError(
            "SESS-LC Dev examples were found in the remaining "
            f"training candidate pool: {sorted(overlap)}"
        )

    print("\n" + "=" * 80)
    print("SESS-LC STAGE COMPLETE")
    print("=" * 80)

    print(
        f"Original candidate pool : {len(examples)}"
    )

    print(
        f"SESS-LC Dev             : {len(dev_set)}"
    )

    print(
        f"Remaining train pool    : "
        f"{len(remaining_train_candidates)}"
    )

    print(
        "Train/Dev overlap       : 0"
    )

    print("=" * 80)

    return (
        dev_set,
        remaining_train_candidates,
    )