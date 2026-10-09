from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

import dspy


DCA_CATEGORIES = [
    "A1",
    "A2",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5a",
    "B5b",
]


def get_dca_category(example: dspy.Example) -> str:
    """
    Extract the human-labelled DCA category.

    Expected field:
        expected_human_label_category

    Fallback:
        human_label_category
        category
    """

    category = getattr(
        example,
        "expected_human_label_category",
        None,
    )

    if category is None:
        category = getattr(
            example,
            "human_label_category",
            None,
        )

    if category is None:
        category = getattr(
            example,
            "category",
            None,
        )

    if category is None:
        raise ValueError(
            f"Example {getattr(example, 'pair_name', 'unknown')} "
            "does not contain a DCA category."
        )

    category = str(category).strip()

    if category not in DCA_CATEGORIES:
        raise ValueError(
            f"Unknown DCA category '{category}' for "
            f"example {getattr(example, 'pair_name', 'unknown')}."
        )

    return category


def _example_id(example: dspy.Example) -> str:
    """
    Return a stable unique identifier.
    """

    pair_name = getattr(example, "pair_name", None)

    if pair_name is not None:
        return str(pair_name)

    example_id = getattr(example, "id", None)

    if example_id is not None:
        return str(example_id)

    raise ValueError(
        "Could not determine a unique ID for DCA example."
    )


def group_by_category(
    examples: List[dspy.Example],
) -> Dict[str, List[dspy.Example]]:
    """
    Group examples according to their DCA category.
    """

    groups = {
        category: []
        for category in DCA_CATEGORIES
    }

    for example in examples:
        category = get_dca_category(example)
        groups[category].append(example)

    return groups


def print_category_distribution(
    examples: List[dspy.Example],
    title: str,
) -> None:
    """
    Print category distribution without modifying examples.
    """

    groups = group_by_category(examples)

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(f"Total examples: {len(examples)}")

    for category in DCA_CATEGORIES:
        print(
            f"  {category:<5}: "
            f"{len(groups[category]):>3}"
        )

    print("=" * 80)


def calculate_balanced_quotas(
    category_counts: Dict[str, int],
    sample_size: int,
) -> Dict[str, int]:
    """
    Allocate sample_size as equally as possible across
    all available categories.

    Categories with fewer examples than their fair quota
    are capped at their available count.

    Remaining slots are redistributed among categories
    that still have unused examples.
    """

    available = {
        category: count
        for category, count in category_counts.items()
        if count > 0
    }

    if not available:
        raise ValueError(
            "No DCA categories are available."
        )

    total_available = sum(available.values())

    if sample_size > total_available:
        raise ValueError(
            f"Requested {sample_size} samples, but only "
            f"{total_available} examples are available."
        )

    quotas = {
        category: 0
        for category in available
    }

    remaining = sample_size

    active_categories = set(available.keys())

    while remaining > 0 and active_categories:

        # Equal share among currently active categories.
        base = remaining // len(active_categories)

        if base == 0:
            # Fewer remaining slots than categories.
            # Give one additional slot to categories
            # in deterministic order.
            for category in sorted(active_categories):

                if remaining == 0:
                    break

                if quotas[category] < available[category]:
                    quotas[category] += 1
                    remaining -= 1

            break

        capped = []

        for category in sorted(active_categories):

            capacity = (
                available[category]
                - quotas[category]
            )

            if capacity <= base:

                quotas[category] += capacity
                remaining -= capacity
                capped.append(category)

            else:

                quotas[category] += base
                remaining -= base

        for category in capped:
            active_categories.remove(category)

    # Final redistribution.
    if remaining > 0:

        for category in sorted(available):

            capacity = (
                available[category]
                - quotas[category]
            )

            if capacity <= 0:
                continue

            take = min(capacity, remaining)

            quotas[category] += take
            remaining -= take

            if remaining == 0:
                break

    if sum(quotas.values()) != sample_size:
        raise RuntimeError(
            f"Could not allocate exactly {sample_size} samples. "
            f"Quotas={quotas}"
        )

    return quotas


def select_true_balanced(
    examples: List[dspy.Example],
    sample_size: int,
    seed: int = 42,
) -> Tuple[List[dspy.Example], Dict[str, int]]:
    """
    Select a category-balanced subset.

    Every available category is represented as equally
    as possible.

    No example is duplicated.
    """

    if sample_size <= 0:
        raise ValueError(
            "sample_size must be greater than zero."
        )

    groups = group_by_category(examples)

    category_counts = {
        category: len(groups[category])
        for category in DCA_CATEGORIES
    }

    quotas = calculate_balanced_quotas(
        category_counts,
        sample_size,
    )

    rng = random.Random(seed)

    selected = []

    for category in DCA_CATEGORIES:

        category_examples = list(
            groups[category]
        )

        if not category_examples:
            continue

        rng.shuffle(category_examples)

        quota = quotas.get(category, 0)

        selected.extend(
            category_examples[:quota]
        )

    rng.shuffle(selected)

    # Verify exact size.
    if len(selected) != sample_size:
        raise RuntimeError(
            f"Expected {sample_size} examples, "
            f"got {len(selected)}."
        )

    # Verify uniqueness.
    ids = [
        _example_id(example)
        for example in selected
    ]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Duplicate examples detected."
        )

    selected_counts = {
        category: 0
        for category in DCA_CATEGORIES
    }

    for example in selected:
        category = get_dca_category(example)
        selected_counts[category] += 1

    return selected, selected_counts


def select_true_balanced_train_and_dev(
    train_pool: List[dspy.Example],
    train_size: int,
    dev_size: int,
    seed: int = 42,
):
    """
    Create category-balanced GEPA Train and Dev sets
    from the 254-example training pool.

    Validation and test are not involved.

    Train and Dev are guaranteed not to overlap.
    """

    required = train_size + dev_size

    if len(train_pool) < required:
        raise ValueError(
            f"Training pool contains {len(train_pool)} examples, "
            f"but Train + Dev requires {required}."
        )

    print_category_distribution(
        train_pool,
        "TRAIN POOL CATEGORY DISTRIBUTION",
    )

    # ---------------------------------------------------------
    # STEP 1
    # Select combined Train + Dev pool in a balanced way.
    # ---------------------------------------------------------

    combined, combined_counts = select_true_balanced(
        examples=train_pool,
        sample_size=required,
        seed=seed,
    )

    print_category_distribution(
        combined,
        "BALANCED TRAIN + DEV POOL",
    )

    # ---------------------------------------------------------
    # STEP 2
    # Divide the balanced pool into Train and Dev.
    #
    # Each category is divided according to the requested
    # Train:Dev ratio as closely as integer counts permit.
    # ---------------------------------------------------------

    grouped = group_by_category(combined)

    train_set = []
    dev_set = []

    train_ratio = train_size / required

    rng = random.Random(seed + 1)

    for category in DCA_CATEGORIES:

        items = list(
            grouped[category]
        )

        if not items:
            continue

        rng.shuffle(items)

        category_train_count = round(
            len(items) * train_ratio
        )

        # Keep at least one Dev example whenever
        # the category has enough examples.
        if len(items) >= 2:

            category_train_count = min(
                category_train_count,
                len(items) - 1,
            )

        category_train_count = max(
            0,
            category_train_count,
        )

        train_set.extend(
            items[:category_train_count]
        )

        dev_set.extend(
            items[category_train_count:]
        )

    # ---------------------------------------------------------
    # STEP 3
    # Correct integer rounding so exact requested sizes
    # are returned.
    # ---------------------------------------------------------

    train_set, dev_set = _rebalance_exact_sizes(
        train_set=train_set,
        dev_set=dev_set,
        train_size=train_size,
        dev_size=dev_size,
        seed=seed + 2,
    )

    rng.shuffle(train_set)
    rng.shuffle(dev_set)

    # ---------------------------------------------------------
    # STEP 4
    # Verify no overlap.
    # ---------------------------------------------------------

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
            f"Train/Dev overlap detected: {overlap}"
        )

    if len(train_set) != train_size:
        raise RuntimeError(
            f"Expected Train={train_size}, "
            f"got {len(train_set)}."
        )

    if len(dev_set) != dev_size:
        raise RuntimeError(
            f"Expected Dev={dev_size}, "
            f"got {len(dev_set)}."
        )

    print_category_distribution(
        train_set,
        "FINAL TRUE-BALANCED GEPA TRAIN",
    )

    print_category_distribution(
        dev_set,
        "FINAL TRUE-BALANCED GEPA DEV",
    )

    return train_set, dev_set


def _rebalance_exact_sizes(
    train_set,
    dev_set,
    train_size,
    dev_size,
    seed,
):
    """
    Correct rounding differences while preserving
    category coverage as much as possible.
    """

    rng = random.Random(seed)

    # ---------------------------------------------------------
    # Train too large
    # Move examples from Train -> Dev.
    # Prefer categories that have multiple Train examples.
    # ---------------------------------------------------------

    while len(train_set) > train_size:

        grouped = group_by_category(train_set)

        candidates = [
            category
            for category in DCA_CATEGORIES
            if len(grouped[category]) > 1
        ]

        if not candidates:
            candidates = [
                category
                for category in DCA_CATEGORIES
                if grouped[category]
            ]

        category = rng.choice(candidates)

        example = grouped[category][-1]

        train_set.remove(example)
        dev_set.append(example)

    # ---------------------------------------------------------
    # Dev too large
    # Move examples from Dev -> Train.
    # ---------------------------------------------------------

    while len(dev_set) > dev_size:

        grouped = group_by_category(dev_set)

        candidates = [
            category
            for category in DCA_CATEGORIES
            if len(grouped[category]) > 1
        ]

        if not candidates:
            candidates = [
                category
                for category in DCA_CATEGORIES
                if grouped[category]
            ]

        category = rng.choice(candidates)

        example = grouped[category][-1]

        dev_set.remove(example)
        train_set.append(example)

    return train_set, dev_set