"""
Performance-based class-adaptive sampling for DCA.

Purpose
-------
Dynamically changes the TRAINING-SET CLASS DISTRIBUTION according to the
current model's error on each DCA category.

Important:
- This file does not modify the existing random/hard/balanced/adaptive/
  IPOMP/APEX/performance_train samplers.
- The caller supplies the DCA scoring function, so this file does not import
  extract_dca.py and avoids circular imports.
- Dev/validation/test are not handled here. The caller keeps Dev fixed and
  passes only the candidate training pool to this sampler.

Expected category attribute
---------------------------
The DCA examples in extract_dca.py are stratified using
"expected_human_label_category". This sampler therefore checks that field
first, then falls back to common alternatives: human_label_category,
category, and expected_category.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable


class PerformanceClassAdaptiveSampler:
    """Select a fixed-size train set by adapting class allocation to error."""

    def __init__(
        self,
        examples: list[Any],
        sample_size: int,
        seed: int = 42,
        min_class_fraction: float = 0.05,
        max_class_fraction: float = 0.50,
    ) -> None:
        if sample_size <= 0:
            raise ValueError("sample_size must be > 0")
        if len(examples) < sample_size:
            raise ValueError(
                f"Candidate pool has {len(examples)} examples, "
                f"but sample_size={sample_size}."
            )
        if not 0.0 <= min_class_fraction <= 1.0:
            raise ValueError("min_class_fraction must be between 0 and 1")
        if not 0.0 <= max_class_fraction <= 1.0:
            raise ValueError("max_class_fraction must be between 0 and 1")
        if min_class_fraction > max_class_fraction:
            raise ValueError("min_class_fraction cannot exceed max_class_fraction")

        self.examples = list(examples)
        self.sample_size = sample_size
        self.rng = random.Random(seed)
        self.seed = seed
        self.min_class_fraction = min_class_fraction
        self.max_class_fraction = max_class_fraction
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def _category(example: Any) -> str:
        """Get the human/DCA category from a DSPy example."""
        for field in (
            "expected_human_label_category",
            "human_label_category",
            "category",
            "expected_category",
        ):
            value = getattr(example, field, None)
            if value is not None and str(value).strip():
                return str(value).strip()

        raise AttributeError(
            "Could not find a DCA category on the example. Expected one of: "
            "expected_human_label_category, human_label_category, category, "
            "expected_category."
        )

    @staticmethod
    def _example_id(example: Any) -> str:
        return str(getattr(example, "pair_name", id(example)))

    def _group_by_class(self) -> dict[str, list[Any]]:
        groups: dict[str, list[Any]] = defaultdict(list)
        for example in self.examples:
            groups[self._category(example)].append(example)
        return dict(groups)

    def _random_initial_selection(self) -> tuple[list[Any], dict[str, Any]]:
        """Round 1: random selection, while keeping class counts visible."""
        selected = self.rng.sample(self.examples, self.sample_size)
        counts = defaultdict(int)
        for example in selected:
            counts[self._category(example)] += 1

        info = {
            "round": 1,
            "strategy": "initial_random",
            "class_counts": dict(counts),
            "class_distribution": {
                k: v / len(selected) for k, v in counts.items()
            },
            "selected_ids": [self._example_id(x) for x in selected],
        }
        self.history.append(info)
        return selected, info

    def score_classes(
        self,
        comparator: Any,
        metric_fn: Callable[[Any, Any], float],
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        """Score every candidate and aggregate performance/error by class."""
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        example_scores: list[dict[str, Any]] = []

        print("\n" + "=" * 80)
        print("PERFORMANCE-BASED CLASS SCORING")
        print("=" * 80)

        for idx, example in enumerate(self.examples):
            category = self._category(example)
            try:
                prediction = comparator(
                    image_original=example.image_original,
                    image_reimage=example.image_reimage,
                )
                score = float(metric_fn(example, prediction))
                score = max(0.0, min(1.0, score))
                error = 1.0 - score
            except Exception as exc:
                # A failed model call is treated as maximum error. This keeps
                # the class visible to the adaptive mechanism instead of
                # silently dropping it.
                print(
                    f"[WARNING] Could not score example {idx} "
                    f"({self._example_id(example)}): {exc}"
                )
                score = 0.0
                error = 1.0

            item = {
                "example": example,
                "category": category,
                "score": score,
                "error": error,
                "index": idx,
            }
            example_scores.append(item)
            groups[category].append(item)

        class_stats: dict[str, dict[str, Any]] = {}
        for category, items in groups.items():
            avg_score = sum(x["score"] for x in items) / len(items)
            avg_error = 1.0 - avg_score
            class_stats[category] = {
                "available": len(items),
                "average_score": avg_score,
                "average_error": avg_error,
                "weight": avg_error,
            }

        print("\nClass performance:")
        print("-" * 80)
        for category in sorted(class_stats):
            stats = class_stats[category]
            print(
                f"{category:20s} | available={stats['available']:3d} "
                f"| score={stats['average_score']:.4f} "
                f"| error={stats['average_error']:.4f}"
            )

        return example_scores, class_stats

    def _allocate_counts(
        self,
        class_stats: dict[str, dict[str, Any]],
    ) -> dict[str, int]:
        """Convert class errors into an exact integer allocation."""
        classes = list(class_stats)
        if not classes:
            raise ValueError("No DCA classes were found in the candidate pool.")

        # Add a tiny floor so a class with zero current error does not become
        # permanently impossible to sample.
        epsilon = 0.01
        weights = {
            c: max(epsilon, float(class_stats[c]["average_error"]))
            for c in classes
        }
        total_weight = sum(weights.values())
        raw = {
            c: self.sample_size * weights[c] / total_weight
            for c in classes
        }

        # Capacity: never request more examples than exist in the pool.
        min_count = max(0, int(self.sample_size * self.min_class_fraction))
        max_count = min(
            self.sample_size,
            max(1, int(self.sample_size * self.max_class_fraction)),
        )

        # If there are too many classes for the requested sample size, a hard
        # minimum for every class is impossible. In that case, use zero minimum
        # and let weighted allocation decide.
        if min_count * len(classes) > self.sample_size:
            min_count = 0

        counts = {c: min_count for c in classes}

        # Respect class availability.
        for c in classes:
            available = int(class_stats[c]["available"])
            counts[c] = min(counts[c], available, max_count)

        remaining = self.sample_size - sum(counts.values())

        # Largest-remainder allocation with availability/max caps.
        while remaining > 0:
            candidates = [
                c
                for c in classes
                if counts[c] < min(
                    int(class_stats[c]["available"]), max_count
                )
            ]
            if not candidates:
                break

            best = max(
                candidates,
                key=lambda c: (
                    raw[c] - counts[c],
                    weights[c],
                    -classes.index(c),
                ),
            )
            counts[best] += 1
            remaining -= 1

        # If caps caused unused slots, fill from classes with remaining
        # capacity, ignoring max cap only when mathematically necessary.
        if remaining > 0:
            candidates = [
                c for c in classes
                if counts[c] < int(class_stats[c]["available"])
            ]
            while remaining > 0 and candidates:
                best = max(
                    candidates,
                    key=lambda c: (weights[c], -classes.index(c)),
                )
                counts[best] += 1
                remaining -= 1
                candidates = [
                    c for c in candidates
                    if counts[c] < int(class_stats[c]["available"])
                ]

        if sum(counts.values()) != self.sample_size:
            raise RuntimeError(
                "Could not allocate the requested training size. "
                f"Requested={self.sample_size}, allocated={sum(counts.values())}."
            )

        return counts

    def select_next_training_set(
        self,
        comparator: Any,
        metric_fn: Callable[[Any, Any], float],
        round_number: int,
    ) -> tuple[list[Any], dict[str, Any]]:
        """Score classes and build the next dynamically adapted train set."""
        _, class_stats = self.score_classes(comparator, metric_fn)
        target_counts = self._allocate_counts(class_stats)
        groups = self._group_by_class()

        selected: list[Any] = []
        for category, count in target_counts.items():
            available = groups[category]
            if count > len(available):
                count = len(available)
            selected.extend(self.rng.sample(available, count))

        # This should already be exact, but guard against implementation bugs.
        if len(selected) != self.sample_size:
            raise RuntimeError(
                f"Adaptive sampler selected {len(selected)} examples; "
                f"expected {self.sample_size}."
            )

        self.rng.shuffle(selected)

        info = {
            "round": round_number,
            "strategy": "performance_class_adaptive",
            "class_counts": target_counts,
            "class_distribution": {
                c: n / self.sample_size
                for c, n in target_counts.items()
            },
            "class_stats": class_stats,
            "selected_ids": [self._example_id(x) for x in selected],
        }
        self.history.append(info)

        print("\n" + "=" * 80)
        print("NEXT ADAPTIVE TRAINING DISTRIBUTION")
        print("=" * 80)
        print(
            f"Training size : {self.sample_size}\n"
            f"Round        : {round_number}"
        )
        print("\nClass allocation:")
        print("-" * 80)
        for category in sorted(target_counts):
            stats = class_stats[category]
            count = target_counts[category]
            print(
                f"{category:20s} | error={stats['average_error']:.4f} "
                f"| next_train={count:3d} "
                f"({count / self.sample_size:.1%})"
            )
        print("=" * 80)

        return selected, info

    def select_initial_training_set(self) -> tuple[list[Any], dict[str, Any]]:
        """Public Round-1 selector."""
        return self._random_initial_selection()
