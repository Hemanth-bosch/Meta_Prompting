from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple


class PerformanceTrainSampler:
    """
    IPOMP-style performance-based training-set sampler for DCA.

    Core idea:

        GEPA candidate prompts
                ↓
        per-example performance
                ↓
        performance matrix
                ↓
        example-to-example correlation
                ↓
        identify redundant examples
                ↓
        prefer contrasting / complementary examples
                ↓
        final GEPA training set

    Important:
        - Only the candidate training pool is used.
        - Dev is NOT selected or modified here.
        - Validation and Test are never used.
        - The candidate pool itself remains unchanged.
    """

    def __init__(
        self,
        examples: List[Any],
        sample_size: int,
        correlation_threshold: float = 0.90,
        hard_pool_fraction: float = 0.50,
        seed: int = 42,
    ):
        self.examples = list(examples)
        self.sample_size = int(sample_size)

        self.correlation_threshold = float(
            correlation_threshold
        )

        self.hard_pool_fraction = float(
            hard_pool_fraction
        )

        self.rng = random.Random(seed)

        # --------------------------------------------------
        # Performance matrix
        #
        # {
        #     example_id: {
        #         prompt_id: score
        #     }
        # }
        #
        # Example:
        #
        # pair_1:
        #   prompt_1 -> 0.0
        #   prompt_2 -> 1.0
        #   prompt_3 -> 0.6
        #
        # pair_2:
        #   prompt_1 -> 0.0
        #   prompt_2 -> 0.0
        #   prompt_3 -> 0.2
        # --------------------------------------------------

        self.performance_matrix: Dict[
            str, Dict[str, float]
        ] = {}

        self.prompt_order: List[str] = []

        # Store example objects by stable ID.
        self.example_by_id: Dict[str, Any] = {
            self._example_id(example): example
            for example in self.examples
        }

    # ======================================================
    # ID
    # ======================================================

    def _example_id(self, example: Any) -> str:
        """
        Get a stable identifier for a DCA example.
        """

        pair_name = getattr(
            example,
            "pair_name",
            None,
        )

        if pair_name is not None:
            return str(pair_name)

        example_id = getattr(
            example,
            "id",
            None,
        )

        if example_id is not None:
            return str(example_id)

        return str(id(example))

    # ======================================================
    # Record GEPA prompt performance
    # ======================================================

    def record_prompt_scores(
        self,
        prompt_id: str,
        scored_examples: List[Dict[str, Any]],
    ) -> None:
        """
        Store the performance of one GEPA candidate prompt.

        scored_examples must contain:

            {
                "example": example,
                "score": score
            }

        This is equivalent to the performance-matrix idea
        used by IPOMP for the Dev set.
        """

        prompt_id = str(prompt_id)

        if prompt_id not in self.prompt_order:
            self.prompt_order.append(prompt_id)

        for item in scored_examples:

            example = item["example"]

            score = float(
                item.get("score", 0.0)
            )

            score = max(
                0.0,
                min(1.0, score),
            )

            example_id = self._example_id(
                example
            )

            if example_id not in self.performance_matrix:
                self.performance_matrix[
                    example_id
                ] = {}

            self.performance_matrix[
                example_id
            ][prompt_id] = score

            self.example_by_id[
                example_id
            ] = example

    # ======================================================
    # Performance vector
    # ======================================================

    def performance_vector(
        self,
        example_id: str,
    ) -> List[float]:
        """
        Return the performance vector of an example
        across all GEPA candidate prompts.

        Example:

            [0.0, 1.0, 0.0, 0.6]
        """

        scores = self.performance_matrix.get(
            example_id,
            {},
        )

        return [
            float(scores[prompt_id])
            for prompt_id in self.prompt_order
            if prompt_id in scores
        ]

    # ======================================================
    # Mean performance
    # ======================================================

    def mean_performance(
        self,
        example_id: str,
    ) -> float:
        """
        Mean model performance across candidate prompts.
        Lower means the example is generally harder.
        """

        vector = self.performance_vector(
            example_id
        )

        if not vector:
            return 0.0

        return sum(vector) / len(vector)

    # ======================================================
    # Pearson correlation
    # ======================================================

    def _pearson_correlation(
        self,
        x: List[float],
        y: List[float],
    ) -> float:
        """
        Calculate Pearson correlation.

        High positive correlation means the two examples
        behave similarly across candidate prompts.
        """

        if len(x) < 2 or len(y) < 2:
            return 0.0

        n = min(
            len(x),
            len(y),
        )

        x = x[:n]
        y = y[:n]

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum(
            (a - mean_x) * (b - mean_y)
            for a, b in zip(x, y)
        )

        denominator_x = math.sqrt(
            sum(
                (a - mean_x) ** 2
                for a in x
            )
        )

        denominator_y = math.sqrt(
            sum(
                (b - mean_y) ** 2
                for b in y
            )
        )

        denominator = (
            denominator_x * denominator_y
        )

        # Constant vectors.
        if denominator == 0:

            # If both vectors are exactly identical,
            # treat them as completely redundant.
            if x == y:
                return 1.0

            return 0.0

        return numerator / denominator

    # ======================================================
    # Example correlation
    # ======================================================

    def example_correlation(
        self,
        example_a: Any,
        example_b: Any,
    ) -> float:
        """
        Calculate correlation between two examples
        based on their performance vectors.
        """

        id_a = self._example_id(
            example_a
        )

        id_b = self._example_id(
            example_b
        )

        vector_a = self.performance_vector(
            id_a
        )

        vector_b = self.performance_vector(
            id_b
        )

        return self._pearson_correlation(
            vector_a,
            vector_b,
        )

    # ======================================================
    # Build hard candidate pool
    # ======================================================

    def _build_hard_pool(
        self,
    ) -> List[Any]:
        """
        Select the hardest examples first.

        Hardness is determined by LOW mean performance
        across the available GEPA prompts.
        """

        scored = []

        for example in self.examples:

            example_id = self._example_id(
                example
            )

            mean_score = self.mean_performance(
                example_id
            )

            scored.append(
                (
                    mean_score,
                    self.rng.random(),
                    example,
                )
            )

        # Lowest performance first.
        scored.sort(
            key=lambda x: (
                x[0],
                x[1],
            )
        )

        hard_pool_size = max(
            self.sample_size,
            int(
                len(scored)
                * self.hard_pool_fraction
            ),
        )

        hard_pool_size = min(
            hard_pool_size,
            len(scored),
        )

        return [
            item[2]
            for item in scored[
                :hard_pool_size
            ]
        ]

    # ======================================================
    # Select contrasting training examples
    # ======================================================

    def _select_contrasting_examples(
        self,
        candidates: List[Any],
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        IPOMP-style selection.

        1. Start from hard examples.
        2. Select one hard example.
        3. For every remaining candidate, calculate:
              - difficulty
              - correlation with selected examples
        4. Prefer examples that:
              - are difficult
              - are not redundant
              - behave differently across prompts
        """

        if not candidates:
            return [], {}

        target_size = min(
            self.sample_size,
            len(candidates),
        )

        # ----------------------------------------------
        # First example:
        # hardest example.
        # ----------------------------------------------

        candidates = list(candidates)

        candidates.sort(
            key=lambda example: (
                self.mean_performance(
                    self._example_id(
                        example
                    )
                ),
                self.rng.random(),
            )
        )

        selected: List[Any] = [
            candidates[0]
        ]

        remaining = [
            example
            for example in candidates[1:]
        ]

        # ----------------------------------------------
        # Greedy contrasting selection
        # ----------------------------------------------

        while (
            len(selected) < target_size
            and remaining
        ):

            best_candidate = None
            best_score = float("-inf")

            for candidate in remaining:

                candidate_id = self._example_id(
                    candidate
                )

                mean_score = self.mean_performance(
                    candidate_id
                )

                # Lower mean performance = harder.
                difficulty = 1.0 - mean_score

                # --------------------------------------
                # Find maximum correlation with anything
                # already selected.
                # --------------------------------------

                correlations = []

                for selected_example in selected:

                    correlation = (
                        self.example_correlation(
                            candidate,
                            selected_example,
                        )
                    )

                    correlations.append(
                        correlation
                    )

                max_correlation = (
                    max(correlations)
                    if correlations
                    else 0.0
                )

                # --------------------------------------
                # Redundancy:
                #
                # correlation close to 1
                # = very similar behavior.
                #
                # correlation close to 0
                # = different behavior.
                # --------------------------------------

                redundancy_penalty = max(
                    0.0,
                    max_correlation,
                )

                # --------------------------------------
                # Final selection score.
                #
                # Difficulty is rewarded.
                # Redundancy is penalized.
                # --------------------------------------

                selection_score = (
                    difficulty
                    - redundancy_penalty
                )

                # Strongly prefer examples below
                # the correlation threshold.
                if (
                    max_correlation
                    >= self.correlation_threshold
                ):
                    selection_score -= 0.25

                # Small deterministic random tie breaker.
                selection_score += (
                    self.rng.random() * 1e-6
                )

                if (
                    selection_score
                    > best_score
                ):
                    best_score = (
                        selection_score
                    )

                    best_candidate = (
                        candidate
                    )

            if best_candidate is None:
                break

            selected.append(
                best_candidate
            )

            remaining.remove(
                best_candidate
            )

        # ----------------------------------------------
        # Selection statistics
        # ----------------------------------------------

        selected_ids = {
            self._example_id(example)
            for example in selected
        }

        redundant_pairs = []

        for i in range(
            len(selected)
        ):

            for j in range(
                i + 1,
                len(selected),
            ):

                example_a = selected[i]
                example_b = selected[j]

                correlation = (
                    self.example_correlation(
                        example_a,
                        example_b,
                    )
                )

                if (
                    correlation
                    >= self.correlation_threshold
                ):
                    redundant_pairs.append(
                        {
                            "example_a":
                                self._example_id(
                                    example_a
                                ),
                            "example_b":
                                self._example_id(
                                    example_b
                                ),
                            "correlation":
                                correlation,
                        }
                    )

        mean_selected_score = (
            sum(
                self.mean_performance(
                    self._example_id(
                        example
                    )
                )
                for example in selected
            )
            / len(selected)
            if selected
            else 0.0
        )

        info = {
            "selected_count":
                len(selected),

            "candidate_count":
                len(candidates),

            "mean_selected_performance":
                mean_selected_score,

            "redundant_pairs":
                redundant_pairs,

            "redundant_pair_count":
                len(redundant_pairs),

            "correlation_threshold":
                self.correlation_threshold,

            "selected_ids":
                list(selected_ids),
        }

        return selected, info

    # ======================================================
    # Main selection function
    # ======================================================

    def select_training_examples(
        self,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Select the next GEPA training set.

        This function should be called after at least one
        GEPA candidate prompt has been evaluated.

        With multiple prompts, it becomes fully
        IPOMP-style because each example has a performance
        vector across prompts.
        """

        if self.sample_size <= 0:
            raise ValueError(
                "sample_size must be greater than 0."
            )

        if len(self.examples) < self.sample_size:
            raise ValueError(
                "Candidate pool is smaller than "
                "requested training size. "
                f"Pool={len(self.examples)}, "
                f"requested={self.sample_size}"
            )

        # ----------------------------------------------
        # No performance information yet.
        #
        # Random initialization.
        # ----------------------------------------------

        if not self.prompt_order:

            selected = self.rng.sample(
                self.examples,
                self.sample_size,
            )

            info = {
                "method":
                    "random_initialization",

                "selected_count":
                    len(selected),

                "prompt_count":
                    0,

                "selected_ids":
                    [
                        self._example_id(
                            example
                        )
                        for example in selected
                    ],
            }

            return selected, info

        # ----------------------------------------------
        # Build hard pool.
        # ----------------------------------------------

        hard_pool = (
            self._build_hard_pool()
        )

        # ----------------------------------------------
        # Apply IPOMP-style contrasting selection.
        # ----------------------------------------------

        selected, info = (
            self._select_contrasting_examples(
                hard_pool
            )
        )

        info["method"] = (
            "ipomp_style_performance"
        )

        info["prompt_count"] = (
            len(self.prompt_order)
        )

        return selected, info

    # ======================================================
    # Print performance matrix
    # ======================================================

    def print_performance_matrix(
        self,
    ) -> None:
        """
        Print the current performance matrix.
        """

        print(
            "\n"
            + "=" * 80
        )

        print(
            "PERFORMANCE MATRIX"
        )

        print(
            "=" * 80
        )

        if not self.prompt_order:

            print(
                "No GEPA candidate prompts "
                "have been recorded yet."
            )

            print(
                "=" * 80
            )

            return

        print(
            "Example".ljust(20),
            end=""
        )

        for prompt_id in self.prompt_order:

            print(
                prompt_id[:15].ljust(17),
                end=""
            )

        print(
            "Mean"
        )

        print(
            "-" * 80
        )

        for example in self.examples:

            example_id = self._example_id(
                example
            )

            vector = (
                self.performance_matrix.get(
                    example_id,
                    {}
                )
            )

            print(
                example_id[:18].ljust(20),
                end=""
            )

            for prompt_id in self.prompt_order:

                score = vector.get(
                    prompt_id,
                    0.0,
                )

                print(
                    f"{score:.3f}".ljust(17),
                    end=""
                )

            mean_score = (
                self.mean_performance(
                    example_id
                )
            )

            print(
                f"{mean_score:.3f}"
            )

        print(
            "=" * 80
        )

    # ======================================================
    # Print selection information
    # ======================================================

    def print_selection(
        self,
        selected: List[Any],
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Print final training-set selection.
        """

        print(
            "\n"
            + "=" * 80
        )

        print(
            "PERFORMANCE-BASED TRAIN SAMPLING"
        )

        print(
            "=" * 80
        )

        print(
            "Strategy            : "
            "IPOMP-style performance"
        )

        print(
            f"Candidate pool      : "
            f"{len(self.examples)}"
        )

        print(
            f"GEPA train size     : "
            f"{len(selected)}"
        )

        print(
            f"Prompt columns      : "
            f"{len(self.prompt_order)}"
        )

        print(
            f"Correlation threshold: "
            f"{self.correlation_threshold}"
        )

        if info is not None:

            print(
                f"Mean selected score : "
                f"{info.get('mean_selected_performance', 0.0):.4f}"
            )

            print(
                f"Redundant pairs     : "
                f"{info.get('redundant_pair_count', 0)}"
            )

        print(
            "\nSelected examples:"
        )

        for index, example in enumerate(
            selected,
            start=1,
        ):

            example_id = self._example_id(
                example
            )

            mean_score = (
                self.mean_performance(
                    example_id
                )
            )

            print(
                f"  {index:3d}. "
                f"{example_id:<30} "
                f"mean={mean_score:.4f}"
            )

        print(
            "=" * 80
        )

    # ======================================================
    # Accessors
    # ======================================================

    def get_performance_matrix(
        self,
    ) -> Dict[str, Dict[str, float]]:
        """
        Return the complete performance matrix.
        """

        return self.performance_matrix

    def get_prompt_count(
        self,
    ) -> int:
        return len(self.prompt_order)