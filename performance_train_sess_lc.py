"""
Performance-Train + SESS-LC sampling for DCA.

Overall method
--------------

1. SESS-LC selects the Dev set using low-confidence DCA examples.
2. The selected Dev examples are removed from the Train candidate pool.
3. This module selects an initial Train set randomly.
4. GEPA generates candidate prompts.
5. Performance of the candidate prompts is recorded for the Train
   candidate pool.
6. A performance matrix is constructed.
7. Redundant Train examples are detected using Pearson correlation.
8. One redundant Train example is removed.
9. An unused candidate with maximum CLIP distance from the removed
   example is selected as the replacement.
10. Train size remains fixed.
11. Dev remains fixed.
12. Validation and Test are never used by this module.

Important
---------

SESS-LC Dev selection is performed by sess_lc_dca.py.

This module therefore receives ONLY the remaining Train candidate pool.

Expected pipeline:

    Original Train pool
            |
            v
       SESS-LC Dev
            |
            v
    Remove Dev examples
            |
            v
    Remaining Train pool
            |
            v
    Initial Train selection
            |
            v
           GEPA
            |
            v
    Performance matrix
            |
            v
    Redundancy detection
            |
            v
    CLIP-contrastive replacement
            |
            v
      Refined Train set

Dev is fixed throughout the experiment.
Validation/Test are untouched.
"""

import math
import random
from typing import Any, Dict, List


# ============================================================
# REUSE DCA PAIR EMBEDDING UTILITIES
# ============================================================

from ipomp_dca import (
    DCAPairEmbedder,
    _cosine_similarity,
)


# ============================================================
# EXAMPLE ID
# ============================================================

def _example_id(example: Any) -> str:
    """
    Return a stable identifier for a DCA example.

    Preference:
        1. pair_name
        2. id
        3. Python object id
    """

    pair_name = getattr(example, "pair_name", None)

    if pair_name is not None:
        return str(pair_name)

    example_id = getattr(example, "id", None)

    if example_id is not None:
        return str(example_id)

    return str(id(example))


# ============================================================
# PERFORMANCE TRAIN + SESS-LC SAMPLER
# ============================================================

class PerformanceTrainSessLCSampler:
    """
    Performance-based Train refinement for the
    Performance-Train + SESS-LC-DCA experiment.

    Dev behaviour
    -------------

    SESS-LC selects Dev BEFORE this sampler is created.

    Dev is therefore:

        - fixed
        - never modified
        - never reselected

    Train behaviour
    ---------------

    Round 0:

        Randomly select the requested Train size.

    Later rounds:

        1. GEPA candidate prompt performance is recorded.
        2. Performance vectors are constructed.
        3. Redundant Train examples are detected using
           Pearson correlation.
        4. One redundant example is removed.
        5. An unused candidate is selected using maximum
           CLIP distance from the removed example.
        6. The Train set size remains unchanged.

    This is a refinement strategy, not a complete GEPA optimizer.

    The outer extract_dca.py pipeline is responsible for:

        GEPA
        -> score examples
        -> record prompt scores
        -> call select_training_examples()
        -> run the next GEPA round

    Important:

        self.examples must be the SESS-LC remaining candidate pool.

    Therefore:

        SESS-LC Dev examples cannot enter Train.
    """

    def __init__(
        self,
        examples,
        train_size,
        correlation_threshold=0.90,
        hard_pool_fraction=0.50,
        seed=42,
    ):
        """
        Parameters
        ----------
        examples:
            Candidate Train pool AFTER SESS-LC Dev removal.

        train_size:
            Number of examples required in the Train set.

        correlation_threshold:
            Absolute Pearson correlation threshold used to identify
            redundant Train examples.

        hard_pool_fraction:
            Retained for CLI/API compatibility with the existing
            performance sampling arguments.

            It is NOT used by this redundancy + CLIP refinement
            strategy.

        seed:
            Random seed for initial Train selection.
        """

        self.examples = list(examples)

        self.train_size = int(train_size)

        self.correlation_threshold = float(
            correlation_threshold
        )

        # Retained for compatibility with existing CLI arguments.
        # This method does not use hard-pool selection.
        self.hard_pool_fraction = float(
            hard_pool_fraction
        )

        self.seed = int(seed)

        self.rng = random.Random(self.seed)

        # --------------------------------------------------------
        # Performance history
        # --------------------------------------------------------

        # Structure:
        #
        # {
        #     example_id: {
        #         prompt_id: score,
        #         ...
        #     },
        #     ...
        # }

        self.performance_matrix: Dict[
            str, Dict[str, float]
        ] = {}

        self.prompt_order: List[str] = []

        # Example lookup.
        self.example_by_id = {
            self._example_id(example): example
            for example in self.examples
        }

        # --------------------------------------------------------
        # CLIP embeddings
        # --------------------------------------------------------

        self.embeddings = {}

        self.embedder = None

        # --------------------------------------------------------
        # Current Train set
        # --------------------------------------------------------

        self.current_train_set = []

        # --------------------------------------------------------
        # Safety
        # --------------------------------------------------------

        if self.train_size <= 0:
            raise ValueError(
                f"train_size must be positive, got {self.train_size}"
            )

        if self.train_size > len(self.examples):
            raise ValueError(
                "Requested Train size is larger than the available "
                "candidate pool. "
                f"Pool={len(self.examples)}, "
                f"Train={self.train_size}"
            )

        if not 0.0 <= self.correlation_threshold <= 1.0:
            raise ValueError(
                "correlation_threshold must be between 0 and 1. "
                f"Got {self.correlation_threshold}"
            )

    # ============================================================
    # ID
    # ============================================================

    def _example_id(self, example):
        return _example_id(example)

    # ============================================================
    # RECORD GEPA PROMPT PERFORMANCE
    # ============================================================

    def record_prompt_scores(
        self,
        prompt_id,
        scored_examples,
    ):
        """
        Record performance for one GEPA candidate prompt.

        Expected scored_examples format:

            [
                {
                    "example": example,
                    "score": 0.95,
                    ...
                },
                ...
            ]

        Only the "example" and "score" fields are required.
        """

        prompt_id = str(prompt_id)

        if prompt_id not in self.prompt_order:
            self.prompt_order.append(prompt_id)

        for item in scored_examples:

            if "example" not in item:
                continue

            example = item["example"]

            score = float(
                item.get("score", 0.0)
            )

            # Protect against numerical issues.
            score = max(
                0.0,
                min(1.0, score),
            )

            example_id = self._example_id(example)

            self.performance_matrix.setdefault(
                example_id,
                {},
            )[prompt_id] = score

            self.example_by_id[
                example_id
            ] = example

        print(
            "\nPerformance scores recorded:"
        )
        print(
            f"  Prompt ID        : {prompt_id}"
        )
        print(
            f"  Examples scored  : {len(scored_examples)}"
        )
        print(
            f"  Total prompts    : {len(self.prompt_order)}"
        )

    # ============================================================
    # PERFORMANCE VECTOR
    # ============================================================

    def performance_vector(
        self,
        example_id,
    ):
        """
        Return the performance vector for one example.

        Example:

            Prompt 1 -> 0.80
            Prompt 2 -> 0.90
            Prompt 3 -> 0.75

        returns:

            [0.80, 0.90, 0.75]
        """

        scores = self.performance_matrix.get(
            example_id,
            {},
        )

        return [
            float(
                scores.get(prompt_id, 0.0)
            )
            for prompt_id in self.prompt_order
        ]

    # ============================================================
    # MEAN PERFORMANCE
    # ============================================================

    def mean_performance(
        self,
        example_id,
    ):
        """
        Return mean performance across recorded prompts.
        """

        vector = self.performance_vector(
            example_id
        )

        if not vector:
            return 0.0

        return sum(vector) / len(vector)

    # ============================================================
    # BUILD PERFORMANCE MATRIX
    # ============================================================

    def build_performance_matrix(
        self,
        evaluation_examples,
    ):
        """
        Build a performance matrix for the supplied examples.

        Returns:

            {
                example_id: [
                    score_prompt_1,
                    score_prompt_2,
                    ...
                ]
            }
        """

        return {
            self._example_id(example):
            self.performance_vector(
                self._example_id(example)
            )
            for example in evaluation_examples
        }

    # ============================================================
    # PEARSON CORRELATION
    # ============================================================

    @staticmethod
    def _correlation(
        a,
        b,
    ):
        """
        Calculate Pearson correlation.

        A minimum of two prompt observations is required.
        """

        if len(a) < 2 or len(b) < 2:
            return 0.0

        if len(a) != len(b):
            return 0.0

        mean_a = sum(a) / len(a)
        mean_b = sum(b) / len(b)

        numerator = sum(
            (x - mean_a) * (y - mean_b)
            for x, y in zip(a, b)
        )

        denominator = math.sqrt(
            sum(
                (x - mean_a) ** 2
                for x in a
            )
            *
            sum(
                (y - mean_b) ** 2
                for y in b
            )
        )

        # Constant vectors do not provide meaningful
        # Pearson correlation.
        if denominator == 0:
            return 0.0

        return numerator / denominator

    # ============================================================
    # EXAMPLE CORRELATION
    # ============================================================

    def example_correlation(
        self,
        example_a,
        example_b,
    ):
        """
        Calculate performance-pattern correlation between
        two examples.
        """

        return self._correlation(
            self.performance_vector(
                self._example_id(example_a)
            ),
            self.performance_vector(
                self._example_id(example_b)
            ),
        )

    # ============================================================
    # FIND REDUNDANT TRAIN EXAMPLES
    # ============================================================

    def find_redundant_examples(
        self,
        evaluation_examples,
    ):
        """
        Find Train example pairs whose performance patterns
        across GEPA prompts have absolute Pearson correlation
        greater than or equal to the configured threshold.
        """

        examples = list(
            evaluation_examples
        )

        matrix = self.build_performance_matrix(
            examples
        )

        redundant = []

        for i in range(
            len(examples)
        ):
            id_a = self._example_id(
                examples[i]
            )

            scores_a = matrix[id_a]

            for j in range(
                i + 1,
                len(examples),
            ):
                id_b = self._example_id(
                    examples[j]
                )

                scores_b = matrix[id_b]

                correlation = self._correlation(
                    scores_a,
                    scores_b,
                )

                if (
                    abs(correlation)
                    >= self.correlation_threshold
                ):
                    redundant.append(
                        (
                            examples[i],
                            examples[j],
                            correlation,
                        )
                    )

        return redundant

    # ============================================================
    # BUILD CLIP EMBEDDINGS
    # ============================================================

    def build_embeddings(self):
        """
        Build DCA pair embeddings using the same
        DCAPairEmbedder used by IPOMP.

        The embeddings are cached after the first call.
        """

        if self.embeddings:
            return self.embeddings

        print("\n" + "=" * 80)
        print(
            "BUILDING PERFORMANCE TRAIN CLIP EMBEDDINGS"
        )
        print("=" * 80)

        print(
            f"Examples to embed : {len(self.examples)}"
        )

        if self.embedder is None:
            self.embedder = DCAPairEmbedder()

        self.embeddings = (
            self.embedder.embed_examples(
                self.examples
            )
        )

        print(
            f"Embeddings created : {len(self.embeddings)}"
        )

        # --------------------------------------------------------
        # Safety check
        # --------------------------------------------------------

        missing_embeddings = []

        for example in self.examples:
            example_id = self._example_id(
                example
            )

            if example_id not in self.embeddings:
                missing_embeddings.append(
                    example_id
                )

        if missing_embeddings:

            preview = (
                missing_embeddings[:10]
            )

            raise RuntimeError(
                "Missing CLIP embeddings for "
                f"{len(missing_embeddings)} examples. "
                f"Examples: {preview}"
            )

        print("=" * 80)

        return self.embeddings

    # ============================================================
    # FIND CLIP-CONTRASTING REPLACEMENT
    # ============================================================

    def find_replacement(
        self,
        redundant_example,
        candidate_pool,
        evaluation_examples,
        protected_examples=None,
    ):
        """
        Find the most CLIP-contrasting unused candidate.

        distance = 1 - cosine_similarity

        The candidate with maximum distance is selected.

        Excluded candidates:

            - current Train examples
            - protected examples
        """

        evaluation_ids = {
            self._example_id(example)
            for example in evaluation_examples
        }

        protected_ids = set()

        if protected_examples is not None:
            protected_ids = {
                self._example_id(example)
                for example in protected_examples
            }

        candidates = [
            example
            for example in candidate_pool
            if self._example_id(example)
            not in evaluation_ids
            and self._example_id(example)
            not in protected_ids
        ]

        if not candidates:
            print(
                "\nNo unused candidate is available "
                "for Train replacement."
            )
            return None

        self.build_embeddings()

        target_id = self._example_id(
            redundant_example
        )

        if target_id not in self.embeddings:
            raise RuntimeError(
                "Missing embedding for redundant "
                f"example: {target_id}"
            )

        target_embedding = (
            self.embeddings[target_id]
        )

        best_candidate = None
        best_distance = -float("inf")

        for candidate in candidates:
            candidate_id = self._example_id(
                candidate
            )

            if candidate_id not in self.embeddings:
                continue

            candidate_embedding = (
                self.embeddings[candidate_id]
            )

            similarity = _cosine_similarity(
                target_embedding,
                candidate_embedding,
            )

            distance = 1.0 - similarity

            if distance > best_distance:
                best_distance = distance
                best_candidate = candidate

        if best_candidate is not None:
            print(
                "\nPerformance Train replacement search:"
            )

            print(
                f"  Removed     : {target_id}"
            )

            print(
                "  Replacement : "
                f"{self._example_id(best_candidate)}"
            )

            print(
                f"  CLIP distance: {best_distance:.4f}"
            )

        return best_candidate

    # ============================================================
    # ONE TRAIN REFINEMENT
    # ============================================================

    def refine_once(
        self,
        evaluation_examples,
        candidate_pool,
        protected_examples=None,
    ):
        """
        Perform exactly ONE:

            redundant example
                ->
            CLIP-contrastive replacement

        Train size is preserved.
        """

        evaluation_examples = list(
            evaluation_examples
        )

        redundant = (
            self.find_redundant_examples(
                evaluation_examples
            )
        )

        if not redundant:
            return (
                list(evaluation_examples),
                None,
            )

        # --------------------------------------------------------
        # Use the first redundant pair.
        # --------------------------------------------------------

        example_a, example_b, correlation = (
            redundant[0]
        )

        replacement = self.find_replacement(
            redundant_example=example_b,
            candidate_pool=candidate_pool,
            evaluation_examples=evaluation_examples,
            protected_examples=protected_examples,
        )

        if replacement is None:
            return (
                list(evaluation_examples),
                None,
            )

        replacement_id = self._example_id(
            replacement
        )

        # --------------------------------------------------------
        # Safety: replacement cannot already be in Train.
        # --------------------------------------------------------

        evaluation_ids = {
            self._example_id(example)
            for example in evaluation_examples
        }

        if replacement_id in evaluation_ids:
            raise RuntimeError(
                "Replacement example is already in "
                "the current Train set."
            )

        # --------------------------------------------------------
        # Remove redundant example.
        # --------------------------------------------------------

        removed_id = self._example_id(
            example_b
        )

        refined = [
            example
            for example in evaluation_examples
            if self._example_id(example)
            != removed_id
        ]

        # --------------------------------------------------------
        # Add replacement.
        # --------------------------------------------------------

        refined.append(
            replacement
        )

        # --------------------------------------------------------
        # Train size safety check.
        # --------------------------------------------------------

        if len(refined) != len(
            evaluation_examples
        ):
            raise RuntimeError(
                "Performance refinement changed "
                "the Train size."
            )

        return (
            refined,
            {
                "removed": removed_id,
                "replacement": replacement_id,
                "correlation": float(
                    correlation
                ),
            },
        )

    # ============================================================
    # MAIN TRAIN SELECTION / REFINEMENT
    # ============================================================

    def select_training_examples(self):
        """
        Select or refine the Train set.

        First call:

            Random initialization.

        Subsequent calls:

            If fewer than two prompt columns exist:
                keep current Train unchanged.

            Otherwise:
                detect redundancy and perform one
                CLIP-contrastive replacement.

        The Dev set is never modified here.
        """

        # --------------------------------------------------------
        # Validate Train size.
        # --------------------------------------------------------

        if self.train_size <= 0:
            raise ValueError(
                "train_size must be greater than 0."
            )

        if len(self.examples) < self.train_size:
            raise ValueError(
                "Candidate pool is smaller than "
                "requested training size. "
                f"Pool={len(self.examples)}, "
                f"requested={self.train_size}"
            )

        # ========================================================
        # ROUND 0: RANDOM INITIALIZATION
        # ========================================================

        if not self.current_train_set:
            self.current_train_set = (
                self.rng.sample(
                    self.examples,
                    self.train_size,
                )
            )

            info = {
                "method": "random_initialization",
                "selected_count": len(
                    self.current_train_set
                ),
                "prompt_count": len(
                    self.prompt_order
                ),
                "selected_ids": [
                    self._example_id(example)
                    for example in self.current_train_set
                ],
            }

            return (
                list(self.current_train_set),
                info,
            )

        # ========================================================
        # WAIT FOR AT LEAST TWO PROMPTS
        # ========================================================

        if len(self.prompt_order) < 2:
            info = {
                "method":
                    "performance_refinement_waiting_for_2_prompts",
                "selected_count": len(
                    self.current_train_set
                ),
                "prompt_count": len(
                    self.prompt_order
                ),
                "selected_ids": [
                    self._example_id(example)
                    for example in self.current_train_set
                ],
            }

            return (
                list(self.current_train_set),
                info,
            )

        # ========================================================
        # PERFORMANCE REFINEMENT
        # ========================================================

        print("\n" + "=" * 80)
        print(
            "PERFORMANCE-BASED TRAIN REFINEMENT"
        )
        print("=" * 80)

        print(
            "Current Train size         : "
            f"{len(self.current_train_set)}"
        )

        print(
            "Candidate replacement pool : "
            f"{len(self.examples)}"
        )

        print(
            "Recorded candidate prompts : "
            f"{len(self.prompt_order)}"
        )

        # --------------------------------------------------------
        # Performance matrix
        # --------------------------------------------------------

        performance_matrix = (
            self.build_performance_matrix(
                self.current_train_set
            )
        )

        print("\nPerformance matrix:")
        print(
            "  Train examples : "
            f"{len(performance_matrix)}"
        )

        print(
            "  Prompt columns : "
            f"{len(self.prompt_order)}"
        )

        # --------------------------------------------------------
        # Find redundant examples.
        # --------------------------------------------------------

        redundant = (
            self.find_redundant_examples(
                self.current_train_set
            )
        )

        print(
            "\nRedundant Train pairs found: "
            f"{len(redundant)}"
        )

        # --------------------------------------------------------
        # No redundancy.
        # --------------------------------------------------------

        if not redundant:
            print(
                "No redundant Train examples found."
            )

            print("=" * 80)

            info = {
                "method":
                    "performance_refinement_no_redundancy",
                "selected_count": len(
                    self.current_train_set
                ),
                "prompt_count": len(
                    self.prompt_order
                ),
                "redundant_pair_count": 0,
                "selected_ids": [
                    self._example_id(example)
                    for example in self.current_train_set
                ],
            }

            return (
                list(self.current_train_set),
                info,
            )

        # --------------------------------------------------------
        # Print redundant pairs.
        # --------------------------------------------------------

        print(
            "\nDetected redundant Train pairs:"
        )

        for index, (
            example_a,
            example_b,
            correlation,
        ) in enumerate(
            redundant,
            start=1,
        ):
            print(
                f"{index:3d}. "
                f"{self._example_id(example_a)} "
                "<-> "
                f"{self._example_id(example_b)} "
                f"| correlation={correlation:.4f}"
            )

        # ========================================================
        # ONE REPLACEMENT
        # ========================================================

        refined_train, refinement_info = (
            self.refine_once(
                evaluation_examples=self.current_train_set,
                candidate_pool=self.examples,
            )
        )

        # --------------------------------------------------------
        # Replacement unavailable.
        # --------------------------------------------------------

        if refinement_info is None:
            info = {
                "method":
                    "performance_refinement_no_replacement",
                "selected_count": len(
                    self.current_train_set
                ),
                "prompt_count": len(
                    self.prompt_order
                ),
                "redundant_pair_count": len(
                    redundant
                ),
                "selected_ids": [
                    self._example_id(example)
                    for example in self.current_train_set
                ],
            }

            return (
                list(self.current_train_set),
                info,
            )

        # --------------------------------------------------------
        # Update current Train set.
        # --------------------------------------------------------

        self.current_train_set = refined_train

        # --------------------------------------------------------
        # Safety checks.
        # --------------------------------------------------------

        if len(
            self.current_train_set
        ) != self.train_size:
            raise RuntimeError(
                "Performance Train refinement "
                "changed the requested Train size."
            )

        train_ids = {
            self._example_id(example)
            for example in self.current_train_set
        }

        if len(train_ids) != self.train_size:
            raise RuntimeError(
                "Duplicate examples detected in "
                "the refined Train set."
            )

        # --------------------------------------------------------
        # Build refinement information.
        # --------------------------------------------------------

        info = {
            "method":
                "performance_redundancy_clip_refinement",
            "selected_count": len(
                self.current_train_set
            ),
            "prompt_count": len(
                self.prompt_order
            ),
            "redundant_pair_count": len(
                redundant
            ),
            "removed": refinement_info[
                "removed"
            ],
            "replacement": refinement_info[
                "replacement"
            ],
            "correlation": refinement_info[
                "correlation"
            ],
            "selected_ids": list(train_ids),
        }

        print("\n" + "=" * 80)
        print("TRAIN REFINEMENT COMPLETE")
        print("=" * 80)

        print(
            "Removed example     : "
            f"{refinement_info['removed']}"
        )

        print(
            "Replacement example : "
            f"{refinement_info['replacement']}"
        )

        print(
            "Correlation         : "
            f"{refinement_info['correlation']:.4f}"
        )

        print(
            "Train size          : "
            f"{len(self.current_train_set)}"
        )

        print(
            "Prompt columns      : "
            f"{len(self.prompt_order)}"
        )

        print("=" * 80)

        return (
            list(self.current_train_set),
            info,
        )

    # ============================================================
    # PRINT PERFORMANCE MATRIX
    # ============================================================

    def print_performance_matrix(self):
        """
        Print the complete performance matrix.
        """

        print("\n" + "=" * 80)
        print("PERFORMANCE MATRIX")
        print("=" * 80)

        if not self.prompt_order:
            print(
                "No GEPA candidate prompts "
                "have been recorded yet."
            )

            print("=" * 80)
            return

        print(
            "Example".ljust(20),
            end="",
        )

        for prompt_id in self.prompt_order:
            print(
                str(prompt_id)[:15].ljust(17),
                end="",
            )

        print("Mean")
        print("-" * 80)

        for example in self.examples:
            example_id = self._example_id(
                example
            )

            print(
                example_id[:18].ljust(20),
                end="",
            )

            for prompt_id in self.prompt_order:
                score = (
                    self.performance_matrix
                    .get(
                        example_id,
                        {},
                    )
                    .get(
                        prompt_id,
                        0.0,
                    )
                )

                print(
                    f"{score:.3f}".ljust(17),
                    end="",
                )

            print(
                f"{self.mean_performance(example_id):.3f}"
            )

        print("=" * 80)

    # ============================================================
    # PRINT CURRENT TRAIN SELECTION
    # ============================================================

    def print_selection(
        self,
        selected,
        info=None,
    ):
        """
        Print the current Train selection.
        """

        print("\n" + "=" * 80)
        print(
            "PERFORMANCE-BASED TRAIN SAMPLING"
        )
        print("=" * 80)

        print(
            "Strategy            : "
            "Performance redundancy + "
            "CLIP contrastive refinement"
        )

        print(
            "Candidate pool      : "
            f"{len(self.examples)}"
        )

        print(
            "GEPA Train size     : "
            f"{len(selected)}"
        )

        print(
            "Prompt columns      : "
            f"{len(self.prompt_order)}"
        )

        print(
            "Correlation threshold: "
            f"{self.correlation_threshold}"
        )

        if info is not None:
            print(
                "Redundant pairs     : "
                f"{info.get('redundant_pair_count', 0)}"
            )

            if info.get("removed"):
                print(
                    "Removed             : "
                    f"{info['removed']}"
                )

            if info.get("replacement"):
                print(
                    "Replacement         : "
                    f"{info['replacement']}"
                )

        print("\nSelected examples:")

        for index, example in enumerate(
            selected,
            start=1,
        ):
            print(
                f"  {index:3d}. "
                f"{self._example_id(example)}"
            )

        print("=" * 80)

    # ============================================================
    # ACCESSORS
    # ============================================================

    def get_performance_matrix(self):
        """
        Return the recorded performance matrix.
        """

        return self.performance_matrix

    def get_prompt_count(self):
        """
        Return the number of recorded GEPA prompts.
        """

        return len(
            self.prompt_order
        )

    def get_current_train_set(self):
        """
        Return the current Train set.
        """

        return list(
            self.current_train_set
        )

    def get_unused_candidates(self):
        """
        Return candidates not currently present in Train.
        """

        train_ids = {
            self._example_id(example)
            for example in self.current_train_set
        }

        return [
            example
            for example in self.examples
            if self._example_id(example)
            not in train_ids
        ]