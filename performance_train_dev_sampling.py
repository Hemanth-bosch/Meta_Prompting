from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

# Reuse the SAME IPOMP utilities and CLIP embedder.
from ipomp_dca import (
    DCAPairEmbedder,
    _cosine_similarity,
    _example_id,
)


# =========================================================
# PERFORMANCE TRAIN + DEV SELECTOR
# =========================================================

class PerformanceTrainDevSelector:
    """
    Performance-based Train + Dev sampler for DCA.

    This sampler uses the SAME performance-guided logic that
    IPOMP currently uses for the Dev set.

    Original IPOMP:

        Stage 1
            Train -> hard + diverse
            Dev   -> diverse

        Stage 2
            Dev   -> performance-guided refinement

    This sampler:

        Stage 1
            Initial Train / Dev sets are supplied.

        Stage 2
            Train -> performance-guided refinement
            Dev   -> performance-guided refinement

    Therefore the SAME Stage-2 logic is applied to BOTH sets.

    Performance-guided process:

        GEPA candidate prompts
                  |
                  v
        Performance matrix
                  |
                  v
        Pearson correlation
                  |
                  v
        Find redundant examples
                  |
                  v
        Remove one redundant example
                  |
                  v
        Search unused candidates
                  |
                  v
        CLIP contrasting replacement
                  |
                  v
        Updated Train / Dev set

    Important:
        - Train and Dev are handled separately.
        - Train size remains fixed.
        - Dev size remains fixed.
        - Validation is never used.
        - Test is never used.
        - Existing ipomp_dca.py is not modified.
        - Existing performance_train_sampling.py is not modified.
    """

    def __init__(
        self,
        examples: List[Any],
        train_size: int,
        dev_size: int,
        correlation_threshold: float = 0.90,
        seed: int = 42,
    ):
        self.examples = list(examples)

        self.train_size = int(train_size)
        self.dev_size = int(dev_size)

        self.correlation_threshold = float(
            correlation_threshold
        )

        self.rng = random.Random(seed)

        # -------------------------------------------------
        # CLIP embedder
        #
        # This is the SAME embedder used by IPOMP.
        # -------------------------------------------------

        self.embedder = DCAPairEmbedder()

        self.embeddings: Dict[
            str, List[float]
        ] = {}

        # -------------------------------------------------
        # Performance scores
        #
        # Same structure as ipomp_dca.py:
        #
        # prompt_id
        #       ->
        #       example_id
        #       ->
        #       score
        #
        # Example:
        #
        # {
        #     "prompt_1": {
        #         "pair_1": 0.0,
        #         "pair_2": 1.0
        #     },
        #
        #     "prompt_2": {
        #         "pair_1": 1.0,
        #         "pair_2": 0.0
        #     }
        # }
        # -------------------------------------------------

        self.prompt_scores: Dict[
            str, Dict[str, float]
        ] = {}

    # =====================================================
    # EMBEDDINGS
    # =====================================================

    def build_embeddings(self):
        """
        Build CLIP embeddings for the complete candidate pool.

        This uses the SAME DCAPairEmbedder from ipomp_dca.py.
        """

        if self.embeddings:
            return self.embeddings

        self.embeddings = (
            self.embedder.embed_examples(
                self.examples
            )
        )

        return self.embeddings

    # =====================================================
    # RECORD PROMPT PERFORMANCE
    # =====================================================

    def record_prompt_scores(
        self,
        prompt_id: str,
        scored_examples: List[Dict[str, Any]],
    ):
        """
        Record per-example performance for one GEPA prompt.

        Expected input:

            [
                {
                    "example": example,
                    "score": 0.8
                },
                {
                    "example": example,
                    "score": 0.3
                }
            ]

        This is the SAME performance-matrix concept used
        by IPOMP.
        """

        scores = {}

        for item in scored_examples:

            example = item[
                "example"
            ]

            score = float(
                item["score"]
            )

            scores[
                _example_id(example)
            ] = score

        self.prompt_scores[
            str(prompt_id)
        ] = scores

    # =====================================================
    # PERFORMANCE MATRIX
    # =====================================================

    def build_performance_matrix(
        self,
        evaluation_examples: List[Any],
    ):
        """
        Build:

            example_id -> [
                prompt1_score,
                prompt2_score,
                ...
            ]

        This is exactly the structure used by
        IPOMP Stage 2.
        """

        matrix = {}

        for example in evaluation_examples:

            example_id = _example_id(
                example
            )

            matrix[
                example_id
            ] = [
                self.prompt_scores[
                    prompt_id
                ].get(
                    example_id,
                    0.0,
                )
                for prompt_id in self.prompt_scores
            ]

        return matrix

    # =====================================================
    # PEARSON CORRELATION
    # =====================================================

    @staticmethod
    def _correlation(
        a,
        b,
    ):
        """
        SAME Pearson correlation calculation as IPOMP.
        """

        if len(a) < 2:
            return 0.0

        mean_a = (
            sum(a)
            / len(a)
        )

        mean_b = (
            sum(b)
            / len(b)
        )

        numerator = sum(
            (
                x - mean_a
            )
            *
            (
                y - mean_b
            )
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

        if denominator == 0:
            return 0.0

        return (
            numerator
            / denominator
        )

    # =====================================================
    # FIND REDUNDANT EXAMPLES
    # =====================================================

    def find_redundant_examples(
        self,
        evaluation_examples: List[Any],
    ):
        """
        SAME redundancy detection as IPOMP.

        Two examples are considered redundant when their
        performance patterns across GEPA candidate prompts
        have:

            abs(correlation)
            >= correlation_threshold
        """

        matrix = (
            self.build_performance_matrix(
                evaluation_examples
            )
        )

        redundant = []

        examples = list(
            evaluation_examples
        )

        for i in range(
            len(examples)
        ):

            id_a = _example_id(
                examples[i]
            )

            scores_a = matrix[
                id_a
            ]

            for j in range(
                i + 1,
                len(examples),
            ):

                id_b = _example_id(
                    examples[j]
                )

                scores_b = matrix[
                    id_b
                ]

                correlation = (
                    self._correlation(
                        scores_a,
                        scores_b,
                    )
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

    # =====================================================
    # FIND CONTRASTING REPLACEMENT
    # =====================================================

    def find_replacement(
        self,
        redundant_example,
        candidate_pool,
        evaluation_examples,
        protected_examples=None,
    ):
        """
        SAME replacement logic as IPOMP Stage 2.

        Steps:

            1. Remove examples already present in the
               current set.

            2. Optionally remove examples belonging to
               the other set.

            3. Compare CLIP embeddings.

            4. Calculate:

                   distance = 1 - cosine_similarity

            5. Select the candidate with the maximum
               distance.

        Therefore the replacement is the most contrasting
        candidate according to the SAME CLIP mechanism
        used by IPOMP.
        """

        evaluation_ids = {
            _example_id(example)
            for example
            in evaluation_examples
        }

        protected_ids = set()

        if protected_examples is not None:

            protected_ids = {
                _example_id(example)
                for example
                in protected_examples
            }

        candidates = [
            example
            for example
            in candidate_pool
            if _example_id(example)
            not in evaluation_ids
            and _example_id(example)
            not in protected_ids
        ]

        if not candidates:
            return None

        # -------------------------------------------------
        # Build CLIP embeddings
        # -------------------------------------------------

        self.build_embeddings()

        target_id = _example_id(
            redundant_example
        )

        target_embedding = (
            self.embeddings[
                target_id
            ]
        )

        # -------------------------------------------------
        # Find most contrasting candidate
        # -------------------------------------------------

        best_candidate = None
        best_distance = -1.0

        for candidate in candidates:

            candidate_id = (
                _example_id(
                    candidate
                )
            )

            candidate_embedding = (
                self.embeddings[
                    candidate_id
                ]
            )

            similarity = (
                _cosine_similarity(
                    target_embedding,
                    candidate_embedding,
                )
            )

            distance = (
                1.0
                - similarity
            )

            if (
                distance
                > best_distance
            ):

                best_distance = (
                    distance
                )

                best_candidate = (
                    candidate
                )

        if (
            best_candidate
            is not None
        ):

            print(
                "\nPerformance Train+Dev "
                "replacement search:"
            )

            print(
                f"  Removed     : "
                f"{target_id}"
            )

            print(
                f"  Replacement : "
                f"{_example_id(best_candidate)}"
            )

            print(
                f"  Distance    : "
                f"{best_distance:.4f}"
            )

        return best_candidate

    # =====================================================
    # ONE REFINEMENT
    # =====================================================

    def refine_once(
        self,
        evaluation_examples,
        candidate_pool,
        protected_examples=None,
    ):
        """
        Perform ONE IPOMP Stage-2 refinement.

        This method is used for BOTH:

            Train
            Dev
        """

        redundant = (
            self.find_redundant_examples(
                evaluation_examples
            )
        )

        if not redundant:

            return (
                list(
                    evaluation_examples
                ),
                None,
            )

        # -------------------------------------------------
        # Take the first redundant pair.
        #
        # SAME as IPOMP.
        # -------------------------------------------------

        (
            example_a,
            example_b,
            correlation,
        ) = redundant[0]

        # -------------------------------------------------
        # Replace example_b.
        #
        # SAME as IPOMP.
        # -------------------------------------------------

        replacement = (
            self.find_replacement(
                redundant_example=(
                    example_b
                ),
                candidate_pool=(
                    candidate_pool
                ),
                evaluation_examples=(
                    evaluation_examples
                ),
                protected_examples=(
                    protected_examples
                ),
            )
        )

        if replacement is None:

            return (
                list(
                    evaluation_examples
                ),
                None,
            )

        # -------------------------------------------------
        # Remove redundant example.
        # -------------------------------------------------

        refined = [
            example
            for example
            in evaluation_examples
            if _example_id(
                example
            )
            != _example_id(
                example_b
            )
        ]

        # -------------------------------------------------
        # Add contrasting replacement.
        # -------------------------------------------------

        refined.append(
            replacement
        )

        # -------------------------------------------------
        # Verify size.
        # -------------------------------------------------

        if (
            len(refined)
            != len(
                evaluation_examples
            )
        ):

            raise RuntimeError(
                "Performance refinement "
                "changed the set size."
            )

        return (
            refined,
            {
                "removed": _example_id(
                    example_b
                ),
                "replacement": _example_id(
                    replacement
                ),
                "correlation": (
                    correlation
                ),
            },
        )

    # =====================================================
    # TRAIN REFINEMENT
    # =====================================================

    def run_train_refinement(
        self,
        train_set,
        candidate_pool,
        protected_examples=None,
    ):
        """
        Apply the SAME IPOMP Stage-2 Dev logic to Train.

        Train:

            performance matrix
                    ↓
            redundant pairs
                    ↓
            remove redundant example
                    ↓
            CLIP contrasting replacement
                    ↓
            updated Train

        Train size remains unchanged.
        """

        print("\n" + "=" * 80)

        print(
            "PERFORMANCE-BASED TRAIN REFINEMENT"
        )

        print("=" * 80)

        print(
            f"Current Train size        : "
            f"{len(train_set)}"
        )

        print(
            f"Candidate replacement pool: "
            f"{len(candidate_pool)}"
        )

        print(
            f"Recorded candidate prompts: "
            f"{len(self.prompt_scores)}"
        )

        # -------------------------------------------------
        # Need at least two prompts.
        # -------------------------------------------------

        if len(
            self.prompt_scores
        ) < 2:

            print(
                "Not enough candidate prompts "
                "for performance refinement."
            )

            print(
                "At least 2 prompt evaluations "
                "are required."
            )

            return (
                list(train_set),
                None,
            )

        # -------------------------------------------------
        # Build performance matrix.
        # -------------------------------------------------

        performance_matrix = (
            self.build_performance_matrix(
                train_set
            )
        )

        print(
            "\nPerformance matrix:"
        )

        print(
            f"  Train examples : "
            f"{len(performance_matrix)}"
        )

        print(
            f"  Prompts        : "
            f"{len(self.prompt_scores)}"
        )

        # -------------------------------------------------
        # Find redundant examples.
        # -------------------------------------------------

        redundant = (
            self.find_redundant_examples(
                train_set
            )
        )

        print(
            f"\nRedundant Train pairs found: "
            f"{len(redundant)}"
        )

        if not redundant:

            print(
                "No redundant Train examples "
                "found."
            )

            print("=" * 80)

            return (
                list(train_set),
                None,
            )

        # -------------------------------------------------
        # Show redundancy.
        # -------------------------------------------------

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
                f"{_example_id(example_a)} "
                f"<-> "
                f"{_example_id(example_b)} "
                f"| correlation="
                f"{correlation:.4f}"
            )

        # -------------------------------------------------
        # Perform SAME IPOMP refinement.
        # -------------------------------------------------

        (
            refined_train,
            refinement_info,
        ) = self.refine_once(
            evaluation_examples=train_set,
            candidate_pool=candidate_pool,
            protected_examples=protected_examples,
        )

        if refinement_info is None:

            print(
                "\nRedundancy was found, "
                "but no replacement was available."
            )

            print("=" * 80)

            return (
                list(train_set),
                None,
            )

        # -------------------------------------------------
        # Verify size.
        # -------------------------------------------------

        if (
            len(refined_train)
            != len(train_set)
        ):

            raise RuntimeError(
                "Performance Train refinement "
                "changed Train size."
            )

        # -------------------------------------------------
        # Print result.
        # -------------------------------------------------

        print(
            "\nTRAIN REFINEMENT"
        )

        print(
            "-" * 80
        )

        print(
            f"Removed example     : "
            f"{refinement_info['removed']}"
        )

        print(
            f"Replacement example : "
            f"{refinement_info['replacement']}"
        )

        print(
            f"Correlation         : "
            f"{refinement_info['correlation']:.4f}"
        )

        print(
            f"Train size          : "
            f"{len(refined_train)}"
        )

        print(
            "-" * 80
        )

        print(
            "Performance-guided "
            "Train refinement complete."
        )

        print("=" * 80)

        return (
            refined_train,
            refinement_info,
        )

    # =====================================================
    # DEV REFINEMENT
    # =====================================================

    def run_dev_refinement(
        self,
        dev_set,
        candidate_pool,
        protected_examples=None,
    ):
        """
        Apply the ORIGINAL IPOMP Stage-2 logic to Dev.

        This keeps the Dev behaviour aligned with
        ipomp_dca.py.
        """

        print("\n" + "=" * 80)

        print(
            "PERFORMANCE-BASED DEV REFINEMENT"
        )

        print("=" * 80)

        print(
            f"Current Dev size          : "
            f"{len(dev_set)}"
        )

        print(
            f"Candidate replacement pool: "
            f"{len(candidate_pool)}"
        )

        print(
            f"Recorded candidate prompts: "
            f"{len(self.prompt_scores)}"
        )

        # -------------------------------------------------
        # Need at least two prompts.
        # -------------------------------------------------

        if len(
            self.prompt_scores
        ) < 2:

            print(
                "Not enough candidate prompts "
                "for Stage 2."
            )

            print(
                "At least 2 prompt evaluations "
                "are required."
            )

            return (
                list(dev_set),
                None,
            )

        # -------------------------------------------------
        # Build performance matrix.
        # -------------------------------------------------

        performance_matrix = (
            self.build_performance_matrix(
                dev_set
            )
        )

        print(
            "\nPerformance matrix:"
        )

        print(
            f"  Dev examples : "
            f"{len(performance_matrix)}"
        )

        print(
            f"  Prompts      : "
            f"{len(self.prompt_scores)}"
        )

        # -------------------------------------------------
        # Find redundant examples.
        # -------------------------------------------------

        redundant = (
            self.find_redundant_examples(
                dev_set
            )
        )

        print(
            f"\nRedundant Dev pairs found: "
            f"{len(redundant)}"
        )

        if not redundant:

            print(
                "No redundant Dev examples "
                "found."
            )

            print("=" * 80)

            return (
                list(dev_set),
                None,
            )

        # -------------------------------------------------
        # Show detected redundancy.
        # -------------------------------------------------

        print(
            "\nDetected redundant Dev pairs:"
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
                f"{_example_id(example_a)} "
                f"<-> "
                f"{_example_id(example_b)} "
                f"| correlation="
                f"{correlation:.4f}"
            )

        # -------------------------------------------------
        # Perform SAME IPOMP refinement.
        # -------------------------------------------------

        (
            refined_dev,
            refinement_info,
        ) = self.refine_once(
            evaluation_examples=dev_set,
            candidate_pool=candidate_pool,
            protected_examples=protected_examples,
        )

        if refinement_info is None:

            print(
                "\nStage 2 found redundancy, "
                "but no replacement was available."
            )

            print("=" * 80)

            return (
                list(dev_set),
                None,
            )

        # -------------------------------------------------
        # Verify size.
        # -------------------------------------------------

        if (
            len(refined_dev)
            != len(dev_set)
        ):

            raise RuntimeError(
                "Performance Dev refinement "
                "changed Dev size."
            )

        # -------------------------------------------------
        # Print result.
        # -------------------------------------------------

        print(
            "\nDEV REFINEMENT"
        )

        print(
            "-" * 80
        )

        print(
            f"Removed example     : "
            f"{refinement_info['removed']}"
        )

        print(
            f"Replacement example : "
            f"{refinement_info['replacement']}"
        )

        print(
            f"Correlation         : "
            f"{refinement_info['correlation']:.4f}"
        )

        print(
            f"Dev size            : "
            f"{len(refined_dev)}"
        )

        print(
            "-" * 80
        )

        print(
            "Performance-guided "
            "Dev refinement complete."
        )

        print("=" * 80)

        return (
            refined_dev,
            refinement_info,
        )

    # =====================================================
    # TRAIN + DEV REFINEMENT
    # =====================================================

    def refine_train_dev(
        self,
        train_set,
        dev_set,
        train_candidate_pool=None,
        dev_candidate_pool=None,
        prevent_overlap=True,
    ):
        """
        Apply performance-based refinement to BOTH Train
        and Dev.

        Train:
            Uses the SAME IPOMP Dev Stage-2 logic.

        Dev:
            Uses the ORIGINAL IPOMP Stage-2 logic.

        Validation and Test are never passed into this method.

        By default, Train and Dev overlap is prevented.
        """

        train_set = list(
            train_set
        )

        dev_set = list(
            dev_set
        )

        if train_candidate_pool is None:

            train_candidate_pool = list(
                self.examples
            )

        if dev_candidate_pool is None:

            dev_candidate_pool = list(
                self.examples
            )

        print("\n" + "=" * 80)

        print(
            "PERFORMANCE TRAIN + DEV SAMPLING"
        )

        print("=" * 80)

        print(
            "Same IPOMP performance-guided "
            "logic is applied to Train and Dev."
        )

        print(
            f"Initial Train size : "
            f"{len(train_set)}"
        )

        print(
            f"Initial Dev size   : "
            f"{len(dev_set)}"
        )

        # -------------------------------------------------
        # TRAIN
        # -------------------------------------------------
        #
        # Do not allow Train replacement candidates
        # from the current Dev set.
        # -------------------------------------------------

        train_protected = (
            dev_set
            if prevent_overlap
            else None
        )

        (
            train_set,
            train_info,
        ) = self.run_train_refinement(
            train_set=train_set,
            candidate_pool=train_candidate_pool,
            protected_examples=train_protected,
        )

        # -------------------------------------------------
        # DEV
        # -------------------------------------------------
        #
        # Do not allow Dev replacement candidates
        # from the updated Train set.
        # -------------------------------------------------

        dev_protected = (
            train_set
            if prevent_overlap
            else None
        )

        (
            dev_set,
            dev_info,
        ) = self.run_dev_refinement(
            dev_set=dev_set,
            candidate_pool=dev_candidate_pool,
            protected_examples=dev_protected,
        )

        # -------------------------------------------------
        # Final overlap check.
        # -------------------------------------------------

        train_ids = {
            _example_id(example)
            for example in train_set
        }

        dev_ids = {
            _example_id(example)
            for example in dev_set
        }

        overlap = (
            train_ids
            &
            dev_ids
        )

        if overlap:

            raise RuntimeError(
                "Train/Dev overlap detected "
                "after performance refinement: "
                f"{sorted(overlap)}"
            )

        # -------------------------------------------------
        # Final result.
        # -------------------------------------------------

        print("\n" + "=" * 80)

        print(
            "PERFORMANCE TRAIN + DEV COMPLETE"
        )

        print("=" * 80)

        print(
            f"Final Train size : "
            f"{len(train_set)}"
        )

        print(
            f"Final Dev size   : "
            f"{len(dev_set)}"
        )

        print(
            f"Train/Dev overlap: "
            f"{len(overlap)}"
        )

        print(
            "Validation/Test were not used."
        )

        print("=" * 80)

        return (
            train_set,
            dev_set,
            {
                "train": train_info,
                "dev": dev_info,
            },
        )

    # =====================================================
    # PRINT PERFORMANCE MATRIX
    # =====================================================

    def print_performance_matrix(
        self,
    ):
        """
        Print the complete performance matrix.
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

        if not self.prompt_scores:

            print(
                "No prompt performance recorded."
            )

            print(
                "=" * 80
            )

            return

        prompt_ids = list(
            self.prompt_scores.keys()
        )

        print(
            "Example".ljust(35),
            end="",
        )

        for prompt_id in prompt_ids:

            print(
                str(prompt_id)[:15].ljust(18),
                end="",
            )

        print()

        print(
            "-" * 80
        )

        for example in self.examples:

            example_id = _example_id(
                example
            )

            print(
                example_id[:33].ljust(35),
                end="",
            )

            for prompt_id in prompt_ids:

                score = (
                    self.prompt_scores[
                        prompt_id
                    ].get(
                        example_id,
                        0.0,
                    )
                )

                print(
                    f"{score:.3f}".ljust(18),
                    end="",
                )

            print()

        print(
            "=" * 80
        )


# =========================================================
# CONVENIENCE FUNCTION
# =========================================================

def performance_train_dev_sampling(
    examples,
    train_set,
    dev_set,
    train_size,
    dev_size,
    prompt_scores,
    correlation_threshold=0.90,
    seed=42,
    train_candidate_pool=None,
    dev_candidate_pool=None,
    prevent_overlap=True,
):
    """
    Convenience function.

    Example usage:

        train_set, dev_set, info = (
            performance_train_dev_sampling(
                examples=candidate_train_pool,
                train_set=train_set,
                dev_set=dev_set,
                train_size=50,
                dev_size=30,
                prompt_scores=prompt_scores,
            )
        )

    prompt_scores must have the same structure as
    self.prompt_scores:

        {
            "prompt_1": {
                "pair_1": 0.8,
                "pair_2": 0.2,
            },

            "prompt_2": {
                "pair_1": 0.9,
                "pair_2": 0.1,
            }
        }
    """

    selector = (
        PerformanceTrainDevSelector(
            examples=examples,
            train_size=train_size,
            dev_size=dev_size,
            correlation_threshold=(
                correlation_threshold
            ),
            seed=seed,
        )
    )

    # Load existing performance information.
    selector.prompt_scores = dict(
        prompt_scores
    )

    return selector.refine_train_dev(
        train_set=train_set,
        dev_set=dev_set,
        train_candidate_pool=(
            train_candidate_pool
        ),
        dev_candidate_pool=(
            dev_candidate_pool
        ),
        prevent_overlap=(
            prevent_overlap
        ),
    )