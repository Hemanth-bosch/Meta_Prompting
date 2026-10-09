# ipomp_dca.py

from __future__ import annotations

import math
import random
from typing import Any, Optional


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def _example_id(example: Any) -> str:
    """Get a stable ID for a DCA example."""

    pair_name = getattr(example, "pair_name", None)

    if pair_name is not None:
        return str(pair_name)

    example_id = getattr(example, "id", None)

    if example_id is not None:
        return str(example_id)

    return str(id(example))


def _cosine_similarity(a, b) -> float:
    numerator = sum(
        x * y for x, y in zip(a, b)
    )

    denominator = (
        math.sqrt(sum(x * x for x in a))
        * math.sqrt(sum(y * y for y in b))
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


# =========================================================
# CLIP PAIR EMBEDDER
# =========================================================

class DCAPairEmbedder:
    """
    Creates an embedding for a DCA image pair.

    Each example contains:
        image_original
        image_reimage
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device

        self.processor = None
        self.model = None
        self.torch = None

    # =====================================================
    # LOAD CLIP MODEL
    # =====================================================

    def _load_model(self):

        if self.model is not None:
            return

        try:
            import torch
            from transformers import (
                CLIPModel,
                CLIPProcessor,
            )
        except ImportError as exc:
            raise ImportError(
                "IPOMP requires torch and transformers."
            ) from exc

        self.torch = torch

        if self.device is None:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        print(
            f"Loading CLIP for IPOMP: "
            f"{self.model_name}"
        )

        self.processor = (
            CLIPProcessor.from_pretrained(
                self.model_name
            )
        )

        self.model = (
            CLIPModel.from_pretrained(
                self.model_name
            )
        )

        self.model.to(self.device)
        self.model.eval()

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    @staticmethod
    def _load_image(image_value):
        """
        Convert a DSPy DCA image object into a PIL Image.

        Supports:
            1. PIL Image
            2. Local file path
            3. DSPy Image object containing PIL image
            4. Image bytes
            5. Local path attributes
            6. data:image/...;base64,... URLs
            7. DSPy/custom wrapped image strings
        """

        from pathlib import Path
        from io import BytesIO
        import base64
        import ast

        from PIL import Image

        # -------------------------------------------------
        # 1. Already a PIL image
        # -------------------------------------------------

        if isinstance(image_value, Image.Image):
            return image_value.convert("RGB")

        # -------------------------------------------------
        # 2. None
        # -------------------------------------------------

        if image_value is None:
            print("\n" + "=" * 80)
            print("WARNING: DCA IMAGE IS NONE")
            print("=" * 80)
            print("Image value:", image_value)

            raise ValueError(
                "DCA image is None."
            )

        # -------------------------------------------------
        # 3. String / Path
        # -------------------------------------------------

        if isinstance(
            image_value,
            (str, Path),
        ):

            image_string = str(image_value)

            path = Path(image_string)

            # ---------------------------------------------
            # 3A. Local path
            # ---------------------------------------------

            if path.exists():
                return Image.open(
                    path
                ).convert("RGB")

            # ---------------------------------------------
            # 3B. Normal data URL
            # ---------------------------------------------

            if image_string.startswith(
                "data:image/"
            ):

                try:
                    header, encoded = (
                        image_string.split(
                            ",",
                            1,
                        )
                    )

                    image_bytes = (
                        base64.b64decode(
                            encoded
                        )
                    )

                    return Image.open(
                        BytesIO(image_bytes)
                    ).convert("RGB")

                except Exception as exc:
                    raise TypeError(
                        "Failed to decode DCA data URL: "
                        f"{exc}"
                    ) from exc

            # ---------------------------------------------
            # 3C. DSPy/custom wrapped image
            # ---------------------------------------------

            if (
                "<<CUSTOM-TYPE-START-IDENTIFIER>>"
                in image_string
            ):

                try:
                    payload = (
                        image_string.replace(
                            "<<CUSTOM-TYPE-START-IDENTIFIER>>",
                            "",
                            1,
                        ).strip()
                    )

                    parsed = ast.literal_eval(
                        payload
                    )

                    if (
                        isinstance(parsed, list)
                        and parsed
                    ):

                        first_item = parsed[0]

                        if isinstance(
                            first_item,
                            dict,
                        ):

                            image_url_data = (
                                first_item.get(
                                    "image_url"
                                )
                            )

                            if isinstance(
                                image_url_data,
                                dict,
                            ):

                                url = (
                                    image_url_data.get(
                                        "url"
                                    )
                                )

                                if (
                                    isinstance(
                                        url,
                                        str,
                                    )
                                    and url.startswith(
                                        "data:image/"
                                    )
                                ):

                                    header, encoded = (
                                        url.split(
                                            ",",
                                            1,
                                        )
                                    )

                                    image_bytes = (
                                        base64.b64decode(
                                            encoded
                                        )
                                    )

                                    return Image.open(
                                        BytesIO(
                                            image_bytes
                                        )
                                    ).convert("RGB")

                except Exception as exc:
                    raise TypeError(
                        "Failed to decode custom wrapped "
                        f"DCA image: {exc}"
                    ) from exc

            raise FileNotFoundError(
                "Image path does not exist or image "
                "format is unsupported: "
                f"{image_string[:200]}"
            )

        # -------------------------------------------------
        # 4. DSPy Image object
        # -------------------------------------------------

        for attr in (
            "image",
            "pil_image",
            "_image",
        ):

            value = getattr(
                image_value,
                attr,
                None,
            )

            if isinstance(
                value,
                Image.Image,
            ):

                return value.convert("RGB")

        # -------------------------------------------------
        # 5. Image bytes
        # -------------------------------------------------

        for attr in (
            "data",
            "image_data",
            "bytes",
            "_data",
        ):

            value = getattr(
                image_value,
                attr,
                None,
            )

            if isinstance(
                value,
                bytes,
            ):

                return Image.open(
                    BytesIO(value)
                ).convert("RGB")

        # -------------------------------------------------
        # 6. Local path attributes
        # -------------------------------------------------

        for attr in (
            "path",
            "filename",
            "file_path",
            "local_path",
        ):

            value = getattr(
                image_value,
                attr,
                None,
            )

            if value:

                path = Path(
                    str(value)
                )

                if path.exists():

                    return Image.open(
                        path
                    ).convert("RGB")

        # -------------------------------------------------
        # 7. DSPy Image containing a data URL
        # -------------------------------------------------

        url = getattr(
            image_value,
            "url",
            None,
        )

        if url:

            if (
                isinstance(
                    url,
                    str,
                )
                and url.startswith(
                    "data:image/"
                )
            ):

                try:

                    header, encoded = (
                        url.split(
                            ",",
                            1,
                        )
                    )

                    image_bytes = (
                        base64.b64decode(
                            encoded
                        )
                    )

                    return Image.open(
                        BytesIO(
                            image_bytes
                        )
                    ).convert("RGB")

                except Exception as exc:

                    raise TypeError(
                        "Failed to decode DSPy DCA "
                        f"data URL: {exc}"
                    ) from exc

            raise TypeError(
                "Unsupported DSPy image URL format: "
                f"{str(url)[:100]}"
            )

        # -------------------------------------------------
        # 8. Nothing worked
        # -------------------------------------------------

        raise TypeError(
            "\nCould not extract a PIL image from:\n"
            f"Type: {type(image_value)}\n"
            f"Attributes: {dir(image_value)}"
        )

    # =====================================================
    # EMBED SINGLE IMAGE
    # =====================================================

    def _embed_image(self, image):

        self._load_model()

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with self.torch.no_grad():

            features = (
                self.model.get_image_features(
                    **inputs
                )
            )

        # Depending on the installed transformers
        # version, get_image_features() may return
        # either a Tensor or a model-output object.

        if not isinstance(
            features,
            self.torch.Tensor,
        ):

            if (
                hasattr(
                    features,
                    "pooler_output",
                )
                and features.pooler_output
                is not None
            ):

                features = (
                    features.pooler_output
                )

            elif (
                hasattr(
                    features,
                    "image_embeds",
                )
                and features.image_embeds
                is not None
            ):

                features = (
                    features.image_embeds
                )

            elif (
                hasattr(
                    features,
                    "last_hidden_state",
                )
                and features.last_hidden_state
                is not None
            ):

                features = (
                    features.last_hidden_state.mean(
                        dim=1
                    )
                )

            else:

                raise TypeError(
                    "Unsupported CLIP output type "
                    "from get_image_features(): "
                    f"{type(features)}"
                )

        features = (
            features
            / features.norm(
                dim=-1,
                keepdim=True,
            )
        )

        return (
            features[0]
            .cpu()
            .tolist()
        )

    # =====================================================
    # EMBED ONE DCA EXAMPLE
    # =====================================================

    def embed_example(
        self,
        example,
    ):

        example_id = _example_id(
            example
        )

        original_value = getattr(
            example,
            "image_original",
            None,
        )

        reimage_value = getattr(
            example,
            "image_reimage",
            None,
        )

        print(
            f"\nIPOMP EMBEDDING EXAMPLE: "
            f"{example_id}"
        )

        print(
            "image_original type:",
            type(original_value),
        )

        print(
            "image_reimage type:",
            type(reimage_value),
        )

        # -------------------------------------------------
        # Load available images
        # -------------------------------------------------

        original = None
        reimage = None

        if original_value is not None:

            original = (
                self._load_image(
                    original_value
                )
            )

        if reimage_value is not None:

            reimage = (
                self._load_image(
                    reimage_value
                )
            )

        # -------------------------------------------------
        # At least one image must exist
        # -------------------------------------------------

        if (
            original is None
            and reimage is None
        ):

            raise ValueError(
                f"DCA example {example_id} has "
                "both image_original and "
                "image_reimage set to None."
            )

        # -------------------------------------------------
        # Create embeddings
        # -------------------------------------------------

        if original is not None:

            original_embedding = (
                self._embed_image(
                    original
                )
            )

        else:

            original_embedding = None

        if reimage is not None:

            reimage_embedding = (
                self._embed_image(
                    reimage
                )
            )

        else:

            reimage_embedding = None

        # -------------------------------------------------
        # Handle missing side
        # -------------------------------------------------

        if original_embedding is None:

            original_embedding = [
                0.0
                for _ in range(
                    len(reimage_embedding)
                )
            ]

        if reimage_embedding is None:

            reimage_embedding = [
                0.0
                for _ in range(
                    len(original_embedding)
                )
            ]

        # -------------------------------------------------
        # Combine image embeddings
        # -------------------------------------------------

        pair_embedding = [
            (a + b) / 2.0
            for a, b in zip(
                original_embedding,
                reimage_embedding,
            )
        ]

        return pair_embedding

    # =====================================================
    # EMBED ALL EXAMPLES
    # =====================================================

    def embed_examples(
        self,
        examples,
    ):

        embeddings = {}

        for index, example in enumerate(
            examples,
            start=1,
        ):

            print(
                f"IPOMP embedding "
                f"{index}/{len(examples)}",
                end="\r",
            )

            embeddings[
                _example_id(example)
            ] = self.embed_example(
                example
            )

        print()

        return embeddings


# =========================================================
# IPOMP DCA SELECTOR
# =========================================================

class IPOMPDCASelector:
    """
    IPOMP selector for DCA.

    train_size:
        Final number of examples used by GEPA
        as train_set.

    dev_size:
        Final number of examples used by GEPA
        as dev_set.

    Stage 1:
        1. Create CLIP embeddings.
        2. Select a diverse Dev set.
        3. Score remaining candidates using a
           fixed baseline comparator.
        4. Select the hardest candidate pool.
        5. Select the final Train set using
           CLIP max-min diversity.

    Stage 2:
        Refine the evaluation/dev set using
        model-performance redundancy.
    """

    def __init__(
        self,
        examples,
        train_size: int,
        dev_size: int,
        cluster_fraction: float = 0.5,
        num_clusters: Optional[int] = None,
        correlation_threshold: float = 0.90,
        seed: int = 42,
        hard_pool_size: int = 100,
    ):

        self.examples = list(
            examples
        )

        self.train_size = train_size
        self.dev_size = dev_size

        self.cluster_fraction = (
            cluster_fraction
        )

        self.num_clusters = num_clusters

        self.correlation_threshold = (
            correlation_threshold
        )

        self.rng = random.Random(
            seed
        )

        self.embedder = (
            DCAPairEmbedder()
        )

        self.embeddings = {}

        # prompt_id -> example_id -> score
        self.prompt_scores = {}

        # example_id -> baseline DCA difficulty
        # Reused by Stage 2 replacement selection.
        self.difficulty_scores = {}

        # Number of difficult examples kept
        # before diversity selection.
        self.hard_pool_size = (
            hard_pool_size
        )

    # =====================================================
    # EMBEDDINGS
    # =====================================================

    def build_embeddings(self):

        if self.embeddings:
            return self.embeddings

        self.embeddings = (
            self.embedder.embed_examples(
                self.examples
            )
        )

        return self.embeddings

    # =====================================================
    # CLUSTER COUNT
    # =====================================================

    def _cluster_count(self):

        if self.num_clusters is not None:

            return max(
                1,
                min(
                    self.num_clusters,
                    len(self.examples),
                ),
            )

        return max(
            1,
            min(
                int(
                    math.sqrt(
                        len(self.examples)
                    )
                ),
                len(self.examples),
            ),
        )

    # =====================================================
    # SIMPLE K-MEANS
    # =====================================================

    def _kmeans(
        self,
        examples,
        k,
    ):

        if k >= len(examples):

            return [
                [example]
                for example in examples
            ]

        vectors = [
            self.embeddings[
                _example_id(example)
            ]
            for example in examples
        ]

        centroid_indices = (
            self.rng.sample(
                range(
                    len(examples)
                ),
                k,
            )
        )

        centroids = [
            list(
                vectors[index]
            )
            for index in centroid_indices
        ]

        assignments = [
            0
            for _ in examples
        ]

        for _ in range(20):

            changed = False

            # -----------------------------------------
            # Assignment
            # -----------------------------------------

            for i, vector in enumerate(
                vectors
            ):

                distances = [
                    1.0
                    - _cosine_similarity(
                        vector,
                        centroid,
                    )
                    for centroid
                    in centroids
                ]

                new_assignment = min(
                    range(k),
                    key=lambda x:
                        distances[x],
                )

                if (
                    assignments[i]
                    != new_assignment
                ):

                    changed = True

                assignments[i] = (
                    new_assignment
                )

            # -----------------------------------------
            # Recalculate centroids
            # -----------------------------------------

            for cluster_index in range(k):

                members = [
                    vectors[i]
                    for i in range(
                        len(vectors)
                    )
                    if assignments[i]
                    == cluster_index
                ]

                if not members:
                    continue

                dimension = len(
                    members[0]
                )

                centroids[
                    cluster_index
                ] = [
                    sum(
                        member[d]
                        for member
                        in members
                    )
                    / len(members)
                    for d in range(
                        dimension
                    )
                ]

            if not changed:
                break

        clusters = [
            []
            for _ in range(k)
        ]

        for (
            example,
            assignment,
        ) in zip(
            examples,
            assignments,
        ):

            clusters[
                assignment
            ].append(
                example
            )

        return clusters

    # =====================================================
    # STAGE 1: DIVERSE DEV SET
    # =====================================================

    def _select_diverse_dev(self):

        self.build_embeddings()

        k = self._cluster_count()

        clusters = self._kmeans(
            self.examples,
            k,
        )

        representatives = []

        # -------------------------------------------------
        # Select cluster representatives
        # -------------------------------------------------

        for cluster in clusters:

            if not cluster:
                continue

            centroid = [
                sum(
                    self.embeddings[
                        _example_id(
                            example
                        )
                    ][dimension]
                    for example
                    in cluster
                )
                / len(cluster)
                for dimension
                in range(
                    len(
                        self.embeddings[
                            _example_id(
                                cluster[0]
                            )
                        ]
                    )
                )
            ]

            representative = min(
                cluster,
                key=lambda example:
                    1.0
                    - _cosine_similarity(
                        self.embeddings[
                            _example_id(
                                example
                            )
                        ],
                        centroid,
                    ),
            )

            representatives.append(
                representative
            )

        # -------------------------------------------------
        # Start with representatives
        # -------------------------------------------------

        selected = representatives[
            :self.dev_size
        ]

        # -------------------------------------------------
        # Fill remaining slots using max-min
        # -------------------------------------------------

        while (
            len(selected)
            < self.dev_size
        ):

            selected_ids = {
                _example_id(
                    example
                )
                for example
                in selected
            }

            remaining = [
                example
                for example
                in self.examples
                if _example_id(
                    example
                )
                not in selected_ids
            ]

            if not remaining:
                break

            # If no example has been selected,
            # choose one randomly.

            if not selected:

                selected.append(
                    self.rng.choice(
                        remaining
                    )
                )

                continue

            # -------------------------------------------------
            # Maximum-minimum distance
            # -------------------------------------------------

            candidate = max(
                remaining,
                key=lambda example:
                    min(
                        1.0
                        - _cosine_similarity(
                            self.embeddings[
                                _example_id(
                                    example
                                )
                            ],
                            self.embeddings[
                                _example_id(
                                    selected_example
                                )
                            ],
                        )
                        for selected_example
                        in selected
                    ),
            )

            selected.append(
                candidate
            )

        return selected[
            :self.dev_size
        ]

    # =====================================================
    # DIFFICULT TRAINING CANDIDATES
    # =====================================================

    def _select_hard_training_pool(
        self,
        train_candidates,
        comparator,
        metric,
    ):
        """
        Select the hardest examples from the remaining
        training candidates using a fixed baseline comparator.

        Difficulty:

            difficulty = 1 - baseline_score

        Higher difficulty means the baseline model
        performs worse on the example.
        """

        print("\n" + "=" * 80)
        print("HARD TRAINING CANDIDATE SELECTION")
        print("=" * 80)

        print(
            f"Candidate examples : "
            f"{len(train_candidates)}"
        )

        print(
            f"Hard pool size     : "
            f"{self.hard_pool_size}"
        )

        scored_examples = []

        # -------------------------------------------------
        # Score every remaining candidate
        # -------------------------------------------------

        for index, example in enumerate(
            train_candidates,
            start=1,
        ):

            try:

                score = float(
                    metric(
                        example,
                        comparator,
                    )
                )

            except TypeError:

                score = float(
                    metric(
                        example,
                        comparator,
                        None,
                    )
                )

            difficulty = (
                1.0 - score
            )

            # Reuse this same baseline difficulty during
            # Stage 2 replacement selection.
            self.difficulty_scores[
                _example_id(example)
            ] = difficulty

            scored_examples.append(
                {
                    "example": example,
                    "score": score,
                    "difficulty": difficulty,
                }
            )

            print(
                f"\rScoring "
                f"{index}/{len(train_candidates)}",
                end="",
            )

        print()

        # -------------------------------------------------
        # Hardest examples first
        # -------------------------------------------------

        scored_examples.sort(
            key=lambda item:
                item["difficulty"],
            reverse=True,
        )

        hard_pool_size = min(
            self.hard_pool_size,
            len(scored_examples),
        )

        hard_pool = [
            item["example"]
            for item
            in scored_examples[
                :hard_pool_size
            ]
        ]

        print(
            f"\nSelected hard candidate pool : "
            f"{len(hard_pool)}"
        )

        print(
            "\nHardest examples:"
        )

        for item in scored_examples[
            :10
        ]:

            print(
                f"  "
                f"{_example_id(item['example'])}"
                f" | score="
                f"{item['score']:.4f}"
                f" | difficulty="
                f"{item['difficulty']:.4f}"
            )

        print("=" * 80)

        return (
            hard_pool,
            scored_examples,
        )

    # =====================================================
    # HARD + DIVERSE TRAINING SELECTION
    # =====================================================

    def _select_diverse_from_hard_pool(
        self,
        hard_pool,
    ):
        """
        Select final training examples from the hard pool
        using CLIP max-min diversity.

        The hard pool already contains difficult examples.

        This step prevents the final training set from
        containing many visually similar difficult examples.
        """

        # -------------------------------------------------
        # If hard pool is already small enough
        # -------------------------------------------------

        if (
            len(hard_pool)
            <= self.train_size
        ):

            return list(
                hard_pool
            )

        self.build_embeddings()

        selected = []

        # -------------------------------------------------
        # Start with one difficult example
        # -------------------------------------------------

        first_example = (
            self.rng.choice(
                hard_pool
            )
        )

        selected.append(
            first_example
        )

        # -------------------------------------------------
        # Max-min diversity
        # -------------------------------------------------

        while (
            len(selected)
            < self.train_size
        ):

            selected_ids = {
                _example_id(
                    example
                )
                for example
                in selected
            }

            remaining = [
                example
                for example
                in hard_pool
                if _example_id(
                    example
                )
                not in selected_ids
            ]

            if not remaining:
                break

            candidate = max(
                remaining,
                key=lambda example:
                    min(
                        1.0
                        - _cosine_similarity(
                            self.embeddings[
                                _example_id(
                                    example
                                )
                            ],
                            self.embeddings[
                                _example_id(
                                    selected_example
                                )
                            ],
                        )
                        for selected_example
                        in selected
                    ),
            )

            selected.append(
                candidate
            )

        return selected[
            :self.train_size
        ]

    # =====================================================
    # COMPLETE STAGE 1 SELECTION
    # =====================================================

    def initial_selection(
        self,
        comparator,
        metric,
    ):
        """
        Complete Stage 1 selection.

        Workflow:

            Candidate pool
                  |
                  v
            Diverse Dev
                  |
                  v
          Remaining candidates
                  |
                  v
          Baseline scoring
                  |
                  v
            Hard pool
                  |
                  v
        CLIP max-min diversity
                  |
                  v
            Final Train
        """

        total_required = (
            self.train_size
            + self.dev_size
        )

        if (
            total_required
            > len(self.examples)
        ):

            raise ValueError(
                "Requested train + dev size "
                "is larger than the available "
                "training pool. "
                f"Available={len(self.examples)}, "
                f"train={self.train_size}, "
                f"dev={self.dev_size}"
            )

        # =================================================
        # SELECT DEV
        # =================================================

        print("\n" + "=" * 80)
        print("IPOMP STAGE 1 - DEV SELECTION")
        print("=" * 80)

        dev_set = (
            self._select_diverse_dev()
        )

        dev_ids = {
            _example_id(
                example
            )
            for example
            in dev_set
        }

        print(
            f"\nSelected Dev examples : "
            f"{len(dev_set)}"
        )

        # =================================================
        # REMAINING CANDIDATES
        # =================================================

        train_candidates = [
            example
            for example
            in self.examples
            if _example_id(
                example
            )
            not in dev_ids
        ]

        print(
            f"Remaining Train candidates : "
            f"{len(train_candidates)}"
        )

        # =================================================
        # SELECT HARD TRAINING POOL
        # =================================================

        (
            hard_pool,
            difficulty_scores,
        ) = (
            self._select_hard_training_pool(
                train_candidates=(
                    train_candidates
                ),
                comparator=comparator,
                metric=metric,
            )
        )

        # =================================================
        # SELECT FINAL TRAIN FROM HARD POOL
        # =================================================

        print("\n" + "=" * 80)
        print("HARD + DIVERSE TRAINING SELECTION")
        print("=" * 80)

        train_set = (
            self._select_diverse_from_hard_pool(
                hard_pool
            )
        )

        # =================================================
        # SAFETY CHECK: TRAIN SIZE
        # =================================================

        if (
            len(train_set)
            != self.train_size
        ):

            raise RuntimeError(
                "Hard + diverse training selection "
                "did not produce the requested train size. "
                f"Requested={self.train_size}, "
                f"Selected={len(train_set)}"
            )

        # =================================================
        # SAFETY CHECK: NO TRAIN/DEV OVERLAP
        # =================================================

        train_ids = {
            _example_id(
                example
            )
            for example
            in train_set
        }

        overlap = (
            train_ids
            .intersection(
                dev_ids
            )
        )

        if overlap:

            raise RuntimeError(
                "Train and Dev sets overlap. "
                f"Overlap={overlap}"
            )

        print(
            f"\nFinal Train size : "
            f"{len(train_set)}"
        )

        print(
            f"Final Dev size   : "
            f"{len(dev_set)}"
        )

        print("=" * 80)

        return (
            train_set,
            dev_set,
        )

    # =====================================================
    # STAGE 2 PERFORMANCE RECORDING
    # =====================================================

    def record_prompt_scores(
        self,
        prompt_id,
        scored_examples,
    ):
        """
        Record per-example scores for a prompt.

        scored_examples should contain:

            {
                "example": example,
                "score": score
            }
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
                _example_id(
                    example
                )
            ] = score

        self.prompt_scores[
            str(prompt_id)
        ] = scores

    # =====================================================
    # PERFORMANCE MATRIX
    # =====================================================

    def build_performance_matrix(
        self,
        evaluation_examples,
    ):
        """
        Build:

            example -> [
                prompt1_score,
                prompt2_score,
                ...
            ]
        """

        matrix = {}

        for example in (
            evaluation_examples
        ):

            example_id = (
                _example_id(
                    example
                )
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
                for prompt_id
                in self.prompt_scores
            ]

        return matrix

    # =====================================================
    # CORRELATION
    # =====================================================

    @staticmethod
    def _correlation(
        a,
        b,
    ):

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
            * (
                y - mean_b
            )
            for x, y
            in zip(a, b)
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
    # REDUNDANCY
    # =====================================================

    def find_redundant_examples(
        self,
        evaluation_examples,
    ):
        """
        Find pairs of evaluation examples whose
        performance patterns across prompts are
        highly correlated.
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
    # REPLACEMENT
    # =====================================================

    def find_replacement(
        self,
        redundant_example,
        candidate_pool,
        evaluation_examples,
    ):
        """
        Select a replacement using a combination of:

            1. CLIP visual diversity
            2. DCA difficulty

        The two signals are equally weighted:

            replacement_score =
                0.5 * normalized_clip_distance
                + 0.5 * difficulty

        Difficulty is the baseline DCA difficulty already
        calculated during Stage 1:

            difficulty = 1 - baseline_score

        Everything else in the original IPOMP replacement
        flow remains unchanged.
        """

        evaluation_ids = {
            _example_id(
                example
            )
            for example
            in evaluation_examples
        }

        candidates = [
            example
            for example
            in candidate_pool
            if _example_id(
                example
            )
            not in evaluation_ids
        ]

        if not candidates:
            return None

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
        # Calculate CLIP distance and DCA difficulty for
        # every available replacement candidate.
        # -------------------------------------------------

        candidate_scores = []

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

            difficulty = (
                self.difficulty_scores.get(
                    candidate_id,
                    0.0,
                )
            )

            candidate_scores.append(
                {
                    "candidate": candidate,
                    "id": candidate_id,
                    "distance": distance,
                    "difficulty": difficulty,
                }
            )

        # -------------------------------------------------
        # Normalize CLIP distance to [0, 1] so that it can
        # be combined fairly with difficulty, which is
        # already in the [0, 1] range.
        # -------------------------------------------------

        distances = [
            item["distance"]
            for item
            in candidate_scores
        ]

        min_distance = min(distances)
        max_distance = max(distances)

        distance_range = (
            max_distance
            - min_distance
        )

        for item in candidate_scores:

            if distance_range > 1e-12:

                normalized_distance = (
                    item["distance"]
                    - min_distance
                ) / distance_range

            else:

                normalized_distance = 0.0

            item["normalized_distance"] = (
                normalized_distance
            )

            # =================================================
            # NEW SAMPLING STRATEGY
            #
            # 50% CLIP diversity
            # 50% DCA difficulty
            # =================================================

            item["replacement_score"] = (
                0.5
                * item["normalized_distance"]
                +
                0.5
                * item["difficulty"]
            )

        # -------------------------------------------------
        # Select the candidate with the highest combined
        # diversity + difficulty score.
        # -------------------------------------------------

        best_item = max(
            candidate_scores,
            key=lambda item:
                item["replacement_score"],
        )

        best_candidate = (
            best_item["candidate"]
        )

        if (
            best_candidate
            is not None
        ):

            print(
                "\nIPOMP-DCA difficulty-aware replacement search:"
            )

            print(
                f"  Removed              : "
                f"{target_id}"
            )

            print(
                f"  Replacement          : "
                f"{best_item['id']}"
            )

            print(
                f"  CLIP distance        : "
                f"{best_item['distance']:.4f}"
            )

            print(
                f"  Normalized diversity : "
                f"{best_item['normalized_distance']:.4f}"
            )

            print(
                f"  DCA difficulty       : "
                f"{best_item['difficulty']:.4f}"
            )

            print(
                f"  Combined score       : "
                f"{best_item['replacement_score']:.4f}"
            )

        return best_candidate

    # =====================================================
    # ONE REFINEMENT
    # =====================================================

    def refine_once(
        self,
        evaluation_examples,
        candidate_pool,
    ):

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

        (
            example_a,
            example_b,
            correlation,
        ) = redundant[0]

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
            )
        )

        if replacement is None:

            return (
                list(
                    evaluation_examples
                ),
                None,
            )

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

        refined.append(
            replacement
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
    # STAGE 2: COMPLETE REFINEMENT
    # =====================================================

    def run_stage2_refinement(
        self,
        evaluation_examples,
        candidate_pool,
    ):
        """
        Run IPOMP Stage 2 on the current
        evaluation/dev set.

        Stage 2:

            1. Build model-performance matrix.
            2. Find redundant examples.
            3. Replace one redundant example with
               a contrasting candidate.

        The evaluation-set size remains unchanged.
        """

        print("\n" + "=" * 80)
        print(
            "IPOMP STAGE 2 - "
            "PERFORMANCE-GUIDED EVALUATION REFINEMENT"
        )
        print("=" * 80)

        print(
            f"Current evaluation set : "
            f"{len(evaluation_examples)}"
        )

        print(
            f"Candidate replacement pool : "
            f"{len(candidate_pool)}"
        )

        print(
            f"Recorded candidate prompts : "
            f"{len(self.prompt_scores)}"
        )

        # -------------------------------------------------
        # Need at least two prompts
        # -------------------------------------------------

        if len(
            self.prompt_scores
        ) < 2:

            print(
                "Not enough candidate prompts "
                "for Stage 2. At least 2 prompt "
                "evaluations are required."
            )

            return (
                list(
                    evaluation_examples
                ),
                None,
            )

        # -------------------------------------------------
        # Build performance matrix
        # -------------------------------------------------

        performance_matrix = (
            self.build_performance_matrix(
                evaluation_examples
            )
        )

        print(
            "\nPerformance matrix:"
        )

        print(
            f"  Examples : "
            f"{len(performance_matrix)}"
        )

        print(
            f"  Prompts  : "
            f"{len(self.prompt_scores)}"
        )

        # -------------------------------------------------
        # Find redundant examples
        # -------------------------------------------------

        redundant = (
            self.find_redundant_examples(
                evaluation_examples
            )
        )

        print(
            f"\nRedundant example pairs found : "
            f"{len(redundant)}"
        )

        if not redundant:

            print(
                "No redundant evaluation "
                "examples found."
            )

            print("=" * 80)

            return (
                list(
                    evaluation_examples
                ),
                None,
            )

        # -------------------------------------------------
        # Show detected redundancy
        # -------------------------------------------------

        print(
            "\nDetected redundant pairs:"
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
        # Perform one refinement
        # -------------------------------------------------

        (
            refined_evaluation,
            refinement_info,
        ) = self.refine_once(
            evaluation_examples=(
                evaluation_examples
            ),
            candidate_pool=(
                candidate_pool
            ),
        )

        if refinement_info is None:

            print(
                "\nStage 2 found redundancy, "
                "but no valid replacement "
                "was available."
            )

            print("=" * 80)

            return (
                list(
                    evaluation_examples
                ),
                None,
            )

        # -------------------------------------------------
        # Verify size preservation
        # -------------------------------------------------

        if (
            len(refined_evaluation)
            != len(
                evaluation_examples
            )
        ):

            raise RuntimeError(
                "IPOMP Stage 2 changed "
                "the evaluation-set size. "
                f"Before="
                f"{len(evaluation_examples)}, "
                f"After="
                f"{len(refined_evaluation)}"
            )

        # -------------------------------------------------
        # Print refinement
        # -------------------------------------------------

        print(
            "\nSTAGE 2 REFINEMENT"
        )

        print(
            "-" * 80
        )

        print(
            f"Removed example    : "
            f"{refinement_info['removed']}"
        )

        print(
            f"Replacement example: "
            f"{refinement_info['replacement']}"
        )

        print(
            f"Correlation        : "
            f"{refinement_info['correlation']:.4f}"
        )

        print(
            f"Evaluation size    : "
            f"{len(refined_evaluation)}"
        )

        print(
            "-" * 80
        )

        print(
            "Stage 2 complete: "
            "redundant example replaced "
            "by a contrasting example."
        )

        print("=" * 80)

        return (
            refined_evaluation,
            refinement_info,
        )

    # =====================================================
    # PRINT SELECTION
    # =====================================================

    @staticmethod
    def print_selection(
        train_set,
        dev_set,
    ):

        print("\n" + "=" * 80)
        print("IPOMP DATA SELECTION")
        print("=" * 80)

        print(
            f"GEPA train_set : "
            f"{len(train_set)}"
        )

        print(
            f"GEPA dev_set   : "
            f"{len(dev_set)}"
        )

        print(
            "\nDev examples:"
        )

        for example in dev_set:

            print(
                f"  {_example_id(example)}"
            )

        print(
            "\nTrain examples:"
        )

        for example in train_set:

            print(
                f"  {_example_id(example)}"
            )

        print("=" * 80)