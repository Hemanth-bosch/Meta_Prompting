"""
SESS-WREP sampling for DCA
===========================

This file is intentionally separate from SESS-REP and SESS-LC.

Purpose:
    1. Select Dev FIRST using SESS-WREP:
         CLIP representation + initial DCA difficulty
         -> weighted facility-location
    2. Remove the selected Dev examples.
    3. Select Train from the remaining candidates using:
         initial DCA difficulty
         -> hard candidate pool
         -> CLIP max-min diversity

Guarantees:
    - Train and Dev are selected from the Train candidate pool only.
    - Dev is reserved before Train selection.
    - Train/Dev overlap is explicitly checked.
    - Validation/Test are not touched by this file.
    - Existing SESS-REP and SESS-LC code does not need to be modified.

Important:
    This file is standalone for WREP sampling. It does not import
    DCAPairEmbedder or the Train selector from sess_rep_dca.py.
    WREPDCAEmbedder below provides the WREP-specific CLIP representation,
    including complete and incomplete DCA image-pair handling.

WREP objective:
    F(S) = sum_j w_j * max_{i in S} sim(i, j)

DCA adaptation:
    DCA score      = initial comparator performance
    difficulty     = 1 - DCA score
    normalized difficulty is converted into the WREP weight.

    lambda = 0.0 -> pure representativeness
    lambda = 1.0 -> maximum emphasis on difficulty

Recommended first experiment:
    train_size = 50
    dev_size   = 30
    wrep_lambda = 0.5
    seed = 42
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
import numpy as np

# ---------------------------------------------------------------------------
# Standalone DCA / CLIP utilities
# ---------------------------------------------------------------------------
#
# This WREP file intentionally does NOT import DCAPairEmbedder or the Train
# selector from sess_rep_dca.py.  The implementations below are WREP-specific
# copies/adaptations of the working SESS-REP utilities, with one important DCA
# improvement: incomplete pairs are supported.
#
# Pair representation:
#   both images      -> normalize each CLIP image embedding, average them,
#                       then normalize the averaged vector
#   original only    -> normalized original embedding
#   reimage only     -> normalized reimage embedding
#   neither          -> cannot be represented and raises an explicit error
#
# This keeps every valid DCA candidate in the WREP candidate pool when at
# least one image is available, while keeping every embedding at the same
# CLIP dimensionality.
# ---------------------------------------------------------------------------

import ast
import base64
import math
import random
from io import BytesIO
from urllib.parse import unquote, urlparse

from PIL import Image


def _example_id(example: Any) -> str:
    pair_name = getattr(example, "pair_name", None)
    if pair_name is not None:
        return str(pair_name)

    example_id = getattr(example, "id", None)
    if example_id is not None:
        return str(example_id)

    return str(id(example))


def _get_example_field(example: Any, field_name: str):
    """Robustly retrieve a field from a DSPy Example."""

    try:
        value = getattr(example, field_name, None)
    except Exception:
        value = None

    if value is not None:
        return value

    try:
        if hasattr(example, "toDict"):
            data = example.toDict()
            if isinstance(data, dict):
                value = data.get(field_name)
                if value is not None:
                    return value
    except Exception:
        pass

    for attr_name in ("_store", "store", "inputs", "_inputs"):
        try:
            store = getattr(example, attr_name, None)
        except Exception:
            store = None

        if isinstance(store, dict):
            value = store.get(field_name)
            if value is not None:
                return value

    return None


def _image_pair_status(examples):
    """Return complete, original-only, reimage-only and empty pairs."""
    complete = []
    original_only = []
    reimage_only = []
    empty = []

    for example in examples:
        original = _get_example_field(example, "image_original")
        reimage = _get_example_field(example, "image_reimage")

        if original is not None and reimage is not None:
            complete.append(example)
        elif original is not None:
            original_only.append(example)
        elif reimage is not None:
            reimage_only.append(example)
        else:
            empty.append(example)

    return complete, original_only, reimage_only, empty


class WREPDCAEmbedder:
    """CLIP embedding utility used only by the WREP sampler."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.processor = None
        self.model = None
        self.torch = None

    def _load_model(self):
        if self.model is not None:
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "WREP requires torch and transformers for CLIP embeddings."
            ) from exc

        self.torch = torch

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading CLIP for WREP: {self.model_name}")

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_image(value):
        """Convert a DCA image value into a PIL RGB image."""

        if value is None:
            raise TypeError("DCA image value is None")

        if isinstance(value, Image.Image):
            return value.convert("RGB")

        if isinstance(value, (bytes, bytearray)):
            return Image.open(BytesIO(value)).convert("RGB")

        def _open_string(text):
            text = str(text).strip()
            if not text:
                return None

            try:
                path = Path(text)
                if path.exists() and path.is_file():
                    return Image.open(path).convert("RGB")
            except (OSError, ValueError):
                pass

            if text.startswith("data:image") and "," in text:
                header, encoded = text.split(",", 1)
                if ";base64" in header:
                    try:
                        image_bytes = base64.b64decode(encoded, validate=False)
                        return Image.open(BytesIO(image_bytes)).convert("RGB")
                    except Exception:
                        pass

            if text.startswith("file://"):
                try:
                    parsed = urlparse(text)
                    file_path = unquote(parsed.path)
                    if parsed.netloc and parsed.netloc not in ("", "localhost"):
                        file_path = f"//{parsed.netloc}{file_path}"
                    path = Path(file_path)
                    if path.exists() and path.is_file():
                        return Image.open(path).convert("RGB")
                except Exception:
                    pass

            if text.startswith(("http://", "https://")):
                try:
                    import requests
                    response = requests.get(text, timeout=30)
                    response.raise_for_status()
                    return Image.open(BytesIO(response.content)).convert("RGB")
                except Exception:
                    pass

            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, str):
                    path = Path(parsed)
                    if path.exists() and path.is_file():
                        return Image.open(path).convert("RGB")
            except Exception:
                pass

            return None

        if isinstance(value, (str, Path)):
            image = _open_string(value)
            if image is not None:
                return image

        for attr in (
            "url", "path", "local_path", "filename", "file_path",
            "data", "content", "value",
        ):
            try:
                attr_value = getattr(value, attr, None)
            except Exception:
                attr_value = None

            if attr_value is None:
                continue

            if isinstance(attr_value, Image.Image):
                return attr_value.convert("RGB")

            if isinstance(attr_value, (bytes, bytearray)):
                try:
                    return Image.open(BytesIO(attr_value)).convert("RGB")
                except Exception:
                    pass

            image = _open_string(attr_value)
            if image is not None:
                return image

        for attr in ("image", "pil_image", "image_data"):
            try:
                image_value = getattr(value, attr, None)
            except Exception:
                image_value = None

            if image_value is None:
                continue

            if isinstance(image_value, Image.Image):
                return image_value.convert("RGB")

            if isinstance(image_value, (bytes, bytearray)):
                try:
                    return Image.open(BytesIO(image_value)).convert("RGB")
                except Exception:
                    pass

            image = _open_string(image_value)
            if image is not None:
                return image

        value_string = str(value)
        if "data:image" in value_string:
            start = value_string.find("data:image")
            candidate = value_string[start:]
            for marker in (
                "}}]<<CUSTOM-TYPE-END-IDENTIFIER>>",
                "<<CUSTOM-TYPE-END-IDENTIFIER>>",
            ):
                if marker in candidate:
                    candidate = candidate.split(marker, 1)[0]
            image = _open_string(candidate)
            if image is not None:
                return image

        raise TypeError(
            "Unsupported DCA image value: "
            f"{type(value)!r}\n"
            f"Available attributes: {dir(value)}"
        )

    def _encode_single_image(self, image):
        """Return one normalized CLIP image embedding."""
        torch = self.torch

        with torch.no_grad():
            inputs = self.processor(
                images=[image],
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            features = self.model.get_image_features(**inputs)

            if not isinstance(features, torch.Tensor):
                if hasattr(features, "pooler_output"):
                    features = features.pooler_output
                elif hasattr(features, "last_hidden_state"):
                    features = features.last_hidden_state[:, 0, :]
                else:
                    raise TypeError(
                        "Unexpected CLIP image feature output type: "
                        f"{type(features)!r}"
                    )

            features = features[0]
            features = features / features.norm().clamp_min(1e-12)

        return features

    def encode_pair(self, example: Any):
        """Encode complete or incomplete DCA image pairs to one fixed-size vector."""
        self._load_model()
        torch = self.torch
        example_id = _example_id(example)

        original_value = _get_example_field(example, "image_original")
        reimage_value = _get_example_field(example, "image_reimage")

        if original_value is None and reimage_value is None:
            raise ValueError(
                f"No usable image is available for DCA example {example_id}."
            )

        original = self._load_image(original_value) if original_value is not None else None
        reimage = self._load_image(reimage_value) if reimage_value is not None else None

        with torch.no_grad():
            features = []

            if original is not None:
                features.append(self._encode_single_image(original))

            if reimage is not None:
                features.append(self._encode_single_image(reimage))

            if len(features) == 2:
                pair = (features[0] + features[1]) / 2.0
            else:
                pair = features[0]

            pair = pair / pair.norm().clamp_min(1e-12)

        return pair.detach().cpu().numpy().astype("float32")

    def encode_examples(self, examples):
        import numpy as np
        return np.asarray([self.encode_pair(example) for example in examples], dtype=np.float32)


def _select_difficulty_diverse_train_examples(
    examples,
    sample_size: int,
    comparator,
    metric,
    model_name: str,
    device=None,
    cache_dir: str = ".cache/dca_embeddings/wrep/train",
    hard_pool_size: int = 100,
    seed: int = 42,
):
    """Standalone DCA difficulty -> hard-pool -> CLIP max-min diversity Train selector."""
    import numpy as np

    examples = list(examples)
    if sample_size <= 0 or sample_size > len(examples):
        raise ValueError(f"Invalid Train sample_size={sample_size}; candidates={len(examples)}")

    scores = []
    for index, example in enumerate(examples, start=1):
        prediction = comparator(
            image_original=example.image_original,
            image_reimage=example.image_reimage,
        )
        score = float(np.clip(metric(example, prediction), 0.0, 1.0))
        scores.append((example, score, 1.0 - score))
        print(f"\rWREP Train scoring: {index}/{len(examples)}", end="")
    print()

    ranked = sorted(scores, key=lambda x: (-x[2], _example_id(x[0])))
    hard_pool_size = min(max(sample_size, hard_pool_size), len(ranked))
    hard_pool = [item[0] for item in ranked[:hard_pool_size]]

    embeddings = _embed_examples_wrep(
        hard_pool,
        model_name=model_name,
        cache_dir=cache_dir,
        device=device,
    )

    # Deterministic seeded first point among the hardest examples, then
    # greedily maximize minimum cosine distance to already-selected points.
    rng = random.Random(seed)
    first = rng.randrange(len(hard_pool))
    selected = [first]
    available = set(range(len(hard_pool)))

    while len(selected) < sample_size:
        available.difference_update(selected)
        best_idx = None
        best_distance = -float("inf")

        for idx in available:
            min_distance = float("inf")
            for chosen in selected:
                similarity = _cosine_similarity(embeddings[idx], embeddings[chosen])
                distance = 1.0 - similarity
                min_distance = min(min_distance, distance)

            if min_distance > best_distance:
                best_distance = min_distance
                best_idx = idx
            elif math.isclose(min_distance, best_distance) and best_idx is not None:
                if _example_id(hard_pool[idx]) < _example_id(hard_pool[best_idx]):
                    best_idx = idx

        if best_idx is None:
            break
        selected.append(best_idx)

    train_set = [hard_pool[i] for i in selected]

    if len(train_set) != sample_size:
        raise RuntimeError(
            f"Difficulty/diverse Train selector returned {len(train_set)} "
            f"examples; expected {sample_size}."
        )

    train_info = {
        "method": "difficulty_hard_pool_plus_clip_maxmin_diversity",
        "hard_pool_size": hard_pool_size,
        "sample_size": sample_size,
        "seed": seed,
        "scores": [
            {
                "example_id": _example_id(example),
                "score": score,
                "difficulty": difficulty,
            }
            for example, score, difficulty in scores
        ],
    }

    return train_set, train_info


# ---------------------------------------------------------------------------
# WREP-specific cache
# ---------------------------------------------------------------------------

def _wrep_cache_key(example: Any, model_name: str) -> str:
    """
    Create a WREP-specific cache key.

    The WREP cache is deliberately separate from the SESS-REP cache so that
    adding WREP experiments cannot overwrite or confuse SESS-REP embeddings.
    """
    try:
        original = getattr(example, "image_original", None)
    except Exception:
        original = None

    try:
        reimage = getattr(example, "image_reimage", None)
    except Exception:
        reimage = None

    raw = "|".join(
        [
            _example_id(example),
            str(original),
            str(reimage),
            model_name,
            "SESS-WREP",
        ]
    )

    return hashlib.sha256(
        raw.encode("utf-8", errors="replace")
    ).hexdigest()


def _embed_examples_wrep(
    examples,
    model_name: str,
    cache_dir: str,
    device=None,
):
    """
    Generate CLIP embeddings using a WREP-specific cache.

    WREPDCAEmbedder is defined in this file and follows the working
    SESS-REP image-loading/CLIP pattern without importing SESS-REP.

    Expected representation:
        - both images: averaged/normalized pair representation
        - original only: original representation
        - reimage only: reimage representation

    All valid representations should have the same dimensionality.
    """

    import numpy as np

    examples = list(examples)

    if not examples:
        return np.empty((0, 0), dtype=np.float32)

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    embedder = WREPDCAEmbedder(
        model_name=model_name,
        device=device,
    )

    embeddings = []

    expected_dim = None

    for idx, example in enumerate(examples, start=1):

        path = (
            cache_path
            / f"wrep_{_wrep_cache_key(example, model_name)}.npy"
        )

        if path.exists():
            emb = np.load(path)
        else:
            emb = embedder.encode_pair(example)
            np.save(path, emb)

        emb = np.asarray(emb, dtype=np.float32)

        if emb.ndim != 1:
            raise ValueError(
                f"Invalid WREP embedding shape for "
                f"{_example_id(example)}: {emb.shape}"
            )

        if expected_dim is None:
            expected_dim = emb.shape[0]

        if emb.shape[0] != expected_dim:
            raise ValueError(
                "WREP embedding dimension mismatch: "
                f"{_example_id(example)} has {emb.shape[0]} dimensions, "
                f"expected {expected_dim}."
            )

        embeddings.append(emb)

        if idx % 25 == 0 or idx == len(examples):
            print(
                f"WREP embedding examples: "
                f"{idx}/{len(examples)}"
            )

    return np.asarray(embeddings, dtype=np.float32)

def _cosine_similarity(a, b):
    """Compute cosine similarity between two embedding vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))
# ---------------------------------------------------------------------------
# WREP weighted facility-location
# ---------------------------------------------------------------------------

def _greedy_wrep_facility_location(
    examples,
    embeddings,
    sample_size: int,
    weights,
):
    """
    Greedy weighted facility-location selection.

    For every candidate x:

        gain(x) =
            sum_j w_j *
            max(0, sim(x,j) - current_best_similarity(j))

    The candidate with the largest marginal weighted coverage is selected.

    This is NOT simply hard-example selection. Difficulty affects the
    importance/weight of covered examples, while CLIP similarity controls
    representativeness.
    """

    import numpy as np

    examples = list(examples)

    n = len(examples)

    if sample_size <= 0:
        return []

    if sample_size > n:
        raise ValueError(
            f"sample_size={sample_size} exceeds "
            f"candidate count={n}"
        )

    weights = np.asarray(weights, dtype=np.float64)

    if weights.shape != (n,):
        raise ValueError(
            f"Expected {n} WREP weights, got shape={weights.shape}"
        )

    if np.any(~np.isfinite(weights)):
        raise ValueError("WREP weights contain NaN or infinity.")

    if np.any(weights < 0):
        raise ValueError("WREP weights cannot be negative.")

    # CLIP embeddings are normalized by WREPDCAEmbedder and normalized again here
    # for numerical safety.
    # Normalize again here for numerical safety.
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )
    norms = np.clip(norms, 1e-12, None)

    normalized_embeddings = embeddings / norms

    # Pairwise cosine similarity.
    sim = normalized_embeddings @ normalized_embeddings.T

    # Convert cosine [-1, 1] to [0, 1], matching the SESS-style
    # facility-location implementation used in sess_rep_dca.py.
    sim = np.clip(
        (sim + 1.0) / 2.0,
        0.0,
        1.0,
    )

    best = np.zeros(
        n,
        dtype=np.float64,
    )

    available = np.ones(
        n,
        dtype=bool,
    )

    selected_indices = []

    for step in range(sample_size):

        marginal_gain = (
            (sim - best[None, :]).clip(min=0.0)
            * weights[None, :]
        ).sum(axis=1)

        marginal_gain[~available] = -np.inf

        chosen = int(
            np.argmax(marginal_gain)
        )

        if not np.isfinite(marginal_gain[chosen]):
            break

        selected_indices.append(chosen)

        available[chosen] = False

        best = np.maximum(
            best,
            sim[chosen],
        )

        print(
            f"\rWREP Dev selection: "
            f"{step + 1}/{sample_size}",
            end="",
        )

    print()

    if len(selected_indices) != sample_size:
        raise RuntimeError(
            "WREP selected the wrong number of Dev examples: "
            f"requested={sample_size}, "
            f"got={len(selected_indices)}"
        )

    return [
        examples[index]
        for index in selected_indices
    ]


# ---------------------------------------------------------------------------
# WREP Dev selection
# ---------------------------------------------------------------------------

def select_sess_wrep_examples(
    examples,
    sample_size: int,
    comparator,
    metric,
    model_name: str = "openai/clip-vit-base-patch32",
    device=None,
    cache_dir: str = ".cache/dca_embeddings/wrep",
    wrep_lambda: float = 0.5,
):
    """
    Select the Dev set using SESS-WREP.

    Steps:
        1. Score every candidate with the initial DCA comparator.
        2. difficulty = 1 - DCA score.
        3. Normalize difficulty.
        4. Convert difficulty into WREP weights.
        5. Generate CLIP representations.
        6. Perform weighted facility-location selection.

    Returns:
        dev_set
        scored_examples
    """

    import numpy as np

    examples = list(examples)

    if sample_size <= 0:
        raise ValueError(
            "sample_size must be positive."
        )

    if sample_size > len(examples):
        raise ValueError(
            f"Requested Dev size={sample_size}, "
            f"but only {len(examples)} candidates exist."
        )

    if not 0.0 <= wrep_lambda <= 1.0:
        raise ValueError(
            "wrep_lambda must be between 0 and 1. "
            f"Received {wrep_lambda}."
        )

    print("\n" + "=" * 80)
    print("SESS-WREP DEV SELECTION")
    print("=" * 80)

    print(
        f"Candidates         : {len(examples)}"
    )
    print(
        f"Requested Dev size : {sample_size}"
    )
    print(
        f"WREP lambda        : {wrep_lambda}"
    )

    # -----------------------------------------------------------------------
    # STEP 1: Initial DCA score
    # -----------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("STEP 1: INITIAL DCA SCORING")
    print("-" * 80)

    scored_examples = []

    for index, example in enumerate(
        examples,
        start=1,
    ):
        prediction = comparator(
            image_original=example.image_original,
            image_reimage=example.image_reimage,
        )

        score = float(
            metric(
                example,
                prediction,
            )
        )

        score = max(
            0.0,
            min(1.0, score),
        )

        difficulty = 1.0 - score

        scored_examples.append(
            {
                "example": example,
                "score": score,
                "difficulty": difficulty,
            }
        )

        print(
            f"\rWREP scoring: "
            f"{index}/{len(examples)}",
            end="",
        )

    print()

    # -----------------------------------------------------------------------
    # STEP 2: Normalize difficulty
    # -----------------------------------------------------------------------

    difficulties = np.asarray(
        [
            item["difficulty"]
            for item in scored_examples
        ],
        dtype=np.float64,
    )

    min_difficulty = float(
        difficulties.min()
    )

    max_difficulty = float(
        difficulties.max()
    )

    if (
        max_difficulty
        - min_difficulty
        < 1e-12
    ):
        normalized_difficulty = np.ones_like(
            difficulties
        )
    else:
        normalized_difficulty = (
            difficulties
            - min_difficulty
        ) / (
            max_difficulty
            - min_difficulty
        )

    # -----------------------------------------------------------------------
    # STEP 3: WREP weights
    # -----------------------------------------------------------------------
    #
    # lambda = 0:
    #     all examples receive weight 1 -> pure SESS-REP
    #
    # lambda = 0.5:
    #     balanced representativeness + difficulty emphasis
    #
    # lambda = 1:
    #     weight follows normalized difficulty
    #
    # We use:
    #
    #     w_j = (1-lambda) + lambda*difficulty_j
    #
    # so low-difficulty examples are not completely ignored when
    # lambda < 1.
    # -----------------------------------------------------------------------

    weights = (
        (1.0 - wrep_lambda)
        + (
            wrep_lambda
            * normalized_difficulty
        )
    )

    print("\nWREP weight statistics:")
    print(
        f"Minimum weight : "
        f"{weights.min():.4f}"
    )
    print(
        f"Maximum weight : "
        f"{weights.max():.4f}"
    )
    print(
        f"Average weight : "
        f"{weights.mean():.4f}"
    )

    # -----------------------------------------------------------------------
    # STEP 4: CLIP representation
    # -----------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("STEP 2: CLIP REPRESENTATION")
    print("-" * 80)

    # Important:
    # This is intentionally a WREP-specific cache directory.
    embeddings = _embed_examples_wrep(
        examples=examples,
        model_name=model_name,
        cache_dir=cache_dir,
        device=device,
    )

    # -----------------------------------------------------------------------
    # STEP 5: Weighted facility-location
    # -----------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("STEP 3: WEIGHTED FACILITY-LOCATION")
    print("-" * 80)

    dev_set = _greedy_wrep_facility_location(
        examples=examples,
        embeddings=embeddings,
        sample_size=sample_size,
        weights=weights,
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    score_by_id = {
        _example_id(item["example"]):
            item["score"]
        for item in scored_examples
    }

    difficulty_by_id = {
        _example_id(item["example"]):
            item["difficulty"]
        for item in scored_examples
    }

    print("\n" + "-" * 80)
    print("WREP DEV SUMMARY")
    print("-" * 80)

    for example in dev_set:
        example_id = _example_id(
            example
        )

        print(
            f"{example_id} | "
            f"score="
            f"{score_by_id[example_id]:.4f} | "
            f"difficulty="
            f"{difficulty_by_id[example_id]:.4f}"
        )

    print(
        f"\nSESS-WREP Dev selected: "
        f"{len(dev_set)}"
    )

    return dev_set, scored_examples


# ---------------------------------------------------------------------------
# Complete WREP experiment:
# Dev first -> Train from remaining candidates
# ---------------------------------------------------------------------------

def select_sess_wrep_dev_and_wrep_train(
    examples,
    train_size: int,
    dev_size: int,
    comparator,
    metric,
    model_name: str = "openai/clip-vit-base-patch32",
    device=None,
    cache_dir: str = ".cache/dca_embeddings/wrep",
    wrep_lambda: float = 0.5,
    seed: int = 42,
):
    """
    Complete SESS-WREP sampling for BOTH Dev and Train.

    The selection order is intentionally: 

        Full Train pool
            -> WREP Dev selection
            -> remove Dev
            -> WREP Train selection from remaining candidates

    Both subsets use exactly the same WREP logic:
        1. initial DCA score
        2. difficulty = 1 - score
        3. min-max difficulty normalization
        4. WREP weights: (1-lambda) + lambda*normalized_difficulty
        5. CLIP pair embeddings
        6. greedy weighted facility-location

    Validation/Test are not passed to or modified by this function.
    """
    examples = list(examples)

    if train_size <= 0 or dev_size <= 0:
        raise ValueError("train_size and dev_size must be positive.")

    if train_size + dev_size > len(examples):
        raise ValueError(
            f"Requested train+dev={train_size + dev_size}, "
            f"but only {len(examples)} candidates are available."
        )

    if not 0.0 <= wrep_lambda <= 1.0:
        raise ValueError(
            "wrep_lambda must be between 0 and 1. "
            f"Received {wrep_lambda}."
        )

    print("\n" + "=" * 80)
    print("SESS-WREP: DEV FIRST -> WREP TRAIN")
    print("=" * 80)
    print(f"Original Train pool : {len(examples)}")
    print(f"Requested Dev       : {dev_size}")
    print(f"Requested Train     : {train_size}")
    print(f"WREP lambda         : {wrep_lambda}")

    # ------------------------------------------------------------------
    # STEP 1: WREP DEV
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: WREP DEV SELECTION")
    print("=" * 80)

    dev_set, dev_scored_examples = select_sess_wrep_examples(
        examples=examples,
        sample_size=dev_size,
        comparator=comparator,
        metric=metric,
        model_name=model_name,
        device=device,
        cache_dir=str(Path(cache_dir) / "dev"),
        wrep_lambda=wrep_lambda,
    )

    dev_ids = {_example_id(example) for example in dev_set}

    # ------------------------------------------------------------------
    # STEP 2: REMOVE DEV
    # ------------------------------------------------------------------
    train_candidates = [
        example for example in examples
        if _example_id(example) not in dev_ids
    ]

    if len(train_candidates) < train_size:
        raise ValueError(
            f"Only {len(train_candidates)} candidates remain after Dev "
            f"selection, but Train size={train_size} was requested."
        )

    print("\n" + "=" * 80)
    print("STEP 2: DEV RESERVED")
    print("=" * 80)
    print(f"Original Train pool : {len(examples)}")
    print(f"WREP Dev            : {len(dev_set)}")
    print(f"Remaining candidates: {len(train_candidates)}")

    # ------------------------------------------------------------------
    # STEP 3: WREP TRAIN -- SAME LOGIC AS DEV
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: WREP TRAIN SELECTION")
    print("=" * 80)

    train_set, train_scored_examples = select_sess_wrep_examples(
        examples=train_candidates,
        sample_size=train_size,
        comparator=comparator,
        metric=metric,
        model_name=model_name,
        device=device,
        cache_dir=str(Path(cache_dir) / "train"),
        wrep_lambda=wrep_lambda,
    )

    # ------------------------------------------------------------------
    # STEP 4: FINAL OVERLAP / SIZE CHECKS
    # ------------------------------------------------------------------
    train_ids = {_example_id(example) for example in train_set}
    train_dev_overlap = train_ids & dev_ids

    if train_dev_overlap:
        raise RuntimeError(
            "SESS-WREP Train/Dev overlap detected: "
            f"{sorted(train_dev_overlap)}"
        )

    if len(dev_set) != dev_size:
        raise RuntimeError(
            f"Incorrect Dev size: requested={dev_size}, got={len(dev_set)}"
        )

    if len(train_set) != train_size:
        raise RuntimeError(
            f"Incorrect Train size: requested={train_size}, got={len(train_set)}"
        )

    # Make it explicit that selected Train + Dev are drawn only from the
    # original candidate pool. This also protects against accidental
    # selector bugs that create foreign examples.
    original_ids = {_example_id(example) for example in examples}
    if not train_ids <= original_ids or not dev_ids <= original_ids:
        raise RuntimeError("WREP selected an example outside the Train pool.")

    train_info = {
        "method": "wrep_weighted_facility_location",
        "train_size": train_size,
        "dev_size": dev_size,
        "wrep_lambda": wrep_lambda,
        "seed": seed,
        "selection_order": "dev_first_then_train_from_remaining",
        "train_scored_examples": train_scored_examples,
        "dev_scored_examples": dev_scored_examples,
    }

    print("\n" + "=" * 80)
    print("SESS-WREP SAMPLING COMPLETE")
    print("=" * 80)
    print(f"Original Train pool : {len(examples)}")
    print(f"WREP Dev            : {len(dev_set)}")
    print(f"Remaining candidates: {len(train_candidates)}")
    print(f"WREP Train          : {len(train_set)}")
    print(f"Train/Dev overlap   : {len(train_dev_overlap)}")
    print("Validation/Test     : untouched by this function")
    print("=" * 80)

    return (
        train_set,
        dev_set,
        train_candidates,
        train_info,
        dev_scored_examples,
    )


# Backward-compatible name for callers that still use the old function.
def select_sess_wrep_dev_and_difficulty_train(*args, **kwargs):
    return select_sess_wrep_dev_and_wrep_train(*args, **kwargs)


__all__ = [
    "select_sess_wrep_examples",
    "select_sess_wrep_dev_and_wrep_train",
    "select_sess_wrep_dev_and_difficulty_train",
]
