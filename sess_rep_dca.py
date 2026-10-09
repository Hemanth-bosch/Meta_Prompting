"""SESS-style subset selection for DCA.

DCA adaptation:
- Dev selection: SESS-REP facility-location selection.
- Train selection: existing IPOMP-Difficulty Hard + Diverse logic.

The Dev set is selected with SESS-REP and reserved first. The remaining
candidates are then passed through the same difficulty -> hard-pool ->
CLIP max-min diversity process used by the DCA IPOMP-Difficulty selector.
"""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from PIL import Image as PILImage
from typing import Any, Optional


def _example_id(example: Any) -> str:
    pair_name = getattr(example, "pair_name", None)
    if pair_name is not None:
        return str(pair_name)
    example_id = getattr(example, "id", None)
    if example_id is not None:
        return str(example_id)
    return str(id(example))


def _cosine_similarity(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _get_example_field(example: Any, field_name: str):
    """
    Robustly retrieve a field from a DSPy Example.

    Tries:
    1. Direct attribute access.
    2. DSPy's toDict() representation.
    3. Common internal dictionary stores.
    """

    # 1. Direct attribute
    try:
        value = getattr(example, field_name, None)
    except Exception:
        value = None

    if value is not None:
        return value

    # 2. DSPy toDict()
    try:
        if hasattr(example, "toDict"):
            data = example.toDict()
            if isinstance(data, dict):
                value = data.get(field_name)
                if value is not None:
                    return value
    except Exception:
        pass

    # 3. Common internal dictionaries
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


def _has_both_images(example: Any) -> bool:
    """Return True only when both images exist and can be used for CLIP."""
    original = _get_example_field(example, "image_original")
    reimage = _get_example_field(example, "image_reimage")
    return original is not None and reimage is not None


def _image_pair_status(examples):
    """Count complete and incomplete DCA image pairs."""
    complete = []
    incomplete = []
    for example in examples:
        if _has_both_images(example):
            complete.append(example)
        else:
            incomplete.append(example)
    return complete, incomplete


class DCAPairEmbedder:
    """CLIP embedding of a DCA image pair."""

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

    def _load_model(self):
        if self.model is not None:
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "SESS-REP requires torch and transformers."
            ) from exc

        self.torch = torch

        if self.device is None:
            self.device = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        print(
            f"Loading CLIP for SESS-REP: {self.model_name}"
        )

        self.processor = CLIPProcessor.from_pretrained(
            self.model_name
        )
        self.model = CLIPModel.from_pretrained(
            self.model_name
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_image(value):
        """
        Convert a DCA image value into a PIL RGB image.

        Supports:
        - PIL images
        - filesystem paths
        - file:// URLs
        - HTTP/HTTPS URLs
        - data:image/... Base64 strings
        - bytes
        - DSPy/custom image objects
        - nested custom image representations
        """

        from io import BytesIO
        import ast
        import base64
        from urllib.parse import unquote, urlparse

        from PIL import Image

        if value is None:
            raise TypeError("DCA image value is None")

        # ---------------------------------------------------------
        # 1. Already a PIL image
        # ---------------------------------------------------------
        if isinstance(value, Image.Image):
            return value.convert("RGB")

        # ---------------------------------------------------------
        # 2. Raw bytes
        # ---------------------------------------------------------
        if isinstance(value, (bytes, bytearray)):
            return Image.open(BytesIO(value)).convert("RGB")

        # ---------------------------------------------------------
        # 3. Helper for strings
        # ---------------------------------------------------------
        def _open_string(text):
            text = str(text).strip()

            if not text:
                return None

            # Local filesystem path
            try:
                path = Path(text)
                if path.exists() and path.is_file():
                    return Image.open(path).convert("RGB")
            except (OSError, ValueError):
                pass

            # data:image/...;base64,...
            if text.startswith("data:image") and "," in text:
                header, encoded = text.split(",", 1)
                if ";base64" in header:
                    try:
                        image_bytes = base64.b64decode(
                            encoded,
                            validate=False,
                        )
                        return Image.open(
                            BytesIO(image_bytes)
                        ).convert("RGB")
                    except Exception:
                        pass

            # file:// URL
            if text.startswith("file://"):
                try:
                    parsed = urlparse(text)
                    file_path = unquote(parsed.path)

                    if parsed.netloc and parsed.netloc not in (
                        "",
                        "localhost",
                    ):
                        file_path = f"//{parsed.netloc}{file_path}"

                    path = Path(file_path)
                    if path.exists() and path.is_file():
                        return Image.open(path).convert("RGB")
                except Exception:
                    pass

            # HTTP / HTTPS URL
            if text.startswith(("http://", "https://")):
                try:
                    import requests

                    response = requests.get(
                        text,
                        timeout=30,
                    )
                    response.raise_for_status()
                    return Image.open(
                        BytesIO(response.content)
                    ).convert("RGB")
                except Exception:
                    pass

            # Literal string representation of a path
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, str):
                    path = Path(parsed)
                    if path.exists() and path.is_file():
                        return Image.open(path).convert("RGB")
            except Exception:
                pass

            return None

        # ---------------------------------------------------------
        # 4. Direct string / Path
        # ---------------------------------------------------------
        if isinstance(value, (str, Path)):
            image = _open_string(value)
            if image is not None:
                return image

        # ---------------------------------------------------------
        # 5. Common custom/DSPy attributes
        # ---------------------------------------------------------
        for attr in (
            "url",
            "path",
            "local_path",
            "filename",
            "file_path",
            "data",
            "content",
            "value",
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
                    return Image.open(
                        BytesIO(attr_value)
                    ).convert("RGB")
                except Exception:
                    pass

            image = _open_string(attr_value)
            if image is not None:
                return image

        # ---------------------------------------------------------
        # 6. Nested image objects
        # ---------------------------------------------------------
        for attr in (
            "image",
            "pil_image",
            "image_data",
        ):
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
                    return Image.open(
                        BytesIO(image_value)
                    ).convert("RGB")
                except Exception:
                    pass

            image = _open_string(image_value)
            if image is not None:
                return image

        # ---------------------------------------------------------
        # 7. Custom serialized representation
        # ---------------------------------------------------------
        # The DCA image may appear in the string representation as:
        # data:image/png;base64,...
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

    def encode_pair(self, example: Any):
        """
        Create one fixed-size CLIP representation for a DCA example.

        Rules:
        - Both images present:
              pair_embedding = average(original_embedding, reimage_embedding)
        - Only original present:
              pair_embedding = original_embedding
        - Only reimage present:
              pair_embedding = reimage_embedding
        - Neither image present:
              raise an error because there is no visual information.
        """
        self._load_model()
        torch = self.torch

        example_id = _example_id(example)

        # ---------------------------------------------------------
        # Retrieve image fields robustly from the DSPy Example
        # ---------------------------------------------------------
        original_value = _get_example_field(
            example,
            "image_original",
        )

        reimage_value = _get_example_field(
            example,
            "image_reimage",
        )

        has_original = original_value is not None
        has_reimage = reimage_value is not None

        print(
            f"\nSESS-REP embedding example: {example_id}"
        )
        print(
            "image_original type:",
            type(original_value),
        )
        print(
            "image_reimage type:",
            type(reimage_value),
        )

        # ---------------------------------------------------------
        # Neither image exists
        # ---------------------------------------------------------
        if not has_original and not has_reimage:
            raise ValueError(
                "Both image_original and image_reimage are missing "
                f"for example {example_id}."
            )

        # ---------------------------------------------------------
        # Load only the images that actually exist
        # ---------------------------------------------------------
        images = []
        image_names = []

        if has_original:
            original = self._load_image(original_value)
            images.append(original)
            image_names.append("original")

        if has_reimage:
            reimage = self._load_image(reimage_value)
            images.append(reimage)
            image_names.append("reimage")

        # ---------------------------------------------------------
        # Generate CLIP embeddings
        # ---------------------------------------------------------
        with torch.no_grad():
            inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True,
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            image_features = self.model.get_image_features(
                **inputs
            )

            if not isinstance(image_features, torch.Tensor):
                if hasattr(image_features, "pooler_output"):
                    image_features = image_features.pooler_output
                elif hasattr(image_features, "last_hidden_state"):
                    image_features = (
                        image_features.last_hidden_state[:, 0, :]
                    )
                else:
                    raise TypeError(
                        "Unexpected CLIP image feature output type: "
                        f"{type(image_features)!r}"
                    )

            # Normalize each image embedding individually
            image_features = (
                image_features
                / image_features.norm(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)
            )

            # -----------------------------------------------------
            # BOTH images available
            # -----------------------------------------------------
            if has_original and has_reimage:
                pair = (
                    image_features[0]
                    + image_features[1]
                ) / 2.0

                representation_type = "AVERAGE(original, reimage)"

            # -----------------------------------------------------
            # ONLY original available
            # -----------------------------------------------------
            elif has_original:
                pair = image_features[0]
                representation_type = "ORIGINAL_ONLY"

            # -----------------------------------------------------
            # ONLY reimage available
            # -----------------------------------------------------
            else:
                pair = image_features[0]
                representation_type = "REIMAGE_ONLY"

            # Final normalization
            pair = pair / pair.norm().clamp_min(1e-12)

        print(
            f"SESS-REP representation: {representation_type}"
        )

        return pair.detach().cpu().numpy()

    def encode_examples(self, examples):
        import numpy as np

        return np.asarray(
            [self.encode_pair(example) for example in examples],
            dtype=np.float32,
        )


def _cache_key(example, model_name: str) -> str:
    original_value = _get_example_field(
        example,
        "image_original",
    )

    reimage_value = _get_example_field(
        example,
        "image_reimage",
    )

    raw = "|".join(
        [
            _example_id(example),
            str(original_value),
            str(reimage_value),
            model_name,
        ]
    )

    return hashlib.sha256(
        raw.encode(
            "utf-8",
            errors="replace",
        )
    ).hexdigest()


def _embed_examples(examples, model_name, cache_dir, device=None):
    import numpy as np

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    embedder = DCAPairEmbedder(model_name=model_name, device=device)

    embeddings = []
    for idx, example in enumerate(examples, 1):
        path = cache_path / f"sess_rep_{_cache_key(example, model_name)}.npy"
        if path.exists():
            emb = np.load(path)
        else:
            emb = embedder.encode_pair(example)
            np.save(path, emb)
        embeddings.append(emb)
        if idx % 25 == 0 or idx == len(examples):
            print(f"Embedding examples: {idx}/{len(examples)}")

    return np.asarray(embeddings, dtype=np.float32)


def _greedy_facility_location(examples, embeddings, sample_size, weights=None):
    """Greedy SESS-REP / WREP facility-location selection."""
    import numpy as np

    n = len(examples)
    if sample_size <= 0:
        return []
    if sample_size > n:
        raise ValueError(f"sample_size={sample_size} exceeds candidates={n}")

    # Pairwise cosine -> SESS similarity in [0, 1].
    sim = embeddings @ embeddings.T
    sim = np.clip((sim + 1.0) / 2.0, 0.0, 1.0)

    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    best = np.zeros(n, dtype=np.float64)
    available = np.ones(n, dtype=bool)
    selected_indices = []

    for step in range(sample_size):
        # gain(x) = sum_j w_j [max(sim(x,j), best_j) - best_j]
        gains = ((sim - best[None, :]).clip(min=0.0) * weights[None, :]).sum(axis=1)
        gains[~available] = -np.inf
        chosen = int(np.argmax(gains))

        if not np.isfinite(gains[chosen]):
            break

        selected_indices.append(chosen)
        available[chosen] = False
        best = np.maximum(best, sim[chosen])

        if (step + 1) % 10 == 0 or step + 1 == sample_size:
            print(f"SESS selection: {step + 1}/{sample_size}")

    return [examples[i] for i in selected_indices]


def select_sess_rep_examples(
    examples,
    sample_size,
    model_name="openai/clip-vit-base-patch32",
    device=None,
    cache_dir=".cache/dca_embeddings",
    seed=42
):
    """Select a representative subset using SESS-REP."""
    examples = list(examples)

    # ------------------------------------------------------------
    # Count complete / incomplete examples for reporting.
    # ------------------------------------------------------------
    complete_examples, incomplete_examples = _image_pair_status(
        examples
    )

    # Only examples with at least one image can receive a visual
    # representation. Examples with neither image are excluded.
    embeddable_examples = [
        example
        for example in examples
        if (
            _get_example_field(example, "image_original") is not None
            or
            _get_example_field(example, "image_reimage") is not None
        )
    ]

    non_embeddable_examples = [
        example
        for example in examples
        if (
            _get_example_field(example, "image_original") is None
            and
            _get_example_field(example, "image_reimage") is None
        )
    ]

    print(
        f"SESS-REP candidates: {len(examples)} total | "
        f"{len(complete_examples)} complete pairs | "
        f"{len(incomplete_examples)} incomplete pairs"
    )

    print(
        f"SESS-REP embeddable candidates: "
        f"{len(embeddable_examples)}"
    )

    print(
        f"SESS-REP non-embeddable candidates: "
        f"{len(non_embeddable_examples)}"
    )

    if sample_size > len(embeddable_examples):
        raise ValueError(
            "SESS-REP cannot select the requested Dev size because "
            f"only {len(embeddable_examples)} candidates contain "
            "at least one image, but "
            f"{sample_size} were requested."
        )

    # ------------------------------------------------------------
    # Embed ALL candidates that have at least one image.
    #
    # Complete pair:
    #     average(original_embedding, reimage_embedding)
    #
    # Original only:
    #     original_embedding
    #
    # Reimage only:
    #     reimage_embedding
    # ------------------------------------------------------------
    embeddings = _embed_examples(
        embeddable_examples,
        model_name,
        cache_dir,
        device,
    )

    # ------------------------------------------------------------
    # SESS-REP facility-location selection
    # ------------------------------------------------------------
    selected = _greedy_facility_location(
        embeddable_examples,
        embeddings,
        sample_size,
    )

    return selected


def select_ipomp_difficulty_train_examples(
    examples,
    sample_size,
    comparator,
    metric,
    model_name="openai/clip-vit-base-patch32",
    device=None,
    cache_dir=".cache/dca_embeddings",
    hard_pool_size=100,
    seed=42,
):
    """Select Train using the existing IPOMP-Difficulty Hard + Diverse flow.

    Workflow (kept the same as the DCA IPOMP-Difficulty implementation):

        remaining candidates
              -> baseline comparator
              -> DCA score
              -> difficulty = 1 - score
              -> hardest candidate pool (default 100)
              -> CLIP max-min diversity
              -> final Train

    The CLIP max-min step starts from one difficult example and then repeatedly
    chooses the candidate whose minimum CLIP distance to the selected set is
    largest. This is intentionally different from SESS-WREP.
    """
    examples = list(examples)
    if sample_size > len(examples):
        raise ValueError(
            f"Requested {sample_size}, but only {len(examples)} examples are available."
        )
    if hard_pool_size <= 0:
        raise ValueError("hard_pool_size must be positive.")

    print("\n" + "=" * 80)
    print("IPOMP-DIFFICULTY HARD TRAINING CANDIDATE SELECTION")
    print("=" * 80)
    print(f"Candidate examples : {len(examples)}")
    print(f"Hard pool size     : {hard_pool_size}")

    # ------------------------------------------------------------
    # 1. Baseline scoring and DCA difficulty
    # ------------------------------------------------------------
    scored_examples = []
    for index, example in enumerate(examples, start=1):
        prediction = comparator(
            image_original=example.image_original,
            image_reimage=example.image_reimage,
        )
        score = float(metric(example, prediction))
        score = max(0.0, min(1.0, score))
        difficulty = 1.0 - score

        scored_examples.append({
            "example": example,
            "score": score,
            "difficulty": difficulty,
        })

        print(f"\rScoring {index}/{len(examples)}", end="")
    print()

    # ------------------------------------------------------------
    # 2. Keep the hardest candidate pool
    # ------------------------------------------------------------
    scored_examples.sort(key=lambda item: item["difficulty"], reverse=True)

    # The difficulty score can be computed for fallback pairs because the
    # DCA comparator supports missing images. However, the later CLIP
    # max-min step requires BOTH images. Therefore build the hard pool from
    # the hardest COMPLETE pairs only. This prevents a None image from ever
    # reaching the CLIP embedder.
    complete_scored = [
        item
        for item in scored_examples
        if _has_both_images(item["example"])
    ]
    incomplete_scored = [
        item
        for item in scored_examples
        if not _has_both_images(item["example"])
    ]

    actual_hard_pool_size = min(
        hard_pool_size,
        len(complete_scored),
    )
    hard_pool = [
        item["example"]
        for item in complete_scored[:actual_hard_pool_size]
    ]

    print(
        f"\nComplete pairs available for CLIP : {len(complete_scored)}"
    )
    print(
        f"Incomplete pairs excluded from CLIP: {len(incomplete_scored)}"
    )
    print(f"Selected hard candidate pool       : {len(hard_pool)}")
    print("\nHardest examples:")
    for item in scored_examples[:10]:
        print(
            f"  {_example_id(item['example'])}"
            f" | score={item['score']:.4f}"
            f" | difficulty={item['difficulty']:.4f}"
        )

    if len(hard_pool) < sample_size:
        raise RuntimeError(
            "Not enough complete image pairs for CLIP-based Train selection. "
            f"Requested Train={sample_size}, but only {len(hard_pool)} "
            "complete pairs are available in the hard pool."
        )

    # If the hard pool itself is exactly the requested train size, return it.
    if len(hard_pool) == sample_size:
        return hard_pool, scored_examples

    # ------------------------------------------------------------
    # 3. CLIP max-min diversity selection from the hard pool
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("HARD + DIVERSE TRAINING SELECTION")
    print("=" * 80)

    embeddings = _embed_examples(hard_pool, model_name, cache_dir, device)
    embedding_by_id = {
        _example_id(example): embeddings[i]
        for i, example in enumerate(hard_pool)
    }

    rng = random.Random(seed)
    first_example = rng.choice(hard_pool)
    selected = [first_example]

    while len(selected) < sample_size:
        selected_ids = {_example_id(example) for example in selected}
        remaining = [
            example for example in hard_pool
            if _example_id(example) not in selected_ids
        ]
        if not remaining:
            break

        # Existing IPOMP-Difficulty rule:
        # choose the candidate with the largest minimum distance to
        # any already-selected example.
        candidate = max(
            remaining,
            key=lambda example: min(
                1.0 - _cosine_similarity(
                    embedding_by_id[_example_id(example)],
                    embedding_by_id[_example_id(selected_example)],
                )
                for selected_example in selected
            ),
        )
        selected.append(candidate)

        if len(selected) % 10 == 0 or len(selected) == sample_size:
            print(f"IPOMP-Difficulty selection: {len(selected)}/{sample_size}")

    if len(selected) != sample_size:
        raise RuntimeError(
            "IPOMP-Difficulty hard + diverse selection produced the wrong "
            f"Train size. Requested={sample_size}, got={len(selected)}"
        )

    return selected, scored_examples


def select_sess_rep_dev_and_difficulty_train(
    examples,
    train_size,
    dev_size,
    comparator,
    metric,
    model_name="openai/clip-vit-base-patch32",
    device=None,
    cache_dir=".cache/dca_embeddings",
    hard_pool_size=100,
    seed=42,
):
    """Select Dev with SESS-REP first, then Train with IPOMP-Difficulty Hard + Diverse.

    Dev is reserved first. Train is selected only from the remaining examples,
    guaranteeing no Train/Dev overlap.
    """
    examples = list(examples)
    if train_size + dev_size > len(examples):
        raise ValueError(
            f"Requested train+dev={train_size + dev_size}, available={len(examples)}"
        )

    print("\n" + "=" * 80)
    print("SESS-REP DEV SELECTION")
    print("=" * 80)
    dev_set = select_sess_rep_examples(
        examples=examples,
        sample_size=dev_size,
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
    )

    dev_ids = {_example_id(x) for x in dev_set}
    train_candidates = [x for x in examples if _example_id(x) not in dev_ids]

    print(f"\nSESS-REP Dev selected : {len(dev_set)}")
    print(f"Remaining train candidates: {len(train_candidates)}")

    print("\n" + "=" * 80)
    print("IPOMP-DIFFICULTY HARD + DIVERSE TRAIN SELECTION")
    print("=" * 80)
    train_set, train_info = select_ipomp_difficulty_train_examples(
        examples=train_candidates,
        sample_size=train_size,
        comparator=comparator,
        metric=metric,
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
        hard_pool_size=hard_pool_size,
        seed=seed,
    )

    train_ids = {_example_id(x) for x in train_set}
    overlap = train_ids & dev_ids
    if overlap:
        raise RuntimeError(f"SESS-REP/IPOMP-Difficulty produced Train/Dev overlap: {sorted(overlap)}")

    return train_set, dev_set, train_candidates, train_info
