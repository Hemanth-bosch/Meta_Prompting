"""
DCA Diverse + Boundary Sampling

Stage A experiment:
    - Operates ONLY on the candidate training pool.
    - Does NOT access validation or test examples.
    - Generates image-pair embeddings.
    - Performs K-means clustering.
    - Selects representative/diverse examples.
    - Selects boundary examples.
    - Returns a fixed-size training subset.

The existing GEPA Dev/Validation/Test sets are not modified.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from transformers import CLIPModel, CLIPProcessor


# ------------------------------------------------------------
# CLIP MODEL
# ------------------------------------------------------------

_MODEL = None
_PROCESSOR = None


def _load_clip():
    global _MODEL, _PROCESSOR

    if _MODEL is None:
        model_name = "openai/clip-vit-base-patch32"

        print("\nLoading CLIP model...")
        print(f"Model: {model_name}")

        _PROCESSOR = CLIPProcessor.from_pretrained(model_name)
        _MODEL = CLIPModel.from_pretrained(model_name)

        _MODEL.eval()

        print("CLIP model loaded.\n")

    return _MODEL, _PROCESSOR


# ------------------------------------------------------------
# IMAGE LOADING
# ------------------------------------------------------------
def _load_image(image_value):
    """
    Convert a DSPy Image object into a PIL Image.

    DCA examples use:
        dspy.adapters.types.image.Image

    The DSPy image may contain the actual image as bytes/base64
    or expose a URL/path representation.
    """

    # --------------------------------------------------------
    # 1. Already a PIL image
    # --------------------------------------------------------
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")

    # --------------------------------------------------------
    # 2. Normal local path
    # --------------------------------------------------------
    if isinstance(image_value, (str, Path)):
        path = Path(image_value)

        if path.exists():
            return Image.open(path).convert("RGB")

        raise FileNotFoundError(
            f"Image path does not exist: {image_value}"
        )

    # --------------------------------------------------------
    # 3. DSPy Image object
    # --------------------------------------------------------

    # Check for an underlying PIL image.
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

        if isinstance(value, Image.Image):
            return value.convert("RGB")

    # --------------------------------------------------------
    # 4. Check for image bytes
    # --------------------------------------------------------

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

        if isinstance(value, bytes):
            from io import BytesIO

            return Image.open(
                BytesIO(value)
            ).convert("RGB")

    # --------------------------------------------------------
    # 5. Check for local file/path attributes
    # --------------------------------------------------------

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
            path = Path(str(value))

            if path.exists():
                return Image.open(
                    path
                ).convert("RGB")

    # --------------------------------------------------------
    # 6. URL -- DO NOT pass URL to Image.open()
    # --------------------------------------------------------

        # --------------------------------------------------------
    # 6. DSPy Image containing a data URL
    # --------------------------------------------------------

    url = getattr(image_value, "url", None)

    if url:
        if isinstance(url, str) and url.startswith("data:image/"):
            try:
                import base64
                from io import BytesIO

                # Format:
                # data:image/png;base64,<BASE64_DATA>
                header, encoded = url.split(",", 1)

                image_bytes = base64.b64decode(encoded)

                return Image.open(
                    BytesIO(image_bytes)
                ).convert("RGB")

            except Exception as e:
                raise TypeError(
                    "Failed to decode DSPy DCA "
                    f"data URL: {e}"
                ) from e

        raise TypeError(
            "Unsupported DSPy image URL format: "
            f"{url[:100]}"
        )

    # --------------------------------------------------------
    # 7. Nothing worked
    # --------------------------------------------------------

    raise TypeError(
        "\nCould not extract a PIL image from:\n"
        f"Type: {type(image_value)}\n"
        f"Attributes: {dir(image_value)}"
    )
def _embed_image(image_value):
    """
    Generate a normalized CLIP embedding for one image.
    """

    model, processor = _load_clip()

    image = _load_image(image_value)

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # Depending on the installed Transformers version,
    # get_image_features() may return either:
    #
    #   1. a Tensor
    #   2. BaseModelOutputWithPooling
    #
    # Handle both cases.

    if isinstance(outputs, torch.Tensor):
        features = outputs

    elif hasattr(outputs, "pooler_output"):
        features = outputs.pooler_output

    elif hasattr(outputs, "last_hidden_state"):
        # Fallback for CLIP model outputs.
        features = outputs.last_hidden_state[:, 0, :]

    else:
        raise TypeError(
            "Unexpected CLIP output type: "
            f"{type(outputs)}"
        )

    # Make sure we have a 2-D tensor.
    if features.ndim == 1:
        features = features.unsqueeze(0)

    # Normalize the embedding.
    features = features / features.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)

    return features[0].cpu().numpy()
# ------------------------------------------------------------
# EMBEDDING CACHE
# ------------------------------------------------------------

def _cache_key(example):
    pair_name = str(
        getattr(example, "pair_name", "unknown")
    )

    return hashlib.sha1(
        pair_name.encode("utf-8")
    ).hexdigest()[:16]

def _pair_embedding(example):
    """
    Create one embedding for a DCA image pair.

    Each DCA example contains:
        image_original
        image_reimage

    We generate a CLIP embedding for both images and combine:

        original embedding
        +
        reimage embedding
        +
        absolute difference
    """

    original_embedding = _embed_image(
        example.image_original
    )

    reimage_embedding = _embed_image(
        example.image_reimage
    )

    difference = np.abs(
        original_embedding - reimage_embedding
    )

    combined = np.concatenate(
        [
            original_embedding,
            reimage_embedding,
            difference,
        ]
    )

    # Normalize the final pair representation.
    norm = np.linalg.norm(combined)

    if norm > 0:
        combined = combined / norm

    return combined




def generate_pair_embeddings(
    examples,
    cache_dir=".cache/dca_embeddings",
):
    """
    Generate embeddings for all supplied examples.

    IMPORTANT:
    The caller must pass ONLY the allowed training pool.

    Embeddings are cached so repeated experiments do not
    repeatedly run CLIP.
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings = []

    print("\n" + "=" * 80)
    print("DCA IMAGE-PAIR EMBEDDING")
    print("=" * 80)

    print(f"Examples to embed: {len(examples)}")
    print(f"Cache directory  : {cache_dir}")

    for index, example in enumerate(
        examples,
        start=1,
    ):

        cache_file = (
            cache_dir
            / f"{_cache_key(example)}.npy"
        )

        if cache_file.exists():
            embedding = np.load(cache_file)

        else:
            print(
                f"Embedding {index:3d}/{len(examples)}: "
                f"{getattr(example, 'pair_name', index)}"
            )

            embedding = _pair_embedding(example)

            np.save(
                cache_file,
                embedding,
            )

        embeddings.append(embedding)

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    print(
        f"\nGenerated embeddings: "
        f"{embeddings.shape}"
    )

    print("=" * 80)

    return embeddings


# ------------------------------------------------------------
# REPRESENTATIVE SELECTION
# ------------------------------------------------------------

def _select_representatives(
    embeddings,
    labels,
    centers,
    count,
    seed=42,
):
    """
    Select examples close to cluster centers.

    Selection is distributed across clusters.
    """

    rng = np.random.default_rng(seed)

    n_clusters = len(centers)

    cluster_members = {
        cluster_id: np.where(
            labels == cluster_id
        )[0].tolist()
        for cluster_id in range(n_clusters)
    }

    selected = []

    # Allocate approximately equal quota.
    base = count // n_clusters
    remainder = count % n_clusters

    for cluster_id in range(n_clusters):

        quota = base + (
            1 if cluster_id < remainder
            else 0
        )

        members = cluster_members[
            cluster_id
        ]

        if not members:
            continue

        center = centers[cluster_id]

        distances = [
            (
                np.linalg.norm(
                    embeddings[index] - center
                ),
                index,
            )
            for index in members
        ]

        distances.sort(
            key=lambda x: x[0]
        )

        chosen = [
            index
            for _, index
            in distances[:quota]
        ]

        selected.extend(chosen)

    rng.shuffle(selected)

    return selected[:count]


# ------------------------------------------------------------
# BOUNDARY SELECTION
# ------------------------------------------------------------

def _select_boundary_examples(
    embeddings,
    labels,
    centers,
    count,
    excluded_indices=None,
):
    """
    Select examples furthest from their cluster
    centers.

    These are treated as boundary / unusual examples.
    """

    excluded_indices = (
        set(excluded_indices or [])
    )

    candidates = []

    for index in range(len(embeddings)):

        if index in excluded_indices:
            continue

        cluster_id = labels[index]

        center = centers[cluster_id]

        distance = np.linalg.norm(
            embeddings[index] - center
        )

        candidates.append(
            (
                distance,
                index,
            )
        )

    # Largest distance = strongest boundary candidate.
    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        index
        for _, index in candidates[:count]
    ]

# ------------------------------------------------------------
# INFORMATIVENESS SELECTION
# ------------------------------------------------------------

def _select_informative_examples(
    examples,
    comparator,
    metric,
    count,
):
    """
    Select examples where the current comparator performs poorly.

    Lower DCA metric score = more informative example.
    """

    if count <= 0:
        return []

    scored_examples = []

    print("\n" + "=" * 80)
    print("DCA INFORMATIVENESS SELECTION")
    print("=" * 80)

    print(f"Examples to evaluate : {len(examples)}")
    print(f"Informative samples  : {count}")

    for index, example in enumerate(examples):

        try:
            prediction = comparator(
                image_original=example.image_original,
                image_reimage=example.image_reimage,
            )

            score = metric(
                example,
                prediction,
            )

            if score is not None:
                scored_examples.append(
                    (float(score), index)
                )

        except Exception as exc:
            print(
                f"Could not score example "
                f"{index}: {exc}"
            )

    # Lower score = harder / more informative
    scored_examples.sort(
        key=lambda x: x[0]
    )

    selected_indices = [
        index
        for _, index in scored_examples[:count]
    ]

    print(
        f"Successfully scored : "
        f"{len(scored_examples)}"
    )

    print("=" * 80)

    return selected_indices
# ------------------------------------------------------------
# DIVERSE + BOUNDARY + INFORMATIVE SELECTION
# ------------------------------------------------------------

def select_diverse_boundary_informative_examples(
    examples,
    sample_size,
    n_clusters=5,
    representative_fraction=0.4,
    boundary_fraction=0.3,
    informative_fraction=0.3,
    seed=42,
    cache_dir=".cache/dca_embeddings",
    comparator=None,
    metric=None,
):
    """
    Select a fixed-size subset using:

        1. Diverse / representative examples
        2. Boundary examples
        3. Informative examples

    The three groups are mutually exclusive.
    """

    if comparator is None:
        raise ValueError(
            "comparator is required for "
            "informative selection."
        )

    if metric is None:
        raise ValueError(
            "metric is required for "
            "informative selection."
        )

    examples = list(examples)

    # --------------------------------------------------------
    # VALIDATE FRACTIONS
    # --------------------------------------------------------

    total_fraction = (
        representative_fraction
        + boundary_fraction
        + informative_fraction
    )

    if abs(total_fraction - 1.0) > 1e-6:
        raise ValueError(
            "representative_fraction + "
            "boundary_fraction + "
            "informative_fraction must equal 1.0"
        )

    # --------------------------------------------------------
    # CALCULATE COUNTS
    # --------------------------------------------------------

    representative_count = int(
        round(
            sample_size
            * representative_fraction
        )
    )

    boundary_count = int(
        round(
            sample_size
            * boundary_fraction
        )
    )

    informative_count = (
        sample_size
        - representative_count
        - boundary_count
    )

    diverse_boundary_count = (
        representative_count
        + boundary_count
    )

    # --------------------------------------------------------
    # STEP 1
    # DIVERSE + BOUNDARY
    # --------------------------------------------------------

    diverse_boundary_examples, base_info = (
        select_diverse_boundary_examples(
            examples=examples,
            sample_size=diverse_boundary_count,
            n_clusters=n_clusters,
            representative_fraction=(
                representative_count
                / diverse_boundary_count
            ),
            seed=seed,
            cache_dir=cache_dir,
        )
    )

    # --------------------------------------------------------
    # STEP 2
    # REMOVE DIVERSE + BOUNDARY
    # --------------------------------------------------------

    selected_ids = {
        id(example)
        for example in diverse_boundary_examples
    }

    remaining_examples = [
        example
        for example in examples
        if id(example) not in selected_ids
    ]

    print("\n" + "=" * 80)
    print("POOL AFTER DIVERSE + BOUNDARY")
    print("=" * 80)

    print(
        f"Original pool    : {len(examples)}"
    )

    print(
        f"Already selected : "
        f"{len(diverse_boundary_examples)}"
    )

    print(
        f"Remaining pool   : "
        f"{len(remaining_examples)}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # STEP 3
    # INFORMATIVE SELECTION
    # --------------------------------------------------------

    informative_indices = (
        _select_informative_examples(
            examples=remaining_examples,
            comparator=comparator,
            metric=metric,
            count=informative_count,
        )
    )

    informative_examples = [
        remaining_examples[index]
        for index in informative_indices
    ]

    # --------------------------------------------------------
    # STEP 4
    # FINAL SELECTION
    # --------------------------------------------------------

    selected_examples = (
        diverse_boundary_examples
        + informative_examples
    )

    # Safety check
    if len(selected_examples) != sample_size:
        raise RuntimeError(
            f"Expected {sample_size} examples, "
            f"but selected "
            f"{len(selected_examples)}."
        )

    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print(
        "DIVERSE + BOUNDARY + INFORMATIVE"
    )
    print("=" * 80)

    print(
        f"Candidate pool : {len(examples)}"
    )

    print(
        f"Final samples  : {len(selected_examples)}"
    )

    print(
        f"Representative : {representative_count}"
    )

    print(
        f"Boundary       : {boundary_count}"
    )

    print(
        f"Informative    : {informative_count}"
    )

    print("=" * 80)

    selection_info = {
        "strategy":
            "diverse_boundary_informative",

        "candidate_pool_size":
            len(examples),

        "sample_size":
            len(selected_examples),
            

        "representative_count":
            representative_count,

        "boundary_count":
            boundary_count,

        "informative_count":
            informative_count,

        "seed":
            seed,

        "selected_pair_names": [
            getattr(
                example,
                "pair_name",
                f"example_{index}",
            )
            for index, example
            in enumerate(selected_examples)
        ],
    }

    return (
        selected_examples,
        selection_info,
    )
# ------------------------------------------------------------
# MAIN SELECTION FUNCTION
# ------------------------------------------------------------

def select_diverse_boundary_examples(
    examples,
    sample_size,
    n_clusters=5,
    representative_fraction=0.5,
    seed=42,
    cache_dir=".cache/dca_embeddings",
):
    """
    Select a training subset using:

        1. semantic diversity
        2. boundary coverage

    Example:

        sample_size = 50
        representative_fraction = 0.5

        => 25 representative
        => 25 boundary

    Returns:

        selected_examples
        selection_info
    """

    # Convert to list first
    examples = list(examples)

    # --------------------------------------------------------
    # FILTER EXAMPLES FOR CLIP EMBEDDING
    # --------------------------------------------------------
    # CLIP pair embeddings require BOTH images.
    # Examples with a missing original or reimage are excluded
    # only from this sampling process.
    #
    # IMPORTANT:
    # This does NOT modify the original Train / Val / Test split.

    embeddable_examples = []

    skipped_missing_original = 0
    skipped_missing_reimage = 0

    for example in examples:

        original = getattr(
            example,
            "image_original",
            None,
        )

        reimage = getattr(
            example,
            "image_reimage",
            None,
        )

        if original is None:
            skipped_missing_original += 1
            continue

        if reimage is None:
            skipped_missing_reimage += 1
            continue

        embeddable_examples.append(example)

    # --------------------------------------------------------
    # PRINT EMBEDDING POOL INFORMATION
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("DCA EMBEDDABLE TRAINING POOL")
    print("=" * 80)

    print(
        f"Original candidate pool : {len(examples)}"
    )

    print(
        f"Embeddable examples     : "
        f"{len(embeddable_examples)}"
    )

    print(
        f"Missing original        : "
        f"{skipped_missing_original}"
    )

    print(
        f"Missing reimage         : "
        f"{skipped_missing_reimage}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # VALIDATE SAMPLE SIZE
    # --------------------------------------------------------

    if sample_size <= 0:
        raise ValueError(
            "sample_size must be > 0"
        )

    if sample_size > len(embeddable_examples):
        raise ValueError(
            f"Requested {sample_size} examples "
            f"but only {len(embeddable_examples)} "
            f"examples have both images available."
        )

    # From this point onward, the sampler works only
    # with examples that can be embedded by CLIP.
    examples = embeddable_examples

    # --------------------------------------------------------
    # ADJUST NUMBER OF CLUSTERS
    # --------------------------------------------------------

    if n_clusters > len(examples):
        n_clusters = len(examples)

    # --------------------------------------------------------
    # GENERATE EMBEDDINGS
    # --------------------------------------------------------

    embeddings = generate_pair_embeddings(
        examples,
        cache_dir=cache_dir,
    )

    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("DCA SEMANTIC CLUSTERING")
    print("=" * 80)

    print(f"Examples  : {len(examples)}")
    print(f"Clusters  : {n_clusters}")
    print(f"Seed      : {seed}")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=10,
    )

    labels = kmeans.fit_predict(
        embeddings
    )

    centers = kmeans.cluster_centers_

    for cluster_id in range(n_clusters):

        count = int(
            np.sum(
                labels == cluster_id
            )
        )

        print(
            f"Cluster {cluster_id}: "
            f"{count} examples"
        )

    # --------------------------------------------------------
    # SPLIT SIZE
    # --------------------------------------------------------

    representative_count = int(
        round(
            sample_size
            * representative_fraction
        )
    )

    boundary_count = (
        sample_size
        - representative_count
    )

    # --------------------------------------------------------
    # REPRESENTATIVE SELECTION
    # --------------------------------------------------------

    representative_indices = (
        _select_representatives(
            embeddings=embeddings,
            labels=labels,
            centers=centers,
            count=representative_count,
            seed=seed,
        )
    )

    # --------------------------------------------------------
    # BOUNDARY SELECTION
    # --------------------------------------------------------

    boundary_indices = (
        _select_boundary_examples(
            embeddings=embeddings,
            labels=labels,
            centers=centers,
            count=boundary_count,
            excluded_indices=representative_indices,
        )
    )

    selected_indices = (
        representative_indices
        + boundary_indices
    )

    # Safety check.
    selected_indices = list(
        dict.fromkeys(selected_indices)
    )

    if len(selected_indices) < sample_size:

        remaining = [
            index
            for index in range(
                len(examples)
            )
            if index not in selected_indices
        ]

        selected_indices.extend(
            remaining[
                :sample_size
                - len(selected_indices)
            ]
        )

    selected_indices = selected_indices[
        :sample_size
    ]

    # --------------------------------------------------------
    # CREATE RESULT
    # --------------------------------------------------------

    selected_examples = [
        examples[index]
        for index in selected_indices
    ]

    representative_set = set(
        representative_indices
    )

    boundary_set = set(
        boundary_indices
    )

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("DIVERSE + BOUNDARY SELECTION")
    print("=" * 80)

    print(
        f"Candidate pool       : {len(examples)}"
    )

    print(
        f"Requested training   : {sample_size}"
    )

    print(
        f"Representative       : "
        f"{len(representative_indices)}"
    )

    print(
        f"Boundary             : "
        f"{len(boundary_indices)}"
    )

    print(
        f"Final selected       : "
        f"{len(selected_examples)}"
    )

    print("\nSELECTED EXAMPLES")
    print("-" * 80)

    for rank, index in enumerate(
        selected_indices,
        start=1,
    ):

        example = examples[index]

        pair_name = getattr(
            example,
            "pair_name",
            f"example_{index}",
        )

        selection_type = (
            "REPRESENTATIVE"
            if index in representative_set
            else "BOUNDARY"
            if index in boundary_set
            else "OTHER"
        )

        print(
            f"{rank:3d}. "
            f"pair={pair_name} | "
            f"type={selection_type} | "
            f"cluster={labels[index]}"
        )

    print("=" * 80)

    selection_info = {
        "strategy": "diverse_boundary",
        "candidate_pool_size": len(examples),
        "sample_size": sample_size,
        "n_clusters": n_clusters,
        "representative_count": len(
            representative_indices
        ),
        "boundary_count": len(
            boundary_indices
        ),
        "seed": seed,
        "selected_indices": selected_indices,
        "selected_pair_names": [
            getattr(
                examples[index],
                "pair_name",
                f"example_{index}",
            )
            for index in selected_indices
        ],
    }

    return (
        selected_examples,
        selection_info,
    )