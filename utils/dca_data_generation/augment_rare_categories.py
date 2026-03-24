"""
Generate synthetic image pairs for underrepresented DCA categories (B1, B2, B4).

Augments the dataset so every category has ≥20 samples, enabling meaningful
stratified splitting.  All generated pairs are tagged with ``is_synthetic``,
``source_pairs``, and ``data_augmentation_explain_in_detail`` for full
traceability.

Categories generated
--------------------
- B1 (corrupted / unreadable)  — destructive transforms on one image
- B2 (full-image cross-mark)   — X overlay on reimage
- B4 (one info + one non-info) — recombination from A1/B3 + B5a/B5b

Example
-------
# Dry-run first
python utils/dca_data_generation/augment_rare_categories.py --dry-run --verbose

# Full run
python utils/dca_data_generation/augment_rare_categories.py --verbose

# Then run experiments on new pairs
python evaluation/dca_baseline_runs/run_human_prompts/run_all_experiments.py  # with --start/--end for new range
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent


def _find_repo_root(start_dir: Path) -> Path:
    """Walk up until we find pyproject.toml."""
    for candidate in [start_dir, *start_dir.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    # Fallback
    return start_dir.parents[2]


REPO_ROOT = _find_repo_root(SCRIPT_DIR)
IMAGE_PAIRS_DIR = REPO_ROOT / "data" / "dca_data" / "image_pairs"
JSON_DIR = REPO_ROOT / "data" / "dca_data" / "image_pairs_llm_output_exp1"

# ---------------------------------------------------------------------------
# Category → pair-number mappings (from analyze_labels.py output, 2026-03-23)
# ---------------------------------------------------------------------------

B5A_PAIRS: List[int] = [
    2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    27, 28, 30, 31, 34, 36, 37, 38, 48, 70, 73, 74, 75, 76, 77, 79, 89, 91,
    94, 96, 98, 99, 100, 103, 104, 116, 119, 122, 124, 125, 150, 152, 155,
    156, 158, 159, 160, 161, 162, 164, 165, 166, 171, 172, 173, 176, 177,
    178, 180, 181, 182, 197, 209, 216, 221, 238, 239, 241, 244, 245, 248,
    264, 266, 267, 268, 269, 271, 275, 276, 277, 278, 279, 287, 288, 291,
    297, 316, 317, 320, 323, 328, 334, 335, 363, 367,
]

B5B_PAIRS: List[int] = [
    7, 10, 29, 45, 47, 53, 55, 81, 83, 101, 102, 106, 112, 121, 149, 157,
    163, 203, 240, 243, 246, 247, 249, 251, 253, 263, 280, 281, 282, 283,
    284, 286, 289, 290, 292, 294, 295, 299, 300, 302, 303, 304, 318, 324,
]

A1_PAIRS: List[int] = [
    8, 61, 63, 65, 66, 68, 69, 84, 87, 88, 92, 105, 107, 108, 110, 111,
    114, 115, 117, 118, 120, 123, 140, 144, 145, 151, 183, 184, 185, 187,
    188, 190, 191, 192, 193, 194, 195, 196, 198, 201, 204, 205, 206, 207,
    208, 211, 212, 213, 214, 219, 220, 222, 223, 224, 226, 227, 228, 229,
    231, 232, 233, 332, 361, 362, 364, 366, 368, 369,
]

B3_PAIRS: List[int] = [
    32, 35, 40, 41, 46, 49, 51, 56, 57, 58, 59, 60, 64, 71, 72, 78, 80,
    85, 86, 90, 93, 95, 97, 200, 215, 217, 315, 343, 344, 345, 346, 347,
    348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 365,
    370, 371, 372, 373, 374, 375, 376,
]

# Two-image informational source pool (both original + reimage exist, both informational)
TWO_IMAGE_INFO_POOL: List[int] = sorted(B5A_PAIRS + B5B_PAIRS)  # 148 pairs

# Non-informational image source pool (A1 = one image, B3 = two non-info images)
NON_INFO_POOL: List[int] = sorted(A1_PAIRS + B3_PAIRS)  # 121 pairs

# Current counts (for summary reporting)
CURRENT_COUNTS: Dict[str, int] = {
    "A1": 68, "A2": 100, "B1": 0, "B2": 2,
    "B3": 53, "B4": 5, "B5a": 104, "B5b": 44,
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def find_next_pair_index() -> int:
    """Scan existing pair folders and JSON files, return the next available index."""
    existing_dirs = [
        int(p.name.split("_")[1])
        for p in IMAGE_PAIRS_DIR.glob("pair_*")
        if p.is_dir() and p.name.split("_")[1].isdigit()
    ]
    existing_jsons = [
        int(p.stem.split("_")[1])
        for p in JSON_DIR.glob("pair_*.json")
        if p.stem.split("_")[1].isdigit()
    ]
    all_existing = existing_dirs + existing_jsons
    return max(all_existing) + 1 if all_existing else 1


def find_noninfo_image(pair_idx: int, rng: random.Random) -> Tuple[Path, str]:
    """Find an available image in a (possibly one-sided) pair folder.

    For A1 pairs only one image exists.  For B3 pairs both exist — randomly
    pick one.

    Returns (path, role_name) where role_name is ``"original"`` or ``"reimage"``.
    """
    pair_dir = IMAGE_PAIRS_DIR / f"pair_{pair_idx:03d}"
    candidates: List[Tuple[Path, str]] = []
    for name, role in [("original.png", "original"), ("reimage.png", "reimage")]:
        path = pair_dir / name
        if path.exists():
            candidates.append((path, role))
    if not candidates:
        raise FileNotFoundError(f"No image found in {pair_dir}")
    return rng.choice(candidates)


def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("augment_rare_categories")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Corruption functions (B1)
# ---------------------------------------------------------------------------

def corrupt_blur(img: Image.Image) -> Image.Image:
    """Extreme Gaussian blur (σ ≈ 25) — makes all text unreadable."""
    return img.filter(ImageFilter.GaussianBlur(radius=7.5))


def corrupt_pixelate(img: Image.Image) -> Image.Image:
    """Downscale to 100×100 then upscale back — severe pixelation."""
    original_size = img.size
    tiny = img.resize((100, 100), Image.NEAREST)
    return tiny.resize(original_size, Image.NEAREST)


def corrupt_noise(img: Image.Image, coverage: float = 0.75, seed: Optional[int] = None) -> Image.Image:
    """Salt-and-pepper noise at *coverage* fraction of all pixels."""
    if seed is not None:
        np.random.seed(seed)
    arr = np.array(img, dtype=np.uint8).copy()
    h, w = arr.shape[:2]
    mask = np.random.random((h, w))
    half = coverage / 2.0
    # Salt (white)
    arr[mask < half] = 255
    # Pepper (black)
    arr[(mask >= half) & (mask < coverage)] = 0
    return Image.fromarray(arr)


def corrupt_tiny(img: Image.Image) -> Image.Image:
    """Resize to genuinely tiny dimensions (~30–40px) — too small to read."""
    return img.resize((40, 40), Image.LANCZOS)


CORRUPTION_METHODS: List[Tuple[str, Any]] = [
    ("extreme_gaussian_blur_sigma25", corrupt_blur),
    ("pixelate_100x100_nearest", corrupt_pixelate),
    ("too_tiny_40x40", corrupt_tiny),
]


# ---------------------------------------------------------------------------
# Cross-mark drawing (B2)
# ---------------------------------------------------------------------------

def draw_cross(img: Image.Image, color: Tuple[int, ...], thickness: int, alpha: int = 255) -> Image.Image:
    """Draw two diagonal lines corner-to-corner (full-image X).

    *alpha* controls line transparency: 255 = fully opaque, lower = more faded.
    Compositing is done via an RGBA overlay so the underlying image shows through.
    """
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = base.size
    line_color = (*color, alpha)
    draw.line([(0, 0), (w - 1, h - 1)], fill=line_color, width=thickness)
    draw.line([(w - 1, 0), (0, h - 1)], fill=line_color, width=thickness)
    composited = Image.alpha_composite(base, overlay)
    return composited.convert("RGB")


# CROSS_STYLES: (color_rgb, thickness, alpha)
# alpha range 40-120 gives subtle, faded lines that are clearly visible but thin-looking
CROSS_STYLES: List[Tuple[Tuple[int, int, int], int, int]] = [
    ((255, 0, 0), 1, 80),    # faded red
    ((0, 0, 0), 1, 90),      # faded black
    ((0, 0, 255), 1, 80),    # faded blue
    ((200, 0, 0), 1, 70),    # faded muted red
    ((0, 0, 0), 1, 90),      # faded black
    ((0, 0, 200), 1, 75),    # faded dark blue
    ((255, 0, 0), 1, 80),    # very faded red
    ((40, 40, 40), 1, 80),   # faded dark gray
    ((0, 0, 180), 1, 75),    # faded navy
    ((0, 0, 0), 1, 100),     # semi-transparent black
    ((255, 0, 0), 1, 100),   # semi-transparent red
    ((0, 0, 255), 1, 100),   # semi-transparent blue
    ((80, 0, 0), 1, 90),     # faded maroon
    ((0, 0, 140), 1, 70),    # faded dark navy
    ((60, 60, 60), 1, 70),   # faded dark gray
    ((200, 0, 0), 1, 80),    # faded muted red
    ((0, 0, 220), 1, 75),    # faded blue
    ((0, 0, 0), 1, 85),      # very faded black
]


# ---------------------------------------------------------------------------
# Pair writer
# ---------------------------------------------------------------------------

def write_pair(
    pair_idx: int,
    original_img: Image.Image,
    reimage_img: Optional[Image.Image],
    json_payload: Dict[str, Any],
    source_paths_text: str,
    logger: logging.Logger,
    dry_run: bool = False,
) -> None:
    """Write a synthetic pair (images + JSON + source_paths.txt) to disk."""
    pair_dir = IMAGE_PAIRS_DIR / f"pair_{pair_idx:03d}"
    json_path = JSON_DIR / f"pair_{pair_idx:03d}.json"

    if dry_run:
        logger.info(
            "[DRY RUN] pair_%03d  category=%s  (would create %s + %s)",
            pair_idx,
            json_payload.get("human_label_category", "?"),
            pair_dir,
            json_path,
        )
        return

    pair_dir.mkdir(parents=True, exist_ok=True)

    original_img.save(pair_dir / "original.png")
    if reimage_img is not None:
        reimage_img.save(pair_dir / "reimage.png")

    (pair_dir / "source_paths.txt").write_text(source_paths_text, encoding="utf-8")
    json_path.write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Created pair_%03d  [%s]",
        pair_idx,
        json_payload.get("human_label_category", "?"),
    )


# ---------------------------------------------------------------------------
# B1 generator — corrupted / unreadable
# ---------------------------------------------------------------------------

def generate_b1_pairs(
    source_pairs: List[int],
    start_idx: int,
    count: int,
    rng: random.Random,
    logger: logging.Logger,
    dry_run: bool,
) -> int:
    """Generate B1 (corrupted/unreadable) pairs.

    Takes an informational image from *source_pairs*, applies a destructive
    transform to one of the two images (randomly chosen), and keeps the other
    image intact.
    """
    logger.info("=== Generating %d B1 (corrupted/unreadable) pairs ===", count)

    created = 0
    for i, src_pair in enumerate(source_pairs[:count]):
        pair_idx = start_idx + i
        method_name, corrupt_fn = CORRUPTION_METHODS[i % len(CORRUPTION_METHODS)]

        # Randomly decide which role gets corrupted
        corrupt_role = rng.choice(["original", "reimage"])
        good_role = "reimage" if corrupt_role == "original" else "original"

        src_dir = IMAGE_PAIRS_DIR / f"pair_{src_pair:03d}"
        original_img = Image.open(src_dir / "original.png").convert("RGB")
        reimage_img = Image.open(src_dir / "reimage.png").convert("RGB")

        if corrupt_role == "original":
            new_original = corrupt_fn(original_img)
            new_reimage = reimage_img
        else:
            new_original = original_img
            new_reimage = corrupt_fn(reimage_img)

        json_payload = {
            "human_label_changes": "Could not infer",
            "human_label_discard_image": False,
            "human_label_category": "B1",
            "is_synthetic": True,
            "source_pairs": [f"pair_{src_pair:03d}"],
            "data_augmentation_explain_in_detail": (
                f"B1 synthetic pair. {corrupt_role.capitalize()} image from "
                f"pair_{src_pair:03d} was corrupted using method "
                f"'{method_name}'. {good_role.capitalize()} image kept intact "
                f"from pair_{src_pair:03d}."
            ),
        }

        source_paths_text = (
            f"synthetic:    true\n"
            f"method:       B1_corruption ({method_name})\n"
            f"source_pair:  pair_{src_pair:03d}\n"
            f"corrupted:    {corrupt_role}\n"
            f"original:     pair_{src_pair:03d}/{corrupt_role}.png"
            f" (corrupted with {method_name})\n"
            f"reimage:      pair_{src_pair:03d}/{good_role}.png (intact)\n"
        ) if corrupt_role == "original" else (
            f"synthetic:    true\n"
            f"method:       B1_corruption ({method_name})\n"
            f"source_pair:  pair_{src_pair:03d}\n"
            f"corrupted:    {corrupt_role}\n"
            f"original:     pair_{src_pair:03d}/original.png (intact)\n"
            f"reimage:      pair_{src_pair:03d}/reimage.png"
            f" (corrupted with {method_name})\n"
        )

        write_pair(pair_idx, new_original, new_reimage, json_payload, source_paths_text, logger, dry_run)
        created += 1

    return created


# ---------------------------------------------------------------------------
# B2 generator — full-image cross-mark / strikethrough
# ---------------------------------------------------------------------------

def generate_b2_pairs(
    source_pairs: List[int],
    start_idx: int,
    count: int,
    rng: random.Random,
    logger: logging.Logger,
    dry_run: bool,
) -> int:
    """Generate B2 (crossed-out) pairs.

    Copies the original image untouched and overlays the reimage with a
    full-image diagonal X cross-mark.
    """
    logger.info("=== Generating %d B2 (crossed-out) pairs ===", count)

    created = 0
    for i, src_pair in enumerate(source_pairs[:count]):
        pair_idx = start_idx + i
        color, thickness, alpha = CROSS_STYLES[i % len(CROSS_STYLES)]

        src_dir = IMAGE_PAIRS_DIR / f"pair_{src_pair:03d}"
        original_img = Image.open(src_dir / "original.png").convert("RGB")
        reimage_img = Image.open(src_dir / "reimage.png").convert("RGB")

        crossed_reimage = draw_cross(reimage_img, color, thickness, alpha)

        json_payload = {
            "human_label_changes": "Image 2 has been crossed out",
            "human_label_discard_image": False,
            "human_label_category": "B2",
            "is_synthetic": True,
            "source_pairs": [f"pair_{src_pair:03d}"],
            "data_augmentation_explain_in_detail": (
                f"B2 synthetic pair. Both images sourced from pair_{src_pair:03d}. "
                f"Reimage overlaid with full-image X cross-mark "
                f"(color=rgb{color}, thickness={thickness}px, alpha={alpha}/255). "
                f"Original image kept intact."
            ),
        }

        source_paths_text = (
            f"synthetic:    true\n"
            f"method:       B2_cross_mark (color=rgb{color}, thickness={thickness}px, alpha={alpha}/255)\n"
            f"source_pair:  pair_{src_pair:03d}\n"
            f"original:     pair_{src_pair:03d}/original.png (untouched)\n"
            f"reimage:      pair_{src_pair:03d}/reimage.png (X cross-mark overlaid)\n"
        )

        write_pair(pair_idx, original_img, crossed_reimage, json_payload, source_paths_text, logger, dry_run)
        created += 1

    return created


# ---------------------------------------------------------------------------
# B4 generator — one informational + one non-informational
# ---------------------------------------------------------------------------

def generate_b4_pairs(
    info_source_pairs: List[int],
    noninfo_pool: List[int],
    start_idx: int,
    count: int,
    rng: random.Random,
    logger: logging.Logger,
    dry_run: bool,
) -> int:
    """Generate B4 (one informational + one non-informational) pairs.

    Combines a non-informational image from A1/B3 pairs with an informational
    image from B5a/B5b pairs.  Layout (which goes to original vs reimage) is
    randomized.
    """
    logger.info("=== Generating %d B4 (one info + one non-info) pairs ===", count)

    # Take count items from each pool
    selected_info = info_source_pairs[:count]
    selected_noninfo = noninfo_pool[:count]
    actual = min(count, len(selected_info), len(selected_noninfo))

    created = 0
    for i in range(actual):
        pair_idx = start_idx + i
        info_pair = selected_info[i]
        noninfo_pair = selected_noninfo[i]

        # Load informational image — randomly pick original or reimage from source pair
        info_dir = IMAGE_PAIRS_DIR / f"pair_{info_pair:03d}"
        info_role = rng.choice(["original", "reimage"])
        info_img = Image.open(info_dir / f"{info_role}.png").convert("RGB")

        # Load non-informational image (handles one-sided A1 pairs)
        noninfo_path, noninfo_role = find_noninfo_image(noninfo_pair, rng)
        noninfo_img = Image.open(noninfo_path).convert("RGB")

        # Randomly decide layout
        info_goes_to_original = rng.choice([True, False])

        if info_goes_to_original:
            new_original = info_img
            new_reimage = noninfo_img
            layout_desc = (
                f"original=informational (from pair_{info_pair:03d}/{info_role}.png), "
                f"reimage=non-informational (from pair_{noninfo_pair:03d}/{noninfo_role}.png)"
            )
        else:
            new_original = noninfo_img
            new_reimage = info_img
            layout_desc = (
                f"original=non-informational (from pair_{noninfo_pair:03d}/{noninfo_role}.png), "
                f"reimage=informational (from pair_{info_pair:03d}/{info_role}.png)"
            )

        json_payload = {
            "human_label_changes": (
                "Only one image is informational; the other is "
                "non-informational (watermark/logo/blank)"
            ),
            "human_label_discard_image": False,
            "human_label_category": "B4",
            "is_synthetic": True,
            "source_pairs": [f"pair_{noninfo_pair:03d}", f"pair_{info_pair:03d}"],
            "data_augmentation_explain_in_detail": (
                f"B4 synthetic pair. Non-informational image from "
                f"pair_{noninfo_pair:03d} ({noninfo_role}.png), informational "
                f"image from pair_{info_pair:03d} ({info_role}.png). "
                f"Layout: {layout_desc}."
            ),
        }

        source_paths_text = (
            f"synthetic:    true\n"
            f"method:       B4_recombination\n"
            f"source_pairs: pair_{noninfo_pair:03d} (non-informational), "
            f"pair_{info_pair:03d} (informational)\n"
            f"original:     {layout_desc.split(', ')[0].split('=', 1)[1]}\n"
            f"reimage:      {layout_desc.split(', ')[1].split('=', 1)[1]}\n"
        )

        write_pair(pair_idx, new_original, new_reimage, json_payload, source_paths_text, logger, dry_run)
        created += 1

    return created


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic image pairs for underrepresented DCA "
            "categories (B1, B2, B4)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--b1-count", type=int, default=20, help="B1 pairs to generate (default: 20)")
    parser.add_argument("--b2-count", type=int, default=18, help="B2 pairs to generate (default: 18)")
    parser.add_argument("--b4-count", type=int, default=15, help="B4 pairs to generate (default: 15)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = setup_logger(args.verbose)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Verify directories exist
    # ------------------------------------------------------------------
    if not IMAGE_PAIRS_DIR.exists():
        raise FileNotFoundError(f"Image pairs dir not found: {IMAGE_PAIRS_DIR}")
    if not JSON_DIR.exists():
        raise FileNotFoundError(f"JSON dir not found: {JSON_DIR}")

    # ------------------------------------------------------------------
    # Determine starting pair index
    # ------------------------------------------------------------------
    start_idx = find_next_pair_index()
    total_to_generate = args.b1_count + args.b2_count + args.b4_count

    logger.info("Repo root:          %s", REPO_ROOT)
    logger.info("Image pairs dir:    %s", IMAGE_PAIRS_DIR)
    logger.info("JSON dir:           %s", JSON_DIR)
    logger.info("Starting pair idx:  %d", start_idx)
    logger.info(
        "Plan: %d B1 + %d B2 + %d B4 = %d synthetic pairs",
        args.b1_count, args.b2_count, args.b4_count, total_to_generate,
    )

    # ------------------------------------------------------------------
    # Allocate source pairs WITHOUT overlap across generators
    # ------------------------------------------------------------------
    info_pool = TWO_IMAGE_INFO_POOL.copy()  # 148 pairs
    rng.shuffle(info_pool)

    needed_from_info = args.b1_count + args.b2_count + args.b4_count
    if needed_from_info > len(info_pool):
        raise ValueError(
            f"Need {needed_from_info} info-pool pairs but only {len(info_pool)} available"
        )

    b1_source = info_pool[: args.b1_count]
    b2_source = info_pool[args.b1_count : args.b1_count + args.b2_count]
    b4_info_source = info_pool[
        args.b1_count + args.b2_count :
        args.b1_count + args.b2_count + args.b4_count
    ]
    remaining_info = len(info_pool) - needed_from_info

    logger.info(
        "Info pool (B5a+B5b): %d total → B1=%d, B2=%d, B4_info=%d, unused=%d",
        len(info_pool), len(b1_source), len(b2_source),
        len(b4_info_source), remaining_info,
    )

    # Non-info pool for B4
    noninfo_pool = NON_INFO_POOL.copy()  # 121 pairs (A1 + B3)
    rng.shuffle(noninfo_pool)

    if args.b4_count > len(noninfo_pool):
        raise ValueError(
            f"Need {args.b4_count} non-info pairs but only {len(noninfo_pool)} available"
        )
    logger.info(
        "Non-info pool (A1+B3): %d total → B4_noninfo=%d, unused=%d",
        len(noninfo_pool), args.b4_count, len(noninfo_pool) - args.b4_count,
    )

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    idx = start_idx

    b1_created = generate_b1_pairs(b1_source, idx, args.b1_count, rng, logger, args.dry_run)
    idx += b1_created

    b2_created = generate_b2_pairs(b2_source, idx, args.b2_count, rng, logger, args.dry_run)
    idx += b2_created

    b4_created = generate_b4_pairs(
        b4_info_source, noninfo_pool, idx, args.b4_count, rng, logger, args.dry_run,
    )
    idx += b4_created

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = b1_created + b2_created + b4_created

    logger.info("")
    logger.info("=" * 65)
    logger.info("GENERATION SUMMARY")
    logger.info("=" * 65)
    if b1_created:
        logger.info(
            "  B1 (corrupted):    %2d pairs  (pair_%03d – pair_%03d)",
            b1_created, start_idx, start_idx + b1_created - 1,
        )
    if b2_created:
        logger.info(
            "  B2 (crossed-out):  %2d pairs  (pair_%03d – pair_%03d)",
            b2_created,
            start_idx + b1_created,
            start_idx + b1_created + b2_created - 1,
        )
    if b4_created:
        logger.info(
            "  B4 (info+noninfo): %2d pairs  (pair_%03d – pair_%03d)",
            b4_created,
            start_idx + b1_created + b2_created,
            idx - 1,
        )
    logger.info("  Total:             %2d pairs  (pair_%03d – pair_%03d)", total, start_idx, idx - 1)

    logger.info("")
    logger.info("Expected new category distribution:")
    new_dist = {
        "A1": CURRENT_COUNTS["A1"],
        "A2": CURRENT_COUNTS["A2"],
        "B1": CURRENT_COUNTS["B1"] + b1_created,
        "B2": CURRENT_COUNTS["B2"] + b2_created,
        "B3": CURRENT_COUNTS["B3"],
        "B4": CURRENT_COUNTS["B4"] + b4_created,
        "B5a": CURRENT_COUNTS["B5a"],
        "B5b": CURRENT_COUNTS["B5b"],
    }
    grand = sum(new_dist.values())
    for cat in ["A1", "A2", "B1", "B2", "B3", "B4", "B5a", "B5b"]:
        marker = " ← augmented" if cat in ("B1", "B2", "B4") else ""
        logger.info(
            "  %3s: %3d  (%5.1f%%)%s",
            cat, new_dist[cat], 100.0 * new_dist[cat] / grand, marker,
        )
    logger.info("  Total: %d", grand)

    if args.dry_run:
        logger.info("")
        logger.info("[DRY RUN] No files were written.")
    else:
        logger.info("")
        logger.info("Done. To run experiments on the new pairs:")
        logger.info("  --start %d --end %d", start_idx, idx - 1)
        logger.info("")
        logger.info("Then re-analyze labels:")
        logger.info(
            "  python evaluation/dca_baseline_runs/run_human_prompts/analyze_labels.py"
        )


if __name__ == "__main__":
    main()
