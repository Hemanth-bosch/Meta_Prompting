"""DCA dataset loading — build labelled ``dspy.Example``s from image pairs + ground-truth JSON."""
import dspy
from typing import List
import json
import logging
import os

logger = logging.getLogger("dca.data_loading")


def load_dca_dataset(
    pairs_dir: str = "data/dca_data/image_pairs",
    gt_dir: str = "data/dca_data/image_pairs_llm_output_exp1"
) -> List[dspy.Example]:
    """Load DCA image pair dataset for discard_image classification.

    Each pair consists of:
      pairs_dir/<pair_name>/original.png   -- original document image
      pairs_dir/<pair_name>/reimage.png    -- re-imaged / updated version
      gt_dir/<pair_name>.json              -- ground truth annotation

    Ground truth JSON must contain:
      {"human_label_discard_image": bool, "human_label_changes": str}

    One-sided pairs (only original or only reimage present) are supported;
    the missing side is set to None so the model sees an explicit absent slot.

    Returns list of dspy.Example with:
      image_original             dspy.Image or None (input)
      image_reimage              dspy.Image or None (input)
      expected_discard_image     str "true" or "false" (label)
      expected_changes           str human change summary (label)
      expected_human_label_category  str category code (metadata)
      doubt                      bool (metadata)
      pair_name                  str (metadata)
    """
    examples = []

    pairs_dir = os.getenv("DCA_IMAGE_PAIRS_DIR", pairs_dir)
    gt_dir = os.getenv("DCA_GT_DIR", gt_dir)

    if not os.path.exists(pairs_dir):
        print(f"Directory {pairs_dir} not found. Returning empty dataset.")
        return examples
    if not os.path.exists(gt_dir):
        print(f"Directory {gt_dir} not found. Returning empty dataset.")
        return examples

    pair_dirs = sorted([
        d for d in os.listdir(pairs_dir)
        if os.path.isdir(os.path.join(pairs_dir, d))
    ])

    print(f"Found {len(pair_dirs)} pair directories in {pairs_dir}")

    skipped_no_gt = 0
    skipped_no_human = 0
    skipped_missing_img = 0
    fallback_missing_original = 0
    fallback_missing_reimage = 0

    for pair_name in pair_dirs:
        pair_path = os.path.join(pairs_dir, pair_name)
        original_path = os.path.join(pair_path, "original.png")
        reimage_path = os.path.join(pair_path, "reimage.png")
        gt_path = os.path.join(gt_dir, f"{pair_name}.json")

        original_exists = os.path.exists(original_path)
        reimage_exists = os.path.exists(reimage_path)

        if not original_exists and not reimage_exists:
            skipped_missing_img += 1
            continue

        # Track one-sided pairs
        if not original_exists and reimage_exists:
            fallback_missing_original += 1
        elif original_exists and not reimage_exists:
            fallback_missing_reimage += 1

        if not os.path.exists(gt_path):
            skipped_no_gt += 1
            continue

        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt = json.load(f)
        except Exception as e:
            print(f"  Warning: Failed to parse {gt_path}: {e}")
            continue

        # Only the current key format is supported
        if "human_label_discard_image" not in gt:
            skipped_no_human += 1
            continue

        human_classification = gt["human_label_discard_image"]
        expected_changes = gt.get("human_label_changes", "")
        expected_category = gt.get("human_label_category", "Unknown")

        # Normalize category to canonical code
        if not isinstance(expected_category, str):
            expected_category = str(expected_category)
        expected_category = expected_category.strip() or "Unknown"
        category_lower = expected_category.lower()
        if category_lower.startswith("a1"):
            expected_category = "A1"
        elif category_lower.startswith("a2"):
            expected_category = "A2"
        elif category_lower.startswith("b1"):
            expected_category = "B1"
        elif category_lower.startswith("b2"):
            expected_category = "B2"
        elif category_lower.startswith("b3"):
            expected_category = "B3"
        elif category_lower.startswith("b4"):
            expected_category = "B4"
        elif category_lower.startswith("b5a"):
            expected_category = "B5a"
        elif category_lower.startswith("b5b"):
            expected_category = "B5b"

        expected_discard = "true" if human_classification else "false"

        # Load images; None for missing side (one-sided pair)
        try:
            image_original = dspy.Image.from_file(original_path) if original_exists else None
            image_reimage = dspy.Image.from_file(reimage_path) if reimage_exists else None
        except Exception as e:
            print(f"  Warning: Failed to load images for {pair_name}: {e}")
            continue

        if not original_exists:
            logger.info("Fallback pair %s: original missing, set to None", pair_name)
        elif not reimage_exists:
            logger.info("Fallback pair %s: reimage missing, set to None", pair_name)

        example = dspy.Example(
            image_original=image_original,
            image_reimage=image_reimage,
            expected_discard_image=expected_discard,
            expected_changes=expected_changes,
            expected_human_label_category=expected_category,
            doubt=gt.get("doubt", False),
            pair_name=pair_name,
        ).with_inputs("image_original", "image_reimage")

        examples.append(example)

    print(f"Loaded {len(examples)} valid DCA image pairs")
    if skipped_no_gt:
        print(f"  Skipped (no ground truth JSON): {skipped_no_gt}")
    if skipped_no_human:
        print(f"  Skipped (no human_label_discard_image key): {skipped_no_human}")
    if skipped_missing_img:
        print(f"  Skipped (both images missing): {skipped_missing_img}")
    if fallback_missing_original:
        print(f"  Fallback (missing original): {fallback_missing_original}")
    if fallback_missing_reimage:
        print(f"  Fallback (missing reimage): {fallback_missing_reimage}")

    if examples:
        discard_true = sum(1 for e in examples if e.expected_discard_image == "true")
        discard_false = len(examples) - discard_true
        print(f"  discard_image=true (discard): {discard_true}")
        print(f"  discard_image=false (keep):   {discard_false}")

    return examples
