"""DCA dataset loading — build labelled ``dspy.Example``s from image pairs + ground-truth JSON."""
import dspy
from typing import List
import json
import logging
import os

logger = logging.getLogger("dca.data_loading")


def load_japanese_order_dataset(images_dir: str = None,
                                 gt_dir: str = None) -> List[dspy.Example]:
    """Load Japanese order dataset with paired images and JSON labels."""
    if images_dir is None:
        images_dir = os.getenv("JAPANESE_ORDER_IMAGES_DIR", "./data/png")
    if gt_dir is None:
        gt_dir = os.getenv("JAPANESE_ORDER_GT_DIR", "./data/json")

    json_dir = gt_dir

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    examples = []

    for json_file in sorted(json_files):
        base_name = json_file.replace('.json', '')

        image_path = os.path.join(images_dir, f"{base_name}.png")
        json_path = os.path.join(json_dir, json_file)

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            doc = data['documents'][0]

            po_numbers = []
            product_info = []
            if 'orderDetail' in doc:
                for order in doc['orderDetail']:
                    if 'PO' in order:
                        po_numbers.append(order['PO'])
                    if 'productDetail' in order:
                        for product in order['productDetail']:
                            product_info.append({
                                'productNumber': product.get('productNumber', ''),
                                'QTY': product.get('QTY', '')
                            })

            dspy_image = dspy.Image.from_file(image_path)

            example = dspy.Example(
                image=dspy_image,
                image_filename=os.path.basename(image_path),
                expected_sold_name1=doc.get("soldName1", ""),
                expected_sold_name2=doc.get("soldName2", ""),
                expected_sold_postal=doc.get("soldPostal", ""),
                expected_sold_address=doc.get("soldAddress", ""),
                expected_sold_tel=doc.get("soldTex", ""),
                expected_sold_fax=doc.get("soldFax", ""),
                expected_order_date=doc.get("CRD", ""),
                expected_ship_postal=doc.get("shipPostal", ""),
                expected_ship_tel=doc.get("shipTEL", ""),
                expected_ship_address=doc.get("shipAddress", ""),
                expected_company_name=doc.get("companyName", ""),
                expected_comments=doc.get("comments", ""),
                expected_po_numbers=po_numbers,
                expected_product_info=product_info
            ).with_inputs("image")

            examples.append(example)

        except Exception as e:
            print(f"  load error {json_file}: {e}")

    print(f"Loaded {len(examples)} Japanese order examples")
    return examples


def load_yanase_order_dataset(images_dir: str = None,
                               ground_truth_dir: str = None) -> List[dspy.Example]:
    """
    Load YANASE order dataset with paired images and ground truth labels

    Args:
        images_dir: Directory containing PNG images (default: from env YANASE_IMAGES_DIR or "data/YANASE_images")
        ground_truth_dir: Directory containing ground truth JSON files (default: from env YANASE_GT_DIR or "data/YANASE_ground_truth")

    Returns:
        List of dspy.Example objects with image inputs and expected outputs
    """
    if images_dir is None:
        images_dir = os.getenv("YANASE_IMAGES_DIR", "data/YANASE_images")
    if ground_truth_dir is None:
        ground_truth_dir = os.getenv("YANASE_GT_DIR", "data/YANASE_ground_truth")
    from pathlib import Path

    gt_path = Path(ground_truth_dir)
    images_path = Path(images_dir)

    if not gt_path.exists():
        raise ValueError(f"Ground truth directory not found: {ground_truth_dir}")
    if not images_path.exists():
        raise ValueError(f"Images directory not found: {images_dir}")

    examples = []

    for gt_file in sorted(gt_path.glob("*_ground_truth.json")):
        base_name = gt_file.stem.replace("_ground_truth", "")
        image_file = images_path / f"{base_name}.png"

        if not image_file.exists():
            print(f"⚠️  Image not found for {gt_file.name}, skipping...")
            continue

        try:
            with open(gt_file, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)

            dspy_image = dspy.Image.from_file(str(image_file))

            example = dspy.Example(
                image=dspy_image,
                image_filename=base_name,
                expected_soldName1=gt_data.get("soldName1", ""),
                expected_soldName2=gt_data.get("soldName2", ""),
                expected_soldTel=gt_data.get("soldTel", ""),
                expected_soldFax=gt_data.get("soldFax", ""),
                expected_receiverCompany=gt_data.get("receiverCompany", ""),
                expected_shipPostal=gt_data.get("shipPostal", ""),
                expected_shipAddress=gt_data.get("shipAddress", ""),
                expected_shipTel=gt_data.get("shipTel", ""),
                expected_shipFax=gt_data.get("shipFax", ""),
                expected_productOrders=gt_data.get("productOrders", [])
            ).with_inputs("image")

            examples.append(example)

        except Exception as e:
            print(f"❌ Error loading {gt_file.name}: {e}")
            continue

    print(f"✅ Successfully loaded {len(examples)} YANASE examples")
    return examples


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
