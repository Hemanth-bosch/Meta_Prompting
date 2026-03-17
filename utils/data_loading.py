import dspy
from typing import List
import json
import os


def load_japanese_order_dataset(images_dir: str = None,
                                 gt_dir: str = None) -> List[dspy.Example]:
    """
    Load Japanese order dataset with paired images and JSON labels

    Args:
        images_dir: Directory containing PNG images (default: from env JAPANESE_ORDER_IMAGES_DIR or "./data/png")
        gt_dir: Directory containing ground truth JSON files (default: from env JAPANESE_ORDER_GT_DIR or "./data/json")

    Returns:
        List of dspy.Example objects
    """
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
            print(f"❌ load error {json_file}: {e}")

    print(f"✅ load success: {len(examples)} ")
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
