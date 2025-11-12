import dspy
from typing import List, Dict, Any
import json
import os
import random

from pytesseract import Output
import pytesseract
from PIL import Image

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
    # Use environment variables or defaults if paths not provided
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
            # load data
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            doc = data['documents'][0]
            
            
            # Analyze order details
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
            
            # load image
            dspy_image = dspy.Image.from_file(image_path)
            
            # Create Example
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


def split_order_dataset(dataset, train_ratio=0.6, val_ratio=0.3, test_ratio=0.1, random_seed=42):
    """Divided order dataset"""
    random.seed(random_seed)
    shuffled_dataset = dataset.copy()
    random.shuffle(shuffled_dataset)
    
    total = len(shuffled_dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_set = shuffled_dataset[:train_size]
    val_set = shuffled_dataset[train_size:train_size + val_size]
    test_set = shuffled_dataset[train_size + val_size:]
    
    print(f" (total: {total}):")
    print(f"  train set: {len(train_set)} samples ({len(train_set)/total:.1%})")
    print(f"  val set: {len(val_set)} samples ({len(val_set)/total:.1%})")
    print(f"  test set: {len(test_set)} samples ({len(test_set)/total:.1%})")
    
    return train_set, val_set, test_set



####################################################
### Yanase order dataset loading and train/val/test splitting functions
####################################################

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
    # Use environment variables or defaults if paths not provided
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

    # Iterate through all ground truth files
    for gt_file in sorted(gt_path.glob("*_ground_truth.json")):
        base_name = gt_file.stem.replace("_ground_truth", "")
        image_file = images_path / f"{base_name}.png"

        if not image_file.exists():
            print(f"⚠️  Image not found for {gt_file.name}, skipping...")
            continue

        try:
            # Load ground truth JSON
            with open(gt_file, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)

            # Load image
            dspy_image = dspy.Image.from_file(str(image_file))

            # Create dspy.Example with inputs and expected outputs
            example = dspy.Example(
                image=dspy_image,
                image_filename=base_name,
                # Seller information
                expected_soldName1=gt_data.get("soldName1", ""),
                expected_soldName2=gt_data.get("soldName2", ""),
                expected_soldTel=gt_data.get("soldTel", ""),
                expected_soldFax=gt_data.get("soldFax", ""),
                # Receiver/shipping information
                expected_receiverCompany=gt_data.get("receiverCompany", ""),
                expected_shipPostal=gt_data.get("shipPostal", ""),
                expected_shipAddress=gt_data.get("shipAddress", ""),
                expected_shipTel=gt_data.get("shipTel", ""),
                expected_shipFax=gt_data.get("shipFax", ""),
                # Product orders (list of dicts)
                expected_productOrders=gt_data.get("productOrders", [])
            ).with_inputs("image")

            examples.append(example)

        except Exception as e:
            print(f"❌ Error loading {gt_file.name}: {e}")
            continue

    print(f"✅ Successfully loaded {len(examples)} YANASE examples")
    return examples


def stratified_split_yanase_dataset(dataset: List[dspy.Example],
                                     train_ratio=0.6,
                                     val_ratio=0.2,
                                     test_ratio=0.2,
                                     max_examples_per_seller=None,
                                     random_seed=42,
                                     verbose=True,
                                     stratify_by=None):
    """
    Split YANASE dataset with optional stratification by specified field(s).

    Args:
        dataset: List of dspy.Example objects
        train_ratio: Proportion for training (default: 0.6)
        val_ratio: Proportion for validation (default: 0.2)
        test_ratio: Proportion for testing (default: 0.2)
        max_examples_per_seller: Maximum examples per group when stratifying.
                                 Downsamples large groups to prevent dominance. (default: None)
        random_seed: Random seed for reproducibility (default: 42)
        verbose: Print detailed split statistics (default: True)
        stratify_by: Field name(s) to stratify by. Options:
                    - None: Simple random split (no stratification)
                    - str: Single field name (e.g., 'expected_soldTel')
                    - List[str]: Multiple fields (e.g., ['expected_soldTel', 'expected_soldName1'])

    Returns:
        Tuple of (train_set, val_set, test_set)

    Examples:
        >>> # Simple random split (no stratification)
        >>> train, val, test = stratified_split_yanase_dataset(dataset)

        >>> # Stratify by single field
        >>> train, val, test = stratified_split_yanase_dataset(dataset, stratify_by='expected_shipPostal')

        >>> # Stratify by multiple fields (seller)
        >>> train, val, test = stratified_split_yanase_dataset(
        ...     dataset, stratify_by=['expected_soldTel', 'expected_soldName1']
        ... )
    """
    from collections import defaultdict

    random.seed(random_seed)

    # ========================================================================
    # NO STRATIFICATION: Simple random split
    # ========================================================================
    if stratify_by is None:
        shuffled_dataset = dataset.copy()
        random.shuffle(shuffled_dataset)

        total = len(shuffled_dataset)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        train_set = shuffled_dataset[:train_size]
        val_set = shuffled_dataset[train_size:train_size + val_size]
        test_set = shuffled_dataset[train_size + val_size:]

        if verbose:
            print(f"\n{'='*70}")
            print(f"📊 Random Split (Seed: {random_seed})")
            print(f"{'='*70}")
            print(f"Total: {total} examples")
            print(f"\n{'Split':<10} {'Examples':<12} {'Percentage':<12}")
            print(f"{'-'*40}")
            print(f"{'Train':<10} {len(train_set):<12} {len(train_set)/total:.1%}")
            print(f"{'Val':<10} {len(val_set):<12} {len(val_set)/total:.1%}")
            print(f"{'Test':<10} {len(test_set):<12} {len(test_set)/total:.1%}")
            print(f"{'='*70}\n")

        return train_set, val_set, test_set

    # ========================================================================
    # STRATIFIED SPLIT: Group by specified field(s)
    # ========================================================================
    # Normalize stratify_by to list
    if isinstance(stratify_by, str):
        stratify_fields = [stratify_by]
    else:
        stratify_fields = list(stratify_by)

    # Auto-detect and fix field names (add "expected_" prefix if needed)
    corrected_fields = []
    for field in stratify_fields:
        # Check if field exists as-is
        test_example = dataset[0] if dataset else None
        if test_example and hasattr(test_example, field):
            corrected_fields.append(field)
        # Try with "expected_" prefix (aligned with dspy object naming)
        elif test_example and not field.startswith('expected_') and hasattr(test_example, f'expected_{field}'):
            corrected_field = f'expected_{field}'
            corrected_fields.append(corrected_field)
            if verbose:
                print(f"ℹ️  Auto-corrected field name: '{field}' → '{corrected_field}'")
        else:
            # Keep original if neither works (will use default empty value)
            corrected_fields.append(field)
            if verbose:
                print(f"⚠️  Field '{field}' not found in dataset, using empty default")

    stratify_fields = corrected_fields

    # Group examples by stratification key
    groups = defaultdict(list)

    for example in dataset:
        # Build stratification key from specified fields
        key_values = []
        for field in stratify_fields:
            value = getattr(example, field, '')
            # Handle both string and non-string values
            key_values.append(str(value).strip() if hasattr(value, 'strip') else str(value))

        # Use tuple as key (hashable)
        key = tuple(key_values)
        groups[key].append(example)

    # Apply downsampling if max_examples_per_seller is specified
    original_total = sum(len(examples) for examples in groups.values())
    if max_examples_per_seller is not None:
        balanced_groups = {}
        downsampled_groups = []

        for group_key, examples in groups.items():
            if len(examples) > max_examples_per_seller:
                # Randomly sample max_examples_per_seller examples
                balanced_groups[group_key] = random.sample(examples, max_examples_per_seller)
                downsampled_groups.append((group_key, len(examples), max_examples_per_seller))
            else:
                balanced_groups[group_key] = examples

        groups = balanced_groups

        if verbose and downsampled_groups:
            downsampled_total = sum(len(examples) for examples in groups.values())
            print(f"\n⚖️  Applied downsampling (max {max_examples_per_seller} examples per group)")
            print(f"   Total examples: {original_total} → {downsampled_total} ({original_total - downsampled_total} removed)")
            print(f"   Downsampled groups:")
            for group_key, original, kept in downsampled_groups:
                key_str = ' | '.join(group_key)
                print(f"     - {key_str}: {original} → {kept} examples")

    # Get list of unique groups
    unique_groups = list(groups.keys())
    total_groups = len(unique_groups)

    # Warn if too few groups for effective stratification
    if verbose and total_groups < 3:
        print(f"\n⚠️  WARNING: Only {total_groups} unique group(s) in dataset!")
        print(f"   Stratification may not be effective with so few groups.")
        print(f"   Consider using different stratification fields or no stratification.\n")

    # Smart stratification: Split large groups proportionally, keep small groups intact
    train_set = []
    val_set = []
    test_set = []

    # Separate large groups (≥3 examples) from small groups
    large_groups = []
    small_groups = []

    for group_key in unique_groups:
        if len(groups[group_key]) >= 3:
            large_groups.append(group_key)
        else:
            small_groups.append(group_key)

    # Split large groups proportionally to maintain ratios
    for group_key in large_groups:
        examples = groups[group_key].copy()
        random.shuffle(examples)
        n = len(examples)

        train_count = max(1, int(n * train_ratio))
        val_count = max(1, int(n * val_ratio))

        train_set.extend(examples[:train_count])
        val_set.extend(examples[train_count:train_count + val_count])
        test_set.extend(examples[train_count + val_count:])

    # Distribute small groups across splits to balance totals
    random.shuffle(small_groups)

    # Calculate how many examples we need in each split
    total_examples = len(dataset)
    target_train = int(total_examples * train_ratio)
    target_val = int(total_examples * val_ratio)
    target_test = total_examples - target_train - target_val

    # Greedily assign small groups to the split that needs more examples
    for group_key in small_groups:
        examples = groups[group_key]

        # Find which split is furthest from its target
        train_deficit = target_train - len(train_set)
        val_deficit = target_val - len(val_set)
        test_deficit = target_test - len(test_set)

        if train_deficit >= val_deficit and train_deficit >= test_deficit:
            train_set.extend(examples)
        elif val_deficit >= test_deficit:
            val_set.extend(examples)
        else:
            test_set.extend(examples)

    # Shuffle each split to avoid ordering bias
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    if verbose:
        total_examples = len(dataset)
        fields_str = ', '.join(stratify_fields)
        print(f"\n{'='*70}")
        print(f"📊 Stratified Split Summary (Seed: {random_seed})")
        print(f"{'='*70}")
        print(f"Stratify by: {fields_str}")
        print(f"Strategy: Split large groups (≥3 examples) proportionally")
        print(f"Total: {total_examples} examples, {total_groups} unique groups")
        print(f"  - Large groups (split proportionally): {len(large_groups)}")
        print(f"  - Small groups (kept intact): {len(small_groups)}")

        print(f"\n{'Split':<10} {'Examples':<12} {'Target':<12} {'Actual %':<12}")
        print(f"{'-'*50}")
        print(f"{'Train':<10} {len(train_set):<12} {int(total_examples*train_ratio):<12} {len(train_set)/total_examples:.1%}")
        print(f"{'Val':<10} {len(val_set):<12} {int(total_examples*val_ratio):<12} {len(val_set)/total_examples:.1%}")
        print(f"{'Test':<10} {len(test_set):<12} {total_examples - int(total_examples*train_ratio) - int(total_examples*val_ratio):<12} {len(test_set)/total_examples:.1%}")

        # Collect detailed group distribution statistics
        from collections import defaultdict

        group_distribution = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})

        for example in train_set:
            key_values = []
            for field in stratify_fields:
                value = getattr(example, field, '')
                key_values.append(str(value).strip() if hasattr(value, 'strip') else str(value))
            group_distribution[tuple(key_values)]['train'] += 1

        for example in val_set:
            key_values = []
            for field in stratify_fields:
                value = getattr(example, field, '')
                key_values.append(str(value).strip() if hasattr(value, 'strip') else str(value))
            group_distribution[tuple(key_values)]['val'] += 1

        for example in test_set:
            key_values = []
            for field in stratify_fields:
                value = getattr(example, field, '')
                key_values.append(str(value).strip() if hasattr(value, 'strip') else str(value))
            group_distribution[tuple(key_values)]['test'] += 1

        # Show group distribution details
        print(f"\n{'Group Distribution Details':<50}")
        print(f"{'-'*70}")
        print(f"{'Group':<30} {'Total':<8} {'Train':<8} {'Val':<8} {'Test':<8}")
        print(f"{'-'*70}")

        # Sort by total size (largest first)
        sorted_groups = sorted(
            group_distribution.items(),
            key=lambda x: sum(x[1].values()),
            reverse=True
        )

        for group_key, counts in sorted_groups:
            group_str = ' | '.join(group_key) if len(group_key) > 1 else group_key[0]
            group_str = group_str[:28] + '..' if len(group_str) > 30 else group_str
            total = counts['train'] + counts['val'] + counts['test']

            # Mark if group was split
            split_marker = ' *' if (counts['train'] > 0 and counts['val'] > 0) or \
                                   (counts['train'] > 0 and counts['test'] > 0) or \
                                   (counts['val'] > 0 and counts['test'] > 0) else ''

            print(f"{group_str:<30} {total:<8} {counts['train']:<8} {counts['val']:<8} {counts['test']:<8}{split_marker}")

        # Count group overlaps
        train_groups = {k for k, v in group_distribution.items() if v['train'] > 0}
        val_groups = {k for k, v in group_distribution.items() if v['val'] > 0}
        test_groups = {k for k, v in group_distribution.items() if v['test'] > 0}

        shared_groups = [k for k, v in group_distribution.items()
                        if sum([v['train'] > 0, v['val'] > 0, v['test'] > 0]) > 1]

        print(f"\n{'Stratification Statistics':<50}")
        print(f"{'-'*70}")
        print(f"Groups unique to train:  {len(train_groups - val_groups - test_groups)}")
        print(f"Groups unique to val:    {len(val_groups - train_groups - test_groups)}")
        print(f"Groups unique to test:   {len(test_groups - train_groups - val_groups)}")
        print(f"Groups split across sets: {len(shared_groups)}")

        if shared_groups:
            print(f"\n  Split groups (* in table above):")
            for group_key in shared_groups[:5]:  # Show top 5 split groups
                group_str = ' | '.join(group_key) if len(group_key) > 1 else group_key[0]
                counts = group_distribution[group_key]
                splits = []
                if counts['train'] > 0:
                    splits.append(f"train={counts['train']}")
                if counts['val'] > 0:
                    splits.append(f"val={counts['val']}")
                if counts['test'] > 0:
                    splits.append(f"test={counts['test']}")
                print(f"    - {group_str}: {', '.join(splits)}")

        # Show stratification quality metrics
        print(f"\n{'Stratification Quality':<50}")
        print(f"{'-'*70}")

        # Calculate diversity: unique groups per split / total groups
        train_diversity = len(train_groups) / total_groups
        val_diversity = len(val_groups) / total_groups
        test_diversity = len(test_groups) / total_groups

        print(f"Group diversity in train: {train_diversity:.1%} ({len(train_groups)}/{total_groups} groups)")
        print(f"Group diversity in val:   {val_diversity:.1%} ({len(val_groups)}/{total_groups} groups)")
        print(f"Group diversity in test:  {test_diversity:.1%} ({len(test_groups)}/{total_groups} groups)")

        print(f"\n{'='*70}\n")

    return train_set, val_set, test_set


####################################################
### PDF to image processing functions
####################################################

# Render a PDF into PIL images using PyMuPDF (fitz)
def process_pdf_into_image(pdf_path: str, dpi: int = 200):
    import fitz  # PyMuPDF
    images = []
    doc = fitz.open(pdf_path)
    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            mode = "RGB" if pix.n < 4 else "RGBA"
            im = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            if mode == "RGBA":
                im = im.convert("RGB")
            images.append(im)
    finally:
        doc.close()
    return images

# process the pdf files within a given directory into images into another directory
def process_pdf_files_into_images_with_rotation(pdf_dir: str, image_dir: str):
    import os
    os.makedirs(image_dir, exist_ok=True)
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            images = process_pdf_into_image(pdf_path)
            base = os.path.splitext(pdf_file)[0]
            for idx, image in enumerate(images, start=1):
                # add page number to the image name (1-based)
                image_name = f"{base}_page_{idx}.png"
                # do the rotation of the image
                osd_data = pytesseract.image_to_osd(image, output_type=Output.DICT)
                rotate_angle = osd_data['rotate']
                print("Detected rotation angle: ", rotate_angle)
                image = image.rotate(-rotate_angle, expand=True)
                out_path = os.path.join(image_dir, image_name)
                image.save(out_path)


if __name__ == "__main__":
    images_dir = "./data/png"
    gt_dir = "./data/json"
    dataset = load_japanese_order_dataset(images_dir=images_dir, gt_dir=gt_dir)
    print("dataset loaded successfully")
    print("# of data points: ", len(dataset))
    print("data instance 0: \n", dataset[0])