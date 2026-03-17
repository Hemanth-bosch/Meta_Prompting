import re

from evaluation.common import (
    normalize_japanese_text,
    calculate_enhanced_similarity_score,
    _compare_quantities,
    _compare_prices,
)


def yanase_order_extraction_metric(ground_truth, prediction, trace=None, return_details=False):
    """
    evaluation metric for YANASE Japanese order extraction task

    Schema:
        - Seller info: soldName1, soldName2, soldTel, soldFax
        - Receiver/Shipping info: receiverCompany, shipPostal, shipAddress, shipTel, shipFax
        - Product orders: list of {receiver, orderCode, productCode, BoschCode, QTY, unitPrice, PCDPrice}

    Args:
        ground_truth: Dict with expected values (from ground truth JSON)
        prediction: Dict with predicted values (from model output)
        trace: Debug trace information (optional)
        return_details: If True, return detailed breakdown instead of just score

    Returns:
        float: Score between 0-1, or dict with detailed breakdown if return_details=True
    """

    # Field weights (seller + receiver/shipping combined)
    # Seller fields: LOW weights (mostly constant across orders)
    # Receiver/shipping fields: HIGH weights (vary per order, more critical)
    seller_weights = {
        # Seller info (mostly constant)
        'soldName1': 0.03,        # Seller company name
        'soldName2': 0.02,        # Seller department/branch
        'soldTel': 0.03,          # Seller telephone
        'soldFax': 0.02,          # Seller fax
        # Receiver/shipping info (varies per order - more important)
        'receiverCompany': 0.12,  # Recipient company name
        'shipPostal': 0.08,       # Shipping postal code
        'shipAddress': 0.15,      # Shipping address (critical)
        'shipTel': 0.05,          # Shipping telephone
        'shipFax': 0.05,          # Shipping fax
    }

    # Product orders weight (remainder of total)
    product_orders_weight = 0.45  # 45% of total score

    total_score = 0.0
    details = {}

    # Evaluate all fields (seller + receiver/shipping combined)
    field_score_total = 0.0
    field_details = {}

    for field_name, weight in seller_weights.items():
        pred_value = prediction.get(field_name, '')
        gt_value = ground_truth.get(field_name, '')

        # Calculate field-specific similarity
        if field_name == 'shipPostal':
            # Postal codes must match exactly
            similarity = 1.0 if normalize_japanese_text(pred_value) == normalize_japanese_text(gt_value) else 0.0
        elif field_name in ['soldTel', 'soldFax', 'shipTel', 'shipFax']:
            # Phone/fax numbers - compare digits only
            pred_digits = re.sub(r'[^\d]', '', str(pred_value))
            gt_digits = re.sub(r'[^\d]', '', str(gt_value))
            similarity = 1.0 if pred_digits == gt_digits else 0.0
        else:
            # Text fields - fuzzy matching
            similarity = calculate_enhanced_similarity_score(pred_value, gt_value, field_name)

        field_score = weight * similarity
        field_score_total += field_score
        field_details[field_name] = {
            'expected': gt_value,
            'predicted': pred_value,
            'similarity': similarity,
            'score': field_score
        }

    total_score += field_score_total
    details['field_info'] = field_details
    details['field_total_score'] = field_score_total

    # Evaluate product orders
    pred_products = prediction.get('productOrders', [])
    gt_products = ground_truth.get('productOrders', [])

    product_score, product_details = _evaluate_yanase_product_orders(pred_products, gt_products)
    total_score += product_orders_weight * product_score

    details['product_orders'] = product_details
    details['product_orders_score'] = product_orders_weight * product_score
    details['total_score'] = total_score

    if return_details:
        return details
    else:
        return min(1.0, total_score)


def _evaluate_yanase_product_orders(pred_products, gt_products):
    """
    Evaluate product orders list for YANASE dataset

    Returns:
        tuple: (score, details_dict)
    """
    if not gt_products:
        return (1.0 if not pred_products else 0.8, {'note': 'No products expected'})

    if not pred_products:
        return (0.0, {'error': 'No products extracted', 'expected_count': len(gt_products)})

    # Field weights within each product order
    field_weights = {
        'receiver': 0.10,
        'orderCode': 0.10,
        'productCode': 0.25,    # Critical field
        'BoschCode': 0.25,      # Critical field
        'QTY': 0.20,            # Important
        'unitPrice': 0.08,
        'PCDPrice': 0.02        # Often empty
    }

    total_score = 0.0
    matched_products = []
    unmatched_gt = []

    # For each ground truth product, find best matching prediction
    for i, gt_product in enumerate(gt_products):
        best_match_score = 0.0
        best_match_idx = -1
        best_match_details = {}

        for j, pred_product in enumerate(pred_products):
            product_score = 0.0
            match_details = {}

            # Compare each field
            for field_name, weight in field_weights.items():
                gt_value = str(gt_product.get(field_name, '')).strip()
                pred_value = str(pred_product.get(field_name, '')).strip()

                # Field-specific comparison
                if field_name in ['receiver', 'orderCode']:
                    similarity = 1.0 if normalize_japanese_text(pred_value) == normalize_japanese_text(gt_value) else 0.0
                elif field_name == 'QTY':
                    # Numeric comparison
                    similarity = _compare_quantities(pred_value, gt_value)
                elif field_name in ['unitPrice', 'PCDPrice']:
                    # Price comparison (may have commas, currency symbols)
                    similarity = _compare_prices(pred_value, gt_value)
                else:
                    # Product codes - allow fuzzy matching
                    similarity = calculate_enhanced_similarity_score(pred_value, gt_value, field_name)

                field_score = weight * similarity
                product_score += field_score

                match_details[field_name] = {
                    'expected': gt_value,
                    'predicted': pred_value,
                    'similarity': similarity
                }

            if product_score > best_match_score:
                best_match_score = product_score
                best_match_idx = j
                best_match_details = match_details

        total_score += best_match_score

        if best_match_score >= 0.5:  # Consider it a match
            matched_products.append({
                'gt_index': i,
                'pred_index': best_match_idx,
                'score': best_match_score,
                'details': best_match_details
            })
        else:
            unmatched_gt.append({
                'gt_index': i,
                'product': gt_product,
                'best_score': best_match_score
            })

    # Calculate final score
    avg_score = total_score / len(gt_products) if gt_products else 0.0

    # Penalty for count mismatch
    count_ratio = len(pred_products) / len(gt_products) if gt_products else 0.0
    if count_ratio > 1.2:  # Too many predictions
        avg_score *= 0.95

    details = {
        'expected_count': len(gt_products),
        'predicted_count': len(pred_products),
        'matched_count': len(matched_products),
        'average_score': avg_score,
        'matched_products': matched_products,
        'unmatched_ground_truth': unmatched_gt
    }

    return (avg_score, details)


def evaluate_yanase_extractions(predictions_dir, ground_truth_dir, verbose=False):
    """
    Batch evaluate YANASE extraction results against ground truth

    Args:
        predictions_dir: Directory containing prediction JSON files
        ground_truth_dir: Directory containing ground truth JSON files
        verbose: If True, print detailed results for each file

    Returns:
        dict: Evaluation results summary
    """
    import json
    from pathlib import Path

    pred_path = Path(predictions_dir)
    gt_path = Path(ground_truth_dir)

    results = {
        'total_files': 0,
        'evaluated_files': 0,
        'scores': [],
        'file_results': {}
    }

    # Find all ground truth files
    for gt_file in gt_path.glob('*_ground_truth.json'):
        base_name = gt_file.stem.replace('_ground_truth', '')
        pred_file = pred_path / f"{base_name}_extracted.json"

        if not pred_file.exists():
            if verbose:
                print(f"⚠️  No prediction found for {gt_file.name}")
            continue

        results['total_files'] += 1

        # Load files
        try:
            with open(gt_file, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)

            with open(pred_file, 'r', encoding='utf-8') as f:
                prediction = json.load(f)

            # Evaluate
            score = yanase_order_extraction_metric(ground_truth, prediction)
            results['scores'].append(score)
            results['evaluated_files'] += 1

            results['file_results'][base_name] = {
                'score': score,
                'gt_products': len(ground_truth.get('productOrders', [])),
                'pred_products': len(prediction.get('productOrders', []))
            }

            if verbose:
                print(f"✅ {base_name}: Score = {score:.3f} "
                      f"(GT: {len(ground_truth.get('productOrders', []))} products, "
                      f"Pred: {len(prediction.get('productOrders', []))} products)")

        except Exception as e:
            if verbose:
                print(f"❌ Error processing {gt_file.name}: {e}")

    # Calculate summary statistics
    if results['scores']:
        results['mean_score'] = sum(results['scores']) / len(results['scores'])
        results['min_score'] = min(results['scores'])
        results['max_score'] = max(results['scores'])
        results['median_score'] = sorted(results['scores'])[len(results['scores']) // 2]

    return results
