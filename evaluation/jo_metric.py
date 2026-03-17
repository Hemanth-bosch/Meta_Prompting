import dspy

from evaluation.common import (
    normalize_japanese_text,
    calculate_enhanced_similarity_score,
    evaluate_po_numbers,
    evaluate_product_details,
)


def japanese_order_extraction_metric(example, prediction, trace=None) -> float:
    """
    Enhanced evaluation metric for Japanese order information extraction

    Args:
        example: Expected output examples
        prediction: DSPy model prediction results
        trace: Debug trace information

    Returns:
        float: Weighted score of the prediction (0-1 for DSPy compatibility)
    """

    # Enhanced field weights - more comprehensive scoring
    # Normalized field weights - ensuring they sum to exactly 1.0
    field_weights = {
        'soldName1': 0.12,      # Company name is critical
        'companyName': 0.12,    # Recipient company equally important
        'soldAddress': 0.09,    # Addresses are complex and important
        'shipAddress': 0.09,
        'soldPostal': 0.07,     # Postal codes should be exact
        'shipPostal': 0.07,
        'soldName2': 0.06,      # Secondary info
        'soldTex': 0.05,        # Phone numbers
        'shipTel': 0.05,
        'soldFax': 0.03,        # Less critical
        'CRD': 0.05,           # Date format matters
        'comments': 0.03,       # Comments can vary
        'PO': 0.07,            # PO numbers are important
        'productDetail': 0.10   # Product details are critical
    }

    total_score = 0.0
    min_similarity = 0.3  # Lowered threshold for partial credit

    # Evaluate string fields
    string_fields = [
        ('soldName1', prediction.soldName1, example.expected_sold_name1),
        ('companyName', prediction.companyName, example.expected_company_name),
        ('soldName2', prediction.soldName2, example.expected_sold_name2),
        ('soldAddress', prediction.soldAddress, example.expected_sold_address),
        ('shipAddress', prediction.shipAddress, example.expected_ship_address),
        ('soldPostal', prediction.soldPostal, example.expected_sold_postal),
        ('shipPostal', prediction.shipPostal, example.expected_ship_postal),
        ('soldTex', prediction.soldTex, example.expected_sold_tel),
        ('shipTel', prediction.shipTel, example.expected_ship_tel),
        ('soldFax', prediction.soldFax, example.expected_sold_fax),
        ('CRD', prediction.CRD, example.expected_order_date),
        ('comments', prediction.comments, example.expected_comments),
    ]

    for field_name, pred_value, expected_value in string_fields:
        if field_name in field_weights:
            similarity = calculate_enhanced_similarity_score(pred_value, expected_value, field_name)
            if similarity >= min_similarity:
                total_score += field_weights[field_name] * similarity

    # Evaluate PO numbers (list comparison)
    po_score = evaluate_po_numbers(prediction.PO, example.expected_po_numbers)
    total_score += field_weights['PO'] * po_score

    # Evaluate product details (complex list comparison)
    product_score = evaluate_product_details(prediction.productDetail, example.expected_product_info)
    total_score += field_weights['productDetail'] * product_score

    # Ensure score never exceeds 1.0
    return min(1.0, total_score)


#############################################################################
# fine-grained feedback for llm prompt optimization
#############################################################################
def japanese_order_extraction_metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Enhanced evaluation metric with detailed feedback for Japanese order extraction
    Compatible with GEPA optimizer for iterative improvement

    Args:
        example: Expected output examples
        prediction: DSPy model prediction results
        trace: Debug trace information
        pred_name: Name of the predictor module being evaluated
        pred_trace: Trace from the predictor (unused - reserved for future enhancement)

    Returns:
        dspy.Prediction with score and feedback, or float score if pred_name is None
    """
    # Note: pred_trace parameter reserved for future use
    _ = pred_trace

    # Calculate the base score using existing metric
    score = japanese_order_extraction_metric(example, prediction, trace)

    # If no predictor name specified, return just the score
    if pred_name is None:
        return score

    # Generate detailed feedback
    feedback_parts = []
    field_weights = {
        'soldName1': 0.12, 'companyName': 0.12, 'soldAddress': 0.09, 'shipAddress': 0.09,
        'soldPostal': 0.07, 'shipPostal': 0.07, 'soldName2': 0.06, 'soldTex': 0.05,
        'shipTel': 0.05, 'soldFax': 0.03, 'CRD': 0.05, 'comments': 0.03,
        'PO': 0.07, 'productDetail': 0.10
    }

    # Generate field-specific feedback
    feedback_parts.extend(_generate_string_field_feedback(prediction, example, field_weights))
    feedback_parts.extend(_generate_po_feedback(prediction.PO, example.expected_po_numbers))
    feedback_parts.extend(_generate_product_feedback(prediction.productDetail, example.expected_product_info))

    # Add performance summary
    performance_level = "excellent" if score >= 0.9 else "good" if score >= 0.7 else "needs improvement"
    feedback_parts.append(f"\nOverall Performance: {performance_level} (Score: {score:.3f})")

    feedback_text = "\n".join(feedback_parts)

    return dspy.Prediction(score=score, feedback=feedback_text)


def _generate_string_field_feedback(prediction, example, field_weights):
    """Generate feedback for string fields"""
    feedback = []

    string_fields = [
        ('soldName1', prediction.soldName1, example.expected_sold_name1, "seller name"),
        ('companyName', prediction.companyName, example.expected_company_name, "company name"),
        ('soldName2', prediction.soldName2, example.expected_sold_name2, "secondary seller info"),
        ('soldAddress', prediction.soldAddress, example.expected_sold_address, "seller address"),
        ('shipAddress', prediction.shipAddress, example.expected_ship_address, "shipping address"),
        ('soldPostal', prediction.soldPostal, example.expected_sold_postal, "seller postal code"),
        ('shipPostal', prediction.shipPostal, example.expected_ship_postal, "shipping postal code"),
        ('soldTex', prediction.soldTex, example.expected_sold_tel, "seller phone"),
        ('shipTel', prediction.shipTel, example.expected_ship_tel, "shipping phone"),
        ('soldFax', prediction.soldFax, example.expected_sold_fax, "seller fax"),
        ('CRD', prediction.CRD, example.expected_order_date, "order date"),
        ('comments', prediction.comments, example.expected_comments, "comments"),
    ]

    critical_errors = []
    minor_issues = []

    for field_name, pred_value, expected_value, display_name in string_fields:
        if field_name not in field_weights:
            continue

        similarity = calculate_enhanced_similarity_score(pred_value, expected_value, field_name)
        weight = field_weights[field_name]

        if similarity < 0.3:
            if weight >= 0.08:  # High importance fields
                critical_errors.append(f"Critical: {display_name} extraction failed (expected: '{expected_value}', got: '{pred_value}')")
            else:
                minor_issues.append(f"Minor: {display_name} not extracted correctly (expected: '{expected_value}', got: '{pred_value}')")
        elif similarity < 0.7:
            minor_issues.append(f"Partial: {display_name} partially correct (expected: '{expected_value}', got: '{pred_value}')")

    if critical_errors:
        feedback.append("🚨 Critical Issues:")
        feedback.extend([f"  - {error}" for error in critical_errors])

    if minor_issues:
        feedback.append("⚠️  Minor Issues:")
        feedback.extend([f"  - {issue}" for issue in minor_issues])

    return feedback


def _generate_po_feedback(pred_pos, expected_pos):
    """Generate feedback for PO numbers"""
    feedback = []

    if not expected_pos:
        if pred_pos:
            feedback.append("ℹ️  PO Numbers: Extracted PO numbers when none expected - may indicate over-extraction")
        return feedback

    if not pred_pos:
        feedback.append("🚨 PO Numbers: Failed to extract any PO numbers")
        feedback.append(f"  Expected: {expected_pos}")
        return feedback

    pred_normalized = [normalize_japanese_text(po) for po in pred_pos]
    expected_normalized = [normalize_japanese_text(po) for po in expected_pos]

    matches = set(pred_normalized) & set(expected_normalized)
    missing = set(expected_normalized) - set(pred_normalized)
    extra = set(pred_normalized) - set(expected_normalized)

    if len(matches) == len(expected_normalized) and not extra:
        feedback.append("✅ PO Numbers: All correctly extracted")
    else:
        feedback.append("⚠️  PO Numbers: Partial extraction")
        if missing:
            feedback.append(f"  Missing: {list(missing)}")
        if extra:
            feedback.append(f"  Extra: {list(extra)}")
        if matches:
            feedback.append(f"  Correct: {list(matches)}")

    return feedback


def _generate_product_feedback(pred_products, expected_products):
    """Generate feedback for product details"""
    feedback = []

    if not expected_products:
        if pred_products:
            feedback.append("ℹ️  Products: Extracted products when none expected")
        return feedback

    if not pred_products:
        feedback.append("🚨 Products: Failed to extract any product information")
        feedback.append(f"  Expected {len(expected_products)} products")
        return feedback

    feedback.append(f"📦 Products: Expected {len(expected_products)}, extracted {len(pred_products)}")

    for i, expected_product in enumerate(expected_products):
        expected_pn = expected_product.get('productNumber', '')
        expected_qty = expected_product.get('QTY', '') or expected_product.get('quantity', '')

        best_match = None
        best_score = 0.0

        for pred_product in pred_products:
            pred_pn = (pred_product.get('productNumber', '') or
                      pred_product.get('商品コード', '') or
                      pred_product.get('productCode', ''))
            pred_qty = (pred_product.get('quantity', '') or
                       pred_product.get('QTY', '') or
                       pred_product.get('注文数', '') or
                       pred_product.get('qty', ''))

            pn_sim = calculate_enhanced_similarity_score(pred_pn, expected_pn, 'productNumber')
            qty_match = 1.0 if str(expected_qty) == str(pred_qty) else 0.0

            total_sim = 0.7 * pn_sim + 0.3 * qty_match
            if total_sim > best_score:
                best_score = total_sim
                best_match = (pred_pn, pred_qty)

        if best_score >= 0.7:
            feedback.append(f"  ✅ Product {i+1}: Good match (score: {best_score:.2f})")
        elif best_score >= 0.3:
            feedback.append(f"  ⚠️  Product {i+1}: Partial match (score: {best_score:.2f})")
            feedback.append(f"      Expected: {expected_pn} x {expected_qty}")
            if best_match:
                feedback.append(f"      Got: {best_match[0]} x {best_match[1]}")
        else:
            feedback.append(f"  🚨 Product {i+1}: No good match found")
            feedback.append(f"      Expected: {expected_pn} x {expected_qty}")

    return feedback


def _generate_japanese_optimization_suggestions(prediction, example):
    """Generate Japanese-specific optimization suggestions"""
    # Note: example parameter available for future context-aware suggestions
    _ = example

    suggestions = []
    suggestions.append("\n🔧 Japanese Order Extraction Optimization Suggestions:")

    # Check for common Japanese extraction issues
    if not prediction.soldName1 or not prediction.companyName:
        suggestions.append("  - Focus on identifying Japanese company name patterns (株式会社, 有限会社, etc.)")
        suggestions.append("  - Look for name variations with and without company suffixes")

    if not prediction.soldAddress or not prediction.shipAddress:
        suggestions.append("  - Improve Japanese address parsing (都道府県, 市区町村 patterns)")
        suggestions.append("  - Pay attention to postal codes (〒 symbol) and address structure")

    if not prediction.soldPostal or not prediction.shipPostal:
        suggestions.append("  - Look for 7-digit postal codes (XXX-XXXX format)")
        suggestions.append("  - Consider postal codes may appear with or without 〒 symbol")

    if not prediction.CRD:
        suggestions.append("  - Focus on Japanese date patterns (年月日 format)")
        suggestions.append("  - Consider various date representations (YYYY-MM-DD, YYYY/MM/DD)")

    if not prediction.PO:
        suggestions.append("  - Look for order numbers, reference numbers, or PO identifiers")
        suggestions.append("  - Check for alphanumeric codes that might be purchase orders")

    if not prediction.productDetail:
        suggestions.append("  - Focus on product information extraction")
        suggestions.append("  - Look for product codes, quantities, and item descriptions")
        suggestions.append("  - Pay attention to tabular data that might contain product details")

    # Add linguistic suggestions
    suggestions.append("  - Consider character normalization (full-width vs half-width)")
    suggestions.append("  - Account for various Japanese text encodings and representations")
    suggestions.append("  - Use context clues from document structure and formatting")

    return suggestions
