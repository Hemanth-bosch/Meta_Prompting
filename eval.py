import re
import unicodedata
from difflib import SequenceMatcher
import dspy

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

def normalize_japanese_text(text: str) -> str:
    """Enhanced text normalization for Japanese text"""
    if not text:
        return ""
    
    # Convert to string and normalize unicode
    text = str(text).strip()
    text = unicodedata.normalize('NFKC', text)
    
    # Remove common variations
    text = re.sub(r'\s+', '', text)  # Remove all whitespace
    text = re.sub(r'[（()）]', '', text)  # Remove parentheses variations
    text = re.sub(r'[〒]', '', text)  # Remove postal symbol
    text = re.sub(r'[年月日]', '-', text)  # Standardize date separators
    text = re.sub(r'[-−‐―]+', '-', text)  # Normalize hyphens
    text = text.replace('株式会社', '').replace('(株)', '').replace('（株）', '')
    text = text.replace('有限会社', '').replace('(有)', '').replace('（有）', '')
    
    return text.lower()

def calculate_enhanced_similarity_score(pred_value, expected_value, field_type):
    """Enhanced similarity calculation with field-specific logic"""
    pred_norm = normalize_japanese_text(pred_value)
    expected_norm = normalize_japanese_text(expected_value)
    
    if not expected_norm:
        return 1.0 if not pred_norm else 0.0
    
    # Field-specific scoring
    if field_type in ['soldPostal', 'shipPostal']:
        # Postal codes should match exactly after normalization
        return 1.0 if pred_norm == expected_norm else 0.0
    
    elif field_type in ['soldTex', 'shipTel', 'soldFax']:
        # Phone/fax numbers - remove formatting and compare digits
        pred_digits = re.sub(r'[^\d]', '', pred_value)
        expected_digits = re.sub(r'[^\d]', '', expected_value)
        return 1.0 if pred_digits == expected_digits else 0.0
    
    elif field_type == 'CRD':
        # Date comparison - extract year, month, day
        pred_date = extract_date_components(pred_value)
        expected_date = extract_date_components(expected_value)
        return 1.0 if pred_date == expected_date else 0.0
    
    else:
        # General text similarity with fuzzy matching
        similarity = SequenceMatcher(None, pred_norm, expected_norm).ratio()
        
        # Moderate boost for partial matches in company names (capped at 1.0)
        if field_type in ['soldName1', 'companyName'] and similarity > 0.5:
            # Check if key terms are present
            key_terms = expected_norm.split()
            if key_terms:  # Avoid division by zero
                matches = sum(1 for term in key_terms if term in pred_norm)
                if matches > 0:
                    boost = 0.1 * matches / len(key_terms)  # Reduced boost
                    similarity = min(1.0, similarity + boost)  # Cap at 1.0
        
        return min(1.0, similarity)  # Ensure never exceeds 1.0

def extract_date_components(date_str):
    """Extract year, month, day from Japanese date format"""
    if not date_str:
        return None
    
    # Match various date patterns
    patterns = [
        r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日',
        r'(\d{4})-(\d{1,2})-(\d{1,2})',
        r'(\d{4})/(\d{1,2})/(\d{1,2})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, date_str)
        if match:
            year, month, day = match.groups()
            return (int(year), int(month), int(day))
    
    return None

def evaluate_po_numbers(pred_pos, expected_pos):
    """Evaluate PO number lists"""
    if not expected_pos:
        return 1.0 if not pred_pos else 0.8
    
    if not pred_pos:
        return 0.0
    
    # Normalize PO numbers
    pred_normalized = [normalize_japanese_text(po) for po in pred_pos]
    expected_normalized = [normalize_japanese_text(po) for po in expected_pos]
    
    # Calculate intersection score
    matches = len(set(pred_normalized) & set(expected_normalized))
    total_expected = len(expected_normalized)
    
    return matches / total_expected if total_expected > 0 else 0.0

def evaluate_product_details(pred_products, expected_products):
    """Evaluate product detail lists"""
    if not expected_products:
        return 1.0 if not pred_products else 0.8
    
    if not pred_products:
        return 0.0
    
    total_score = 0.0
    
    for expected_product in expected_products:
        best_match_score = 0.0
        
        for pred_product in pred_products:
            # Score both product number and quantity
            product_score = 0.0
            
            # Product number comparison - handle both English and Japanese field names
            expected_pn = expected_product.get('productNumber', '')
            pred_pn = (pred_product.get('productNumber', '') or 
                      pred_product.get('商品コード', '') or 
                      pred_product.get('productCode', ''))
            pn_similarity = calculate_enhanced_similarity_score(pred_pn, expected_pn, 'productNumber')
            product_score += 0.7 * pn_similarity
            
            # Quantity comparison - handle both field name variations (English and Japanese)
            expected_qty = expected_product.get('QTY', '') or expected_product.get('quantity', '')
            pred_qty = (pred_product.get('quantity', '') or 
                       pred_product.get('QTY', '') or
                       pred_product.get('注文数', '') or
                       pred_product.get('qty', ''))
            
            # More flexible quantity matching
            if str(expected_qty) == str(pred_qty):
                qty_similarity = 1.0
            elif expected_qty and pred_qty:
                try:
                    exp_num = float(re.sub(r'[^\d.]', '', str(expected_qty)))
                    pred_num = float(re.sub(r'[^\d.]', '', str(pred_qty)))
                    if exp_num == pred_num:
                        qty_similarity = 1.0
                    else:
                        # Partial credit for close quantities
                        diff_ratio = abs(exp_num - pred_num) / max(exp_num, pred_num)
                        qty_similarity = max(0.0, 1.0 - diff_ratio)
                except:
                    qty_similarity = 0.0
            else:
                qty_similarity = 0.0
            product_score += 0.3 * qty_similarity
            
            best_match_score = max(best_match_score, product_score)
        
        total_score += best_match_score
    
    return min(1.0, total_score / len(expected_products)) if expected_products else 0.0

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
    
    # # Add Japanese-specific optimization suggestions (reduced for token efficiency)
    # if score < 0.6:  # Only for very poor performance
    #     feedback_parts.extend(_generate_japanese_optimization_suggestions(prediction, example))
    
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


# ============================================================================
# YANASE Dataset Evaluation Functions
# ============================================================================

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
                    # Should match exactly
                    # print("Comparing", field_name, ":", pred_value, gt_value, normalize_japanese_text(pred_value), normalize_japanese_text(gt_value))
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


def _compare_quantities(pred_qty, gt_qty):
    """Compare quantity values, handling various formats"""
    if not gt_qty:
        return 1.0 if not pred_qty else 0.0

    if str(pred_qty) == str(gt_qty):
        return 1.0

    try:
        # Extract numeric values
        pred_num = float(re.sub(r'[^\d.]', '', str(pred_qty)))
        gt_num = float(re.sub(r'[^\d.]', '', str(gt_qty)))

        if pred_num == gt_num:
            return 1.0
        else:
            return 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def _compare_prices(pred_price, gt_price):
    """Compare price values, handling commas and currency symbols"""
    if not gt_price or gt_price == '':
        return 1.0 if (not pred_price or pred_price == '') else 0.5

    try:
        # Remove commas, currency symbols, spaces
        pred_clean = re.sub(r'[,¥$\s]', '', str(pred_price))
        gt_clean = re.sub(r'[,¥$\s]', '', str(gt_price))

        if pred_clean == gt_clean:
            return 1.0

        # Try numeric comparison
        pred_num = float(re.sub(r'[^\d.]', '', pred_clean))
        gt_num = float(re.sub(r'[^\d.]', '', gt_clean))

        if abs(pred_num - gt_num) < 0.01:  # Account for floating point
            return 1.0
        else:
            # Partial credit for close prices
            diff_ratio = abs(pred_num - gt_num) / max(pred_num, gt_num)
            return max(0.0, 1.0 - diff_ratio * 2)  # Stricter penalty for prices
    except (ValueError, ZeroDivisionError):
        return 0.0


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
        # Find corresponding prediction file
        # Ground truth: xxx_ground_truth.json
        # Prediction: xxx_extracted.json
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