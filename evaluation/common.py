import re
import unicodedata
from difflib import SequenceMatcher


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
