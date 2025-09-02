import re
import unicodedata
from difflib import SequenceMatcher

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
            
            # Product number comparison
            expected_pn = expected_product.get('productNumber', '')
            pred_pn = pred_product.get('productNumber', '')
            pn_similarity = calculate_enhanced_similarity_score(pred_pn, expected_pn, 'productNumber')
            product_score += 0.7 * pn_similarity
            
            # Quantity comparison  
            expected_qty = expected_product.get('QTY', '')  # Note: JSON uses 'QTY'
            pred_qty = pred_product.get('quantity', '')     # Note: prediction uses 'quantity'
            qty_similarity = 1.0 if str(expected_qty) == str(pred_qty) else 0.0
            product_score += 0.3 * qty_similarity
            
            best_match_score = max(best_match_score, product_score)
        
        total_score += best_match_score
    
    return min(1.0, total_score / len(expected_products)) if expected_products else 0.0