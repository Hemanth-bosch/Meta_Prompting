import dspy
from typing import List, Dict


class JapaneseOrderExtraction(dspy.Signature):
    """Extract structured order information from Japanese order images with high precision. Output empty string "" if not found"""
    image: dspy.Image = dspy.InputField(desc="Japanese order document image - analyze all visible text carefully")

    # Seller Information - Company issuing the order
    soldName1: str = dspy.OutputField(desc="Primary seller company name (会社名) - exact text from top right corner")
    soldName2: str = dspy.OutputField(desc="Secondary seller info like branch/department (営業所/支店) - below company name")
    soldPostal: str = dspy.OutputField(desc="Seller postal code (〒) - 7 digits with hyphen format like 177-0032")
    soldAddress: str = dspy.OutputField(desc="Complete seller address - include prefecture, city, and street details")
    soldTex: str = dspy.OutputField(desc="Seller telephone number with area code in parentheses format like (03)5372-6111")
    soldFax: str = dspy.OutputField(desc="Seller fax number with area code in parentheses format like (03)5372-5043")

    # Basic order information
    CRD: str = dspy.OutputField(desc="Order date in Japanese format YYYY年MM月DD日 like 2024年7月31日")

    # Recipient/Shipping information - Where goods are being sent
    shipPostal: str = dspy.OutputField(desc="Shipping postal code (〒) - 7 digits with hyphen format.")
    shipTel: str = dspy.OutputField(desc="Shipping telephone number with area code in parentheses.")
    shipAddress: str = dspy.OutputField(desc="Complete shipping address - include all prefecture, city, street details.")
    companyName: str = dspy.OutputField(desc="Recipient company name - company receiving the goods.")

    # Order Details
    comments: str = dspy.OutputField(desc="Special comments for delivery instructions (from 特記事項): output '直送' if direct delivery is indicated, otherwise output empty string ''")
    PO: List[str] = dspy.OutputField(desc="Purchase Order numbers - usually starts with P followed by digits like P45829")
    productDetail: List[Dict[str, str]] = dspy.OutputField(desc="Product details with exact 'productNumber' (from 商品コード) and 'QTY' (from 注文数) from product table")
