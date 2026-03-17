import dspy
from typing import List


class ReceiverAndOrder(dspy.Signature):
    receiver: str = dspy.OutputField(desc="Recipient entity code -  5 digits from field like '配送先'")
    orderCode: str = dspy.OutputField(desc="Customer order code - 5 digits from field like '注文No.'")
    productCode: str = dspy.OutputField(desc="Product code/number - 得意先品目ｺｰﾄﾞ")
    BoschCode: str = dspy.OutputField(desc="""Bosch manufacturer part number containing letters and numbers from field like 'メーカーパーツNo' from product table, excluding '商品名'.""")
    QTY: str = dspy.OutputField(desc="Quantity of the product - 数量")
    unitPrice: str = dspy.OutputField(desc="Unit price of the product - 単価")
    PCDPrice: str = dspy.OutputField(desc="Original price of the product before discount. By default, it is EMPTY string unless the price is annotated.")


class JapaneseOrderExtraction(dspy.Signature):
    """Extract structured order information from Japanese order images with exact text and high precision. Output empty string "" if not found"""
    image: dspy.Image = dspy.InputField(desc="Japanese order document image - analyze all visible text carefully")

    # Seller Information - Company issuing the order
    soldName1: str = dspy.OutputField(desc="Primary ORDER ISSUER legal company name from TOP-RIGHT block only. Extract EXACT first line only.")
    soldName2: str = dspy.OutputField(desc="ORDER ISSUER division/section line(s) IMMEDIATELY BELOW soldName1 in TOP-RIGHT block (e.g. 部 / 課).")
    soldTel: str = dspy.OutputField(desc="Seller telephone number - 10 digits in parentheses or hyphen format")
    soldFax: str = dspy.OutputField(desc="Seller fax number - 10 digits in parentheses or hyphen format")

    # Recipent/Shipping information - Where goods are being sent
    receiverCompany: str = dspy.OutputField(desc="Recipient full entity name from fields like '得意先名' BUT NOT shipping entity '配送先'. Include company name and org-units like 本部/支店/部/課. Return empty '' if not present.")
    shipPostal: str = dspy.OutputField(desc="Shipping address postal code (〒) - 7 digits with hyphen format like 177-0032")
    shipAddress: str = dspy.OutputField(desc="Complete shipping address - include all prefecture, city, street details")
    shipTel: str = dspy.OutputField(desc="Shipping telephone number - 10 digits in parentheses or hyphen format.")
    shipFax: str = dspy.OutputField(desc="Shipping fax number - 10 digits in parentheses or hyphen format")

    # Recipient and order information - What are goods and Where goods are being sent
    productOrders: List[ReceiverAndOrder] = dspy.OutputField(desc="List of recipient and order information - What are goods and customer/recipient details")
