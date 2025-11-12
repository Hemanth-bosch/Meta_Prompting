import dspy
from config import *
from config import reconfigure_lm
from typing import List, Dict
from pathlib import Path
import json
import os
import dotenv
import argparse

dotenv.load_dotenv(override=True)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Extract order information from Japanese order images')
parser.add_argument('--input-dir', type=str, default='data/YANASE_images',
                    help='Directory containing input images (default: data/YANASE_images)')
parser.add_argument('--output-dir', type=str, default='data/YANASE_jsons_updated',
                    help='Directory to save extracted JSON files (default: data/YANASE_jsons_updated)')
parser.add_argument('--model-name', type=str, default=None,
                    help='Model name to use (default: from .env MODEL_NAME)')
parser.add_argument('--no-cache', action='store_true',
                    help='Disable LLM response caching (default: caching enabled)')
parser.add_argument('--load-extractor', type=str, default=None,
                    help='Path to previously saved extractor to load (default: from .env CACHED_EXTRACTOR_PATH)')
parser.add_argument('--pattern', type=str, default='*.png',
                    help='File pattern to match (default: *.png)')

args = parser.parse_args()

# Reconfigure model if custom model name or cache setting provided
cache_enabled = not args.no_cache
if args.model_name or args.no_cache:
    model_to_use = args.model_name or os.getenv('MODEL_NAME')
    print(f"\n🔄 Reconfiguring DSPy with model: {model_to_use}, cache: {cache_enabled}")
    reconfigure_lm(model_to_use, cache=cache_enabled)
    print(f"✅ Model configured: {model_to_use}, cache: {cache_enabled}\n")

# Display configuration
print(f"\n{'='*80}")
print(f"🔧 Configuration:")
print(f"  Model: {args.model_name or os.getenv('MODEL_NAME', 'Unknown')}")
print(f"  Cache: {not args.no_cache}")
print(f"  Input directory: {args.input_dir}")
print(f"  Output directory: {args.output_dir}")
print(f"  File pattern: {args.pattern}")
print(f"  Load extractor: {args.load_extractor or dotenv.get_key(dotenv.find_dotenv(), 'CACHED_EXTRACTOR_PATH') or 'No'}")
print(f"{'='*80}\n")

# Define extraction schema (matching current mainv2.py)
# Extractor
class ReceiverAndOrder(dspy.Signature):
    receiver: str = dspy.OutputField(desc="Recipient entity code -  5 digits from field like '配送先'")
    orderCode: str = dspy.OutputField(desc="Customer order code - 5 digits from field like '注文No.'")
    productCode: str = dspy.OutputField(desc="Product code/number - 得意先品目ｺｰﾄﾞ")
    BoschCode: str = dspy.OutputField(desc="""Bosch manufacturer part number containing letters and numbers from field like 'メーカーパーツNo' from product table, excluding ‘商品名’.""")
    QTY: str = dspy.OutputField(desc="Quantity of the product - 数量")
    unitPrice: str = dspy.OutputField(desc="Unit price of the product - 単価")
    PCDPrice: str = dspy.OutputField(desc="Original price of the product before discount. By default, it is EMPTY string unless the price is annotated.")

class JapaneseOrderExtraction(dspy.Signature):
    """Extract structured order information from Japanese order images with exact text and high precision. Output empty string \"\" if not found"""
    image: dspy.Image = dspy.InputField(desc="Japanese order document image - analyze all visible text carefully")
    
    # Seller Information - Company issuing the order
    soldName1: str = dspy.OutputField(desc="Primary ORDER ISSUER legal company name from TOP-RIGHT block only. Extract EXACT first line only.")
    soldName2: str = dspy.OutputField(desc="ORDER ISSUER division/section line(s) IMMEDIATELY BELOW soldName1 in TOP-RIGHT block (e.g. 部 / 課).")
    soldTel: str = dspy.OutputField(desc="Seller telephone number - 10 digits in parentheses or hyphen format")
    soldFax: str = dspy.OutputField(desc="Seller fax number - 10 digits in parentheses or hyphen format")

    # Recipent/Shipping information - Where goods are being sent
    receiverCompany: str = dspy.OutputField(desc="Recipient full entity name from fields like '得意先名' BUT NOT shipping entity '配送先'. Include company name and org-units like 本部/支店/部/課. Return empty '' if not present.")
    shipPostal: str = dspy.OutputField(desc="Shipping address postal code (〒), often above product table - 7 digits with hyphen format like 177-0032")
    shipAddress: str = dspy.OutputField(desc="Complete shipping address - include all prefecture, city, street details")
    shipTel: str = dspy.OutputField(desc="Shipping telephone number, often above product table - 10 digits in parentheses or hyphen format")
    shipFax: str = dspy.OutputField(desc="Shipping fax number, often above product table - 10 digits in parentheses or hyphen format")
    
    
    # Recipient and order information - What are goods and Where goods are being sent
    productOrders: List[ReceiverAndOrder] = dspy.OutputField(desc="List of recipient and order information - What are goods and customer/recipient details")

# Create extractor
order_extractor = dspy.ChainOfThought(JapaneseOrderExtraction)

# Load cached extractor if available
cached_extractor_path = args.load_extractor or dotenv.get_key(dotenv.find_dotenv(), "CACHED_EXTRACTOR_PATH")
if cached_extractor_path and os.path.exists(cached_extractor_path) and os.path.isfile(cached_extractor_path):
    print(f"📂 Loading cached extractor from: {cached_extractor_path}")
    order_extractor.load(cached_extractor_path)
    print("✅ Extractor loaded successfully\n")
else:
    print(f"ℹ️  No cached extractor found, using base model\n")

# Convert dspy prediction to JSON-serializable dict
def convert_dspy_result_to_json(pred: dspy.Prediction) -> Dict[str, object]:
    """Convert DSPy prediction to JSON-serializable dictionary"""
    # Prefer dspy's built-in conversion
    data = pred.toDict() if hasattr(pred, "toDict") else {}
    # Drop non-serializable inputs if present
    data.pop("image", None)

    # Ensure nested product rows are dicts
    if "productOrders" in data and isinstance(data["productOrders"], list):
        normalized_rows: List[Dict[str, str]] = []
        for row in data["productOrders"]:
            if hasattr(row, "toDict"):
                row_dict = row.toDict()
            elif isinstance(row, dict):
                row_dict = row
            else:
                # Fallback: extract expected fields
                row_dict = {
                    "receiver": getattr(row, "receiver", ""),
                    "orderCode": getattr(row, "orderCode", ""),
                    "productCode": getattr(row, "productCode", ""),
                    "BoschCode": getattr(row, "BoschCode", ""),
                    "QTY": getattr(row, "QTY", ""),
                    "unitPrice": getattr(row, "unitPrice", ""),
                    "PCDPrice": getattr(row, "PCDPrice", ""),
                }
            normalized_rows.append(row_dict)
        data["productOrders"] = normalized_rows
    return data

# Process images
input_path = Path(args.input_dir)
output_path = Path(args.output_dir)

# Ensure output directory exists
output_path.mkdir(parents=True, exist_ok=True)

# Find all matching images
image_files = list(input_path.glob(args.pattern))

if not image_files:
    print(f"⚠️  No images found matching pattern '{args.pattern}' in {args.input_dir}")
    exit(1)

print(f"📸 Found {len(image_files)} images to process\n")
print(f"{'='*80}")

# Process each image
for idx, image_file in enumerate(image_files, 1):
    print(f"\n[{idx}/{len(image_files)}] Processing: {image_file.name}")

    try:
        # Load image and extract
        dspy_image = dspy.Image.from_file(str(image_file))
        result = order_extractor(image=dspy_image)

        # Convert result to JSON format
        json_result = convert_dspy_result_to_json(result)
        # Include source image for traceability
        json_result["source_image"] = str(image_file)

        # Create output filename based on input image
        output_filename = image_file.stem + "_extracted.json"
        output_file = output_path / output_filename

        # Save to JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_result, f, ensure_ascii=False, indent=4)

        print(f"✅ Saved to: {output_file}")

    except Exception as e:
        print(f"❌ Error processing {image_file.name}: {str(e)}")
        continue

print(f"\n{'='*80}")
print(f"✅ Extraction complete! Processed {len(image_files)} images")
print(f"📁 Results saved to: {output_path.resolve()}")
print(f"{'='*80}\n")
