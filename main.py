import dspy
from utils import load_japanese_order_dataset, split_order_dataset
from config import *
from config import reconfigure_lm
from eval import japanese_order_extraction_metric
from opt import create_custom_strategy, OptimizationRunner, create_bootstrap_strategy, create_miprov2_strategy
from posthoc_analysis import analyze_model_performance

from datetime import datetime
from typing import List, Dict
import json
import os
import dotenv
import argparse

dotenv.load_dotenv(override=True)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Japanese Order Extraction Optimization')
parser.add_argument('--model-name', type=str, default=None,
                    help='Model name to use (default: from .env MODEL_NAME)')
parser.add_argument('--load-extractor', type=str, default=None,
                    help='Path to previously saved extractor to load (default: from .env CACHED_EXTRACTOR_PATH)')
parser.add_argument('--skip-load', action='store_true',
                    help='Skip loading cached extractor even if it exists')
parser.add_argument('--analyze-sets', type=str, default='test', choices=['val', 'test', 'both'],
                    help='Which dataset to analyze: val, test, or both (default: test)')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='Save the optimized model after training')
parser.add_argument('--save-results', action='store_true', default=False,
                    help='Save performance analysis results to JSON')

args = parser.parse_args()

# Reconfigure model if custom model name provided
if args.model_name:
    print(f"\n🔄 Reconfiguring DSPy with custom model: {args.model_name}")
    reconfigure_lm(args.model_name)
    print(f"✅ Model switched to: {args.model_name}\n")

# Display configuration
print(f"\n{'='*80}")
print(f"🔧 Configuration:")
print(f"  Model: {args.model_name or os.getenv('MODEL_NAME', 'Unknown')}")
print(f"  Load extractor: {args.load_extractor or dotenv.get_key(dotenv.find_dotenv(), 'CACHED_EXTRACTOR_PATH') if not args.skip_load else 'No'}")
print(f"  Analyze sets: {args.analyze_sets}")
print(f"  Save model: {args.save_model}")
print(f"  Save results: {args.save_results}")
print(f"{'='*80}\n")

# Load data
dataset_path = "./data"
dataset = load_japanese_order_dataset(dataset_path)
train_set, val_set, test_set = split_order_dataset(dataset)
print("train set: ", len(train_set))
print("val set: ", len(val_set))
print("test set: ", len(test_set))

# Extractor
class JapaneseOrderExtraction(dspy.Signature):
    """Extract structured order information from Japanese order images with high precision. Output empty string \"\" if not found"""
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

# Create extractor
order_extractor = dspy.ChainOfThought(JapaneseOrderExtraction)

# Load cached extractor if available
if not args.skip_load:
    cached_extractor_path = args.load_extractor or dotenv.get_key(dotenv.find_dotenv(), "CACHED_EXTRACTOR_PATH")
    if cached_extractor_path and os.path.exists(cached_extractor_path) and os.path.isfile(cached_extractor_path):
        print(f"📂 Loading cached extractor from: {cached_extractor_path}")
        order_extractor.load(cached_extractor_path)
        print("✅ Extractor loaded successfully\n")
    else:
        print(f"ℹ️  No cached extractor found, starting from scratch\n")
else:
    print(f"⏭️  Skipping extractor loading (--skip-load)\n")

# Evaluation function
evaluate_order_extraction = dspy.Evaluate(
    devset=None,  
    metric=japanese_order_extraction_metric,
    num_threads=4,  
    display_progress=True,
    display_table=True  
)

# Start optimization
custom_strategies = [
    create_bootstrap_strategy(
        name="Enhanced Bootstrap",
        description="Enhanced BootstrapFewShotWithRandomSearch with balanced parameters",
        max_labeled_demos=4,
        max_bootstrapped_demos=4,
        num_candidate_programs=16,
        num_threads=4,
        max_rounds=2,
    ),
    create_bootstrap_strategy(
        name="Conservative Bootstrap",
        description="Conservative approach with more labeled examples, fewer bootstrapped",
        max_labeled_demos=6,
        max_bootstrapped_demos=2,
        num_candidate_programs=20,
        num_threads=4,
        max_rounds=1,
    ),
    # create_miprov2_strategy(
    #     name="MiProV2",
    #     description="MiProV2 with balanced demo and candidate settings",
    #     auto="medium"
    # )
]

runner = OptimizationRunner(
    metric=japanese_order_extraction_metric,
    student_model=order_extractor,
    train_set=train_set,
    val_set=val_set,
    evaluate_fn=evaluate_order_extraction
)

optimized_extractor, best_score = runner.run_optimization(custom_strategies)

# Analyze performance based on --analyze-sets argument
print(f"\n{'='*80}")
print(f"📊 Performance Analysis")
print(f"{'='*80}\n")

all_field_scores = {}
all_total_scores = []

if args.analyze_sets in ['val', 'both']:
    print(f"Analyzing validation set ({len(val_set)} examples)...")
    val_field_scores, val_total_scores = analyze_model_performance(optimized_extractor, val_set)
    print(f"\n✅ Validation set average score: {sum(val_total_scores)/len(val_total_scores):.3f}\n")
    all_field_scores['val'] = val_field_scores
    all_total_scores.extend(val_total_scores)

if args.analyze_sets in ['test', 'both']:
    print(f"Analyzing test set ({len(test_set)} examples)...")
    test_field_scores, test_total_scores = analyze_model_performance(optimized_extractor, test_set)
    print(f"\n✅ Test set average score: {sum(test_total_scores)/len(test_total_scores):.3f}\n")
    all_field_scores['test'] = test_field_scores
    all_total_scores.extend(test_total_scores)

# Use the last analyzed set for saving
field_score = test_field_scores if args.analyze_sets in ['test', 'both'] else val_field_scores
total_score = test_total_scores if args.analyze_sets in ['test', 'both'] else val_total_scores

# Save results (if requested)
model_name = (args.model_name or os.getenv("MODEL_NAME", "Unknown Model")).replace("/", "_")
current_time = datetime.now().strftime("%Y%m%d%H")
best_strategy_name = runner.best_strategy_name.replace(" ", "_") if hasattr(runner, 'best_strategy_name') else "Unknown"

if args.save_model or args.save_results:
    # Create directory if not exists
    cache_dir = dotenv.get_key(dotenv.find_dotenv(), "CACHED_EXTRACTOR_DIR")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Save optimized model
    if args.save_model:
        model_path = os.path.join(
            cache_dir,
            f"{model_name}_{best_strategy_name}_optimized_extractor_{current_time}.json"
        )
        optimized_extractor.save(model_path)
        print(f"\n💾 Saved optimized model to: {model_path}")

    # Save performance results
    if args.save_results:
        results_path = os.path.join(
            cache_dir,
            f"{model_name}_{best_strategy_name}_performance_analysis_{current_time}.json"
        )
        with open(results_path, "w") as f:
            json.dump({
                "analyze_sets": args.analyze_sets,
                "field_scores": all_field_scores,
                "total_scores": all_total_scores
            }, f, indent=2)
        print(f"💾 Saved performance results to: {results_path}")

print(f"\n{'='*80}")
print(f"✅ Pipeline complete!")
print(f"{'='*80}\n")

