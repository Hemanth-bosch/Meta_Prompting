import dspy
from signatures.yanase import ReceiverAndOrder, JapaneseOrderExtraction
from utils import load_yanase_order_dataset, stratified_split_yanase_dataset, split_order_dataset
from config import *
from config import reconfigure_lm
from evaluation.yanase_metric import yanase_order_extraction_metric
from optimization.opt import OptimizationRunner, create_miprov2_strategy, create_bootstrap_strategy
from evaluation.posthoc_analysis import analyze_yanase_model_performance

from datetime import datetime
import json
import os
import dotenv
import argparse

dotenv.load_dotenv(override=True)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='YANASE Order Extraction Optimization')
# Data path arguments
parser.add_argument('--images-dir', type=str, default=None,
                    help='Path to YANASE images directory (default: from env YANASE_IMAGES_DIR or "data/YANASE_images")')
parser.add_argument('--ground-truth-dir', type=str, default=None,
                    help='Path to ground truth JSON directory (default: from env YANASE_GT_DIR or "data/YANASE_ground_truth")')
# Model and caching arguments
parser.add_argument('--model-name', type=str, default=None,
                    help='Model name to use (default: from .env MODEL_NAME)')
parser.add_argument('--no-cache', action='store_true',
                    help='Disable LLM response caching (default: caching enabled)')
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

# Reconfigure model if custom model name or cache setting provided
cache_enabled = not args.no_cache
if args.model_name or args.no_cache:
    model_to_use = args.model_name or os.getenv('MODEL_NAME')
    print(f"\n🔄 Reconfiguring DSPy with model: {model_to_use}, cache: {cache_enabled}")
    reconfigure_lm(model_to_use, cache=cache_enabled)
    print(f"✅ Model configured: {model_to_use}, cache: {cache_enabled}\n")

# Determine data paths (from args, env, or defaults)
images_dir = args.images_dir or os.getenv("YANASE_IMAGES_DIR", "data/YANASE_images")
ground_truth_dir = args.ground_truth_dir or os.getenv("YANASE_GT_DIR", "data/YANASE_ground_truth")

# Display configuration
print(f"\n{'='*80}")
print(f"🔧 Configuration:")
print(f"  Model: {args.model_name or os.getenv('MODEL_NAME', 'Unknown')}")
print(f"  Cache: {not args.no_cache}")
print(f"  Load extractor: {args.load_extractor or dotenv.get_key(dotenv.find_dotenv(), 'CACHED_EXTRACTOR_PATH') if not args.skip_load else 'No'}")
print(f"  Analyze sets: {args.analyze_sets}")
print(f"  Save model: {args.save_model}")
print(f"  Save results: {args.save_results}")
print(f"\n📁 Data Paths:")
print(f"  Images: {images_dir}")
print(f"  Ground truth: {ground_truth_dir}")
print(f"{'='*80}\n")

# Load YANASE data with ground truth
dataset = load_yanase_order_dataset(images_dir=images_dir, ground_truth_dir=ground_truth_dir)

# original splitting method
# train_set, val_set, test_set = split_order_dataset(
#     dataset,
#     train_ratio=0.5,
#     val_ratio=0.3,
#     test_ratio=0.2,
#     random_seed=42
# )

train_set, val_set, test_set = stratified_split_yanase_dataset(
    dataset,
    train_ratio=0.5,
    val_ratio=0.3,
    test_ratio=0.2,
    random_seed=42,
    stratify_by="shipPostal",  # Use expected_* prefix and shipPostal for better stratification
    verbose=True
)
print("train set: ", len(train_set))
print("val set: ", len(val_set))
print("test set: ", len(test_set))

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


# Metric wrapper for DSPy optimization
def yanase_metric_for_dspy(example, prediction, trace=None):
    """Wrapper around yanase_order_extraction_metric for DSPy"""
    # Convert prediction to dict
    pred_dict = prediction.toDict() if hasattr(prediction, "toDict") else {}
    pred_dict.pop("image", None)

    # Normalize productOrders
    if "productOrders" in pred_dict and isinstance(pred_dict["productOrders"], list):
        normalized = []
        for row in pred_dict["productOrders"]:
            if hasattr(row, "toDict"):
                normalized.append(row.toDict())
            elif isinstance(row, dict):
                normalized.append(row)
            else:
                normalized.append({
                    "receiver": getattr(row, "receiver", ""),
                    "orderCode": getattr(row, "orderCode", ""),
                    "productCode": getattr(row, "productCode", ""),
                    "BoschCode": getattr(row, "BoschCode", ""),
                    "QTY": getattr(row, "QTY", ""),
                    "unitPrice": getattr(row, "unitPrice", ""),
                    "PCDPrice": getattr(row, "PCDPrice", ""),
                })
        pred_dict["productOrders"] = normalized

    # Create ground truth dict
    gt_dict = {
        "soldName1": example.expected_soldName1,
        "soldName2": example.expected_soldName2,
        "soldTel": example.expected_soldTel,
        "soldFax": example.expected_soldFax,
        "receiverCompany": example.expected_receiverCompany,
        "shipPostal": example.expected_shipPostal,
        "shipAddress": example.expected_shipAddress,
        "shipTel": example.expected_shipTel,
        "shipFax": example.expected_shipFax,
        "productOrders": example.expected_productOrders
    }

    return yanase_order_extraction_metric(gt_dict, pred_dict, trace=trace)

# Evaluation function
evaluate_order_extraction = dspy.Evaluate(
    devset=None,
    metric=yanase_metric_for_dspy,
    num_threads=4,
    display_progress=True,
    display_table=True
)

# Start optimization with improved strategies for stratified split
custom_strategies = [
    create_bootstrap_strategy(
        name="Balanced Bootstrap",
        description="Balanced approach with equal labeled and bootstrapped demos",
        max_labeled_demos=3,
        max_bootstrapped_demos=3,
        num_candidate_programs=12,
        num_threads=4,
        max_rounds=2,
    ),
    create_bootstrap_strategy(
        name="Label-Heavy Bootstrap",
        description="Focus on labeled examples with fewer bootstrapped demos",
        max_labeled_demos=4,
        max_bootstrapped_demos=2,
        num_candidate_programs=16,
        num_threads=4,
        max_rounds=2,
    ),
    create_bootstrap_strategy(
        name="Bootstrap-Heavy",
        description="Focus on bootstrapped examples with fewer labeled demos",
        max_labeled_demos=2,
        max_bootstrapped_demos=4,
        num_candidate_programs=12,
        num_threads=4,
        max_rounds=2,
    ),
    create_miprov2_strategy(
        name="MiProV2_YANASE_Medium",
        description="MiProV2 optimizer with medium preset for YANASE order extraction",
        auto="medium",
        max_labeled_demos=3,
        max_bootstrapped_demos=3
    )
]

runner = OptimizationRunner(
    metric=yanase_metric_for_dspy,
    student_model=order_extractor,
    train_set=train_set,
    val_set=val_set,
    evaluate_fn=evaluate_order_extraction
)

# Evaluate baseline (unoptimized) extractor for comparison
print(f"\n{'='*80}")
print(f"📊 Baseline Evaluation (Before Optimization)")
print(f"{'='*80}\n")

print("Evaluating baseline extractor on validation set...")
baseline_val_result = evaluate_order_extraction(order_extractor, devset=val_set)
# Extract score from EvaluationResult object - access the score attribute or use float()
baseline_val_score = float(baseline_val_result) if hasattr(baseline_val_result, '__float__') else baseline_val_result
print(f"Baseline Val Score: {baseline_val_score:.3f} ({baseline_val_score:.1%})\n")

print("Evaluating baseline extractor on test set...")
baseline_test_result = evaluate_order_extraction(order_extractor, devset=test_set)
baseline_test_score = float(baseline_test_result) if hasattr(baseline_test_result, '__float__') else baseline_test_result
print(f"Baseline Test Score: {baseline_test_score:.3f} ({baseline_test_score:.1%})")

print(f"\n{'='*80}")
print(f"Starting Optimization...")
print(f"{'='*80}\n")
optimized_extractor, best_score = runner.run_optimization(custom_strategies)

# Calculate and display improvement
print(f"\n{'='*80}")
print(f"📈 Optimization Results Summary")
print(f"{'='*80}")
print(f"Best Strategy Val Score: {best_score:.3f} ({best_score:.1%})")
print(f"Improvement from Baseline: {(best_score - baseline_val_score):.3f} ({(best_score - baseline_val_score):.1%})")
print(f"{'='*80}\n")

# Analyze performance based on --analyze-sets argument
print(f"\n{'='*80}")
print(f"📊 Performance Analysis")
print(f"{'='*80}\n")

all_field_scores = {}
all_total_scores = []

if args.analyze_sets in ['val', 'both']:
    print(f"Analyzing validation set ({len(val_set)} examples)...")
    val_field_scores, val_total_scores = analyze_yanase_model_performance(optimized_extractor, val_set)
    print(f"\n✅ Validation set average score: {sum(val_total_scores)/len(val_total_scores):.3f}\n")
    all_field_scores['val'] = val_field_scores
    all_total_scores.extend(val_total_scores)

if args.analyze_sets in ['test', 'both']:
    print(f"Analyzing test set ({len(test_set)} examples)...")
    test_field_scores, test_total_scores = analyze_yanase_model_performance(optimized_extractor, test_set)
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
            f"{model_name}_{best_strategy_name}_yanase_optimized_{current_time}.json"
        )
        optimized_extractor.save(model_path)
        print(f"\n💾 Saved optimized model to: {model_path}")

    # Save performance results
    if args.save_results:
        results_path = os.path.join(
            cache_dir,
            f"{model_name}_{best_strategy_name}_yanase_analysis_{current_time}.json"
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
