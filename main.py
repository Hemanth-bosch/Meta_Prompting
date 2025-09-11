from utils import load_japanese_order_dataset, split_order_dataset
from config import *
from eval import japanese_order_extraction_metric
from opt import create_custom_strategy, OptimizationRunner, create_bootstrap_strategy
from posthoc_analysis import analyze_model_performance

from datetime import datetime
import dotenv

dotenv.load_dotenv(override=True)

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
cached_extractor_path = dotenv.get_key(dotenv.find_dotenv(), "CACHED_EXTRACTOR_PATH")
if os.path.exists(cached_extractor_path) and os.path.isfile(cached_extractor_path) and cached_extractor_path.endswith(".pkl"):
    order_extractor.load(cached_extractor_path)

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
    )
]

runner = OptimizationRunner(
    metric=japanese_order_extraction_metric,
    student_model=order_extractor,
    train_set=train_set,
    val_set=val_set,
    evaluate_fn=evaluate_order_extraction
)

optimized_extractor, best_score = runner.run_optimization(custom_strategies)

field_score, total_score = analyze_model_performance(optimized_extractor, val_set)

# # save results
model_name = os.getenv("MODEL_NAME", "Unknown Model").replace("/", "_")
current_time = datetime.now().strftime("%Y%m%d%H")

# mkdir if not exists
if not os.path.exists(dotenv.get_key(dotenv.find_dotenv(), "CACHED_EXTRACTOR_DIR")):
    os.makedirs(dotenv.get_key(dotenv.find_dotenv(), "CACHED_EXTRACTOR_DIR"))
optimized_extractor.save(os.path.join(dotenv.get_key(dotenv.find_dotenv(), "CACHED_EXTRACTOR_DIR"), f"{model_name}_optimized_extractor_{current_time}.pkl"))

# with open(f"{model_name}_performance_analysis_{current_time}.json", "w") as f:
#     json.dump({"field_scores": field_score, "total_scores": total_score}, f)

