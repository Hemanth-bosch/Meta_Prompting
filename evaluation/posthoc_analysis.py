from evaluation.common import (
    normalize_japanese_text,
    calculate_enhanced_similarity_score,
    evaluate_po_numbers,
    evaluate_product_details,
    _compare_quantities,
    _compare_prices,
)
from evaluation.jo_metric import japanese_order_extraction_metric
from evaluation.yanase_metric import yanase_order_extraction_metric


def analyze_model_performance(extractor, dataset):
    """Detailed performance analysis by field"""
    print(f"\n🔍 Analyzing Dataset Performance:")
    print("="*50)

    field_scores = {
        'soldName1': [], 'soldName2': [], 'soldAddress': [], 'soldPostal': [],
        'soldTex': [], 'soldFax': [], 'CRD': [], 'shipPostal': [], 'shipTel': [],
        'shipAddress': [], 'companyName': [], 'comments': [], 'PO': [], 'productDetail': []
    }

    total_scores = []

    for i, example in enumerate(dataset):
        try:
            prediction = extractor(image=example.image)

            # Calculate individual field scores
            string_fields = [
                ('soldName1', prediction.soldName1, example.expected_sold_name1),
                ('soldName2', prediction.soldName2, example.expected_sold_name2),
                ('soldAddress', prediction.soldAddress, example.expected_sold_address),
                ('soldPostal', prediction.soldPostal, example.expected_sold_postal),
                ('soldTex', prediction.soldTex, example.expected_sold_tel),
                ('soldFax', prediction.soldFax, example.expected_sold_fax),
                ('CRD', prediction.CRD, example.expected_order_date),
                ('shipPostal', prediction.shipPostal, example.expected_ship_postal),
                ('shipTel', prediction.shipTel, example.expected_ship_tel),
                ('shipAddress', prediction.shipAddress, example.expected_ship_address),
                ('companyName', prediction.companyName, example.expected_company_name),
                ('comments', prediction.comments, example.expected_comments),
            ]

            for field_name, pred_value, expected_value in string_fields:
                score = calculate_enhanced_similarity_score(pred_value, expected_value, field_name)
                field_scores[field_name].append(score)

            # PO and product scores
            po_score = evaluate_po_numbers(prediction.PO, example.expected_po_numbers)
            field_scores['PO'].append(po_score)

            product_score = evaluate_product_details(prediction.productDetail, example.expected_product_info)
            field_scores['productDetail'].append(product_score)

            # Total score for this example
            total_score = japanese_order_extraction_metric(example, prediction)
            total_scores.append(total_score)

        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue

    # Print field-by-field results
    print(f"Field-by-Field Performance:")
    print("-" * 30)

    for field_name, scores in field_scores.items():
        if scores:
            avg_score = sum(scores) * 100 / len(scores)
            perfect_matches = sum(1 for s in scores if s >= 0.95)
            partial_matches = sum(1 for s in scores if 0.5 <= s < 0.95)
            poor_matches = sum(1 for s in scores if s < 0.5)

            print(f"{field_name:15s}: {avg_score:5.1f}% avg | Perfect: {perfect_matches:2d} | Partial: {partial_matches:2d} | Poor: {poor_matches:2d}")

    if total_scores:
        overall_avg = sum(total_scores) / len(total_scores)
        print(f"\nOverall Average: {overall_avg * 100:.3f}%")
        print(f"Best Score: {max(total_scores) * 100:.3f}%")
        print(f"Worst Score: {min(total_scores) * 100:.3f}%")

    return field_scores, total_scores


def analyze_yanase_model_performance(extractor, dataset, verbose=False):
    """
    Detailed performance analysis for YANASE order extraction

    Args:
        extractor: The DSPy model to evaluate
        dataset: List of dspy.Example objects with YANASE schema
        verbose: If True, print per-example details

    Returns:
        tuple: (field_scores_dict, total_scores_list)
    """
    print(f"\n🔍 Analyzing YANASE Dataset Performance:")
    print("="*80)

    # Initialize score tracking (seller + receiver/shipping fields)
    seller_field_scores = {
        'soldName1': [], 'soldName2': [], 'soldTel': [], 'soldFax': [],
        'receiverCompany': [], 'shipPostal': [], 'shipAddress': [],
        'shipTel': [], 'shipFax': []
    }

    product_field_scores = {
        'receiver': [], 'orderCode': [], 'productCode': [],
        'BoschCode': [], 'QTY': [], 'unitPrice': [], 'PCDPrice': []
    }

    total_scores = []
    product_count_errors = []
    failed_examples = []

    for i, example in enumerate(dataset):
        try:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Example {i+1}/{len(dataset)}: {example.image_filename}")

            prediction = extractor(image=example.image)
            pred_dict = _convert_yanase_prediction_to_dict(prediction)

            gt_dict = {
                'soldName1': example.expected_soldName1,
                'soldName2': example.expected_soldName2,
                'soldTel': example.expected_soldTel,
                'soldFax': example.expected_soldFax,
                'receiverCompany': example.expected_receiverCompany,
                'shipPostal': example.expected_shipPostal,
                'shipAddress': example.expected_shipAddress,
                'shipTel': example.expected_shipTel,
                'shipFax': example.expected_shipFax,
                'productOrders': example.expected_productOrders
            }

            details = yanase_order_extraction_metric(gt_dict, pred_dict, return_details=True)
            total_score = details['total_score']
            total_scores.append(total_score)

            for field_name in seller_field_scores.keys():
                if field_name in details['field_info']:
                    similarity = details['field_info'][field_name]['similarity']
                    seller_field_scores[field_name].append(similarity)

            pred_products = pred_dict.get('productOrders', [])
            gt_products = gt_dict.get('productOrders', [])
            product_count_errors.append(abs(len(pred_products) - len(gt_products)))
            _analyze_product_fields(pred_products, gt_products, product_field_scores)

            if verbose:
                print(f"  Score: {total_score:.3f}")
                print(f"  Products: GT={len(gt_products)}, Pred={len(pred_products)}")

        except Exception as e:
            print(f"❌ Error processing example {i} ({example.image_filename}): {e}")
            failed_examples.append((i, example.image_filename, str(e)))
            continue

    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 YANASE Performance Summary")
    print(f"{'='*80}")

    if total_scores:
        print(f"\n🎯 Overall Scores:")
        print(f"  Mean:   {sum(total_scores)/len(total_scores):.3f} ({sum(total_scores)/len(total_scores)*100:.1f}%)")
        print(f"  Median: {sorted(total_scores)[len(total_scores)//2]:.3f}")
        print(f"  Min:    {min(total_scores):.3f}")
        print(f"  Max:    {max(total_scores):.3f}")
        print(f"  Std:    {_calculate_std(total_scores):.3f}")

    print(f"\n🏢 Seller Information Fields:")
    print(f"  {'Field':<15} {'Avg':<8} {'Perfect':<8} {'Good':<8} {'Poor':<8}")
    print(f"  {'-'*50}")
    for field_name, scores in seller_field_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            perfect = sum(1 for s in scores if s >= 0.95)
            good = sum(1 for s in scores if 0.7 <= s < 0.95)
            poor = sum(1 for s in scores if s < 0.7)
            print(f"  {field_name:<15} {avg:<8.3f} {perfect:<8d} {good:<8d} {poor:<8d}")

    print(f"\n📦 Product Order Fields:")
    print(f"  {'Field':<15} {'Avg':<8} {'Perfect':<8} {'Good':<8} {'Poor':<8}")
    print(f"  {'-'*50}")
    for field_name, scores in product_field_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            perfect = sum(1 for s in scores if s >= 0.95)
            good = sum(1 for s in scores if 0.7 <= s < 0.95)
            poor = sum(1 for s in scores if s < 0.7)
            print(f"  {field_name:<15} {avg:<8.3f} {perfect:<8d} {good:<8d} {poor:<8d}")

    if product_count_errors:
        perfect_counts = sum(1 for e in product_count_errors if e == 0)
        avg_error = sum(product_count_errors) / len(product_count_errors)
        print(f"\n📊 Product Count Accuracy:")
        print(f"  Perfect count matches: {perfect_counts}/{len(product_count_errors)}")
        print(f"  Average count error: {avg_error:.2f} products")

    if failed_examples:
        print(f"\n❌ Failed Examples: {len(failed_examples)}")
        for idx, filename, error in failed_examples:
            print(f"  - Example {idx}: {filename} - {error}")

    print(f"\n{'='*80}\n")

    all_field_scores = {**seller_field_scores, **product_field_scores}
    return all_field_scores, total_scores


def _convert_yanase_prediction_to_dict(prediction):
    """Convert DSPy prediction to dict for YANASE evaluation"""
    pred_dict = prediction.toDict() if hasattr(prediction, "toDict") else {}
    pred_dict.pop("image", None)

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

    return pred_dict


def _analyze_product_fields(pred_products, gt_products, field_scores):
    """Analyze individual product fields"""
    if not gt_products:
        return

    field_names = ['receiver', 'orderCode', 'productCode', 'BoschCode', 'QTY', 'unitPrice', 'PCDPrice']

    for gt_product in gt_products:
        best_match_scores = {field: 0.0 for field in field_names}

        for pred_product in pred_products:
            for field_name in field_names:
                gt_value = str(gt_product.get(field_name, '')).strip()
                pred_value = str(pred_product.get(field_name, '')).strip()

                if field_name in ['receiver', 'orderCode']:
                    similarity = 1.0 if normalize_japanese_text(pred_value) == normalize_japanese_text(gt_value) else 0.0
                elif field_name == 'QTY':
                    similarity = _compare_quantities(pred_value, gt_value)
                elif field_name in ['unitPrice', 'PCDPrice']:
                    similarity = _compare_prices(pred_value, gt_value)
                else:
                    similarity = calculate_enhanced_similarity_score(pred_value, gt_value, field_name)

                best_match_scores[field_name] = max(best_match_scores[field_name], similarity)

        for field_name, score in best_match_scores.items():
            field_scores[field_name].append(score)


def _calculate_std(values):
    """Calculate standard deviation"""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def analyze_val_test_gap(val_field_scores: dict,
                          test_field_scores: dict,
                          gap_threshold: float = 0.20,
                          verbose: bool = True):
    """
    Analyze the performance gap between validation and test sets.

    Args:
        val_field_scores: Dictionary of field_name -> list of scores for validation set
        test_field_scores: Dictionary of field_name -> list of scores for test set
        gap_threshold: Threshold for flagging problematic gaps (default: 0.20 = 20%)
        verbose: Print detailed analysis (default: True)

    Returns:
        Dictionary with gap analysis
    """
    if not val_field_scores or not test_field_scores:
        print("⚠️  Cannot compute gap: missing val or test scores")
        return None

    field_gaps = {}
    all_field_names = set(val_field_scores.keys()) | set(test_field_scores.keys())

    for field_name in all_field_names:
        val_scores = val_field_scores.get(field_name, [])
        test_scores = test_field_scores.get(field_name, [])

        if val_scores and test_scores:
            val_avg = sum(val_scores) / len(val_scores)
            test_avg = sum(test_scores) / len(test_scores)
            gap = val_avg - test_avg
            field_gaps[field_name] = gap
        else:
            field_gaps[field_name] = None

    flagged_fields = []
    for field_name, gap in field_gaps.items():
        if gap is not None and abs(gap) > gap_threshold:
            flagged_fields.append((field_name, gap))

    flagged_fields.sort(key=lambda x: abs(x[1]), reverse=True)

    valid_gaps = [gap for gap in field_gaps.values() if gap is not None]
    avg_gap = sum(valid_gaps) / len(valid_gaps) if valid_gaps else 0.0
    max_gap = max(valid_gaps, key=abs) if valid_gaps else 0.0

    if verbose:
        print(f"\n{'='*80}")
        print(f"📊 Validation-Test Gap Analysis")
        print(f"{'='*80}")
        print(f"Gap Threshold: {gap_threshold*100:.0f}%")
        print(f"Average Gap: {avg_gap:.3f} ({avg_gap*100:+.1f}%)")
        print(f"Max Gap: {max_gap:.3f} ({max_gap*100:+.1f}%)")

        if flagged_fields:
            print(f"\n⚠️  FLAGGED FIELDS (Gap > {gap_threshold*100:.0f}%):")
            print(f"{'='*80}")
            print(f"{'Field':<20} {'Val Avg':<10} {'Test Avg':<10} {'Gap':<10} {'Status'}")
            print(f"{'-'*80}")

            for field_name, gap in flagged_fields:
                val_avg = sum(val_field_scores.get(field_name, [0])) / len(val_field_scores.get(field_name, [1]))
                test_avg = sum(test_field_scores.get(field_name, [0])) / len(test_field_scores.get(field_name, [1]))
                status = "Overfitting" if gap > 0 else "Underfitting"
                print(f"{field_name:<20} {val_avg:<10.3f} {test_avg:<10.3f} {gap:+.3f} ({gap*100:+.1f}%)  {status}")

            print(f"\n{'='*80}")
            print(f"📋 Interpretation:")
            print(f"  - Positive gap (Val > Test): Model may be overfitting to val set patterns")
            print(f"  - Negative gap (Test > Val): Model generalizes better to test, or test is easier")
            print(f"  - Large gaps indicate distribution shift or data leakage issues")
            print(f"{'='*80}\n")
        else:
            print(f"\n✅ No fields exceed {gap_threshold*100:.0f}% gap threshold")
            print(f"   Model shows good generalization across val and test sets\n")

        print(f"\n📋 Field-by-Field Gap Summary:")
        print(f"{'Field':<20} {'Val Avg':<10} {'Test Avg':<10} {'Gap':<10}")
        print(f"{'-'*60}")

        sorted_fields = sorted(field_gaps.items(), key=lambda x: abs(x[1]) if x[1] is not None else 0, reverse=True)
        for field_name, gap in sorted_fields:
            if gap is not None:
                val_avg = sum(val_field_scores.get(field_name, [0])) / len(val_field_scores.get(field_name, [1]))
                test_avg = sum(test_field_scores.get(field_name, [0])) / len(test_field_scores.get(field_name, [1]))
                print(f"{field_name:<20} {val_avg:<10.3f} {test_avg:<10.3f} {gap:+.3f} ({gap*100:+.1f}%)")

        print(f"\n{'='*80}\n")

    return {
        'field_gaps': field_gaps,
        'flagged_fields': [f[0] for f in flagged_fields],
        'avg_gap': avg_gap,
        'max_gap': max_gap
    }


def analyze_dca_performance(extractor, dataset, verbose=True):
    """Analyze binary classification performance for DCA image comparison.

    positive class = discard_image=true (discard / images are identical/trivial)
    negative class = discard_image=false (keep / meaningful changes exist)

    Computes confusion matrix, accuracy, precision, recall, F1.

    Args:
        extractor: Compiled ChainOfThought(DCAImageComparison) module
        dataset: List of dspy.Example with expected_discard_image
        verbose: Print detailed output

    Returns:
        dict with accuracy, precision, recall, f1, confusion matrix, misclassified
    """
    results = []
    true_values = {"true", "yes", "1"}

    for example in dataset:
        try:
            prediction = extractor(
                image_original=example.image_original,
                image_reimage=example.image_reimage,
            )
            pred_raw = str(getattr(prediction, "discard_image", "")).strip().lower()
            pred_bool = pred_raw in true_values
            pred_label = "discard" if pred_bool else "keep"
        except Exception as e:
            pair = getattr(example, "pair_name", "?")
            print(f"  Error on {pair}: {e}")
            pred_label = "error"

        expected_raw = str(example.expected_discard_image).strip().lower()
        expected_bool = expected_raw in true_values
        expected_label = "discard" if expected_bool else "keep"

        results.append({
            "pair_name": getattr(example, "pair_name", "unknown"),
            "expected": expected_label,
            "predicted": pred_label,
            "correct": pred_label == expected_label,
        })

    # Confusion matrix (positive class = discard)
    tp = sum(1 for r in results if r["expected"] == "discard" and r["predicted"] == "discard")
    tn = sum(1 for r in results if r["expected"] == "keep" and r["predicted"] == "keep")
    fp = sum(1 for r in results if r["expected"] == "keep" and r["predicted"] == "discard")
    fn = sum(1 for r in results if r["expected"] == "discard" and r["predicted"] == "keep")
    errors = sum(1 for r in results if r["predicted"] == "error")

    total = len(results)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    if verbose:
        print(f"\n{'='*55}")
        print(f"DCA IMAGE COMPARISON ANALYSIS ({total} examples)")
        print(f"{'='*55}")

        print(f"\nConfusion Matrix (positive = 'discard'):")
        print(f"  +-------------+--------------+--------------+")
        print(f"  |             | Pred Keep    | Pred Discard |")
        print(f"  +-------------+--------------+--------------+")
        print(f"  | GT Keep     | TN = {tn:<6}  | FP = {fp:<6}  |")
        print(f"  | GT Discard  | FN = {fn:<6}  | TP = {tp:<6}  |")
        print(f"  +-------------+--------------+--------------+")
        if errors > 0:
            print(f"  Errors (invalid output): {errors}")

        print(f"\nMetrics:")
        print(f"  Accuracy:  {accuracy:.1%}")
        print(f"  Precision: {precision:.1%} (of predicted 'discard', how many correct)")
        print(f"  Recall:    {recall:.1%} (of actual 'discard', how many found)")
        print(f"  F1 Score:  {f1:.1%}")

        misclassified = [r for r in results if not r["correct"]]
        if misclassified:
            print(f"\nMisclassified ({len(misclassified)}):")
            for r in misclassified[:20]:
                print(f"  {r['pair_name']}: expected={r['expected']}, predicted={r['predicted']}")
            if len(misclassified) > 20:
                print(f"  ... and {len(misclassified) - 20} more")
        else:
            print(f"\nAll examples classified correctly!")

        discard_total = sum(1 for r in results if r["expected"] == "discard")
        keep_total = sum(1 for r in results if r["expected"] == "keep")

        print(f"\nPer-class accuracy:")
        if keep_total > 0:
            print(f"  Keep (discard=false):    {tn}/{keep_total} ({tn/keep_total:.1%})")
        if discard_total > 0:
            print(f"  Discard (discard=true):  {tp}/{discard_total} ({tp/discard_total:.1%})")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "errors": errors,
        "misclassified": [r for r in results if not r["correct"]],
    }
