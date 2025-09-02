from eval import calculate_enhanced_similarity_score, evaluate_po_numbers, evaluate_product_details, japanese_order_extraction_metric

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