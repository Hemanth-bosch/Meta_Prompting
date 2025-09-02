import dspy
from typing import List, Dict, Any
import json
import os
import random

def load_japanese_order_dataset(data_dir: str) -> List[dspy.Example]:

    images_dir = os.path.join(data_dir, "png")
    json_dir = os.path.join(data_dir, "json")
    
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    examples = []
    
    for json_file in sorted(json_files):
        base_name = json_file.replace('.json', '')
        
        
        image_path = os.path.join(images_dir, f"{base_name}.png")
        json_path = os.path.join(json_dir, json_file)
        
        try:
            # load data
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            doc = data['documents'][0]
            
            
            # Analyze order details
            po_numbers = []
            product_info = []
            if 'orderDetail' in doc:
                for order in doc['orderDetail']:
                    if 'PO' in order:
                        po_numbers.append(order['PO'])
                    if 'productDetail' in order:
                        for product in order['productDetail']:
                            product_info.append({
                                'productNumber': product.get('productNumber', ''),
                                'QTY': product.get('QTY', '')
                            })
            
            # load image
            dspy_image = dspy.Image.from_file(image_path)
            
            # Create Example
            example = dspy.Example(
                image=dspy_image,
                image_filename=os.path.basename(image_path),
                expected_sold_name1=doc.get("soldName1", ""),
                expected_sold_name2=doc.get("soldName2", ""),
                expected_sold_postal=doc.get("soldPostal", ""),
                expected_sold_address=doc.get("soldAddress", ""),
                expected_sold_tel=doc.get("soldTex", ""),
                expected_sold_fax=doc.get("soldFax", ""),
                expected_order_date=doc.get("CRD", ""),
                expected_ship_postal=doc.get("shipPostal", ""),
                expected_ship_tel=doc.get("shipTEL", ""),
                expected_ship_address=doc.get("shipAddress", ""),
                expected_company_name=doc.get("companyName", ""),
                expected_comments=doc.get("comments", ""),
                expected_po_numbers=po_numbers,
                expected_product_info=product_info
            ).with_inputs("image")
            
            examples.append(example)
            
        except Exception as e:
            print(f"❌ load error {json_file}: {e}")
    
    print(f"✅ load success: {len(examples)} ")
    return examples


def split_order_dataset(dataset, train_ratio=0.6, val_ratio=0.3, test_ratio=0.1, random_seed=42):
    """Divided order dataset"""
    random.seed(random_seed)
    shuffled_dataset = dataset.copy()
    random.shuffle(shuffled_dataset)
    
    total = len(shuffled_dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_set = shuffled_dataset[:train_size]
    val_set = shuffled_dataset[train_size:train_size + val_size]
    test_set = shuffled_dataset[train_size + val_size:]
    
    print(f" (total: {total}):")
    print(f"  train set: {len(train_set)} samples ({len(train_set)/total:.1%})")
    print(f"  val set: {len(val_set)} samples ({len(val_set)/total:.1%})")
    print(f"  test set: {len(test_set)} samples ({len(test_set)/total:.1%})")
    
    return train_set, val_set, test_set


if __name__ == "__main__":
    dataset_path = "./data"
    dataset = load_japanese_order_dataset(dataset_path)
    print("dataset loaded successfully")
    print("# of data points: ", len(dataset))
    print("data instance 0: \n", dataset[0])