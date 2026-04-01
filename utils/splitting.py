import dspy
from typing import List
import random


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

def stratified_split_yanase_dataset(dataset: List[dspy.Example],
                                     train_ratio=0.6,
                                     val_ratio=0.2,
                                     test_ratio=0.2,
                                     max_examples_per_seller=None,
                                     random_seed=42,
                                     verbose=True,
                                     stratify_by=None):
    """
    Split YANASE dataset with optional stratification by specified field(s).

    Args:
        dataset: List of dspy.Example objects
        train_ratio: Proportion for training (default: 0.6)
        val_ratio: Proportion for validation (default: 0.2)
        test_ratio: Proportion for testing (default: 0.2)
        max_examples_per_seller: Maximum examples per group when stratifying.
        random_seed: Random seed for reproducibility (default: 42)
        verbose: Print detailed split statistics (default: True)
        stratify_by: Field name(s) to stratify by.

    Returns:
        Tuple of (train_set, val_set, test_set)
    """
    from collections import defaultdict

    random.seed(random_seed)

    # NO STRATIFICATION: Simple random split
    if stratify_by is None:
        shuffled_dataset = dataset.copy()
        random.shuffle(shuffled_dataset)

        total = len(shuffled_dataset)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        train_set = shuffled_dataset[:train_size]
        val_set = shuffled_dataset[train_size:train_size + val_size]
        test_set = shuffled_dataset[train_size + val_size:]

        if verbose:
            print(f"\n{'='*70}")
            print(f"📊 Random Split (Seed: {random_seed})")
            print(f"{'='*70}")
            print(f"Total: {total} examples")
            print(f"\n{'Split':<10} {'Examples':<12} {'Percentage':<12}")
            print(f"{'-'*40}")
            print(f"{'Train':<10} {len(train_set):<12} {len(train_set)/total:.1%}")
            print(f"{'Val':<10} {len(val_set):<12} {len(val_set)/total:.1%}")
            print(f"{'Test':<10} {len(test_set):<12} {len(test_set)/total:.1%}")
            print(f"{'='*70}\n")

        return train_set, val_set, test_set

    # STRATIFIED SPLIT: Group by specified field(s)
    if isinstance(stratify_by, str):
        stratify_fields = [stratify_by]
    else:
        stratify_fields = list(stratify_by)

    # Auto-detect and fix field names (add "expected_" prefix if needed)
    corrected_fields = []
    for field in stratify_fields:
        test_example = dataset[0] if dataset else None
        if test_example and hasattr(test_example, field):
            corrected_fields.append(field)
        elif test_example and not field.startswith('expected_') and hasattr(test_example, f'expected_{field}'):
            corrected_field = f'expected_{field}'
            corrected_fields.append(corrected_field)
            if verbose:
                print(f"ℹ️  Auto-corrected field name: '{field}' → '{corrected_field}'")
        else:
            corrected_fields.append(field)
            if verbose:
                print(f"⚠️  Field '{field}' not found in dataset, using empty default")

    stratify_fields = corrected_fields

    # Group examples by stratification key
    groups = defaultdict(list)
    for example in dataset:
        key_values = []
        for field in stratify_fields:
            value = getattr(example, field, '')
            key_values.append(str(value).strip() if hasattr(value, 'strip') else str(value))
        key = tuple(key_values)
        groups[key].append(example)

    # Apply downsampling if max_examples_per_seller is specified
    original_total = sum(len(examples) for examples in groups.values())

    if max_examples_per_seller is not None:
        balanced_groups = {}
        downsampled_groups = []

        for group_key, examples in groups.items():
            if len(examples) > max_examples_per_seller:
                balanced_groups[group_key] = random.sample(examples, max_examples_per_seller)
                downsampled_groups.append((group_key, len(examples), max_examples_per_seller))
            else:
                balanced_groups[group_key] = examples

        groups = balanced_groups

        if verbose and downsampled_groups:
            downsampled_total = sum(len(examples) for examples in groups.values())
            print(f"\n⚖️  Applied downsampling (max {max_examples_per_seller} examples per group)")
            print(f"   Total examples: {original_total} → {downsampled_total} ({original_total - downsampled_total} removed)")
            for group_key, original, kept in downsampled_groups:
                key_str = ' | '.join(group_key)
                print(f"     - {key_str}: {original} → {kept} examples")

    unique_groups = list(groups.keys())
    total_groups = len(unique_groups)

    if verbose and total_groups < 3:
        print(f"\n⚠️  WARNING: Only {total_groups} unique group(s) in dataset!")
        print(f"   Stratification may not be effective with so few groups.\n")

    train_set = []
    val_set = []
    test_set = []

    large_groups = [k for k in unique_groups if len(groups[k]) >= 3]
    small_groups = [k for k in unique_groups if len(groups[k]) < 3]

    for group_key in large_groups:
        examples = groups[group_key].copy()
        random.shuffle(examples)
        n = len(examples)
        train_count = max(1, int(n * train_ratio))
        val_count = max(1, int(n * val_ratio))
        train_set.extend(examples[:train_count])
        val_set.extend(examples[train_count:train_count + val_count])
        test_set.extend(examples[train_count + val_count:])

    random.shuffle(small_groups)

    total_examples = len(dataset)
    target_train = int(total_examples * train_ratio)
    target_val = int(total_examples * val_ratio)
    target_test = total_examples - target_train - target_val

    for group_key in small_groups:
        examples = groups[group_key]
        train_deficit = target_train - len(train_set)
        val_deficit = target_val - len(val_set)
        test_deficit = target_test - len(test_set)

        if train_deficit >= val_deficit and train_deficit >= test_deficit:
            train_set.extend(examples)
        elif val_deficit >= test_deficit:
            val_set.extend(examples)
        else:
            test_set.extend(examples)

    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    if verbose:
        total_examples = len(dataset)
        fields_str = ', '.join(stratify_fields)
        print(f"\n{'='*70}")
        print(f"📊 Stratified Split Summary (Seed: {random_seed})")
        print(f"{'='*70}")
        print(f"Stratify by: {fields_str}")
        print(f"Strategy: Split large groups (≥3 examples) proportionally")
        print(f"Total: {total_examples} examples, {total_groups} unique groups")
        print(f"  - Large groups (split proportionally): {len(large_groups)}")
        print(f"  - Small groups (kept intact): {len(small_groups)}")

        print(f"\n{'Split':<10} {'Examples':<12} {'Target':<12} {'Actual %':<12}")
        print(f"{'-'*50}")
        print(f"{'Train':<10} {len(train_set):<12} {int(total_examples*train_ratio):<12} {len(train_set)/total_examples:.1%}")
        print(f"{'Val':<10} {len(val_set):<12} {int(total_examples*val_ratio):<12} {len(val_set)/total_examples:.1%}")
        print(f"{'Test':<10} {len(test_set):<12} {total_examples - int(total_examples*train_ratio) - int(total_examples*val_ratio):<12} {len(test_set)/total_examples:.1%}")

        from collections import defaultdict as _dd
        group_distribution = _dd(lambda: {'train': 0, 'val': 0, 'test': 0})

        for example in train_set:
            key_values = tuple(str(getattr(example, f, '')).strip() for f in stratify_fields)
            group_distribution[key_values]['train'] += 1
        for example in val_set:
            key_values = tuple(str(getattr(example, f, '')).strip() for f in stratify_fields)
            group_distribution[key_values]['val'] += 1
        for example in test_set:
            key_values = tuple(str(getattr(example, f, '')).strip() for f in stratify_fields)
            group_distribution[key_values]['test'] += 1

        print(f"\n{'Group Distribution Details':<50}")
        print(f"{'-'*70}")
        print(f"{'Group':<30} {'Total':<8} {'Train':<8} {'Val':<8} {'Test':<8}")
        print(f"{'-'*70}")

        sorted_groups = sorted(group_distribution.items(), key=lambda x: sum(x[1].values()), reverse=True)
        for group_key, counts in sorted_groups:
            group_str = ' | '.join(group_key) if len(group_key) > 1 else group_key[0]
            group_str = group_str[:28] + '..' if len(group_str) > 30 else group_str
            total = counts['train'] + counts['val'] + counts['test']
            split_marker = ' *' if sum([counts['train'] > 0, counts['val'] > 0, counts['test'] > 0]) > 1 else ''
            print(f"{group_str:<30} {total:<8} {counts['train']:<8} {counts['val']:<8} {counts['test']:<8}{split_marker}")

        train_groups = {k for k, v in group_distribution.items() if v['train'] > 0}
        val_groups = {k for k, v in group_distribution.items() if v['val'] > 0}
        test_groups = {k for k, v in group_distribution.items() if v['test'] > 0}
        shared_groups = [k for k, v in group_distribution.items()
                        if sum([v['train'] > 0, v['val'] > 0, v['test'] > 0]) > 1]

        print(f"\n{'Stratification Statistics':<50}")
        print(f"{'-'*70}")
        print(f"Groups unique to train:  {len(train_groups - val_groups - test_groups)}")
        print(f"Groups unique to val:    {len(val_groups - train_groups - test_groups)}")
        print(f"Groups unique to test:   {len(test_groups - train_groups - val_groups)}")
        print(f"Groups split across sets: {len(shared_groups)}")

        if shared_groups:
            print(f"\n  Split groups (* in table above):")
            for group_key in shared_groups[:5]:
                group_str = ' | '.join(group_key) if len(group_key) > 1 else group_key[0]
                counts = group_distribution[group_key]
                splits = []
                if counts['train'] > 0: splits.append(f"train={counts['train']}")
                if counts['val'] > 0: splits.append(f"val={counts['val']}")
                if counts['test'] > 0: splits.append(f"test={counts['test']}")
                print(f"    - {group_str}: {', '.join(splits)}")

        print(f"\n{'Stratification Quality':<50}")
        print(f"{'-'*70}")
        train_diversity = len(train_groups) / total_groups
        val_diversity = len(val_groups) / total_groups
        test_diversity = len(test_groups) / total_groups
        print(f"Group diversity in train: {train_diversity:.1%} ({len(train_groups)}/{total_groups} groups)")
        print(f"Group diversity in val:   {val_diversity:.1%} ({len(val_groups)}/{total_groups} groups)")
        print(f"Group diversity in test:  {test_diversity:.1%} ({len(test_groups)}/{total_groups} groups)")
        print(f"\n{'='*70}\n")

    return train_set, val_set, test_set


def stratified_split_dca_dataset(dataset: List[dspy.Example],
                                  train_ratio=0.6,
                                  val_ratio=0.2,
                                  test_ratio=0.2,
                                  random_seed=42,
                                  verbose=True,
                                  stratify_by=None):
    """
    Split DCA dataset with optional stratification by specified field(s).

    Defaults to stratifying by ["expected_discard_image", "expected_human_label_category", "doubt"]
    to preserve class balance across the binary classification task and label categories.

    Args:
        dataset: List of dspy.Example objects (DCA image pairs)
        train_ratio: Proportion for training (default: 0.6)
        val_ratio: Proportion for validation (default: 0.2)
        test_ratio: Proportion for testing (default: 0.2)
        random_seed: Random seed for reproducibility (default: 42)
        verbose: Print detailed split statistics (default: True)
        stratify_by: Field name(s) to stratify by.
                     Defaults to ["expected_discard_image", "expected_human_label_category", "doubt"].

    Returns:
        Tuple of (train_set, val_set, test_set)
    """
    from collections import defaultdict

    if stratify_by is None:
        stratify_by = ["expected_discard_image", "expected_human_label_category", "doubt"]

    random.seed(random_seed)

    if isinstance(stratify_by, str):
        stratify_fields = [stratify_by]
    else:
        stratify_fields = list(stratify_by)

    # NO STRATIFICATION: Simple random split
    if len(stratify_fields) == 0:
        shuffled_dataset = dataset.copy()
        random.shuffle(shuffled_dataset)

        total = len(shuffled_dataset)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        train_set = shuffled_dataset[:train_size]
        val_set = shuffled_dataset[train_size:train_size + val_size]
        test_set = shuffled_dataset[train_size + val_size:]

        if verbose:
            print(f"\n{'='*70}")
            print(f"📊 Random Split — DCA Image Pairs (Seed: {random_seed})")
            print(f"{'='*70}")
            print(f"Total: {total} image pairs")
            print(f"\n{'Split':<10} {'Pairs':<12} {'Percentage':<12}")
            print(f"{'-'*40}")
            print(f"{'Train':<10} {len(train_set):<12} {len(train_set)/total:.1%}")
            print(f"{'Val':<10} {len(val_set):<12} {len(val_set)/total:.1%}")
            print(f"{'Test':<10} {len(test_set):<12} {len(test_set)/total:.1%}")
            print(f"{'='*70}\n")

        return train_set, val_set, test_set

    # STRATIFIED SPLIT: group by stratification key
    groups = defaultdict(list)
    for example in dataset:
        key_values = []
        for field in stratify_fields:
            value = getattr(example, field, '')
            key_values.append(str(value).strip() if hasattr(value, 'strip') else str(value))
        groups[tuple(key_values)].append(example)

    unique_groups = list(groups.keys())
    total_groups = len(unique_groups)

    if verbose and total_groups < 3:
        print(f"\n⚠️  WARNING: Only {total_groups} unique group(s) in dataset!")
        print(f"   Stratification may not be effective with so few groups.\n")

    train_set: List[dspy.Example] = []
    val_set: List[dspy.Example] = []
    test_set: List[dspy.Example] = []

    large_groups = [k for k in unique_groups if len(groups[k]) >= 3]
    small_groups = [k for k in unique_groups if len(groups[k]) < 3]

    for group_key in large_groups:
        examples = groups[group_key].copy()
        random.shuffle(examples)
        n = len(examples)
        train_count = max(1, int(n * train_ratio))
        val_count = max(1, int(n * val_ratio))
        train_set.extend(examples[:train_count])
        val_set.extend(examples[train_count:train_count + val_count])
        test_set.extend(examples[train_count + val_count:])

    random.shuffle(small_groups)

    total_examples = len(dataset)
    target_train = int(total_examples * train_ratio)
    target_val = int(total_examples * val_ratio)
    target_test = total_examples - target_train - target_val

    for group_key in small_groups:
        examples = groups[group_key]
        train_deficit = target_train - len(train_set)
        val_deficit = target_val - len(val_set)
        test_deficit = target_test - len(test_set)

        if train_deficit >= val_deficit and train_deficit >= test_deficit:
            train_set.extend(examples)
        elif val_deficit >= test_deficit:
            val_set.extend(examples)
        else:
            test_set.extend(examples)

    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    if verbose:
        fields_str = ', '.join(stratify_fields)
        print(f"\n{'='*70}")
        print(f"📊 Stratified Split Summary — DCA Image Pairs (Seed: {random_seed})")
        print(f"{'='*70}")
        print(f"Stratify by: {fields_str}")
        print(f"Strategy: Split large groups (≥3 pairs) proportionally")
        print(f"Total: {total_examples} image pairs, {total_groups} unique groups")
        print(f"  - Large groups (split proportionally): {len(large_groups)}")
        print(f"  - Small groups (kept intact): {len(small_groups)}")

        print(f"\n{'Split':<10} {'Pairs':<12} {'Target':<12} {'Actual %':<12}")
        print(f"{'-'*50}")
        print(f"{'Train':<10} {len(train_set):<12} {target_train:<12} {len(train_set)/total_examples:.1%}")
        print(f"{'Val':<10} {len(val_set):<12} {target_val:<12} {len(val_set)/total_examples:.1%}")
        print(f"{'Test':<10} {len(test_set):<12} {target_test:<12} {len(test_set)/total_examples:.1%}")

        from collections import defaultdict as _dd
        group_distribution = _dd(lambda: {'train': 0, 'val': 0, 'test': 0})
        for example in train_set:
            key_values = tuple(str(getattr(example, f, '')).strip() for f in stratify_fields)
            group_distribution[key_values]['train'] += 1
        for example in val_set:
            key_values = tuple(str(getattr(example, f, '')).strip() for f in stratify_fields)
            group_distribution[key_values]['val'] += 1
        for example in test_set:
            key_values = tuple(str(getattr(example, f, '')).strip() for f in stratify_fields)
            group_distribution[key_values]['test'] += 1

        print(f"\n{'Group Distribution Details':<50}")
        print(f"{'-'*70}")
        print(f"{'Group':<30} {'Total':<8} {'Train':<8} {'Val':<8} {'Test':<8}")
        print(f"{'-'*70}")

        sorted_groups = sorted(group_distribution.items(), key=lambda x: sum(x[1].values()), reverse=True)
        for group_key, counts in sorted_groups:
            group_str = ' | '.join(group_key) if len(group_key) > 1 else group_key[0]
            group_str = group_str[:28] + '..' if len(group_str) > 30 else group_str
            total = counts['train'] + counts['val'] + counts['test']
            split_marker = ' *' if sum([counts['train'] > 0, counts['val'] > 0, counts['test'] > 0]) > 1 else ''
            print(f"{group_str:<30} {total:<8} {counts['train']:<8} {counts['val']:<8} {counts['test']:<8}{split_marker}")

        train_groups = {k for k, v in group_distribution.items() if v['train'] > 0}
        val_groups = {k for k, v in group_distribution.items() if v['val'] > 0}
        test_groups = {k for k, v in group_distribution.items() if v['test'] > 0}
        shared_groups = [k for k, v in group_distribution.items()
                         if sum([v['train'] > 0, v['val'] > 0, v['test'] > 0]) > 1]

        print(f"\n{'Stratification Statistics':<50}")
        print(f"{'-'*70}")
        print(f"Groups unique to train:  {len(train_groups - val_groups - test_groups)}")
        print(f"Groups unique to val:    {len(val_groups - train_groups - test_groups)}")
        print(f"Groups unique to test:   {len(test_groups - train_groups - val_groups)}")
        print(f"Groups split across sets: {len(shared_groups)}")

        if shared_groups:
            print(f"\n  Split groups (* in table above):")
            for group_key in shared_groups[:5]:
                group_str = ' | '.join(group_key) if len(group_key) > 1 else group_key[0]
                counts = group_distribution[group_key]
                splits = []
                if counts['train'] > 0: splits.append(f"train={counts['train']}")
                if counts['val'] > 0: splits.append(f"val={counts['val']}")
                if counts['test'] > 0: splits.append(f"test={counts['test']}")
                print(f"    - {group_str}: {', '.join(splits)}")

        print(f"\n{'Stratification Quality':<50}")
        print(f"{'-'*70}")
        train_diversity = len(train_groups) / total_groups
        val_diversity = len(val_groups) / total_groups
        test_diversity = len(test_groups) / total_groups
        print(f"Group diversity in train: {train_diversity:.1%} ({len(train_groups)}/{total_groups} groups)")
        print(f"Group diversity in val:   {val_diversity:.1%} ({len(val_groups)}/{total_groups} groups)")
        print(f"Group diversity in test:  {test_diversity:.1%} ({len(test_groups)}/{total_groups} groups)")
        print(f"\n{'='*70}\n")

    return train_set, val_set, test_set
