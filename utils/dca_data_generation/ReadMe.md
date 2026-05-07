# DCA Data Generation

Tools for creating and augmenting the DCA image-pair dataset.

The DCA dataset contains 429 image pairs in total: 376 original pairs and 53 synthetically generated pairs added to address class imbalance.

---

## Folder Structure

```
utils/dca_data_generation/
├── generate_from_zips.py         # Initial dataset creation from PDF zip archives
└── augment_rare_categories.py    # Synthetic data generation for B1, B2, B4 categories
```

---

## Dataset Overview

### DCA Categories

| Code | Description | Discard? | Original Count |
|------|-------------|----------|---------------|
| A1 | Both images non-informational | true | ~68 |
| A2 | Both informational, trivial differences only | true | ~80 |
| B1 | Either image corrupted/unreadable/too tiny | false | 0 (synthetic: 20) |
| B2 | Either image crossed out | false | 2 (synthetic: 18) |
| B3 | Only one image present (informational) | false | ~53 |
| B4 | One informational + one non-informational | false | 5 (synthetic: 15) |
| B5a | Both informational, meaningful differences | false | ~104 |
| B5b | Both informational, meaningful differences (sub-type) | false | ~44 |

### Data Location

```
data/dca_data/
├── image_pairs/
│   ├── pair_001/
│   │   ├── original.png           # Image 1 (reference version)
│   │   └── reimage.png            # Image 2 (updated version)
│   ├── pair_002/ ...
│   └── pair_429/
└── image_pairs_llm_output_exp1/
    ├── pair_001.json              # Ground truth labels + LLM outputs per run
    ├── pair_002.json ...
    └── pair_429.json
```

Each `pair_NNN.json` contains:
```json
{
  "human_label_discard_image": true,
  "human_label_changes": "Images are identical - no changes",
  "human_label_category": "A2",
  "doubt": false,
  "is_synthetic": false
}
```

Additional keys are added by each inference run (human-prompt runs, optimized model runs) and are keyed by run identifier.

---

## `generate_from_zips.py` — Initial Dataset Creation

Creates the original 376 image pairs from PDF zip archives. This script was used once to generate the initial dataset and does not need to be re-run unless new source data is available.

### What it does

1. Extracts PDF zip archives from `data/dca_data/`.
2. Parses `image_matches_*.txt` files inside each archive to find original/updated image pairs.
3. Sends each pair to Azure OpenAI GPT-4.1 with a structured comparison prompt.
4. Saves image files (`original.png`, `reimage.png`) and annotation JSON per pair.

### Configuration

Edit the `ZIP_FILES` list and `COMPARISON_PROMPT` constant at the top of the script to adapt to new source data.

### Usage

```bash
# Requires Azure CLI login
az login

# Generate pairs (resumes from last existing pair_NNN folder)
python utils/dca_data_generation/generate_from_zips.py
```

### Output

- Images: `data/dca_data/image_pairs/pair_NNN/original.{ext}` + `reimage.{ext}`
- Annotations: `data/dca_data/image_pairs_llm_output_exp1/pair_NNN.json`
- Traceability: `pair_NNN/source_paths.txt` records which zip file and path produced each image.

---

## `augment_rare_categories.py` — Synthetic Data Generation

Generates synthetic image pairs for underrepresented categories to bring each to at least 20 samples. The script is idempotent — it appends after the last existing pair index and records full provenance metadata.

### Methods

| Category | Method | Description |
|----------|--------|-------------|
| B1 (corrupted/unreadable) | `corrupt_blur` | Gaussian blur, sigma ~7.5, applied to one side |
| B1 | `corrupt_pixelate` | Downsample to 100×100 (nearest-neighbor), then upscale |
| B1 | `corrupt_tiny` | Resize to 40×40 pixels |
| B2 (crossed out) | `draw_cross` | Semi-transparent diagonal X overlay, 18 predefined styles (color, thickness, alpha 45–100) |
| B4 (one info + one non-info) | recombination | Pair an informational image (from B5a/B5b pool) with a non-informational image (from A1/B3 pool) |

B2 specifically: the original image is kept intact, and the destructive transform is applied only to the reimage side — simulating a document that has been crossed out in the updated version.

### Source Pool Allocation

The script draws from two non-overlapping pools (seeded at `--seed` for reproducibility):

| Pool | Sources | Used by |
|------|---------|---------|
| Info pool (148 pairs) | B5a (104) + B5b (44) | B1 (20 pairs), B2 (18 pairs), B4 informational side (15 pairs) |
| Non-info pool (121 pairs) | A1 (68) + B3 (53) | B4 non-informational side (15 pairs) |

### Usage

```bash
# Preview without writing any files
python utils/dca_data_generation/augment_rare_categories.py --dry-run --verbose

# Generate synthetic pairs with default counts (appended after pair_376)
python utils/dca_data_generation/augment_rare_categories.py --verbose

# Custom counts
python utils/dca_data_generation/augment_rare_categories.py \
    --b1-count 20 \
    --b2-count 18 \
    --b4-count 15 \
    --seed 42 \
    --verbose
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--b1-count` | `20` | Number of B1 (corrupted) pairs to generate |
| `--b2-count` | `18` | Number of B2 (crossed-out) pairs to generate |
| `--b4-count` | `15` | Number of B4 (info+non-info) pairs to generate |
| `--seed` | `42` | Random seed for pool shuffling |
| `--dry-run` | False | Simulate without writing any files |
| `--verbose` | False | Print per-pair details |
| `--image-pairs-dir` | `data/dca_data/image_pairs` | Directory where image pairs are stored |
| `--json-dir` | `data/dca_data/image_pairs_llm_output_exp1` | Directory for annotation JSONs |

### Output Format

Each synthetic pair JSON includes full provenance:
```json
{
  "human_label_discard_image": false,
  "human_label_changes": "Could not infer",
  "human_label_category": "B1",
  "doubt": false,
  "is_synthetic": true,
  "source_pairs": ["pair_042", "pair_117"],
  "data_augmentation_explain_in_detail": "B1 pair generated by applying corrupt_blur (Gaussian sigma=7.5) to the reimage side of pair_042 (B5a). Original side sourced from pair_117 (B5b)."
}
```

### Result

After running with default settings on the original 376-pair dataset:

| Category | Before | Synthetic added | After |
|----------|--------|----------------|-------|
| B1 | 0 | 20 | 20 |
| B2 | 2 | 18 | 20 |
| B4 | 5 | 15 | 20 |
| All others | unchanged | 0 | unchanged |
| **Total** | **376** | **53** | **429** |

---

## Dataset Split

The frozen train/val/test split is saved under `.cache/splits/` after the first run of `extract_dca.py` or the tournament pipeline, using a fingerprinted filename like `.cache/splits/dca_split_seed42_<fingerprint>.json`. Some tooling may also maintain the legacy compatibility alias `.cache/dca_split_seed42.json`. It stratifies by `expected_discard_image`, `human_label_category`, and `doubt`. New pairs added after the split manifest was created are automatically assigned to the train set.

Split ratios: 60% train / 20% val / 20% test (seed=42).

Do not regenerate the split manifest unless intentional — all evaluation scripts depend on it for reproducibility.
