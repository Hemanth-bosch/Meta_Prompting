# DCA Data Generation

Tools for creating and augmenting the DCA image-pair dataset.

## Scripts

### `generate_from_zips.py`

Generates the original image pairs and LLM annotations from source PDF zip archives. Requires Azure CLI login and the zip files placed in `data/dca_data/`.

### `augment_rare_categories.py`

Generates synthetic image pairs for underrepresented DCA categories (B1, B2, B4) to address class imbalance. All generated pairs are tagged with full provenance metadata.

#### Methods

| Category | Corruption / Overlay | Description |
|----------|---------------------|-------------|
| **B1** (corrupted/unreadable) | `corrupt_blur` | Extreme Gaussian blur (σ ≈ 25) |
| | `corrupt_pixelate` | Downscale to 100×100 nearest-neighbor, then upscale |
| | `corrupt_tiny` | Resize to 40×40 pixels |
| **B2** (crossed-out) | `draw_cross` | Semi-transparent diagonal X overlay (1px, alpha 45–100, colors: red/black/blue) |
| **B4** (info + non-info) | recombination | Pairs a non-informational image (from A1/B3 pool) with an informational image (from B5a/B5b pool) |

#### Usage

```bash
# Preview without writing files
python utils/dca_data_generation/augment_rare_categories.py --dry-run --verbose

# Generate synthetic pairs (appends after the last existing pair)
python utils/dca_data_generation/augment_rare_categories.py --verbose

# Custom counts
python utils/dca_data_generation/augment_rare_categories.py --b1-count 20 --b2-count 18 --b4-count 15 --seed 42
```

#### Output

- Image pairs in `data/dca_data/image_pairs/pair_NNN/`
- JSON metadata in `data/dca_data/image_pairs_llm_output_exp1/pair_NNN.json`
- Each JSON includes:
  - `is_synthetic: true`
  - `source_pairs`: which original pairs were used
  - `data_augmentation_explain_in_detail`: full description of the transform applied
  - `human_label_category`: the target category (B1/B2/B4)
  - `doubt: false`

#### Source Pool Allocation

The script draws from two non-overlapping pools (seed=42 for reproducibility):
- **Info pool** (B5a + B5b = 148 pairs): B1 draws 20, B2 draws 18, B4 draws 15
- **Non-info pool** (A1 + B3 = 121 pairs): B4 draws 15 for the non-informational side
