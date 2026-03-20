# Data Directory

Place datasets here locally. 
- **Yanase**: YANASE/, YANASE_images/, YANASE_ground_truth/
- **Japanese Order**: png/, json/
- **DCA**: dca_data/ (image_pairs/, image_pairs_llm_output_exp1/)

```
data/
├── YANASE/                  # Raw order PDFs
├── YANASE_images/           # Rendered page PNGs (from PDFs)
├── YANASE_ground_truth/     # Curated ground truth JSONs
├── dca_data/
│   ├── image_pairs/         # Pair folders (pair_001, pair_002, ...) with original/reimage images
│   └── image_pairs_llm_output_exp1/  # Pair-level JSON outputs used in baseline experiments
├── png/                     # Japanese order images
└── json/                    # Japanese order ground truth
```

## DCA note

DCA data is currently stored here for active experimentation and baseline runs. This location is temporary and is expected to be removed or reorganized later once the workflow is stabilized.

## Loading

```python
from utils import load_yanase_order_dataset, load_japanese_order_dataset

# YANASE — reads from YANASE_images/ + YANASE_ground_truth/
dataset = load_yanase_order_dataset()

# Japanese orders — reads from png/ + json/
dataset = load_japanese_order_dataset()
```

Paths default to the above but can be overridden via `.env` (see `.env.template`).
