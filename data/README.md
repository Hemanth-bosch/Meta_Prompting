# Data Directory

Place datasets here locally. 
- **Yanase**: YANASE/, YANASE_images/, YANASE_ground_truth/
- **Japanese Order**: png/, json/

```
data/
├── YANASE/                  # Raw order PDFs
├── YANASE_images/           # Rendered page PNGs (from PDFs)
├── YANASE_ground_truth/     # Curated ground truth JSONs
├── png/                     # Japanese order images
└── json/                    # Japanese order ground truth
```

## Loading

```python
from utils import load_yanase_order_dataset, load_japanese_order_dataset

# YANASE — reads from YANASE_images/ + YANASE_ground_truth/
dataset = load_yanase_order_dataset()

# Japanese orders — reads from png/ + json/
dataset = load_japanese_order_dataset()
```

Paths default to the above but can be overridden via `.env` (see `.env.template`).
