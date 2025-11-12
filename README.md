# Meta-Prompt Development Pipeline

A meta-prompting system for extracting structured information from images using multimodal language models with automated prompt optimization.

## Use Case
- Japanese Order Extraction from Images


### Key Features

- **Meta-Prompt Optimization**: Automated prompt engineering for information extraction tasks, including optimizers: 
   - Fewshot learning optimization: boostrap examples with random search
   - Instruction optimization: MIPROv2, SIMBA
- **Comprehensive Evaluation**: Field-specific scoring with text normalization and extact/fuzzy matching

## 📋 Project Structure

```
meta-prompting/
├── main.py                   # Japanese order extraction pipeline
├── mainv2.py                 # YANASE order extraction pipeline
├── config.py                 # Model configuration and DSPy setup
├── utils.py                  # Dataset loading and preprocessing utilities
├── eval.py                   # Evaluation metrics and scoring functions
├── opt.py                    # Optimization strategies and runner classes
├── posthoc_analysis.py       # Performance analysis and reporting
├── debug_products.py         # Product detail debugging utilities
├── data/                     # Dataset directory
│   ├── json/                # Japanese order ground truth JSON (50 files)
│   ├── png/                 # Japanese order images (50 files)
│   ├── YANASE_ground_truth/ # YANASE ground truth JSON annotations
│   ├── YANASE_images/       # YANASE processed PNG images
│   └── YANASE/              # YANASE original PDF documents
├── .cache/                   # Cached optimized extractors
└── .env                      # Environment configuration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Azure OpenAI API access
- DSPy framework
- uv package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.boschdevcloud.com/WAX3SYV/meta-prompting.git
cd meta-prompting
```

2. Install dependencies with [`uv`](https://docs.astral.sh/uv/) package and activate the virtual environment:
```bash
uv sync
# activate the virtual env:
source .venv/bin/activate # Windows: .venv\Scripts\activate
```

3. Configure environment variables in `.env`:
```bash
# Azure OpenAI configuration
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_API_VERSION=2025-04-01-preview

# Model configuration
MODEL_NAME=azure/gpt-4o-mini
REFLECTION_LM_NAME=azure/gpt-5

# Path to the cached extractor
CACHED_EXTRACTOR_PATH=.cache/
CACHED_EXTRACTOR_DIR=.cache/

# YANASE Dataset Paths (optional - defaults shown)
YANASE_IMAGES_DIR=data/YANASE_images
YANASE_GT_DIR=data/YANASE_ground_truth

# Japanese Order Dataset Paths (optional - defaults shown)
JAPANESE_ORDER_IMAGES_DIR=./data/png
JAPANESE_ORDER_GT_DIR=./data/json
```

4. Prepare data within the `data` folder:

**4.1 Japanese Order (JO) Dataset**

Download from: [JO data - OneDrive link (access granted on demand)](https://bosch-my.sharepoint.com/:u:/r/personal/wax3syv_bosch_com/Documents/aim205/meta-agents/meta-prompting-data.zip?csf=1&web=1&e=HGZVem)

Extract to create the following structure:
```
data/
├── json/               # Ground truth JSON annotations (50 files)
└── png/                # Order document images (50 files)
```

**4.2 YANASE Order Dataset**

Download from: [YANASE data - OneDrive link (access granted on demand)](https://bosch-my.sharepoint.com/:u:/r/personal/wax3syv_bosch_com/Documents/aim205/meta-agents/yanase_data-20251112.zip?csf=1&web=1&e=HVqYqe)

Extract to create the following structure:
```
data/
├── YANASE_ground_truth/  # Ground truth JSON annotations
├── YANASE_images/        # Processed PNG images from YANASE PDFs
└── YANASE/               # Original source PDF documents (optional)
```


    
### Running the Pipeline

#### Japanese Order Extraction (main.py)
Execute the Japanese order extraction pipeline:
```bash
# Basic usage (uses defaults from .env)
python main.py

# With custom configuration
python main.py \
    --images-dir ./data/png \
    --gt-dir ./data/json \
    --model-name azure/gpt-4o-mini \
    --analyze-sets both \
    --save-model \
    --save-results

# Other available options:
# --no-cache              Disable LLM response caching
# --load-extractor PATH   Load previously saved extractor
# --skip-load             Skip loading cached extractor
# --analyze-sets [val|test|both]  Which dataset to analyze (default: test)
```

#### YANASE Order Extraction (mainv2.py)
Execute the YANASE order extraction pipeline:
```bash
# Basic usage (uses defaults from .env)
python mainv2.py

# With custom configuration
python mainv2.py \
    --images-dir data/YANASE_images \
    --ground-truth-dir data/YANASE_ground_truth \
    --model-name azure/gpt-4o-mini \
    --analyze-sets both \
    --save-model \
    --save-results

# Same options as main.py available
```


### Performance

#### Japanese Order (JO) Dataset
Extraction accuracy on Japanese business order documents with 13 fields:

| Model       | Baseline Prompting | Meta-Prompting | Improvement |
|-------------|--------------------| ---------------|-------------|
| gpt-4o-mini | 73.68%             | 84.29%         | +10.62%     |
| gpt-4o      | 49.81%             | 91.85%         | +42.04%     |
| gpt-5       | 86.98%             | **98.26%**     | +11.28%     |


<!-- 
**Extracted Fields:**
- Seller information: company name (2 lines), postal code, address, telephone, fax
- Order details: order date, comments
- Shipping information: postal code, telephone, address, company name
- Product details: PO numbers, product codes and quantities 
-->


#### YANASE Order Dataset
Extraction accuracy on YANASE automotive parts order documents with 16 fields:

| Model       | Baseline Prompting | Meta-Prompting | Improvement |
|-------------|--------------------| ---------------|-------------|
| gpt-4o-mini | TBD                | TBD            | TBD         |
| gpt-5       | TBD                | TBD            | TBD         |

<!-- 
**Extracted Fields:**
- Seller information: company name (primary and secondary names), telephone, fax
- Receiver information: company name, shipping postal code, address, telephone, fax
- Product orders (list): receiver code, order code, product code, Bosch code, quantity, unit price, PCD price 
-->

---

**Note**: This project requires access to Azure OpenAI services and is optimized for image order document processing. Ensure proper API quotas and rate limiting when running optimization experiments.
