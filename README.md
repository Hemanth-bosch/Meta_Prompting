# Meta-Prompt Development Pipeline

A meta-prompting system for extracting structured information from images using multimodal language models with automated prompt optimization.

## Use Case
- Information Extraction from Images:
    - Japanese Order (JO)
    - YANASE
- Image comparision: Classification, and Change summary for DCA:
    - DCA: Document Comparison Assist is a document image-pair comparison, classification, and change summary tool. It outputs a json with classification labels (whether to discard image pairs) and change summary for each image pair. The current baseline workflow is human-prompt based, and we are using it as a basis for testing the effectiveness of meta-prompt optimization in improving LLM performance on this task.


### Key Features

- **Meta-Prompt Optimization**: Automated prompt engineering for information extraction tasks, including optimizers: 
   - Fewshot learning optimization: boostrap examples with random search
   - Instruction optimization: MIPROv2, SIMBA
- **DCA Human-Run Baselines**: Human Prompt-based image-pair comparison workflow with classification and change-summary generation.
- **Comprehensive Evaluation**: Field-specific scoring with text normalization and extact/fuzzy matching

## 📋 Project Structure

```
meta-prompting/
├── extract_dca.py             # DCA extraction/experimentation entrypoint
├── extract_jo.py              # Japanese order extraction pipeline
├── extract_yanase.py          # YANASE order extraction pipeline
├── config.py                  # Model configuration and DSPy setup (multi-provider)
│
├── signatures/                # DSPy Signature definitions
│   ├── dca.py                 #   DCA fields
│   ├── jo.py                  #   JO extraction fields
│   └── yanase.py              #   YANASE extraction fields
│
├── evaluation/                # Evaluation metrics and analysis
│   ├── common.py              #   Shared helpers (normalization, similarity)
│   ├── dca_metric.py          #   DCA weighted metric (0.6 classification + 0.4 change summary)
│   ├── jo_metric.py           #   JO-specific metrics
│   ├── yanase_metric.py       #   YANASE-specific metrics
│   ├── posthoc_analysis.py    #   Performance analysis and reporting
│   └── dca_baseline_runs/     #   DCA human-prompt baseline experiments
│       ├── eval_human_runs.py #     Metrics, plots, confusion matrices
│       ├── eval_results/      #     Generated charts and eval_summary.csv
│       ├── human_prompts/     #     Prompt markdown files (v0–v4)
│       └── run_human_prompts/ #     Runners: single, batch, v4 triage, categorization
│
├── utils/                     # Data utilities
│   ├── data_loading.py        #   Dataset loading functions
│   ├── splitting.py           #   Train/val/test splitting
│   ├── pdf_processing.py      #   PDF-to-image conversion
│   └── dca_data_generation/   #   DCA dataset generation tools
│       ├── generate_from_zips.py  # Generate image pairs from PDF zips
│       └── augment_rare_categories.py  # Synthetic data for B1/B2/B4 categories
│
├── optimization/              # Optimization strategies
│   └── opt.py                 #   OptimizationRunner and strategy builders
│
├── tests/                     # Test and example scripts
│   ├── test_model_comparison.py
│   ├── test_vertex_gemini.py
│   └── ...
│
├── data/                      # Dataset directory
│   ├── dca_data/              #   DCA pairs and experiment JSON outputs (temporary for experimentation)
│   ├── json/                  #   JO ground truth JSON (50 files)
│   ├── png/                   #   JO order images (50 files)
│   ├── YANASE_ground_truth/   #   YANASE ground truth JSON annotations
│   ├── YANASE_images/         #   YANASE processed PNG images
│   └── YANASE/                #   YANASE original PDF documents
│
├── .cache/                    # Cached optimized extractors
└── .env                       # Environment configuration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- LLM API access (Azure OpenAI, Vertex AI, OpenAI, or Anthropic)
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

3. Configure environment variables in `.env`. A template is provided in `.env.template`:
```bash
# Model configuration (prefix determines provider)
MODEL_NAME=azure/gpt-4o-mini          # Azure OpenAI
# MODEL_NAME=vertex_ai/gemini-2.0-flash # Vertex AI Gemini
# MODEL_NAME=openai/gpt-4o             # OpenAI direct
# MODEL_NAME=anthropic/claude-sonnet-4-20250514  # Anthropic direct

# Azure OpenAI (required for azure/ models)
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_API_VERSION=2025-04-01-preview

# Vertex AI (required for vertex_ai/ models)
# VERTEX_PROJECT=your_project_id
# VERTEX_LOCATION=us-central1

# Caching and data paths
CACHED_EXTRACTOR_PATH=.cache/
CACHED_EXTRACTOR_DIR=.cache/
YANASE_IMAGES_DIR=data/YANASE_images
YANASE_GT_DIR=data/YANASE_ground_truth
JAPANESE_ORDER_IMAGES_DIR=./data/png
JAPANESE_ORDER_GT_DIR=./data/json
```

4. Prepare data within the `data` folder:

>Please email data owners of the JO and YANASE data (Matthew Deng (Matthew.DENG@cn.bosch.com) and Vinayak Kumar (Vinayak.Kumar2@sg.bosch.com)) and send their confirmation email proofs to xingbo.wang@us.bosch.com

**4.1 Japanese Order (JO) Dataset**

Download from: [JO data - OneDrive link (access granted on demand)](https://bosch-my.sharepoint.com/:u:/r/personal/wax3syv_bosch_com/Documents/aim205/meta-agents/JO-data.zip?csf=1&web=1&e=liwUyV) 

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


    
## Running the Pipeline

### PDF Preprocessing (Optional)

If your source data is in PDF format, you can convert it to images with automatic rotation correction:

**Prerequisites:**
- Install Tesseract OCR for rotation detection:
  ```bash
  # macOS
  brew install tesseract

  # Ubuntu/Debian
  sudo apt-get install tesseract-ocr

  # Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
  ```

    **Usage:**
    ```python
    from utils import process_pdf_files_into_images_with_rotation

    # Convert PDFs to rotation-corrected images
    pdf_dir = "data/YANASE"              # Directory containing PDF files
    output_dir = "data/YANASE_images"     # Output directory for PNG images

    process_pdf_files_into_images_with_rotation(pdf_dir, output_dir)
    ```


### Japanese Order Extraction with Prompt Optimization (extract_jo.py)
Execute the Japanese order extraction pipeline:
```bash
# Basic usage (uses defaults from .env)
python extract_jo.py

# With custom configuration
python extract_jo.py \
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

### YANASE Order Extraction with Prompt Optimization (extract_yanase.py)
Execute the YANASE order extraction pipeline:
```bash
# Basic usage (uses defaults from .env)
python extract_yanase.py

# With custom configuration
python extract_yanase.py \
    --images-dir data/YANASE_images \
    --ground-truth-dir data/YANASE_ground_truth \
    --model-name azure/gpt-4o-mini \
    --analyze-sets both \
    --save-model \
    --save-results

# Same options as extract_jo.py available
```

### DCA Human-Run Baseline Pipeline (Brief)

The DCA workflow currently runs from `evaluation/dca_baseline_runs/` and focuses on image-pair triage/change assessment using human-authored prompts.

- Single run: `evaluation/dca_baseline_runs/run_human_prompts/run_with_human_prompts.py`
- Multi-step v4 pipeline: `evaluation/dca_baseline_runs/run_human_prompts/run_v4_human_prompt.py`
- Batch orchestration: `evaluation/dca_baseline_runs/run_human_prompts/run_all_experiments.py`
- Metrics and reports: `evaluation/dca_baseline_runs/eval_human_runs.py`

For detailed usage and prompt/model combinations, see `evaluation/dca_baseline_runs/README.md`.

### DCA Data Augmentation

The original DCA dataset had severe class imbalance (B1=0, B2=2, B4=5 samples). Synthetic pairs were generated to bring each underrepresented category to ≥20 samples:

```bash
# Preview what would be generated
python utils/dca_data_generation/augment_rare_categories.py --dry-run --verbose

# Generate synthetic pairs (appended after pair_376)
python utils/dca_data_generation/augment_rare_categories.py --verbose
```

| Category | Method | Count |
|----------|--------|-------|
| B1 (corrupted/unreadable) | Gaussian blur, pixelation, resize-to-tiny | 20 |
| B2 (crossed-out) | Semi-transparent diagonal X overlay (1px, alpha 45–100) | 18 |
| B4 (one info + one non-info) | Recombination from A1/B3 + B5a/B5b pools | 15 |

All synthetic pairs include `is_synthetic: true`, `source_pairs`, and `data_augmentation_explain_in_detail` for traceability. Total dataset: 429 pairs (376 original + 53 synthetic).

### DCA Prompt Optimization Pipeline

The `extract_dca.py` script runs DSPy-based prompt optimization on the DCA dataset:

```bash
# MIPROv2 optimization (default)
uv run python extract_dca.py --optimizer miprov2 --miprov2-auto medium

# SIMBA optimization
uv run python extract_dca.py --optimizer simba --simba-max-steps 8

# Run all optimizers
uv run python extract_dca.py --optimizer all --analyze-sets both --save-model --save-results
```

The pipeline uses a frozen train/val/test split (saved in `.cache/dca_split_seed42.json`) stratified by `expected_discard_image`, `expected_human_label_category`, and `doubt`. New pairs added to the dataset are automatically assigned to the train set.

Evaluation covers two dimensions:
- **Classification** (discard_image): accuracy, precision, recall, F1, confusion matrix
- **Change summary** (changes): weighted F-beta via GPT-5.2 judge (precision 60% / recall 40%)


### Performance

#### Japanese Order (JO) Dataset
Extraction accuracy on Japanese business order documents with 13 fields:

| Model       | Human-crafted Prompts | Meta-Prompting | Improvement |
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

| Model       | Human-crafted Prompts | Meta-Prompting | Improvement |
|-------------|-----------------------| ---------------|-------------|
| gpt-4o-mini | 71.3%                 | 79.8%          | +8.5%       |
| gpt-4o      | 73.9%                 | 83.8%          | +9.9%       |
| gpt-4.1     | 68.9%                 | **88.1%**      | +19.2%      |
| gpt-5       | 71.1%                 | 79.0%          | +7.9%       |

<!-- 
**Extracted Fields:**
- Seller information: company name (primary and secondary names), telephone, fax
- Receiver information: company name, shipping postal code, address, telephone, fax
- Product orders (list): receiver code, order code, product code, Bosch code, quantity, unit price, PCD price 
-->

---

**Note**: This project supports multiple LLM providers (Azure OpenAI, Vertex AI, OpenAI, Anthropic) via LiteLLM routing. Set the `MODEL_NAME` prefix to select a provider. Ensure proper API quotas and rate limiting when running optimization experiments.
