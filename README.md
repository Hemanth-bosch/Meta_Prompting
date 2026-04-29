# Meta-Prompt Development Pipeline

A meta-prompting system for extracting structured information from images using multimodal language models with automated prompt optimization.

## Use Cases

- **Information Extraction from Images**:
  - Japanese Order (JO) — structured field extraction from Japanese business order documents
  - YANASE — structured field extraction from YANASE automotive parts order documents
- **Image Comparison: Classification and Change Summary for DCA**:
  - DCA (Document Comparison Assist) — image-pair comparison that outputs a classification label (whether to discard the pair) and a change summary. The current baseline workflow is human-prompt based; this repository tests and improves it using DSPy meta-prompt optimization.

---

## Key Features

- **Multi-Stage Tournament Optimization**: Cascaded pipeline (Seed → Bootstrap → MIPROv2 → GEPA) that progressively refines prompts using DSPy optimizers.
- **Single-Optimizer Runs**: Standalone `extract_dca.py` for quick MIPROv2, Bootstrap, SIMBA, or GEPA experiments.
- **Grid Runner**: `run_dca_grid.py` for parallel combinatorial experiments across models and hyperparameters.
- **DCA Human-Run Baselines**: Human-authored prompt experiments with full evaluation pipeline.
- **Comprehensive Evaluation**: Composite metric (0.6 × classification + 0.4 × LLM-judge change summary), per-category breakdown, Pareto analysis of score vs token cost.
- **Pareto Selection**: Optional token-efficiency-aware candidate selection during optimization.
- **Multi-Provider LLM Support**: Azure OpenAI, Vertex AI, OpenAI, Anthropic via LiteLLM routing.

---

## Project Structure

```
meta-prompting/
├── extract_dca.py              # DCA single-optimizer CLI entrypoint
├── run_dca_grid.py             # Grid runner for parallel DCA optimization experiments
├── extract_jo.py               # Japanese Order extraction pipeline
├── extract_yanase.py           # YANASE order extraction pipeline
├── config.py                   # Model configuration and DSPy setup (multi-provider)
│
├── signatures/                 # DSPy Signature definitions
│   ├── README.md               #   Signature guide and DCA task specification
│   ├── dca.py                  #   DCA image-pair comparison signature
│   ├── jo.py                   #   JO extraction fields
│   └── yanase.py               #   YANASE extraction fields
│
├── dca_tournaments/            # Multi-stage tournament optimization pipeline
│   ├── README.md               #   Full pipeline guide
│   ├── run_tournament.py       #   CLI entrypoint — orchestrates all stages
│   ├── tour_config.py          #   TournamentConfig dataclass — all hyperparameters
│   ├── leaderboard.py          #   Candidate tracking, Pareto selection, persistence
│   ├── tour_utils.py           #   Shared helpers: evaluate_candidate, logging, sampling
│   ├── seed_prompt_generator.py #  Stage 0: LLM-written seed prompt generation
│   ├── stage_runner.py         #   Stages 1-3: subprocess-isolated compile() + eval
│   ├── optimizer_subprocess.py #   Isolated compile() worker
│   └── gepa_utils.py           #   GEPA image-aware reflection utilities
│
├── evaluation/                 # Evaluation metrics, analysis, and baseline runners
│   ├── README.md               #   DCA metric guide
│   ├── dca_metric.py           #   DCA composite metric (0.6×cls + 0.4×change summary)
│   ├── jo_metric.py            #   JO extraction metric
│   ├── yanase_metric.py        #   YANASE extraction metric
│   ├── common.py               #   Shared helpers (normalization, fuzzy matching)
│   ├── posthoc_analysis.py     #   Performance analysis and reporting
│   ├── dca_baseline_runs/      #   Human-prompt baseline experiments
│   │   ├── README.md           #     Usage guide
│   │   ├── human_prompts/      #     Prompt markdown files (v0–v5)
│   │   ├── run_human_prompts/  #     Inference scripts
│   │   └── eval/               #     Classification and change-summary eval scripts
│   └── dca_run_val_test_set_eval/  # Val/test evaluation for human + optimized runs
│       ├── README.md           #     Usage guide
│       ├── run_val_test_set_eval_on_human_runs.py
│       └── run_val_test_set_eval_on_optimized_prompt.py
│
├── optimization/               # Single-optimizer strategy builders
│   └── opt.py                  #   OptimizationRunner, strategy factories
│
├── utils/                      # Data utilities
│   ├── data_loading.py         #   load_dca_dataset()
│   ├── splitting.py            #   stratified_split_dca_dataset()
│   ├── pdf_processing.py       #   PDF-to-image conversion
│   └── dca_data_generation/    #   DCA dataset creation and augmentation
│       ├── ReadMe.md           #     Data generation guide
│       ├── generate_from_zips.py  # Generate image pairs from PDF zips
│       └── augment_rare_categories.py  # Synthetic data for B1/B2/B4 categories
│
├── data/                       # Dataset directory
│   ├── dca_data/               #   DCA image pairs and annotation JSONs
│   ├── json/                   #   JO ground truth (50 files)
│   ├── png/                    #   JO order images (50 files)
│   ├── YANASE_ground_truth/    #   YANASE ground truth JSON annotations
│   └── YANASE_images/          #   YANASE processed PNG images
│
├── .cache/                     # Cached optimized programs and split manifests
└── .env                        # Environment configuration (see env.template)
```

---

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- LLM API access (Azure OpenAI, Vertex AI, OpenAI, or Anthropic)

### Installation

1. Clone the repository:
```bash
git clone https://github.boschdevcloud.com/WAX3SYV/meta-prompting.git
cd meta-prompting
```

2. Install dependencies and activate the virtual environment:
```bash
uv sync
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

3. Configure environment variables by creating a `.env` file in the repo root:
```bash
# Model configuration (prefix determines provider)
MODEL_NAME=azure/gpt-4.1-mini

# Azure OpenAI (leave API key empty to use az login)
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# Corporate TLS (Bosch / proxy environments)
REQUESTS_CA_BUNDLE=

# Cache and data paths (defaults shown)
CACHED_EXTRACTOR_PATH=.cache/
DCA_IMAGE_PAIRS_DIR=data/dca_data/image_pairs
DCA_GT_DIR=data/dca_data/image_pairs_llm_output_exp1
```

See `env.template` for the full list of supported variables.

4. Prepare data (see [Data Setup](#data-setup) section below).

---

## Data Setup

### DCA Dataset

The DCA dataset (429 image pairs) is stored under `data/dca_data/`. It is not included in the repository — contact the data owner for access.

After placing the data, the frozen train/val/test split is created automatically on first run and saved to `.cache/dca_split_seed42.json`. Stratification is by `expected_discard_image`, `human_label_category`, and `doubt`. The split is fixed — do not delete this file.

For dataset details and synthetic augmentation, see [utils/dca_data_generation/ReadMe.md](utils/dca_data_generation/ReadMe.md).

### Japanese Order (JO) Dataset

Download from the OneDrive link (access granted on demand) and extract to:
```
data/
├── json/   # Ground truth JSON annotations (50 files)
└── png/    # Order document images (50 files)
```

### YANASE Dataset

Download from the OneDrive link and extract to:
```
data/
├── YANASE_ground_truth/   # Ground truth JSON annotations
├── YANASE_images/         # Processed PNG images
└── YANASE/                # Original PDFs (optional)
```

---

## Running the DCA Pipeline

### Option 1: Multi-Stage Tournament (Recommended)

The tournament pipeline runs up to four sequential optimization stages and selects the best candidate by val-set score (and optionally inference token cost).

```bash
# Default: seed → bootstrap → miprov2
uv run python -m dca_tournaments.run_tournament

# Dry run — print config and estimated runs, then exit
uv run python -m dca_tournaments.run_tournament --dry-run

# Full pipeline including GEPA
uv run python -m dca_tournaments.run_tournament --stages seed,bootstrap,miprov2,gepa

# Parallel: 4 concurrent optimizer subprocesses + 4 concurrent seed evaluations
uv run python -m dca_tournaments.run_tournament \
    --max-parallel 4 \
    --seed-parallel-evaluations 4

# Pareto selection (score + token efficiency)
uv run python -m dca_tournaments.run_tournament \
    --selection-criteria pareto

# Resume an interrupted run
uv run python -m dca_tournaments.run_tournament \
    --resume .cache/tournament/tournament_20260428234355
```

See [dca_tournaments/README.md](dca_tournaments/README.md) for the full argument reference and architecture details.

### Option 2: Single-Optimizer Run

```bash
# MIPROv2 optimization
uv run python extract_dca.py --optimizer miprov2 --miprov2-auto medium

# Bootstrap
uv run python extract_dca.py --optimizer bootstrap

# GEPA
uv run python extract_dca.py --optimizer gepa --gepa-auto medium

# SIMBA
uv run python extract_dca.py --optimizer simba --simba-max-steps 8

# Run all optimizers
uv run python extract_dca.py --optimizer all
```

### Option 3: Grid Runner

Run combinatorial experiments across models and hyperparameters in parallel:

```bash
# Default grid (models × intensities × demo configs)
uv run python run_dca_grid.py --parallel 4

# Dry run
uv run python run_dca_grid.py --dry-run
```

### Evaluate a Saved Model

```bash
# Evaluate an optimized model on the val set
uv run python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py \
    --model-path .cache/my_model.json \
    --model-name azure/gpt-4.1-mini \
    --split-file .cache/dca_split_seed42.json \
    --which-set val

# Evaluate human-prompt runs on val/test
uv run python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_human_runs.py --which-set both
```

See [evaluation/dca_run_val_test_set_eval/README.md](evaluation/dca_run_val_test_set_eval/README.md) for details.

---

## Running the JO and YANASE Pipelines

```bash
# Japanese Order extraction with prompt optimization
python extract_jo.py --model-name azure/gpt-4o-mini --analyze-sets both --save-model

# YANASE order extraction with prompt optimization
python extract_yanase.py --model-name azure/gpt-4o-mini --analyze-sets both --save-model
```

---

## DCA Human Prompt Baselines

Run LLM inference using hand-authored prompts and evaluate the results:

```bash
# Single run: v2 prompt with gpt-4.1-mini
python evaluation/dca_baseline_runs/run_human_prompts/run_with_human_prompts.py \
    --key-prefix v2_prompt_cr_gpt41mini \
    --prompt v2_prompt_cr.md \
    --deployment gpt-4.1-mini \
    --no-thinking-area \
    --verbose

# Batch run (configured via RUNS list inside the script)
python evaluation/dca_baseline_runs/run_human_prompts/run_all_experiments.py
```

See [evaluation/dca_baseline_runs/README.md](evaluation/dca_baseline_runs/README.md) for the full guide.

---

## DCA Data Augmentation

The original dataset had severe class imbalance (B1=0, B2=2, B4=5 samples). Synthetic pairs were generated to bring each underrepresented category to at least 20 samples:

```bash
# Preview what would be generated
python utils/dca_data_generation/augment_rare_categories.py --dry-run --verbose

# Generate synthetic pairs (appended after pair_376)
python utils/dca_data_generation/augment_rare_categories.py --verbose
```

| Category | Method | Pairs added |
|----------|--------|------------|
| B1 (corrupted/unreadable) | Gaussian blur, pixelation, resize-to-40px | 20 |
| B2 (crossed-out) | Semi-transparent diagonal X overlay | 18 |
| B4 (one info + one non-info) | Recombination from A1/B3 + B5a/B5b pools | 15 |

Total: 376 original + 53 synthetic = 429 pairs. See [utils/dca_data_generation/ReadMe.md](utils/dca_data_generation/ReadMe.md) for details.

---

## Performance

### Japanese Order (JO)

Extraction accuracy on Japanese business order documents (13 fields):

| Model | Human-crafted Prompts | Meta-Prompting | Improvement |
|-------|-----------------------|----------------|-------------|
| gpt-4o-mini | 73.68% | 84.29% | +10.62% |
| gpt-4o | 49.81% | 91.85% | +42.04% |
| gpt-5 | 86.98% | 98.26% | +11.28% |

### YANASE Order

Extraction accuracy on YANASE automotive parts order documents (16 fields):

| Model | Human-crafted Prompts | Meta-Prompting | Improvement |
|-------|-----------------------|----------------|-------------|
| gpt-4o-mini | 71.3% | 79.8% | +8.5% |
| gpt-4o | 73.9% | 83.8% | +9.9% |
| gpt-4.1 | 68.9% | 88.1% | +19.2% |
| gpt-5 | 71.1% | 79.0% | +7.9% |

---

## PDF Preprocessing (Optional)

To convert source PDFs to rotation-corrected images, install Tesseract OCR:

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows: download from https://github.com/UB-Mannheim/tesseract/wiki
```

Then:
```python
from utils import process_pdf_files_into_images_with_rotation
process_pdf_files_into_images_with_rotation("data/YANASE", "data/YANASE_images")
```

---

## Provider Configuration

This project supports multiple LLM providers via LiteLLM routing. Set the `MODEL_NAME` prefix to select a provider:

| Prefix | Provider | Example |
|--------|----------|---------|
| `azure/` | Azure OpenAI | `azure/gpt-4.1-mini` |
| `vertex_ai/` | Google Vertex AI | `vertex_ai/gemini-2.0-flash` |
| `openai/` | OpenAI | `openai/gpt-4o` |
| `anthropic/` | Anthropic | `anthropic/claude-sonnet-4-20250514` |
