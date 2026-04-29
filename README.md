# Meta-Prompt Development Pipeline

This repository provides a framework for automatically optimizing LLM prompts using [DSPy](https://github.com/stanfordnlp/dspy). It supports multiple document-processing use cases and includes tools for data loading, stratified splitting, evaluation metrics, and multi-stage prompt optimization.

**In plain terms:** you give it a task description (a DSPy Signature), a dataset with ground-truth labels, and a scoring metric — and it automatically finds a better prompt than you started with.

---

## Use Cases in This Repository

Three use cases are currently implemented. Each has its own data, signature, metric, and pipeline scripts.

| Use Case | Task | Key Output |
|----------|------|------------|
| **DCA** (Document Comparison Assist) | Compare two document images: classify whether to discard the pair and describe what changed | `discard_image` (true/false) + `changes` (text) |
| **JO** (Japanese Order) | Extract structured fields from Japanese business order document images | 13 structured fields (seller, order, shipping, product) |
| **YANASE** | Extract structured fields from YANASE automotive parts order documents | 16 structured fields (seller, receiver, product list) |

> **New to this repo?** Start by reading this README, then go to the section for your use case. If you are adding a new use case entirely, see [ReadMe-Onboarding_new_use_case/ONBOARDING_NEW_USE_CASE.md](ReadMe-Onboarding_new_use_case/ONBOARDING_NEW_USE_CASE.md).

---

## Project Structure

```
meta-prompting/
│
│   ── DCA-specific top-level scripts ──
├── extract_dca.py              # (DCA) Single-optimizer CLI: MIPROv2 / Bootstrap / GEPA / SIMBA
├── run_dca_grid.py             # (DCA) Grid runner: parallel experiments across models + hyperparams
│
│   ── Other use case scripts ──
├── extract_jo.py               # (JO) Japanese Order extraction pipeline
├── extract_yanase.py           # (YANASE) YANASE order extraction pipeline
│
│   ── Shared infrastructure ──
├── config.py                   # LM configuration and DSPy setup (multi-provider via LiteLLM)
├── env.template                # Template for all environment variables
│
├── signatures/                 # DSPy Signature definitions (task contracts)
│   ├── README.md               #   What signatures are + DCA decision policy
│   ├── dca.py                  #   (DCA) Image-pair comparison signature
│   ├── jo.py                   #   (JO) Order extraction fields
│   └── yanase.py               #   (YANASE) Parts order extraction fields
│
├── dca_tournaments/            # (DCA) Multi-stage tournament optimization pipeline
│   ├── README.md               #   Full guide — architecture, CLI, stages, selection
│   ├── run_tournament.py       #   CLI entrypoint — orchestrates all stages
│   ├── tour_config.py          #   All hyperparameters in one dataclass
│   ├── leaderboard.py          #   Candidate tracking, Pareto front, persistence
│   ├── tour_utils.py           #   Shared helpers: evaluate_candidate, logging, sampling
│   ├── seed_prompt_generator.py #  Stage 0: LLM writes improved instruction candidates
│   ├── stage_runner.py         #   Stages 1-3: subprocess-isolated compile() + eval
│   ├── optimizer_subprocess.py #   Isolated compile() worker process
│   └── gepa_utils.py           #   (DCA) GEPA image-aware reflection utilities
│
├── evaluation/                 # Metrics and evaluation runners
│   ├── README.md               #   (DCA) Metric guide: scoring formula, LLM judge
│   ├── dca_metric.py           #   (DCA) Composite metric: 0.6×classification + 0.4×change summary
│   ├── jo_metric.py            #   (JO) Field extraction metric
│   ├── yanase_metric.py        #   (YANASE) Field extraction metric
│   ├── common.py               #   Shared helpers: text normalization, fuzzy matching
│   ├── posthoc_analysis.py     #   Post-hoc performance breakdown
│   │
│   ├── dca_baseline_runs/      # (DCA) Human-prompt baseline experiments
│   │   ├── README.md           #   Step-by-step guide
│   │   ├── human_prompts/      #   Prompt markdown files v0–v5
│   │   ├── run_human_prompts/  #   Inference scripts (single run, batch, category coding)
│   │   └── eval/               #   Classification + change-summary eval scripts
│   │
│   └── dca_run_val_test_set_eval/  # (DCA) Val/test evaluation for any saved model
│       ├── README.md               #   Usage guide
│       ├── run_val_test_set_eval_on_human_runs.py
│       └── run_val_test_set_eval_on_optimized_prompt.py
│
├── optimization/               # Single-optimizer strategy builders (shared)
│   └── opt.py                  #   OptimizationRunner + Bootstrap / MIPROv2 / GEPA / SIMBA factories
│
├── utils/                      # Data utilities (shared)
│   ├── data_loading.py         #   load_dca_dataset(), load_jo_dataset(), load_yanase_dataset()
│   ├── splitting.py            #   Stratified train/val/test splitting per use case
│   ├── pdf_processing.py       #   PDF-to-image conversion (used for JO and YANASE)
│   └── dca_data_generation/    #   (DCA) Dataset creation and augmentation tools
│       ├── ReadMe.md           #     Guide
│       ├── generate_from_zips.py
│       └── augment_rare_categories.py
│
├── data/                       # Datasets (not committed to git — see Data Setup below)
│   ├── dca_data/               #   (DCA) Image pairs + annotation JSONs (429 pairs)
│   ├── json/                   #   (JO) Ground truth annotations (50 files)
│   ├── png/                    #   (JO) Order document images (50 files)
│   ├── YANASE_ground_truth/    #   (YANASE) Ground truth annotations
│   └── YANASE_images/          #   (YANASE) Processed PNG images
│
├── .cache/                     # Auto-generated: saved optimized programs + split manifests
└── .env                        # Your credentials (copy from env.template, do not commit)
```

---

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- LLM API access — Azure OpenAI, Vertex AI, OpenAI, or Anthropic

### Installation

1. Clone the repository:
```bash
git clone https://github.boschdevcloud.com/WAX3SYV/meta-prompting.git
cd meta-prompting
```

2. Install dependencies and activate the virtual environment:
```bash
uv sync
source .venv/bin/activate         # Linux / Mac
.venv\Scripts\activate            # Windows cmd
& ".venv\Scripts\Activate.ps1"    # Windows PowerShell
```

3. Configure credentials — copy `env.template` to `.env` and fill in your values:
```env
# Provider prefix in MODEL_NAME determines which API is used
MODEL_NAME=azure/gpt-4.1-mini

# Azure OpenAI (leave API key blank to use "az login" instead)
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# Corporate TLS proxy (Bosch networks)
REQUESTS_CA_BUNDLE=

# (DCA) Data paths — change these if your data is elsewhere
DCA_IMAGE_PAIRS_DIR=data/dca_data/image_pairs
DCA_GT_DIR=data/dca_data/image_pairs_llm_output_exp1
```

See `env.template` for all supported variables including Vertex AI, OpenAI, and Anthropic.

---

## Data Setup

### (DCA) DCA Dataset

The DCA dataset (429 image pairs) lives under `data/dca_data/` and is not included in the repository. Contact the data owner for access.

Once data is in place, the frozen train/val/test split is created automatically on first run and saved to `.cache/dca_split_seed42.json`. Do not delete this file — all evaluation scripts depend on it for reproducibility.

The DCA dataset has 8 categories (A1/A2/B1/B2/B3/B4/B5a/B5b). Three categories (B1, B2, B4) were originally underrepresented and were augmented with 53 synthetic pairs. Total: 376 original + 53 synthetic = 429 pairs. See [utils/dca_data_generation/ReadMe.md](utils/dca_data_generation/ReadMe.md).

### (JO) Japanese Order Dataset

Download from the OneDrive link (access granted on demand) and extract to:
```
data/
├── json/   # Ground truth JSON annotations (50 files)
└── png/    # Order document images (50 files)
```

### (YANASE) YANASE Dataset

Download from the OneDrive link and extract to:
```
data/
├── YANASE_ground_truth/   # Ground truth JSON annotations
├── YANASE_images/         # Processed PNG images
└── YANASE/                # Original PDFs (optional)
```

---

## Running the DCA Pipeline

The DCA pipeline has three modes depending on how much you want to run:

### Mode 1 — Multi-Stage Tournament (Recommended for best results)

Runs four sequential optimization stages (Seed → Bootstrap → MIPROv2 → GEPA). Each stage uses the best candidates from the previous stage as its starting point. Produces a leaderboard and selects the overall best.

```bash
# Default: seed → bootstrap → miprov2, student model = gpt-4.1-mini
uv run python -m dca_tournaments.run_tournament

# First time — use dry-run to see what will run without calling the API
uv run python -m dca_tournaments.run_tournament --dry-run

# Full pipeline including GEPA (all 4 stages)
uv run python -m dca_tournaments.run_tournament --stages seed,bootstrap,miprov2,gepa

# Run in parallel (4 concurrent optimizer subprocesses + 4 concurrent seed evaluations)
uv run python -m dca_tournaments.run_tournament \
    --max-parallel 4 \
    --seed-parallel-evaluations 4

# Select candidates by Pareto front (score + inference token efficiency)
uv run python -m dca_tournaments.run_tournament --selection-criteria pareto

# Resume an interrupted run
uv run python -m dca_tournaments.run_tournament \
    --resume .cache/tournament/tournament_20260428234355
```

See [dca_tournaments/README.md](dca_tournaments/README.md) for the full argument reference, stage details, and output artifacts.

### Mode 2 — Single-Optimizer Run

Run one optimizer in isolation for quick experiments:

```bash
# (DCA) MIPROv2 optimization
uv run python extract_dca.py --optimizer miprov2 --miprov2-auto medium

# (DCA) Bootstrap few-shot
uv run python extract_dca.py --optimizer bootstrap

# (DCA) GEPA (Generative Prompt Adaptation)
uv run python extract_dca.py --optimizer gepa --gepa-auto medium

# (DCA) SIMBA
uv run python extract_dca.py --optimizer simba --simba-max-steps 8
```

### Mode 3 — Grid Runner

Run combinatorial experiments across models and hyperparameters in parallel:

```bash
# (DCA) Default grid: 2 models × 1 intensity × 2 demo configs = 4 parallel runs
uv run python run_dca_grid.py --parallel 4

# Preview what would run without calling the API
uv run python run_dca_grid.py --dry-run
```

### Evaluating a Saved DCA Model

After optimization, evaluate the saved model on the frozen val or test split:

```bash
# (DCA) Evaluate an optimized model on the val set
uv run python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py \
    --model-path .cache/my_optimized_model.json \
    --model-name azure/gpt-4.1-mini \
    --split-file .cache/dca_split_seed42.json \
    --which-set val

# (DCA) Evaluate human-prompt baseline runs on val + test
uv run python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_human_runs.py \
    --which-set both
```

See [evaluation/dca_run_val_test_set_eval/README.md](evaluation/dca_run_val_test_set_eval/README.md).

---

## Running the JO and YANASE Pipelines

```bash
# (JO) Japanese Order extraction with prompt optimization
python extract_jo.py --model-name azure/gpt-4o-mini --analyze-sets both --save-model

# (JO) With custom configuration
python extract_jo.py \
    --images-dir ./data/png \
    --gt-dir ./data/json \
    --model-name azure/gpt-4o-mini \
    --analyze-sets both \
    --save-model \
    --save-results

# (YANASE) YANASE order extraction with prompt optimization
python extract_yanase.py --model-name azure/gpt-4o-mini --analyze-sets both --save-model

# (YANASE) With custom configuration
python extract_yanase.py \
    --images-dir data/YANASE_images \
    --ground-truth-dir data/YANASE_ground_truth \
    --model-name azure/gpt-4o-mini \
    --analyze-sets both
```

---

## (DCA) Human Prompt Baseline Pipeline

Run LLM inference using hand-authored prompts and compare against the optimized results:

```bash
# (DCA) Single run: v2 prompt with gpt-4.1-mini
python evaluation/dca_baseline_runs/run_human_prompts/run_with_human_prompts.py \
    --key-prefix v2_prompt_cr_gpt41mini \
    --prompt v2_prompt_cr.md \
    --deployment gpt-4.1-mini \
    --no-thinking-area \
    --verbose

# (DCA) Batch run (configured in the RUNS list inside the script)
python evaluation/dca_baseline_runs/run_human_prompts/run_all_experiments.py
```

See [evaluation/dca_baseline_runs/README.md](evaluation/dca_baseline_runs/README.md) for the full guide including all prompt versions (v0–v5) and eval scripts.

---

## (DCA) Data Augmentation

The DCA dataset originally had severe class imbalance (B1=0, B2=2, B4=5 samples). Synthetic pairs were generated to bring each underrepresented category to at least 20 samples:

```bash
# Preview — see what would be generated without writing files
python utils/dca_data_generation/augment_rare_categories.py --dry-run --verbose

# Generate and append after pair_376
python utils/dca_data_generation/augment_rare_categories.py --verbose
```

| Category | Augmentation method | Pairs added |
|----------|---------------------|-------------|
| B1 (corrupted / unreadable) | Gaussian blur, pixelation, resize to 40px | 20 |
| B2 (crossed out) | Semi-transparent diagonal X overlay | 18 |
| B4 (one info + one non-info) | Recombination from A1/B3 + B5a/B5b pools | 15 |

All synthetic pairs include `is_synthetic: true` and full provenance metadata. See [utils/dca_data_generation/ReadMe.md](utils/dca_data_generation/ReadMe.md).

---

## Performance Results

### (JO) Japanese Order — 13 extracted fields

| Model | Human-written Prompt | Meta-Prompting | Improvement |
|-------|----------------------|----------------|-------------|
| gpt-4o-mini | 73.68% | 84.29% | +10.62% |
| gpt-4o | 49.81% | 91.85% | +42.04% |
| gpt-5 | 86.98% | 98.26% | +11.28% |

### (YANASE) YANASE Order — 16 extracted fields

| Model | Human-written Prompt | Meta-Prompting | Improvement |
|-------|----------------------|----------------|-------------|
| gpt-4o-mini | 71.3% | 79.8% | +8.5% |
| gpt-4o | 73.9% | 83.8% | +9.9% |
| gpt-4.1 | 68.9% | 88.1% | +19.2% |
| gpt-5 | 71.1% | 79.0% | +7.9% |

---

## PDF Preprocessing (Optional)

If your source data is in PDF format, convert it to images first. Requires Tesseract OCR for rotation detection:

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows: download from https://github.com/UB-Mannheim/tesseract/wiki
```

Then convert:
```python
from utils import process_pdf_files_into_images_with_rotation
process_pdf_files_into_images_with_rotation("data/YANASE", "data/YANASE_images")
```

---

## LLM Provider Configuration

This project supports multiple LLM providers via LiteLLM. Set the `MODEL_NAME` prefix in `.env` to choose:

| Prefix | Provider | Example value |
|--------|----------|---------------|
| `azure/` | Azure OpenAI | `azure/gpt-4.1-mini` |
| `vertex_ai/` | Google Vertex AI | `vertex_ai/gemini-2.0-flash` |
| `openai/` | OpenAI direct | `openai/gpt-4o` |
| `anthropic/` | Anthropic | `anthropic/claude-sonnet-4-20250514` |

Ensure you have sufficient API quota before running optimization experiments — optimizer runs can make hundreds of LLM calls.

---

## Adding a New Use Case

To add a new document-processing task to this framework, follow the step-by-step guide in [ReadMe-Onboarding_new_use_case/ONBOARDING_NEW_USE_CASE.md](ReadMe-Onboarding_new_use_case/ONBOARDING_NEW_USE_CASE.md). The guide covers defining a signature, writing a data loader and metric, creating a pipeline script, and optionally adapting the tournament pipeline.
