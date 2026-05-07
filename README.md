# Meta-Prompt Development Pipeline

This repository provides a framework for automatically optimizing LLM prompts using [DSPy](https://github.com/stanfordnlp/dspy). It supports multiple document-processing use cases and includes tools for data loading, stratified splitting, evaluation metrics, and multi-stage prompt optimization.

**In plain terms:** you give it a task description (a DSPy Signature), a dataset with ground-truth labels, and a scoring metric — and it automatically finds a better prompt than you started with.

---

## Table of Contents

1. [(DCA) Getting Started in 5 Minutes](#dca-getting-started-in-5-minutes)
2. [Use Cases](#use-cases-in-this-repository)
3. [Project Structure](#project-structure)
4. [Setup](#setup)
5. [Data Setup](#data-setup)
6. [(DCA) Running the DCA Pipeline](#dca-running-the-dca-pipeline)
7. [Running JO and YANASE Pipelines](#running-the-jo-and-yanase-pipelines)
8. [(DCA) Human Prompt Baselines](#dca-human-prompt-baseline-pipeline)
9. [Performance Results](#performance-results)
10. [LLM Provider Configuration](#llm-provider-configuration)
11. [Adding a New Use Case](#adding-a-new-use-case)
12. [(DCA) Runtime Expectations](#dca-runtime-expectations)
13. [(DCA) Cost Estimates](#dca-cost-estimates)
14. [(DCA) Example Output Files](#dca-example-output-files)
15. [(DCA) Resuming an Interrupted Run](#dca-resuming-an-interrupted-run)
16. [(DCA) Test Set Evaluation](#dca-test-set-evaluation)
17. [FAQ](#faq)
18. [Troubleshooting](#troubleshooting)

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

## (DCA) Getting Started in 5 Minutes

> **Fastest path to running a DCA tournament on your machine.**

**Step 1 — Install**
```bash
git clone https://github.boschdevcloud.com/WAX3SYV/meta-prompting.git
cd meta-prompting
uv sync
```

**Step 2 — Configure credentials**
```bash
cp env.template .env
# Edit .env: set AZURE_OPENAI_ENDPOINT and either AZURE_OPENAI_API_KEY or use az login
```

**Step 3 — Get data**

Download `dca_data.zip` from the SharePoint link in [data/README.md](data/README.md) and extract it to `data/dca_data/`.

**Step 4 — Dry-run (no API calls)**
```bash
uv run python -m dca_tournaments.run_tournament --dry-run
```
This prints the full configuration and estimated API calls, then exits. Verify the models, dataset size, and stage list look correct.

**Step 5 — Run**
```bash
# Fastest config: mini dataset, gpt-4.1-mini student (3.5–8 h)
uv run python -m dca_tournaments.run_tournament \
    --stages seed,bootstrap,miprov2,gepa \
    --student-model azure/gpt-4.1-mini \
    --seed-model azure/gpt-5.2 \
    --reflection-model azure/gpt-5.2 \
    --mini-train-size 120 --mini-val-size 45 \
    --max-parallel 2 --eval-num-threads 4
```

Results land in `.cache/tournament/<timestamp>/`. See [Running the DCA Pipeline](#running-the-dca-pipeline) for the full options reference.

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

Once data is in place, the frozen train/val/test split is created automatically on first run and saved under `.cache/splits/` with a fingerprinted name like `.cache/splits/dca_split_seed42_<fingerprint>.json`. Some tooling also writes or reuses the legacy compatibility alias `.cache/dca_split_seed42.json`. Do not delete these split manifests — evaluation scripts depend on them for reproducibility.

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

## (DCA) Running the DCA Pipeline

The DCA pipeline has three modes depending on how much you want to run:

### (DCA) Mode 1 — Multi-Stage Tournament (Recommended for best results)

Runs four sequential optimization stages (Seed → Bootstrap → MIPROv2 → GEPA). Each stage uses the best candidates from the previous stage as its starting point. Produces a leaderboard plus final score and Pareto summaries for manual or criteria-based selection.

```bash
# Default: seed → bootstrap → miprov2 → gepa, student model = gpt-4.1-mini
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
    --resume-from .cache/tournament/tournament_20260428234355
```

See [dca_tournaments/README.md](dca_tournaments/README.md) for the full argument reference, stage details, and output artifacts.

### (DCA) Mode 2 — Single-Optimizer Run

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

# If an optimizer stage fails after retries, rerun from the preserved seed state
uv run python extract_dca.py --optimizer all --load-extractor log\..._seed_state.json
```

Single-optimizer runs are now fail-fast: if Bootstrap, MIPROv2, GEPA, or SIMBA hits an unrecoverable optimizer error, the script stops immediately, writes an error summary JSON, preserves the seed state JSON, and prints a rerun command that resumes from that saved seed state instead of silently continuing with a degraded result.

### (DCA) Mode 3 — Grid Runner

Run combinatorial experiments across models and hyperparameters in parallel:

```bash
# (DCA) Default grid: 2 models × 1 intensity × 2 demo configs = 4 parallel runs
uv run python run_dca_grid.py --parallel 4

# Preview what would run without calling the API
uv run python run_dca_grid.py --dry-run
```

### (DCA) Evaluating a Saved DCA Model

After optimization, evaluate the saved model on the frozen val or test split:

```bash
# (DCA) Evaluate an optimized model on the val set
uv run python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py \
    --model-path .cache/my_optimized_model.json \
    --model-name azure/gpt-4.1-mini \
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

> **Start here if you are adapting this framework for a new document-processing task.**

Follow the step-by-step guide in [ReadMe-Onboarding_new_use_case/ONBOARDING_NEW_USE_CASE.md](ReadMe-Onboarding_new_use_case/ONBOARDING_NEW_USE_CASE.md). The guide covers:
1. Defining a DSPy `Signature` for your task
2. Writing a data loader and stratified splitter
3. Implementing an evaluation metric
4. Creating a pipeline entry-point script (like `extract_dca.py`)
5. Optionally wiring the tournament pipeline for multi-stage optimization

See also [ReadMe-Understand-Optimizers/](ReadMe-Understand-Optimizers/) for guidance on choosing between Bootstrap, MIPROv2, GEPA, and SIMBA for your use case.

---

## (DCA) Runtime Expectations

Optimizer runtime depends on the **student model**, **dataset size**, and **Azure API throughput** at the time of the run. Numbers below are measured from completed DCA tournaments with the default configuration (`mini_train=120`, `mini_val=45`, `top_k=2`, `n_threads=4`).

### Default config — student: `gpt-4.1-mini`, seed/reflection: `gpt-5.2`

| Stage | What runs | Typical duration | Notes |
|-------|-----------|-----------------|-------|
| **Seed** (Stage 0) | `gpt-5.2` generates 2 candidate instructions; evaluates 6 programs (bare + demo variants) on mini_val | 10–45 min | Can reach 130 min under heavy API load |
| **Bootstrap** (Stage 1) | `n=16, max_rounds=2, d3b5` — 2 seeds × 1 config = 2 subprocess runs, teacher=`gpt-5.2` | 30–90 min per run; **~1–2 h total** | 4 threads inside each subprocess |
| **MIPROv2** (Stage 2) | `medium` intensity, d3b5 — 2 subprocess runs, prompt_model=`gpt-5.2` | 25–140 min per run; **~1–3 h total** | Varies with API capacity |
| **GEPA** (Stage 3) | `medium` auto, image reflection, `gpt-5.2` — 2 subprocess runs | 75–160 min per run; **~2–3 h total** | Image reflection adds cost; up to 10 images per mini_val example |
| **Full tournament** | All 4 stages in sequence | **3.5–8 h** | gpt-4.1-mini student, default DCA dataset |

### With student: `gpt-4.1` (full model — approx. 5× slower inference)

| Stage | Typical duration |
|-------|-----------------|
| Seed | 30–75 min |
| Bootstrap | ~90 min |
| MIPROv2 | ~6 h |
| GEPA | ~12 h |
| **Full tournament** | **~22 h** |

> These measurements are from DCA image-pair comparisons (429 pairs). Your use case may be faster (text-only, smaller dataset) or slower (larger images, more output fields).

---

## (DCA) Cost Estimates

Costs depend on Azure pricing tier, model versions, and API load. The estimates below are rough ballparks for the DCA use case (image-pair inputs — **images are the dominant cost factor**).

### gpt-4.1-mini student + gpt-5.2 seed/reflection (default config)

| Stage | Main model cost driver | Approx. cost |
|-------|----------------------|-------------|
| Seed (Stage 0) | gpt-5.2 seed generation + mini_val eval (45 examples) | $1–5 |
| Bootstrap (Stage 1) | gpt-4.1-mini mini_val eval across candidates | $1–4 |
| MIPROv2 (Stage 2) | gpt-5.2 proposal calls + gpt-4.1-mini eval (~30 trials) | $3–10 |
| GEPA (Stage 3) | gpt-5.2 image reflection + gpt-4.1-mini mini_val eval | $3–15 |
| **Full tournament** | | **~$8–35** |

### gpt-4.1 student (full model, 5× more expensive inference)

| Stage | Approx. cost |
|-------|-------------|
| Seed | $3–12 |
| Bootstrap | $5–15 |
| MIPROv2 | $15–40 |
| GEPA | $25–60 |
| **Full tournament** | **~$50–130** |

> **Notes:**
> - DCA image inputs are JPEG/PNG pairs — each example sends 2 images to the model. Token counts vary with image size and Azure vision billing mode.
> - Costs scale roughly linearly with `mini_val_size` and number of optimizer trials.
> - Check current Azure OpenAI pricing at [azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/).
> - Text-only use cases (no images) are significantly cheaper — typically 5–10× less per example.

---

## (DCA) Resuming an Interrupted Run

If a tournament is interrupted (network timeout, crash, manual stop), resume it from where it stopped:

```bash
# Each tournament saves state to .cache/tournament/<tournament_id>/
uv run python -m dca_tournaments.run_tournament \
    --resume-from .cache/tournament/tournament_20260428234355
```

**What gets reused vs re-run:**
- Stages with a `stage_completed.json` marker are **skipped entirely**.
- Interrupted stages are **resumed** — compiled candidates already in the leaderboard are reused; only missing work items run.
- The tournament config is reloaded from `<tournament_id>/config.json` — no need to pass flags again.

If an optimizer subprocess fails, the error summary and seed state are preserved. The console prints the exact `--resume-from` command to restart.

---

## (DCA) Test Set Evaluation

By default the tournament does **not** evaluate on the hold-out test set. Test evaluation should happen once, on a final selected candidate, to avoid repeated measurement leakage.

To enable test evaluation at the end of a tournament:

```bash
uv run python -m dca_tournaments.run_tournament --run-test-eval
```

The test score appears in `tournament_summary.json` under `test_evaluation` and in the final console output. It does **not** influence candidate selection — selection always uses the val set score.

---

## (DCA) Example Output Files

After a completed tournament, `.cache/tournament/<tournament_id>/` contains:

```
tournament_20260507_123456/
├── config.json                      # Full TournamentConfig used for this run (reload with --resume-from)
├── leaderboard.json                 # All candidates ever evaluated, with scores and token stats
├── leaderboard.xlsx                 # Excel version with per-stage sheets and Pareto highlights
├── tournament_summary.json          # Final result: best candidate, scores, stage timing, test eval
├── summary_scores.csv               # Per-candidate score table (easy to diff across runs)
├── pareto_plot.png                  # Score vs. avg-tokens scatter, Pareto front highlighted
│
├── seed/
│   ├── stage_completed.json         # Written on success — resume skips this stage
│   ├── seed_0_bare/                 # Candidate: seed instruction 0, no demos
│   │   └── program.json            #   Saved DSPy program (load with comparator.load())
│   ├── seed_0_with_demos/
│   └── seed_baseline_bare/         # Safety-net: original signature bare (cannot regress below this)
│
├── bootstrap/
│   ├── stage_completed.json
│   └── Balanced_DCA_Bootstrap_seed0/
│       └── program.json
│
├── miprov2/
│   ├── stage_completed.json
│   └── MIPROv2_light_seed0/
│       └── program.json
│
└── gepa/
    ├── stage_completed.json
    └── GEPA_medium_seed0/
        └── program.json
```

**Key files to inspect after a run:**

| File | What to look at |
|------|-----------------|
| `tournament_summary.json` | `best_candidate.val_total_score`, `best_candidate.program_path`, `stage_timings` |
| `leaderboard.json` | Full candidate list — sort by `val_total_score` or `val_avg_total_tokens` |
| `leaderboard.xlsx` | Pareto sheet highlights token-efficient candidates; use for manual selection |
| `summary_scores.csv` | Quick diff across multiple tournament runs |
| `<stage>/<config>/program.json` | Load directly: `comparator.load("program.json")` then run inference |

To load the best program for inference:
```python
import dspy
from signatures.dca import DCAImageComparison

comparator = dspy.Predict(DCAImageComparison)
comparator.load(".cache/tournament/tournament_20260507_123456/gepa/GEPA_medium_seed0/program.json")
```

---

## FAQ

**Q: Do I need to run all four stages every time?**

No. Pass `--stages` to choose which stages to run. A common starting point is `--stages seed,bootstrap` which is fast (~1–2 h with gpt-4.1-mini) and often sufficient for a first optimized prompt. Add `miprov2` or `gepa` once you want to push further.

**Q: Can I use this for a use case other than DCA?**

Yes — the tournament pipeline is use-case-agnostic. You need a DSPy `Signature`, a data loader, a metric function, and a small wiring script. Follow the [ReadMe-Onboarding_new_use_case/ONBOARDING_NEW_USE_CASE.md](ReadMe-Onboarding_new_use_case/ONBOARDING_NEW_USE_CASE.md) guide for step-by-step instructions. The DCA tournament is the reference implementation.

**Q: I don't have Azure CLI. Can I still use the pipeline?**

Yes. Set `AZURE_OPENAI_API_KEY` in your `.env` file with a direct API key. The pipeline checks for a key first and only falls back to Azure CLI if the key is absent. See the Troubleshooting section for the exact `.env` entry.

**Q: How do I pick the best result after a tournament?**

Open `leaderboard.xlsx` in the tournament output directory. The **Pareto sheet** highlights candidates that are on the score-vs-token efficiency frontier. For deployment, pick from the Pareto front:
- If inference cost does not matter → take the row with the highest `val_total_score`.
- If you want to minimize token usage → pick the Pareto candidate with the lowest `val_avg_total_tokens` that still meets your minimum score threshold.

The `program_path` column gives the exact `.json` file to load with `comparator.load(path)`.

**Q: Can I continue optimization starting from a previous best candidate?**

Yes. Use `--seed-from-file` to skip Stage 0 and start Bootstrap/MIPROv2/GEPA from an existing saved program:
```bash
uv run python -m dca_tournaments.run_tournament \
    --seed-from-file .cache/tournament/tournament_20260507_123456/gepa/GEPA_medium_seed0/program.json \
    --stages bootstrap,miprov2,gepa
```

---

## Troubleshooting

### Azure CLI not found / `az` is not recognized

```
Error: Azure CLI executable 'az' was not found in PATH (after venv removal)
```

1. Install Azure CLI: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli
2. Log in: `az login`
3. Verify: `az account show` — should print your subscription
4. Re-run the script

Alternatively, set `AZURE_OPENAI_API_KEY` in `.env` to use direct API-key auth and skip Azure CLI entirely.

### Python not recognized / venv activation fails (Windows)

```
python : The term 'python' is not recognized
```

Use `uv run` (recommended — it activates the venv automatically):
```powershell
uv run python -m dca_tournaments.run_tournament
```

Or activate manually first:
```powershell
.\.meta-prompting\Scripts\Activate.ps1
python -m dca_tournaments.run_tournament
```

### `UnicodeEncodeError` on Windows

```powershell
$env:PYTHONIOENCODING='utf-8'
uv run python -m dca_tournaments.run_tournament
```

### Tournament produces ~0.0 scores on all candidates

This is the nested-ThreadPoolExecutor / Azure ContextVar bug. The tournament pipeline already isolates each optimizer in a subprocess to prevent it. If you see near-zero scores from your own code:
- Do not call `optimizer.compile()` directly from a thread pool — use the subprocess isolation pattern in `optimizer_subprocess.py`.
- Verify Azure credentials are valid: auth failures also produce silent 0-scores in some DSPy versions.

### `FileNotFoundError` during GEPA (Windows MAX_PATH)

Windows limits paths to 260 characters. After cross-stage cascading, config names can become very long. Fix: clone the repo to a short root path (e.g. `C:\mp\`) rather than a deeply nested OneDrive folder.
