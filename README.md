# DCA Prompt Optimization — Document Comparison Assist

Automatically optimize the prompt for **DCA (Document Comparison Assist)** — a vision-language task that compares an original document image against a re-imaged version, decides whether to **discard** the pair, and **describes what changed**. Built on [DSPy](https://github.com/stanfordnlp/dspy).

> **In one line:** give it the task (a DSPy Signature), the labelled image-pair dataset, and the scoring metric — it finds a materially better prompt than a human wrote, and shows you the score/token trade-off.

**Headline result:** a full tournament lifts DCA from a **~72%** human-prompt baseline to **92.28%** (gpt‑4.1, val n=82) in a single automated run — and a prompt optimized cheaply on `gpt‑4.1‑mini` transfers to `gpt‑5.2` at **90.5%**, matching the best human prompt at ~5× lower optimizer cost.

---

## Table of Contents
1. [What DCA is](#1-what-dca-is)
2. [The metric](#2-the-metric)
3. [Results](#3-results)
4. [Setup](#4-setup)
5. [Quick start](#5-quick-start)
6. [Run the tournament (detailed)](#6-run-the-tournament-detailed)
7. [Run a single optimizer — extract_dca.py (detailed)](#7-run-a-single-optimizer--extract_dcapy-detailed)
8. [Evaluate a saved prompt](#8-evaluate-a-saved-prompt)
9. [Outputs](#9-outputs)
10. [Repository structure](#10-repository-structure)
11. [Troubleshooting](#11-troubleshooting)
12. [Adding a new use case](#12-adding-a-new-use-case)

---

## 1. What DCA is

Given two document images (`image_original`, `image_reimage`), the model outputs:

| Output | Meaning |
|--------|---------|
| `discard_image` | `"true"` → images identical / only trivial differences → discard the pair. `"false"` → a meaningful change exists (or one image is informational-only / crossed-out / unreadable). |
| `changes` | A branch-consistent explanation (e.g. `"Images are identical - no changes"`, `"NoMatchFound"`, `"Image has been crossed out"`, or a detailed `OVERVIEW … DETAILS …` for real changes). |

The task contract lives in [signatures/dca.py](signatures/dca.py) (`DCAImageComparison`). All prompt guidance lives in the **instruction docstring** ("instructions-only" mode); field descriptions are kept empty so the optimizers have one clean surface to improve.

**Dataset:** 429 image pairs (376 real + 53 synthetic augmentations for rare categories), 8 categories (A1–B5b), stratified 60/20/20 split (seed 42) → **train 254 / val 82 / test 93**.

---

## 2. The metric

[evaluation/dca_metric.py](evaluation/dca_metric.py):

```
DCA Score = 0.60 × classification_score + 0.40 × change_score
```

- **classification_score** — exact match of predicted `discard_image` vs. ground truth (rule-based, no LLM).
- **change_score** — quality of the `changes` text vs. the human reference, via rule-based fast-paths + a persistent judge cache + an **LLM judge** (`gpt-4.1`) that returns precision/recall combined with a weighted F-beta.

**Optional hybrid metrics** (token-efficiency pressure, used *only* to steer optimizer candidate selection — final ranking always uses the raw DCA score):

```
Hybrid Linear      = 0.5 × DCA + 0.5 × max(0, 1 - tokens/12000)
Hybrid Exponential = DCA × exp(-2.0 × max(0, tokens/2500 - 1))
```

---

## 3. Results

All on the val set (n=82), temp 0.1, teacher/seed/reflection = `gpt-5.2`.

**Standalone vs. tournament (student = gpt-4.1):**

| Approach | DCA | Notes |
|----------|-----|-------|
| Human v1 (MiDAS) | 74.5% | manual baseline |
| Human v2 (CR) | 83.3% | manual baseline |
| MIPROv2 (standalone) | 90.7% | |
| Bootstrap (standalone) | 89.3% | |
| GEPA (standalone) | 91.0% | multimodal reflection |
| **Tournament (Pareto)** | **92.28%** | best — GEPA was the winning stage |
| Tournament (Hybrid Linear) | 92.84% @ 6,624 tok | best cost-efficiency |

**Cross-model transfer** (prompt optimized on `gpt-4.1-mini`, run on larger models): gpt-4.1-mini 86.5% → gpt-4.1 90.0% → gpt-5.2 **90.5%** (vs. 91.8% best human on gpt-5.2).

**Key findings:** Pareto selection beats best-score (+2.43 pp and ~1.9× fewer tokens); **GEPA with multimodal reflection is the single most impactful stage** (won 9 of ~15 tournament runs); hybrid metrics reliably keep prompts small (<5K tokens/sample).

---

## 4. Setup

**Requirements:** Python 3.13, an LLM provider (Azure OpenAI / Vertex / OpenAI / Anthropic). The DCA **judge** authenticates via `az login` (Azure CLI), so an active Azure login is required even when the student LM uses an API key.

```bash
# 1. Install deps (uv recommended)
uv sync
# then activate the venv:
#   Windows PowerShell:  & ".venv\Scripts\Activate.ps1"
#   Linux/Mac:           source .venv/bin/activate

# 2. Credentials — copy env.template to .env and fill in:
```

```env
MODEL_NAME=azure/gpt-4.1-mini            # student LM (provider prefix picks the API)
REFLECTION_LM_NAME=azure/gpt-5.2         # GEPA reflection / seed / teacher LM
AZURE_OPENAI_API_KEY=...                 # leave blank to use `az login` instead
AZURE_OPENAI_ENDPOINT=https://<your>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2025-01-01-preview
REQUESTS_CA_BUNDLE=C:\path\to\corp-ca.pem   # corporate TLS proxy (Bosch)
DCA_IMAGE_PAIRS_DIR=data/dca_data/image_pairs
DCA_GT_DIR=data/dca_data/image_pairs_llm_output_exp1
```

```bash
# 3. Azure login (required for the LLM judge)
az login
```

> **Windows note:** prefix runs with `$env:PYTHONIOENCODING="utf-8"`. If `uv` isn't installed, call the venv Python directly, e.g. `.\.meta-prompting\Scripts\python.exe extract_dca.py ...`.

The frozen train/val/test split is created on first run and cached under `.cache/splits/` — don't delete it (evaluation reproducibility depends on it).

---

## 5. Quick start

```bash
# See what a full tournament would run — no API calls
python -m dca_tournaments.run_tournament --dry-run

# Full tournament with defaults (seed → bootstrap → miprov2 → gepa-medium)
python -m dca_tournaments.run_tournament

# One optimizer only
python extract_dca.py --optimizer gepa --gepa-auto medium --skip-load
```

---

## 6. Run the tournament (detailed)

Four sequential stages, each seeded by the best candidates of the previous one:

```
Stage 0  Seed       gpt-5.2 writes N instruction variants (+ best/worst demos)
Stage 1  Bootstrap  BootstrapFewShotWithRandomSearch (teacher gpt-5.2)
Stage 2  MIPROv2    Bayesian instruction + demo search (proposer/teacher gpt-5.2)
Stage 3  GEPA       reflection LM SEES failing image-pairs and rewrites the prompt
                    ── top-K by Pareto (score + tokens) carried between stages ──
```

**Full command with every argument spelled out** (PowerShell; use `` ` `` for line continuation, `\` in bash):

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m dca_tournaments.run_tournament `
  --stages seed,bootstrap,miprov2,gepa `
  --student-model azure/gpt-4.1-mini `
  --seed-model azure/gpt-5.2 --reflection-model azure/gpt-5.2 `
  --no-cot `
  --top-k 2 --selection-criteria pareto --cross-stage-selection --track-tokens --instructions-only `
  --split-seed 42 --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 `
  --mini-train-size 120 --mini-val-size 45 `
  --eval-temperature 0.1 --seed-eval-temperature 0.1 --seed-generation-temperature 1.0 `
  --n-seed-candidates 2 --max-context-examples 8 --seed-context-mode best `
  --seed-reasoning-effort low --seed-add-demos --seed-parallel-evaluations 2 `
  --bootstrap-num-threads 4 --bootstrap-teacher-model azure/gpt-5.2 `
  --miprov2-auto medium --miprov2-max-labeled 3 --miprov2-max-bootstrapped 5 `
  --miprov2-num-threads 4 --miprov2-prompt-model azure/gpt-5.2 `
  --miprov2-prompt-reasoning-effort low --miprov2-teacher-model azure/gpt-5.2 `
  --gepa-auto medium --gepa-reflection-images --gepa-reflection-detail auto `
  --gepa-reflection-max-images 10 --gepa-minibatch-size 5 `
  --gepa-reflection-model-name azure/gpt-5.2 --gepa-teacher-model azure/gpt-5.2 `
  --gepa-perfect-score 1.0 --gepa-lowest-score 0.0 --gepa-candidate-selection pareto `
  --gepa-add-format-failure-feedback `
  --max-parallel 2 --eval-num-threads 2
```

**Common variations** (swap into the command above):

| Goal | Change |
|------|--------|
| Best-score selection | `--selection-criteria score_only` |
| Student = gpt-4.1 | `--student-model azure/gpt-4.1` |
| Seeds shown failing examples | `--seed-context-mode worst` |
| Hybrid Linear metric | `--hybrid-metric --hybrid-penalty-mode linear --hybrid-max-prompt-tokens 12000` |
| Hybrid Exponential metric | `--hybrid-metric --hybrid-penalty-mode exponential --hybrid-target-tokens 2500 --hybrid-penalty-steepness 2.0` |
| Seed-only study | `--stages seed` |
| Resume an interrupted run | `--resume-from .cache/tournament/tournament_<timestamp>` |

`--resume-from` reloads the saved `config.json` and skips completed stages — you don't re-pass the other flags.

Full argument reference: [dca_tournaments/README.md](dca_tournaments/README.md).

---

## 7. Run a single optimizer — extract_dca.py (detailed)

One optimizer over the full train/val split. Every knob is a CLI flag (no hard-coding); teacher/prompt/reflection models default to the student LM (GEPA reflection defaults to `REFLECTION_LM_NAME`).

```powershell
# GEPA (multimodal reflection ON by default; reflection LM = gpt-5.2 from .env)
python extract_dca.py --optimizer gepa --gepa-auto medium `
  --gepa-reflection-images --gepa-reflection-detail auto --gepa-reflection-max-images 10 `
  --skip-load

# MIPROv2 (with a gpt-5.2 teacher + proposer, deck demo budget 3/5)
python extract_dca.py --optimizer miprov2 --miprov2-auto medium `
  --max-labeled-demos 3 --max-bootstrapped-demos 5 `
  --miprov2-teacher-model azure/gpt-5.2 --miprov2-prompt-model azure/gpt-5.2 `
  --skip-load

# Bootstrap (few-shot random search; tune the search here)
python extract_dca.py --optimizer bootstrap `
  --bootstrap-labeled-demos 3 --bootstrap-bootstrapped-demos 5 `
  --bootstrap-num-candidate-programs 16 --bootstrap-max-rounds 2 `
  --bootstrap-teacher-model azure/gpt-5.2 `
  --skip-load

# Any of the above on gpt-4.1:  add  --model-name azure/gpt-4.1
```

Key flags: `--optimizer {gepa,miprov2,bootstrap,simba,all}`, `--gepa-auto/--miprov2-auto {light,medium,heavy}`, `--model-name`, `--no-cot`, `--skip-load` (start clean), `--split-seed/--train-ratio/--val-ratio/--test-ratio`, `--eval-num-threads`. Run `python extract_dca.py --help` for the complete list.

---

## 8. Evaluate a saved prompt

```bash
# Optimized prompt on the val (or test) set
python evaluation/dca_run_val_test_set_eval/run_val_test_set_eval_on_optimized_prompt.py \
    --model-path .cache/<saved_model>.json --model-name azure/gpt-4.1-mini --which-set val

# Human-prompt baselines
python evaluation/dca_baseline_runs/run_human_prompts/run_all_experiments.py
```

See [evaluation/dca_run_val_test_set_eval/README.md](evaluation/dca_run_val_test_set_eval/README.md) and [evaluation/dca_baseline_runs/README.md](evaluation/dca_baseline_runs/README.md).

---

## 9. Outputs

- **Tournament:** `.cache/tournament/tournament_<ts>/` → `leaderboard.json` / `.xlsx` / `.txt`, `config.json`, `summary.json`, `pareto_plot.png`, and per-stage `program.json` + `program.meta.json` (scores, per-category, tokens/sample). Logs under `log/tournament/tournament_<ts>/`.
- **Single optimizer:** `.cache/<model>_<strategy>_..._dca_optimized_<ts>.json` (the saved program), plus seed/optimized state JSONs and a run-summary; the log is renamed to match the model.

---

## 10. Repository structure

```
extract_dca.py            Single-optimizer CLI (MIPROv2 / Bootstrap / GEPA / SIMBA)
run_dca_grid.py           Grid runner — parallel experiments across models/hyperparams
config.py                 LM configuration (multi-provider via LiteLLM) + az-login/CA bundle

signatures/dca.py         DCAImageComparison signature (the task contract)
evaluation/
  dca_metric.py           Composite metric + LLM judge + hybrid metrics
  dca_baseline_runs/       Human-prompt baseline experiments (v0–v5) + eval
  dca_run_val_test_set_eval/  Val/test evaluation for any saved model
optimization/opt.py       OptimizationRunner + Bootstrap/MIPROv2/GEPA/SIMBA factories
utils/
  data_loading.py         load_dca_dataset()
  splitting.py            stratified_split_dca_dataset()
  pdf_processing.py       PDF→image (optional)
  dca_data_generation/    Dataset creation + rare-category augmentation

dca_tournaments/          Multi-stage tournament pipeline
  run_tournament.py         CLI entrypoint (orchestrates all stages)
  tour_config.py            TournamentConfig — all hyperparameters
  leaderboard.py            Candidate tracking, Pareto front, persistence
  tour_utils.py             Shared helpers (evaluate, sampling, demo normalization)
  seed_prompt_generator.py  Stage 0 — LLM-written seed prompts
  stage_runner.py           Stages 1–3 — subprocess-isolated compile() + eval
  optimizer_subprocess.py   Isolated compile() worker
  gepa_utils.py             Multimodal GEPA reflection (image store + LM wrapper)

data/dca_data/            Image pairs + annotation JSONs (not committed — see Setup)
ReadMe-Understand-Optimizers/       How each optimizer works
ReadMe-Onboarding_new_use_case/     Adapt this framework to a new task
tests/                    test_tournament_defenses.py
```

---

## 11. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `python: command not found` / not recognized | Activate the venv (`uv sync` then activate), or call the venv Python directly. |
| Judge/`az` error: "Azure CLI token retrieval failed" | Run `az login`; ensure network/VPN reaches the Azure endpoint. |
| First LLM call times out | You're off the network/VPN that can reach `AZURE_OPENAI_ENDPOINT`. |
| Windows Unicode errors | `$env:PYTHONIOENCODING="utf-8"` before the command. |
| A tournament stage failed mid-run | Re-run with `--resume-from .cache/tournament/tournament_<ts>` (skips completed stages). |

---

## 12. Adding a new use case

This tournament framework is task-agnostic. To adapt it to a new document task (new signature + metric + data loader), follow [ReadMe-Onboarding_new_use_case/ONBOARDING_NEW_USE_CASE.md](ReadMe-Onboarding_new_use_case/ONBOARDING_NEW_USE_CASE.md).
