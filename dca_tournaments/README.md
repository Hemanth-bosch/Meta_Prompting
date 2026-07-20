# DCA Tournaments — Cascaded Prompt Optimization Pipeline

This folder contains the end-to-end multi-stage tournament pipeline for optimizing DCA (Document Comparison Assist) prompts using DSPy. It runs up to four sequential optimization stages, each seeding from the winners of the previous stage, and selects the best candidate by val-set score and/or inference token efficiency.

---

## Table of Contents

1. [What It Does](#1-what-it-does)
2. [File Structure](#2-file-structure)
3. [Quick Start](#3-quick-start)
4. [Full Command Reference](#4-full-command-reference)
5. [The Four Stages](#5-the-four-stages)
6. [Candidate Selection](#6-candidate-selection)
7. [Configuration Reference](#7-configuration-reference)
8. [Output Artifacts](#8-output-artifacts)
9. [Resume and Recovery](#9-resume-and-recovery)
10. [Parallelism Architecture](#10-parallelism-architecture)
11. [Token Tracking](#11-token-tracking)
12. [Adapting to a New Use Case](#12-adapting-to-a-new-use-case)

---

## 1. What It Does

The tournament pipeline applies four increasingly powerful DSPy optimizers in sequence:

```
Stage 0 — Seed         LLM-written instruction candidates (gpt-5.2 generates, student model evaluates)
Stage 1 — Bootstrap    Few-shot example selection + random search
Stage 2 — MIPROv2      Bayesian instruction + demo optimization
Stage 3 — GEPA         Generative prompt adaptation with image-aware reflection
```

At each transition the top-K candidates (by score and/or token efficiency) are carried forward as seeds for the next stage. The result is a leaderboard of all candidates tried across all stages, with the global best selected at the end and evaluated on the full val set and optionally the test set.

---

## 2. File Structure

```
dca_tournaments/
├── run_tournament.py          # CLI entrypoint — main() orchestrates the full pipeline
├── tour_config.py             # TournamentConfig dataclass — all hyperparameters in one place
├── leaderboard.py             # OptimizationLeaderboard + CandidateEntry — persistence, Pareto, selection
├── tour_utils.py              # Shared helpers: evaluate_candidate, logging, save/load, sampling
├── seed_prompt_generator.py   # Stage 0: LLM-written seed prompts + baseline safety nets
├── stage_runner.py            # Stages 1-3: subprocess-isolated compile() + in-process evaluation
├── optimizer_subprocess.py    # Isolated worker: receives config JSON, calls DSPy compile(), saves result
├── gepa_utils.py              # GEPA-specific: MultimodalReflectionLM, image store, patched dataset
└── __init__.py                # Package exports
```

---

## 3. Quick Start

```bash
# Activate environment
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# Default run: seed → bootstrap → miprov2 → gepa, student = gpt-4.1-mini
python -m dca_tournaments.run_tournament

# Dry run — print config and estimated API calls, then exit
python -m dca_tournaments.run_tournament --dry-run

# Stage 0 only — smoke test with 1 seed candidate
python -m dca_tournaments.run_tournament --stages seed --n-seed-candidates 1

# Full pipeline including GEPA
python -m dca_tournaments.run_tournament --stages seed,bootstrap,miprov2,gepa

# Skip seed generation, use an existing saved program as starting point
python -m dca_tournaments.run_tournament \
    --seed-from-file .cache/my_best.json \
    --stages bootstrap,miprov2

# Run with parallel optimizers and parallel seed evaluations
python -m dca_tournaments.run_tournament \
    --max-parallel 4 \
    --seed-parallel-evaluations 4

# Resume an interrupted run
python -m dca_tournaments.run_tournament \
    --resume-from .cache/tournament/tournament_20260428234355
```

---

## 4. Full Command Reference

### Stage control

| Argument | Default | Description |
|----------|---------|-------------|
| `--stages` | `seed,bootstrap,miprov2,gepa` | Comma-separated list of stages to run. Valid: `seed,bootstrap,miprov2,gepa` |
| `--top-k` | `2` | Candidates passed from one stage to the next |
| `--skip-final-tournament` | False | Skip the final full-val-set re-evaluation |
| `--test-eval` | False | Run the optional test-set evaluation subprocess at the end |
| `--test-eval-criteria` | `best_score` | Candidate selection rule for test-set evaluation |
| `--dry-run` | False | Print config + estimated runs, then exit without running anything |
| `--resume-from` | None | Path to a previous tournament output directory to continue from |

### Models

| Argument | Default | Description |
|----------|---------|-------------|
| `--student-model` | `azure/gpt-4.1-mini` | Model used during stage evaluation and optimization |
| `--seed-model` | `azure/gpt-5.2` | Strong model that writes improved seed prompts (Stage 0) |
| `--reflection-model` | `azure/gpt-5.2` | Model used for GEPA reflection (Stage 3) |
| `--bootstrap-model` | student model | Teacher model for Bootstrap stage |
| `--miprov2-model` | student model | Prompt model for MIPROv2 stage |
| `--gepa-model` | student model | Override for GEPA stage |
| `--cot` / `--no-cot` | `--no-cot` | Use ChainOfThought vs plain Predict wrapper |

### Temperatures

| Argument | Default | Description |
|----------|---------|-------------|
| `--optimization-temperature` | None | Temperature for the LM during compile() calls |
| `--eval-temperature` | `0.1` | Temperature for all val-set evaluations |
| `--seed-generation-temperature` | `1.0` | Temperature for the seed-writing LLM |
| `--seed-eval-temperature` | `0.1` | Temperature when evaluating seed candidates |

### Stage 0 — Seed

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-seed-candidates` | `3` | Number of LLM-generated seed instruction variants |
| `--seed-from-file` | None | Skip seed generation; load this saved program file instead |
| `--seed-context-mode` | `best` | Examples shown to seed LLM: `best` or `worst` |
| `--max-context-examples` | `6` | Max example image pairs shown in the seed meta-prompt |
| `--no-add-demos-variant` | False | Disable the `with_demos` variant of each seed candidate |
| `--seed-parallel-evaluations` | `4` | Max concurrent seed candidate evaluations |
| `--seed-reasoning-effort` | `low` | Reasoning effort for the seed-writing LLM (`low/medium/high`) |

### Stage 1 — Bootstrap

| Argument | Default | Description |
|----------|---------|-------------|
| `--bootstrap-n` | `[2]` | List of `n` values (random search candidates) |
| `--bootstrap-max-rounds` | `[1]` | Max bootstrap rounds per config |
| `--bootstrap-labeled-demos` | `[6]` | Max labeled demos per config |
| `--bootstrap-bootstrapped-demos` | `[4]` | Max bootstrapped demos per config |
| `--bootstrap-stop-at-score` | None | Stop early if this score is reached |
| `--bootstrap-num-threads` | `4` | Threads for Bootstrap's internal evaluation |

### Stage 2 — MIPROv2

| Argument | Default | Description |
|----------|---------|-------------|
| `--miprov2-intensities` | `light` | Comma-separated: `light`, `medium`, `heavy` |
| `--miprov2-models` | student model | Comma-separated model list to try |
| `--miprov2-prompt-model` | `azure/gpt-5.2` | LLM used to generate instruction proposals |
| `--miprov2-prompt-model-reasoning-effort` | `low` | Reasoning effort for the prompt model |
| `--miprov2-labeled-demos` | `[6]` | Max labeled demos |
| `--miprov2-bootstrapped-demos` | `[6]` | Max bootstrapped demos |
| `--miprov2-minibatch-size` | None | Minibatch size for MIPROv2 internal eval |
| `--miprov2-metric-threshold` | None | Score threshold for demo selection |

### Stage 3 — GEPA

| Argument | Default | Description |
|----------|---------|-------------|
| `--gepa-auto` | `medium` | GEPA intensity: `light`, `medium`, `heavy` |
| `--gepa-reflection-model` | `azure/gpt-5.2` | Model for reflective feedback generation |
| `--gepa-no-reflection-images` | False | Disable image context in GEPA reflection |
| `--gepa-reflection-max-images` | `6` | Max images attached to reflection prompt |
| `--gepa-candidate-selection` | `pareto` | Internal GEPA candidate selection: `pareto`, `score_only`, `tokens_only` |

### Selection and output

| Argument | Default | Description |
|----------|---------|-------------|
| `--selection-criteria` | `score_only` | Cross-stage selection: `score_only`, `pareto`, `tokens_only` |
| `--output-dir` | `.cache/tournament/<timestamp>` | Where to save all artifacts |
| `--log-dir` | `log/tournament/<timestamp>` | Where to save logs |
| `--mini-train-size` | `120` | Size of the mini training set used during optimization |
| `--mini-val-size` | `41` | Size of the mini val set used during evaluation |
| `--max-parallel` | `1` | Max concurrent optimizer subprocesses |
| `--eval-num-threads` | `4` | Threads for parallel val-set evaluation |

---

## 5. The Four Stages

### Stage 0: Seed Prompt Generation

`seed_prompt_generator.py`

Uses a strong LLM (default: gpt-5.2) to generate N improved instruction variants from the current signature. The process:

1. Runs single-pass inference on the train set to identify failing examples per category.
2. Builds a multimodal meta-prompt: current instruction + per-category failure analysis + up to 6 example image pairs.
3. Asks the seed LLM to produce N instruction variants as a JSON array.
4. Evaluates each variant (and optionally a `+demos` variant) on the mini val set.
5. Also evaluates the original signature bare and with demos as safety-net baselines (`seed_baseline_bare`, `seed_baseline_with_demos`). These compete in selection — the pipeline can never regress below the original.

Each evaluation uses its own isolated LM instance for correct per-sample token tracking.

### Stage 1: Bootstrap

`stage_runner.py` + `optimizer_subprocess.py`

Applies `BootstrapFewShotWithRandomSearch`. For each combination of (seed candidate × bootstrap config), an isolated Python subprocess runs `optimizer.compile()` and saves the compiled program. The parent process then loads the result and evaluates it on the mini val set.

### Stage 2: MIPROv2

Same subprocess architecture as Bootstrap. Applies `MIPROv2` with configurable intensity (`light/medium/heavy`), optional separate prompt model (default: gpt-5.2 with low reasoning), and configurable demo counts. Can run multiple models and intensities in parallel.

### Stage 3: GEPA

Applies `GEPA` with `MultimodalReflectionLM`. The reflection model receives failing examples with their image pairs and generates textual feedback used to improve the instruction. GEPA performs internal Pareto selection across its own candidate set.

---

## 6. Candidate Selection

At every stage transition, `OptimizationLeaderboard.top_k()` selects the seeds for the next stage.

### Criteria

| Criteria | Behavior |
|----------|----------|
| `score_only` | Sort by `val_total_score` descending, take top K |
| `pareto` | Maximize `val_total_score`, minimize `val_avg_total_tokens`. Select `ceil(K/2)` score leaders + `floor(K/2)` efficiency leaders from the Pareto front, fill remaining by score. Requires token data — raises a hard error if any candidate is missing tokens. |
| `tokens_only` | Sort by `val_avg_total_tokens` ascending (cheapest first), break ties by score. Requires token data. |

### Token dimensions

Two distinct token metrics are tracked but serve different purposes:

- `total_tokens` — total tokens consumed across the entire optimization phase (training cost). Not used for selection.
- `val_avg_total_tokens` — per-sample average inference tokens from the mini val evaluation (inference cost). This is the value used for Pareto and `tokens_only` selection.

### Stage exclusion

Entries with `stage="seed_baseline"` (train-set diagnostic run) are always excluded from cross-stage selection. The val-set baselines (`seed_baseline_bare`, `seed_baseline_with_demos`, `stage="seed"`) compete normally.

---

## 7. Configuration Reference

All hyperparameters live in `TournamentConfig` (`tour_config.py`). Factory presets:

```python
from dca_tournaments.tour_config import TournamentConfig

cfg = TournamentConfig.default()       # Full pipeline defaults
cfg = TournamentConfig.quick_test()    # Fast smoke test settings
cfg = TournamentConfig.bootstrap_only() # Stage 1 only
cfg = TournamentConfig.full_pipeline()  # All 4 stages
```

Key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `stages` | `["seed","bootstrap","miprov2"]` | Ordered list of stages to run |
| `top_k` | `2` | Candidates propagated between stages |
| `student_model_name` | `azure/gpt-4.1-mini` | Inference + optimization LM |
| `seed_model_name` | `azure/gpt-5.2` | Seed-writing LM |
| `mini_train_size` | `120` | Training subset for optimization |
| `mini_val_size` | `41` | Val subset for intermediate scoring |
| `selection_criteria` | `score_only` | `score_only`, `pareto`, or `tokens_only` |
| `max_parallel_optimizations` | `1` | Concurrent optimizer subprocesses |
| `eval_num_threads` | `4` | Threads per evaluate_candidate() call |
| `instructions_only` | `True` | P1 mode: all guidance in docstring, field descs stay empty |

---

## 8. Output Artifacts

```
.cache/tournament/tournament_<timestamp>/
├── config.json                      # Full TournamentConfig snapshot
├── leaderboard.json                 # All candidates across all stages
├── leaderboard.xlsx                 # Excel export (one row per candidate)
├── leaderboard.txt                  # Human-readable table
├── summary.json                     # End-of-run statistics
├── pareto_plot.png                  # Score vs avg-tokens scatter plot
├── seed/
│   ├── seed_candidate_0/
│   │   ├── program.json             # Saved DSPy program
│   │   └── program.meta.json        # Scores, token stats, config metadata
│   └── ...
├── bootstrap/
│   └── ...
├── miprov2/
│   └── ...
└── gepa/
    └── ...

log/tournament/tournament_<timestamp>/
├── tournament.log                   # Full stdout+stderr mirror
├── stage_seed_snapshot.json         # Leaderboard state after seed stage
├── stage_bootstrap_snapshot.json    # Leaderboard state after bootstrap stage
└── ...
```

Each `program.meta.json` sidecar records:
- `val_total_score`, `val_cls_score`, `val_change_score`
- `per_category_scores` — per DCA category breakdown
- `val_avg_total_tokens`, `val_avg_input_tokens`, `val_avg_output_tokens`, `val_avg_reasoning_tokens`
- `val_eval_time_seconds`
- `n_demos`, `instruction_length`
- `stage`, `config_name`, `parent_path`, `timestamp`

---

## 9. Resume and Recovery

Pass `--resume-from <output_dir>` to continue an interrupted run:

```bash
python -m dca_tournaments.run_tournament \
    --resume-from .cache/tournament/tournament_20260428234355
```

Tournament execution is now fail-fast for optimizer stages: if any Bootstrap, MIPROv2, or GEPA subprocess run fails after its own retries, the tournament aborts, keeps the partial leaderboard, writes `summary_error.json`, and prints the exact `--resume-from` command.

Resume behavior uses stage completion markers instead of just checking whether a stage has any entries. That means a partially completed stage is resumed in place: already-evaluated candidates are reused from the leaderboard, and only the missing optimizer runs are executed.

Fully completed stages are skipped during resume. Incomplete stages are not skipped.

The current Azure CLI auth path is also hardened for Windows virtual environments: the code avoids the venv-local `az.BAT` shim that can fail with `'python' is not recognized as an internal or external command`.


---

## 10. Parallelism Architecture

Three levels of parallelism:

1. **Optimizer-level** (`--max-parallel`): Multiple optimizer subprocesses run concurrently. Each subprocess is a fully isolated Python interpreter — no shared DSPy ContextVar state, no nested thread pools. This prevents the 0-score bug caused by nested ThreadPoolExecutors interfering with the Azure client.

2. **Seed-level** (`--seed-parallel-evaluations`): Multiple seed candidate evaluations run concurrently inside `seed_prompt_generator.py`, each with its own LM instance.

3. **Eval-level** (`--eval-num-threads`): Within a single `evaluate_candidate()` call, samples are evaluated in parallel using a `ThreadPoolExecutor`. Each worker thread uses `dspy.settings.context(lm=lm)` to force the correct LM regardless of Python ContextVar propagation behavior.

Each concurrent `_evaluate_and_save` call creates its own `eval_lm` instance so that `lm.history` entries from concurrent calls never interleave and produce wrong token counts.

---

## 11. Token Tracking

Per-sample token stats are extracted by diffing `lm.history` before and after the evaluation pass:

```python
# tour_utils.py
history_before = len(lm.history)
# ... run all samples ...
new_entries = lm.history[history_before:]
# sum usage.prompt_tokens, usage.completion_tokens, usage.completion_tokens_details.reasoning_tokens
# divide by n_samples
```

This gives `val_avg_total_tokens`, `val_avg_input_tokens`, `val_avg_output_tokens`, `val_avg_reasoning_tokens` — all per-sample averages representing inference cost, not optimization cost.

Token data is required for `pareto` and `tokens_only` selection. A missing token value raises a hard `ValueError` immediately — there is no fallback.

---

## 12. Adapting to a New Use Case

To use this pipeline for a task other than DCA:

1. **Define a signature** in `signatures/` — a DSPy Signature subclass with your input/output fields and a task-description docstring.

2. **Write a metric** in `evaluation/` — a function `(example, prediction, trace=None) -> float` returning a score in [0, 1]. Also provide a `_with_feedback` variant returning `dspy.Prediction(score=..., feedback=...)` for GEPA.

3. **Update data loading** in `run_tournament.py` — replace `load_dca_dataset` and `stratified_split_dca_dataset` with your own data loading and splitting logic. Update the `mini_val` loop to call your program with the correct field names.

4. **Update `evaluate_candidate`** in `tour_utils.py` — change the `program(image_original=..., image_reimage=...)` call to match your signature's input field names.

5. Run with `--dry-run` first to verify configuration, then run normally.
