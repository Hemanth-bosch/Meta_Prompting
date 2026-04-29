# Tournament Pipeline — Complete Guide

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [File Dependency Order](#2-file-dependency-order)
3. [Data Flow — End to End](#3-data-flow--end-to-end)
4. [File-by-File Deep Dive](#4-file-by-file-deep-dive)
5. [How to Run](#5-how-to-run)
6. [Temperature Strategy](#6-temperature-strategy)
7. [Directory & Artifact Layout](#7-directory--artifact-layout)
8. [Resume & Recovery](#8-resume--recovery)
9. [Overfitting Safeguards](#9-overfitting-safeguards)

---

## 1. Architecture Overview

The tournament is a **sequential cascaded prompt optimization pipeline** for the DCA (Document Change Analysis) task. Each stage takes the top-K winners from the previous stage as seeds, optimizes them using a different DSPy optimizer, and passes results forward.

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                         CLI / argparse                              │
  │                    (run_tournament.py)                               │
  └──────────────┬──────────────────────────────────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────┐
  │   TournamentConfig       │  ◄── config.py
  │   (all hyperparameters)  │
  └──────────┬───────────────┘
             │
             ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                    main() orchestration                           │
  │                                                                  │
  │  1. Load dataset ──► stratified split (train/val/test)           │
  │  2. Init leaderboard                                             │
  │  3. Stage 0: SeedPromptGenerator → list[CandidateEntry]          │
  │  4. For each subsequent stage:                                   │
  │       current_top[:top_k] ──► StageRunner.run_stage()            │
  │       ──► new list[CandidateEntry] → becomes current_top         │
  │  5. Final re-evaluation at eval_temperature                      │
  │  6. Test eval (subprocess)                                       │
  │  7. Save summary                                                 │
  └──────────────────────────────────────────────────────────────────┘
```

---

## 2. File Dependency Order

Files are ordered by dependency — each file only imports from files above it plus root project modules (which are never modified).

```
#   File                           Imports from                       Purpose
─── ──────────────────────────── ──────────────────────────────────── ──────────────────────────────
1   __init__.py                   config, leaderboard,                Package exports
                                  seed_prompt_generator,
                                  stage_runner, run_tournament

2   config.py                     (stdlib only)                       TournamentConfig dataclass

3   leaderboard.py                (stdlib only)                       CandidateEntry +
                                                                      OptimizationLeaderboard

4   utils.py                      ROOT config.create_lm               TeeLogger, evaluate_candidate,
                                  ROOT signatures/dca                 save/load, sampling, summary
                                  ROOT evaluation/dca_metric
                                  leaderboard

5   seed_prompt_generator.py      ROOT config.create_lm               Stage 0 — LLM seed generation
                                  ROOT signatures/dca
                                  ROOT evaluation/dca_metric
                                  leaderboard, utils

6   stage_runner.py               ROOT config.create_lm               Stages 1-4 — optimizer runner
                                  ROOT signatures/dca
                                  ROOT evaluation/dca_metric
                                  config, leaderboard, utils

7   run_tournament.py             ALL above                           CLI entrypoint + orchestration
                                  ROOT utils (data loading)
                                  ROOT evaluation/dca_metric
```

**"ROOT" prefix** = imports from the project root (`config.py`, `utils/`, `evaluation/`, `signatures/`).
The tournament **never modifies** any root file — read-only dependency.

---

## 3. Data Flow — End to End

### 3.1 Phase 1: Configuration

```
CLI args (argparse)
    │
    ▼
TournamentConfig.from_cli_args(args)         ◄── config.py
    │
    ├── Validates stage names against VALID_STAGES
    ├── Generates timestamp: "20260413225829"
    ├── Resolves output_dir: ".cache/tournament/tournament_20260413225829"
    └── Resolves log_dir:    "log/tournament/tournament_20260413225829"
    │
    ▼
config.save()  →  .cache/tournament/tournament_<timestamp>/config.json   (output_dir — persisted immediately)
config.save()  →  log/tournament/tournament_<timestamp>/config.json      (log_dir)
```

### 3.2 Phase 2: Data Loading

```
load_dca_dataset(pairs_dir, gt_dir)          ◄── ROOT utils/data_loading.py
    │
    ▼
list[dspy.Example]  (each has: image_original, image_reimage,
                      expected_discard_image, expected_changes,
                      expected_human_label_category, pair_name)
    │
    ▼
stratified_split_dca_dataset(dataset, seed=42)  ◄── ROOT utils/splitting.py
    │
    ├── train_set (60%)  ── used by optimizer.compile()
    ├── val_set   (20%)  ── used by evaluate_candidate() at every stage
    └── test_set  (20%)  ── NEVER touched until final winner eval
    │
    ▼
split_manifest.json  →  saved to output_dir + .cache/dca_split_seed42.json
```

### 3.3 Phase 3: Stage 0 — Seed Generation

```
SeedPromptGenerator.generate(train_set, val_set, metric, leaderboard, output_dir, log_dir)
    │
    ├── Step 1: Create baseline Predict(DCAImageComparison)
    │           Configure student LM (azure/gpt-4.1-mini, temp=0.0)
    │           find_worst_performing_examples(baseline, train_set) → 6 failures
    │
    ├── Step 2: Build meta-prompt with:
    │           - Current DCAImageComparison instruction
    │           - DCA category reference (A1-B5b)
    │           - Formatted failure examples
    │
    ├── Step 3: Call seed LM (azure/gpt-5.2, temp=1.0) 3 times
    │           Parse JSON response → {instruction, field_descs}
    │
    ├── Step 4: For each candidate:
    │           module = Predict(DCAImageComparison)
    │           Inject via with_instructions() + with_updated_fields()
    │           evaluate_candidate(module, val_set, metric)
    │           save_candidate(module, path, metadata)
    │           CandidateEntry → leaderboard.add(entry)
    │           │
    │           ▼
    │           Saved files:
    │             stage_0_seed/seed_candidate_0.json      (DSPy state)
    │             stage_0_seed/seed_candidate_0.meta.json (metadata sidecar)
    │
    └── Return: list[CandidateEntry]  →  becomes initial current_top
```

### 3.4 Phase 4: Stages 1-3 — Optimization Loop

```
current_top = seed_entries  (list[CandidateEntry])

for stage in ["bootstrap", "miprov2", "gepa", "simba"]:
    │
    ├── current_top = sorted(current_top)[:top_k]     ◄── only best K flow forward
    │
    ├── runner.run_stage(stage, current_top)           ◄── stage_runner.py
    │       │
    │       ├── stage_dir = output_dir / "stage_1_bootstrap"
    │       │
    │       ├── For each (seed_entry × optimizer_config):
    │       │       │
    │       │       ├── _load_student(seed_entry)
    │       │       │     load_candidate(program_path) → deepcopy
    │       │       │
    │       │       ├── Create optimizer:
    │       │       │     bootstrap → BootstrapFewShotWithRandomSearch
    │       │       │     miprov2   → MIPROv2 (reconfigure LM if different model)
    │       │       │     gepa      → GEPA(metric=metric_WITH_FEEDBACK)
    │       │       │     simba     → SIMBA (NO valset)
    │       │       │
    │       │       ├── compiled = optimizer.compile(student, trainset, valset)
    │       │       │
    │       │       └── _evaluate_and_save(compiled)
    │       │             ├── evaluate_candidate(compiled, val_set)
    │       │             ├── save_candidate(compiled, path, metadata)
    │       │             └── leaderboard.add(CandidateEntry)
    │       │
    │       └── Returns: leaderboard.top_k(top_k, stage=stage)
    │
    └── current_top = new_entries  (feeds next stage)
```

**Key data flowing between stages:**

```
CandidateEntry  (the "currency" between stages)
  ├── program_path       →  .json path on disk (next stage loads this)
  ├── stage              →  "seed" / "bootstrap" / "miprov2" / etc.
  ├── config_name        →  human-readable label
  ├── parent_path        →  which seed produced this (lineage tracking)
  ├── val_total_score    →  0.6*cls + 0.4*change  (ranking criterion)
  ├── val_cls_score      →  classification accuracy
  ├── val_change_score   →  change description quality
  ├── per_category_scores → {A1: 0.95, A2: 0.88, ...}
  ├── n_demos            →  demo count in the optimized program
  ├── instruction_length →  char count of instruction (prompt bloat detector)
  └── metadata           →  {model, stage, parent, eval_result, ...}
```

### 3.5 Phase 5: Final Re-evaluation

```
leaderboard.top_k(top_k * 2)
    │
    For each entry:
    │
    ├── evaluate_candidate_at_temperature(
    │       program_path, val_set, metric,
    │       temperature=0.1,
    │       model_name=entry.metadata["model"]   ◄── entry-specific model
    │   )
    │   Internally:
    │     1. create_lm(model, temp=0.1, cache=False)
    │     2. dspy.configure(lm=eval_lm)
    │     3. program = load_candidate(path)
    │     4. program.lm = eval_lm    ← prevents saved LM from overriding
    │     5. evaluate_candidate(program, val_set)
    │     6. Restore original LM in finally block
    │
    ├── Update entry scores in-place
    └── leaderboard.entries.sort() → leaderboard.save()
```

### 3.6 Phase 6: Test Evaluation

```
winner = leaderboard.best()
    │
    ▼
_run_test_eval(config, winner, split_path)
    │
    ├── subprocess.run([python, eval_script,
    │     --model-path   winner.program_path,
    │     --model-name   winner.metadata["model"],
    │     --split-file   .cache/dca_split_seed42.json,
    │     --temperature  0.1,
    │     --which-set    test,
    │     --output-dir   output_dir/test_eval/,
    │     --no-cot])                                  ◄── (if use_cot=False)
    │
    └── Parse: eval_results_*_test.json
          max(results, key=avg_total_score)
          Return: {avg_total_score, avg_cls_score, avg_change_score, ...}
```

### 3.7 Phase 7: Summary

```
generate_summary(config, leaderboard, stages_run, total_time, test_eval_result)
    │
    ├── Builds dict: tournament_id, timestamps, config, per_stage, global_best,
    │                test_eval, all_candidates_ranked
    │
    ├── save_summary() → output_dir/summary.json + log_dir/summary.json
    └── print_final_summary()
```

---

## 4. File-by-File Deep Dive

### 4.1 `config.py` — TournamentConfig

**Data in:** CLI args via `from_cli_args()`, or factory presets.
**Data out:** The config object is passed to every other component.

```
TournamentConfig
  ├──→ run_tournament.py   (reads all fields)
  ├──→ stage_runner.py     (reads optimizer params, temps, models, dirs)
  ├──→ seed_prompt_gen.py  (reads seed/student model, temp, n_candidates)
  └──→ summary generation  (serialized into summary.json via to_dict())
```

**Key method:** `print_summary()` — also computes estimated run count:
- seed: `n_seed_candidates`
- bootstrap: `top_k × len(bootstrap_configs)`
- miprov2: `top_k × len(intensities) × len(models)`
- gepa/simba: `top_k`

### 4.2 `leaderboard.py` — CandidateEntry + OptimizationLeaderboard

**Data in:** `CandidateEntry` objects from seed generator, stage runner, and run_tournament.
**Data out:** `top_k()` queries feed the next stage. `get_stage_summary()` feeds summaries.

```
                add(entry)
                    │
 seed_prompt_gen ──►│
 stage_runner    ──►│──► entries (sorted desc by val_total_score)
 run_tournament  ──►│       │
                    │       ├──► top_k(k, stage) → feeds next stage as seeds
                    │       ├──► best()           → winner for test eval
                    │       ├──► get_stage_summary() → per-stage stats
                    │       ├──► get_full_summary()  → global stats
                    │       └──► save() → leaderboard.json (auto on every add)
                    │
                    └──► from_file() ← resume mode
```

**Persistence:** Auto-saves to `leaderboard.json` on every `add()`. This is the crash-recovery checkpoint.

### 4.3 `utils.py` — Shared Utilities

Core functions and their data contracts:

| Function | IN | OUT |
|---|---|---|
| `evaluate_candidate(program, val_set, metric)` | dspy.Module, examples, metric fn | `{total_score, cls_score, change_score, per_category_scores, n_evaluated}` |
| `evaluate_candidate_at_temperature(path, val_set, metric, temp, model)` | file path, temp, model name | Same dict as above |
| `save_candidate(program, path, metadata)` | dspy.Module, file path, dict | `.json` + `.meta.json` files on disk |
| `load_candidate(path, use_cot)` | file path, bool | dspy.Module |
| `find_worst_performing_examples(program, dataset, metric, n)` | Module, train data | `list[{example, prediction, score, category}]` |
| `generate_summary(config, leaderboard, ...)` | All runtime info | summary dict |

### 4.4 `seed_prompt_generator.py` — Stage 0

```
IN:  train_set, val_set, metric, leaderboard, output_dir, log_dir
     │
     ├── ① student LM → baseline → find failures on train set
     ├── ② build meta-prompt (instruction + categories + failures)
     ├── ③ seed LM at temp=1.0 × N calls → parsed instructions
     ├── ④ inject instructions → evaluate on val_set → save → leaderboard.add()
     │
OUT: list[CandidateEntry]
```

### 4.5 `stage_runner.py` — Stages 1-4

```
IN:  config, train_set, val_set, metric, metric_with_feedback, leaderboard
     │
     Per run_stage(stage, seed_entries):
     ├── expand configs → list[dict]
     ├── for (seed × config): load → deepcopy → optimizer.compile → eval → save
     │     leaderboard.add(entry)
     │
OUT: leaderboard.top_k(top_k, stage=stage)
```

**GEPA-specific:** uses `metric_with_feedback` (not regular metric) — returns structured feedback for GEPA's reflection loop.

**SIMBA-specific:** does NOT pass `valset` to `compile()`.

### 4.6 `run_tournament.py` — Orchestrator

Connects everything:

```
CLI → Config → Dataset → Split → Leaderboard
  → Stage 0 (SeedPromptGenerator)
  → Stage loop (StageRunner per stage)
  → Final re-eval (evaluate_candidate_at_temperature)
  → Test eval (subprocess)
  → Summary (generate_summary + save_summary)
```

---

## 5. How to Run

| Scenario | Command |
|---|---|
| **Dry run** | `python -m tournament.run_tournament --dry-run` |
| **Default pipeline** | `python -m tournament.run_tournament` |
| **Quick smoke test** | `python -m tournament.run_tournament --stages seed --n-seed-candidates 1 --top-k 1 --skip-test-eval` |
| **Bootstrap only from file** | `python -m tournament.run_tournament --stages bootstrap --seed-from-file .cache/my_seed.json --top-k 1` |
| **Full with GEPA** | `python -m tournament.run_tournament --stages seed,bootstrap,miprov2,gepa` |
| **Custom models** | `python -m tournament.run_tournament --student-model azure/gpt-4.1 --seed-model azure/gpt-4.1` |
| **Resume interrupted** | `python -m tournament.run_tournament --resume-from .cache/tournament/tournament_20260413225829` |
| **MIPROv2 light only** | `python -m tournament.run_tournament --stages miprov2 --seed-from-file .cache/my_seed.json --miprov2-auto light --miprov2-models azure/gpt-4.1-mini` |

---

## 6. Temperature Strategy

| Where | Temp | Set by | Why |
|---|---|---|---|
| Student LM during optimization | `0.0` (dspy default) | `optimization_temperature=None` → no kwarg → dspy.LM default | Deterministic traces |
| Seed generation (Stage 0) | `1.0` | `seed_generation_temperature=1.0` → `create_lm(temp=1.0)` | Diverse prompts from strong LLM |
| MIPROv2 proposal generation | `1.0` | DSPy hardcodes via `GroundedProposer(init_temperature=1.0)` | Cannot override |
| Bootstrap retry rounds (>0) | `1.0` | DSPy hardcodes via `lm.copy(temperature=1.0)` | Cannot override |
| Final tournament re-eval | `0.1` | `eval_temperature=0.1` → `evaluate_candidate_at_temperature()` | Realistic scoring |
| Test set evaluation | `0.1` | Passed as `--temperature 0.1` to subprocess | Matches eval convention |

**Critical:** `evaluate_candidate_at_temperature()` forces `program.lm = eval_lm` after loading to prevent the saved program's LM from silently overriding the desired temperature.

---

## 7. Directory & Artifact Layout

```
.cache/tournament/tournament_<timestamp>/
├── config.json                       ◄── TournamentConfig snapshot
├── split_manifest.json               ◄── train/val/test pair names
├── leaderboard.json                  ◄── crash-recovery checkpoint
├── stage_0_seed/
│   ├── seed_candidate_0.json         ◄── DSPy state (instruction + demos)
│   └── seed_candidate_0.meta.json    ◄── metadata sidecar
├── stage_1_bootstrap/
│   ├── bootstrap_balanced_from_seed_candidate_0.json
│   └── bootstrap_balanced_from_seed_candidate_0.meta.json
├── stage_2_miprov2/
│   └── miprov2_light_gpt-4.1-mini_from_*.json
├── stage_3_gepa/  (if enabled)
│   └── gepa_medium_from_*.json
├── test_eval/
│   └── eval_results_*_test.json
└── summary.json

log/tournament/tournament_<timestamp>/
├── config.json
├── tournament_main.log               ◄── full stdout+stderr capture
├── stage_bootstrap.log               ◄── per-stage log
├── stage_miprov2.log
└── summary.json
```

### Metadata Sidecar (`.meta.json`)

Every saved `.json` program has a companion `.meta.json`:

```json
{
  "stage": "bootstrap",
  "config_name": "bootstrap_label_heavy_from_seed_candidate_0",
  "parent": ".cache/tournament/.../seed_candidate_0.json",
  "model": "azure/gpt-4.1-mini",
  "timestamp": "2026-04-13T22:58:29",
  "eval_result": {
    "total_score": 0.7892,
    "cls_score": 0.8123,
    "change_score": 0.7456,
    "per_category_scores": {"A1": 0.95, "A2": 0.88, "B1": 0.90, "...": "..."},
    "n_evaluated": 30
  },
  "name": "label_heavy",
  "max_labeled_demos": 6,
  "max_bootstrapped_demos": 4,
  "saved_at": "2026-04-13T22:58:29"
}
```

---

## 8. Resume & Recovery

The leaderboard auto-saves on every `add()`. If the process crashes:

1. Re-run with `--resume-from .cache/tournament/tournament_<ts>`
2. The pipeline loads the existing `leaderboard.json`
3. For each stage, checks `leaderboard.stage_entries(stage)`:
   - If entries exist → **skip** (stage already completed)
   - If no entries → **run** (stage not yet done)
4. `current_top` is reconstructed from the latest completed stage's top-K
5. Continues from first incomplete stage

---

## 9. Overfitting Safeguards

| Safeguard | Implementation | File |
|---|---|---|
| **Per-category tracking** | `evaluate_candidate()` groups by A1-B5b, stores scores in every `CandidateEntry` | utils.py |
| **Demo count caps** | Configurable per stage: bootstrap 4-8, MIPROv2 6+6, SIMBA 4 | config.py |
| **Test set isolation** | test_set is NEVER passed to any optimizer or eval — only used once for final winner | run_tournament.py |
| **Val-only optimization** | All `optimizer.compile()` and `evaluate_candidate()` use val_set only | stage_runner.py |
| **Instruction length tracking** | `CandidateEntry.instruction_length` detects prompt bloat across stages | leaderboard.py |
| **Zero root file changes** | Tournament only imports from root modules — never modifies them | all files |
| **Leaderboard lineage** | Every entry has `parent_path` → trace any winner back to its original seed | leaderboard.py |

---

## Quick Reference: Data Objects

| Object | Created in | Consumed by | Purpose |
|---|---|---|---|
| `TournamentConfig` | config.py | everywhere | All hyperparameters |
| `CandidateEntry` | leaderboard.py | stage_runner, run_tournament | One optimized program + its scores |
| `OptimizationLeaderboard` | leaderboard.py | stage_runner, run_tournament | Sorted collection of all candidates |
| `TeeLogger` | utils.py | run_tournament, stage_runner | Dual stdout/file logging |
| `eval_result` dict | utils.py `evaluate_candidate()` | saved in metadata + entry | `{total_score, cls_score, change_score, per_category_scores}` |
| `summary` dict | utils.py `generate_summary()` | saved to disk, printed | End-of-run report |
| `split_manifest` dict | run_tournament.py | test eval subprocess | Train/val/test pair names |
