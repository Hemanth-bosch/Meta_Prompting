# Tournament Pipeline — Constitution, Specification & Design Reference

**Version:** 1.0 | **Date:** 2026-04-20  
**Audience:** Software engineers implementing or verifying the DCA optimization tournament pipeline  
**Repo root:** `C:\Users\BHS1RNG\OneDrive - Bosch Group\AIQ-212\Code\meta-prompting\`  
**Tournament package:** `tournament/`

---

## Table of Contents

1. [Constitutional Principles](#1-constitutional-principles)
2. [System Overview](#2-system-overview)
3. [Stage Map](#3-stage-map)
4. [Feature Specifications (Requirements 1–24)](#4-feature-specifications)
5. [Data Flow & Contracts](#5-data-flow--contracts)
6. [Configuration Reference](#6-configuration-reference)
7. [Module Design](#7-module-design)
8. [Implementation Status & Gap Analysis](#8-implementation-status--gap-analysis)
9. [Verification Checklist](#9-verification-checklist)

---

## 1. Constitutional Principles

These principles govern every design decision. All implementations MUST comply.

### P1 — Instructions-Only Signature
All LLM instructions live in the DSPy `Signature` class docstring. All field descriptions (`desc=`) are **always empty strings** (`""`). This applies in every stage (seed, bootstrap, MIPROv2, GEPA) and must never be violated. Config parameter `instructions_only: bool = True` controls this; when `False`, field descs may be populated but should remain consistent across stages.

### P2 — Config-Driven Everything
Every hyperparameter — parallelism limits, temperatures, model names, dataset sizes, demo counts, selection criteria, stage ordering — must be expressed as a named field in `TournamentConfig`. Nothing is hardcoded in runner code. CLI arguments map 1-to-1 to config fields via `TournamentConfig.from_cli_args()`.

### P3 — Modularity & Stage Composability
Each stage (seed/bootstrap/miprov2/gepa) is complete, independently runnable, and skippable. Stage ordering must be driven purely by `config.stages: List[str]`. If tomorrow a user wants order `["seed", "gepa", "bootstrap"]` that must work without code changes.

### P4 — Pareto-Aware Selection
Candidate selection across and within stages considers both **total val score** (maximize) and **total tokens** (minimize). Selection mode is configurable per run: `score_only`, `pareto` (default), or `tokens_only`. Token semantics = `total_tokens = prompt_tokens + completion_tokens` (i.e., input + output + reasoning = all).

### P5 — Parallel-Batch Execution
All optimizations and evaluations within a stage run as parallel batches with a configurable `max_parallel_optimizations` limit. Pattern: launch `min(N, max_parallel)` workers, wait for all to finish, then launch next batch.

### P6 — Reproducibility
Dataset splits use a fixed `split_seed`. Mini-train and mini-val are stratified (category-balanced). All candidates are saved to disk with `.meta.json` sidecars. Config is saved at run start. Leaderboard is persisted and resumable.

### P7 — Token Tracking
When `track_tokens=True` (default), every inference and optimization call is wrapped with `dspy.track_usage()`. Token counts are stored on every `CandidateEntry`. Pareto and `tokens_only` selection modes require `track_tokens=True`.

### P8 — Evaluation Temperature
All evaluation/validation runs use `eval_temperature = 0.1`. Optimization runs may use a different `optimization_temperature`. Seed generation uses `seed_generation_temperature = 1.0`. Seed evaluation uses `seed_eval_temperature = 0.1`.

**Design decision (accepted):** Intermediate stage evaluations (inside bootstrap/MIPROv2/GEPA workers) run at `optimization_temperature` (default: dspy ~0.0), NOT `eval_temperature`. This is intentional — intermediate scores are preliminary. The **final full-val re-evaluation** at `eval_temperature = 0.1` is the authoritative ranking pass.

---

## 2. System Overview

```
Raw DCA Dataset
      │
      ▼
Stratified Split (split_seed)
 ├── mini_train (size: mini_train_size, default 120)
 ├── mini_val   (size: mini_val_size,   default 40)
 ├── full_val
 └── test
      │
      ▼
┌─────────────────────────────────────────────────┐
│  OptimizationLeaderboard  (persistent, sorted)   │
└─────────────────────────────────────────────────┘
      │
      ▼
Stage 0: Seed Prompt Generation  ──► top_k candidates ─►┐
      │                                                   │
Stage 1: Bootstrap FewShot       ──► top_k candidates ─►┤ cross-stage
      │                                                   │ leaderboard
Stage 2: MIPROv2                 ──► top_k candidates ─►┤
      │                                                   │
Stage 3: GEPA                    ──► top_k candidates ─►┘
      │
      ▼
Final Re-eval (full val set, eval_temperature)
      │
      ▼
Top-K 2D Pareto Plot (score vs tokens)
      │
      ▼
Test Set Evaluation
```

---

## 3. Stage Map

| Stage | Name | DSPy Optimizer | Key Output |
|-------|------|---------------|-----------|
| 0 | Seed Prompt Generation | LLM meta-prompt (GPT-5.x) | N × instruction docstrings ± few shots |
| 1 | Bootstrap | `BootstrapFewShotWithRandomSearch` | labeled+bootstrapped demo programs |
| 2 | MIPROv2 | `MIPROv2` | grid of intensity × model × demo configs |
| 3 | GEPA | `GEPA` + multimodal reflection | instruction-optimized programs |

---

## 4. Feature Specifications

### REQ-01 — Stage-0 Baseline Run

**Requirement:**  
Run the current DCA signature as-is on the mini-train dataset. All instructions must be in the class docstring. Field descriptions must be empty. Use `seed_eval_temperature = 0.1` for this run.

**Implementation contract:**
- Load `DCAImageComparison` signature (from `signatures/dca.py`)
- Assert all field `desc` values are `""` before running
- Run `evaluate_candidate()` on `mini_train` at `temp = config.seed_eval_temperature`
- Store result as the baseline `CandidateEntry` with `stage="seed_baseline"`
- Save program + meta JSON to `{output_dir}/stage0_seed/baseline/`

**Config fields:** `seed_eval_temperature`, `mini_train_size`

---

### REQ-02 — Worst-Example Mining

**Requirement:**  
From the baseline run, select the **worst-performing** examples as few-shot failure demonstrations. Maximum 6 examples. Selection is balanced: exactly 1 worst example per branch/category. Final selection = top-6 worst overall, each from a different category (8 DCA categories, take worst 6).

**Implementation contract:**
- Call `find_worst_performing_examples(results, per_category_limit=1, total_limit=config.max_context_examples)`
- `per_category_limit=1` guarantees one per category
- `total_limit` (default 6) is the global cap
- Examples are sorted ascending by score; worst first
- Output: list of `(example, prediction, score)` tuples — max 6, all from distinct categories

**Config fields:** `max_context_examples: int = 6`

**DCA Categories (8 total):**  
`A1, A2, B1, B2, B3, B4, B5a, B5b`

---

### REQ-03 — LLM Seed Candidate Generation

**Requirement:**  
Pass the current signature instructions + worst-example context to a high-capability LLM (GPT-5.x). The LLM generates `n_seed_candidates` different instruction docstrings as seed candidates. Temperature = `seed_generation_temperature`. Model = `seed_model_name`. Reasoning effort = `seed_reasoning_effort`.

**Implementation contract:**
- Build a meta-prompt containing:
  1. The current signature docstring (the full instruction set)
  2. The worst-performing examples (formatted as failure demonstrations)
  3. Output format instructions: return a JSON list of `n_seed_candidates` instruction strings
- Call `dspy.LM(config.seed_model_name, temperature=config.seed_generation_temperature, reasoning_effort=config.seed_reasoning_effort)`
- Parse JSON response → list of instruction strings
- Model must support reasoning effort (e.g., `azure/gpt-5.2`, `azure/gpt-5.3` or equivalent)
- Each candidate = a new class docstring to replace `DCAImageComparison.__doc__`

**Config fields:**  
`seed_model_name: str = "azure/gpt-5.2"`  
`seed_generation_temperature: float = 1.0`  
`seed_reasoning_effort: Optional[str] = "low"`  (choices: `"low"`, `"medium"`, `None`)  
`n_seed_candidates: int = 3`

---

### REQ-04 — Dual-Mode Seed Evaluation

**Requirement:**  
Evaluate each of the N seed candidates in two modes:
1. **Bare** — instruction-only (no few-shot demos)
2. **With-demos** — instruction + the 6 worst-example few-shots from REQ-02

This produces 2×N evaluation runs total. Evaluation is on `mini_val`, at `seed_eval_temperature = 0.1`.

When `seed_add_demos_variant = False`, only bare mode is run (N runs total).

**Field description behavior:**
- `instructions_only = True` (default): field descs remain `""` throughout; instructions go only in docstring
- `instructions_only = False`: field descs may be set; implementations must decide per-stage whether to populate them — however they remain **fixed** (not mutated by MIPROv2/GEPA) once set; only the docstring/instructions portion is optimized

**Implementation contract:**
- For each candidate instruction string:
  - Create a copy of the signature with the new docstring
  - Variant 1 (bare): compile signature with 0 demos, evaluate on mini_val
  - Variant 2 (with_demos): add the 6 few-shot examples as `demos`, evaluate on mini_val
- Track tokens per variant if `track_tokens=True`
- Save each variant as a `CandidateEntry` to the leaderboard

**Config fields:**  
`seed_add_demos_variant: bool = True`  
`instructions_only: bool = True`

---

### REQ-05 — Top-K Selection After Stage 0

**Requirement:**  
After all 2×N (or N) evaluations, select the top-K best candidates using the configured selection criteria. Selection is from all evaluated variants (bare + with-demos combined).

**Selection modes:**
- `score_only`: rank by `val_total_score` descending, take top K
- `pareto` (default): compute Pareto front maximizing score + minimizing tokens; if front < K, fill remainder by score rank
- `tokens_only`: rank by `total_tokens` ascending, take top K

**Implementation contract:**
- Call `leaderboard.top_k(k=config.top_k, criteria=config.selection_criteria, stage_filter="seed")`
- These K candidates become the inputs to Stage 1
- If `cross_stage_selection=True`, selection pool includes all stages so far

**Config fields:**  
`top_k: int = 2`  
`selection_criteria: str = "pareto"`  (choices: `"score_only"`, `"pareto"`, `"tokens_only"`)  
`cross_stage_selection: bool = True`

---

### REQ-06 — Stage-0 Parallel Evaluation

**Requirement:**  
All seed eval variants (up to 2×N) are dispatched using `ThreadPoolExecutor`. Max concurrent workers = `min(total_variants, max(1, seed_parallel_evaluations))`. Batching pattern: launch batch of `seed_parallel_evaluations` workers, wait for all to complete, then launch next batch.

**Implementation contract:**
- `max_workers = min(len(eval_items), max(1, self.seed_parallel_evaluations))`
- `eval_items` = list of `(candidate_instruction, variant_mode, demos_or_None)` tuples
- Use `concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)`
- Submit all tasks, call `.result()` on each future
- Print: `Parallel evals: {seed_parallel_evaluations}`

**Config fields:**  
`seed_parallel_evaluations: int = 4`

**CLI flag:** `--seed-parallel-evaluations INT`

---

### REQ-07 — Configurable Selection Criteria (All Stages)

Same as REQ-05, applied globally. The `selection_criteria` field applies to ALL stages. Each stage performs evaluation on mini_val, then selects top-K using the same criteria. The selection pool for a given stage is:
- `cross_stage_selection=True`: all candidates across all stages so far  
- `cross_stage_selection=False`: only candidates from the current stage

**Config fields:** `selection_criteria`, `top_k`, `cross_stage_selection`

---

### REQ-08 — Stage-1 Bootstrap Configuration Grid

**Requirement:**  
Stage 1 runs `BootstrapFewShotWithRandomSearch` with a configurable grid of hyperparameter combinations.

**Default grid (6 configs):**

| name | n | max_rounds | max_labeled_demos | max_bootstrapped_demos |
|------|---|-----------|-------------------|----------------------|
| n16_d3b3 | 16 | 2 | 3 | 3 |
| n16_d5b3 | 16 | 2 | 5 | 3 |
| n32_d3b3 | 32 | 2 | 3 | 3 |
| n32_d5b3 | 32 | 2 | 5 | 3 |
| n16_d3b5 | 16 | 2 | 3 | 5 |
| n32_d3b5 | 32 | 2 | 3 | 5 |

Note: `traces` parameter (e.g., `(1,4)` or `(2,5)`) is mentioned in the original requirements. This refers to the trace filtering range. If `BootstrapFewShotWithRandomSearch` exposes a `traces` or `min_score`/`max_score` argument, it should be included and wired. **Currently unimplemented — marked as a gap.**

**Config fields:**
```python
bootstrap_configs: List[dict] = [
    {"name": "...", "n": int, "max_rounds": int, 
     "max_labeled_demos": int, "max_bootstrapped_demos": int}
]
bootstrap_model_name: Optional[str] = None  # falls back to student_model_name
bootstrap_num_threads: int = 4
```

**CLI flags:** `--bootstrap-model`, `--bootstrap-num-threads`

---

### REQ-09 — Stage-1 Parallel Execution

**Requirement:**  
Each combination of `(seed_candidate × bootstrap_config)` is a work item. All work items run in parallel batches with `max_parallel_optimizations` cap.

Example: 2 seed candidates × 4 bootstrap configs = 8 work items. With `max_parallel_optimizations=4`: launch 4, wait, launch 4 more.

**Implementation contract:**
- Build full work item list: `[(seed_entry, bs_config) for seed in top_k_seeds for bs_config in bootstrap_configs]`
- Batch: `chunks = [items[i:i+max_parallel] for i in range(0, len(items), max_parallel)]`
- For each chunk: `ThreadPoolExecutor(max_workers=len(chunk))`, submit all, collect all results before next chunk
- Each worker: `BootstrapFewShotWithRandomSearch(**bs_config, num_threads=bootstrap_num_threads).compile(student, trainset=mini_train, ...)`

**Config fields:** `max_parallel_optimizations: int = 4`

---

### REQ-10 — Mini Dataset Usage Across All Stages

**Requirement:**  
All optimization stages (1–3) use `mini_train` and `mini_val` for training and validation during optimization. The reasoning: mini datasets are stratified subsets that approximate the full distribution. Results must generalize — verified by the **final full-val re-evaluation** before test.

**Implementation contract:**
- `mini_train`: used as `trainset` for all optimizers (Bootstrap, MIPROv2, GEPA)
- `mini_val`: used for intermediate candidate evaluation after each stage
- `full_val`: used only in the final re-evaluation tournament at run end
- `test`: used only in the final test eval after full-val selection
- Sizes must be configurable; if `None`, the full split is used

**Config fields:**  
`mini_train_size: Optional[int] = 120`  
`mini_val_size: Optional[int] = 40`

GEPA may use different-sized subsets:  
`gepa_train_size: Optional[int] = None`  (default: use `mini_train_size`)  
`gepa_val_size: Optional[int] = None`    (default: use `mini_val_size`)

---

### REQ-11 — Per-Stage Model Configuration

Every stage has an independently configurable model name for the **optimizer/teacher** (not the student):

| Config field | Default | Used in |
|---|---|---|
| `student_model_name` | `"azure/gpt-4.1-mini"` | All inference (student) |
| `seed_model_name` | `"azure/gpt-5.2"` | Stage-0 meta-prompt generation |
| `bootstrap_model_name` | `None` → falls back to student | Stage-1 teacher |
| `miprov2_model_name` | `None` → config grid | Stage-2 (override per-config) |
| `gepa_model_name` | `None` → student | Stage-3 optimization |
| `reflection_model_name` | `None` → student | GEPA reflection LM |

**Temperature fields:**

| Config field | Default | Used in |
|---|---|---|
| `eval_temperature` | `0.1` | All mini-val / final re-eval |
| `seed_eval_temperature` | `0.1` | Stage-0 baseline + candidate eval |
| `seed_generation_temperature` | `1.0` | Stage-0 LLM meta-prompt call |
| `optimization_temperature` | `None` | All optimizer internal LM calls |

---

### REQ-12 — Post-Stage-1 Evaluation & Selection

**Requirement:**  
After Bootstrap finishes, evaluate all produced candidates on mini_val at `eval_temperature=0.1`. Then select top-K from the combined pool (all stage-0 and stage-1 candidates) if `cross_stage_selection=True`.

**Implementation contract:**
- Each bootstrap worker saves its compiled program + evaluates on `mini_val`
- Result stored as `CandidateEntry(stage="bootstrap", config_name=bs_config["name"], ...)`
- After all bootstrap workers finish: `leaderboard.top_k(k=top_k, criteria=selection_criteria)`
- If `cross_stage_selection=True`, the pool includes stage-0 entries as well
- Top-K entries are passed to Stage 2

**Config fields:** `top_k`, `selection_criteria`, `cross_stage_selection`, `eval_temperature`

---

### REQ-13 — Stage-2 MIPROv2 Configuration Grid

**Requirement:**  
Stage 2 runs a grid search over `intensities × models × demo_configs`.

**Default values:**

```python
miprov2_intensities: List[str] = ["light", "medium"]
miprov2_models: List[str] = ["azure/gpt-4.1-mini", "azure/gpt-4.1"]
miprov2_demo_configs: List[dict] = [
    {"max_labeled_demos": 3, "max_bootstrapped_demos": 3},
    {"max_labeled_demos": 6, "max_bootstrapped_demos": 6},
    {"max_labeled_demos": 3, "max_bootstrapped_demos": 5},
    {"max_labeled_demos": 5, "max_bootstrapped_demos": 3},
]
```

Work items = `top_k_inputs × intensities × models × demo_configs`.

**`miprov2_model_name` override:** When `miprov2_model_name` is set (non-None), it **overrides** the `miprov2_models` grid — every MIPROv2 work item uses that single model instead of iterating the grid. When `None` (default), the `miprov2_models` list is used.

**Field description behavior during MIPROv2:**
- `instructions_only=True`: MIPROv2 optimizes only the instruction docstring; does NOT populate field descs
- `instructions_only=False`: MIPROv2 may populate field descs — but original field descs (set in Stage 0) remain fixed as a baseline; only the docstring is varied
- `miprov2_clear_field_descs: bool = False` — if `True`, MIPROv2 will explicitly clear field descs before optimizing

**Config fields:**
```python
miprov2_intensities: List[str]
miprov2_models: List[str]
miprov2_demo_configs: List[dict]
miprov2_clear_field_descs: bool = False
miprov2_num_threads: int = 4
```

---

### REQ-14 — Stage-2 Parallel Execution

Same pattern as REQ-09 but for MIPROv2 work items. All `(input_candidate × intensity × model × demo_config)` combos are batched with `max_parallel_optimizations`.

**Config fields:** `max_parallel_optimizations`

---

### REQ-15 — Config-Driven, Fully Parameterized

All hyperparameters across all stages must be in `TournamentConfig`. No magic numbers or hardcoded values in any stage runner, generator, or utility code. Every field must have a documented type, default, and CLI flag.

**Audit checklist:**
- [x] All parallelism limits → config fields
- [x] All temperatures → config fields (including `seed_generation_temperature` CLI flag)
- [x] All model names → config fields
- [x] All dataset sizes → config fields
- [x] All demo counts and optimizer settings → config fields
- [x] Selection criteria → config field
- [x] Stage list → config field
- [x] Token tracking → config field
- [x] Split ratios (`train_ratio`, `val_ratio`, `test_ratio`) → config fields
- [x] Data paths (`pairs_dir`, `gt_dir`, `log_base_dir`) → config fields with CLI flags
- [x] `split_seed`, `eval_num_threads` → config fields with CLI flags

---

### REQ-16 — Stage Composability

`config.stages: List[str]` drives which stages run and in what order. Any subset and any permutation of `["seed", "bootstrap", "miprov2", "gepa"]` must work.

**Implementation contract:**
- `run_tournament.py:main()` iterates: `for stage_name in config.stages:`
- Each stage entry point receives: `(leaderboard, config, mini_train, mini_val, output_dir)`
- Stages are not allowed to have hard dependencies on each other's internal state beyond the leaderboard
- Resume mode must reconstruct leaderboard from saved candidates for any stage ordering

---

### REQ-17 — Post-Stage-2 Evaluation & Selection

Same as REQ-12 but for MIPROv2 output. After all MIPROv2 workers finish, evaluate all candidates on mini_val. Select top-K from all stages 0+1+2 (if `cross_stage_selection=True`) using configured criteria. These K candidates advance to GEPA.

**Config fields:** `top_k`, `selection_criteria`, `cross_stage_selection`, `eval_temperature`

---

### REQ-18 — Early Stopping (Optional)

**Requirement (optional, lower priority):**  
Investigate and optionally implement early stopping for Bootstrap and MIPROv2:

- **Bootstrap**: `BootstrapFewShotWithRandomSearch` accepts a `stop_at_score` parameter. Expose as `bootstrap_stop_at_score: Optional[float]`.
- **MIPROv2**: Check if `MIPROv2.compile()` supports early stopping via a score threshold. Investigate DSPy source at `.meta-prompting/Lib/site-packages/dspy/teleprompt/mipro_optimizer_v2.py`.
- **GEPA**: Check `dspy.teleprompt.gepa` for early stopping hooks.

**Implementation contract (if implemented):**
- `bootstrap_stop_at_score: Optional[float] = None` — pass to `BootstrapFewShotWithRandomSearch(stop_at_score=...)`
- For MIPROv2 and GEPA: document finding; if not natively supported, can use a callback/wrapper

**Status: NOT YET IMPLEMENTED. Gap acknowledged.**

---

### REQ-19 — Dataset Size & Generalization Validation

**Requirement:**  
Mini datasets must be representative enough that ranking on mini_val approximates ranking on full_val.

**Implementation contract:**
- Use `category_balanced_sample()` (stratified by DCA category) for all mini-set creation
- `split_seed` must be fixed and saved in config
- The **final re-evaluation** on full_val validates this: run every leaderboard candidate on full_val, re-rank
- If ranking changes significantly between mini_val and full_val, consider increasing `mini_val_size`
- Recommended minimum: `mini_train_size ≥ 80`, `mini_val_size ≥ 30`

**Config fields:**  
`mini_train_size`, `mini_val_size`, `split_seed`, `gepa_train_size`, `gepa_val_size`

---

### REQ-20 — Cross-Stage Top-K for GEPA Input

After Stage 2, select top-K from the global leaderboard (all stages). These become GEPA inputs. GEPA input count = `top_k` (same parameter).

---

### REQ-21 — Stage-3 GEPA — Instructions Only

**Requirement:**  
GEPA optimizes instruction docstrings only. Field descriptions remain `""`. This must be enforced in the GEPA stage setup. If DSPy GEPA internally tries to set field descs, they must be cleared after the run.

**Implementation contract:**
- Before passing any program to GEPA: assert all field descs are `""`
- After GEPA optimization: if `instructions_only=True`, explicitly clear any field descs that GEPA may have set on the output program
- The `DCAImageComparison` fields are defined with `desc=""` in `signatures/dca.py`; this must not change

---

### REQ-22 — Stage-3 GEPA Configuration Grid

**Requirement:**  
GEPA runs with configurable parameter combinations.

**Default config:**
```python
gepa_configs: List[dict] = [
    {"name": "gepa_medium", "auto": "medium", "use_merge": True, "track_stats": True}
]
```

**Key GEPA-specific parameters:**

| Config field | Default | Description |
|---|---|---|
| `gepa_train_size` | `None` → mini_train_size | Training set size for GEPA |
| `gepa_val_size` | `None` → mini_val_size | Val set size for GEPA |
| `gepa_reflection_minibatch_size` | `5` | Minibatch size for reflection step |
| `gepa_reflection_images` | `False` | Send images to reflection LM via vision API |
| `gepa_reflection_image_detail` | `"low"` | Azure vision detail level (`"low"`, `"high"`) |
| `gepa_reflection_max_images` | `6` | Max images per reflection call |
| `gepa_reflection_model` | `None` → reflection_model_name | Model for GEPA reflection LM |
| `gepa_num_threads` | `4` | Thread count for GEPA internal parallelism |

**Custom feedback metric:**  
GEPA uses `dca_comparison_metric_with_feedback` (a feedback-aware wrapper around the standard DCA metric). This is passed as `metric_with_feedback` to the `OptimizationRunner`.

**Multimodal reflection (from colleague's code — REQ-23):**  
See [Section 4, REQ-23](#req-23--gepa-multimodal-reflection-implementation) and [Module Design §7.7](#77-gepa-utilitiesgepa_utilspy).

---

### REQ-23 — GEPA Multimodal Reflection Implementation

**Requirement:**  
Integrate the multimodal GEPA reflection code provided by a colleague. This code:

1. Intercepts GEPA's `make_reflective_dataset` call
2. Extracts base64 image URLs from reflection input samples
3. Replaces image fields with text placeholders (always, to prevent token explosion)
4. If `gepa_reflection_images=True`: also injects the images as proper `image_url` content blocks into the reflection LM call via Azure Vision API

**Key components to implement/verify in `tournament/gepa_utils.py`:**

#### `_GepaImageStore` (thread-local)
```python
class _GepaImageStore(threading.local):
    def store(self, urls: List[str]) -> None: ...
    def get_recent_images(self) -> List[str]: ...
    # Note: do NOT call clear() destructively before reflecting — 
    # images must persist until after the reflection LM call reads them
```

#### `MultimodalReflectionLM`
```python
@dataclass
class MultimodalReflectionLM:
    base_lm: Any          # underlying dspy.LM
    detail: str = "low"   # image detail level: "low" | "high"
    max_images: int = 6   # cap on injected images
    
    def __call__(self, prompt: str):
        # 1. Get stored image URLs from _GepaImageStore
        # 2. If no images: call base_lm(prompt) directly
        # 3. Else: build multimodal content = [text_block] + [image_url_blocks...]
        #    and call base_lm(messages=[{"role": "user", "content": content}])
    
    def __getattr__(self, name):
        # Proxy all other attrs to base_lm
        return getattr(self.base_lm, name)
```

#### `make_image_storing_metric()`
```python
def make_image_storing_metric(base_metric):
    # Returns a wrapped metric that calls gepa_image_store.store(image_urls)
    # for each evaluation example, so images are available for reflection
```

#### `patch_gepa_reflective_dataset()`
```python
def patch_gepa_reflective_dataset(send_images: bool = False):
    # Monkey-patches DspyAdapter.make_reflective_dataset
    # The patch:
    # 1. Calls original method to get result
    # 2. Walks result dict/list recursively
    # 3. Finds fields containing "data:image/" strings
    # 4. Extracts data URLs with regex r"(data:image/[^'\"}\s]+)"
    # 5. Replaces field val with placeholder string
    # 6. Stores extracted URLs in _gepa_image_store.urls (thread-local sideband)
    # CRITICAL: Do NOT call gepa_image_store.clear() before reflection call
```

**Import path for patch:**  
`from dspy.teleprompt.gepa.gepa_utils import DspyAdapter`

---

### REQ-24 — Final Visualization & Selection

**Requirement:**  
At the very end of the pipeline:

1. Re-evaluate **every leaderboard candidate** on the **full val set** at `eval_temperature`
2. Select top-K using configured criteria from ALL stages
3. Generate a **2D scatter plot**: x-axis = total val score (%), y-axis = total tokens
4. Plot must show: all candidates (grey), Pareto front (highlighted), **top 3–4** labeled with name/stage
5. Save plot to `{output_dir}/pareto_plot.png`

**Implementation contract:**
- `plot_pareto_candidates(leaderboard, top_k=config.top_k, output_path=..., title=...)` in `tournament/utils.py`
- Plot shows ALL candidates; top-K are labeled and use distinct markers
- Pareto front is drawn as a step function connecting non-dominated candidates
- X-axis: val_total_score (0–100%), Y-axis: total_tokens (log scale optional)
- Function must handle the case where `track_tokens=False` (tokens unknown) — skip token axis or use N/A

**Config fields:** `top_k`, `selection_criteria`

**Status: Partially implemented. Current `plot_pareto_candidates` plots all entries but may not label specifically top 3–4 with markers. Verify and fix.**

---

## 5. Data Flow & Contracts

### 5.1 CandidateEntry Contract

Every candidate that enters the leaderboard must have these fields populated:

```python
@dataclass
class CandidateEntry:
    program_path: str         # path to saved DSPy program (.json)
    stage: str                # "seed" | "bootstrap" | "miprov2" | "gepa"
    config_name: str          # e.g. "bare_0", "with_demos_0", "n16_d3b3"
    parent_path: Optional[str]  # program_path of input seed (for stages 1+)
    
    val_total_score: float    # REQUIRED — score on mini_val [0.0..1.0]
    val_cls_score: float      # classification sub-score
    val_change_score: float   # change-detection sub-score
    per_category_scores: dict # {category: score} for all 8 DCA categories
    
    n_demos: int              # number of few-shot demos
    instruction_length: int   # len(docstring) in characters
    timestamp: str            # ISO datetime string
    
    # Token tracking (None if track_tokens=False)
    total_tokens: Optional[int]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    
    metadata: dict            # stage-specific extra info
```

### 5.2 Mini Dataset Contract

- Created once at run start via `create_mini_dataset(full_dataset, size, seed)`
- Stratified by `example.category` field
- If `size >= len(full_dataset)`, return full dataset without sampling
- Same `split_seed` across all calls for reproducibility

### 5.3 Leaderboard Contract

- Thread-safe: all `add()` and `top_k()` calls protected by `threading.Lock`
- Auto-saves after every `add()` call
- Sorted by `val_total_score` descending always
- `top_k(k, criteria, stage_filter=None)`:
  - `stage_filter=None` → all stages
  - `stage_filter="seed"` → only seed stage entries
  - Returns at most K entries; may return fewer if leaderboard has < K entries

### 5.4 Stage Runner Contract

Each stage function signature:
```python
def run_stage(
    stage_name: str,
    input_candidates: List[CandidateEntry],
    leaderboard: OptimizationLeaderboard,
    config: TournamentConfig,
    mini_train: List[Example],
    mini_val: List[Example],
    output_dir: Path,
) -> List[CandidateEntry]:  # new candidates added this stage
```

---

## 6. Configuration Reference

Complete `TournamentConfig` field reference:

### 6.1 Pipeline Control

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `stages` | `List[str]` | `["seed","bootstrap","miprov2"]` | Ordered list of stages to run |
| `top_k` | `int` | `2` | Candidates to keep per stage |
| `cross_stage_selection` | `bool` | `True` | Pool all stages for top-K selection |
| `selection_criteria` | `str` | `"pareto"` | `"score_only"` \| `"pareto"` \| `"tokens_only"` |
| `skip_final_tournament` | `bool` | `False` | Skip final full-val re-evaluation |
| `run_test_eval` | `bool` | `True` | Run test set evaluation at end |

### 6.2 Models

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `student_model_name` | `str` | `"azure/gpt-4.1-mini"` | Inference model for all stages |
| `seed_model_name` | `str` | `"azure/gpt-5.2"` | Meta-prompt LLM for Stage 0 |
| `reflection_model_name` | `Optional[str]` | `None` | GEPA reflection LM (falls back to student) |
| `bootstrap_model_name` | `Optional[str]` | `None` | Bootstrap teacher (falls back to student) |
| `miprov2_model_name` | `Optional[str]` | `None` | MIPROv2 override (falls back to grid) |
| `gepa_model_name` | `Optional[str]` | `None` | GEPA student override |
| `gepa_reflection_model` | `Optional[str]` | `None` | GEPA-specific reflection model |

### 6.3 Temperatures

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `eval_temperature` | `float` | `0.1` | All eval/val inference |
| `seed_eval_temperature` | `float` | `0.1` | Stage-0 candidate eval |
| `seed_generation_temperature` | `float` | `1.0` | Stage-0 meta-prompt LLM call |
| `optimization_temperature` | `Optional[float]` | `None` | Optimizer internal calls |

### 6.4 Dataset Sizes

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mini_train_size` | `Optional[int]` | `120` | Stratified mini train set size |
| `mini_val_size` | `Optional[int]` | `40` | Stratified mini val set size |
| `train_ratio` | `float` | `0.6` | Train split ratio |
| `val_ratio` | `float` | `0.2` | Val split ratio |
| `test_ratio` | `float` | `0.2` | Test split ratio |
| `split_seed` | `int` | `42` | Stratified split random seed |
| `gepa_train_size` | `Optional[int]` | `None` | GEPA train override (fallback: mini_train_size) |
| `gepa_val_size` | `Optional[int]` | `None` | GEPA val override (fallback: mini_val_size) |
| `split_seed` | `int` | `42` | Stratified split random seed |

### 6.5 Stage-0 Seed

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_seed_candidates` | `int` | `3` | LLM-generated instruction variants |
| `max_context_examples` | `int` | `6` | Max worst examples in meta-prompt |
| `seed_reasoning_effort` | `Optional[str]` | `"low"` | Reasoning effort for seed LLM |
| `seed_add_demos_variant` | `bool` | `True` | Also evaluate with-demos variant |
| `seed_parallel_evaluations` | `int` | `4` | Max concurrent Stage-0 eval workers |
| `seed_from_file` | `Optional[str]` | `None` | Skip generation, load seeds from JSON |
| `instructions_only` | `bool` | `True` | Keep all instructions in docstring only |

### 6.6 Stage-1 Bootstrap

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `bootstrap_configs` | `List[dict]` | 6-config grid | Bootstrap hyperparameter grid |
| `bootstrap_num_threads` | `int` | `4` | Threads per bootstrap worker |
| `bootstrap_stop_at_score` | `Optional[float]` | `None` | Early stop threshold (if supported) |

### 6.7 Stage-2 MIPROv2

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `miprov2_intensities` | `List[str]` | `["light","medium"]` | MIPROv2 auto= intensities |
| `miprov2_models` | `List[str]` | `["gpt-4.1-mini","gpt-4.1"]` | Model grid for MIPROv2 |
| `miprov2_demo_configs` | `List[dict]` | 4-entry grid | labeled/bootstrapped demo counts |
| `miprov2_clear_field_descs` | `bool` | `False` | Clear field descs before MIPROv2 |
| `miprov2_num_threads` | `int` | `4` | Threads per MIPROv2 worker |

### 6.8 Stage-3 GEPA

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gepa_configs` | `List[dict]` | `[{"name":"gepa_medium","auto":"medium",...}]` | GEPA config grid |
| `gepa_num_threads` | `int` | `4` | GEPA internal threads |
| `gepa_reflection_images` | `bool` | `False` | Send images to reflection LM |
| `gepa_reflection_image_detail` | `str` | `"low"` | Vision detail (`"low"` \| `"high"`) |
| `gepa_reflection_max_images` | `int` | `6` | Max images per reflection call |
| `gepa_reflection_model` | `Optional[str]` | `None` | Reflection LM override |
| `gepa_reflection_minibatch_size` | `int` | `5` | Reflection minibatch size |

### 6.9 Parallelism

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_parallel_optimizations` | `int` | `4` | Max concurrent optimizer workers (stages 1–3) |
| `seed_parallel_evaluations` | `int` | `4` | Max concurrent Stage-0 eval workers |
| `eval_num_threads` | `int` | `4` | Threads inside a single eval_candidate call |

### 6.10 Token Tracking

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `track_tokens` | `bool` | `True` | Enable dspy.track_usage() wrapping |

Token semantics: `total_tokens = prompt_tokens + completion_tokens = input + output + reasoning = ALL tokens`.

---

## 7. Module Design

### 7.1 `tournament/config.py`

- **Class:** `TournamentConfig` (dataclass)
- **Responsibility:** Single source of truth for all hyperparameters
- **Key methods:**
  - `from_cli_args(args) -> TournamentConfig` — maps argparse namespace to config
  - `get_stage_model(stage: str) -> str` — returns model name for given stage
  - `print_summary()` — human-readable config table
  - `save(path)` / `load(path)` — JSON persistence
  - Factory presets: `default()`, `quick_test()`, `bootstrap_only()`, `full_pipeline()`

### 7.2 `tournament/run_tournament.py`

- **Responsibility:** CLI entrypoint, orchestration, pipeline glue
- **Key functions:**
  - `build_parser() -> argparse.ArgumentParser`
  - `main()` — full pipeline execution
  - `_run_test_eval(config, val_entry, output_dir)` — subprocess test eval
- **Flow:**
  1. Parse args → `TournamentConfig.from_cli_args()`
  2. Setup dirs, logging, save config
  3. Load dataset, stratified split, create mini sets
  4. Load or initialize leaderboard
  5. For each stage in `config.stages`: run stage, update leaderboard, select top-K
  6. Final full-val re-eval tournament
  7. Generate summary, plot, save
  8. Test eval

### 7.3 `tournament/stage_runner.py`

- **Class:** `StageRunner`
- **Responsibility:** Run optimization for stages 1–3 (Bootstrap, MIPROv2, GEPA)
- **Key methods:**
  - `run_stage(stage_name, input_candidates, leaderboard, config, ...) -> List[CandidateEntry]`
  - `_run_bootstrap_stage()`
  - `_run_miprov2_stage()`
  - `_run_gepa_stage()`
  - `_build_*_configs()` — work item builders
  - `_load_student(candidate, config)` — load and configure DSPy program
  - `_evaluate_and_save(program, mini_val, config, ...)` — evaluate + save + create `CandidateEntry`

### 7.4 `tournament/seed_prompt_generator.py`

- **Class:** `SeedPromptGenerator`
- **Responsibility:** Stage-0 seed prompt generation pipeline
- **Key method:**
  - `generate(mini_train, mini_val, output_dir) -> List[CandidateEntry]`
- **Internal steps:**
  1. Baseline eval on mini_train
  2. Worst-example mining
  3. Meta-prompt construction
  4. LLM generation of N instruction docstrings
  5. Parallel eval: bare mode + with-demos mode
  6. Leaderboard insertion

### 7.5 `tournament/leaderboard.py`

- **Classes:** `CandidateEntry`, `OptimizationLeaderboard`
- **Responsibility:** Persistent, thread-safe candidate registry with Pareto selection
- **Key methods:**
  - `add(entry)` — thread-safe insert + auto-save
  - `top_k(k, criteria, stage_filter)` — ranked selection
  - `_pareto_top_k(entries, k)` — Pareto front computation
  - `save()` / `load()` / `from_file()`
  - `export_to_excel(path)` — pandas export

### 7.6 `tournament/utils.py`

- **Responsibility:** Shared utilities
- **Key functions:**
  - `evaluate_candidate(program, dataset, metric, ...)` — parallel per-example inference + scoring
  - `find_worst_performing_examples(results, per_category_limit, total_limit)` — worst-N selection
  - `create_mini_dataset(dataset, size, seed)` — stratified sampling
  - `token_usage_from_tracker(tracker)` — extract `{total, prompt, completion}_tokens` from DSPy tracker
  - `plot_pareto_candidates(leaderboard, top_k, output_path)` — 2D scatter plot
  - `TeeLogger` — stdout+stderr tee to log file
  - `generate_summary()`, `save_summary()`, `print_final_summary()`

### 7.7 GEPA Utilities (`tournament/gepa_utils.py`)

- **Responsibility:** Multimodal GEPA reflection support
- **Components:**
  - `_GepaImageStore(threading.local)` — thread-local image URL sideband
  - `MultimodalReflectionLM` — wraps `dspy.LM` to inject images into reflection calls
  - `make_image_storing_metric(base_metric)` — wraps metric to populate image store
  - `patch_gepa_reflective_dataset(send_images)` — monkey-patches `DspyAdapter.make_reflective_dataset`

**Critical implementation note:**  
Do NOT call `gepa_image_store.clear()` between `make_reflective_dataset` and the reflection LM call. The store must remain populated until `MultimodalReflectionLM.__call__()` has read and injected the images. Clearing before reflection = images lost.

### 7.8 `signatures/dca.py`

- **Class:** `DCAImageComparison(dspy.Signature)`
- **Constitution:** ALL instructions in class docstring. ALL field `desc=""`. This is fixed and immutable.
- **Fields (always with `desc=""`):**
  ```python
  image_original: Optional[dspy.Image] = dspy.InputField(desc="", default=None)
  image_reimage:  Optional[dspy.Image] = dspy.InputField(desc="", default=None)
  discard_image:  str                  = dspy.OutputField(desc="")
  changes:        str                  = dspy.OutputField(desc="")
  ```

---

## 8. Implementation Status & Gap Analysis

| REQ | Status | File(s) | Notes |
|-----|--------|---------|-------|
| REQ-01 | ✅ DONE | `seed_prompt_generator.py` | Baseline run on mini_train |
| REQ-02 | ✅ DONE | `utils.py`, `seed_prompt_generator.py` | `per_category_limit=1`, `total_limit=max_context_examples` |
| REQ-03 | ✅ DONE | `seed_prompt_generator.py` | GPT-5.x, temp=1.0, reasoning_effort configurable |
| REQ-04 | ✅ DONE | `seed_prompt_generator.py` | bare + with_demos variants; `seed_add_demos_variant` |
| REQ-05 | ✅ DONE | `leaderboard.py`, `seed_prompt_generator.py` | `score_only`, `pareto`, `tokens_only` |
| REQ-06 | ✅ DONE | `seed_prompt_generator.py`, `config.py`, `run_tournament.py` | `seed_parallel_evaluations` field + CLI flag |
| REQ-07 | ✅ DONE | `leaderboard.py`, `stage_runner.py` | `selection_criteria` applied in all stages |
| REQ-08 | ✅ DONE | `config.py`, `stage_runner.py` | 6-config grid; `traces` param **NOT wired** — gap |
| REQ-09 | ✅ DONE | `stage_runner.py` | Batch parallelism with `max_parallel_optimizations` |
| REQ-10 | ✅ DONE | `run_tournament.py`, `stage_runner.py` | mini_train/val for all stages |
| REQ-11 | ✅ DONE | `config.py`, `stage_runner.py` | Per-stage model + temp fields |
| REQ-12 | ✅ DONE | `stage_runner.py` | Post-bootstrap eval + cross-stage selection. Note: intermediate eval uses optimization_temperature (design decision — final re-eval at eval_temperature is authoritative) |
| REQ-13 | ✅ DONE | `config.py`, `stage_runner.py` | Grid over intensities × models × demo_configs |
| REQ-14 | ✅ DONE | `stage_runner.py` | Parallel batching for MIPROv2 |
| REQ-15 | ✅ DONE | `config.py` | All params in TournamentConfig; all have CLI flags |
| REQ-16 | ✅ DONE | `run_tournament.py` | `config.stages` drives stage loop |
| REQ-17 | ✅ DONE | `stage_runner.py` | Post-MIPROv2 eval + cross-stage selection |
| REQ-18 | ⚠️ PARTIAL | `config.py` | `bootstrap_stop_at_score` exists; MIPROv2/GEPA early stop **not investigated** |
| REQ-19 | ✅ DONE | `utils.py` | Stratified sampling; final full-val re-eval validates |
| REQ-20 | ✅ DONE | `run_tournament.py` | Cross-stage top-K feeds GEPA |
| REQ-21 | ✅ DONE | `stage_runner.py`, `signatures/dca.py` | GEPA enforces `desc=""` |
| REQ-22 | ✅ DONE | `config.py`, `stage_runner.py` | GEPA config grid with all params |
| REQ-23 | ✅ DONE | `tournament/gepa_utils.py` | Multimodal reflection + image store sideband |
| REQ-24 | ⚠️ PARTIAL | `utils.py` | `plot_pareto_candidates` exists but **top-3-4 labeling** may not distinguish — verify |

### Known Gaps

| ID | Description | Priority |
|----|-------------|----------|
| G1 | `traces` parameter in bootstrap configs not wired to `BootstrapFewShotWithRandomSearch` | Medium |
| ~~G2~~ | ~~MIPROv2 early stopping not investigated/implemented~~ | ~~Low (optional)~~ |
| ~~G3~~ | ~~GEPA early stopping not investigated/implemented~~ | ~~Low (optional)~~ |
| ~~G4~~ | ~~`plot_pareto_candidates` may not label top 3–4 with distinct markers~~ — Verified: labels top 4 | ~~Closed~~ |
| ~~G5~~ | ~~Resume mode GEPA stage not included in fallback reconstruction chain~~ | ~~Low~~ |

---

## 9. Verification Checklist

Use this checklist when verifying an implementation against all 24 requirements:

### Stage 0 (Seed)

- [ ] All field `desc=` values in `DCAImageComparison` are `""` before and after every run
- [ ] Baseline run uses `seed_eval_temperature` (not `eval_temperature`)
- [ ] Worst-example mining: `per_category_limit=1`, `total_limit=max_context_examples` (default 6)
- [ ] Worst examples span distinct DCA categories (verify: no two examples from same category)
- [ ] Meta-prompt includes: current docstring + failure examples + output format instructions (JSON list)
- [ ] LLM used = `seed_model_name`; temp = `seed_generation_temperature`; effort = `seed_reasoning_effort`
- [ ] Exactly `n_seed_candidates` instruction strings generated and parsed from JSON
- [ ] Both bare and with-demos variants evaluated (if `seed_add_demos_variant=True`)
- [ ] Eval on `mini_val` at `seed_eval_temperature` 
- [ ] Max workers = `min(total_variants, max(1, seed_parallel_evaluations))`
- [ ] All variants saved as `CandidateEntry` objects with `stage="seed"`
- [ ] Token counts populated on all entries (if `track_tokens=True`)
- [ ] Top-K selected using `selection_criteria` from seed stage entries

### Stage 1 (Bootstrap)

- [ ] Input = top-K from stage-0 (or all stages if `cross_stage_selection=True`)
- [ ] Work items = all `(seed × bootstrap_config)` combinations
- [ ] Batched parallel execution with `max_parallel_optimizations` cap
- [ ] Model for optimization = `bootstrap_model_name` (or `student_model_name` if None)
- [ ] `optimization_temperature` forwarded to bootstrap LM
- [ ] Post-bootstrap eval on `mini_val` at `eval_temperature`
- [ ] All candidates added to leaderboard with `stage="bootstrap"`
- [ ] Top-K selected from cross-stage pool

### Stage 2 (MIPROv2)

- [ ] Grid = `intensities × models × demo_configs` 
- [ ] Input = top-K from all previous stages
- [ ] Work items = `(seed × intensity × model × demo_config)` — all combinations
- [ ] Batched parallel with `max_parallel_optimizations`
- [ ] `instructions_only=True`: field descs remain `""` after MIPROv2
- [ ] Post-MIPROv2 eval on `mini_val` at `eval_temperature`
- [ ] All candidates added with `stage="miprov2"`
- [ ] Top-K selected from cross-stage pool

### Stage 3 (GEPA)

- [ ] Input = top-K from all previous stages
- [ ] `instructions_only=True`: field descs remain `""` after GEPA
- [ ] GEPA uses `dca_comparison_metric_with_feedback` as feedback metric
- [ ] `gepa_reflection_minibatch_size` passed to reflection setup
- [ ] If `gepa_reflection_images=True`: `MultimodalReflectionLM` wraps reflection LM
- [ ] `patch_gepa_reflective_dataset()` applied before GEPA runs
- [ ] `_GepaImageStore.clear()` is NOT called between `make_reflective_dataset` and reflection call
- [ ] `gepa_train_size` / `gepa_val_size` override mini dataset sizes for GEPA
- [ ] Parallel batching with `max_parallel_optimizations`
- [ ] Post-GEPA eval on `mini_val` at `eval_temperature`
- [ ] Candidates added with `stage="gepa"`

### Final Evaluation & Output

- [ ] Final tournament: re-evaluate ALL leaderboard candidates on full_val at `eval_temperature`
- [ ] Top-K selected from all stages combined after full-val re-rank
- [ ] 2D plot generated: score (x) vs tokens (y), all candidates shown
- [ ] Top 3–4 candidates labeled with distinct markers on plot
- [ ] Pareto front visually highlighted
- [ ] Plot saved to `{output_dir}/pareto_plot.png`
- [ ] Summary JSON saved
- [ ] Test eval launched (if `run_test_eval=True`)

### Config & Architecture

- [ ] Every hyperparameter in `TournamentConfig`
- [ ] Every field has a CLI flag in `build_parser()`
- [ ] `TournamentConfig.from_cli_args()` maps all CLI flags to config
- [ ] `config.stages` ordering is the only thing that drives stage execution order
- [ ] Token semantics: `total_tokens = prompt_tokens + completion_tokens` (all tokens)
- [ ] `track_tokens=False` does not break any non-token-dependent features
- [ ] All parallelism uses `ThreadPoolExecutor` with batch-wait pattern
- [ ] Leaderboard auto-saves after every insertion
- [ ] Run is resumable from last saved leaderboard state

---

*End of specification. For questions, refer to the codebase at `tournament/` or the implementation notes in `TOURNAMENT_GUIDE.md`.*
