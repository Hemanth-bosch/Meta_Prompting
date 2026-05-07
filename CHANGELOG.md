# Changelog

All notable changes to this project are documented here.

---

## [0.2.0] — 2026-05-01

### Fixed

#### Azure CLI credential resolution on Windows virtual environments
- Added a PATH-sanitizing context manager in `config.py` that temporarily removes the active
  venv `Scripts/` directory before acquiring Azure CLI tokens.  This prevents the venv-local
  `az.BAT` shim from delegating to plain `python`, which broke on Windows shells where only
  the venv interpreter is on PATH.
- If no system `az` is found after PATH sanitization the code now fails early with a direct
  error instead of failing silently inside a model call.

#### Tournament stages now fail fast on optimizer errors
- Stage worker failures are no longer logged-and-ignored.
- Bootstrap, MIPROv2, and GEPA stages raise `StageExecutionError` when any optimizer run fails.
- `run_tournament.py` records the failed stage in `stage_execution_history`, writes
  `summary_error.json`, and prints a ready-to-run `--resume-from` command.
- A tournament can no longer exit `0` after losing a stage and quietly continuing with only
  seed results from later stages.

#### Partial-stage resume
- Added per-stage completion markers (`stage_completed.json`).
- Resume mode (`--resume-from`) now skips only stages that have a completion marker.
- Interrupted stages are resumed from where they left off; already-completed work items are
  reused from the leaderboard without re-optimization.

#### `extract_dca.py` abort and rerun path
- `OptimizationRunner` records a structured failure summary and raises `StrategyExecutionError`.
- `extract_dca.py` catches unrecoverable failures, writes `*_error_summary.json`, preserves
  the seed-state JSON, and prints a rerun command pointing at that seed state.

### Changed

#### Tournament pipeline defaults
- Default stage order is now `seed → bootstrap → miprov2 → gepa`.
- Test evaluation is opt-in (`run_test_eval=False`); uses explicit `test_eval_criteria="best_score"`.
- MIPROv2 inherits `student_model_name` when no stage-specific model override is provided.
- Resume now reloads `config.json` from the resumed tournament directory.

#### Tournament execution and reporting
- `run_tournament.py` records stage execution timing and writes richer audit snapshots.
- Final full-val re-evaluation is forced onto `config.student_model_name`.
- Leaderboard and `tour_utils.py` separate `score_leader` from `selection_leader` for all
  three selection modes (`score_only`, `tokens_only`, `pareto`).
- Excel export includes candidate timing columns and an optional `StageTimings` sheet.
- Diagnostic `seed_baseline` stage is consistently excluded from final summary, leader
  selection, and the Pareto plot.

#### Seed, stage, and optimizer timing metadata
- `seed_prompt_generator.py` writes timing metadata for baselines, fallback seeds,
  generated variants, and demo-enhanced variants.
- `stage_runner.py` records worker timing, compile duration, evaluation duration,
  optimization duration, and the evaluation model in candidate metadata.
- `optimization/opt.py` exposes a structured `run_summary` with per-strategy timing and
  best-strategy markers.

#### `extract_dca.py` hardening
- Model saving is always-on (no longer conditional).
- Added `--auto-load-compatible-cache` flag; automatic cache matching is now opt-in.
- Tighter cache matching and split-manifest fingerprinting.
- Dataset/split tags included in artifact file names.
- Run summary written on completion.

#### Judge consistency (`evaluation/dca_metric.py`)
- Prompt template extracted to a module-level constant.
- Normalized exact-text fast path before cache lookup.
- Stronger judge-cache key that includes prompt/rule identity.

#### Evaluation tooling
- Evaluation entrypoints use `resolve_default_split_file()` instead of hard-coding
  `.cache/dca_split_seed42.json`.
- Resolver checks both `.cache/` (legacy alias) and `.cache/splits/` (current location).
- `run_cache_grid_eval.py` excludes split manifests from model discovery by pattern.

### Removed

#### Azure auth preflight checks
- Removed startup Azure auth validation from `extract_dca.py` and `run_tournament.py`.
- The preflight gate was aborting runs before any work started and did not fix the underlying
  runtime credential path.
- Auth failures now surface at the actual call site where a model or token is needed.
- Direct API-key users are no longer forced through Azure CLI-oriented startup checks.

---

## [0.1.0] — 2026-04-01

Initial working implementation:
- DCA, JO, and YANASE use-case pipelines
- DSPy integration: Bootstrap, MIPROv2, GEPA, SIMBA optimizers
- Tournament multi-stage pipeline (`dca_tournaments/`)
- Azure OpenAI backend with Azure CLI and API-key auth modes
- Hybrid metric (linear and exponential token-efficiency penalty)
