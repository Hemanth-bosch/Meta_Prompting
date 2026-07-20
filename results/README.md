# DCA Results Archive

Raw experiment data behind the DCA results deck, plus a **master table mapping every plotted run/data-point to the exact hyperparameters used** to produce it.

All scores are **DCA = 0.60·classification + 0.40·change** on the val set (n=82) unless a row says `test` (n=93). Teacher / seed / reflection / MIPROv2-prompt models are `azure/gpt-5.2` in every tournament run.

## Contents

| File | What it is |
|------|-----------|
| `dca_experiments.xlsx` | Master spreadsheet — `val` / `test` / `task` sheets with per-run scores (human baselines, single-optimizer runs, tournaments). |
| `tournaments_leaderboards.txt` | Full per-stage leaderboard for each tournament run (Total / Cls / Chg / Demos / InstLen / AvgTok per candidate). |
| `excel/tournament_results.xlsx`, `tournament_top2.xlsx` | Tournament candidate exports (all candidates / top-2 per tournament). |
| `excel/singlerun_results.xlsx` | Standalone single-optimizer run exports (MIPROv2 / Bootstrap / GEPA). |
| `excel/combined_tournaments_single_run.xlsx` | Combined tournament + single-run comparison. |
| `eval_results/cross_model_20260415170019/` | The gpt-4.1-mini-optimized prompt (`20260415170019`) evaluated on gpt-4.1-mini / gpt-4.1 / gpt-5-mini / gpt-5.2 (cross-model transfer). |
| `eval_results/miprov2_light_20260402141458_gpt41mini_test.json` | Per-example test-set output for the MIPROv2-light optimized prompt. |
| `initial_classification_eval/` | Initial classification-only experiments (v1–v4 human prompts): metrics / confusion / failures / token / time plots + `eval_summary.csv`. |

## Shared configuration (every tournament run)

`use_cot=false` · split 60/20/20 seed 42 · `mini_train=120`, `mini_val=45` · `eval_temp=0.1`, `seed_gen_temp=1.0`, `seed_eval_temp=0.1` · seed: `max_context_examples=8`, `add_demos=true`, `reasoning=low` · bootstrap: `n16`, `rounds=2`, `labeled=3`, `bootstrapped=5` · miprov2: `medium`, `labeled=3`, `bootstrapped=5`, prompt-reasoning `low` · gepa: `medium`, `use_merge=true`, reflection-images `true`, detail `auto`, `max_images=10`, `minibatch=5`, candidate-selection `pareto`, `perfect=1.0`, `lowest=0.0` · `max_parallel=2`, `eval_num_threads=2` · instructions-only.

Metric variants: **DCA** (raw); **Hybrid-Linear** = `0.5·DCA + 0.5·max(0,1−tok/12000)`; **Hybrid-Exp** = `DCA·exp(−2·max(0,tok/2500−1))` (hybrid used only for optimizer selection, never final ranking).

## Master run → parameters table (tournament & seed-only runs)

| Run ID | Exp | Student | Stages | Selection | Seed ctx | Seed source | N seeds | top-k | Metric | Best DCA | Best stage |
|--------|-----|---------|--------|-----------|----------|-------------|:------:|:----:|--------|:--------:|-----------|
| 20260504234953 | 1a/2 | gpt-4.1 | seed→boot→mipro→gepa | Pareto | best | v1-MiDAS | 2 | 2 | DCA | **92.28%** | GEPA |
| 20260504235001 | 1a | gpt-4.1 | full | Best-Score | best | CR-DCA | 2 | 2 | DCA | 89.85% | GEPA |
| 20260507121046 | 2 | gpt-4.1 | full | Best-Score | best | CR-DCA | 3 | 2 | DCA | 92.22% | GEPA |
| 20260504233802 | 1b | gpt-4.1-mini | full | Pareto | best | CR-DCA | 2 | 2 | DCA | 86.48% | seed |
| 20260504233822 | 1b | gpt-4.1-mini | full | Best-Score | best | CR-DCA | 2 | 2 | DCA | 86.56% | MIPROv2 |
| 20260505172652 | 3 | gpt-4.1-mini | full | Pareto | best | CR-DCA | 2 | 2 | Hybrid-Linear (max=12000) | 87.41% | GEPA |
| 20260505153708 | 3 | gpt-4.1-mini | full | Best-Score | best | CR-DCA | 2 | 2 | Hybrid-Exp (T=2500, k=2) | 86.28% | GEPA |
| 20260506090909 | 4a | gpt-4.1 | seed only | — | **worst** | CR-DCA | 3 | — | DCA | 92.55% | seed |
| 20260506090846 | 4a | gpt-4.1 | seed only | — | best | CR-DCA | 3 | — | DCA | 87.29% | seed |
| 20260506120758 | 4b | gpt-4.1-mini | seed only | — | **worst** | CR-DCA | 3 | — | DCA | 86.40% | seed |
| 20260506120822 | 4b | gpt-4.1-mini | seed only | — | best | CR-DCA | 3 | — | DCA | 86.35% | seed |
| 20260507113839 | 5 | gpt-4.1 | full | Best-Score | worst | CR-DCA | 3 | 2 | DCA | (repeat of 1a) | GEPA |
| 20260507113820 | 5 | gpt-4.1 | full | Best-Score | worst | CR-DCA | 3 | 2 | DCA | (repeat of 1a) | GEPA |
| 20260505160645 | 6 | gpt-4.1-mini | seed only | — | worst | CR-DCA | 2 | — | DCA | — | seed |

*Seed source: "CR-DCA" = the current CR DCA signature docstring; "v1-MiDAS" = the MiDAS human prompt. "Seed ctx" = whether the seed LLM was shown best- or worst-performing train examples per class.*

## Standalone / manual / cross-model runs (`extract_dca.py` + human-prompt evals)

| Run ID | Type | Student | Config | Notes |
|--------|------|---------|--------|-------|
| 20260402141458 | Single optimizer | gpt-4.1-mini | MIPROv2 **light**, NoCoT, demos 6/6 | test-set + n=10 repeatability (`eval_results/miprov2_light_..._test.json`) |
| 20260415170019 | Single optimizer | gpt-4.1-mini (trained) | GEPA **medium**, CoT | cross-model transfer (val): mini 86.5% → gpt-4.1 90.0% → gpt-5-mini 88.5% → gpt-5.2 90.5% (`eval_results/cross_model_20260415170019/`) |
| 20260413204859 | Optimized v1 | gpt-4.1-mini | MIPROv2 | v1 prompt line in `dca_experiments.xlsx` (val) |
| 20260413221304 / 305 / 306 | Optimized v2 | gpt-4.1-mini | MIPROv2 | v2 prompt lines in `dca_experiments.xlsx` |
| 20260415135923 | "Best performance" opt | gpt-4.1-mini | — | perf-optimized prompt |
| 20260331173715 / 20260413191910 | Repeatability (n=10) | gpt-4.1-mini | — | mean ± 95% CI (`dca_experiments.xlsx` val) |
| INITIAL (v1–v4) | Human baselines | gpt-4.1-mini / 4.1 | manual prompts (v1 MiDAS, v2 CR, v3 +thinking, v4 agentic) | classification-only eval → `initial_classification_eval/` |

> For per-example scores and exact token counts of any run, open `dca_experiments.xlsx` (val/test sheets) or the matching `eval_results/*.json`; for per-stage tournament candidate breakdowns, see `tournaments_leaderboards.txt`.
