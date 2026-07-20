# DCA scatter-plus-mean plots

DCA total score with **individual run points + the mean marker only** (no box, no median), on the **val set**.

## Pipeline

```bash
python results/plots/build_datapoints.py    # xlsx + tournament excel -> dca_datapoints.csv
python results/plots/plot_dca_scatter.py    # dca_datapoints.csv -> figures/*.png
```

- **`dca_datapoints.csv`** — one row per individual run: `source_row, optimizer, prompt_version, prompt_type, model, seed_source, cot, score`. `source_row` traces each point back to its source (spreadsheet row or `tour:<run-id>`), so every point is auditable.

## The 6 figures (`figures/`)

`2 inference models × 3 optimizer seed-sources`. Every plot uses the same x-axis:

```
MiDAS pipeline (v1, non-DSPy) | v1 human (DSPy) | v2 human (DSPy) | bootstrap | miprov2 | gepa | tournament
```

| File | Inference model | Optimizer seed |
|------|-----------------|----------------|
| `dca_gpt41mini_crseed.png` | gpt-4.1-mini | CR-default signature |
| `dca_gpt41mini_v1seed.png` | gpt-4.1-mini | v1 (MiDAS) prompt |
| `dca_gpt41mini_v2seed.png` | gpt-4.1-mini | v2 (CR) prompt |
| `dca_gpt41_crseed.png` | gpt-4.1 | CR-default signature |
| `dca_gpt41_v1seed.png` | gpt-4.1 | v1 (MiDAS) prompt |
| `dca_gpt41_v2seed.png` | gpt-4.1 | v2 (CR) prompt |

**Markers:** ● circle = No-CoT run, ▲ triangle = CoT run, ◆ green diamond = mean. Categories with no matching run stay on the axis and show a **"no run yet"** box.

## How the columns are sourced

- **MiDAS pipeline (v1, non-DSPy)** = val rows labelled `"using midas like image passing prompt"` (human prompt run through the *original* MiDAS pipeline, no DSPy).
- **v1 / v2 human (DSPy)** = rows labelled `"using DSPY style signature"` (same human prompt reorganised into a DSPy signature — changes performance). Where a model has no explicit DSPy-labelled row (gpt-4.1), the single unlabelled human run is used.
- **Optimizer (bootstrap / miprov2 / gepa)** — from `dca_experiments.xlsx`; seed source from the "Type of initial seed prompt" col (`signature`→CR-default, `v1-prompt`, `v2-prompt`), GEPA carries its seed in the notes col (`v1`/`v2`). Col-7 is trusted as the total only when cls + chg are present.
- **Tournament** — best-score per full tournament run (`1a/1b/2/3/5`) from `combined_tournaments_single_run.xlsx`, which carries the run-IDs. Seed source per run-ID is taken from the **deck (slides 29 & 37)**: run `20260504234953` used the **v1-MiDAS seed**; all other tournaments used CR-default. The "Only seeds" experiments (`4a/4b`) are seed-only, not tournaments, and are excluded.

## Tournament / seed data availability

| | gpt-4.1-mini | gpt-4.1 |
|---|---|---|
| tournament, CR-default | 5 runs | 4 runs |
| tournament, v1-seed | none | **1 run (0.9228, run 20260504234953)** |
| tournament, v2-seed | none | none |
| bootstrap / gepa / miprov2, v1 or v2 seed | miprov2 (v1: 18, v2: 28), gepa (v1/v2: 1 each) | none |

Everything else shows "no run yet". Nearly all val inference is temp 0.1, so a temperature slice isn't supported by this data.

## Adding a plot

Add a `scatter_mean([...], title, outfile)` call in `plot_dca_scatter.py`; each group is `(label, sub(optimizer=..., model=..., seed_source=..., cot=...))`.
