# DCA scatter-plus-mean plots

DCA total score with **individual run points + the mean marker only** (no box, no median), on the **val set**.

## Pipeline

```bash
python results/plots/build_datapoints.py    # dca_experiments.xlsx + tournament_top2.xlsx -> dca_datapoints.csv
python results/plots/plot_dca_scatter.py    # dca_datapoints.csv -> figures/*.png
```

- **`dca_datapoints.csv`** — one row per individual run: `source_row, optimizer, prompt_version, prompt_type, model, seed_source, cot, score`. `source_row` traces each point back to the spreadsheet, so every point is auditable.

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

- **MiDAS pipeline (v1, non-DSPy)** = val rows labelled `"using midas like image passing prompt"` (the human prompt run through the *original* MiDAS pipeline, no DSPy).
- **v1 / v2 human (DSPy)** = rows labelled `"using DSPY style signature"` (same human prompt, but reorganised into a DSPy signature — this changes performance). Where a model has no explicit DSPy-labelled row (gpt-4.1), the single unlabelled human run is used.
- **Optimizer seed source** from col "Type of initial seed prompt" (`signature`→CR-default, `v1-prompt`, `v2-prompt`); GEPA carries its seed in the notes column (`v1`/`v2`).
- Optimizer scores are trusted from col-7 only when cls + chg are also present (guards mis-aligned rows).

## Data-availability caveats

- **v1 / v2 seed** optimizer data exists mainly for **MIPROv2** (+ one GEPA point each). **Bootstrap** and **tournament** have no v1/v2-seed runs → those columns show "no run yet".
- Nearly all val inference is temp 0.1, so a temperature slice isn't supported by this data.

## Adding a plot

Add a `scatter_mean([...], title, outfile)` call in `plot_dca_scatter.py`; each group is `(label, sub(optimizer=..., model=..., seed_source=..., cot=...))`.
