# DCA scatter-plus-mean plots

Reproducible plots of DCA total score with **individual run data points + the mean marker only** (no box, no median). All on the **val set**.

## Pipeline

```
dca_experiments.xlsx (val sheet)  ─┐
excel/tournament_top2.xlsx        ─┤─►  build_datapoints.py  ─►  dca_datapoints.csv  ─►  plot_dca_scatter.py  ─►  figures/*.png
```

```bash
python results/plots/build_datapoints.py   # xlsx -> dca_datapoints.csv (auditable)
python results/plots/plot_dca_scatter.py   # csv  -> figures/*.png
```

- **`dca_datapoints.csv`** — one row per individual run: `source_row, optimizer, prompt_version, model, seed_source, cot, score`. `source_row` traces each point back to the spreadsheet (or `tournament_top2:<instance>`), so every point is auditable.
- **Data guard:** a val-sheet row's score is only trusted as the *total* when that same row also carries cls + chg (filters mis-aligned rows where col-7 held a classification list). Repeatability rows expand to one point each. Validated against the deck (v1-mini ≈ 0.46, v2-mini ≈ 0.68, tournament gpt-4.1 ≈ 0.92).

## Figures (`figures/`)

| File | What it shows |
|------|---------------|
| `dca_cr_default_gpt41mini.png` | gpt-4.1-mini, CR-default seed: human v1 · human v2 · bootstrap · miprov2 · gepa · tournament |
| `dca_cr_default_gpt41.png` | same, gpt-4.1 |
| `dca_seed_source_miprov2_gpt41mini.png` | MIPROv2 (gpt-4.1-mini) by starting seed: CR-default vs v1 vs v2 |
| `dca_model_compare_cr_default.png` | each optimizer, gpt-4.1-mini vs gpt-4.1 |
| `dca_cot_vs_nocot_miprov2_gpt41mini.png` | MIPROv2 (gpt-4.1-mini): CoT vs No-CoT |
| `dca_human_v2_by_model.png` | v2 (CR) human prompt across gpt-4.1-mini / gpt-4.1 / gpt-5-mini / gpt-5.2 |

## Adding a plot

Add one `scatter_mean([...], title, outfile)` call in `plot_dca_scatter.py`; each group is `(label, sc(optimizer=..., model=..., seed_source=..., cot=...))`.

## Caveats (data availability)

- **Tournament** points are the per-instance "Best Score" candidate from `tournament_top2.xlsx`, tagged `cr-default`; the one v1-seed tournament run (`20260504234953`) is included in the gpt-4.1 tournament group.
- **v1 / v2 seed** data exists only for MIPROv2 in the sheet, so the seed-source plot is MIPROv2-only.
- **Temperature** is ≈0.1 for nearly all val inference, so a clean temperature slice isn't supported by this data.
- The **non-DSPy MiDAS-pipeline** baseline (classification-only, `initial_classification_eval/`) is not a DCA-total score and is not plotted here.
