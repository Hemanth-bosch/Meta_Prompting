# Agentic optimizer on DCA — filling the `agentic` distribution on the crseed plots

The two crseed plots (`results/plots/figures/dca_gpt41{mini,}_crseed.png`) now include an
**`agentic`** category between `gepa` and `tournament`, currently *"no run yet"* because the
optimization runs need Azure OpenAI connectivity (worker + gpt-5.2 controller/reflection +
gpt-4.1 judge, all on `*.openai.azure.com`), which was blocked from the build machine.

## Method (matches the baseline optimizers)
- **Selection (same protocol as the DCA tournament):** the agentic loop optimizes, then
  **re-evaluates every version on the FULL val set (82)** and picks the best — "re-eval
  candidates on full val, pick best". That best-by-val is one plotted point (fallback:
  best-by-dev, which is the same number when val selection doesn't finish).
- **Distribution:** run the pipeline **3× per inference model** → 3 best-by-val points per plot,
  exactly like the gepa/miprov2/bootstrap single-run-best points. Distribution-vs-distribution.
- **Caching (selective, best of both):** `run_dca_agentic.py` sets `AGENTIC_LM_CACHE=1` so the
  **worker/predictor** LM caches (same image+prompt → same output → token savings on the many
  repeated evals) and the **gpt-4.1 judge** caches (stable, plot-comparable measurement) — matching
  the cache-ON protocol that produced the baseline points. The **controller/memory/recommendation**
  agents (smolagents `AzureOpenAIServerModel`) and the **reflection / GEPA-reflection** LM
  (`reflectin_lm`, temp 1.0) stay **uncached**, so the meta-level stays stochastic and the 3 runs
  explore different trajectories → a real distribution. `--worker-cache off` reverts to all-off.
- **Fidelity:** val/test are pinned by `pair_name` to the exact extract_dca split
  (`aiq212-.../task-adaptation/data_loading/dca_split_seed42.json`), verified byte-identical to the
  16-07 `.cache/splits` manifests (val=82, test=93); train (60%) → train~120 / dev~46, same
  stratifier/seed; val/test never sub-sampled. Metric vendored verbatim (0.6/0.4, F-beta,
  gpt-4.1 judge, 16-07 judge cache reused).

## Commands (on a network with Azure access)

Add two **DCA-scoped** vars to `meta-prompting/.env` ONCE. They take effect ONLY for
`run_agentic_loop.py dca` (translated to the generic vars before config/utils load) and are
ignored by MBD/every other task, so there's no interference:

    DCA_AGENTIC_WORKER_MODEL=azure/gpt-4.1-mini   # inference model = which PNG
    DCA_AGENTIC_LM_CACHE=1                          # worker+judge cache ON; controller/reflection OFF

Then run `run_agentic_loop.py dca` **directly**, like MBD — from the `meta-prompting` dir (so `.env` loads):

    cd "C:\Users\BHS1RNG\OneDrive - Bosch Group\AIQ-212\Code\meta-prompting"
    $LOOP="..\aiq212-prompt-optimization\aiq-prompt-optimization"

    # ---- RUN 1 (repeat 3x; each prints an outputs\agentic\dca\<ts> path) ----
    .\.meta-prompting\Scripts\python.exe "$LOOP\run_agentic_loop.py" dca --iterations 15 `
        --num-eval-threads 4 --val-version-workers 2 `
        --enable-demo-combination --enable-extra-decision-signals --output-predictions
    # step 4 record for RUN 1 (no-cache repeatability; --latest = newest run for the model)
    .\.meta-prompting\Scripts\python.exe "$LOOP\eval_topk_dca_nocache.py" --latest azure/gpt-4.1-mini --topk 6 --draws 3 --with-test
    # ... repeat for RUN 2, RUN 3 ...

    # collect the 3 best-by-val points onto the plot (pass the 3 printed run dirs), then regenerate
    .\.meta-prompting\Scripts\python.exe "$LOOP\collect_dca_points.py" --model azure/gpt-4.1-mini --run-dir <dir1> <dir2> <dir3>
    cd "..\meta-prompting -16-07"
    ..\meta-prompting\.meta-prompting\Scripts\python.exe results\plots\build_datapoints.py
    ..\meta-prompting\.meta-prompting\Scripts\python.exe results\plots\plot_dca_scatter.py

For the gpt-4.1 PNG, set `DCA_AGENTIC_WORKER_MODEL=azure/gpt-4.1` (edit `.env`, or override once with
`$env:DCA_AGENTIC_WORKER_MODEL="azure/gpt-4.1"`) and repeat.

**vs MBD:** DCA has NO 32-bit DLL, so use `--num-eval-threads 4 --val-version-workers 2` (MBD needs
`1/1`). Set `AGENTIC_LM_CACHE=1` to cache worker+judge (MBD ran the repo default = cache-off). Drop
to `--val-version-workers 1` / `--num-eval-threads 1` if Azure 429s appear (worker `num_retries=0`).

**Optional one-command wrapper** (does the env + 3-run loop + collect for you; not required):

    run_dca_agentic.py azure/gpt-4.1-mini --runs 3 --iterations 15

## Verify multimodal GEPA at runtime
Expect `  [GEPA] Multimodal dataset detected — automatically enabling image reflection.` and, during
reflection, `[MultimodalReflection] sending N image(s) ...` with N>0. A
`  [GEPA] WARNING: multimodal reflection ON but 0 image URLs captured ...` means images aren't
reaching the reflection LM (capture patch in `optimization_loop/dspy_tools.py` was made recursive
to handle nested/object image inputs).

## Output files
- `dca_agentic_results.json` — plotted rows, one per run's best-by-val:
  `[{"model":"gpt-4.1-mini","val_score":0.87,"cot":"nocot","seed_source":"cr-default","source":"agentic:gpt41mini:<ts>:valbest"}]`
- `dca_agentic_runs.json` — per-run metadata (run_dir, best_version, program_path, val/dev/test).
- `dca_agentic_topk_report.json` — step-4 record: per-candidate no-cache full-val draws, mean±std,
  overall best by val-mean and best single draw.
