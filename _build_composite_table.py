r"""[wrapper-only] Build ONE composite-metric table across all 9 runs/varieties, with source references.
Output: apples_compare/COMPOSITE_ALL_RUNS.csv  (+ printed table). Composite only (per-example not included).
Re-runnable: `.meta-prompting\Scripts\python.exe _build_composite_table.py`.
"""
import os
import json
import statistics

ROOT = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(ROOT, "apples_compare", "batch_meanstd")
T = os.path.join(ROOT, "apples_compare", "true_midas_baseline")
Lp = "log/PRESERVED_GEPA"


def load(p):
    return json.load(open(os.path.join(ROOT, p) if not os.path.isabs(p) else p, encoding="utf-8"))


def ms(xs):
    return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)


rows = []  # each: dict(run, experiment, model, variety, cot, kb, kind, comp_mean, comp_std, n, raw, source)


def add(run, experiment, model, variety, cot, kb, kind, cm, cs, n, raw, source):
    rows.append(dict(run=run, experiment=experiment, model=model, variety=variety, cot=cot, kb=kb,
                     kind=kind, comp_mean=cm, comp_std=cs, n=n, raw=raw, source=source))


# ── apples #1-#6 ──────────────────────────────────────────────────────────────────────────────────
CATS = [
    ("1", "apples", ["batch_meanstd_intermediate_20260715_002626.json"], "azure/gpt-4.1-mini", "CoT", "intermediate", ["V1", "V2", "V3", "V4", "V5"]),
    ("2", "apples", ["batch_meanstd_intermediate_20260715_002601.json"], "azure/gpt-4.1",      "CoT", "intermediate", ["V1", "V2", "V3", "V4", "V5"]),
    ("3", "apples", ["batch_meanstd_final_20260714_164247.json", "batch_meanstd_final_20260714_232805.json"], "azure/gpt-4.1-mini", "CoT", "final", ["V1", "V2", "V3", "V4", "V5"]),
    ("4", "apples", ["batch_meanstd_final_20260715_003950.json"], "azure/gpt-4.1", "CoT", "final", ["V1", "V2", "V3", "V4", "V5"]),
    ("5", "apples", ["batch_meanstd_V1_final_nocot_20260715_003755.json"], None, "NoCoT", "final", ["V1"]),
    ("6", "apples", ["batch_meanstd_V1_intermediate_nocot_20260715_002557.json"], None, "NoCoT", "intermediate", ["V1"]),
]
for run, exp, files, model, cot, kb, varis in CATS:
    ds = [load(os.path.join("apples_compare", "batch_meanstd", f)) for f in files]
    models = [model] if model else ds[0]["config"].get("models", [])
    for m in models:
        for v in varis:
            comps = []
            for d in ds:
                comps += [r.get("composite", 0.0) for r in (d.get("raw_per_repeat", {}).get(m, {}) or {}).get(v, [])]
            if not comps:
                continue
            cm, cs = ms(comps)
            add(run, exp, m, v, cot, kb, "apples-variety", round(cm, 2), round(cs, 2), len(comps),
                ";".join(f"{c:.2f}" for c in comps), " + ".join("batch_meanstd/" + f for f in files))

# ── #7 GEPA 150-call (preserved) ────────────────────────────────────────────────────────────────
for kb, f in [("final", f"{Lp}/GEPA_final_150call_20260715_run_summary.json"),
              ("intermediate", f"{Lp}/GEPA_intermediate_150call_20260715_run_summary.json")]:
    d = load(f)
    add("7", f"GEPA-{kb}-150call", d.get("model_name"), "-", "CoT", kb, "GEPA-baseline(seed)",
        round(d["baseline_score"], 2), "", 1, "", f)
    add("7", f"GEPA-{kb}-150call", d.get("model_name"), "-", "CoT", kb, "GEPA-OPTIMIZED",
        round(d["best_score"], 2), "", 1, f"improvement={d['improvement']:+.2f}; cost=${(d.get('cost') or {}).get('total_usd')}", f)

# ── #8 true-MIDAS baseline ──────────────────────────────────────────────────────────────────────
d = load(os.path.join("apples_compare", "true_midas_baseline", "true_midas_baseline_final_20260715_114414.json"))
rep = d["report"]["composite"]
raw = [r.get("composite", 0.0) for r in d.get("raw_per_repeat", [])]
add("8", "true-MIDAS (run.py prompts)", d["config"]["gen_model"], "run.py-orig", "-", "final",
    "true-midas-baseline", round(rep["mean"], 2), round(rep["std"], 2), len(raw) or d["config"]["repeats"],
    ";".join(f"{c:.2f}" for c in raw), "true_midas_baseline/true_midas_baseline_final_20260715_114414.json")

# ── #9 300-call runs — KILLED mid-run (no run_summary); best-so-far RE-EXTRACTED from the preserved logs ──
def gepa_log_best(log_path):
    """[wrapper-only] best-so-far from a killed GEPA run's log: isolate the 300-call section (after the last
    'budget=300 calls' line, skipping any appended prior run) and return (baseline_pct, best_pct, latest_iter)."""
    import re as _re
    txt = open(os.path.join(ROOT, log_path), encoding="utf-8", errors="ignore").read()
    i = txt.rfind("budget=300 calls")
    sec = txt[i:] if i >= 0 else txt
    base = _re.findall(r"Base program full valset score: ([0-9.]+)", sec)
    best = _re.findall(r"Best score on valset: ([0-9.]+)", sec)
    it = _re.findall(r"Iteration (\d+):", sec)
    return (round(float(base[-1]) * 100, 2) if base else "",
            round(float(best[-1]) * 100, 2) if best else "",
            it[-1] if it else "?")


for cot, log in [("CoT", "log/PRESERVED_GEPA/GEPA_final_cot_300call_STUCK_20260715.log"),
                 ("NoCoT", "log/PRESERVED_GEPA/GEPA_final_nocot_300call_STUCK_20260715.log")]:
    b, best, it = gepa_log_best(log)
    add("9", f"GEPA-final-{cot}-300call-STUCK", "azure/gpt-4.1-mini", "-", cot, "final", "GEPA-STUCK-baseline(seed)",
        b, "", 1, f"KILLED mid-run @iter{it}", log)
    add("9", f"GEPA-final-{cot}-300call-STUCK", "azure/gpt-4.1-mini", "-", cot, "final", "GEPA-STUCK-BEST-so-far",
        best, "", 1, f"best-so-far @iter{it}; NO optimized_state (killed) - prompts NOT saved", log)

# ── write CSV ───────────────────────────────────────────────────────────────────────────────────
cols = ["run", "experiment", "model", "variety", "cot", "kb", "kind", "comp_mean", "comp_std", "n", "raw", "source"]
outp = os.path.join(ROOT, "apples_compare", "COMPOSITE_ALL_RUNS.csv")
with open(outp, "w", encoding="utf-8") as fp:
    fp.write(",".join(cols) + "\n")
    for r in rows:
        fp.write(",".join('"%s"' % str(r[c]) if c in ("raw", "source", "experiment") else str(r[c]) for c in cols) + "\n")

# ── print a readable table ──────────────────────────────────────────────────────────────────────
print(f"COMPOSITE-ONLY master table — {len(rows)} rows  (source cited per row)\n")
print(f"{'run':4}{'experiment':30}{'model':20}{'var':5}{'cot':7}{'kb':13}{'kind':20}{'comp%':>8}{'+/-std':>8}{'n':>3}  raw")
print("-" * 150)
for r in rows:
    cm = f"{r['comp_mean']}" if r['comp_mean'] != "" else "  --"
    cs = f"{r['comp_std']}" if r['comp_std'] != "" else ""
    print(f"{r['run']:4}{r['experiment'][:29]:30}{(r['model'] or '')[:19]:20}{r['variety']:5}{r['cot']:7}{r['kb']:13}"
          f"{r['kind'][:19]:20}{cm:>8}{cs:>8}{str(r['n']):>3}  {r['raw'][:40]}")
print(f"\nWrote {outp}")
