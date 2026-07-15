r"""[wrapper-only] Collect ALL data points (every repeat, per-example) from apples varieties, no-cot V1,
GEPA runs, and the true-MIDAS baseline into referenced tables. Every row cites its SOURCE FILE + JSON path
so it can be manually re-verified. Re-runnable: `.meta-prompting\Scripts\python.exe _collect_datapoints.py`.

Outputs (under apples_compare/):
  MASTER_DATAPOINTS.txt        human-readable: per-variety aggregate + per-repeat + per-example, with sources
  ALL_perrepeat.csv            one row per (category,model,variety,repeat) = mean M1..M8 + composite (over examples)
  ALL_perexample.csv           one row per (category,model,variety,example,repeat) = composite%
  GEPA_and_baseline.csv        one row per GEPA run / true-MIDAS baseline = baseline/optimized/improvement or composite
"""
import os
import json
import glob
import statistics

ROOT = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(ROOT, "apples_compare", "batch_meanstd")
T = os.path.join(ROOT, "apples_compare", "true_midas_baseline")
L = os.path.join(ROOT, "log")
METRICS = ["m1", "m2", "m3", "m4", "m7", "m8"]
VLAB = {"V1": "M4old,no-inject", "V2": "M4new,no-inject", "V3": "injOLD,1retry",
        "V4": "injNEW,1retry", "V5": "injNEW,4retry"}
txt = []
rep_rows = []   # (category, model, variety, cot, kb, repeat_idx, m1..m8, composite, source)
ex_rows = []    # (category, model, variety, cot, kb, example, repeat_idx, composite, source)
gepa_rows = []  # (category, run, baseline, optimized, improvement, seed_retained, cost, source)


def P(s=""):
    print(s); txt.append(s)


def _ms(xs):
    return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def load(p):
    return json.load(open(p, encoding="utf-8"))


# ── category metadata (verified from inventory) ──────────────────────────────────────────────────
CATS = [
    dict(id="1", title="apples V1-V5 | gpt-4.1-MINI | INTERMEDIATE KB | CoT | 3 reps",
         files=["batch_meanstd_intermediate_20260715_002626.json"], model="azure/gpt-4.1-mini",
         cot=True, kb="intermediate", varieties=["V1", "V2", "V3", "V4", "V5"], kind="batch"),
    dict(id="2", title="apples V1-V5 | gpt-4.1 | INTERMEDIATE KB | CoT | 3 reps",
         files=["batch_meanstd_intermediate_20260715_002601.json"], model="azure/gpt-4.1",
         cot=True, kb="intermediate", varieties=["V1", "V2", "V3", "V4", "V5"], kind="batch"),
    dict(id="3", title="apples V1-V5 | gpt-4.1-MINI | FINAL KB | CoT | POOLED 3 reps (2+1)",
         files=["batch_meanstd_final_20260714_164247.json", "batch_meanstd_final_20260714_232805.json"],
         model="azure/gpt-4.1-mini", cot=True, kb="final", varieties=["V1", "V2", "V3", "V4", "V5"], kind="batch"),
    dict(id="4", title="apples V1-V5 | gpt-4.1 | FINAL KB | CoT | 3 reps",
         files=["batch_meanstd_final_20260715_003950.json"], model="azure/gpt-4.1",
         cot=True, kb="final", varieties=["V1", "V2", "V3", "V4", "V5"], kind="batch"),
    dict(id="5", title="apples V1 | FINAL KB | NO-CoT | gpt-4.1 + gpt-4.1-mini | 3 reps",
         files=["batch_meanstd_V1_final_nocot_20260715_003755.json"], model=None,
         cot=False, kb="final", varieties=["V1"], kind="batch"),
    dict(id="6", title="apples V1 | INTERMEDIATE KB | NO-CoT | gpt-4.1 + gpt-4.1-mini | 3 reps",
         files=["batch_meanstd_V1_intermediate_nocot_20260715_002557.json"], model=None,
         cot=False, kb="intermediate", varieties=["V1"], kind="batch"),
]
GEPA_150 = [
    ("7-final-GEPA150", "log/PRESERVED_GEPA/GEPA_final_150call_20260715_run_summary.json"),
    ("7-intermediate-GEPA150", "log/PRESERVED_GEPA/GEPA_intermediate_150call_20260715_run_summary.json"),
]
TRUEMIDAS = ("8-true-midas-final", "apples_compare/true_midas_baseline/true_midas_baseline_final_20260715_114414.json")


def do_batch(cat):
    P("=" * 108); P(f"#{cat['id']}  {cat['title']}")
    for f in cat["files"]:
        P(f"  source: apples_compare/batch_meanstd/{f}")
    P("-" * 108)
    ds = [load(os.path.join(B, f)) for f in cat["files"]]
    models = cat["model"] and [cat["model"]] or ds[0]["config"].get("models", [])
    for model in models:
        P(f"  MODEL {model}   (cot={cat['cot']}, KB={cat['kb']})")
        P(f"    {'variety':22} {'composite%(mean+/-std,n)':26} {'  '.join(m.upper() for m in METRICS)}")
        for v in cat["varieties"]:
            pooled = []
            for f, d in zip(cat["files"], ds):
                for i, rep in enumerate((d.get("raw_per_repeat", {}).get(model, {}) or {}).get(v, [])):
                    pooled.append((rep, f, i))
                    rep_rows.append([f"#{cat['id']}", model, v, cat["cot"], cat["kb"], f"{f}:rep{i}"]
                                    + [round(float(rep.get(m) or 0), 4) for m in METRICS]
                                    + [round(rep.get("composite", 0), 3)])
            if not pooled:
                continue
            comps = [r[0].get("composite", 0.0) for r in pooled]
            cmean, cstd = _ms(comps)
            mm = {m: statistics.mean([float(r[0].get(m) or 0.0) for r in pooled]) for m in METRICS}
            P(f"    {v+' '+VLAB.get(v,''):22} {cmean:6.2f} +/- {cstd:4.2f} (n={len(pooled)})  "
              + "  ".join(f"{mm[m]:.2f}" for m in METRICS)
              + f"   RAW={[round(c,2) for c in comps]}")
        # per-example (composite per repeat)
        for f, d in zip(cat["files"], ds):
            pe = (d.get("per_example_composite", {}) or {}).get(model, {})
            for v in cat["varieties"]:
                for exk, lst in (pe.get(v, {}) or {}).items():
                    for i, c in enumerate(lst):
                        ex_rows.append([f"#{cat['id']}", model, v, cat["cot"], cat["kb"], exk, f"{f}:rep{i}", round(c, 2)])
    P("")


for cat in CATS:
    do_batch(cat)

# ── #7 GEPA 150-call ──
P("=" * 108); P("#7  COMPLETED GEPA (150 metric calls) — FINAL & INTERMEDIATE (CoT, gpt-4.1-mini)  [PRESERVED]")
for tag, f in GEPA_150:
    d = load(os.path.join(ROOT, f))
    P(f"  source: {f}")
    P(f"    BASELINE={d['baseline_score']:.2f}%  OPTIMIZED={d['best_score']:.2f}%  IMPROVEMENT={d['improvement']:+.2f}%  "
      f"seed_retained={d.get('seed_retained')}  cost=${(d.get('cost') or {}).get('total_usd')}")
    gepa_rows.append([tag, os.path.basename(f), round(d['baseline_score'], 2), round(d['best_score'], 2),
                      round(d['improvement'], 2), d.get('seed_retained'), (d.get('cost') or {}).get('total_usd'), f])
P("")

# ── #8 true-MIDAS ──
tag, f = TRUEMIDAS
d = load(os.path.join(ROOT, f))
c = d["config"]; rep = d["report"]
P("=" * 108); P("#8  TRUE-MIDAS (run.py prompts) BASELINE | gpt-4.1-mini | FINAL KB | max_retries=2 | 3 reps")
P(f"  source: {f}")
P("  " + "  ".join(f"{m.upper()}={rep[m]['mean']:.3f}+/-{rep[m]['std']:.3f}" for m in METRICS)
  + f"   COMPOSITE={rep['composite']['mean']:.2f}% +/- {rep['composite']['std']:.2f}")
for exk, lst in (d.get("per_example_composite") or {}).items():
    mn, sd = _ms(lst)
    P(f"      {exk:32} {mn:6.2f} +/- {sd:4.2f}   vals={lst}")
    for i, cc in enumerate(lst):
        ex_rows.append(["#8", "azure/gpt-4.1-mini", "run.py", False, "final", exk, f"{os.path.basename(f)}:rep{i}", round(cc, 2)])
gepa_rows.append([tag, os.path.basename(f), "", round(rep['composite']['mean'], 2), "", "", "", f])
P(f"  fail_count={d.get('fail_count')}\n")

# ── #9 in-progress note ──
P("=" * 108); P("#9  IN-PROGRESS GEPA (300 calls) — FINAL CoT + FINAL NoCot — NOT YET COMPLETE (no run_summary).")
P("  (add when done: log/..._CoT_..._final_run1_run_summary.json and ..._NoCot_..._final_run1_run_summary.json)")
P("  NOTE: the CoT-final log is shared with the completed 150-call run — read its run_summary, not the log.\n")

# ── write files ──
oc = os.path.join(ROOT, "apples_compare")
open(os.path.join(oc, "MASTER_DATAPOINTS.txt"), "w", encoding="utf-8").write("\n".join(txt))
with open(os.path.join(oc, "ALL_perrepeat.csv"), "w", encoding="utf-8") as fp:
    fp.write("category,model,variety,cot,kb,repeat_source," + ",".join(METRICS) + ",composite\n")
    for r in rep_rows:
        fp.write(",".join(str(x) for x in r) + "\n")
with open(os.path.join(oc, "ALL_perexample.csv"), "w", encoding="utf-8") as fp:
    fp.write("category,model,variety,cot,kb,example,repeat_source,composite\n")
    for r in ex_rows:
        fp.write(",".join(str(x) for x in r) + "\n")
with open(os.path.join(oc, "GEPA_and_baseline.csv"), "w", encoding="utf-8") as fp:
    fp.write("tag,file,baseline,optimized,improvement,seed_retained,cost_usd,source\n")
    for r in gepa_rows:
        fp.write(",".join(str(x) for x in r) + "\n")
P(f"Wrote apples_compare/MASTER_DATAPOINTS.txt")
P(f"Wrote apples_compare/ALL_perrepeat.csv   ({len(rep_rows)} rows)")
P(f"Wrote apples_compare/ALL_perexample.csv  ({len(ex_rows)} rows)")
P(f"Wrote apples_compare/GEPA_and_baseline.csv ({len(gepa_rows)} rows)")
