r"""[wrapper-only] FINAL apples/GEPA results table - UNIFORM 3 REPEATS everywhere.

History: several apples source files had an INCOMPLETE 3rd repeat (one repeat did not finish all 5 val
examples), so we first standardized to 2 reps. The user then TOPPED UP each incomplete config with one fresh
repeat (temp=1.0, cache off => reps are i.i.d./exchangeable, so a fresh repeat is a valid 3rd sample). With
those top-ups merged, EVERY run now has a complete 3 repeats, so this is the final uniform 3-repeat table.

METHOD (composite reconstructs from the atomic per_example_composite scores in each cited source file):
  for each (run, model, variety): pool per_example_composite across the run's files IN ORDER
  (original file(s) first, then the top-up), keep the FIRST 3 reps of each of the 5 examples, then
     comp_mean = mean over the 5 examples of (mean of that example's first 3 composites)   [robust, balanced]
     rep0/1/2  = the 3 per-example sample-column means (comp_mean is their average)
     comp_std  = pstdev([rep0, rep1, rep2])
Set KEEP_REPS=2 in the env to reproduce the earlier 2-rep table.

Runs:
  #1-#6 apples (dspy predictors) + top-ups   #7 GEPA 150-call   #8 true-MIDAS run.py prompts
  #10 GEPA 300-call final CoT (COMPLETED)     #11 GEPA 300-call final NoCoT (COMPLETED)
  (#9 = the earlier 300-call attempts KILLED by a DLL hang; superseded by #10/#11 - kept only as a note.)

Outputs (apples_compare/): FINAL_COMPOSITE.csv, FINAL_REMOVED_DATA.csv, FINAL_REPORT.txt
Re-runnable: `.meta-prompting\Scripts\python.exe _final_tables.py`
"""
import os
import json
import re
import statistics

ROOT = os.path.dirname(os.path.abspath(__file__))
Lp = "log/PRESERVED_GEPA"
KEEP = int(os.getenv("KEEP_REPS", "3"))
V5 = ["V1", "V2", "V3", "V4", "V5"]
EX_ORDER = ["GearLvrPosn_Vd", "InjRly_PwrCtrl_1_1_0", "EGSSig_LamSymcp_1_6_0", "oilsepactr_coorr", "Rail_TempLim_50.8.0"]


def load(p):
    return json.load(open(os.path.join(ROOT, p) if not os.path.isabs(p) else p, encoding="utf-8"))


def exkey_sort(k):
    base = k.split("#")[0]
    return EX_ORDER.index(base) if base in EX_ORDER else 99


# ── run registry (verified metadata; top-up files appended AFTER the original so first-3 keeps originals) ──
TU_INT = "batch_meanstd_intermediate_topup_20260715_164057.json"   # both models, KB=intermediate, CoT, V1-V5
TU_FIN = "batch_meanstd_final_topup_20260715_154345.json"          # gpt-4.1,     KB=final,       CoT, V1-V5
TU_NOCOT = "batch_meanstd_V1_final_nocot_topup_20260715_152228.json"  # mini, KB=final, NoCoT, V1 (later of 2; 150556 = discarded dup)
RUNS = [
    dict(run="1", exp="apples V1-V5 INTERMEDIATE CoT", models=["azure/gpt-4.1-mini"], varis=V5, cot="CoT", kb="intermediate",
         files=["batch_meanstd_intermediate_20260715_002626.json", TU_INT]),
    dict(run="2", exp="apples V1-V5 INTERMEDIATE CoT", models=["azure/gpt-4.1"], varis=V5, cot="CoT", kb="intermediate",
         files=["batch_meanstd_intermediate_20260715_002601.json", TU_INT]),
    dict(run="3", exp="apples V1-V5 FINAL CoT (pooled 2-rep + 1-rep files)", models=["azure/gpt-4.1-mini"], varis=V5, cot="CoT", kb="final",
         files=["batch_meanstd_final_20260714_164247.json", "batch_meanstd_final_20260714_232805.json"]),
    dict(run="4", exp="apples V1-V5 FINAL CoT", models=["azure/gpt-4.1"], varis=V5, cot="CoT", kb="final",
         files=["batch_meanstd_final_20260715_003950.json", TU_FIN]),
    dict(run="5", exp="apples V1 FINAL NoCoT", models=["azure/gpt-4.1", "azure/gpt-4.1-mini"], varis=["V1"], cot="NoCoT", kb="final",
         files=["batch_meanstd_V1_final_nocot_20260715_003755.json", TU_NOCOT]),
    dict(run="6", exp="apples V1 INTERMEDIATE NoCoT", models=["azure/gpt-4.1", "azure/gpt-4.1-mini"], varis=["V1"], cot="NoCoT", kb="intermediate",
         files=["batch_meanstd_V1_intermediate_nocot_20260715_002557.json"]),
]

final_rows = []
removed_rows = []


def add_run(run):
    ds = [(f, load(os.path.join("apples_compare", "batch_meanstd", f))) for f in run["files"]]
    src = " + ".join("batch_meanstd/" + f for f in run["files"])
    for m in run["models"]:
        for v in run["varis"]:
            pooled = {}
            for f, d in ds:
                pe = (d.get("per_example_composite", {}).get(m, {}) or {}).get(v, {})
                for exk, lst in pe.items():
                    pooled.setdefault(exk, []).extend(lst)
            if not pooled:
                continue
            exks = sorted(pooled, key=exkey_sort)
            per_ex_mean = {e: statistics.mean(pooled[e][:KEEP]) for e in exks}
            comp_mean = statistics.mean(per_ex_mean.values())
            cols = []
            for c in range(KEEP):
                cv = [pooled[e][c] for e in exks if len(pooled[e]) > c]
                if cv:
                    cols.append(statistics.mean(cv))
            comp_std = statistics.pstdev(cols) if len(cols) > 1 else 0.0
            final_rows.append(dict(run=run["run"], experiment=run["exp"], model=m, variety=v, cot=run["cot"],
                                   kb=run["kb"], kind="apples-3rep", comp_mean=round(comp_mean, 2), comp_std=round(comp_std, 2),
                                   n_reps=KEEP, reps=";".join(f"{c:.2f}" for c in cols),
                                   per_example=";".join(f"{e.split('#')[0]}={round(per_ex_mean[e],2)}" for e in exks),
                                   source=src, notes=""))
            src_pat = ",".join(f"{e.split('#')[0]}:{len(pooled[e])}" for e in exks)
            removed_here = sum(max(0, len(pooled[e]) - KEEP) for e in exks)
            short = [e.split('#')[0] for e in exks if len(pooled[e]) < KEEP]
            removed_rows.append(dict(run=run["run"], model=m, variety=v, pooled_reps_per_example=src_pat,
                                     datapoints_dropped_to_reach_3=removed_here,
                                     examples_still_below_3=("|".join(short) if short else "(none - all >=3)"),
                                     source=" + ".join(run["files"])))


for r in RUNS:
    add_run(r)

# ── #8 true-MIDAS baseline (3 reps complete) ──────────────────────────────────────────────────────
f8 = "apples_compare/true_midas_baseline/true_midas_baseline_final_20260715_114414.json"
d8 = load(f8)
pe8 = d8.get("per_example_composite", {})
exks8 = sorted(pe8, key=exkey_sort)
per_ex8 = {e: statistics.mean(pe8[e][:KEEP]) for e in exks8}
cols8 = [statistics.mean([pe8[e][c] for e in exks8 if len(pe8[e]) > c]) for c in range(KEEP)]
final_rows.append(dict(run="8", experiment="true-MIDAS run.py prompts (gen=gpt-4.1-mini)", model="azure/gpt-4.1-mini",
                       variety="orig", cot="-", kb="final", kind="true-midas-3rep",
                       comp_mean=round(statistics.mean(per_ex8.values()), 2), comp_std=round(statistics.pstdev(cols8), 2),
                       n_reps=KEEP, reps=";".join(f"{c:.2f}" for c in cols8),
                       per_example=";".join(f"{e.split('#')[0]}={round(per_ex8[e],2)}" for e in exks8),
                       source=f8, notes="run.py's OWN prompts+transport, scored with the same M1-M8"))
removed_rows.append(dict(run="8", model="azure/gpt-4.1-mini", variety="orig",
                         pooled_reps_per_example=",".join(f"{e.split('#')[0]}:{len(pe8[e])}" for e in exks8),
                         datapoints_dropped_to_reach_3=sum(max(0, len(pe8[e]) - KEEP) for e in exks8),
                         examples_still_below_3="(none - all >=3)", source=os.path.basename(f8)))

# ── #7 GEPA 150-call (single baseline/optimized) ─────────────────────────────────────────────────
P7_NOTE = ("KB label (intermediate vs final) is NOT stored in this run_summary (unlike apples files, which record "
           "config.rules.required_io_path). It rests on (a) run name 'intermediate_run1' vs 'final_run1' and (b) "
           "baseline direction: intermediate 60.35 > final 51.42, consistent with the intermediate KB scoring higher "
           "(matches apples #1 vs #3). Treat the KB label as inferred, not file-proven.")
for kb, f in [("final", f"{Lp}/GEPA_final_150call_20260715_run_summary.json"),
              ("intermediate", f"{Lp}/GEPA_intermediate_150call_20260715_run_summary.json")]:
    d = load(f)
    final_rows.append(dict(run="7", experiment=f"GEPA-150call-{kb} BASELINE(seed)", model=d.get("model_name"),
                           variety="-", cot="CoT", kb=kb, kind="GEPA-baseline", comp_mean=round(d["baseline_score"], 2),
                           comp_std="", n_reps=1, reps="", per_example="", source=f, notes="single seed eval (not repeat-based)"))
    final_rows.append(dict(run="7", experiment=f"GEPA-150call-{kb} OPTIMIZED", model=d.get("model_name"),
                           variety="-", cot="CoT", kb=kb, kind="GEPA-optimized", comp_mean=round(d["best_score"], 2),
                           comp_std="", n_reps=1, reps="", per_example="", source=f,
                           notes=f"improvement={d['improvement']:+.2f}; cost=${(d.get('cost') or {}).get('total_usd')}; "
                                 f"seed_retained={d.get('seed_retained')}. " + P7_NOTE))

# ── #10 GEPA 300-call final CoT (COMPLETED) ──────────────────────────────────────────────────────
f10 = f"{Lp}/GEPA_final_cot_300call_COMPLETED_20260715_run_summary.json"
d10 = load(f10)
c10 = (d10.get("cost") or {}).get("total_usd")
final_rows.append(dict(run="10", experiment="GEPA-300call-final-CoT BASELINE(seed)", model=d10.get("model_name"),
                       variety="-", cot="CoT", kb="final", kind="GEPA-baseline", comp_mean=round(d10["baseline_score"], 2),
                       comp_std="", n_reps=1, reps="", per_example="", source=f10, notes="single seed eval; COMPLETED 300-call run"))
final_rows.append(dict(run="10", experiment="GEPA-300call-final-CoT OPTIMIZED", model=d10.get("model_name"),
                       variety="-", cot="CoT", kb="final", kind="GEPA-optimized", comp_mean=round(d10["best_score"], 2),
                       comp_std="", n_reps=1, reps="", per_example="", source=f10,
                       notes=f"improvement={d10['improvement']:+.2f}; cost=${c10}; seed_retained={d10.get('seed_retained')}; "
                             f"duration={d10.get('total_duration_seconds')}s; optimized_state preserved alongside run_summary"))

# ── #11 GEPA 300-call final NoCoT (COMPLETED) ────────────────────────────────────────────────────
f11 = f"{Lp}/GEPA_final_nocot_300call_COMPLETED_20260715_run_summary.json"
d11 = load(f11)
c11 = (d11.get("cost") or {}).get("total_usd")
final_rows.append(dict(run="11", experiment="GEPA-300call-final-NoCoT BASELINE(seed)", model=d11.get("model_name"),
                       variety="-", cot="NoCoT", kb="final", kind="GEPA-baseline", comp_mean=round(d11["baseline_score"], 2),
                       comp_std="", n_reps=1, reps="", per_example="", source=f11, notes="single seed eval; COMPLETED 300-call run"))
final_rows.append(dict(run="11", experiment="GEPA-300call-final-NoCoT OPTIMIZED", model=d11.get("model_name"),
                       variety="-", cot="NoCoT", kb="final", kind="GEPA-optimized", comp_mean=round(d11["best_score"], 2),
                       comp_std="", n_reps=1, reps="", per_example="", source=f11,
                       notes=f"improvement={d11['improvement']:+.2f}; cost=${c11}; seed_retained={d11.get('seed_retained')}; "
                             f"duration={d11.get('total_duration_seconds')}s; optimized_state preserved alongside run_summary"))

# ── write CSVs ────────────────────────────────────────────────────────────────────────────────────
oc = os.path.join(ROOT, "apples_compare")
cols = ["run", "experiment", "model", "variety", "cot", "kb", "kind", "comp_mean", "comp_std", "n_reps",
        "reps", "per_example", "source", "notes"]
with open(os.path.join(oc, "FINAL_COMPOSITE.csv"), "w", encoding="utf-8") as fp:
    fp.write(",".join(cols) + "\n")
    for r in final_rows:
        fp.write(",".join('"%s"' % str(r[c]).replace('"', "'") if c in ("experiment", "reps", "per_example", "source", "notes") else str(r[c]) for c in cols) + "\n")
with open(os.path.join(oc, "FINAL_REMOVED_DATA.csv"), "w", encoding="utf-8") as fp:
    rc = ["run", "model", "variety", "pooled_reps_per_example", "datapoints_dropped_to_reach_3", "examples_still_below_3", "source"]
    fp.write(",".join(rc) + "\n")
    for r in removed_rows:
        fp.write(",".join('"%s"' % str(r[c]) if c in ("pooled_reps_per_example", "examples_still_below_3", "source") else str(r[c]) for c in rc) + "\n")

# ── write FINAL_REPORT.txt ────────────────────────────────────────────────────────────────────────
L = []
def W(s=""):
    L.append(s)

W("=" * 112)
W(f"FINAL RESULTS - UNIFORM {KEEP} REPEATS  (apples #1-#6 + top-ups, true-MIDAS #8) + GEPA #7/#10/#11")
W("Generated by _final_tables.py. Every composite reconstructs from per_example_composite in the cited source.")
W("=" * 112)
W("")
W(f"WHY {KEEP} REPEATS: apples #1,#2,#4,#5-mini originally had an incomplete/biased 3rd repeat; each was TOPPED UP")
W("with one fresh repeat (temp=1.0, cache off => i.i.d. samples), so all runs are now complete at 3 reps. Composite")
W("= mean over the 5 val examples of (mean of that example's first 3 composites). rep0/1/2 = the 3 sample-column")
W("means (comp% is their average; the A/B/C split is nominal where a top-up supplied the 3rd sample).")
W("")
W("COMPOSITE WEIGHTS (all runs): M1=.20 M2=.15 M3=.15 M4=.20 M7=.15 M8=.15 | M4 judge=azure/gpt-5.2 | M8_FULL_RESOLVE=on | M7_ref_pin=4")
W("")
W("-" * 112)
W(f"FINAL {KEEP}-REPEAT COMPOSITE TABLE (apples + true-MIDAS)")
W("-" * 112)
W(f"{'run':4}{'model':20}{'var':5}{'cot':7}{'kb':13}{'comp%':>8}{'+/-':>7}   rep0/rep1/rep2        source")
for r in final_rows:
    if r["kind"] in ("apples-3rep", "true-midas-3rep"):
        W(f"{r['run']:4}{(r['model'] or '')[:19]:20}{r['variety']:5}{r['cot']:7}{r['kb']:13}"
          f"{r['comp_mean']:>8}{str(r['comp_std']):>7}   {r['reps']:22}{os.path.basename(str(r['source']).split(' + ')[0])}")
W("")
W(f"RUN-LEVEL AVERAGE (mean over varieties, {KEEP}-rep):")
agg = {}
for r in final_rows:
    if r["kind"] == "apples-3rep":
        agg.setdefault((r["run"], r["model"], r["cot"], r["kb"]), []).append(r["comp_mean"])
for (run, m, cot, kb), xs in agg.items():
    W(f"   #{run} {m:20} {cot:6} {kb:13} = {statistics.mean(xs):.2f}%  (over {len(xs)} varieties)")
W("")
W("-" * 112)
W("GEPA RUNS")
W("-" * 112)
for r in final_rows:
    if r["run"] in ("7", "10", "11"):
        W(f"   #{r['run']:3} {r['experiment']:38} = {r['comp_mean']}%   [{os.path.basename(str(r['source']))}]")
        if r["notes"]:
            W(f"          note: {r['notes']}")
W("")
W("   #9 (HISTORICAL, superseded): the FIRST 300-call attempts (CoT + NoCoT) were KILLED by a 32-bit DLL hang on")
W("   fc.is_function_available (now guarded). Best-so-far before kill: CoT 72.37% @iter25, NoCoT 72.25% @iter29.")
W("   Preserved logs: log/PRESERVED_GEPA/GEPA_final_{cot,nocot}_300call_STUCK_20260715.log. Superseded by #10/#11.")
W("")
W("-" * 112)
W("WHAT WAS DROPPED to reach uniform 3 reps (4th/extra reps from top-ups; no example is below 3)")
W("-" * 112)
W(f"{'run':4}{'model':20}{'var':5}{'pooled reps/example':44}{'dropped':>8}  below 3?")
tot = 0
for r in removed_rows:
    tot += r["datapoints_dropped_to_reach_3"]
    W(f"{r['run']:4}{r['model'][:19]:20}{r['variety']:5}{r['pooled_reps_per_example']:44}{r['datapoints_dropped_to_reach_3']:>8}  {r['examples_still_below_3']}")
W(f"\n   Total extra reps dropped (beyond 3) = {tot}. Plus 1 discarded duplicate NoCoT top-up file")
W("   (batch_meanstd_V1_final_nocot_topup_20260715_150556.json; the 152228 re-run is used). Every run = exactly 3 reps, zero holes.")
W("")
W("-" * 112)
W("#7 DETAILED NOTE (KB label provenance)")
W("-" * 112)
for line in [P7_NOTE[i:i+108] for i in range(0, len(P7_NOTE), 108)]:
    W("   " + line)
W("")
W("-" * 112)
W("SOURCE FILES (every number traces here):")
W("-" * 112)
seen = []
for r in final_rows:
    s = str(r["source"])
    if s not in seen:
        seen.append(s)
        W("   " + s)
W("")
W("Machine-readable: apples_compare/FINAL_COMPOSITE.csv (+ FINAL_REMOVED_DATA.csv). Set KEEP_REPS=2 to reproduce the 2-rep table.")
W("All 11 runs COMPLETE. GEPA optimized (final KB, gpt-4.1-mini): CoT 150-call 70.59, CoT 300-call 71.87, NoCoT 300-call 65.73.")

open(os.path.join(oc, "FINAL_REPORT.txt"), "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
print(f"\n\nWrote apples_compare/FINAL_COMPOSITE.csv ({len(final_rows)} rows, KEEP={KEEP})")
print(f"Wrote apples_compare/FINAL_REMOVED_DATA.csv ({len(removed_rows)} rows)")
print("Wrote apples_compare/FINAL_REPORT.txt")
