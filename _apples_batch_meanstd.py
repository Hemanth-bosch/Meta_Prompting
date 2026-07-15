"""BATCH: V1..V5 x N repeats x M inference models -> per-metric (M1..M8) + composite mean/std.

Inference LMs run at temperature=1.0, cache=False, so the repeats are INDEPENDENT draws (real
mean/std = repeatability). The M4 coverage-judge is DEDICATED to azure/gpt-5.2 (see
mbd_metric._get_judge_predictor) regardless of the inference model, so M4 is comparable across models.
ALL scoring goes through evaluation.mbd_metric._compute_all_components + the WEIGHT_* constants imported
from that module — so this script CANNOT drift from the pipeline's own metric definitions/weights.

Varieties (== injection-matrix C1..C5):
  V1  no-inject       | M4 old KB | 1 retry     (pipeline shared with V2 — only M4's KB differs)
  V2  no-inject       | M4 new KB | 1 retry
  V3  inject OLD KB    | M4 old KB | 1 retry
  V4  inject NEW KB    | M4 new KB | 1 retry
  V5  inject NEW KB    | M4 new KB | 4 retries

Latest metric rules in force (all via _compute_all_components — this script inherits them automatically):
  * M1 = OUTPUT-ROLE vs ground-truth instance_names (output-verified + output-not-set-as-input + fixed-const);
         every check weighs M1_IO_CHECK_WEIGHT (see banner).
  * M2 = UNIQUE-name resolution (each distinct signal counts once).
  * M3 = structural; act/exp + precond/exp overlaps weigh M3_OVERLAP_WEIGHT on failure; NO-Wait steps skipped.
  * M4 = KB-guideline coverage (LLM judge, dedicated gpt-5.2). old-KB vs new-KB re-judged (cache keyed on guidelines).
  * M7 = graded DLL solve-retry efficiency (ref pinned to 4 -> all varieties share one scale, see below);
         function-call weakspots have no arithmetic to solve -> neutral 0.5.
  * M8 = functional correctness on the DLL; GATED to 0 when the test targets a non-measurable output
         (M8_GATE_ON_WRONG_IO). For the systemically-disconnected weakspots M8 is 0/degenerate — read M1 for
         output-correctness. (M6 removed.)

Run a SMOKE test first (1 example, 1 repeat) then the full run:
  .meta-prompting\\Scripts\\python.exe _apples_batch_meanstd.py --limit 1 --repeats 1
  .meta-prompting\\Scripts\\python.exe _apples_batch_meanstd.py --repeats 5
  .meta-prompting\\Scripts\\python.exe _apples_batch_meanstd.py --repeats 5 --split all   # train+val (9 weakspots)
"""
import os, json, csv, argparse, statistics, datetime, traceback
import dotenv; dotenv.load_dotenv(override=True)

import logging
logging.basicConfig(level=logging.WARNING)
for _n in ("pipeline.function_caller", "LiteLLM", "litellm", "httpx", "dspy"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

import config  # noqa: F401  pin repo config before mbd_data swaps it
import dspy
from config import reconfigure_lm
import utils.kb_parser as kb
_REAL_NEW_KB = dict(kb.NEW_WEAKSPOT_GUIDELINES)     # the BU-Excel new KB (before any toggle)
from utils.mbd_data import load_mbd_dataset, all_fc_entries, stratified_split_mbd_dataset
from extract_mbd_full import MBDFullPipelineModule, prepare_full_examples
from signatures.raw_completion_adapter import RawCompletionAdapter
# Import scoring + the ACTUAL weights/rule-constants so this script never drifts from the pipeline.
from evaluation import mbd_metric
from evaluation.mbd_metric import (
    _compute_all_components,
    WEIGHT_M1_IO, WEIGHT_M2_SIGNAL, WEIGHT_M3_STRUCTURAL,
    WEIGHT_M4_COVERAGE, WEIGHT_M7_DLL, WEIGHT_M8_FUNCTIONAL,
    M1_IO_CHECK_WEIGHT, M3_OVERLAP_WEIGHT, M8_GATE_ON_WRONG_IO,
)

# composite weights = the EXACT WEIGHT_* the pipeline uses (mbd_testcase_metric)
W = [("m1", WEIGHT_M1_IO), ("m2", WEIGHT_M2_SIGNAL), ("m3", WEIGHT_M3_STRUCTURAL),
     ("m4", WEIGHT_M4_COVERAGE), ("m7", WEIGHT_M7_DLL), ("m8", WEIGHT_M8_FUNCTIONAL)]
COMPONENTS = [k for k, _ in W]
CFG = ["V1", "V2", "V3", "V4", "V5"]
M7_REF_PIN = 4   # pin M7 reference so ALL varieties share one retry scale (V5's 4-retry solve == 0.2,
                 # V1-V4's 1-retry solve == 0.8); equals the max(4,max_retries) default for these budgets.

ap = argparse.ArgumentParser(description="V1..V5 x repeats x models -> per-metric + composite mean/std.")
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--models", nargs="+", default=["azure/gpt-4.1-mini", "azure/gpt-5-mini"])
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--split", choices=["val", "train", "all"], default="val",
                help="which weakspots to score: val (default), train, or all (train+val)")
ap.add_argument("--limit", type=int, default=0, help="limit #examples (0 = all in the chosen split)")
ap.add_argument("--required-io-outputs", action="store_true",
                help="Override Prompt-2 Expected Results with the Required-I/O KB required_outputs (steer generation to an expert-chosen output). No KB entry -> keep instance_names.")
ap.add_argument("--required-io-path", default="",
                help="Path to the Required-I/O KB JSON (default utils/required_io.json); drives scoring + the output override.")
ap.add_argument("--label", default="",
                help="Tag inserted into the output JSON/CSV filenames (e.g. 'final' vs 'intermediate') to distinguish runs.")
ap.add_argument("--no-cot", action="store_true",
                help="Use plain dspy.Predict instead of ChainOfThought (matches run.py's raw call_llm; default is CoT).")
ap.add_argument("--varieties", nargs="+", default=["V1", "V2", "V3", "V4", "V5"],
                choices=["V1", "V2", "V3", "V4", "V5"],
                help="Which varieties to run (default all 5). Run a subset to save cost, e.g. --varieties V1. "
                     "Only the pipelines needed for the requested varieties are executed (V1/V2 share one).")
args = ap.parse_args()
CFG = list(args.varieties)   # [wrapper-only] restrict the run to requested varieties (per_rep/report/loop all key off CFG)
if args.required_io_path:  # [wrapper-only] re-point the KB (scoring + prompt override) BEFORE any resolve
    from utils.required_io import set_kb_path as _set_kb_path
    _set_kb_path(args.required_io_path)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apples_compare", "batch_meanstd")
os.makedirs(OUT, exist_ok=True)
_exkey = lambda e: f"{e.fc_name}#ws{e.weakspot_id}"


def _set_kb(use_new):  # toggle the KB the injection AND M4 resolve against
    kb.NEW_WEAKSPOT_GUIDELINES = _REAL_NEW_KB if use_new else {}


def _score(ex, pred, use_new_kb):  # score all M1..M8 under the chosen KB
    _set_kb(use_new_kb)
    return _compute_all_components(ex, pred)


def _composite(comp):  # weighted composite in [0,1] using the pipeline's own weights
    return sum(w * float(comp.get(k) or 0.0) for k, w in W)


def _ms(vals):
    return (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)


# ── active-rules banner (self-documents the run + goes into the saved JSON) ──
RULES = {
    "composite_weights": {k: w for k, w in W},
    "M1_IO_CHECK_WEIGHT": M1_IO_CHECK_WEIGHT,
    "M3_OVERLAP_WEIGHT": M3_OVERLAP_WEIGHT,
    "M8_GATE_ON_WRONG_IO": M8_GATE_ON_WRONG_IO,
    "M7_REF_PIN": M7_REF_PIN,
    "required_io_in_prompt": bool(args.required_io_outputs),
    "required_io_path": args.required_io_path or "utils/required_io.json (default)",
    "M8_FULL_RESOLVE": os.getenv("MBD_M8_FULL_RESOLVE", "1") not in ("0", "false", "False"),
    "M4_judge_model": os.getenv("M4_JUDGE_MODEL_NAME", "azure/gpt-5.2"),
    "temperature": args.temperature, "cache": False,
    "cot": (not args.no_cot), "varieties": list(CFG),
}
print("=" * 78)
print("ACTIVE METRIC RULES (imported live from evaluation.mbd_metric):")
print(f"  composite weights: " + "  ".join(f"{k}={w}" for k, w in W) + f"  (sum={sum(w for _,w in W):.2f})")
print(f"  M1_IO_CHECK_WEIGHT={M1_IO_CHECK_WEIGHT}  M3_OVERLAP_WEIGHT={M3_OVERLAP_WEIGHT}  "
      f"M8_GATE_ON_WRONG_IO={M8_GATE_ON_WRONG_IO}  M7_ref_pin={M7_REF_PIN}")
print(f"  M4 judge={RULES['M4_judge_model']}  temp={args.temperature}  cache=off  "
      f"repeats={args.repeats}  split={args.split}  models={args.models}")
print(f"  CoT={'on' if not args.no_cot else 'OFF (plain Predict)'}  varieties={CFG}")
print("=" * 78)

results = {}      # model -> variety -> [per-repeat {component mean over examples, composite, _n_examples}]
per_example = {}  # model -> variety -> exkey -> [per-repeat composite]
fail_count = {}   # model -> #failed (example,repeat) generations

for model in args.models:
    print(f"\n########## MODEL: {model}  (temp={args.temperature}, cache=off; M4 judge={RULES['M4_judge_model']}) ##########")
    reconfigure_lm(model, cache=False, temperature=args.temperature)   # inference LM (independent draws)
    dspy.configure(adapter=RawCompletionAdapter())                     # run.py-faithful transport
    ds = prepare_full_examples(load_mbd_dataset(fc_entries=all_fc_entries()))
    tr, val, _ = stratified_split_mbd_dataset(ds, train_ratio=0.6, val_ratio=0.4, test_ratio=0.0)
    pool = {"val": list(val), "train": list(tr), "all": list(tr) + list(val)}[args.split]
    examples = pool[:args.limit] if args.limit else pool
    print(f"  {args.split} examples: {len(examples)}  ({', '.join(_exkey(e) for e in examples)})")

    gen = MBDFullPipelineModule(use_cot=not args.no_cot)  # [wrapper-only] CoT default; --no-cot -> plain Predict
    gen.m7_reference_retries = M7_REF_PIN   # one M7 scale across ALL varieties (comparability)
    gen.required_io_in_prompt = bool(args.required_io_outputs)  # [wrapper-only] opt-in Prompt-2 output override (fc/ws flow via ex.inputs() from prepare_full_examples)

    per_rep = {c: [] for c in CFG}
    pex = {c: {} for c in CFG}
    fails = 0
    for r in range(args.repeats):
        rep_comp = {c: [] for c in CFG}          # per-example component dicts, this repeat
        for _ei, ex in enumerate(examples, 1):   # [progress] 1-indexed example counter for a live heartbeat
            k = _exkey(ex)
            print(f"  [repeat {r+1}/{args.repeats}] example {_ei}/{len(examples)}  {k}  running...", flush=True)  # [progress] print BEFORE the slow pipelines so silence never looks like a hang
            try:
                need = set(CFG)                          # [wrapper-only] only run the pipelines the requested varieties need
                res = {}                                 # [wrapper-only] variety -> component dict
                # V1/V2: shared no-inject pipeline; only M4's KB differs (one pipeline serves both)
                if need & {"V1", "V2"}:
                    gen.inject_kb_guidelines = False; gen.max_retries = 1; _set_kb(False)
                    pred_ni = gen(**ex.inputs())
                    if "V1" in need: res["V1"] = _score(ex, pred_ni, use_new_kb=False)
                    if "V2" in need: res["V2"] = _score(ex, pred_ni, use_new_kb=True)
                    print("      " + "  ".join(f"{v}={100*_composite(res[v]):.1f}%" for v in ("V1", "V2") if v in res), flush=True)  # [progress]
                # V3: inject OLD KB (new emptied), M4 old KB
                if "V3" in need:
                    gen.inject_kb_guidelines = True; gen.max_retries = 1; _set_kb(False)
                    res["V3"] = _score(ex, gen(**ex.inputs()), use_new_kb=False)
                    print(f"      V3={100*_composite(res['V3']):.1f}%", flush=True)  # [progress]
                # V4: inject NEW KB, M4 new KB
                if "V4" in need:
                    gen.inject_kb_guidelines = True; gen.max_retries = 1; _set_kb(True)
                    res["V4"] = _score(ex, gen(**ex.inputs()), use_new_kb=True)
                    print(f"      V4={100*_composite(res['V4']):.1f}%", flush=True)  # [progress]
                # V5: inject NEW KB + 4 retries, M4 new KB  (SLOWEST — 4 solver retries)
                if "V5" in need:
                    gen.inject_kb_guidelines = True; gen.max_retries = 4; _set_kb(True)
                    res["V5"] = _score(ex, gen(**ex.inputs()), use_new_kb=True)
                    gen.max_retries = 1
                    print(f"      V5={100*_composite(res['V5']):.1f}%", flush=True)  # [progress]
                print(f"      (example {_ei}/{len(examples)} done)", flush=True)  # [progress]
                for c in CFG:                            # [wrapper-only] record only the requested varieties
                    rep_comp[c].append(res[c])
                    pex[c].setdefault(k, []).append(round(100.0 * _composite(res[c]), 2))
            except Exception as e:  # noqa: BLE001 — keep the long batch alive; log which example failed
                fails += 1
                print(f"    [WARN] {model} repeat {r+1} {k} FAILED: {e}")
                traceback.print_exc()
        # per-repeat mean of each component (over examples) + weighted composite
        for c in CFG:
            comps = rep_comp[c]
            if not comps:
                continue
            rm = {k: statistics.mean([float(x.get(k) or 0.0) for x in comps]) for k in COMPONENTS}
            rm["composite"] = 100.0 * sum(w * rm[k] for k, w in W)
            rm["_n_examples"] = len(comps)
            per_rep[c].append(rm)
        print(f"  repeat {r+1}/{args.repeats} done — " +
              "  ".join(f"{c}={per_rep[c][-1]['composite']:.1f}%" for c in CFG if per_rep[c]))
    results[model] = per_rep
    per_example[model] = pex
    fail_count[model] = fails

# ── aggregate + print tables ──
report = {}
for model, per_rep in results.items():
    report[model] = {}
    print(f"\n================ {model} — mean ± std over {args.repeats} repeat(s)"
          f"{' (%d gen failures)' % fail_count[model] if fail_count[model] else ''} ================")
    print("variety   " + "  ".join(f"{k.upper():>13}" for k in COMPONENTS + ["COMPOSITE"]))
    for c in CFG:
        reps = per_rep[c]
        if not reps:
            print(f"{c:8}  (no data)"); continue
        row, cells = {}, []
        for k in COMPONENTS + ["composite"]:
            m, s = _ms([rep[k] for rep in reps])
            row[k] = {"mean": m, "std": s}
            cells.append(f"{m:6.1f}±{s:4.1f}" if k == "composite" else f"{m:5.3f}±{s:5.3f}")
        report[model][c] = row
        print(f"{c:8}  " + "  ".join(cells))
    # per-example composite mean±std (which weakspot drives the spread / repeatability)
    print(f"  --- per-example composite % (mean±std over repeats) ---")
    exkeys = sorted({k for c in CFG for k in per_example[model][c]})
    print("  weakspot".ljust(34) + "  ".join(f"{c:>11}" for c in CFG))
    for k in exkeys:
        cells = []
        for c in CFG:
            vals = per_example[model][c].get(k, [])
            cells.append(f"{statistics.mean(vals):5.1f}±{(statistics.stdev(vals) if len(vals)>1 else 0.0):4.1f}" if vals else "     -    ")
        print(f"  {k[:32]:32}" + "  ".join(cells))

# ── write JSON + CSV ──
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_tag = f"{args.label}_" if getattr(args, "label", "") else ""  # [wrapper-only] label the output files (e.g. 'final' vs 'intermediate')
json_path = os.path.join(OUT, f"batch_meanstd_{_tag}{ts}.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "config": {"repeats": args.repeats, "models": args.models, "temperature": args.temperature,
                   "split": args.split, "limit": args.limit, "rules": RULES,
                   "metric_code_version": {"M1_IO_CHECK_WEIGHT": M1_IO_CHECK_WEIGHT,
                                            "M3_OVERLAP_WEIGHT": M3_OVERLAP_WEIGHT,
                                            "M8_GATE_ON_WRONG_IO": M8_GATE_ON_WRONG_IO}},
        "report": report, "raw_per_repeat": results, "per_example_composite": per_example,
        "fail_count": fail_count,
    }, f, indent=2, default=str)

csv_path = os.path.join(OUT, f"batch_meanstd_{_tag}{ts}.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["model", "variety", "metric", "mean", "std"])
    for model in args.models:
        for c in CFG:
            row = report.get(model, {}).get(c, {})
            for k in COMPONENTS + ["composite"]:
                if k in row:
                    wr.writerow([model, c, k, f"{row[k]['mean']:.4f}", f"{row[k]['std']:.4f}"])
print(f"\nsaved -> {json_path}\nsaved -> {csv_path}")
