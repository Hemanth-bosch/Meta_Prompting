"""Capture prompts + inputs + responses for the 5 injection/KB/retry VARIETIES (C1..C5),
for ONE val weakspot. Records BOTH the pipeline prompts (context/testcase/solver via forward())
AND the M4 coverage-JUDGE prompt (which is where the old-vs-new KB choice shows up).

  V1  M4 old KB      | no inject | 1 retry
  V2  M4 new KB      | no inject | 1 retry        (pipeline prompts == V1; only M4 judge differs)
  V3  inject OLD KB  | M4 old KB | 1 retry
  V4  inject NEW KB  | M4 new KB | 1 retry
  V5  inject NEW KB  | M4 new KB | 4 retries      (pipeline prompts == V4 + more solver retries)

The KB dict NEW_WEAKSPOT_GUIDELINES drives BOTH the Prompt-2 injection and the M4 judge (see
kb_parser.format_kb_guidelines_block / resolve_guidelines) — same toggle the injection matrix uses.
"""
import os, json, argparse
import dotenv; dotenv.load_dotenv(override=True)

import logging
logging.basicConfig(level=logging.WARNING)
for _n in ("pipeline.function_caller", "LiteLLM", "litellm", "httpx", "dspy"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

import config  # noqa: F401  pin repo config before mbd_data swaps it
import dspy
from config import reconfigure_lm
import utils.kb_parser as kb
_REAL_NEW_KB = dict(kb.NEW_WEAKSPOT_GUIDELINES)   # the BU-Excel new KB (before any toggle)
from utils.kb_parser import format_kb_guidelines_block
from utils.mbd_data import load_mbd_dataset, all_fc_entries, stratified_split_mbd_dataset
from extract_mbd_full import MBDFullPipelineModule, prepare_full_examples
from evaluation.mbd_metric import (
    _m4_test_coverage, _compute_all_components,
    WEIGHT_M1_IO, WEIGHT_M2_SIGNAL, WEIGHT_M3_STRUCTURAL,
    WEIGHT_M4_COVERAGE, WEIGHT_M7_DLL, WEIGHT_M8_FUNCTIONAL,
)
from signatures.raw_completion_adapter import RawCompletionAdapter

VARIETY = {
    1: dict(inject=False, kb="old", retries=1, label="V1_M4old_noinject_1retry"),
    2: dict(inject=False, kb="new", retries=1, label="V2_M4new_noinject_1retry"),
    3: dict(inject=True,  kb="old", retries=1, label="V3_injectOLD_M4old_1retry"),
    4: dict(inject=True,  kb="new", retries=1, label="V4_injectNEW_M4new_1retry"),
    5: dict(inject=True,  kb="new", retries=4, label="V5_injectNEW_M4new_4retry"),
}

ap = argparse.ArgumentParser(description="Capture prompts/inputs/responses for one C1..C5 variety.")
ap.add_argument("--variety", type=int, required=True, choices=[1, 2, 3, 4, 5])
ap.add_argument("--fc", default="EGSSig_LamSymcp_1_6_0")
ap.add_argument("--ws", type=int, default=2)
ap.add_argument("--required-io-outputs", action="store_true",
                help="Override Prompt-2 Expected Results with the Required-I/O KB required_outputs (steer generation to an expert-chosen output). No KB entry -> keep instance_names.")
ap.add_argument("--required-io-path", default="",
                help="Path to the Required-I/O KB JSON (default utils/required_io.json); drives scoring + the output override.")
ap.add_argument("--out-suffix", default="",
                help="Suffix appended to the output folder name (distinguish runs of the SAME weakspot, e.g. 'final' vs 'intermediate').")
ap.add_argument("--op", default="",
                help="Select the example by weakspot_op_key too (disambiguates duplicate fc#ws, e.g. a2mamg#0 getBit $op4 vs putBit $op5).")
ap.add_argument("--m8-full-resolve", action="store_true",
                help="M8 FULL-CHAIN resolve: widen inst_to_type with equation_results chain leaves so the DLL runs from the test's PHYSICAL inputs (else M8 gates/returns None). Unresolvable chains still flag 0.")
ap.add_argument("--no-cot", action="store_true",
                help="Use plain dspy.Predict instead of ChainOfThought (matches run.py's raw call_llm; default is CoT).")
args = ap.parse_args()
if args.m8_full_resolve:  # [wrapper-only] env read at M8 call-time -> set before the pipeline runs
    os.environ["MBD_M8_FULL_RESOLVE"] = "1"
v = VARIETY[args.variety]
if args.required_io_path:  # [wrapper-only] re-point the KB (scoring + prompt override) BEFORE any resolve
    from utils.required_io import set_kb_path as _set_kb_path
    _set_kb_path(args.required_io_path)

# KB toggle (controls BOTH the Prompt-2 injection text AND the M4 judge's guidelines)
kb.NEW_WEAKSPOT_GUIDELINES = _REAL_NEW_KB if v["kb"] == "new" else {}

model = os.getenv("MODEL_NAME", "azure/gpt-4.1-mini")
reconfigure_lm(model, cache=False)
dspy.configure(adapter=RawCompletionAdapter())

ds = prepare_full_examples(load_mbd_dataset(fc_entries=all_fc_entries()))
_tr, _val, _ = stratified_split_mbd_dataset(ds, train_ratio=0.6, val_ratio=0.4, test_ratio=0.0)
_allex = list(_tr) + list(_val)  # search the WHOLE dataset (train+val) so any weakspot can be captured
ex = next(e for e in _allex if e.fc_name == args.fc and int(e.weakspot_id) == args.ws
          and (not args.op or getattr(e, "weakspot_op_key", "") == args.op))  # [wrapper-only] --op disambiguates duplicate fc#ws
inp = ex.inputs()
input_dict = {k: inp[k] for k in inp.keys()}

REPO = os.path.dirname(os.path.abspath(__file__))
_suffix = f"_{args.out_suffix}" if getattr(args, "out_suffix", "") else ""  # [wrapper-only] distinguish runs of the SAME weakspot (e.g. final vs intermediate KB)
OUT = os.path.join(REPO, "apples_compare", f"variety_{v['label']}_{args.fc.replace('.', '_')}_ws{args.ws}{_suffix}")
os.makedirs(OUT, exist_ok=True)

# what fn the KB resolves against (as forward()/M4 compute it)
_op = json.loads(input_dict.get("equation_json", "{}") or "{}").get(input_dict.get("weakspot_op_key", ""), {}) or {}
_fn = _op.get("function_name") or _op.get("class_method_name") or _op.get("method_name") or ""
injected_block = format_kb_guidelines_block(_fn) if v["inject"] else ""

gen = MBDFullPipelineModule(use_cot=not args.no_cot)  # [wrapper-only] CoT default; --no-cot -> plain Predict
gen.inject_kb_guidelines = v["inject"]
gen.max_retries = v["retries"]
gen.required_io_in_prompt = bool(args.required_io_outputs)  # [wrapper-only] opt-in Prompt-2 output override (fc/ws flow via input_dict from prepare_full_examples)

lm = dspy.settings.lm


def _render_messages(msgs):
    parts = []
    for m in msgs:
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join((b.get("text") if isinstance(b, dict) and "text" in b else json.dumps(b, default=str))
                                if isinstance(b, dict) else str(b) for b in content)
        parts.append(f"================ role: {m.get('role','?')} ================\n{content}")
    return "\n\n".join(parts)


def _render_outputs(h):
    outs = h.get("outputs", h.get("response"))
    if isinstance(outs, list):
        return "\n----\n".join((o.get("text") or json.dumps(o, default=str)) if isinstance(o, dict) else str(o) for o in outs)
    return str(outs)


def _dump(calls, prefix):
    for i, h in enumerate(calls, 1):
        msgs = h.get("messages") or []
        open(os.path.join(OUT, f"{prefix}_{i:02d}_prompt.txt"), "w", encoding="utf-8").write(
            _render_messages(msgs) if msgs else str(h.get("prompt", "")))
        open(os.path.join(OUT, f"{prefix}_{i:02d}_response.txt"), "w", encoding="utf-8").write(_render_outputs(h))
    return len(calls)


# ── attach per-metric step logging -> metrics_log.txt (mbd_metric + pipeline loggers at DEBUG) ──
# Every M1..M8 check (PASS/FAIL, resolved/abstract, DLL step, etc.) and the forward() pipeline
# trace (incl. the per-equation M7 inline computation) are written here for manual verification.
_mlog = logging.FileHandler(os.path.join(OUT, "metrics_log.txt"), mode="w", encoding="utf-8")
_mlog.setLevel(logging.DEBUG)
_mlog.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
for _ln in ("mbd_metric", "extract_mbd_full", "mbd.data"):
    _lg = logging.getLogger(_ln); _lg.setLevel(logging.DEBUG); _lg.addHandler(_mlog)

# 1) pipeline prompts (context / testcase[+injection] / solver[+retries])
start = len(lm.history)
pred = gen(**input_dict)
n_pipe = _dump(lm.history[start:], "llm_call")

# 2) FULL metric suite M1..M8 — each component's step-by-step calc is logged to metrics_log.txt;
#    M4 also fires the coverage-judge LLM (captured below). Uses the SAME extract_mbd_full metric code.
metrics_start = len(lm.history)
comps = _compute_all_components(ex, pred)               # runs M1,M2,M3,M4,M6,M7,M8 with step logging
n_m4 = _dump(lm.history[metrics_start:], "m4_judge")   # capture the M4 judge LLM call (prompt + response)
_W = [("m1", WEIGHT_M1_IO), ("m2", WEIGHT_M2_SIGNAL), ("m3", WEIGHT_M3_STRUCTURAL),
      ("m4", WEIGHT_M4_COVERAGE), ("m7", WEIGHT_M7_DLL), ("m8", WEIGHT_M8_FUNCTIONAL)]  # live weights (no drift); M6 removed
_composite = 100.0 * sum(w * float(comps.get(k) or 0.0) for k, w in _W)  # weighted composite (mbd_testcase_metric weights)
comps["_weighted_composite_pct"] = _composite
_mlog.flush()

# 3) inputs / outputs / metrics / injected block / summary
json.dump(input_dict, open(os.path.join(OUT, "00_pipeline_inputs.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
json.dump(comps, open(os.path.join(OUT, "metrics_components.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
open(os.path.join(OUT, "00_injected_kb_block.txt"), "w", encoding="utf-8").write(injected_block or "(no injection for this variety / no KB match)")
open(os.path.join(OUT, "final_test_cases.json"), "w", encoding="utf-8").write(getattr(pred, "test_cases", "") or "")
_scores = "  ".join(f"{k.upper()}={comps.get(k)}" for k, _ in _W)
open(os.path.join(OUT, "INFO.txt"), "w", encoding="utf-8").write(
    f"VARIETY {args.variety}: {v['label']}\n  weakspot={args.fc} ws={args.ws} fn={_fn}\n"
    f"  inject_kb_guidelines={v['inject']}  KB={v['kb']}  max_retries={v['retries']}  model={model}  required_io_in_prompt={gen.required_io_in_prompt}\n"
    f"  pipeline LLM calls (llm_call_*) = {n_pipe}\n  M4 judge LLM calls (m4_judge_*) = {n_m4}\n"
    f"  injected block length = {len(injected_block)} chars\n"
    f"  METRICS: {_scores}\n  weighted composite = {_composite:.1f}%\n"
    f"  M4 detail: {comps.get('m4_details')}\n  (per-step calc trace -> metrics_log.txt; scores -> metrics_components.json)\n")

print(f"\nVARIETY {args.variety} ({v['label']}) -> {OUT}")
print(f"  pipeline calls={n_pipe}  M4 judge calls={n_m4}")
print(f"  {_scores}  composite={_composite:.1f}%")
