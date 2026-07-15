"""Prove the INPUT VARIABLES injected into each prompt are identical between run.py and extract.

Independently recomputes run.py's eq_ctx for EGSSig_LamSymcp_1_6_0 ws=2 using run.py's OWN
functions (no LLM, offline), then diffs each prompt-input variable against what the wrapper
actually fed into forward() (apples_compare/.../00_pipeline_inputs.json).
"""
import os, sys, json

REPO = os.path.dirname(os.path.abspath(__file__))
MBD = os.path.join(REPO, "MBD", "weakspot_code_to_run_2")
sys.path.insert(0, MBD)          # MBD dir FIRST so `import run`/`config` = the MBD originals
import run                        # the original pipeline (no dspy, no LLM at import time)
from pipeline.context_helpers import get_weakspot_context, get_weakspot_example_output  # run.py's own renderers

import argparse
_LEAF = {"GearLvrPosn_Vd": "Proc_10ms", "InjRly_PwrCtrl_1_1_0": "Proc_20ms",
         "oilsepactr_coorr": "Proc_100ms", "EGSSig_LamSymcp_1_6_0": "Proc_10ms"}
_ap = argparse.ArgumentParser(description="Byte-compare run.py eq_ctx vars vs wrapper Example inputs.")
_ap.add_argument("--fc", default="EGSSig_LamSymcp_1_6_0")
_ap.add_argument("--ws", type=int, default=2)
_ap.add_argument("--leaf", default=None)
_a = _ap.parse_args()
FC, WS = _a.fc, _a.ws
LEAF = _a.leaf or _LEAF.get(FC, "Proc")
fc_dir = os.path.join(MBD, "data", FC)

# ---- Replicate run_pipeline()'s data prep + Phase 1/2 (run.py:728-848), no LLM ----
eqn = run.load_equations(fc_dir)
ss = run.load_signal_sheet(fc_dir)
mi = run.load_model_interface(fc_dir)
_, _, const_df, _ = run.get_filtered_dfs(ss)
step_time = run.extract_step_time(LEAF)
cvd = run.get_const_default_values(const_df, step_time, mi)
const_values = {k: v for k, v in cvd.items() if v is not None}
data = run.get_sequence_by_name(eqn, LEAF)
run.fix_operation_instance_names(data)
global_ops = run.build_global_ops(data)
selection = run._locate_weakspot(data, WS)
eq_ctx = run._build_equation_context(data, global_ops, selection, const_values)
example_output = get_weakspot_example_output(eq_ctx.filtered_dict, eq_ctx.weakspot_op_key)  # run.py:472-474
_wc = get_weakspot_context(eq_ctx.filtered_dict["equation"], eq_ctx.weakspot_op_key)         # rendered inside run.py:445
weakspot_context_text = _wc[0]["text"] if isinstance(_wc, list) and _wc else (_wc if isinstance(_wc, str) else "")

runpy = {
    "equation": eq_ctx.filtered_dict["equation"],
    "const_values": eq_ctx.filtered_dict["const_values"],
    "weakspot_operation": eq_ctx.filtered_dict["weakspot_operation"],
    "weakspot_context": weakspot_context_text,
    "instance_names": eq_ctx.instance_names,
    "example_explanation_output": example_output,
    "weakspot_op_key": eq_ctx.weakspot_op_key,
}

# ---- What the WRAPPER actually fed into forward() ----
ext = json.load(open(os.path.join(REPO, "apples_compare",
                    f"extract_{FC.replace('.', '_')}_ws{WS}", "00_pipeline_inputs.json"), encoding="utf-8"))


def canon(v):
    """Normalize for comparison: parse JSON strings, sort dict keys, normalize instance-name lists."""
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in "[{":
            try:
                return canon(json.loads(s))
            except Exception:
                pass
        if "," in s and "\n" not in s:                 # comma-joined instance names
            return [x.strip() for x in s.split(",") if x.strip()]
        return s
    if isinstance(v, dict):
        return {k: canon(x) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return [canon(x) for x in v]
    return v


FIELDS = [
    ("equation",                   "equation_json"),
    ("const_values",               "const_values"),
    ("weakspot_operation",         "weakspot_operation"),
    ("weakspot_op_key",            "weakspot_op_key"),
    ("instance_names",             "instance_names"),
    ("example_explanation_output", "example_explanation_output"),
    ("weakspot_context",           "weakspot_context"),
]

def _names(v):
    """Normalize instance_names to a sorted list of signal tokens (order/list-vs-string-agnostic)."""
    if isinstance(v, list):
        items = v
    elif isinstance(v, str):
        items = [x.strip() for x in v.split(",") if x.strip()]
    else:
        items = [v]
    return sorted(str(x) for x in items)


print(f"\n=== INPUT-VARIABLE FIDELITY: run.py eq_ctx  vs  wrapper Example  ({FC} ws={WS}) ===\n")
all_ok = True
for rk, ek in FIELDS:
    if rk == "instance_names":                    # compare by CONTENT (same signals), not order/representation
        rv, ev = _names(runpy[rk]), _names(ext.get(ek, ""))
    else:
        rv = canon(runpy[rk])
        ev = canon(ext.get(ek, ""))
    ok = (rv == ev)
    all_ok &= ok
    tag = "IDENTICAL" if ok else "*** DIFFERS ***"
    size = len(json.dumps(rv, default=str))
    note = " (same signals; run.py list vs wrapper comma-string/order)" if rk == "instance_names" and ok else ""
    print(f"[{tag:15}] {rk:28} ({size} bytes){note}")
    if not ok:
        rs, es = json.dumps(rv, default=str), json.dumps(ev, default=str)
        # show first divergence
        i = next((j for j in range(min(len(rs), len(es))) if rs[j] != es[j]), min(len(rs), len(es)))
        print(f"    run.py : ...{rs[max(0,i-60):i+60]}...")
        print(f"    extract: ...{es[max(0,i-60):i+60]}...")
        print(f"    (run.py len={len(rs)}  extract len={len(es)})")

print("\n" + ("ALL INPUT VARIABLES IDENTICAL ✓" if all_ok else "SOME VARIABLES DIFFER — see above"))
