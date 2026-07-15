"""Compare the FINAL output test cases (run.py vs extract) for each val example.
Evaluates: schema, #test-cases, and the signals used (precond / action-input / expected-output),
which is the semantic-fidelity question (values differ due to LLM nondeterminism)."""
import json, re, os

PAIRS = [
    ("GearLvrPosn_Vd ws0", "apples_compare/run_py/run_20260707_122340/GearLvrPosn_Vd/getBit_unsigned_U8_testcases.json",
     "apples_compare/extract_GearLvrPosn_Vd_ws0/final_test_cases.json"),
    ("InjRly ws0", "apples_compare/run_py_injrly/run_20260707_124328/InjRly_PwrCtrl_1_1_0/InjRly_T15TOffD_I_testcases.json",
     "apples_compare/extract_InjRly_PwrCtrl_1_1_0_ws0/final_test_cases.json"),
    ("EGSSig ws2", "apples_compare/run_py_egssig/run_20260706_200555/EGSSig_LamSymcp_1_6_0/ratLamAdpn_LT_I_testcases.json",
     "apples_compare/extract_EGSSig_LamSymcp_1_6_0_ws2/final_test_cases.json"),
    ("oilsepactr ws20", "apples_compare/run_py_oilsep/run_20260707_124621/oilsepactr_coorr/OilSepActr_ShwdnTOnD_I_testcases.json",
     "apples_compare/extract_oilsepactr_coorr_ws20/final_test_cases.json"),
]


def sig(s):
    """'Signal = value' -> signal name; skip Wait/blank."""
    s = str(s).strip()
    if s.lower().startswith("wait") or "=" not in s:
        return None
    return s.split("=")[0].strip()


def collect(tcs):
    pre, act, exp = set(), set(), set()
    for tc in tcs:
        for p in tc.get("Pre-Condition", []):
            n = sig(p);  pre.add(n) if n else None
        for st in tc.get("Steps", []):
            for a in st.get("Actions", []):
                n = sig(a); act.add(n) if n else None
            for e in st.get("Expected Results", []):
                n = sig(e); exp.add(n) if n else None
    return pre, act, exp


def show(label, s):
    return "{" + ", ".join(sorted(x for x in s if x)) + "}"


for name, rp, ep in PAIRS:
    print("\n" + "=" * 72 + f"\n{name}\n" + "=" * 72)
    if not (os.path.exists(rp) and os.path.exists(ep)):
        print("  MISSING FILE:", rp if not os.path.exists(rp) else ep); continue
    r = json.load(open(rp, encoding="utf-8"))
    e = json.load(open(ep, encoding="utf-8"))
    rk = sorted(r[0].keys()) if r else []
    ek = sorted(e[0].keys()) if e else []
    print(f"  #test-cases      run.py={len(r):<3} extract={len(e):<3}  {'MATCH' if len(r)==len(e) else 'differ'}")
    print(f"  schema keys      run.py={rk}")
    print(f"                   extract={ek}   {'SAME SCHEMA' if rk==ek else 'DIFF SCHEMA'}")
    rp_pre, rp_act, rp_exp = collect(r)
    ep_pre, ep_act, ep_exp = collect(e)
    print(f"  Expected-output signals  run.py ={show('', rp_exp)}")
    print(f"                           extract={show('', ep_exp)}   {'SAME' if rp_exp==ep_exp else ('OVERLAP' if rp_exp&ep_exp else 'DISJOINT')}")
    print(f"  Action/input signals     run.py ={show('', rp_act)}")
    print(f"                           extract={show('', ep_act)}   {'SAME' if rp_act==ep_act else ('OVERLAP' if rp_act&ep_act else 'DISJOINT')}")
    print(f"  Pre-Condition signals    run.py ={show('', rp_pre)}")
    print(f"                           extract={show('', ep_pre)}")
    print(f"  Byte-identical output?   {'YES' if r==e else 'NO (values differ — expected, LLM nondeterminism)'}")
