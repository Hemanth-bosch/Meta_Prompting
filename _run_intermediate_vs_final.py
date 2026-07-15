"""[wrapper-only] Driver: capture EVERY weakspot twice — once steered to the FINAL physical output
(utils/required_io.json) and once to the INTERMEDIATE weakspot-op output (utils/required_io_intermediate.json).

Runs variety 4 (inject NEW KB, M4 new, 1 retry) via _apples_capture_variety.py, with --required-io-outputs
ON so Prompt 2's Expected Results is driven by the chosen KB. Covers all 10 examples (both a2mamg#0 ops,
disambiguated by --op $op4 getBit / $op5 putBit) x {final, intermediate} = 20 captures.

Output folders: apples_compare/variety_V4_injectNEW_M4new_1retry_<fc>_ws<ws>_<final|intermediate>[_<tag>]

Run:  .meta-prompting\\Scripts\\python.exe _run_intermediate_vs_final.py
      (optionally set MODEL_NAME, e.g. $env:MODEL_NAME="azure/gpt-4.1-mini")
"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable                                   # the .meta-prompting python running this driver
CAP = os.path.join(HERE, "_apples_capture_variety.py")
VARIETY = 4

# (fc, ws, weakspot_op_key or "", tag or "") — op+tag disambiguate the a2mamg#0 duplicate
WEAKSPOTS = [
    ("a2mamg",                0, "$op4", "getBit"),   # example 0: getBit  -> msg finals / getBit_unsigned_U8
    ("a2mamg",                0, "$op5", "putBit"),   # example 1: putBit  -> PT_stTraShftOp / putBit_unsigned
    ("InjRly_PwrCtrl_1_1_0",  0, "",     ""),
    ("Rail_TempLim_50.8.0",   2, "",     ""),
    ("GearLvrPosn_Vd",        0, "",     ""),
    ("EGSSig_LamSymcp_1_6_0", 2, "",     ""),
    ("EGSSig_LamSymcp_1_6_0", 5, "",     ""),
    ("GbxFanCoorn_Sp_1_0_2",  2, "",     ""),
    ("moftqptdveh_fil",       0, "",     ""),
    ("oilsepactr_coorr",     20, "",     ""),
]
KBS = [("final", "utils/required_io.json"),
       ("intermediate", "utils/required_io_intermediate.json")]

NO_COT = "--no-cot" in sys.argv   # [wrapper-only] pass-through: `python _run_intermediate_vs_final.py --no-cot` runs every capture with plain Predict
runs = []
for fc, ws, op, tag in WEAKSPOTS:
    for kb_label, kb_path in KBS:
        suffix = f"{kb_label}_{tag}" if tag else kb_label
        cmd = [PY, CAP, "--variety", str(VARIETY), "--fc", fc, "--ws", str(ws),
               "--required-io-outputs", "--required-io-path", kb_path, "--out-suffix", suffix]
        if op:
            cmd += ["--op", op]
        if NO_COT:
            cmd.append("--no-cot")
        label = f"{fc}#{ws}{('/'+op) if op else ''} [{kb_label}]"
        runs.append((label, cmd))

PER_CAPTURE_TIMEOUT_S = 900   # kill a single capture after 15 min so ONE hung DLL can't block the other 19

def main():
    print(f"Running {len(runs)} captures — variety {VARIETY}, model={os.getenv('MODEL_NAME','azure/gpt-4.1-mini')}, "
          f"M8_FULL_RESOLVE={os.getenv('MBD_M8_FULL_RESOLVE','1') not in ('0','false','False')}\n")
    fails = []
    for i, (label, cmd) in enumerate(runs, 1):
        print(f"[{i:2d}/{len(runs)}] {label}")
        print(f"        {' '.join(cmd)}")
        try:                                                   # per-capture timeout so one hang doesn't stop the sweep
            rc = subprocess.run(cmd, cwd=HERE, timeout=PER_CAPTURE_TIMEOUT_S).returncode
        except subprocess.TimeoutExpired:
            fails.append(label + " [TIMEOUT]")
            print(f"        [TIMEOUT >{PER_CAPTURE_TIMEOUT_S}s — killed, continuing]")
            continue
        if rc != 0:
            fails.append(label + f" [rc={rc}]")
            print(f"        [FAILED rc={rc}]")
    print("\n" + "=" * 66)
    print(f"DONE — {len(runs) - len(fails)}/{len(runs)} succeeded.")
    if fails:
        print("NOT-OK:", *fails, sep="\n  ")
    else:
        print("All 20 captures written under apples_compare/  (suffix _final / _intermediate).")


if __name__ == "__main__":
    main()
