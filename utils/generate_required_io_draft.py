"""[wrapper-only] Bootstrap helper: auto-DRAFT candidate Required-I/O entries from equation metadata — no run.py analog.

For every loadable weakspot it derives a CANDIDATE required set from the filtered equation graph and writes
``utils/required_io_draft.json``. This is a REVIEW AID ONLY — a human must curate it (keep only what they are
sure of; trim forward-cone signals and fixed `_C` calibrations) into the real ``utils/required_io.json``.

Candidate derivation (see the confounds flagged inline):
  required_inputs   = (union of unique_variables ∪ unique_params over all ops) − instance_names − {dT}
                      (`_C`-suffixed params are FLAGGED — some are fixed calibrations the LLM should NOT set)
  required_preconds = union of unique_consts (SystemConstants) − instance_names  (config the precond declares)
  required_outputs  = instance_names  (already matches every expert we checked)

Run:  .meta-prompting\\Scripts\\python.exe utils\\generate_required_io_draft.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root on path

import logging
logging.basicConfig(level=logging.CRITICAL)
for _n in ("pipeline.function_caller", "LiteLLM", "litellm", "httpx", "dspy", "mbd.data", "run"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

import config  # noqa: F401  pin repo config before mbd_data swaps it
from utils.mbd_data import load_mbd_dataset, all_fc_entries


def _candidate(ex) -> dict:
    """Derive a candidate {required_inputs, required_preconds, required_outputs, _flags} from one Example."""
    eq = json.loads(getattr(ex, "equation_json", "{}") or "{}")
    outs = {x.strip() for x in (getattr(ex, "instance_names", "") or "").split(",") if x.strip()}
    req_in, req_pc, flags = set(), set(), []
    for v in eq.values():
        if not isinstance(v, dict):
            continue
        req_in |= set(v.get("unique_variables") or []) | set(v.get("unique_params") or [])
        req_pc |= set(v.get("unique_consts") or [])
    req_in = {x for x in req_in if x and x.lower() != "dt"} - outs   # exclude dT + the outputs
    req_pc = req_pc - outs
    calib = sorted(x for x in req_in if x.endswith("_C"))            # FLAG fixed-calibration candidates
    if calib:
        flags.append(f"REVIEW _C params (maybe fixed calibrations, drop if not test-driven): {calib}")
    if len(req_in) > 8:
        flags.append(f"LARGE input set ({len(req_in)}) - likely includes forward-cone signals; trim to what a test must drive")
    return {
        "required_inputs": sorted(req_in),
        "required_preconds": sorted(req_pc),
        "required_outputs": sorted(outs),
        "_flags": flags,
    }


def main():
    ds = load_mbd_dataset(fc_entries=all_fc_entries())
    draft = {"_README": "AUTO-DRAFT candidates for review. Curate into required_io.json — keep ONLY what you are sure of. "
                         "required_inputs excludes dT (Prompt 1 forbids altering it); review _C params & large sets (_flags)."}
    for ex in ds:
        key = f"{getattr(ex, 'fc_name', '?')}#{getattr(ex, 'weakspot_id', '?')}"
        draft[key] = _candidate(ex)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "required_io_draft.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(draft, f, indent=2, ensure_ascii=False)
    print(f"wrote {len([k for k in draft if not k.startswith('_')])} candidate entries -> {out}")
    for k, v in draft.items():
        if k.startswith("_"):
            continue
        print(f"\n{k}:  inputs={len(v['required_inputs'])}  preconds={len(v['required_preconds'])}  outputs={v['required_outputs']}")
        for fl in v["_flags"]:
            print(f"    [REVIEW] {fl}")


if __name__ == "__main__":
    main()
