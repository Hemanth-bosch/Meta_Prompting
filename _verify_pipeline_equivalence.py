#!/usr/bin/env python3
"""Equivalence harness — prove the GEPA wrapper runs the MIDAS pipeline identically.

Goal (Phase B of the MBD-UTA optimization plan): show that, GIVEN the SAME LLM
outputs at every stage, `extract_mbd_full.MBDFullPipelineModule.forward()` produces
the SAME final test-case JSON as the original `run.py` pipeline — i.e. the wrapper's
pipeline logic is faithful and only the prompt WORDING is what differs/optimizes.

Method (offline, real DLL):
  1. Serve the RECORDED LLM responses (saved under
     output/run_20260331_131417/Rail_TempLim_50.8.0/prompts/) to BOTH pipelines, so
     neither makes a live LLM call. The 32-bit ASCET DLL still runs for real.
  2. ORIGINAL: monkeypatch run.call_llm + pipeline.test_generator.call_llm to a
     content-addressed server that ALSO records the order of responses served.
     Run run.run_pipeline for Rail_TempLim weakspot 2; read its output JSON.
  3. WRAPPER: build the dspy.Example for the same weakspot via utils.mbd_data, replace
     the 4 predictors with fakes that REPLAY the recorded order, run forward().
  4. Diff the two final test-case JSONs (order-insensitive).

Run (project venv):
    .meta-prompting\\Scripts\\python.exe _verify_pipeline_equivalence.py
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import re
import sys
import tempfile
from types import SimpleNamespace

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
# Cache the REPO config in sys.modules BEFORE any MBD-config swap below. Without this,
# utils.mbd_data (imported in main()) finds no cached `config`, so on cleanup it DELETES
# sys.modules['config']; extract_mbd_full:80 (`from config import reconfigure_lm`) then
# re-imports and — because _MBD_ROOT is now on sys.path — resolves the MBD config and fails.
import config  # noqa: F401,E402  (repo config; keeps it pinned as sys.modules['config'])
_MBD_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "MBD", "weakspot_code_to_run_2"))
_SAVED_DIR = os.path.join(
    _MBD_ROOT, "output", "run_20260331_131417", "Rail_TempLim_50.8.0", "prompts"
)
_FC_NAME = "Rail_TempLim_50.8.0"
_LEAF = "_Proc"
_WS_ID = 2
_WS_INSTANCE = "Rail_trqRampSwitch"  # output JSON filename stem for ws2


# ─────────────────────────────────────────────────────────────────────────────
# Recorded-response loading + content-addressed server (mirrors _trace_run.py)
# ─────────────────────────────────────────────────────────────────────────────
def _read(name: str) -> str:
    with open(os.path.join(_SAVED_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def _eq_key(text: str):
    """Stable content key for a Phase-4 solve prompt: (equation block, LHS value)."""
    m = re.search(r"given below:\s*(.*?)\s*This expression involves", text, re.S)
    eq = m.group(1).strip() if m else ""
    m2 = re.search(r"known to be (.+?)\.", text)
    lhs = m2.group(1).strip() if m2 else ""
    return (eq, lhs)


def _load_recorded():
    ctx = _read("prompt_1_context_response.txt")
    tc = _read("prompt_2_testcase_response.txt")
    init_resp, retry_resp = {}, {}
    for fn in os.listdir(_SAVED_DIR):
        if not fn.startswith("prompt_4_dll_") or fn.endswith("_response.txt"):
            continue
        resp_name = fn[:-4] + "_response.txt"
        if not os.path.isfile(os.path.join(_SAVED_DIR, resp_name)):
            continue
        key = _eq_key(_read(fn))
        if "_retry_" in fn:
            retry_resp[key] = _read(resp_name)
        else:
            init_resp[key] = _read(resp_name)
    return ctx, tc, init_resp, retry_resp


def _text_of(content) -> str:
    if isinstance(content, list):
        return "\n".join(
            it.get("text", "") if isinstance(it, dict) else str(it) for it in content
        )
    return str(content)


# ─────────────────────────────────────────────────────────────────────────────
# Config-swap helper (import MBD `run` under MBD config, like extract_mbd_full)
# ─────────────────────────────────────────────────────────────────────────────
def _import_under_mbd_config(module_names: list[str]):
    if _MBD_ROOT not in sys.path:
        sys.path.insert(0, _MBD_ROOT)
    repo_cfg = sys.modules.get("config")
    spec = importlib.util.spec_from_file_location(
        "mbd_config_equiv", os.path.join(_MBD_ROOT, "config.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["config"] = mod
    spec.loader.exec_module(mod)
    try:
        imported = [importlib.import_module(n) for n in module_names]
    finally:
        if repo_cfg is not None:
            sys.modules["config"] = repo_cfg
        else:
            sys.modules.pop("config", None)
    return imported


# ─────────────────────────────────────────────────────────────────────────────
# Comparison
# ─────────────────────────────────────────────────────────────────────────────
def _canon(tcs) -> str:
    """Order-insensitive canonical form of a list of test cases for diffing."""
    return json.dumps(tcs, sort_keys=True, ensure_ascii=False, indent=2)


def main() -> int:
    if not os.path.isdir(_SAVED_DIR):
        print(f"ERROR: recorded-response dir not found: {_SAVED_DIR}")
        return 2

    ctx_resp, tc_resp, init_resp, retry_resp = _load_recorded()
    print(f"Loaded recorded responses: ctx + tc + {len(init_resp)} init + {len(retry_resp)} retry")

    # Import the repo-config-dependent modules FIRST so sys.modules['config'] is the
    # REPO config (cached) before we temporarily swap to MBD config to import `run`.
    # (These modules each do their own internal MBD-config swap and restore repo config.)
    from utils.mbd_data import load_mbd_dataset
    from extract_mbd_full import MBDFullPipelineModule, prepare_full_examples

    served_order: list[str] = []
    _noneq = [0]

    def server(content, *a, **k):
        text = _text_of(content)
        if "given below:" in text and "final output value (LHS)" in text:
            key = _eq_key(text)
            if "does not match the expected result" in text:
                resp = retry_resp.get(key) or init_resp.get(key) or "```json\n[]\n```"
            else:
                resp = init_resp.get(key) or "```json\n[]\n```"
        else:
            resp = ctx_resp if _noneq[0] == 0 else tc_resp
            _noneq[0] += 1
        served_order.append(resp)
        return resp

    async def _aserver(content, *a, **k):
        return server(content, *a, **k)

    # ── 1) ORIGINAL run.py pipeline (offline LLM, real DLL) ───────────────────
    print("\n[1/3] Running ORIGINAL run.py pipeline for ws2 (offline LLM, real DLL)...")
    (runmod, testgen) = _import_under_mbd_config(["run", "pipeline.test_generator"])
    runmod.call_llm = _aserver
    testgen.call_llm = _aserver

    out_dir = tempfile.mkdtemp(prefix="equiv_orig_")
    ts = "equiv"
    args = SimpleNamespace(
        fc_dir=os.path.join(_MBD_ROOT, "data", _FC_NAME),
        leaf_node=_LEAF,
        weakspot_ids=[_WS_ID],
        dry_run=False,
        save_prompts=False,
        no_dll=False,
        output_json=True,
        output_dir=out_dir,
    )
    asyncio.run(runmod.run_pipeline(args, ts))
    orig_json_path = os.path.join(out_dir, f"run_{ts}", _FC_NAME, f"{_WS_INSTANCE}_testcases.json")
    if not os.path.isfile(orig_json_path):
        print(f"ERROR: original output JSON not written at {orig_json_path}")
        return 3
    with open(orig_json_path, "r", encoding="utf-8") as f:
        original_final = json.load(f)
    print(f"  original final_json: {len(original_final)} test case(s); "
          f"recorded {len(served_order)} LLM call(s) in order")

    # ── 2) Build the wrapper Example for the same weakspot ────────────────────
    print("\n[2/3] Building wrapper Example via utils.mbd_data...")
    ds = load_mbd_dataset(fc_entries=[
        {"fc_name": _FC_NAME, "leaf_node": _LEAF, "weakspot_ids": [_WS_ID], "raster_time": 100}
    ])
    ds = prepare_full_examples(ds)
    if not ds:
        print("ERROR: wrapper dataset empty (ws2 should produce 1 example)")
        return 4
    ex = ds[0]

    # ── 3) WRAPPER forward() replaying the recorded order ─────────────────────
    print("[3/3] Running WRAPPER forward() replaying recorded responses (real DLL)...")
    replay = iter(served_order)

    def _ctx_fake(**kw):
        return SimpleNamespace(context_analysis=next(replay))

    def _tc_fake(**kw):
        return SimpleNamespace(test_cases=next(replay))

    def _solve_fake(**kw):
        return SimpleNamespace(variable_assignments=next(replay))

    module = MBDFullPipelineModule(use_cot=True)
    module.context_extractor = _ctx_fake
    module.testcase_generator = _tc_fake
    module.equation_solver_dll = _solve_fake
    module.equation_solver_retry = _solve_fake

    pred = module(**ex.inputs())
    wrapper_final = json.loads(pred.test_cases)
    print(f"  wrapper final_json: {len(wrapper_final)} test case(s)")

    # ── Diff ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    o_canon, w_canon = _canon(original_final), _canon(wrapper_final)
    if o_canon == w_canon:
        print("RESULT: ✅ IDENTICAL — wrapper forward() == original run.py final_json")
        return 0
    print("RESULT: ❌ DIVERGENCE — see first differing lines below")
    o_lines, w_lines = o_canon.splitlines(), w_canon.splitlines()
    for i in range(max(len(o_lines), len(w_lines))):
        ol = o_lines[i] if i < len(o_lines) else "<missing>"
        wl = w_lines[i] if i < len(w_lines) else "<missing>"
        if ol != wl:
            print(f"  line {i}: ORIG | {ol}")
            print(f"  line {i}: WRAP | {wl}")
            break
    print("\n--- ORIGINAL final_json ---")
    print(o_canon[:2000])
    print("\n--- WRAPPER final_json ---")
    print(w_canon[:2000])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
