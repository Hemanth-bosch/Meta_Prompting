#!/usr/bin/env python3
"""SECOND DATA POINT: baseline on the CURRENT val set scored with the OLD KB ONLY (no new KB).

Compares, on the SAME 4 current-val examples and the SAME seed prompts, the M1..M8 composite
under NEW-KB vs OLD-KB-only. To remove LLM run-to-run variance it runs each example's pipeline
ONCE and scores M4 both ways on the identical prediction — so M1/M2/M3/M6/M7/M8 are byte-identical
and ONLY M4 (and the composite) change. This isolates the pure knowledge-base effect.

NO production code is changed. "Old KB only" is achieved by emptying the in-memory new-KB dict
(utils.kb_parser.NEW_WEAKSPOT_GUIDELINES = {}) so the existing resolve_guidelines() falls back to
the old KB via its normal path.

Run (project venv, needs Azure like a normal run):
    .meta-prompting\\Scripts\\python.exe _eval_oldkb_current_val.py
"""
import os, sys, json, logging
import dotenv; dotenv.load_dotenv(override=True)

# Silence the benign MIDAS DLL-fallback flood ("Function not available in the DLL …" for
# Operation/Local/Variable/etc. node types — handled in Python, not the DLL). extract_mbd_full.main()
# suppresses these via a log filter; this standalone script must do it too, else M7/M8's DLL calls
# print thousands of lines (×2, since we score each example with both KBs). Also quiet litellm/httpx.
logging.basicConfig(level=logging.WARNING)
for _noisy in ("pipeline.function_caller", "LiteLLM", "litellm", "httpx", "dspy"):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

# Pin the REPO config in sys.modules BEFORE importing mbd_data/extract_mbd_full (which do the
# MBD-config swap). Without this, the swap can delete sys.modules['config'] and later
# `from config import reconfigure_lm` resolves the MBD config instead. (Same guard as
# _verify_pipeline_equivalence.py.)
import config  # noqa: F401

# ── Import kb_parser and CAPTURE the real new KB; toggling this global flips the KB source ──
import utils.kb_parser as kb
_REAL_NEW_KB = dict(kb.NEW_WEAKSPOT_GUIDELINES)     # the BU-Excel-derived new KB
print(f"[setup] new KB has {len(_REAL_NEW_KB)} blocks; old KB has {len(kb.WEAKSPOT_GUIDELINES)} blocks")

import dspy
from config import reconfigure_lm
from utils.mbd_data import load_mbd_dataset, all_fc_entries, stratified_split_mbd_dataset
from extract_mbd_full import MBDFullPipelineModule, prepare_full_examples
from evaluation.mbd_metric import _compute_all_components

W = [("m1", 0.15), ("m2", 0.15), ("m3", 0.10), ("m4", 0.20), ("m6", 0.00), ("m7", 0.20), ("m8", 0.20)]

def _blk(ex):
    op = json.loads(ex.equation_json).get(ex.weakspot_op_key, {})
    return op.get("function_name") or op.get("class_method_name") or op.get("method_name") or "?"

def _composite(s):
    return sum(w * s[k] for k, w in W)

def main():
    model = os.getenv("MODEL_NAME", "azure/gpt-4.1-mini")
    reconfigure_lm(model, cache=True)   # same as a normal baseline run (az path is cache=False anyway)

    # SAME dataset + split as the GEPA runs (all-tasks, stratified 60/40, seed 42) → the current
    # 4-example VAL set, so this lines up 1:1 with the --eval-only / GEPA-baseline val numbers.
    ds = prepare_full_examples(load_mbd_dataset(fc_entries=all_fc_entries()))
    train, val, test = stratified_split_mbd_dataset(ds, train_ratio=0.6, val_ratio=0.4, test_ratio=0.0)
    eval_set = list(val)
    print(f"\ncurrent VAL set (n={len(eval_set)}):")
    for ex in eval_set:
        print(f"   {ex.fc_name:24} ws={ex.weakspot_id:<3} {_blk(ex)}")

    gen = MBDFullPipelineModule(use_cot=True)   # SEED prompts (baseline; no optimized state loaded)

    rows_new, rows_old, per_ex = [], [], []
    for ex in eval_set:
        pred = gen(**ex.inputs())               # ONE pipeline run (LLM + DLL) per example
        kb.NEW_WEAKSPOT_GUIDELINES = _REAL_NEW_KB      # score with NEW KB (new->old)
        s_new = _compute_all_components(ex, pred)
        kb.NEW_WEAKSPOT_GUIDELINES = {}                # score with OLD KB ONLY
        s_old = _compute_all_components(ex, pred)
        rows_new.append(s_new); rows_old.append(s_old)
        per_ex.append((_blk(ex), s_new, s_old))

    def table(title, rows):
        n = len(rows)
        avg = {k: sum(r[k] for r in rows) / n for k, _ in W}
        comp = sum(w * avg[k] for k, w in W)
        print(f"\n================ {title} (n={n}) ================")
        print(f"  {'metric':6} {'weight':6} {'avgScore':9} {'contrib':8} points")
        for k, w in W:
            print(f"  {k.upper():6} {w:<6.2f} {avg[k]:<9.4f} {w*avg[k]:<8.4f} {w*avg[k]*100:.1f}")
        print(f"  COMPOSITE baseline = {comp*100:.1f}%")
        return comp * 100

    print("\n--- per-example M4 (only M4 differs between the two KBs) ---")
    for blk, sn, so in per_ex:
        print(f"   {blk:26} M4_new={sn['m4']:.3f} [{sn['m4_details'][:34]}]   M4_old={so['m4']:.3f} [{so['m4_details'][:34]}]")

    new_pct = table("NEW-KB baseline (current val)", rows_new)
    old_pct = table("OLD-KB-ONLY baseline (current val)", rows_old)
    print(f"\n==> DATA POINTS (same 4 val, same seed prompts, same predictions):")
    print(f"    NEW KB : {new_pct:.1f}%   |   OLD KB only : {old_pct:.1f}%   |   KB effect : {new_pct-old_pct:+.1f} pts")

if __name__ == "__main__":
    main()
