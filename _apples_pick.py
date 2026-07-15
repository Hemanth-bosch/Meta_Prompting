"""TEMP: list the wrapper's VAL split so we pick ONE weakspot for the apples-to-apples run."""
import os, json
os.environ.setdefault("MBD_INJECT_KB_GUIDELINES", "0")
import config  # noqa: F401  pin repo config before mbd_data swaps it
from utils.mbd_data import load_mbd_dataset, all_fc_entries, stratified_split_mbd_dataset
from extract_mbd_full import prepare_full_examples

ds = prepare_full_examples(load_mbd_dataset(fc_entries=all_fc_entries()))
train, val, test = stratified_split_mbd_dataset(ds, train_ratio=0.6, val_ratio=0.4, test_ratio=0.0)
print(f"\n=== dataset={len(ds)} train={len(train)} val={len(val)} test={len(test)} ===")
print("\nVAL SET (index | fc_name | ws_id | op_key | fn | raster | #equations | dll?):")
for i, ex in enumerate(val):
    eqj = json.loads(ex.equation_json) if ex.equation_json else {}
    op = eqj.get(ex.weakspot_op_key, {}) or {}
    fn = op.get("function_name") or op.get("class_method_name") or op.get("method_name") or "?"
    # crude: count solvable relations (proxy for how much Prompt-4 solving happens)
    try:
        rel = json.loads(getattr(ex, "relations_json", "{}") or "{}")
    except Exception:
        rel = {}
    try:
        eqres = json.loads(getattr(ex, "equation_results_json", "[]") or "[]")
    except Exception:
        eqres = []
    dll = bool(getattr(ex, "dll_path", "")) and os.path.exists(getattr(ex, "dll_path", ""))
    print(f"  [{i}] {ex.fc_name:24} ws={ex.weakspot_id:<3} op={ex.weakspot_op_key:<6} fn={fn:<28} "
          f"raster={ex.raster_time:<6} rel={len(rel)} eqres={len(eqres)} dll={dll}")
