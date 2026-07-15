#!/usr/bin/env python3
"""2x2 MATRIX: does injecting KB guidelines into the test-case-generator prompt (Prompt 2) help?

Two factors, on the SAME current 4-example VAL set (same split/seed as GEPA):
    A. inject guidelines into Prompt-2 context   : OFF / ON
    B. M4 knowledge base                          : old KB / new KB (new->old fallback)

4 configurations:
    C1  no-inject  + M4 old KB      (today's baseline, old KB)
    C2  no-inject  + M4 new KB      (today's baseline, new KB)      <- C1/C2 share ONE prediction
    C3  inject OLD-KB guidelines + M4 old KB
    C4  inject NEW-KB guidelines + M4 new KB

Both the prompt-injection and M4 read the SAME resolve_guidelines() (utils.kb_parser), so emptying
NEW_WEAKSPOT_GUIDELINES flips BOTH to old-KB in lockstep (item-5 semantics: search new KB -> append
to prompt AND use for M4; else old KB). No production code is changed; injection is driven by the
public `generator.inject_kb_guidelines` attribute + the in-memory NEW-KB dict toggle.

Noise control (n=4 val is small, so per-run sampling noise is large — esp. M2 signal-resolution):
  --repeat N   run the whole matrix N INDEPENDENT times (cache OFF) and report mean +/- std, so the
               sampling noise on M2/M4/composite averages out. Use --repeat 3 to settle the M2 dip.
  reuse        when a config's injection is EMPTY for a block (e.g. C3 on a gap block with no old-KB
               entry), REUSE the no-inject prediction instead of regenerating. The prompt is then
               byte-identical to the baseline, so this removes the spurious re-roll noise and makes
               C3 differ from C1 ONLY on blocks that were actually injected (isolates the real effect).

Cost: up to 3 forward passes/example x 4 val x N repeats; reuse skips the empty-injection re-rolls
      (C3 typically costs ~1x, not 3x). --repeat 3 on n=4 is ~35-70 min.

Run (project venv, needs Azure like a normal run):
    .meta-prompting\\Scripts\\python.exe _eval_guideline_injection_matrix.py --repeat 3
"""
import os, sys, json, logging, argparse, statistics
import dotenv; dotenv.load_dotenv(override=True)

# Silence the benign MIDAS DLL-fallback flood (see _eval_oldkb_current_val.py). x3 preds here.
logging.basicConfig(level=logging.WARNING)
for _noisy in ("pipeline.function_caller", "LiteLLM", "litellm", "httpx", "dspy"):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

# Pin the REPO config BEFORE importing mbd_data/extract_mbd_full (which do the MBD-config swap).
import config  # noqa: F401

import utils.kb_parser as kb
_REAL_NEW_KB = dict(kb.NEW_WEAKSPOT_GUIDELINES)     # the BU-Excel-derived new KB
print(f"[setup] new KB has {len(_REAL_NEW_KB)} blocks; old KB has {len(kb.WEAKSPOT_GUIDELINES)} blocks")

import dspy  # noqa: F401
from config import reconfigure_lm
from utils.mbd_data import load_mbd_dataset, all_fc_entries, stratified_split_mbd_dataset
from extract_mbd_full import MBDFullPipelineModule, prepare_full_examples
from evaluation.mbd_metric import _compute_all_components

W = [("m1", 0.15), ("m2", 0.15), ("m3", 0.10), ("m4", 0.20), ("m6", 0.00), ("m7", 0.20), ("m8", 0.20)]


def _blk(ex):
    op = json.loads(ex.equation_json).get(ex.weakspot_op_key, {})
    return op.get("function_name") or op.get("class_method_name") or op.get("method_name") or "?"


def _fn_of(ex):
    """function_name used for KB resolution — EXACTLY as forward() computes it (extract_mbd_full.py:313)."""
    op = json.loads(ex.equation_json).get(ex.weakspot_op_key, {}) if ex.equation_json else {}
    return op.get("function_name") or op.get("class_method_name") or op.get("method_name") or ""


def _score(ex, pred, use_new_kb):
    """Score M1..M8 on a fixed prediction under the chosen KB (toggles the in-memory new-KB dict)."""
    kb.NEW_WEAKSPOT_GUIDELINES = _REAL_NEW_KB if use_new_kb else {}
    return _compute_all_components(ex, pred)


def _avg(metric, score_list):
    return sum(s[metric] for s in score_list) / len(score_list) if score_list else 0.0


def _composite_pct(score_list):
    """Composite % from a list of per-example score dicts (weights W)."""
    return 100.0 * sum(w * _avg(k, score_list) for k, w in W)


def main():
    ap = argparse.ArgumentParser(description="2x2 KB-guideline-injection matrix (repeat-averaged).")
    ap.add_argument("--repeat", type=int, default=1,
                    help="independent repeats to average — cuts the n=4 sampling noise on M2/M4/composite.")
    ap.add_argument("--c5-retries", type=int, default=10,
                    help="DLL retry budget for C5 (C1-C4 fixed at 1). M7 is graded per this budget. Default 10.")
    args = ap.parse_args()
    N = max(1, args.repeat)
    C5_RETRIES = max(1, args.c5_retries)

    model = os.getenv("MODEL_NAME", "azure/gpt-4.1-mini")
    # cache=False so each --repeat is an INDEPENDENT draw (averaging only reduces noise if draws differ).
    reconfigure_lm(model, cache=False)
    # RawCompletionAdapter = run.py-faithful transport: it kills ChatAdapter's JSONAdapter fallback and
    # hands the RAW LLM completion to the output field, so the solver's ```json fence survives to
    # extract_json_from_text_dll. Matches extract_mbd_full.py exactly (was the reverted-JSONAdapter note;
    # neither JSONAdapter nor the default ChatAdapter parses the solver's fenced-array output reliably).
    from signatures.raw_completion_adapter import RawCompletionAdapter
    dspy.configure(adapter=RawCompletionAdapter())

    ds = prepare_full_examples(load_mbd_dataset(fc_entries=all_fc_entries()))
    train, val, test = stratified_split_mbd_dataset(ds, train_ratio=0.6, val_ratio=0.4, test_ratio=0.0)
    eval_set = list(val)
    labels = [_blk(ex) for ex in eval_set]
    print(f"\ncurrent VAL set (n={len(eval_set)}), repeats={N}, C5 retries={C5_RETRIES}:")
    for ex in eval_set:
        print(f"   {ex.fc_name:24} ws={ex.weakspot_id:<3} {_blk(ex)}")

    gen = MBDFullPipelineModule(use_cot=True)   # SEED prompts (baseline; no optimized state)
    gen.m7_reference_retries = C5_RETRIES       # score M7 on ONE common budget-C5_RETRIES scale for ALL configs
                                                # (C1-C4 still run 1 retry, but a 1-retry solve = 0.8 like C5 -> M7 comparable)

    CFG = ("C1", "C2", "C3", "C4", "C5")
    all_rows = {c: [] for c in CFG}              # every (repeat,example) score dict -> averaged tables
    comp_by_rep = {c: [] for c in CFG}           # per-repeat composite %  -> mean +/- std
    m2_by_rep = {c: [] for c in CFG}             # per-repeat M2 %         -> mean +/- std
    m7_by_rep = {c: [] for c in CFG}             # per-repeat M7 %         -> mean +/- std (C5's target metric)
    m4_by_idx = [{c: [] for c in CFG} for _ in eval_set]   # per-example M4 across repeats
    reuse_note = [None] * len(eval_set)          # per-example: was C3/C4 injected or reused?

    for r in range(N):
        rep_rows = {c: [] for c in CFG}
        for i, ex in enumerate(eval_set):
            fn = _fn_of(ex)

            # C1/C2 baseline (NO injection, 1 retry = run.py) — shared prediction
            gen.max_retries = 1
            gen.inject_kb_guidelines = False
            pred_ni = gen(**ex.inputs())

            # C3: inject OLD-KB guidelines (new KB emptied). REUSE the baseline when nothing is injected
            #     — gap blocks have no old-KB entry -> identical prompt -> avoid a spurious re-roll (noise).
            gen.inject_kb_guidelines = True
            kb.NEW_WEAKSPOT_GUIDELINES = {}
            blk_old = kb.format_kb_guidelines_block(fn)
            pred_io = gen(**ex.inputs()) if blk_old else pred_ni

            # C4: inject NEW-KB guidelines (new -> old fallback). REUSE only if NOTHING resolves at all.
            kb.NEW_WEAKSPOT_GUIDELINES = _REAL_NEW_KB
            blk_new = kb.format_kb_guidelines_block(fn)
            pred_in = gen(**ex.inputs()) if blk_new else pred_ni

            # C5: C4 (inject NEW-KB) + C5_RETRIES DLL retries (vs C4's 1; --c5-retries, default 10). Its own
            #     forward pass (retries change the solve loop). Tests whether more backtrace attempts lift
            #     M7 (graded) and M8. inject stays ON and NEW KB is still set from C4 above.
            gen.max_retries = C5_RETRIES
            pred_in10 = gen(**ex.inputs())
            gen.max_retries = 1                                  # reset for the next example's C1-C4

            # score the 5 configs (KB toggled inside _score)
            s = {"C1": _score(ex, pred_ni, use_new_kb=False),
                 "C2": _score(ex, pred_ni, use_new_kb=True),
                 "C3": _score(ex, pred_io, use_new_kb=False),
                 "C4": _score(ex, pred_in, use_new_kb=True),
                 "C5": _score(ex, pred_in10, use_new_kb=True)}
            for c in CFG:
                all_rows[c].append(s[c]); rep_rows[c].append(s[c])
                m4_by_idx[i][c].append(s[c]["m4"])
            if reuse_note[i] is None:
                reuse_note[i] = f"C3={'inject' if blk_old else 'REUSE'}, C4={'inject' if blk_new else 'REUSE'}"

        for c in CFG:
            comp_by_rep[c].append(_composite_pct(rep_rows[c]))
            m2_by_rep[c].append(100.0 * _avg("m2", rep_rows[c]))
            m7_by_rep[c].append(100.0 * _avg("m7", rep_rows[c]))
        print(f"  repeat {r + 1}/{N} done — " + "  ".join(f"{c}={comp_by_rep[c][-1]:.1f}%" for c in CFG))

    def _ms(vals):
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    def table(title, rows_cfg):
        print(f"\n================ {title} (n={len(rows_cfg)} = {N}x{len(eval_set)}) ================")
        print(f"  {'metric':6} {'weight':6} {'avgScore':9} {'contrib':8} points")
        for k, w in W:
            a = _avg(k, rows_cfg)
            print(f"  {k.upper():6} {w:<6.2f} {a:<9.4f} {w * a:<8.4f} {w * a * 100:.1f}")
        print(f"  COMPOSITE (mean) = {_composite_pct(rows_cfg):.1f}%")

    print("\n--- per-example M4 across the 5 configs (mean over repeats) ---")
    for i, lbl in enumerate(labels):
        v = m4_by_idx[i]
        print(f"   {lbl:24} " + "  ".join(f"{c}={statistics.mean(v[c]):.3f}" for c in CFG)
              + f"   [{reuse_note[i]}]")

    print(f"\n--- M7 (DLL pass rate; graded on ONE common budget-{C5_RETRIES} scale for all configs -> fair) per repeat ---")
    for c in CFG:
        m, sd = _ms(m7_by_rep[c])
        print(f"   {c}: " + " ".join(f"{x:.1f}" for x in m7_by_rep[c]) + f"    mean={m:.1f}  std={sd:.1f}")

    print("\n--- M2 (signal resolution) per repeat ---")
    for c in CFG:
        m, sd = _ms(m2_by_rep[c])
        print(f"   {c}: " + " ".join(f"{x:.1f}" for x in m2_by_rep[c]) + f"    mean={m:.1f}  std={sd:.1f}")

    # Solver JSON health across the WHOLE run (all configs/repeats share the one `gen` instance).
    _sj_att = getattr(gen, "_solver_attempts", 0)
    _sj_fail = getattr(gen, "_solver_json_fails", 0)
    _sj_rate = (100.0 * _sj_fail / _sj_att) if _sj_att else 0.0
    print(f"\n--- Solver JSON health (whole run): {_sj_fail}/{_sj_att} DLL solve attempts returned no "
          f"parseable JSON ({_sj_rate:.1f}% fail) ---")
    print("    high => solver starved: equations rarely solve, retries wasted, M7/M8 pinned low")

    _titles = ("C1 no-inject  | M4 old KB | 1 retry", "C2 no-inject  | M4 new KB | 1 retry",
               "C3 inject OLD | M4 old KB | 1 retry", "C4 inject NEW | M4 new KB | 1 retry",
               f"C5 inject NEW | M4 new KB | {C5_RETRIES} retries")
    for c, ttl in zip(CFG, _titles):
        table(ttl, all_rows[c])

    comp = {c: _ms(comp_by_rep[c]) for c in CFG}
    plural = "s" if N > 1 else ""
    print(f"\n==> DATA POINTS (same {len(eval_set)} val, same seed prompts; mean +/- std over {N} repeat{plural}):")
    _labs = ("no-inject/oldKB   ", "no-inject/newKB   ", "inject-old/oldKB  ",
             "inject-new/newKB  ", f"inject-new/{C5_RETRIES}ret".ljust(18))
    for c, lab in zip(CFG, _labs):
        print(f"    {c} {lab}: {comp[c][0]:.1f}%   (+/- {comp[c][1]:.1f})")
    c1, c2, c3, c4, c5 = (comp[c][0] for c in CFG)
    _m7 = {c: _ms(m7_by_rep[c])[0] for c in CFG}
    print(f"\n    KB effect  (no inject) : C2-C1 = {c2 - c1:+.1f} pts")
    print(f"    inject effect (old KB) : C3-C1 = {c3 - c1:+.1f} pts   (C3 reuses baseline on un-injected blocks)")
    print(f"    inject effect (new KB) : C4-C2 = {c4 - c2:+.1f} pts")
    print(f"    combined (new KB+inject vs old baseline) : C4-C1 = {c4 - c1:+.1f} pts")
    _m8 = {c: 100.0 * _avg("m8", all_rows[c]) for c in CFG}
    print(f"    RETRY effect ({C5_RETRIES} vs 1) : C5-C4 = {c5 - c4:+.1f} pts composite")
    print(f"       M7 (graded on the SAME budget-{C5_RETRIES} scale for all configs -> now FAIR): "
          f"{_m7['C4']:.1f}% -> {_m7['C5']:.1f}% ({_m7['C5'] - _m7['C4']:+.1f})")
    print(f"       M8 (functional, same scale): "
          f"{_m8['C4']:.1f}% -> {_m8['C5']:.1f}% ({_m8['C5'] - _m8['C4']:+.1f})")


if __name__ == "__main__":
    main()
