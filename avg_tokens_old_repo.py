r"""avg_tokens_old_repo.py — average inference tokens per val sample, measured through
the OLD (meta-prompting -16-07) repo's own pipeline, on the fixed 82-pair seed-42 val set.

Mirrors avg_tokens_new_repo.py so the two are directly comparable. Runs the seed program
`dspy.Predict(DCAImageComparison)` with the v1 MIDAS instruction injected (exactly like the
tournament seed) on the same 82 val pairs from dca_split_seed42.json, and reports avg
tokens/val-sample computed TWO ways:
  * history-diff  — the OLD repo's NATIVE method (tour_utils._extract_token_stats_from_history):
                    sum(prompt+completion[+reasoning]) from lm.history entries / N
  * get_lm_usage  — the NEW repo's method: sum(prompt+completion) from pred.get_lm_usage()

If the two repos' scripts print the SAME totals on the same 82 val, then the historical
3046-vs-3544 difference was NOT a pipeline/counting difference but a DSPy-version artifact
(the old runs were made months earlier) or a provenance issue (a raw-client number). Only
WORKER-forward inference is counted (the DCA judge is a raw Azure client, outside DSPy).

RUN (in the (meta-prompt-dev) terminal, env loaded):
  python avg_tokens_old_repo.py \
      --split-json ".\.cache\dca_split_seed42.json" \
      --prompt-file ".\evaluation\dca_baseline_runs\human_prompts\v1_prompt_midas.md" \
      --out old_repo_val_tokens.json
"""

import argparse
import json
import os
import sys
from statistics import mean

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def _g(o, k):
    if isinstance(o, dict):
        return o.get(k)
    return getattr(o, k, None)


def measure(program, inputs, lm):
    import dspy

    h0 = len(getattr(lm, "history", []))
    with dspy.context(track_usage=True):
        pred = program(**inputs)
    u = (pred.get_lm_usage() if hasattr(pred, "get_lm_usage") else {}) or {}
    gp = sum(int(v.get("prompt_tokens", 0) or 0) for v in u.values() if isinstance(v, dict))
    gc = sum(int(v.get("completion_tokens", 0) or 0) for v in u.values() if isinstance(v, dict))
    hp = hc = hr = ht = 0
    for e in lm.history[h0:]:
        usage = _g(e, "usage") or _g(_g(e, "response"), "usage") or {}
        pt = int(_g(usage, "prompt_tokens") or _g(usage, "input_tokens") or 0)
        ct = int(_g(usage, "completion_tokens") or _g(usage, "output_tokens") or 0)
        tt = _g(usage, "total_tokens")
        rt = int(_g(_g(usage, "completion_tokens_details"), "reasoning_tokens") or 0)
        hp += pt; hc += ct; hr += rt
        ht += int(tt) if tt is not None else (pt + ct)
    return {"getlm_prompt": gp, "getlm_completion": gc, "getlm_total": gp + gc,
            "hist_prompt": hp, "hist_completion": hc, "hist_reasoning": hr, "hist_total": ht,
            "n_calls": len(lm.history) - h0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--prompt-file", required=True, help="v1_prompt_midas.md (injected as seed instruction)")
    ap.add_argument("--images-dir", default="", help="override pairs_dir from the split json")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="worker LM max output tokens (set the SAME in both scripts, e.g. 4096, "
                         "to normalize the output-token difference; default = repo default)")
    ap.add_argument("--temperature", type=float, default=0.0, help="worker LM temperature (default 0)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import dspy
    from config import create_lm, DEFAULT_MODEL_NAME
    from signatures.dca import DCAImageComparison

    split = json.load(open(args.split_json, encoding="utf-8"))
    val_names = list(split["val"]["pair_names"])
    pairs_dir = args.images_dir or split.get("pairs_dir")
    midas = open(args.prompt_file, encoding="utf-8").read()

    # OLD repo's worker LM (Azure gpt-4.1-mini), cache OFF so tokens are counted fresh.
    _lm_kwargs = {"temperature": args.temperature, "cache": False}
    if args.max_tokens:
        _lm_kwargs["max_tokens"] = args.max_tokens
    lm = create_lm(DEFAULT_MODEL_NAME, **_lm_kwargs)
    dspy.configure(lm=lm)
    dspy.settings.configure(track_usage=True)
    print(f"[old repo] LM kwargs: {_lm_kwargs}")

    program = dspy.Predict(DCAImageComparison.with_instructions(midas))  # seed = v1 MIDAS
    print(f"[old repo] worker={DEFAULT_MODEL_NAME} | seed instr chars={len(midas)} | val pairs={len(val_names)}")

    rows = []
    for i, name in enumerate(val_names):
        op = os.path.join(pairs_dir, name, "original.png")
        rp = os.path.join(pairs_dir, name, "reimage.png")
        try:
            img_o = dspy.Image.from_file(op) if os.path.exists(op) else None
            img_r = dspy.Image.from_file(rp) if os.path.exists(rp) else None
            m = measure(program, {"image_original": img_o, "image_reimage": img_r}, lm)
            m["pair"] = name
            rows.append(m)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(val_names)}] {name:<22} hist_total={m['hist_total']:>6} "
                      f"getlm_total={m['getlm_total']:>6} calls={m['n_calls']}")
        except Exception as e:
            print(f"  [{i+1}/{len(val_names)}] {name}: ERROR {type(e).__name__}: {e}")

    if not rows:
        print("[old repo] no rows"); return
    agg = {k: round(mean(r[k] for r in rows), 1) for k in
           ("getlm_prompt", "getlm_completion", "getlm_total",
            "hist_prompt", "hist_completion", "hist_reasoning", "hist_total")}
    agg["avg_calls_per_sample"] = round(mean(r["n_calls"] for r in rows), 2)
    print(f"\n================ [old repo] AVG tokens / val sample (N={len(rows)}) ================")
    print(f"  NATIVE (old = history-diff): input {agg['hist_prompt']} + output {agg['hist_completion']}"
          f" + reasoning {agg['hist_reasoning']} = TOTAL {agg['hist_total']}")
    print(f"  get_lm_usage (new method)  : input {agg['getlm_prompt']} + output {agg['getlm_completion']}"
          f" = TOTAL {agg['getlm_total']}")
    print(f"  avg LM calls / sample      : {agg['avg_calls_per_sample']}")
    print(f"  method delta (getlm-hist)  : {agg['getlm_total'] - agg['hist_total']:+.1f}  (≈0 => method agnostic)")
    if args.out:
        json.dump({"meta": {"repo": "old (meta-prompting-16-07)", "model": DEFAULT_MODEL_NAME,
                            "method_native": "history-diff", "seed_instr_chars": len(midas)},
                   "aggregate": agg, "n": len(rows), "pairs": rows},
                  open(args.out, "w", encoding="utf-8"), indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
