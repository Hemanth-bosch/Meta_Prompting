"""APPLES-TO-APPLES capture (wrapper side).

Runs extract_mbd_full's MBDFullPipelineModule.forward() on ONE val weakspot (the same one
run.py processes) and saves, for manual inspection:
  - 00_pipeline_inputs.json   : the exact variables/equations fed into forward() (the Example)
  - llm_call_NN_prompt.txt     : the FULL chat sent to the LLM for each call (system + user)
  - llm_call_NN_response.txt   : the raw completion returned
  - llm_call_NN_raw.json       : the raw dspy history entry (messages/outputs/kwargs/usage)
  - final_test_cases.json      : the pipeline's final test-case JSON (== run.py's final_json)
  - final_prediction_all_fields.txt : context_analysis / pre-postprocess / final / m7
  - INDEX.txt                  : call order → which prompt (1=context, 2=testcase, 3+=solver+retry)

Faithful-to-run.py settings: KB injection OFF and max_retries=1 (== run.py's single retry),
so this is the byte-faithful baseline (NOT the --inject / --max-retries 4 variant).
"""
import os, json
import dotenv; dotenv.load_dotenv(override=True)

# Quiet the benign MIDAS DLL-fallback flood so the console stays readable.
import logging
logging.basicConfig(level=logging.WARNING)
for _n in ("pipeline.function_caller", "LiteLLM", "litellm", "httpx", "dspy"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

os.environ["MBD_INJECT_KB_GUIDELINES"] = "0"   # match run.py (no KB)
os.environ["MBD_MAX_RETRIES"] = "1"            # match run.py (1 retry)

import config  # noqa: F401  pin repo config before mbd_data swaps it
import dspy
from config import reconfigure_lm
from utils.mbd_data import load_mbd_dataset, all_fc_entries, stratified_split_mbd_dataset
from extract_mbd_full import MBDFullPipelineModule, prepare_full_examples
from signatures.raw_completion_adapter import RawCompletionAdapter

import argparse
_ap = argparse.ArgumentParser(description="Apples-to-apples wrapper-side capture for ONE val weakspot.")
_ap.add_argument("--fc", default="GearLvrPosn_Vd", help="FC name (must be in the val split)")
_ap.add_argument("--ws", type=int, default=0, help="weakspot id")
_args = _ap.parse_args()
FC, WS = _args.fc, _args.ws                     # the SAME weakspot run.py processes
REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO, "apples_compare", f"extract_{FC.replace('.', '_')}_ws{WS}")
os.makedirs(OUT, exist_ok=True)

model = os.getenv("MODEL_NAME", "azure/gpt-4.1-mini")
reconfigure_lm(model, cache=False)             # cache OFF -> a real, fresh LLM call each time
dspy.configure(adapter=RawCompletionAdapter())  # run.py-faithful plain-text transport

ds = prepare_full_examples(load_mbd_dataset(fc_entries=all_fc_entries()))
train, val, test = stratified_split_mbd_dataset(ds, train_ratio=0.6, val_ratio=0.4, test_ratio=0.0)
ex = next(e for e in val if e.fc_name == FC and int(e.weakspot_id) == WS)
inp = ex.inputs()
input_dict = {k: inp[k] for k in inp.keys()}

# 1) Save the exact pipeline INPUTS (variables + equations) that feed forward().
with open(os.path.join(OUT, "00_pipeline_inputs.json"), "w", encoding="utf-8") as f:
    json.dump(input_dict, f, indent=2, ensure_ascii=False, default=str)

# 2) Run the pipeline once, capturing every LLM call via the LM history.
gen = MBDFullPipelineModule(use_cot=True)
gen.inject_kb_guidelines = False
gen.max_retries = 1
lm = dspy.settings.lm
start = len(lm.history)
pred = gen(**input_dict)
calls = lm.history[start:]


def _render_messages(msgs):
    parts = []
    for m in msgs:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):  # structured content blocks
            content = "\n".join(
                (b.get("text") if isinstance(b, dict) and "text" in b else json.dumps(b, default=str))
                if isinstance(b, dict) else str(b) for b in content
            )
        parts.append(f"================ role: {role} ================\n{content}")
    return "\n\n".join(parts)


def _render_outputs(h):
    outs = h.get("outputs", h.get("response"))
    if isinstance(outs, list):
        rendered = []
        for o in outs:
            if isinstance(o, dict):
                rendered.append(o.get("text") or json.dumps(o, default=str))
            else:
                rendered.append(str(o))
        return "\n----\n".join(rendered)
    return str(outs)


index = []
for i, h in enumerate(calls, 1):
    msgs = h.get("messages") or []
    prompt_txt = _render_messages(msgs) if msgs else str(h.get("prompt", ""))
    resp_txt = _render_outputs(h)
    with open(os.path.join(OUT, f"llm_call_{i:02d}_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt_txt)
    with open(os.path.join(OUT, f"llm_call_{i:02d}_response.txt"), "w", encoding="utf-8") as f:
        f.write(resp_txt)
    with open(os.path.join(OUT, f"llm_call_{i:02d}_raw.json"), "w", encoding="utf-8") as f:
        json.dump(h, f, indent=2, ensure_ascii=False, default=str)
    index.append(f"llm_call_{i:02d}: prompt={len(prompt_txt)} chars, response={len(resp_txt)} chars")

# 3) Final outputs (== run.py's final_json + the intermediate fields).
with open(os.path.join(OUT, "final_test_cases.json"), "w", encoding="utf-8") as f:
    f.write(getattr(pred, "test_cases", "") or "")
with open(os.path.join(OUT, "final_prediction_all_fields.txt"), "w", encoding="utf-8") as f:
    for k in ("context_analysis", "test_cases_pre_postprocess", "test_cases", "m7_score", "m7_details"):
        f.write(f"===== {k} =====\n{getattr(pred, k, None)}\n\n")

with open(os.path.join(OUT, "INDEX.txt"), "w", encoding="utf-8") as f:
    f.write(f"weakspot : {FC} ws={WS}\nmodel    : {model}\ninject_kb_guidelines: False (== run.py)\n"
            f"max_retries: 1 (== run.py)\nLLM calls captured: {len(calls)}\n\n"
            "Expected call order:\n  call 1  = PROMPT 1 (context_extractor)\n"
            "  call 2  = PROMPT 2 (testcase_generator)\n"
            "  call 3+ = PROMPT 4 (equation_solver_dll) per equation, each optionally + 1 retry\n\n"
            + "\n".join(index) + "\n")

print(f"\nDONE. Captured {len(calls)} LLM call(s) -> {OUT}")
for line in index:
    print("  " + line)
