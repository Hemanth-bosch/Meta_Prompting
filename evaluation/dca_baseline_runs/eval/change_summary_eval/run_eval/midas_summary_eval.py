import json
import os
from pathlib import Path

# ---------- PATHS ----------
# File lives at: evaluation/dca_baseline_runs/eval/change_summary_eval/run_eval/
SCRIPT_DIR      = Path(__file__).resolve().parent
REPO_ROOT       = SCRIPT_DIR.parents[4]            # 5 levels up to repo root
EVAL_RESULTS_DIR = SCRIPT_DIR.parent / "eval_results"  # change_summary_eval/eval_results/

# Set CA bundle before importing bert_score / huggingface_hub so their httpx
# clients pick up the corporate root CA and don't hit proxy SSL errors.
from config import _build_ca_bundle, _get_azure_client
_ca = _build_ca_bundle()
os.environ.setdefault("REQUESTS_CA_BUNDLE", _ca)
os.environ.setdefault("CURL_CA_BUNDLE", _ca)
os.environ.setdefault("SSL_CERT_FILE", _ca)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- CONFIG ----------
DATA_DIR  = REPO_ROOT / "data" / "dca_data" / "image_pairs_llm_output_exp1"
MAX_PAIRS = None
client    = _get_azure_client()

# ---------- MODELS TO EVALUATE ----------
MODELS = {
    "v1/gpt41": "v1_prompt_midas_gpt41_changes",
    "v1/gpt41m": "v1_prompt_midas_gpt41_mini_changes",
    "v1/gpt52": "v1_prompt_midas_gpt52_changes",
    "v1/gpt52rl": "v1_prompt_midas_gpt52_reason_low_changes",
    "v1/gpt5m": "v1_prompt_midas_gpt_5_mini_reason_low_changes",
    "v2/gpt41": "v2_prompt_cr_gpt41_changes",
    "v2/gpt41m": "v2_prompt_cr_gpt41_mini_changes",
    "v2/gpt52": "v2_prompt_cr_gpt52_changes",
    "v2/gpt52rl": "v2_prompt_cr_gpt52_reason_low_changes",
    "v2/gpt5m": "v2_prompt_cr_gpt_5_mini_reason_low_changes",
    "v3/gpt41": "v3_prompt_cr_gpt41_changes",
    "v3/gpt41m": "v3_prompt_cr_gpt41_mini_changes",
    "v3/gpt52": "v3_prompt_cr_gpt52_changes",
    "v3/gpt52rl": "v3_prompt_cr_gpt52_reason_low_changes",
    "v3/gpt5m": "v3_prompt_cr_gpt_5_mini_reason_low_changes",
    "v4/gpt41": "v4_prompt_cr_gpt41_changes",
    "v4/gpt41m": "v4_prompt_cr_gpt41_mini_changes",
    "v4/gpt52": "v4_prompt_cr_gpt52_changes",
    "v4/gpt52rl": "v4_prompt_cr_gpt52_reason_low_changes",
    "v4/gpt5m": "v4_prompt_cr_gpt_5_mini_reason_low_changes"
}

# ---------- LLM JUDGE ----------
def judge_summary(human, model):

    prompt = f"""
You are a judge evaluating model-generated summaries of changes between two document images.

The human (ground truth) summary was produced using these rules:
- Only one image present AND it is informational → "NoMatchFound"
- Only one image present AND it is non-informational → "Only one image and it is non informational"
- Both images present AND both non-informational → "Both images are non informational and contain only logo/watermark etc. no meaningful content"
- Both images present AND both informational AND fully identical → "Images are identical - no changes"
- Both images present AND one is non-informational → describe the informational image and note that the other is non-informational
- Image is crossed out → "Image has been crossed out"
- Image is tiny/corrupted → "Could not infer"
- Both images present AND both informational AND meaningfully different → describe the meaningful visual/content differences (ignoring trivial differences like formatting)

---

Human summary:
{human}

Model summary:
{model}

---

STEP 1 — Determine evaluation mode:

**DIRECT mode:** Use this when the human summary falls into any special case:
  "NoMatchFound", "Images are identical - no changes", one/both images non-informational,
  image crossed out, could not infer, or only one image present.
  → The model is expected to produce a summary that conveys the same category/conclusion as the human.
  → Treat the human's conclusion as a single "change" item and the model's conclusion as a single "change" item.
  → Score 5 if they match, score 1 if they do not.
  → human_changes: [human conclusion], model_changes: [model conclusion]
  → correct_matches: 1 if they match else 0, missed_changes: 0 if they match else 1, hallucinated_changes: 0 if they match else 1.

**BREAKDOWN mode:** Use this only when the human summary describes actual meaningful differences
  (i.e. both images are informational and different).
  → Extract distinct changes from each summary, match them, and score 1–5 based on coverage and accuracy.
  → Score guide: 5 = all key changes captured with no hallucinations, 3 = roughly half captured or some hallucinations, 1 = no meaningful overlap.

STEP 2 — Return JSON:

{{
  "evaluation_mode": "direct" or "breakdown",
  "score": 1-5,
  "human_changes": [],
  "model_changes": [],
  "correct_matches": 0,
  "missed_changes": 0,
  "hallucinated_changes": 0
}}
"""

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": prompt}]
    )

    usage = response.usage
    text = response.choices[0].message.content
    token_usage = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
    return text, token_usage


# ---------- LOAD ----------
files = sorted(DATA_DIR.glob("pair_*.json"))

data = []
for f in files:
    with open(f, "r", encoding="utf-8") as file:
        data.append(json.load(file))

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- FILTER VALID PAIRS ----------
def is_false(val):
    return val is False or str(val).lower() == "false"

valid_pairs = [
    d for d in data
    if is_false(d.get("human_label_discard_image"))
]

pairs = valid_pairs[:MAX_PAIRS]

print(f"Evaluating {len(pairs)} pairs\n")

# ---------- SENTENCE TRANSFORMER SIMILARITY SCORE ----------
def semantic_similarity(a, b):
    emb = model.encode([a, b])
    return cosine_similarity([emb[0]], [emb[1]])[0][0]

# ---------- RUN ----------
results = []

for i, pair in enumerate(pairs):

    print("="*80)
    print(f"PAIR {i+1}")

    human = pair["human_label_changes"]

    print("\nHUMAN:", human)

    pair_result = {"pair_id": i+1, "models": {}}

    for model_name, key in MODELS.items():

        if key not in pair:
            continue

        model_summary = pair[key]

        print("\n-----------------------------")
        print(f"{model_name.upper()}:", model_summary)
        
        sim = float(semantic_similarity(model_summary, human))
        print(f"Similarity: {sim:.3f}")

        # ---------- LLM JUDGE ----------
        judge_output, token_usage = judge_summary(human, model_summary)

        print("\nLLM JUDGE:")
        print(judge_output)

        pair_result["models"][model_name] = {
            "summary": model_summary,
            "sim_score": sim,
            "llm_judge_raw": judge_output,
            "token_usage": token_usage,
        }

    results.append(pair_result)

# ---------- COMPUTE PER-PAIR METRICS FROM LLM JUDGE ----------
def compute_metrics(judge_raw: str) -> dict:
    """Parse LLM judge JSON and compute precision, recall, hallucination_rate."""
    try:
        # Strip markdown code fences if present
        text = judge_raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text.strip())
    except Exception:
        return {}

    correct = parsed.get("correct_matches", 0)
    missed = parsed.get("missed_changes", 0)
    hallucinated = parsed.get("hallucinated_changes", 0)
    human_total = correct + missed
    model_total = correct + hallucinated

    precision = correct / model_total if model_total > 0 else None
    recall = correct / human_total if human_total > 0 else None
    hallucination_rate = hallucinated / model_total if model_total > 0 else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and (precision + recall) > 0
          else (0 if precision is not None and recall is not None else None))

    return {
        "evaluation_mode": parsed.get("evaluation_mode"),
        "llm_score": parsed.get("score"),
        "correct_matches": correct,
        "missed_changes": missed,
        "hallucinated_changes": hallucinated,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "hallucination_rate": round(hallucination_rate, 4) if hallucination_rate is not None else None,
    }

# Enrich results with computed metrics
for pair_result in results:
    for model_name, model_data in pair_result["models"].items():
        metrics = compute_metrics(model_data.get("llm_judge_raw", ""))
        model_data.update(metrics)

# ---------- AGGREGATE: AVERAGE PER MODEL ACROSS ALL PAIRS ----------
from collections import defaultdict

model_accum = defaultdict(lambda: defaultdict(list))
metric_keys = ["llm_score", "sim_score", "precision", "recall", "f1", "hallucination_rate",
               "prompt_tokens", "completion_tokens", "total_tokens"]

for pair_result in results:
    for model_name, model_data in pair_result["models"].items():
        for k in metric_keys:
            # token fields are nested under token_usage
            if k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                v = (model_data.get("token_usage") or {}).get(k)
            else:
                v = model_data.get(k)
            if v is not None:
                model_accum[model_name][k].append(v)

model_averages = {}
for model_name, metrics in model_accum.items():
    model_averages[model_name] = {
        k: round(sum(vs) / len(vs), 4) for k, vs in metrics.items() if vs
    }

print("\n" + "="*80)
print("AVERAGE METRICS PER MODEL")
print("="*80)
for model_name, avgs in model_averages.items():
    print(f"\n{model_name.upper()}")
    for k, v in avgs.items():
        print(f"  {k:<22}: {v}")

# ---------- SAVE ----------
output = {
    "pairs": results,
    "model_averages": model_averages,
}
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
out_path = EVAL_RESULTS_DIR / "summary_eval_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved results to {out_path}")