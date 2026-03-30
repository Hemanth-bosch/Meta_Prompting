import dspy
import hashlib
import json
import os

from config import _build_ca_bundle, _get_azure_client, DCA_JUDGE_MODEL_NAME

_ca = _build_ca_bundle()
os.environ.setdefault("REQUESTS_CA_BUNDLE", _ca)
os.environ.setdefault("CURL_CA_BUNDLE", _ca)
os.environ.setdefault("SSL_CERT_FILE", _ca)

# ---------- CONFIG ----------
# TODO Azure client is lazily initialised on the first LLM call so that importing
# this module (e.g. in evaluation scripts) does not spin up the client.
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = _get_azure_client()
    return _client


#  TODO Judge model — defined in config.DCA_JUDGE_MODEL_NAME
MODEL = DCA_JUDGE_MODEL_NAME

# ---------------------------------------------------------------------------
# Judge cache — persistent file-based cache for judge_summary_metric results.
# Keyed on SHA-256(human + "\0" + model) to keep file names short.
# Loaded once on import; flushed to disk after every new LLM call.
# ---------------------------------------------------------------------------
_JUDGE_CACHE_PATH = os.path.join(".cache", "judge_summary_cache.json")
_judge_cache: dict[str, float] = {}

def _load_judge_cache() -> None:
    global _judge_cache
    if os.path.exists(_JUDGE_CACHE_PATH):
        try:
            with open(_JUDGE_CACHE_PATH, "r", encoding="utf-8") as f:
                _judge_cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            _judge_cache = {}

def _save_judge_cache() -> None:
    os.makedirs(os.path.dirname(_JUDGE_CACHE_PATH), exist_ok=True)
    with open(_JUDGE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_judge_cache, f, indent=2)

def _judge_cache_key(human: str, model: str) -> str:
    return hashlib.sha256(f"{human}\0{model}".encode()).hexdigest()

_load_judge_cache()

# ---------------------------------------------------------------------------
# TODO Field weights
# ---------------------------------------------------------------------------
FIELD_WEIGHT_CLS    = 0.6   # discard_image classification weight
FIELD_WEIGHT_CHANGE = 0.4   # change summary quality weight

# ---------------------------------------------------------------------------
# TODO Rule-based fast-path for DIRECT-mode special cases
#
# These canonical phrases are output verbatim (or near-verbatim) and can be
# matched without an LLM call.  Any human summary that does NOT match one of
# these patterns is a BREAKDOWN-mode case and still goes to the LLM.
# ---------------------------------------------------------------------------
_SPECIAL_CASE_RULES = [
    {
        "label": "NoMatchFound",
        "detect": lambda h: ("nomatchfound" in h.replace(" ", "") or "no match found" in h) and len(h) < 20,
        "match":  lambda h, m: ("nomatchfound" in m.replace(" ", "") or "no match found" in m) and len(m) < 20,
    },
    {
        "label": "Could not infer",
        "detect": lambda h: "could not infer" in h and len(h) < 25,
        "match":  lambda h, m: "could not infer" in m and len(m) < 25,
    },
    {
        "label": "Image has been crossed out",
        "detect": lambda h: "crossed out" in h.replace(" ", "") and len(h) < 40,
        "match":  lambda h, m: "crossed out" in m.replace(" ", "") and len(m) < 40,
    },
    {
        "label": "Images are identical - no changes.",
        "detect": lambda h: "identical" in h.replace(" ", "") and "no change" in h.replace(" ", "") and  len(h) < 50,
        "match":  lambda h, m: "identical" in m.replace(" ", "") and "no change" in m.replace(" ", "") and len(m) < 50,
    },
    # "Images are identical - no changes" is intentionally omitted:
    # phrasing is too variable across runs → handled by the LLM judge instead.
]

# TODO Add
def _try_rule_based_match(human: str, model: str):
    """Try to score the pair without an LLM call for known special-case summaries.

    Returns (fbeta, reasoning, matched) where matched=True means a rule fired.
    fbeta is 1.0 (agreement) or 0.0 (disagreement).
    If no rule applies, returns (None, "", False) → caller should use the LLM.
    """
    h = human.lower().strip()
    m = model.lower().strip()

    for rule in _SPECIAL_CASE_RULES:
        if rule["detect"](h):
            agrees = rule["match"](h, m)
            label  = rule["label"]
            if agrees:
                return (
                    1.0,
                    f"Rule-based DIRECT match [{label}]: human and model summaries agree.",
                    True,
                )
            else:
                return (
                    0.0,
                    f"Rule-based DIRECT mismatch [{label}]: human='{human[:80]}' but model='{model[:80]}'.",
                    True,
                )

    return None, "", False

#############################################################################
# DCA (Document Change Analysis) Metrics
#############################################################################


def compute_classification_score(expected: bool, predicted: bool) -> float:
    """Return 1.0 if predicted matches expected discard_image, else 0.0."""
    return 1.0 if predicted == expected else 0.0


def compute_total_score(cls_score: float, change_score: float) -> float:
    """Weighted combination: FIELD_WEIGHT_CLS * cls + FIELD_WEIGHT_CHANGE * change."""
    return FIELD_WEIGHT_CLS * cls_score + FIELD_WEIGHT_CHANGE * change_score


# TODO Check the metric in depth

def dca_comparison_metric(example, prediction, trace=None):
    """ Weighted classification metric for image discard classification and change summary generation for DCA image comparison.

    Compares predicted discard_image against human ground truth.
    Returns 1.0 for correct classification, 0.0 for incorrect.

    Compares the change field against the ground truth change description.
    Returns F1 score between 0 to 1, based on correctness, missed changes, and hallucinated changes.

    Args:
        example: dspy.Example with expected_discard_image ("true"/"false")  and change field
        prediction: dspy.Prediction with discard_image and change field
        trace: Optional trace (unused, required by DSPy interface)

    Returns:
        float: 1.0 if correct, 0.0 if incorrect
    """
    expected_discard = str(getattr(example, "expected_discard_image", "")).strip().lower()
    predicted_discard = str(getattr(prediction, "discard_image", "")).strip().lower()
    expected_changes = str(getattr(example, "expected_changes", "")).strip()
    predicted_changes = str(getattr(prediction, "changes", "")).strip()


    # TODO I think we can change this to 0.5 and 0.5
    true_values = {"true", "yes", "1"}
    expected_bool = expected_discard in true_values
    predicted_bool = predicted_discard in true_values

    cls_score = compute_classification_score(expected_bool, predicted_bool)

    # Evaluate change description accuracy using F1 score
    change_score = judge_summary_metric(expected_changes, predicted_changes)

    if change_score is None:
        change_score = 0.0
    total_score = compute_total_score(cls_score, change_score)

    return total_score

# TODO in dca metric file, i think we can change- Both images present AND both informational AND fully identical → "Images are identical - no changes"
# to
# - Both images present AND both informational AND fully identical or just trivial differences (minor such as layout shift, spelling errors, etc.) → "Images are identical - no changes"

# TODO → correct_matches: 1 if they match else 0, missed_changes: 0 if they match else 1, hallucinated_changes: 0 if they match else 4.
# Why 4?

# TODO We can reduce prompt calls here

# TODO Remove score in prompt, because in output JSON from this prompt - we are not using it


# ── Rule-based fast-path ───────────────────────────────────────────────────


def judge_summary_metric(human, model):

    # ── Fast-path: rule-based for DIRECT-mode special cases ──────────────────
    fbeta, reasoning, matched = _try_rule_based_match(human, model)
    if matched:
        return fbeta

    # ── Cache lookup ──────────────────────────────────────────────────────────
    cache_key = _judge_cache_key(human, model)
    if cache_key in _judge_cache:
        return _judge_cache[cache_key]

    # ── Fallback: LLM judge for BREAKDOWN-mode (meaningful differences) ───────

    prompt = f"""
You are a judge evaluating model-generated summaries of changes between two document images.

The human (ground truth) summary was produced using these rules:
- Only one image present AND it is informational → "NoMatchFound"
- Only one image present AND it is non-informational → "Only one image and it is non informational"
- Both images present AND both non-informational → "Both images are non informational and contain only logo/watermark etc. no meaningful content"
- Both images present AND both informational AND fully identical or just trivial differences (minor such as layout shift, spelling errors, etc.) → "Images are identical - no changes"
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

**BREAKDOWN mode:** Use this ONLY when BOTH the human summary AND the model summary describe actual meaningful differences between two images in detail (i.e. both are informational and substantively different).
  → Extract distinct changes from each summary, compare them, and calculate counts based on overlap.
  → Example: human describes 3 changes, model correctly captures 2, misses 1, adds 1 hallucinated → correct_matches=2, missed_changes=1, hallucinated_changes=1.

**DIRECT mode:** Use this whenever the human summary OR the model summary falls into a special category that does not describe detailed meaningful differences. Special categories include:
  - "NoMatchFound" (only one image and it is informational)
  - "Only one image and it is non informational"
  - "Both images are non informational"
  - "One image is non informational" (one present, one not)
  - "Images are identical - no changes" (trivial or no differences)
  - "Image has been crossed out"
  - "Could not infer"
  - A detailed breakdown description when the other summary is one of the above special categories

  In DIRECT mode, reduce each summary to its single conclusion/category label and compare:
  → human_changes: [one-phrase label for the human conclusion]
  → model_changes: [one-phrase label for the model conclusion]
  → They MATCH only if both convey the same conclusion (e.g. both say "no meaningful changes" / "identical both", or both say "only one image present and it is non informational", etc.).
  → They DO NOT MATCH if one says "meaningful differences exist" (breakdown) while the other says "identical" or vice versa.
  → correct_matches: 1 if they match else 0
  → missed_changes:  0 if they match else 1
  → hallucinated_changes: 0 if they match else 1

  Example (mixed case): human summary describes several meaningful differences (breakdown), model says "Images are identical - no changes".
  → DIRECT mode, because model output is a special-category conclusion.
  → human_changes: ["meaningful differences exist"], model_changes: ["images are identical - no changes"]
  → They do NOT match → correct_matches=0, missed_changes=1, hallucinated_changes=1

STEP 2 — Return JSON:

{{
  "evaluation_mode": "direct" or "breakdown",
  "human_changes": [],
  "model_changes": [],
  "correct_matches": 0,
  "missed_changes": 0,
  "hallucinated_changes": 0
}}

"""
    # TODO add response format and add temperature
    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    text = response.choices[0].message.content
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    parsed = json.loads(text.strip())
    correct = parsed.get("correct_matches", 0)
    missed = parsed.get("missed_changes", 0)
    hallucinated = parsed.get("hallucinated_changes", 0)
    human_total = correct + missed
    model_total = correct + hallucinated

    # TODO print to debug
    # print(f"    [judge] mode={parsed.get('evaluation_mode','?')}  "
    #       f"correct={correct}  missed={missed}  hallucinated={hallucinated}  "
    #       f"human_total={human_total}  model_total={model_total}", flush=True)

    # Degenerate cases where F-beta denominator is 0 — return 0.0 with a warning:
    #   model_total=0, human_total>0  → model described nothing, missed all human changes
    #   model_total>0, human_total=0  → model hallucinated changes when human had none
    #   model_total=0, human_total=0  → judge returned all-zero counts (judge failure)
    if model_total == 0 or human_total == 0:
        if model_total == 0 and human_total == 0:
            print(f"    [WARN judge] all-zero counts — treating as 0.0", flush=True)
        result = 0.0
        _judge_cache[cache_key] = result
        _save_judge_cache()
        return result

    precision = correct / model_total if model_total > 0 else 0.0
    recall = correct / human_total if human_total > 0 else 0.0
    
    # Weighted F-score: precision carries 60% weight, recall 40%
    # Equivalent to F_beta with beta^2 = 2/3  →  5*P*R / (2*P + 3*R)
    fbeta = (5 * precision * recall / (2 * precision + 3 * recall)
             if (2 * precision + 3 * recall) > 0
             else 0.0)

    _judge_cache[cache_key] = fbeta
    _save_judge_cache()
    return fbeta


# TODO Jahnavi to look into this
def _change_score_label(score: float) -> str:
    """Map a change_score (weighted F-beta, 0–1) to a quality label.

    Thresholds derived from empirical score distribution (precision-weighted 60 / recall 40):
      Exact match  >= 0.95  — effectively perfect; aligns with the dominant 1.0 cluster
      Good         >= 0.70  — solid quality, minor misses or hallucinations
      Partial      >= 0.35  — meaningful overlap but notable gaps or hallucinations - change summary needs major improvement
      No match      < 0.35  — aligns with the 0.0 cluster and near-zero scores - unacceptable quality
    """
    if score >= 0.95:
        return "Exact match"
    elif score >= 0.70:
        return "Good - needs improvement"
    elif score >= 0.35:
        return "Partial - needs major improvement"
    else:
        return "No match - Unacceptable quality"

# TODO Jahnavi to look into this
def dca_comparison_metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """DCA comparison metric with textual feedback for GEPA/MIPROv2 optimization.

    Returns dspy.Prediction with score and feedback text when pred_name is set.
    """
    _ = trace
    _ = pred_trace

    expected_discard_image = str(getattr(example, "expected_discard_image", "")).strip().lower()
    predicted_discard_image = str(getattr(prediction, "discard_image", "")).strip().lower()
    expected_changes = getattr(example, "expected_changes", "")
    predicted_changes = getattr(prediction, "changes", "")
    
    true_values = {"true", "yes", "1"}
    false_values = {"false", "no", "0"}

    expected_bool = expected_discard_image in true_values
    predicted_bool = predicted_discard_image in true_values
    predicted_valid = predicted_discard_image in true_values or predicted_discard_image in false_values
    pair_name = getattr(example, "pair_name", "unknown")

    if not predicted_valid:
        classification_score = 0.0
        total_score = 0.0  # no change score available for invalid prediction
        invalid_feedback = (
            f"INVALID OUTPUT: Predicted_Discard_Image '{predicted_discard_image}' which is not a valid boolean. "
            f"The output for discard_image must be exactly 'true' or 'false'. "
        )
        feedback = invalid_feedback + f"\nClassification score: {classification_score:.3f}"
    else:
        classification_score = compute_classification_score(expected_bool, predicted_bool)

        # Compute change description score via judge_summary_metric (robust to None)
        change_score = judge_summary_metric(expected_changes, predicted_changes)
        if change_score is None:
            change_score = 0.0

        change_label = _change_score_label(change_score)

        total_score = compute_total_score(classification_score, change_score)

        if classification_score == 1.0:
            feedback = (
                f"Feedback for Discard Image Classification: CORRECT: Predicted discard_image='{predicted_discard_image}' "
                f"matches ground truth (expected='{expected_discard_image}'). "
                f"Feedback for Changes [{change_label} | score={change_score:.3f}]"
            )

        else:
            if expected_bool and not predicted_bool:
                feedback = (
                    f"Feedback for Discard Image Classification: FALSE NEGATIVE (pair={pair_name}): "
                    f"Predicted discard_image='false' but the images should be discarded. "
                    f"Feedback for Changes [{change_label} | score={change_score:.3f}]"
                )
            else:
                feedback = (
                    f"Feedback for Discard Image Classification: FALSE POSITIVE (pair={pair_name}): "
                    f"Predicted discard_image='true' but meaningful changes exist between the images. "
                    f"Feedback for Changes [{change_label} | score={change_score:.3f}]"
                )

        feedback += f"\nTotal score: {total_score:.3f}"

    if pred_name is None:
        return total_score

    return dspy.Prediction(score=total_score, feedback=feedback)


