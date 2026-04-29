import dspy
import hashlib
import json
import os
import threading

from config import _build_ca_bundle, _get_azure_client, DCA_JUDGE_MODEL_NAME

_ca = _build_ca_bundle()
os.environ.setdefault("REQUESTS_CA_BUNDLE", _ca)
os.environ.setdefault("CURL_CA_BUNDLE", _ca)
os.environ.setdefault("SSL_CERT_FILE", _ca)

# Azure client is lazily initialised on the first judge call so that
# importing this module does not spin up a connection.
# Thread-safe via double-checked locking — multiple eval threads may
# call _get_client() concurrently when max_parallel > 1.
_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = _get_azure_client()
    return _client


MODEL = DCA_JUDGE_MODEL_NAME

# ---------------------------------------------------------------------------
# Judge cache -- persistent file-based cache for judge_summary_metric results.
# Keyed on SHA-256(human + "\0" + model).
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
# Field weights for the combined DCA metric
# ---------------------------------------------------------------------------
FIELD_WEIGHT_CLS    = 0.6   # discard_image classification weight
FIELD_WEIGHT_CHANGE = 0.4   # change summary quality weight

# ---------------------------------------------------------------------------
# Rule-based fast-path for DIRECT-mode special cases.
#
# These canonical phrases can be matched without an LLM call.
# Any human summary that does NOT match falls through to the LLM judge.
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
        "label": "Images are identical - no changes",
        "detect": lambda h: "identical" in h.replace(" ", "") and "no change" in h.replace(" ", "") and len(h) < 50,
        "match":  lambda h, m: "identical" in m.replace(" ", "") and "no change" in m.replace(" ", "") and len(m) < 50,
    },
]


def _try_rule_based_match(human: str, model: str):
    """Try to score the pair without an LLM call for known special-case summaries.

    Returns (fbeta, reasoning, matched) where matched=True means a rule fired.
    fbeta is 1.0 (agreement) or 0.0 (disagreement).
    If no rule applies, returns (None, "", False) -- caller should use the LLM.
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
# DCA Metrics
#############################################################################


def compute_classification_score(expected: bool, predicted: bool) -> float:
    """Return 1.0 if predicted matches expected discard_image, else 0.0."""
    return 1.0 if predicted == expected else 0.0


def compute_total_score(cls_score: float, change_score: float) -> float:
    """Weighted combination: FIELD_WEIGHT_CLS * cls + FIELD_WEIGHT_CHANGE * change."""
    return FIELD_WEIGHT_CLS * cls_score + FIELD_WEIGHT_CHANGE * change_score


def dca_comparison_metric(example, prediction, trace=None):
    """Weighted DCA metric: 0.6 * classification + 0.4 * change summary quality.
    Compares predicted discard_image against human ground truth.
    Returns 1.0 for correct classification, 0.0 for incorrect.

    Compares the change field against the ground truth change description.
    Returns F1Beta score between 0 to 1, based on correctness, missed changes, and hallucinated changes.

    Returns float in [0, 1].
    """
    expected_discard = str(getattr(example, "expected_discard_image", "")).strip().lower()
    predicted_discard = str(getattr(prediction, "discard_image", "")).strip().lower()
    expected_changes = str(getattr(example, "expected_changes", "")).strip()
    predicted_changes = str(getattr(prediction, "changes", "")).strip()

    true_values = {"true", "yes", "1"}
    expected_bool = expected_discard in true_values
    predicted_bool = predicted_discard in true_values

    cls_score = compute_classification_score(expected_bool, predicted_bool)
    change_score = judge_summary_metric(expected_changes, predicted_changes)

    if change_score is None:
        change_score = 0.0

    return compute_total_score(cls_score, change_score)


def judge_summary_metric(human, model):
    """Score model change summary against human ground truth.

    Returns a weighted F-beta score in [0, 1] (precision 60%, recall 40%).
    Uses rule-based fast-path for special cases, persistent cache, and
    falls back to LLM judge for breakdown-mode comparisons.
    """
    # Fast-path: rule-based for DIRECT-mode special cases
    fbeta, reasoning, matched = _try_rule_based_match(human, model)
    if matched:
        return fbeta

    # Cache lookup
    cache_key = _judge_cache_key(human, model)
    if cache_key in _judge_cache:
        return _judge_cache[cache_key]

    # LLM judge for BREAKDOWN-mode (meaningful differences)
    prompt = f"""
You are a judge evaluating model-generated summaries of changes between two document images.

The human (ground truth) summary was produced using these rules:
- Only one image present AND it is informational -> "NoMatchFound"
- Only one image present AND it is non-informational -> "Only one image and it is non informational"
- Both images present AND both non-informational -> "Both images are non informational and contain only logo/watermark etc. no meaningful content"
- Both images present AND both informational AND fully identical or just trivial differences (minor such as layout shift, spelling errors, etc.) -> "Images are identical - no changes"
- Both images present AND one is non-informational -> describe the informational image and note that the other is non-informational
- Image is crossed out -> "Image has been crossed out"
- Image is tiny/corrupted -> "Could not infer"
- Both images present AND both informational AND meaningfully different -> describe the meaningful visual/content differences (ignoring trivial differences like formatting)

---

Human summary:
{human}

Model summary:
{model}

---

STEP 1 -- Determine evaluation mode:

**BREAKDOWN mode:** Use this ONLY when BOTH the human summary AND the model summary describe actual meaningful differences between two images in detail (i.e. both are informational and substantively different).
  -> Extract distinct changes from each summary, compare them, and calculate counts based on overlap.
  -> Example: human describes 3 changes, model correctly captures 2, misses 1, adds 1 hallucinated -> correct_matches=2, missed_changes=1, hallucinated_changes=1.

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
  -> human_changes: [one-phrase label for the human conclusion]
  -> model_changes: [one-phrase label for the model conclusion]
  -> They MATCH only if both convey the same conclusion (e.g. both say "no meaningful changes" / "identical both", or both say "only one image present and it is non informational", etc.).
  -> They DO NOT MATCH if one says "meaningful differences exist" (breakdown) while the other says "identical" or vice versa.
  -> correct_matches: 1 if they match else 0
  -> missed_changes:  0 if they match else 1
  -> hallucinated_changes: 0 if they match else 1

  Example (mixed case): human summary describes several meaningful differences (breakdown), model says "Images are identical - no changes".
  -> DIRECT mode, because model output is a special-category conclusion.
  -> human_changes: ["meaningful differences exist"], model_changes: ["images are identical - no changes"]
  -> They do NOT match -> correct_matches=0, missed_changes=1, hallucinated_changes=1

STEP 2 -- Return JSON:

{{
  "evaluation_mode": "direct" or "breakdown",
  "human_changes": [],
  "model_changes": [],
  "correct_matches": 0,
  "missed_changes": 0,
  "hallucinated_changes": 0
}}

"""
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

    # Degenerate cases: F-beta denominator is 0
    if model_total == 0 or human_total == 0:
        if model_total == 0 and human_total == 0:
            print(f"    [WARN judge] all-zero counts -- treating as 0.0", flush=True)
        result = 0.0
        _judge_cache[cache_key] = result
        _save_judge_cache()
        return result

    precision = correct / model_total
    recall = correct / human_total

    # Weighted F-score: precision 60%, recall 40%
    # F_beta with beta^2 = 2/3  ->  5*P*R / (2*P + 3*R)
    fbeta = (5 * precision * recall / (2 * precision + 3 * recall)
             if (2 * precision + 3 * recall) > 0
             else 0.0)

    _judge_cache[cache_key] = fbeta
    _save_judge_cache()
    return fbeta


def _change_score_label(score: float) -> str:
    """Map a change_score (weighted F-beta, 0-1) to a quality label.

    Thresholds:
      Exact match  >= 0.95
      Good         >= 0.70
      Partial      >= 0.35
      No match      < 0.35
    """
    if score >= 0.95:
        return "Exact match"
    elif score >= 0.70:
        return "Good - needs improvement"
    elif score >= 0.35:
        return "Partial - needs major improvement"
    else:
        return "No match - Unacceptable quality"


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


# ── Hybrid metric (score + prompt-token efficiency penalty) ─────────────────

def _prompt_tokens_from_lm_history(trace) -> int:
    """Read prompt (input) token count for the most recent LM call(s).

    Uses ``dspy.settings.lm.history`` — the live API-reported token counts
    from the LM response JSON.  This is accurate for multimodal prompts
    (base64 images included) because the LM itself counts the tokens.

    ``n_calls`` is estimated as max(1, len(trace)) so multi-module programs
    (e.g. chain-of-thought + classifier) sum all their input tokens.

    Returns 0 when trace is None (pure evaluation mode, no penalty applied)
    or when lm.history is unavailable.
    """
    if not trace:
        return 0  # pure eval mode — no penalty
    try:
        import dspy as _dspy
        lm = _dspy.settings.lm
        if not hasattr(lm, "history") or not lm.history:
            return 0
        n_calls = max(1, len(trace))
        recent = lm.history[-n_calls:]
        total_prompt = 0
        for entry in recent:
            usage = entry.get("usage") or {}
            if hasattr(usage, "get"):
                pt = (usage.get("prompt_tokens")
                      or usage.get("input_tokens")
                      or 0)
            else:
                pt = getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0
            total_prompt += int(pt or 0)
        return total_prompt
    except Exception:
        return 0


def make_hybrid_metric(
    base_metric,
    base_metric_with_feedback,
    score_weight: float = 0.5,
    length_weight: float = 0.5,
    max_prompt_tokens: int = 4000,
):
    """Wrap base metrics with a prompt-token efficiency penalty.

    hybrid_score = score_weight * dca_score
                 + length_weight * max(0, 1 - prompt_tokens / max_prompt_tokens)

    Prompt token count is read from ``dspy.settings.lm.history`` after each
    forward pass — these are the real API-reported input tokens, so multimodal
    prompts (images, long contexts) are counted correctly.

    During plain evaluation (trace=None) the wrapper returns the raw DCA score
    unchanged so leaderboard final evals are always on the true metric.

    Parameters
    ----------
    base_metric : callable
        The plain per-example metric, e.g. dca_comparison_metric.
    base_metric_with_feedback : callable
        The feedback-aware metric, e.g. dca_comparison_metric_with_feedback.
    score_weight : float
        Weight applied to the DCA accuracy score (default 0.5).
    length_weight : float
        Weight applied to the token-efficiency term (default 0.5).
    max_prompt_tokens : int
        Input token count at which length_score reaches 0 (default 4000).

    Returns
    -------
    (hybrid_metric, hybrid_metric_with_feedback) — two callables with the
    same signatures as the base functions.
    """

    def hybrid_metric(example, prediction, trace=None):
        raw_score = base_metric(example, prediction, trace=trace)
        tokens = _prompt_tokens_from_lm_history(trace)
        if tokens > 0:
            length_score = max(0.0, 1.0 - tokens / max_prompt_tokens)
            return score_weight * raw_score + length_weight * length_score
        # trace=None → pure eval mode; return unblended score for true leaderboard evals
        return raw_score

    def hybrid_metric_with_feedback(
        example, prediction, trace=None, pred_name=None, pred_trace=None
    ):
        result = base_metric_with_feedback(
            example, prediction,
            trace=trace, pred_name=pred_name, pred_trace=pred_trace,
        )
        tokens = _prompt_tokens_from_lm_history(trace)
        if tokens == 0:
            return result  # no trace → return as-is

        length_score = max(0.0, 1.0 - tokens / max_prompt_tokens)

        if pred_name is None:
            # scalar return path
            return score_weight * result + length_weight * length_score

        # dspy.Prediction return path
        adjusted = score_weight * result.score + length_weight * length_score
        return dspy.Prediction(score=adjusted, feedback=result.feedback)

    return hybrid_metric, hybrid_metric_with_feedback
