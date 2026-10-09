"""DCA evaluation metric \u2014 classification score, judge-based change score, hybrid penalty.

WHAT IS THE DCA METRIC?
-----------------------
DCA (Document Change Analysis) measures how well the model describes
meaningful differences between two document images.

The final score for one example is:

    total_score = FIELD_WEIGHT_CLS \xd7 cls_score + FIELD_WEIGHT_CHANGE \xd7 change_score
                = 0.6 \xd7 cls_score + 0.4 \xd7 change_score

  cls_score    (classification):   1.0 if model correctly flags the change
               category; 0.0 otherwise.  Rule-based, instant, no LLM needed.

  change_score (summary quality):  How well the model\u2019s change description
               matches the human ground truth.  Measured by a judge LLM that
               returns precision/recall of content items, combined via F-beta
               with \u03b2\u00b2 = FIELD_WEIGHT_CHANGE/FIELD_WEIGHT_CLS = 2/3.

JUDGE PIPELINE
--------------
  1. \u201cFast-path\u201d rules in _SPECIAL_CASE_RULES handle NoMatchFound,
     Could not infer, Images are identical, crossed out, etc. \u2014 no LLM call.
  2. Persistent on-disk cache at JUDGE_CACHE_FILE avoids re-judging the
     same (human, model) pair across runs.  Cache key = sha256(human+model).
  3. For genuine change descriptions, an Azure OpenAI judge call is made.
     The judge returns JSON with precision (0\u20131) and recall (0\u20131).
  4. F-beta combines them: score = (1+\u03b2\u00b2)\xd7P\xd7R / (\u03b2\u00b2\xd7P + R)

HYBRID METRIC (optional)
------------------------
make_hybrid_metric() wraps any base metric with a token-efficiency penalty:

  Exponential: score \xd7 exp(-k \xd7 max(0, tokens/T - 1))
               Full score up to T tokens, then decays smoothly.
               Default: T=2500, k=2.0

  Linear:      score \xd7 max(0, 1 - (tokens - T) / T)
               Full score up to T tokens, linear decrease to 0 at 2\xd7T.

  Both modes only apply during optimization (trace is not None).
  During pure evaluation (trace=None), the raw metric score is returned
  unchanged so leaderboard scores stay comparable across candidates.

MODULE EXPORTS
--------------
  dca_comparison_metric(example, prediction, trace) \u2192 float [0,1]
      Primary metric for DSPy optimizers.  Returns total_score.

  dca_comparison_metric_with_feedback(example, prediction, trace)
      \u2192 dspy.Prediction(score, feedback)
      Used by GEPA optimizer which needs textual feedback per example.

  judge_summary_metric(human, model) \u2192 float [0,1]
      Judge the summary quality of one example.  Exported for testing.

  make_hybrid_metric(base_metric, \u2026) \u2192 callable
      Factory returning a new metric with token penalty applied.

  compute_classification_score(example, prediction) \u2192 float
      Classification component only (no LLM).  Used for fast pre-screening.

CONSTANTS
---------
  FIELD_WEIGHT_CLS    = 0.6   (classification weight)
  FIELD_WEIGHT_CHANGE = 0.4   (change summary weight)
  JUDGE_CACHE_FILE    = \".cache/judge_summary_cache.json\"
"""

import dspy
import hashlib
import json
import math
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

_JUDGE_PROMPT_TEMPLATE = """
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
__HUMAN_SUMMARY__

Model summary:
__MODEL_SUMMARY__

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

{
    "evaluation_mode": "direct" or "breakdown",
    "human_changes": [],
    "model_changes": [],
    "correct_matches": 0,
    "missed_changes": 0,
    "hallucinated_changes": 0
}
""".strip()

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
    namespace = {
        "judge_model": MODEL,
        "prompt_template": _JUDGE_PROMPT_TEMPLATE,
        "special_case_rules": [
            {
                "label": rule["label"],
                "detect_code": rule["detect"].__code__.co_code.hex(),
                "detect_consts": repr(rule["detect"].__code__.co_consts),
                "match_code": rule["match"].__code__.co_code.hex(),
                "match_consts": repr(rule["match"].__code__.co_consts),
            }
            for rule in _SPECIAL_CASE_RULES
        ],
    }
    payload = json.dumps(
        {
            "namespace": namespace,
            "human": human,
            "model": model,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_summary_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())

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
        "detect": lambda h: ("nomatchfound" in h or "no match found" in h) and len(h) < 20,
        "match":  lambda h, m: ("nomatchfound" in m or "no match found" in m) and len(m) < 20,
    },
    {
        "label": "Could not infer",
        "detect": lambda h: "could not infer" in h and len(h) < 25,
        "match":  lambda h, m: "could not infer" in m and len(m) < 25,
    },
    {
        "label": "Image has been crossed out",
        "detect": lambda h: ("crossed out" in h or "crossedout" in h) and len(h) < 40,
        "match":  lambda h, m: ("crossed out" in m or "crossedout" in m) and len(m) < 40,
    },
    {
        "label": "Images are identical - no changes",
        "detect": lambda h: "identical" in h and ("no change" in h or "nochanges" in h) and len(h) < 50,
        "match":  lambda h, m: "identical" in m and ("no change" in m or "nochanges" in m) and len(m) < 50,
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

def get_dca_score_breakdown(example, prediction):
    """
    Calculate the individual DCA scores used by adaptive sampling.
    """

    expected_discard = str(
        getattr(example, "expected_discard_image", "")
    ).strip().lower()

    predicted_discard = str(
        getattr(prediction, "discard_image", "")
    ).strip().lower()

    expected_changes = str(
        getattr(example, "expected_changes", "")
    ).strip()

    predicted_changes = str(
        getattr(prediction, "changes", "")
    ).strip()

    true_values = {"true", "yes", "1"}

    expected_bool = expected_discard in true_values
    predicted_bool = predicted_discard in true_values

    cls_score = compute_classification_score(
        expected_bool,
        predicted_bool,
    )

    change_score = judge_summary_metric(
        expected_changes,
        predicted_changes,
    )

    if change_score is None:
        change_score = 0.0

    total_score = compute_total_score(
        cls_score,
        change_score,
    )

    error = 1.0 - total_score

    return {
        "classification_score": cls_score,
        "change_score": change_score,
        "total_score": total_score,
        "error": error,
    }
# HARD EXAMPLE SAMPLING
# ────────────────────────────────────────────────────────────────

def select_hard_examples(
    examples,
    comparator,
    metric,
    sample_size,
):
    """Select the lowest-scoring examples as hard examples."""

    scored_examples = []

    print("\n" + "=" * 80)
    print("HARD EXAMPLE SAMPLING")
    print("=" * 80)

    for idx, example in enumerate(examples):
        try:
            prediction = comparator(
                image_original=example.image_original,
                image_reimage=example.image_reimage,
            )

            score = float(
                metric(example, prediction)
            )

            scored_examples.append(
                (score, idx, example)
            )

        except Exception as exc:
            print(
                f"[WARNING] Could not score example "
                f"{idx}: {exc}"
            )

    if len(scored_examples) < sample_size:
        raise ValueError(
            f"Only {len(scored_examples)} examples were scored, "
            f"but {sample_size} are required."
        )

    # Lower score = harder example
    scored_examples.sort(
        key=lambda x: x[0]
    )

    selected = scored_examples[:sample_size]

    print("\nHard examples selected:")

    for rank, (score, idx, example) in enumerate(
        selected,
        start=1
    ):
        print(
            f"{rank:3d}. "
            f"index={idx:4d} "
            f"score={score:.4f}"
        )

    print("=" * 80)
    print(f"Successfully scored : {len(scored_examples)}")
    print(f"Selected             : {len(selected)}")
    print("=" * 80)

    return [
        example
        for score, idx, example in selected
    ]
    
    
def judge_summary_metric(human, model):
    """Score model change summary against human ground truth.

    Returns a weighted F-beta score in [0, 1] (precision 60%, recall 40%).
    Uses rule-based fast-path for special cases, persistent cache, and
    falls back to LLM judge for breakdown-mode comparisons.
    """
    if _normalize_summary_text(human) == _normalize_summary_text(model):
        return 1.0

    # Fast-path: rule-based for DIRECT-mode special cases
    fbeta, reasoning, matched = _try_rule_based_match(human, model)
    if matched:
        return fbeta

    # Cache lookup
    cache_key = _judge_cache_key(human, model)
    if cache_key in _judge_cache:
        return _judge_cache[cache_key]

    # LLM judge for BREAKDOWN-mode (meaningful differences)
    prompt = (
        _JUDGE_PROMPT_TEMPLATE
        .replace("__HUMAN_SUMMARY__", str(human))
        .replace("__MODEL_SUMMARY__", str(model))
    )
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
    penalty_mode: str = "linear",
    target_tokens: int = 2500,
    penalty_steepness: float = 2.0,
):
    """Wrap base metrics with a prompt-token efficiency penalty.

    Two penalty modes are supported:

    ``'linear'`` (default, backwards-compatible):
        hybrid_score = score_weight * dca_score
                     + length_weight * max(0, 1 - prompt_tokens / max_prompt_tokens)

    ``'exponential'`` (recommended):
        hybrid_score = dca_score * exp(-k * max(0, t/T - 1))

        Zero penalty below *target_tokens* T; exponential cliff above it.
        ``penalty_steepness`` k controls severity (k=1 gentle, k=2 standard,
        k=3 aggressive).

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
        Weight applied to the DCA accuracy score (linear mode, default 0.5).
    length_weight : float
        Weight applied to the token-efficiency term (linear mode, default 0.5).
    max_prompt_tokens : int
        Input token count at which length_score reaches 0 (linear mode, default 4000).
    penalty_mode : str
        ``'linear'`` or ``'exponential'`` (default ``'linear'``).
    target_tokens : int
        Token budget for exponential mode — no penalty below this (default 2500).
    penalty_steepness : float
        Exponential steepness k (default 2.0).

    Returns
    -------
    (hybrid_metric, hybrid_metric_with_feedback) — two callables with the
    same signatures as the base functions.
    """

    def hybrid_metric(example, prediction, trace=None):
        raw_score = base_metric(example, prediction, trace=trace)
        tokens = _prompt_tokens_from_lm_history(trace)
        if tokens > 0:
            if penalty_mode == "exponential":
                factor = math.exp(-penalty_steepness * max(0.0, tokens / target_tokens - 1.0))
                return raw_score * factor
            else:  # "linear"
                length_score = max(0.0, 1.0 - tokens / max_prompt_tokens)
                return score_weight * raw_score + length_weight * length_score
        # trace=None or no LM history → pure eval mode; return unblended score for true leaderboard evals
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

        if penalty_mode == "exponential":
            factor = math.exp(-penalty_steepness * max(0.0, tokens / target_tokens - 1.0))
            if pred_name is None:
                # scalar return path
                return result * factor
            # dspy.Prediction return path
            adjusted = result.score * factor
            return dspy.Prediction(score=adjusted, feedback=result.feedback)
        else:  # "linear"
            length_score = max(0.0, 1.0 - tokens / max_prompt_tokens)
            if pred_name is None:
                # scalar return path
                return score_weight * result + length_weight * length_score
            # dspy.Prediction return path
            adjusted = score_weight * result.score + length_weight * length_score
            return dspy.Prediction(score=adjusted, feedback=result.feedback)

    return hybrid_metric, hybrid_metric_with_feedback

def dca_passed(example, prediction):
    """
    Return True if the DCA prediction is completely correct,
    otherwise return False.
    """

    score = dca_comparison_metric(
        example,
        prediction,
    )

    return score >= 1.0
