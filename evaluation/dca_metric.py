import dspy
import json
import os

from config import _build_ca_bundle, _get_azure_client
_ca = _build_ca_bundle()
os.environ.setdefault("REQUESTS_CA_BUNDLE", _ca)
os.environ.setdefault("CURL_CA_BUNDLE", _ca)
os.environ.setdefault("SSL_CERT_FILE", _ca)

# ---------- CONFIG ----------
client = _get_azure_client()

# TODO add this model into config with a unique name
MODEL = "gpt-5.2"

#############################################################################
# DCA (Document Change Analysis) Metrics
#############################################################################

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
    field_weights = {
        'discard_image': 0.6,  # Primary focus on discard_image classification
        'changes': 0.4          # Secondary focus on change description accuracy
    }

    total_score = 0.0

    #Evaluate discard_image classification
    # Normalize various boolean representations
    true_values = {"true", "yes", "1"}

    expected_bool = expected_discard in true_values
    predicted_bool = predicted_discard in true_values

    if predicted_bool == expected_bool:
        total_score += field_weights['discard_image'] * 1.0  # Correct classification
    else:
        total_score += field_weights['discard_image'] * 0.0  # Incorrect classification

    # Evaluate change description accuracy using F1 score
    change_score, reasoning = judge_summary_metric(expected_changes, predicted_changes)
    total_score += field_weights['changes'] * change_score

    return total_score

# TODO in dca metric file, i think we can change- Both images present AND both informational AND fully identical → "Images are identical - no changes"
# to
# - Both images present AND both informational AND fully identical or just trivial differences (minor such as layout shift, spelling errors, etc.) → "Images are identical - no changes"


# TODO Remove score in prompt, because in output JSON from this prompt - we are not using it
def judge_summary_metric(human, model):

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

**BREAKDOWN mode:** Use this only when the human summary describes actual meaningful differences between two images in detail
  (i.e. both images are informational and different).
  → Extract distinct changes from each summary (human and model), compare them, and calculate correct_matches, missed_changes, and hallucinated_changes based on the overlap of change details.
  → For example, if human describes 3 distinct changes and model describes 2 of them correctly but misses 1 and adds 1 hallucinated change, then correct_matches=2, missed_changes=1, hallucinated_changes=1.

**DIRECT mode:** Use this when the human summary falls into any special case (and does not describe meaningful differences) such as:
  "NoMatchFound", "Images are identical - no changes", one/both images non-informational,
  image fully crossed out, could not infer, or only one image present.
  → The model is expected to produce a summary that conveys the same category/conclusion as the human.
  → Treat the human's conclusion as a single "change" item and the model's conclusion as a single "change" item.
  → human_changes: [human conclusion], model_changes: [model conclusion]
  → correct_matches: 1 if they match else 0, missed_changes: 0 if they match else 1, hallucinated_changes: 0 if they match else 4.


STEP 2 — Return JSON:

{{
  "evaluation_mode": "direct" or "breakdown",
  "human_changes": [],
  "model_changes": [],
  "correct_matches": 0,
  "missed_changes": 0,
  "hallucinated_changes": 0,
  "reasoning": "If there are missed_changes or hallucinated_changes, mention briefly the difference between human/ ground-truth and LLM model summary, and your reasoning for deciding the missed changes and hallucinated changes. Else mention human/ ground-truth and LLM model summaries are fully in agreement."
}}

"""
    # TODO add response format
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format="json"
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
    reasoning = parsed.get("reasoning", "")
    human_total = correct + missed
    model_total = correct + hallucinated

    precision = correct / model_total if model_total > 0 else None
    recall = correct / human_total if human_total > 0 else None

    # Weighted F-score: precision carries 60% weight, recall 40%
    # Equivalent to F_beta with beta^2 = 2/3  →  5*P*R / (2*P + 3*R)
    fbeta = (5 * precision * recall / (2 * precision + 3 * recall)
             if precision is not None and recall is not None and (precision + recall) > 0
             else (0 if precision is not None and recall is not None else None))

    return fbeta, reasoning


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
        return "Good"
    elif score >= 0.35:
        return "Partial"
    else:
        return "No match"

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
    # Use same field weights as dca_comparison_metric
    field_weights = {
        'discard_image': 0.6,
        'changes': 0.4
    }
    pair_name = getattr(example, "pair_name", "unknown")
    
    if not predicted_valid:
        classification_score = 0.0
        invalid_feedback = (
            f"INVALID OUTPUT: Predicted_Discard_Image '{predicted_discard_image}' which is not a valid boolean. "
            f"The output for discard_image must be exactly 'true' or 'false'. "
        )
    
        feedback = invalid_feedback + f"\nClassification score: {classification_score:.3f}"
    else:
        classification_score = 1.0 if predicted_bool == expected_bool else 0.0

        # Compute change description score via judge_summary_metric (robust to None)
        change_score, reasoning = judge_summary_metric(expected_changes, predicted_changes)
        if change_score is None:
            change_score = 0.0

        change_label = _change_score_label(change_score)

        total_score = field_weights['discard_image'] * classification_score + field_weights['changes'] * change_score

        if classification_score == 1.0:
            feedback = (
                f"Feedback for Discard Image Classification: CORRECT: Predicted discard_image='{predicted_discard_image}' "
                f"matches ground truth (expected='{expected_discard_image}'). "
                f"Feedback for Changes [{change_label} | score={change_score:.3f}]: {reasoning}"
            )

        else:
            if expected_bool and not predicted_bool:
                feedback = (
                    f"Feedback for Discard Image Classification: FALSE NEGATIVE (pair={pair_name}): "
                    f"Predicted discard_image='false' but the images should be discarded. "
                    f"Feedback for Changes [{change_label} | score={change_score:.3f}]: {reasoning}"
                )
            else:
                feedback = (
                    f"Feedback for Discard Image Classification: FALSE POSITIVE (pair={pair_name}): "
                    f"Predicted discard_image='true' but meaningful changes exist between the images. "
                    f"Feedback for Changes [{change_label} | score={change_score:.3f}]: {reasoning}"
                )

        feedback += f"\nTotal score: {total_score:.3f}"

    if pred_name is None:
        return total_score

    return dspy.Prediction(score=total_score, feedback=feedback)


