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
MODEL = "gpt-5.2"

#############################################################################
# DCA (Document Change Analysis) Metrics
#############################################################################

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
    expected_changes = str(getattr(example, "expected_changes", "")).strip().lower()
    predicted_changes = str(getattr(prediction, "changes", "")).strip().lower()


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

def judge_summary_metric(human, model):

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
  → correct_matches: 1 if they match else 0, missed_changes: 0 if they match else 1, hallucinated_changes: 0 if they match else 4.

**BREAKDOWN mode:** Use this only when the human summary describes actual meaningful differences
  (i.e. both images are informational and different).
  → Extract distinct changes from each summary, match them, and score 1–5 based on coverage and accuracy.
  → Score guide: 5 = all key changes captured with no hallucinations, 3 = roughly half captured or some hallucinations, 1 = no meaningful overlap.

STEP 2 — Return JSON:

{{
  "evaluation_mode": "direct" or "breakdown",
  "human_changes": [],
  "model_changes": [],
  "correct_matches": 0,
  "missed_changes": 0,
  "hallucinated_changes": 0,
  "reasoning": "If there are missed_changes or hallucinated_changes, mention briefly the difference between human and model summary, and your reasoning for deciding what are the correct matches, missed changes and hallucinated changes. Else mention human and model summaries are in agreement."
}}

"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
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

    fbeta = (5 * precision * recall / (2*precision + 3*recall)
          if precision is not None and recall is not None and (precision + recall) > 0
          else (0 if precision is not None and recall is not None else None))

    return fbeta, reasoning


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

        total_score = field_weights['discard_image'] * classification_score + field_weights['changes'] * change_score

        if classification_score == 1.0:
            feedback = (
                f"Feedback for Discard Image Classification: CORRECT: Predicted discard_image='{predicted_discard_image}' matches ground truth (expected='{expected_discard_image}'). Feedback for Changes:"+reasoning
            )

        else:
            if expected_bool and not predicted_bool:
                feedback = (
                    f"Feedback for Discard Image Classification: FALSE NEGATIVE (pair={pair_name}): Predicted discard_image='false' "
                    f"but the images should be discarded."
                    f"Feedback for Changes: {reasoning}"
                )
            else:
                feedback = (
                    f"Feedback for Discard Image Classification: FALSE POSITIVE (pair={pair_name}): Predicted discard_image='true' "
                    f"but meaningful changes exist between the images. "
                    f"Feedback for Changes: {reasoning}"
                )

        feedback += f"\nTotal score: {total_score:.3f}"

    if pred_name is None:
        return total_score

    return dspy.Prediction(score=total_score, feedback=feedback)


