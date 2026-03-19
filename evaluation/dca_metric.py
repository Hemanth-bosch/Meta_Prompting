import dspy


def dca_comparison_metric(example, prediction, trace=None):
    """Binary classification metric for DCA image comparison.

    Compares predicted discard_image against human ground truth.
    Returns 1.0 for correct classification, 0.0 for incorrect.

    Args:
        example: dspy.Example with expected_discard_image ("true"/"false")
        prediction: dspy.Prediction with discard_image
        trace: Optional trace (unused, required by DSPy interface)

    Returns:
        float: 1.0 if correct, 0.0 if incorrect
    """
    expected = str(getattr(example, "expected_discard_image", "")).strip().lower()
    predicted = str(getattr(prediction, "discard_image", "")).strip().lower()

    # Normalize various boolean representations
    true_values = {"true", "yes", "1"}
    false_values = {"false", "no", "0"}

    expected_bool = expected in true_values
    predicted_bool = predicted in true_values

    # If predicted value is not clearly true or false, score 0
    if predicted not in true_values and predicted not in false_values:
        return 0.0

    return 1.0 if predicted_bool == expected_bool else 0.0


def dca_comparison_metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """DCA comparison metric with textual feedback for GEPA/MIPROv2 optimization.

    Returns dspy.Prediction with score and feedback text when pred_name is set.
    """
    _ = trace
    _ = pred_trace

    expected = str(getattr(example, "expected_discard_image", "")).strip().lower()
    predicted = str(getattr(prediction, "discard_image", "")).strip().lower()
    reasoning = getattr(prediction, "reasoning", "No reasoning provided")
    changes = getattr(prediction, "changes", "")

    true_values = {"true", "yes", "1"}
    false_values = {"false", "no", "0"}

    expected_bool = expected in true_values
    predicted_bool = predicted in true_values
    predicted_valid = predicted in true_values or predicted in false_values

    if not predicted_valid:
        score = 0.0
        feedback = (
            f"INVALID OUTPUT: Predicted '{predicted}' which is not a valid boolean. "
            f"The output for discard_image must be exactly 'true' or 'false'. "
            f"Reasoning was: '{reasoning[:200]}'"
        )
    elif predicted_bool == expected_bool:
        score = 1.0
        feedback = (
            f"CORRECT: Predicted discard_image='{predicted}' matches "
            f"ground truth (expected='{expected}')."
        )
    else:
        score = 0.0
        pair_name = getattr(example, "pair_name", "unknown")
        expected_changes = getattr(example, "expected_changes", "")

        if expected_bool and not predicted_bool:
            # Should discard but model said keep
            feedback = (
                f"FALSE NEGATIVE (pair={pair_name}): Predicted discard_image='false' "
                f"but the images should be discarded (identical/trivial). "
                f"Human note: '{expected_changes[:200]}'. "
                f"Model reasoning: '{reasoning[:200]}'. "
                f"The prompt should emphasize that when images are visually identical "
                f"or differences are only trivial (resolution, watermarks, minor border "
                f"changes), discard_image should be true."
            )
        else:
            # Should keep but model said discard
            feedback = (
                f"FALSE POSITIVE (pair={pair_name}): Predicted discard_image='true' "
                f"but meaningful changes exist between the images. "
                f"Human note: '{expected_changes[:200]}'. "
                f"Model reasoning: '{reasoning[:200]}'. "
                f"The prompt should emphasize that text changes, label changes, "
                f"diagram modifications, cross-out marks, or structural differences "
                f"should result in discard_image=false."
            )

    if pred_name is None:
        return score

    return dspy.Prediction(score=score, feedback=feedback)
