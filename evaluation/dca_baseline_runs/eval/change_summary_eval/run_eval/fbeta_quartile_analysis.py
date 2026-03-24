"""Precision-60 / Recall-40 weighted F-beta quartile analysis.

Computes per-(pair, version) F-beta scores for gpt41m and gpt5m and
reports quartile distribution (Q1-Q4), sample counts, and percentages.

Formula:  fbeta = 5*P*R / (2*P + 3*R)   [beta^2 = 2/3, precision-heavy]
  precision = correct / (correct + hallucinated)
  recall    = correct / (correct + missed)
"""

import json
from pathlib import Path
from collections import defaultdict

RESULTS_FILE = Path(__file__).parents[1] / "eval_results" / "summary_eval_results.json"
TARGET_MODELS = ["gpt41m", "gpt5m"]
VERSIONS = ["v2"]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_fbeta(correct: int, missed: int, hallucinated: int) -> float | None:
    """Precision-60 / recall-40 weighted F-score (beta^2 = 2/3).

    Returns None when both totals are zero (no ground-truth + no predictions).
    """
    human_total = correct + missed      # ground-truth change count
    model_total = correct + hallucinated  # predicted change count

    precision = correct / model_total if model_total > 0 else None
    recall    = correct / human_total  if human_total  > 0 else None

    if precision is not None and recall is not None:
        denom = 2 * precision + 3 * recall
        return 5 * precision * recall / denom if denom > 0 else 0.0
    return None


def percentile(sorted_data: list[float], p: float) -> float:
    """Linear-interpolation percentile (same as numpy default)."""
    if len(sorted_data) == 1:
        return sorted_data[0]
    idx  = p / 100.0 * (len(sorted_data) - 1)
    lo   = int(idx)
    hi   = min(lo + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_quartile_table(scores: list[float], model_name: str) -> None:
    """Print quartile breakdown table for a model's score distribution."""
    if not scores:
        print(f"\n  {model_name}: no scores found")
        return

    n = len(scores)
    s = sorted(scores)

    q1 = percentile(s, 25)
    q2 = percentile(s, 50)   # median
    q3 = percentile(s, 75)

    # Closed-right intervals; Q1 bucket is left-closed too
    in_q1 = [v for v in scores if v <= q1]
    in_q2 = [v for v in scores if q1 < v <= q2]
    in_q3 = [v for v in scores if q2 < v <= q3]
    in_q4 = [v for v in scores if v > q3]

    print(f"\n{'='*65}")
    print(f"  Model : {model_name}   |   N = {n}")
    print(f"  Mean  : {sum(scores)/n:.4f}   "
          f"Min : {min(scores):.4f}   Max : {max(scores):.4f}")
    print(f"  Q1 (25th pct) : {q1:.4f}")
    print(f"  Q2 (50th pct) : {q2:.4f}  ← median")
    print(f"  Q3 (75th pct) : {q3:.4f}")
    print()
    print(f"  {'Quartile':<14} {'Range':<26} {'Count':>6}  {'%':>7}")
    print(f"  {'-'*57}")

    rows = [
        ("Q1  (bottom)",  f"[0.0000, {q1:.4f}]",        in_q1),
        ("Q2",            f"({q1:.4f}, {q2:.4f}]",  in_q2),
        ("Q3",            f"({q2:.4f}, {q3:.4f}]",  in_q3),
        ("Q4  (top)",     f"({q3:.4f}, 1.0000]",         in_q4),
    ]
    for label, rng, bucket in rows:
        print(f"  {label:<14} {rng:<26} {len(bucket):>6}  {len(bucket)/n:>7.1%}")


def print_per_pair_table(
    details: dict[str, list[tuple]],
    pair_ids: list[int],
    version: str,
) -> None:
    """Print fbeta per pair_id for each model, scoped to one version."""
    print(f"\n  Per-pair F-beta  (version={version})")
    print()
    header = f"  {'pair_id':<10}" + "".join(f"  {m:>12}" for m in TARGET_MODELS)
    print(header)
    print(f"  {'-'*55}")
    for pid in sorted(pair_ids):
        row = f"  {pid:<10}"
        for m in TARGET_MODELS:
            p_scores = [s for p, ver, s in details[m] if p == pid and ver == version]
            if p_scores:
                row += f"  {p_scores[0]:>12.4f}"  # one entry per (pair, version, model)
            else:
                row += f"  {'n/a':>12}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    with open(RESULTS_FILE, encoding="utf-8") as fh:
        data = json.load(fh)

    # version -> model -> list[float]
    version_model_scores: dict[str, dict[str, list[float]]] = {
        v: defaultdict(list) for v in VERSIONS
    }
    # model -> list[(pair_id, version, score)]  — used by per-pair table
    model_details: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    pair_ids: set[int] = set()

    for pair in data["pairs"]:
        pair_id = pair["pair_id"]
        pair_ids.add(pair_id)

        for key, entry in pair["models"].items():
            version, model_suffix = key.split("/", 1)
            if model_suffix not in TARGET_MODELS:
                continue
            if version not in VERSIONS:
                continue

            correct      = entry.get("correct_matches",      0)
            missed       = entry.get("missed_changes",       0)
            hallucinated = entry.get("hallucinated_changes", 0)

            fbeta = compute_fbeta(correct, missed, hallucinated)
            if fbeta is not None:
                version_model_scores[version][model_suffix].append(fbeta)
                model_details[model_suffix].append((pair_id, version, fbeta))

    # ── Header ──────────────────────────────────────────────────────────────
    print("\nPrecision-60 / Recall-40 weighted F-beta analysis")
    print("Formula : fbeta = 5·P·R / (2·P + 3·R)   [beta² = 2/3]")
    print(f"Versions: {', '.join(VERSIONS)}")
    print(f"Models  : {', '.join(TARGET_MODELS)}")

    # ── Per-version, per-model quartile + per-pair breakdown ─────────────────
    for version in VERSIONS:
        print(f"\n\n{'#'*65}")
        print(f"  VERSION: {version}")
        print(f"{'#'*65}")

        for model in TARGET_MODELS:
            scores = version_model_scores[version][model]
            print_quartile_table(scores, model)

        print_per_pair_table(model_details, pair_ids, version)

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    main()
