from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "dca_data" / "image_pairs_llm_output_exp1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "dca_data" / "human_label_stats"


CLASSIFICATION_KEY = "human_label_discard_image"
CHANGE_SUMMARY_KEY = "human_label_changes"
CATEGORY_KEY = "human_label_category"
VALID_CATEGORY_CODES = ("A1", "A2", "B1", "B2", "B3", "B4", "B5a", "B5b")


@dataclass
class Record:
    file_name: str
    classification_raw: Any
    classification_norm: bool | None
    change_summary_raw: Any
    change_summary_norm: str
    change_summary_category: str
    category_raw: Any
    category_norm: str | None
    category_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze human-labeled DCA JSON files and generate stats + JPEG chart for "
            "human_label_discard_image, human_label_changes, and human_label_category."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing pair_XXX JSON files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save stats outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Top-N entries to include for frequent change summaries (default: 15)",
    )
    return parser.parse_args()


def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
    return None


def normalize_summary(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return " ".join(value.strip().split())


def normalize_category_code(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw:
        return None

    lowered = raw.lower()
    if lowered in {"a1", "a2", "b1", "b2", "b3", "b4", "b5a", "b5b"}:
        if lowered.startswith("b5"):
            return "B5a" if lowered == "b5a" else "B5b"
        return lowered.upper()

    if lowered.startswith("a1"):
        return "A1"
    if lowered.startswith("a2"):
        return "A2"
    if lowered.startswith("b1"):
        return "B1"
    if lowered.startswith("b2"):
        return "B2"
    if lowered.startswith("b3"):
        return "B3"
    if lowered.startswith("b4"):
        return "B4"
    if lowered.startswith("b5a"):
        return "B5a"
    if lowered.startswith("b5b"):
        return "B5b"

    return None


def classify_change_summary(text: str) -> str:
    if not text:
        return "missing"

    lowered = text.lower()

    if "nomatchfound" in lowered:
        return "nomatchfound"
    if "crossed out" in lowered or "cross mark" in lowered or "strikethrough" in lowered:
        return "crossed_out"
    if "identical" in lowered and "no changes" in lowered:
        return "identical_no_changes"

    non_info_tokens = [
        "non informational",
        "non-informational",
        "watermark",
        "logo",
        "blank",
    ]
    has_non_info = any(token in lowered for token in non_info_tokens)

    if ("single image" in lowered or "only one image" in lowered) and has_non_info:
        return "single_non_informational"
    if "both" in lowered and has_non_info:
        return "both_non_informational"
    if has_non_info:
        return "non_informational_other"

    meaningful_tokens = [
        "text changed",
        "text added",
        "text removed",
        "removed",
        "added",
        "different",
        "diagram",
        "table",
        "replaced",
        "changed",
    ]
    if any(token in lowered for token in meaningful_tokens):
        return "meaningful_change_described"

    return "other"


def percent(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return (count * 100.0) / total


def safe_numeric_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
        }

    sorted_vals = sorted(values)

    def q(p: float) -> float:
        if len(sorted_vals) == 1:
            return float(sorted_vals[0])
        idx = p * (len(sorted_vals) - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return float(sorted_vals[lo])
        ratio = idx - lo
        return sorted_vals[lo] * (1 - ratio) + sorted_vals[hi] * ratio

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
        "p25": round(q(0.25), 3),
        "p75": round(q(0.75), 3),
    }


def truncate_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def draw_horizontal_bar_panel(
    draw: ImageDraw.ImageDraw,
    panel: tuple[int, int, int, int],
    title: str,
    labels: list[str],
    values: list[int],
    color: tuple[int, int, int],
    font_title: ImageFont.ImageFont,
    font_body: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = panel
    draw.rectangle(panel, outline=(180, 180, 180), width=2)
    draw.text((x0 + 12, y0 + 10), title, fill=(0, 0, 0), font=font_title)

    if not labels or not values:
        draw.text((x0 + 12, y0 + 44), "No data", fill=(90, 90, 90), font=font_body)
        return

    max_value = max(values) if values else 1
    chart_left = x0 + 210
    chart_right = x1 - 12
    chart_top = y0 + 46
    chart_bottom = y1 - 12
    chart_height = max(1, chart_bottom - chart_top)
    n = len(labels)
    row_h = chart_height / n

    for i, (label, val) in enumerate(zip(labels, values)):
        y_center = chart_top + (i + 0.5) * row_h
        bar_h = max(8, int(row_h * 0.55))
        bar_w = 0 if max_value == 0 else int(((chart_right - chart_left) * val) / max_value)

        draw.text((x0 + 12, int(y_center - 8)), truncate_text(label, 32), fill=(30, 30, 30), font=font_body)
        draw.rectangle(
            (chart_left, int(y_center - bar_h / 2), chart_left + bar_w, int(y_center + bar_h / 2)),
            fill=color,
        )
        draw.text((chart_left + bar_w + 6, int(y_center - 8)), str(val), fill=(30, 30, 30), font=font_body)


def draw_histogram_panel(
    draw: ImageDraw.ImageDraw,
    panel: tuple[int, int, int, int],
    title: str,
    values: list[int],
    bins: int,
    color: tuple[int, int, int],
    font_title: ImageFont.ImageFont,
    font_body: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = panel
    draw.rectangle(panel, outline=(180, 180, 180), width=2)
    draw.text((x0 + 12, y0 + 10), title, fill=(0, 0, 0), font=font_title)

    if not values:
        draw.text((x0 + 12, y0 + 44), "No data", fill=(90, 90, 90), font=font_body)
        return

    vmin, vmax = min(values), max(values)
    if vmin == vmax:
        vmin = 0
    span = max(1, vmax - vmin)

    bin_counts = [0] * bins
    for value in values:
        idx = int(((value - vmin) / span) * bins)
        if idx == bins:
            idx = bins - 1
        bin_counts[idx] += 1

    max_count = max(bin_counts) if bin_counts else 1
    chart_left = x0 + 18
    chart_right = x1 - 18
    chart_top = y0 + 44
    chart_bottom = y1 - 38
    chart_w = max(1, chart_right - chart_left)
    chart_h = max(1, chart_bottom - chart_top)

    for i, count in enumerate(bin_counts):
        bx0 = chart_left + int((i / bins) * chart_w) + 2
        bx1 = chart_left + int(((i + 1) / bins) * chart_w) - 2
        bh = 0 if max_count == 0 else int((count / max_count) * chart_h)
        by0 = chart_bottom - bh
        draw.rectangle((bx0, by0, bx1, chart_bottom), fill=color)

    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=(120, 120, 120), width=1)
    draw.text((chart_left, chart_bottom + 8), f"min={min(values)}", fill=(50, 50, 50), font=font_body)
    draw.text((chart_right - 130, chart_bottom + 8), f"max={max(values)}", fill=(50, 50, 50), font=font_body)


def save_visualization_jpeg(
    output_path: Path,
    total_files: int,
    classification_counter: Counter[str],
    category_counter: Counter[str],
    top_summaries: list[tuple[str, int]],
    summary_char_lengths: list[int],
) -> None:
    width, height = 2200, 1500
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    font_title = ImageFont.load_default()
    font_body = ImageFont.load_default()

    draw.text(
        (24, 20),
        f"Human Label Analysis ({CLASSIFICATION_KEY} + {CATEGORY_KEY})",
        fill=(0, 0, 0),
        font=font_title,
    )
    draw.text(
        (24, 42),
        f"Total JSON files analyzed: {total_files}",
        fill=(40, 40, 40),
        font=font_body,
    )

    margin_x = 24
    margin_top = 74
    gap_x = 20
    gap_y = 20
    panel_w = (width - margin_x * 2 - gap_x) // 2
    panel_h = (height - margin_top - 24 - gap_y) // 2

    p1 = (margin_x, margin_top, margin_x + panel_w, margin_top + panel_h)
    p2 = (margin_x + panel_w + gap_x, margin_top, width - margin_x, margin_top + panel_h)
    p3 = (margin_x, margin_top + panel_h + gap_y, margin_x + panel_w, height - 24)
    p4 = (margin_x + panel_w + gap_x, margin_top + panel_h + gap_y, width - margin_x, height - 24)

    class_labels = ["True", "False", "Unknown"]
    class_values = [classification_counter.get(k, 0) for k in class_labels]
    draw_horizontal_bar_panel(
        draw,
        p1,
        "Classification Count",
        class_labels,
        class_values,
        (66, 133, 244),
        font_title,
        font_body,
    )

    top_categories = category_counter.most_common(10)
    draw_horizontal_bar_panel(
        draw,
        p2,
        "Human Label Category Distribution",
        [k for k, _ in top_categories],
        [v for _, v in top_categories],
        (15, 157, 88),
        font_title,
        font_body,
    )

    draw_histogram_panel(
        draw,
        p3,
        "Change-Summary Length Distribution (characters)",
        summary_char_lengths,
        bins=12,
        color=(255, 140, 0),
        font_title=font_title,
        font_body=font_body,
    )

    top_exact = top_summaries[:10]
    draw_horizontal_bar_panel(
        draw,
        p4,
        "Top Exact Change-Summary Values",
        [k for k, _ in top_exact],
        [v for _, v in top_exact],
        (128, 90, 213),
        font_title,
        font_body,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=95)


def analyze(input_dir: Path, top_n: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Path]:
    json_files = sorted(input_dir.glob("*.json"))

    records: list[Record] = []
    parse_errors: list[dict[str, str]] = []

    for path in json_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            parse_errors.append({"file": path.name, "error": str(exc)})
            continue

        cls_raw = payload.get(CLASSIFICATION_KEY)
        cls_norm = normalize_bool(cls_raw)

        change_raw = payload.get(CHANGE_SUMMARY_KEY)
        change_norm = normalize_summary(change_raw)
        change_category = classify_change_summary(change_norm)

        category_raw = payload.get(CATEGORY_KEY)
        category_norm = normalize_category_code(category_raw)
        if category_raw is None or (isinstance(category_raw, str) and not category_raw.strip()):
            category_status = "missing"
        elif category_norm is None:
            category_status = "invalid"
        else:
            category_status = "valid"

        records.append(
            Record(
                file_name=path.name,
                classification_raw=cls_raw,
                classification_norm=cls_norm,
                change_summary_raw=change_raw,
                change_summary_norm=change_norm,
                change_summary_category=change_category,
                category_raw=category_raw,
                category_norm=category_norm,
                category_status=category_status,
            )
        )

    total_parsed = len(records)
    classification_counter = Counter(
        "True" if r.classification_norm is True else "False" if r.classification_norm is False else "Unknown"
        for r in records
    )

    summaries_non_empty = [r.change_summary_norm for r in records if r.change_summary_norm]
    summary_exact_counter = Counter(summaries_non_empty)
    category_counter = Counter(r.category_norm for r in records if r.category_norm is not None)
    missing_category_count = sum(1 for r in records if r.category_status == "missing")
    invalid_category_count = sum(1 for r in records if r.category_status == "invalid")

    summary_char_lengths = [len(s) for s in summaries_non_empty]
    summary_word_lengths = [len(s.split()) for s in summaries_non_empty]

    crosstab: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        cls_key = "True" if r.classification_norm is True else "False" if r.classification_norm is False else "Unknown"
        category_key = r.category_norm if r.category_norm is not None else ("Missing" if r.category_status == "missing" else "Invalid")
        crosstab[cls_key][category_key] += 1

    top_summary_rows: list[dict[str, Any]] = []
    for idx, (summary, count) in enumerate(summary_exact_counter.most_common(top_n), start=1):
        top_summary_rows.append(
            {
                "rank": idx,
                "summary": summary,
                "count": count,
                "percentage": round(percent(count, total_parsed), 3),
            }
        )

    category_rows: list[dict[str, Any]] = []
    for category in VALID_CATEGORY_CODES:
        count = category_counter.get(category, 0)
        category_rows.append(
            {
                "category": category,
                "count": count,
                "percentage": round(percent(count, total_parsed), 3),
            }
        )

    crosstab_rows: list[dict[str, Any]] = []
    crosstab_columns = list(VALID_CATEGORY_CODES) + ["Missing", "Invalid"]
    for cls_name in ["True", "False", "Unknown"]:
        row = {"classification": cls_name}
        for category in crosstab_columns:
            row[category] = crosstab[cls_name].get(category, 0)
        crosstab_rows.append(row)

    invalid_category_details = [
        {
            "file": r.file_name,
            "raw_value": r.category_raw,
        }
        for r in records
        if r.category_status == "invalid"
    ]

    stats: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(input_dir),
        "file_stats": {
            "json_files_found": len(json_files),
            "json_files_parsed": total_parsed,
            "json_parse_errors": len(parse_errors),
            "parse_error_details": parse_errors,
        },
        CLASSIFICATION_KEY: {
            "counts": {
                "true": classification_counter.get("True", 0),
                "false": classification_counter.get("False", 0),
                "unknown": classification_counter.get("Unknown", 0),
            },
            "percentages": {
                "true": round(percent(classification_counter.get("True", 0), total_parsed), 3),
                "false": round(percent(classification_counter.get("False", 0), total_parsed), 3),
                "unknown": round(percent(classification_counter.get("Unknown", 0), total_parsed), 3),
            },
        },
        CHANGE_SUMMARY_KEY: {
            "missing_count": sum(1 for r in records if not r.change_summary_norm),
            "non_empty_count": len(summaries_non_empty),
            "unique_exact_values": len(summary_exact_counter),
            "character_length_stats": safe_numeric_summary(summary_char_lengths),
            "word_length_stats": safe_numeric_summary(summary_word_lengths),
            "top_exact_values": top_summary_rows,
        },
        CATEGORY_KEY: {
            "valid_codes": list(VALID_CATEGORY_CODES),
            "counts": {
                **{code: category_counter.get(code, 0) for code in VALID_CATEGORY_CODES},
                "missing": missing_category_count,
                "invalid": invalid_category_count,
            },
            "percentages": {
                **{code: round(percent(category_counter.get(code, 0), total_parsed), 3) for code in VALID_CATEGORY_CODES},
                "missing": round(percent(missing_category_count, total_parsed), 3),
                "invalid": round(percent(invalid_category_count, total_parsed), 3),
            },
            "distribution": category_rows,
            "invalid_details": invalid_category_details,
        },
        "cross_tab_classification_vs_human_label_category": crosstab_rows,
    }

    return stats, top_summary_rows, category_rows, crosstab_rows, input_dir


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    top_n: int = max(1, args.top_n)

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")

    stats, top_summary_rows, category_rows, crosstab_rows, _ = analyze(input_dir=input_dir, top_n=top_n)

    output_dir.mkdir(parents=True, exist_ok=True)

    stats_json_path = output_dir / "human_label_stats.json"
    top_values_csv_path = output_dir / "human_change_summary_top_values.csv"
    category_csv_path = output_dir / "human_label_category_counts.csv"
    crosstab_csv_path = output_dir / "human_classification_vs_label_category.csv"
    chart_path = output_dir / "human_label_stats_chart.jpeg"

    stats_json_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(top_values_csv_path, top_summary_rows)
    write_csv(category_csv_path, category_rows)
    write_csv(crosstab_csv_path, crosstab_rows)

    classification_counter = Counter(
        {
            "True": stats[CLASSIFICATION_KEY]["counts"]["true"],
            "False": stats[CLASSIFICATION_KEY]["counts"]["false"],
            "Unknown": stats[CLASSIFICATION_KEY]["counts"]["unknown"],
        }
    )
    category_counter = Counter({row["category"]: row["count"] for row in category_rows})
    top_exact = [(row["summary"], row["count"]) for row in top_summary_rows]

    non_empty_rows = stats[CHANGE_SUMMARY_KEY]["non_empty_count"]
    summary_char_lengths = []
    if non_empty_rows > 0:
        all_json = sorted(input_dir.glob("*.json"))
        for path in all_json:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            summary = normalize_summary(payload.get(CHANGE_SUMMARY_KEY))
            if summary:
                summary_char_lengths.append(len(summary))

    save_visualization_jpeg(
        output_path=chart_path,
        total_files=stats["file_stats"]["json_files_parsed"],
        classification_counter=classification_counter,
        category_counter=category_counter,
        top_summaries=top_exact,
        summary_char_lengths=summary_char_lengths,
    )

    print("Human label analysis completed.")
    print(f"Input JSON directory : {input_dir}")
    print(f"Stats JSON          : {stats_json_path}")
    print(f"Top values CSV      : {top_values_csv_path}")
    print(f"Label category CSV  : {category_csv_path}")
    print(f"Cross-tab CSV       : {crosstab_csv_path}")
    print(f"Chart JPEG          : {chart_path}")


if __name__ == "__main__":
    main()
