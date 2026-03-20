"""
check_and_sort_jsons.py

1. Takes pair_001.json as the reference — extracts its keys and count.
2. Checks every other pair_NNN.json has EXACTLY the same keys (same names, same count).
   Reports any missing or extra keys per file.
3. Sorts ALL pair_NNN.json files (including pair_001) alphabetically by key and saves them.

Usage:
    python utils/dca_data_generation/check_and_sort_jsons.py
    python utils/dca_data_generation/check_and_sort_jsons.py --json-dir path/to/jsons
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_DIR = REPO_ROOT / "data" / "dca_data" / "image_pairs_llm_output_exp1"
REFERENCE_FILE = "pair_001.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_sorted(path: Path, data: dict) -> None:
    sorted_data = dict(sorted(data.items()))
    path.write_text(json.dumps(sorted_data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    args = parser.parse_args()

    json_dir: Path = args.json_dir
    files = sorted(json_dir.glob("pair_*.json"))
    if not files:
        print(f"[ERROR] No pair_*.json files in {json_dir}")
        return

    # -----------------------------------------------------------------------
    # Step 1: Load reference (pair_001.json)
    # -----------------------------------------------------------------------
    ref_path = json_dir / REFERENCE_FILE
    if not ref_path.exists():
        print(f"[ERROR] Reference file not found: {ref_path}")
        return

    ref_data = load_json(ref_path)
    ref_keys = set(ref_data.keys())
    ref_count = len(ref_keys)

    print(f"Reference: {REFERENCE_FILE}")
    print(f"  Key count : {ref_count}")
    print(f"  Keys      : {sorted(ref_keys)}\n")

    # -----------------------------------------------------------------------
    # Step 2: Check all files against reference
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("KEY CONSISTENCY CHECK")
    print("=" * 70)

    all_ok = True
    problems: list[str] = []

    for f in files:
        data = load_json(f)
        keys = set(data.keys())
        missing = ref_keys - keys          # in reference but not in this file
        extra   = keys   - ref_keys        # in this file but not in reference

        if missing or extra:
            all_ok = False
            msg = f"[MISMATCH] {f.name}  (count={len(keys)})"
            if missing:
                msg += f"\n    MISSING keys: {sorted(missing)}"
            if extra:
                msg += f"\n    EXTRA   keys: {sorted(extra)}"
            print(msg)
            problems.append(f.name)
        else:
            pass  # silent for matching files

    if all_ok:
        print(f"✓ All {len(files)} files have exactly the same {ref_count} keys as {REFERENCE_FILE}")
    else:
        print(f"\n{len(problems)} file(s) had key mismatches: {problems}")

    # -----------------------------------------------------------------------
    # Step 3: Sort all JSONs alphabetically by key
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SORTING ALL JSONs by key (alphabetical)")
    print("=" * 70)

    for f in files:
        data = load_json(f)
        save_json_sorted(f, data)

    print(f"✓ Sorted and saved {len(files)} files.")
    print(f"\nKey order after sorting:")
    sample = load_json(ref_path)
    for i, k in enumerate(sorted(sample.keys()), 1):
        print(f"  {i:02d}. {k}")


if __name__ == "__main__":
    main()
