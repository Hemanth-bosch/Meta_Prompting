#!/usr/bin/env python
"""Integrity guard for the NEW KB (mirror of utils/verify_kb_integrity.py for the old KB).

Confirms utils/new_weakspot_guidelines.json is a faithful projection of
utils/block_to_tc.json + the BU Excel:
  1. every block in block_to_tc.json maps to a TC sheet that EXISTS in the Excel,
  2. every mapped block appears in new_weakspot_guidelines.json with >=1 guideline,
  3. the JSON has no blocks that are absent from the mapping (no spurious entries),
  4. the JSON _metadata fingerprints match the current Excel + mapping (not stale).

Exit code 0 = faithful; 1 = mismatch (regenerate with generate_new_kb_json.py).

    .meta-prompting/Scripts/python utils/verify_new_kb_integrity.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import xlrd

_ROOT = Path(__file__).resolve().parent.parent
XLS = _ROOT / "MBD" / "weakspot_code_to_run_2" / "new_knowledge_base" / "TDI-mapped-FCI-Lib-Dynamic_Elements 2 (1).xls"
MAP_JSON = _ROOT / "utils" / "block_to_tc.json"
OUT_JSON = _ROOT / "utils" / "new_weakspot_guidelines.json"


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> int:
    problems: list[str] = []

    for p in (XLS, MAP_JSON, OUT_JSON):
        if not p.exists():
            print(f"FAIL: missing file {p}")
            return 1

    mapping = {k: v for k, v in json.loads(MAP_JSON.read_text(encoding="utf-8")).items() if not k.startswith("_")}
    kb = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    meta = kb.pop("_metadata", {})
    sheets = set(xlrd.open_workbook(str(XLS)).sheet_names())

    # 1. mapping -> real TC sheet
    for block, tc in mapping.items():
        if tc not in sheets:
            problems.append(f"mapping {block} -> {tc}: TC sheet not in Excel")

    # 2. every mapped block present with >=1 guideline
    for block in mapping:
        gl = kb.get(block)
        if not gl:
            problems.append(f"block {block}: absent or empty in new_weakspot_guidelines.json")
        elif not isinstance(gl, list) or not all(isinstance(x, str) and x.strip() for x in gl):
            problems.append(f"block {block}: malformed guideline list")

    # 3. no spurious blocks in the JSON
    for block in kb:
        if block not in mapping:
            problems.append(f"JSON block {block}: not in block_to_tc.json (spurious)")

    # 4. fingerprints fresh
    if meta.get("xls", {}).get("md5") != _md5(XLS):
        problems.append("stale: Excel MD5 differs from JSON _metadata (regenerate)")
    if meta.get("block_to_tc", {}).get("md5") != _md5(MAP_JSON):
        problems.append("stale: block_to_tc.json MD5 differs from JSON _metadata (regenerate)")

    if problems:
        print("NEW-KB INTEGRITY: FAIL")
        for p in problems:
            print("  -", p)
        return 1

    print(f"NEW-KB INTEGRITY: OK — {len(mapping)} block(s), all map to real TC sheets with guidelines; fingerprints fresh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
