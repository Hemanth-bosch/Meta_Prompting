#!/usr/bin/env python
"""One-time (or on-change) preprocessing: build the NEW knowledge base for M4 from the
BU's FCI test-case Excel + a block->TC mapping.

Run from the project root:

    .meta-prompting/Scripts/python utils/generate_new_kb_json.py

Inputs:
  - MBD/weakspot_code_to_run_2/new_knowledge_base/TDI-mapped-FCI-Lib-Dynamic_Elements 2 (1).xls
      (33 TC_xx sheets, cols: Label | Action | Description | ...; Action in Part/Info/Set/Wait/Check)
  - utils/block_to_tc.json   (ASCET block name -> TC_id; ONLY strong matches)

Output: utils/new_weakspot_guidelines.json   (block-keyed, SAME shape as the old KB
        utils/weakspot_guidelines.json, so evaluation.mbd_metric can look it up with the
        exact same find_block_type() logic).

Guideline granularity = PER TEST-CASE "Case" (scenario level, ~3-6 per block), matching the old
KB's granularity. Deterministic (no LLM). IDEMPOTENT: re-run is a no-op when the Excel + mapping
are unchanged (MD5+size stored in the JSON _metadata).

This mirrors utils/generate_kb_json.py (which does the same for the OLD HTML KB), minus the LLM
cross-reference resolution (not needed — the mapping is explicit).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import xlrd  # binary .xls reader (installed for the new KB)

_PROJ_ROOT = Path(__file__).resolve().parent.parent

# ── Paths ────────────────────────────────────────────────────────────────────
KB_XLS = (
    _PROJ_ROOT / "MBD" / "weakspot_code_to_run_2" / "new_knowledge_base"
    / "TDI-mapped-FCI-Lib-Dynamic_Elements 2 (1).xls"
)
MAP_JSON = _PROJ_ROOT / "utils" / "block_to_tc.json"
OUT_JSON = _PROJ_ROOT / "utils" / "new_weakspot_guidelines.json"

# TC sheet layout (verified): row 7 = header (Label|Action|Description|...), data from row 8.
_COL_ACTION = 1
_COL_DESC = 2
_MAX_GUIDELINES = 12   # verbatim per-scenario guidelines (init + cases + max/min/zero repeats)


# ── Staleness fingerprint (mirror generate_kb_json._html_metadata) ────────────
def _file_md5(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "size": stat.st_size,
        "md5": hashlib.md5(path.read_bytes()).hexdigest(),
    }


def _is_fresh(existing: dict) -> bool:
    """True if the output JSON's stored fingerprints still match the current inputs."""
    meta = (existing or {}).get("_metadata", {})
    try:
        return (
            meta.get("xls", {}).get("md5") == _file_md5(KB_XLS)["md5"]
            and meta.get("xls", {}).get("size") == KB_XLS.stat().st_size
            and meta.get("block_to_tc", {}).get("md5") == _file_md5(MAP_JSON)["md5"]
        )
    except Exception:
        return False


# ── Verbatim per-scenario guideline extraction ────────────────────────────────
# Each TC_xx sheet is Label|Action|Description with Action in Part/Info/Set/Wait/Check/Condition.
# A GUIDELINE = one test SCENARIO captured SAME-TO-SAME from the Excel: its header (an Info
# "Case N …"/"Checking …" line, or the sheet-title Part) PLUS the verbatim Set/Wait/Check/Condition
# /Info-note steps under it — INCLUDING the Check expected-outputs. The trailing Part
# "Repeat all cases with max/min/zero …" lines become their own guidelines (boundary-parameter tests).
# No paraphrasing/summarizing (that would deviate from the BU KB and needs an LLM); the only added
# text is the Action-column tag (Set:/Wait:/Check:/Given:) which is itself in the Excel.
_ACTION_TAG = {"Set": "Set", "Wait": "Wait", "Check": "Check", "Condition": "Given", "Info": ""}


def extract_guidelines(sheet) -> List[str]:
    rows = [
        (str(sheet.cell_value(r, _COL_ACTION)).strip(), str(sheet.cell_value(r, _COL_DESC)).strip())
        for r in range(sheet.nrows)
    ]

    scenarios: List[list] = []       # each = [header_str, [verbatim step strings]]
    cur = None
    title_seen = False
    for act, desc in rows:
        if not desc:
            continue
        is_case = act == "Info" and (re.match(r"Case\s*\d", desc, re.I) or desc.lower().startswith("checking"))
        is_repeat = act == "Part" and desc.lower().startswith("repeat")
        if act == "Part" and not is_repeat:
            if not title_seen:            # sheet-title Part → seed a header for any pre-Case steps
                title_seen = True
                cur = [desc, []]
            continue
        if is_case:
            if cur and cur[1]:            # flush the previous scenario (drop an empty title seed)
                scenarios.append(cur)
            cur = [re.sub(r"^Case\s*\d+\s*[:\-]?\s*", "", desc, flags=re.I).strip(), []]
        elif is_repeat:
            if cur and cur[1]:
                scenarios.append(cur)
                cur = None
            scenarios.append([desc, []])  # "repeat with max/min/zero" → its own guideline
        else:                             # Set / Wait / Check / Condition / Info-note → verbatim step
            if cur is None:
                cur = ["Checking", []]
            tag = _ACTION_TAG.get(act, act)
            cur[1].append(f"{tag}: {desc}" if tag else desc)
    if cur and cur[1]:
        scenarios.append(cur)

    guidelines = [f"{h} — " + "; ".join(steps) if steps else h for h, steps in scenarios]

    # dedupe (case-insensitive), preserve order, cap
    seen, out = set(), []
    for x in guidelines:
        x = x.strip()
        if x and x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out[:_MAX_GUIDELINES]


def _load_mapping() -> Dict[str, str]:
    raw = json.loads(MAP_JSON.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}  # skip _about/_tier metadata


def main() -> int:
    if not KB_XLS.exists():
        print(f"ERROR: Excel not found: {KB_XLS}")
        return 2
    if not MAP_JSON.exists():
        print(f"ERROR: mapping not found: {MAP_JSON}")
        return 2

    existing = {}
    if OUT_JSON.exists():
        try:
            existing = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    if existing and _is_fresh(existing):
        n = len([k for k in existing if not k.startswith("_")])
        print(f"[fresh] {OUT_JSON.name} up to date ({n} blocks); Excel + mapping unchanged. No-op.")
        return 0

    wb = xlrd.open_workbook(str(KB_XLS))
    sheet_names = set(wb.sheet_names())
    mapping = _load_mapping()

    out: Dict[str, object] = {}
    missing_tc, empty = [], []
    for block, tc_id in mapping.items():
        if tc_id not in sheet_names:
            missing_tc.append((block, tc_id))
            continue
        guides = extract_guidelines(wb.sheet_by_name(tc_id))
        if not guides:
            empty.append((block, tc_id))
            continue
        out[block] = guides

    if missing_tc:
        print(f"ERROR: mapping references missing TC sheet(s): {missing_tc}")
        return 3
    if empty:
        print(f"WARNING: no guidelines extracted for: {empty}")

    out["_metadata"] = {
        "source": "BU new_knowledge_base FCI test-case Excel (per-Case guidelines) + block_to_tc.json",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_blocks": len(mapping),
        "xls": _file_md5(KB_XLS),
        "block_to_tc": _file_md5(MAP_JSON),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[written] {OUT_JSON.name}: {len(mapping)} block(s)")
    for block, tc_id in mapping.items():
        print(f"   {block:26} <- {tc_id}  ({len(out.get(block, []))} guidelines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
