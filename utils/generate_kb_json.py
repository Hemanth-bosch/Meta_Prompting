#!/usr/bin/env python
"""One-time (or on-change) preprocessing script: extract ASCET weakspot guidelines
from the HTML knowledge base and resolve all cross-references using LLM.

Run from the project root:

    .meta-prompting/Scripts/python utils/generate_kb_json.py

Output: utils/weakspot_guidelines.json

The script is IDEMPOTENT: it compares the MD5 + size of the HTML source against
metadata stored inside the JSON.  If the HTML has not changed since the last run,
the existing JSON is reused and no LLM call is made.

Table structure (3 columns):
    S.No | Weakspot elements | Hint: Critical checks to perform

Cross-reference resolution rules applied by the LLM:
  REPLACE  "Refer X", "Same as X", "Tested with X", "Verified along with X",
           "Used to get the result from calling X and tested along with it"
           "Please refer row: X"          -> use X's guidelines only
  MERGE    "Similar test cases to X when Y"  -> own items + all of X's
  EMPTY    no text / only empty tags / referenced block not in KB   -> []
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Project root so config / dspy are importable ──────────────────────────────
_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

import config  # noqa: F401  -- runs dspy.configure(lm=...) as a side effect
import dspy
from bs4 import BeautifulSoup  # type: ignore[import]


# ── Paths ──────────────────────────────────────────────────────────────────────
KB_HTML = (
    _PROJ_ROOT
    / "MBD"
    / "weakspot_code_to_run_2"
    / "knowledge_base"
    / "ASCET Block Weakspot list - Software Testing Knowledge Base.html"
)
OUT_JSON = _PROJ_ROOT / "utils" / "weakspot_guidelines.json"

# All blocks in a single LLM call so cross-batch reference resolution works
# (93 blocks ~ 4 k tokens, well within 128 k context).
BATCH_SIZE = 200


# ═══════════════════════════════════════════════════════════════════════════════
# HTML metadata helpers  (for staleness detection)
# ═══════════════════════════════════════════════════════════════════════════════

def _html_metadata(html_path: Path) -> dict:
    """Compute fingerprint of the HTML source file."""
    stat = html_path.stat()
    digest = hashlib.md5(html_path.read_bytes()).hexdigest()
    return {
        "html_path": str(html_path),
        "html_mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "html_size": stat.st_size,
        "html_md5": digest,
    }


def _json_is_fresh(json_path: Path, html_path: Path) -> bool:
    """Return True if the JSON was generated from the exact same HTML file."""
    if not json_path.exists():
        return False
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        stored = saved.get("_metadata", {})
        current = _html_metadata(html_path)
        return (
            stored.get("html_md5") == current["html_md5"]
            and stored.get("html_size") == current["html_size"]
        )
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# DSPy Signature
# ═══════════════════════════════════════════════════════════════════════════════

class _ResolveKBBatch(dspy.Signature):
    """Resolve ASCET weakspot guideline cross-references and output structured JSON."""

    task: str = dspy.InputField(
        desc="Full instructions including all block names and their raw guideline text"
    )
    resolved_json: str = dspy.OutputField(
        desc=(
            "Valid JSON object -- no markdown fences, no explanation. "
            "Maps block_name -> list of resolved guideline strings. "
            "Use [] for blocks with no guidelines. "
            'Example: {"Srv_Debounce": ["Check rising edge", "Check falling edge"]}'
        )
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HTML extraction
# ═══════════════════════════════════════════════════════════════════════════════

_UNICODE_CLEAN_RE = re.compile(r"[\ufffd\u00b7]")


def _cell_text(cell) -> str:
    """Return normalised guideline text from a <td>/<th>.

    - Removes <s> (strikethrough) content -- typically stale/overridden text.
    - Joins <p>/<li> children with ' | '.
    - Falls back to full cell text when no block-level children exist.
    - Strips unicode replacement characters.
    """
    for s_tag in cell.find_all("s"):
        s_tag.decompose()

    parts = [
        tag.get_text(separator=" ", strip=True)
        for tag in cell.find_all(["p", "li"])
    ]
    text = " | ".join(p for p in parts if p) if parts else cell.get_text(separator=" | ", strip=True)
    text = _UNICODE_CLEAN_RE.sub("", text)
    text = re.sub(r"\s*\|\s*\|", " |", text)
    text = re.sub(r"^\s*\|\s*", "", text)
    return text.strip()


def extract_raw_blocks(html_path: Path) -> List[Tuple[str, str]]:
    """Parse the HTML knowledge base -> list of (block_name, raw_hint_text).

    Table columns: S.No (0) | Weakspot elements (1) | Hint (2)
    """
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    tables = soup.find_all("table")
    if not tables:
        raise ValueError("No <table> found in HTML")
    main_table = max(tables, key=lambda t: len(t.find_all("tr")))
    result: List[Tuple[str, str]] = []
    for row in main_table.find_all("tr")[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) < 3:
            continue
        name = cells[1].get_text(strip=True)   # col 1 = Weakspot elements
        if not name:
            continue
        text = _cell_text(cells[2])             # col 2 = Hint
        result.append((name, text))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# LLM resolution
# ═══════════════════════════════════════════════════════════════════════════════

_TASK_TEMPLATE = """\
You are resolving the ASCET block weakspot software testing knowledge base.

All available block names in the KB (use ONLY these for cross-reference lookup):
{all_names}

Resolution rules -- apply to EVERY block in the list below:

RULE 1 -- REPLACE (use ONLY the referenced block's resolved guidelines, no own text):
  Triggers (any of these phrases):
    "Refer X"
    "do Refer X"          (strikethrough prefix "do" can be ignored)
    "Please refer row: X"
    "Please refer row X"
    "Same as X"
    "Tested with X"
    "Verified along with X when ..."
    "Used to get the result from calling X and tested along with it"
  Action: output the fully-resolved guidelines of block X.
  If X is not in the available block names list -> output [].
  Short-name mappings (use these when the name is abbreviated):
    "Pwin"   -> "Srv_PWin"
    "IWin"   -> "Srv_IWin"
    "DT1Win" -> "Srv_DT1Win"

RULE 2 -- MERGE (own items + referenced block's guidelines):
  Trigger: "Similar test cases to X when Y" (appearing anywhere in the text)
  Action: take the block's OWN numbered items listed BEFORE the trigger phrase,
          then append ALL of block X's resolved guidelines.
  Example: Srv_RampSwitch has items 1 and 2, then "Similar test cases to Srv_Ramp ..."
           -> output [item1, item2] + all of Srv_Ramp's items.

RULE 3 -- EMPTY:
  If the raw text is blank, or all referenced blocks don't exist in the KB -> [].

RULE 4 -- CLEAN direct guidelines:
  a) Strip lead-in sentences that are not test scenarios themselves, e.g.:
       "Test the following scenarios"
       "Test the followig scenarios"   (typo variant)
       "Calculates the ramp slope based on Input."
  b) Numbered items like "1. text", "1: text", "2.text" -> strip the prefix,
     keep only the test description.
  c) Pipe-separated segments (from multi-paragraph HTML) -> split into entries.
  d) Each output string should be a single readable test-scenario sentence.

IMPORTANT:
- Resolve cross-references recursively (if X itself has a cross-reference, follow it).
- If a block references multiple blocks (e.g. "Tested with A, B, C") and NONE of them
  exist in the available block names list, output [].
- "Tested with SrvF_PDT1" means use SrvF_PDT1's guidelines (not SrvF_PDT1Win's).
- Output ONLY valid JSON -- absolutely no markdown fences, no explanatory prose.

Blocks to resolve:
{block_lines}

Required output format (JSON object, one key per block):
{{"BlockName": ["resolved guideline 1", "resolved guideline 2", ...], ...}}"""


def _build_task(batch: List[Tuple[str, str]], all_names: List[str]) -> str:
    block_lines = "\n".join(
        f"  {name}: {text if text else '(empty)'}"
        for name, text in batch
    )
    return _TASK_TEMPLATE.format(
        all_names=", ".join(all_names),
        block_lines=block_lines,
    )


def _strip_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _parse_json(raw: str) -> Dict[str, List[str]]:
    cleaned = _strip_fence(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError(
            f"LLM output is not valid JSON.\n--- start ---\n{cleaned[:800]}\n--- end ---"
        )


def resolve_batch(
    batch: List[Tuple[str, str]],
    all_names: List[str],
    predictor: dspy.Predict,
) -> Dict[str, List[str]]:
    task_str = _build_task(batch, all_names)
    result = predictor(task=task_str)
    return _parse_json(result.resolved_json)


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════

def _validate(
    resolved: Dict[str, List[str]],
    all_names: List[str],
) -> List[str]:
    warnings: List[str] = []

    for name in all_names:
        if name not in resolved:
            warnings.append(f"MISSING block: {name}")

    known_replace: Dict[str, str] = {
        "Srv_DebounceParam_t": "Srv_Debounce",
        "SrvF_PT1":            "Srv_PT1",
        "SrvF_PT1Param_t":     "Srv_PT1",
        "Srv_DT1Param_t":      "Srv_DT1",
        "Srv_PWinParam_t":     "Srv_PWin",
        "Srv_IWinParam_t":     "Srv_IWin",
        "Srv_DT1WinParam_t":   "Srv_DT1Win",
        "Srv_IntLimParam_t":   "Srv_IntLimit",
        "Pwmin_ResultXt_t":    "PWM_InXt",
        "Pwmin_Result_t":      "PWM_In",
        "Pwmin_ResultPer_t":   "PWM_InPer",
        "Srv_PDT1Param_t":     "SrvF_PDT1",
        "Srv_PDT1WinParam_t":  "SrvF_PDT1",
        "Pwmout_appar_t":      "PWM_Out",
        "SrvF_Ramp":           "Srv_Ramp",
    }
    for block, target in known_replace.items():
        tgt_g = resolved.get(target, [])
        blk_g = resolved.get(block, [])
        if tgt_g and not blk_g:
            warnings.append(f"REPLACE failed: {block} -> {target} (got [])")
        elif tgt_g and blk_g != tgt_g:
            warnings.append(
                f"REPLACE mismatch: {block} ({len(blk_g)}) != {target} ({len(tgt_g)})"
            )

    ramp_switch = resolved.get("Srv_RampSwitch", [])
    ramp        = resolved.get("Srv_Ramp", [])
    if ramp and len(ramp_switch) < len(ramp):
        warnings.append(
            f"MERGE failed: Srv_RampSwitch ({len(ramp_switch)}) < Srv_Ramp ({len(ramp)})"
        )

    empty_prefixes = ("SrvMT_", "Byp_", "BYP_", "getEb", "Bypass_", "Call_Dependant")
    for name, guidelines in resolved.items():
        if any(name.startswith(p) for p in empty_prefixes) and guidelines:
            warnings.append(f"UNEXPECTED non-empty: {name}")

    return warnings


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"KB HTML  : {KB_HTML}")
    print(f"Output   : {OUT_JSON}")

    if not KB_HTML.exists():
        print("ERROR: KB HTML file not found.")
        sys.exit(1)

    # ── Staleness check ────────────────────────────────────────────────────────
    if _json_is_fresh(OUT_JSON, KB_HTML):
        print("\nJSON is up-to-date (HTML unchanged since last generation). Nothing to do.")
        print("To force regeneration, delete the JSON file and re-run.")
        return

    print("\nStep 1 - Extracting raw blocks from HTML (col1=name, col2=hint)...")
    raw_blocks = extract_raw_blocks(KB_HTML)
    print(f"  {len(raw_blocks)} blocks extracted")
    if raw_blocks:
        print(f"  First: [{raw_blocks[0][0]}] -> {raw_blocks[0][1][:80]}")

    all_names = [name for name, _ in raw_blocks]
    predictor = dspy.Predict(_ResolveKBBatch)
    resolved: Dict[str, List[str]] = {}

    batches = [raw_blocks[i : i + BATCH_SIZE] for i in range(0, len(raw_blocks), BATCH_SIZE)]
    print(f"\nStep 2 - Resolving via LLM ({len(batches)} batch(es) x up to {BATCH_SIZE} blocks each)...")

    for idx, batch in enumerate(batches, 1):
        names_in_batch = [n for n, _ in batch]
        print(f"  Batch {idx}/{len(batches)}: [{names_in_batch[0]}] ... [{names_in_batch[-1]}] ({len(batch)} blocks)")
        batch_result = resolve_batch(batch, all_names, predictor)
        resolved.update(batch_result)
        non_empty = sum(1 for v in resolved.values() if v)
        print(f"    -> {len(batch_result)} blocks resolved | {non_empty} non-empty so far")

    missing = set(all_names) - set(resolved.keys())
    if missing:
        print(f"\nWARNING: {len(missing)} block(s) absent from LLM output -- filling with []:")
        for name in sorted(missing):
            print(f"  {name}")
            resolved[name] = []

    # ── Save with embedded metadata ────────────────────────────────────────────
    print(f"\nStep 3 - Saving to {OUT_JSON} ...")
    meta = _html_metadata(KB_HTML)
    meta["generated_at"] = datetime.now().isoformat(timespec="seconds")

    output_doc: Dict = {"_metadata": meta}
    output_doc.update(resolved)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_doc, f, indent=2, ensure_ascii=False)

    non_empty = sum(1 for k, v in resolved.items() if v)
    print(f"  Saved {len(resolved)} blocks ({non_empty} with guidelines, {len(resolved)-non_empty} empty)")
    print(f"  HTML fingerprint: md5={meta['html_md5'][:12]}... size={meta['html_size']} mtime={meta['html_mtime']}")

    # ── Validation ─────────────────────────────────────────────────────────────
    print("\nStep 4 - Validating output...")
    warnings = _validate(resolved, all_names)
    if warnings:
        print(f"  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  [WARN] {w}")
    else:
        print("  All validation checks passed.")

    # ── Sample ─────────────────────────────────────────────────────────────────
    print("\nSample -- first 5 blocks with guidelines:")
    count = 0
    for name, guidelines in resolved.items():
        if guidelines:
            print(f"  [{name}]  ({len(guidelines)} items)")
            for g in guidelines[:3]:
                print(f"    - {g[:100]}")
            count += 1
            if count >= 5:
                break


if __name__ == "__main__":
    main()
