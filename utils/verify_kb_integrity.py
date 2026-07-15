#!/usr/bin/env python3
"""KB integrity guard — is weakspot_guidelines.json a faithful 1:1 of the source HTML?

The KB is single-source-of-truth: `MBD/weakspot_code_to_run_2/knowledge_base/*.html`.
`utils/generate_kb_json.py` projects it into `utils/weakspot_guidelines.json`. This guard
confirms the projection dropped (or invented) NOTHING:

  - blocks in HTML but MISSING from JSON  -> a real generator gap (would warrant a fix/supplement)
  - blocks in JSON but NOT in HTML        -> an unexpected addition

Per policy: we only ever supplement the KB if the generator dropped a block that IS in the
HTML. Blocks genuinely absent from the HTML (e.g. Srv_TrnOffDly, getBit, putBit, EdgeRising)
are correctly absent and are handled by M4's neutral fallback — NOT supplemented.

Run (project venv):
    .meta-prompting\\Scripts\\python.exe utils\\verify_kb_integrity.py
Exit code 0 = faithful (no gap); 1 = mismatch found.
"""
from __future__ import annotations

import glob
import importlib
import json
import os
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HTML_GLOB = os.path.join(_ROOT, "MBD", "weakspot_code_to_run_2", "knowledge_base", "*.html")
_JSON = os.path.join(_ROOT, "utils", "weakspot_guidelines.json")


def main() -> int:
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    html_files = glob.glob(_HTML_GLOB)
    if not html_files:
        print(f"KB-INTEGRITY: ERROR — no source HTML found at {_HTML_GLOB}")
        return 1
    html_path = Path(html_files[0])

    gen = importlib.import_module("utils.generate_kb_json")
    html_blocks = {b[0].strip() for b in gen.extract_raw_blocks(html_path)}

    kb = json.load(open(_JSON, encoding="utf-8"))
    kb.pop("_metadata", None)
    json_keys = set(kb.keys())

    missing = sorted(html_blocks - json_keys)   # in HTML, dropped from JSON  -> real gap
    added = sorted(json_keys - html_blocks)      # in JSON, not in HTML        -> unexpected

    print(f"KB-INTEGRITY: HTML blocks={len(html_blocks)}  JSON keys={len(json_keys)}")
    ok = not missing and not added
    if missing:
        print(f"  ❌ in HTML but MISSING from JSON (generator gap — supplement these): {missing}")
    if added:
        print(f"  ❌ in JSON but NOT in HTML (unexpected addition): {added}")
    if ok:
        print("  ✅ faithful 1:1 projection — no dropped or invented blocks. No supplement needed.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
