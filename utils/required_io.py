"""[wrapper-only] Loader for the expert-curated Required-I/O KB — no run.py analog.

Reads ``utils/required_io.json`` (keyed by weakspot INSTANCE ``"fc_name#ws_id"``) and exposes a single
resolver. The KB feeds two metrics in ``evaluation/mbd_metric.py``:
  * M1  — ``required_outputs`` OVERRIDES the auto-derived ``instance_names`` when a curated entry exists.
  * M2  — ``required_inputs`` + ``required_preconds`` drive an input/precond COMPLETENESS check.

Design (mirrors the M4 KB skip-behaviour): a missing key, or an entry whose lists are all empty, resolves
to ``None`` so the caller keeps its DEFAULT behaviour (no penalty on uncurated weakspots).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("mbd_metric")  # [wrapper-only] share the metric logger for trace continuity

# [wrapper-only] KB path is CONFIGURABLE via env MBD_REQUIRED_IO_PATH (one KB drives BOTH scoring — M1/M2 —
# AND the optional Prompt-2 output override). Unset -> the default curated file next to this module. This is
# the single "define the path" knob the BU asked for; point it at an expert-curated KB per run.
_DEFAULT_REQ_IO_JSON = Path(__file__).parent / "required_io.json"  # [wrapper-only] default curated KB file
_REQ_IO_JSON = Path(os.getenv("MBD_REQUIRED_IO_PATH") or _DEFAULT_REQ_IO_JSON)  # [wrapper-only] env-overridable KB path
_LISTS = ("required_inputs", "required_preconds", "required_outputs")  # [wrapper-only] the three curated lists


def _load(path: Path) -> Dict[str, dict]:  # [wrapper-only] parse the KB JSON once (cached below)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))  # [wrapper-only] read the whole KB
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:  # [wrapper-only] missing/bad file -> empty KB (skip everything)
        logger.warning("required_io: could not load %s (%s) -> KB empty (all checks skipped)", path, e)  # [wrapper-only] trace
        return {}
    out: Dict[str, dict] = {}  # [wrapper-only] normalized {key: {list: [...]}} (drops _README/_source meta keys)
    for key, entry in raw.items():  # [wrapper-only] iterate curated weakspot entries
        if key.startswith("_") or not isinstance(entry, dict):  # [wrapper-only] skip meta keys (_README) / malformed
            continue
        out[key] = {lst: [str(x).strip() for x in (entry.get(lst) or []) if str(x).strip()] for lst in _LISTS}  # [wrapper-only] keep only the 3 lists, cleaned
    return out


_REQ_IO: Dict[str, dict] = _load(_REQ_IO_JSON)  # [wrapper-only] module-level cache (loaded once at import)


def _key(fc_name: str, ws_id, op_key=None) -> str:  # [wrapper-only] canonical KB key "fc#ws" or "fc#ws#op"
    return f"{fc_name}#{ws_id}#{op_key}" if op_key else f"{fc_name}#{ws_id}"  # [wrapper-only] op-disambiguated when op_key given


def _lookup(fc_name: str, ws_id, op_key=None) -> Optional[dict]:
    """[wrapper-only] Look up an entry, preferring the OP-DISAMBIGUATED key 'fc#ws#op' (needed when two
    different weakspot ops share one fc#ws, e.g. a2mamg#0's getBit $op4 vs putBit $op5), then falling
    back to the plain 'fc#ws' key. Non-duplicated weakspots keep their plain 'fc#ws' entries (fallback)."""
    if op_key:  # [wrapper-only] try op-specific first
        e = _REQ_IO.get(_key(fc_name, ws_id, op_key))  # [wrapper-only] "fc#ws#op"
        if e is not None:  # [wrapper-only] op-specific entry wins
            return e
    return _REQ_IO.get(_key(fc_name, ws_id))  # [wrapper-only] plain "fc#ws" fallback


def set_kb_path(path) -> int:
    """[wrapper-only] Re-point the KB at `path` and reload it AT RUNTIME (for a `--required-io-path` CLI
    flag). Rebinds the module-level cache, so BOTH scoring (mbd_metric's resolve_required_io) and the
    Prompt-2 override (resolve_required_outputs_str) pick up the new KB. Returns the #entries loaded."""
    global _REQ_IO_JSON, _REQ_IO  # [wrapper-only] rebind the module cache both resolvers read
    _REQ_IO_JSON = Path(path)  # [wrapper-only] new KB path
    _REQ_IO = _load(_REQ_IO_JSON)  # [wrapper-only] reload + re-normalize
    logger.info("required_io: KB re-pointed to %s (%d entries)", _REQ_IO_JSON, len(_REQ_IO))  # [wrapper-only] trace
    return len(_REQ_IO)  # [wrapper-only] #curated entries now active


def resolve_required_io(fc_name: str, ws_id, op_key=None) -> Optional[Dict[str, List[str]]]:
    """[wrapper-only] Return the curated {required_inputs, required_preconds, required_outputs} for this
    weakspot instance, or None when there is no curated entry OR all three lists are empty (=> the caller
    should keep its default behaviour and NOT penalise). ``op_key`` (weakspot_op_key) disambiguates two
    weakspot ops sharing one fc#ws (tries 'fc#ws#op' then 'fc#ws')."""
    entry = _lookup(fc_name, ws_id, op_key)  # [wrapper-only] op-specific then fc#ws fallback
    if not entry or not any(entry.get(lst) for lst in _LISTS):  # [wrapper-only] absent / fully empty -> skip
        return None
    return entry  # [wrapper-only] curated lists (any individual list may still be empty)


def resolve_required_outputs_str(fc_name: str, ws_id, op_key=None) -> Optional[str]:
    """[wrapper-only] Return the curated required_outputs for this weakspot as a comma-joined string
    (the shape Prompt 2's `instance_names` field expects), or None when there is no curated entry or
    it lists no outputs. Used by extract_mbd_full.forward() to OVERRIDE the Prompt-2 'use this Expected
    Results' field so experts can steer generation to an expert-chosen (e.g. intermediate) output.
    ``op_key`` disambiguates duplicate fc#ws weakspots. Fallback to the dataset instance_names is the
    caller's responsibility (None => keep instance_names)."""
    entry = _lookup(fc_name, ws_id, op_key)  # [wrapper-only] op-specific then fc#ws fallback (same KB as scoring)
    outs = [str(x).strip() for x in ((entry or {}).get("required_outputs") or []) if str(x).strip()]  # [wrapper-only] cleaned list
    return ",".join(outs) if outs else None  # [wrapper-only] comma-joined (instance_names shape) or None -> caller keeps default
