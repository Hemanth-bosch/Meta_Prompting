"""
V4 multi-step pipeline for DCA image-pair analysis.

Architecture
------------
Phase 0  — Python rule-based pre-check (zero LLM cost)
           • File existence  → image_count = 0/1/2
           • PIL sanity      → flag corrupted images
Phase 1  — LLM Step 1: Triage
           • 1-image pairs   → v4_step1_triage_1img.md  (is_informational?)
           • 2-image pairs   → v4_step1_triage_2img.md  (branch B1-B4 or both_informational)
Phase 2  — LLM Step 2: Diff analysis (only when branch == "both_informational")
           → v4_step2_diff.md  (trivial vs meaningful changes + description)
Phase 3  — Assemble & save three keys into pair_N.json:
           {prefix}_discard_image, {prefix}_changes, {prefix}_metadata

The _metadata key is eval-compatible with eval_human_runs.py:
  response.usage tokens and timing.response_time_seconds are SUMMED across steps.

Examples
--------
python evaluation/dca_baseline_runs/run_human_prompts/run_v4_human_prompt.py \\
    --key-prefix v4_prompt_cr_gpt41 \\
    --deployment gpt-4.1

python evaluation/dca_baseline_runs/run_human_prompts/run_v4_human_prompt.py \\
    --key-prefix v4_prompt_cr_gpt52_rlow \\
    --deployment gpt-5.2 --reasoning-effort low
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

import certifi
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI, DefaultHttpxClient
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DCA_DIR = REPO_ROOT / "data" / "dca_data"
DEFAULT_JSON_DIR = DCA_DIR / "image_pairs_llm_output_exp1"
DEFAULT_IMAGE_PAIRS_DIR = DCA_DIR / "image_pairs"
HUMAN_PROMPTS_DIR = REPO_ROOT / "evaluation" / "dca_baseline_runs" / "human_prompts"

PROMPT_1IMG = "v4_step1_triage_1img.md"
PROMPT_2IMG = "v4_step1_triage_2img.md"
PROMPT_DIFF = "v4_step2_diff.md"

_MIN_PIL_SIZE = (5, 5)


# ---------------------------------------------------------------------------
# Utilities (shared with run_with_human_prompts.py style)
# ---------------------------------------------------------------------------

def load_prompt(name: str) -> str:
    path = HUMAN_PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("run_v4_human_prompt")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if logger.handlers:
        return logger
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.DEBUG if verbose else logging.INFO)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(h)
    return logger


def _build_ca_bundle(logger: logging.Logger) -> str:
    for var in ("AOAI_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        value = os.getenv(var)
        if value and Path(value).exists():
            logger.debug("Using CA bundle from %s: %s", var, value)
            return value
    return certifi.where()


def build_client(endpoint: str, api_version: str, logger: logging.Logger) -> AzureOpenAI:
    ca_bundle = _build_ca_bundle(logger)
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if api_key:
        logger.info("Using AZURE_OPENAI_API_KEY authentication")
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            http_client=DefaultHttpxClient(verify=ca_bundle),
        )
    logger.info("Using Azure CLI authentication")
    credential = AzureCliCredential()
    token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
    return AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
        http_client=DefaultHttpxClient(verify=ca_bundle),
    )


def image_to_data_url(image_path: Path) -> str:
    ext_to_mime = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
        ".tif": "image/tiff", ".tiff": "image/tiff",
    }
    media_type = ext_to_mime.get(image_path.suffix.lower(), "image/png")
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{media_type};base64,{b64}"


def parse_pair_index(json_path: Path) -> Optional[int]:
    match = re.match(r"^pair_(\d+)\.json$", json_path.name)
    return int(match.group(1)) if match else None


def find_image(pair_dir: Path, stem: str) -> Optional[Path]:
    allowed = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
    matches = [p for p in sorted(pair_dir.glob(f"{stem}.*")) if p.suffix.lower() in allowed]
    return matches[0] if matches else None


def normalize_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "1"}:
            return True
        if v in {"false", "no", "0"}:
            return False
    if isinstance(value, int):
        return bool(value)
    return None


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    for method in ("model_dump", "dict"):
        if hasattr(value, method):
            try:
                return to_jsonable(getattr(value, method)())
            except Exception:
                pass
    if hasattr(value, "__dict__"):
        try:
            return to_jsonable(vars(value))
        except Exception:
            pass
    return str(value)


def compact_json(data: Any) -> str:
    return json.dumps(to_jsonable(data), ensure_ascii=False, separators=(",", ":"))


def _strip_base64(content: object) -> object:
    if isinstance(content, list):
        result = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url", "")
                stripped = url[:40] + "...<base64_omitted>" if url.startswith("data:") else url
                result.append({"type": "image_url", "image_url": {"url": stripped}})
            else:
                result.append(item)
        return result
    return content


def loggable_messages(messages: list) -> list:
    return [{"role": m["role"], "content": _strip_base64(m["content"])} for m in messages]


# ---------------------------------------------------------------------------
# Phase 0: PIL rule-based pre-check
# ---------------------------------------------------------------------------

def phase0_check(original: Optional[Path], reimage: Optional[Path], logger: logging.Logger) -> dict:
    """
    Returns:
      image_count      : 0 / 1 / 2
      original_ok      : bool or None
      reimage_ok       : bool or None
      corrupted        : bool  (True if PIL check failed for any existing file)
      detail           : list[str]
    """
    result: dict[str, Any] = {
        "original_path": str(original) if original else None,
        "reimage_path": str(reimage) if reimage else None,
        "original_exists": original is not None and original.exists(),
        "reimage_exists": reimage is not None and reimage.exists(),
        "original_ok": None,
        "reimage_ok": None,
        "image_count": 0,
        "corrupted": False,
        "detail": [],
    }

    paths_to_check: list[tuple[str, Optional[Path]]] = [
        ("original", original if result["original_exists"] else None),
        ("reimage", reimage if result["reimage_exists"] else None),
    ]
    count = 0
    corrupted = False

    for label, path in paths_to_check:
        if path is None:
            result[f"{label}_ok"] = None
            continue

        count += 1
        ok = True

        if not _PIL_AVAILABLE:
            # Can't do PIL check; assume OK
            result["detail"].append(f"{label}: PIL not available, assuming OK")
            result[f"{label}_ok"] = True
            continue

        try:
            if path.stat().st_size == 0:
                ok = False
                result["detail"].append(f"{label}: zero-byte file")
            else:
                with PILImage.open(path) as img:
                    w, h = img.size
                    if w < _MIN_PIL_SIZE[0] or h < _MIN_PIL_SIZE[1]:
                        ok = False
                        result["detail"].append(f"{label}: too tiny ({w}x{h})")
                    else:
                        result["detail"].append(f"{label}: OK ({w}x{h})")
        except Exception as exc:
            ok = False
            result["detail"].append(f"{label}: PIL error — {exc}")

        result[f"{label}_ok"] = ok
        if not ok:
            corrupted = True

    result["image_count"] = count
    result["corrupted"] = corrupted
    return result


# ---------------------------------------------------------------------------
# Model call (generic, reusable for both steps)
# ---------------------------------------------------------------------------

class ModelCallFailure(Exception):
    def __init__(self, message: str, metadata: dict):
        super().__init__(message)
        self.metadata = metadata


def call_model(
    client: AzureOpenAI,
    deployment: str,
    images: list[Path],          # ordered list of image paths to send
    prompt_text: str,
    logger: logging.Logger,
    save_full_response: bool,
    max_retries: int = 3,
    retry_backoff_base: float = 2.0,
    reasoning_effort: Optional[str] = None,
    step_label: str = "step",
) -> dict:
    """
    Calls the LLM with given images + prompt.
    Returns a metadata dict with keys: status, request, response, timing, parse.
    Raises ModelCallFailure on unrecoverable error.
    """
    user_content: list[dict] = []
    for img_path in images:
        user_content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img_path)}})
    user_content.append({"type": "text", "text": prompt_text})

    messages = [
        {"role": "system", "content": "You are an expert assistant for comparing document image pairs."},
        {"role": "user", "content": user_content},
    ]
    messages_log = loggable_messages(messages)

    total_start = time.perf_counter()
    retry_log: list[dict] = []
    last_exc: Optional[Exception] = None
    response = None

    _RATELIMIT_DEFAULT_WAIT = 30.0

    def _retry_after(exc: RateLimitError) -> float:
        try:
            val = exc.response.headers.get("Retry-After")
            if val is not None:
                return float(val)
        except Exception:
            pass
        return _RATELIMIT_DEFAULT_WAIT

    for attempt in range(1, max_retries + 2):
        ts_start = datetime.now(timezone.utc).isoformat()
        t_start = time.perf_counter()
        try:
            extra: dict = {}
            if reasoning_effort is not None:
                extra["reasoning_effort"] = reasoning_effort
            response = client.chat.completions.create(
                model=deployment,
                messages=messages,
                response_format={"type": "json_object"},
                **extra,
            )
            t_end = time.perf_counter()
            ts_end = datetime.now(timezone.utc).isoformat()
            retry_log.append({
                "attempt": attempt, "status": "success",
                "timestamp_start": ts_start, "timestamp_end": ts_end,
                "duration_seconds": round(t_end - t_start, 3),
            })
            last_exc = None
            break
        except RateLimitError as exc:
            last_exc = exc
            t_end = time.perf_counter()
            ts_end = datetime.now(timezone.utc).isoformat()
            wait = _retry_after(exc)
            retry_log.append({
                "attempt": attempt, "status": "failed", "error": "RateLimitError",
                "timestamp_start": ts_start, "timestamp_end": ts_end,
                "duration_seconds": round(t_end - t_start, 3), "wait_seconds": wait, "detail": str(exc),
            })
            logger.warning("[%s attempt %d/%d] RateLimitError — waiting %.1fs", step_label, attempt, max_retries + 1, wait)
        except APITimeoutError as exc:
            last_exc = exc
            t_end = time.perf_counter()
            ts_end = datetime.now(timezone.utc).isoformat()
            wait = retry_backoff_base ** attempt
            retry_log.append({
                "attempt": attempt, "status": "failed", "error": "APITimeoutError",
                "timestamp_start": ts_start, "timestamp_end": ts_end,
                "duration_seconds": round(t_end - t_start, 3), "wait_seconds": wait, "detail": str(exc),
            })
            logger.warning("[%s attempt %d/%d] APITimeoutError — waiting %.1fs", step_label, attempt, max_retries + 1, wait)
        except APIConnectionError as exc:
            last_exc = exc
            t_end = time.perf_counter()
            ts_end = datetime.now(timezone.utc).isoformat()
            wait = retry_backoff_base ** attempt
            retry_log.append({
                "attempt": attempt, "status": "failed", "error": "APIConnectionError",
                "timestamp_start": ts_start, "timestamp_end": ts_end,
                "duration_seconds": round(t_end - t_start, 3), "wait_seconds": wait, "detail": str(exc),
            })
            logger.warning("[%s attempt %d/%d] APIConnectionError — waiting %.1fs", step_label, attempt, max_retries + 1, wait)
        except APIStatusError as exc:
            last_exc = exc
            t_end = time.perf_counter()
            ts_end = datetime.now(timezone.utc).isoformat()
            if exc.status_code in (500, 502, 503, 504):
                wait = retry_backoff_base ** attempt
                retry_log.append({
                    "attempt": attempt, "status": "failed", "error": f"APIStatusError_{exc.status_code}",
                    "timestamp_start": ts_start, "timestamp_end": ts_end,
                    "duration_seconds": round(t_end - t_start, 3), "wait_seconds": wait, "detail": str(exc),
                })
                logger.warning("[%s attempt %d/%d] APIStatusError %d — waiting %.1fs", step_label, attempt, max_retries + 1, exc.status_code, wait)
            else:
                retry_log.append({
                    "attempt": attempt, "status": "failed", "error": f"APIStatusError_{exc.status_code}",
                    "timestamp_start": ts_start, "timestamp_end": ts_end,
                    "duration_seconds": round(t_end - t_start, 3), "wait_seconds": 0, "detail": str(exc),
                })
                logger.error("[%s attempt %d] APIStatusError %d — not retryable", step_label, attempt, exc.status_code)
                raise ModelCallFailure(str(exc), _build_fail_meta(
                    deployment, images, messages_log, retry_log, total_start,
                    f"APIStatusError_{exc.status_code}", str(exc)
                )) from exc

        if attempt <= max_retries and last_exc is not None:
            time.sleep(wait)
    else:
        logger.error("[%s] All %d retries exhausted. Last: %s", step_label, max_retries, last_exc)
        raise ModelCallFailure(str(last_exc), _build_fail_meta(
            deployment, images, messages_log, retry_log, total_start,
            type(last_exc).__name__ if last_exc else "UnknownError",
            str(last_exc) if last_exc else "Unknown"
        ))

    total_elapsed = round(time.perf_counter() - total_start, 3)
    succ = retry_log[-1]
    first_choice = response.choices[0] if response and response.choices else None
    usage = getattr(response, "usage", None)

    meta: dict = {
        "status": "success",
        "request": {
            "deployment": deployment,
            "response_format": {"type": "json_object"},
            "messages": messages_log,
            "images": {"paths": [str(p) for p in images], "count": len(images)},
        },
        "response": {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "created": getattr(response, "created", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "finish_reason": getattr(first_choice, "finish_reason", None),
            "usage": to_jsonable(usage),
        },
        "timing": {
            "timestamp_start": succ["timestamp_start"],
            "timestamp_end": succ["timestamp_end"],
            "response_time_seconds": succ["duration_seconds"],
            "total_elapsed_seconds": total_elapsed,
            "attempts_made": len(retry_log),
            "retry_log": retry_log,
        },
    }

    if save_full_response and response:
        meta["response"]["full_response"] = to_jsonable(response)

    raw = (first_choice.message.content or "").strip() if first_choice else ""
    meta["response"]["raw_content"] = raw

    if not raw:
        logger.warning("[%s] Empty model response", step_label)
        meta["parse"] = {"json_valid": False, "error": "empty_content"}
        return meta

    try:
        payload = json.loads(raw)
        meta["parse"] = {"json_valid": True, "error": None, "parsed_payload": to_jsonable(payload)}
    except json.JSONDecodeError:
        logger.warning("[%s] Non-JSON response", step_label)
        meta["parse"] = {"json_valid": False, "error": "json_decode_error"}

    return meta


def _build_fail_meta(
    deployment: str,
    images: list[Path],
    messages_log: list,
    retry_log: list,
    total_start: float,
    error_type: str,
    error_message: str,
) -> dict:
    ts_start = retry_log[0]["timestamp_start"] if retry_log else None
    ts_end = retry_log[-1]["timestamp_end"] if retry_log else None
    return {
        "status": "failed",
        "error_type": error_type,
        "error_message": error_message,
        "request": {
            "deployment": deployment,
            "messages": messages_log,
            "images": {"paths": [str(p) for p in images], "count": len(images)},
        },
        "response": {"id": None, "model": None, "usage": None},
        "timing": {
            "timestamp_start": ts_start,
            "timestamp_end": ts_end,
            "response_time_seconds": None,
            "total_elapsed_seconds": round(time.perf_counter() - total_start, 3),
            "attempts_made": len(retry_log),
            "retry_log": retry_log,
        },
        "parse": {"json_valid": False, "error": "model_call_failed"},
    }


# ---------------------------------------------------------------------------
# Token / timing aggregation helpers (for eval compatibility)
# ---------------------------------------------------------------------------

def _safe_int(val: Any) -> int:
    try:
        return int(val) if val is not None else 0
    except (TypeError, ValueError):
        return 0


def _safe_float(val: Any) -> float:
    try:
        return float(val) if val is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def aggregate_usage(*step_metas: Optional[dict]) -> Tuple[dict, float]:
    """Sum token counts and response_time_seconds from step metadata dicts."""
    pt = ct = tt = rt = 0
    rsp_time = 0.0

    for m in step_metas:
        if m is None:
            continue
        usage = (m.get("response") or {}).get("usage") or {}
        pt += _safe_int(usage.get("prompt_tokens"))
        ct += _safe_int(usage.get("completion_tokens"))
        tt += _safe_int(usage.get("total_tokens"))
        ct_details = usage.get("completion_tokens_details") or {}
        rt += _safe_int(ct_details.get("reasoning_tokens"))
        rsp_time += _safe_float((m.get("timing") or {}).get("response_time_seconds"))

    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
        "completion_tokens_details": {"reasoning_tokens": rt},
    }, rsp_time


# ---------------------------------------------------------------------------
# Pipeline for one pair
# ---------------------------------------------------------------------------

def process_pair(
    pair_name: str,
    original: Optional[Path],
    reimage: Optional[Path],
    client: AzureOpenAI,
    deployment: str,
    prompts: dict[str, str],
    logger: logging.Logger,
    save_full_response: bool,
    max_retries: int,
    reasoning_effort: Optional[str],
    run_settings: dict,
) -> Tuple[Optional[bool], str, dict]:
    """
    Run the full v4 pipeline for one pair.

    Returns (discard_image, changes, metadata_dict).
    """

    # ── Phase 0: PIL rule-based pre-check ──────────────────────────────────
    p0 = phase0_check(original, reimage, logger)
    image_count = p0["image_count"]

    if image_count == 0:
        changes = "No images found for this pair"
        final_meta = _assemble_meta(
            discard_image=False,
            changes=changes,
            branch="rule_no_images",
            phase0=p0,
            step1_meta=None,
            step2_meta=None,
            run_settings=run_settings,
            status="success",
        )
        logger.warning("[%s] Phase 0: no images found — skipping LLM", pair_name)
        return False, changes, final_meta

    if p0["corrupted"]:
        changes = "Could not infer"
        final_meta = _assemble_meta(
            discard_image=False,
            changes=changes,
            branch="rule_corrupted",
            phase0=p0,
            step1_meta=None,
            step2_meta=None,
            run_settings=run_settings,
            status="success",
        )
        logger.warning("[%s] Phase 0: PIL corruption detected — skipping LLM: %s", pair_name, p0["detail"])
        return False, changes, final_meta

    # Collect image list in order (original first, reimage second if available)
    img_list: list[Path] = []
    if original is not None and p0["original_exists"]:
        img_list.append(original)
    if reimage is not None and p0["reimage_exists"]:
        img_list.append(reimage)

    # ── Phase 1: LLM Step 1 (Triage) ───────────────────────────────────────
    if image_count == 1:
        prompt_step1 = prompts["1img"]
        logger.debug("[%s] Step 1: 1-image triage", pair_name)
    else:
        prompt_step1 = prompts["2img"]
        logger.debug("[%s] Step 1: 2-image triage", pair_name)

    step1_meta = call_model(
        client=client,
        deployment=deployment,
        images=img_list,
        prompt_text=prompt_step1,
        logger=logger,
        save_full_response=save_full_response,
        max_retries=max_retries,
        reasoning_effort=reasoning_effort,
        step_label="step1",
    )

    # Parse step1 output
    step1_payload: dict = {}
    if step1_meta.get("parse", {}).get("json_valid"):
        step1_payload = step1_meta["parse"]["parsed_payload"] or {}

    # Map step1 output to discard_image, changes, branch
    if image_count == 1:
        is_info = normalize_bool(step1_payload.get("is_informational"))
        if is_info is False:
            # A1 — non-informational single image
            discard_image = True
            changes = "Only one image which is non-informational and it contains logo (or watermark / blank block / single line / standalone section number / etc.)"
            branch = "A1"
        else:
            # A2 — informational single image (default if None too)
            discard_image = False
            changes = "NoMatchFound"
            branch = "A2"

        final_meta = _assemble_meta(
            discard_image=discard_image,
            changes=changes,
            branch=branch,
            phase0=p0,
            step1_meta=step1_meta,
            step2_meta=None,
            run_settings=run_settings,
            status="success",
        )
        logger.debug("[%s] Step 1 (1-img): branch=%s discard=%s", pair_name, branch, discard_image)
        return discard_image, changes, final_meta

    else:
        # 2-image case
        branch = step1_payload.get("branch", "")
        llm_changes = step1_payload.get("changes", "") or ""

        if not step1_meta.get("parse", {}).get("json_valid"):
            # JSON parse failure — treat as B1
            branch = "B1_parse_error"
            discard_image = False
            changes = "Could not infer"
            final_meta = _assemble_meta(
                discard_image=discard_image, changes=changes, branch=branch,
                phase0=p0, step1_meta=step1_meta, step2_meta=None,
                run_settings=run_settings, status="success",
            )
            return discard_image, changes, final_meta

        if branch == "B1":
            discard_image = False
            changes = "Could not infer"
            final_meta = _assemble_meta(
                discard_image=discard_image, changes=changes, branch=branch,
                phase0=p0, step1_meta=step1_meta, step2_meta=None,
                run_settings=run_settings, status="success",
            )
            return discard_image, changes, final_meta

        if branch == "B2":
            discard_image = False
            changes = llm_changes if llm_changes else "Image has been crossed out"
            final_meta = _assemble_meta(
                discard_image=discard_image, changes=changes, branch=branch,
                phase0=p0, step1_meta=step1_meta, step2_meta=None,
                run_settings=run_settings, status="success",
            )
            return discard_image, changes, final_meta

        if branch == "B3":
            discard_image = True
            changes = "Both images are non informational and contain only logos / etc.."
            final_meta = _assemble_meta(
                discard_image=discard_image, changes=changes, branch=branch,
                phase0=p0, step1_meta=step1_meta, step2_meta=None,
                run_settings=run_settings, status="success",
            )
            return discard_image, changes, final_meta

        if branch == "B4":
            discard_image = False
            changes = llm_changes if llm_changes else "One image is informational, the other is non-informational"
            final_meta = _assemble_meta(
                discard_image=discard_image, changes=changes, branch=branch,
                phase0=p0, step1_meta=step1_meta, step2_meta=None,
                run_settings=run_settings, status="success",
            )
            return discard_image, changes, final_meta

        if branch != "both_informational":
            # Unknown branch — log and treat as non-stop with diff
            logger.warning("[%s] Step 1 returned unknown branch '%s'; proceeding to diff anyway", pair_name, branch)

        # ── Phase 2: LLM Step 2 (Diff) ──────────────────────────────────
        logger.debug("[%s] Step 2: diff analysis", pair_name)
        step2_meta = call_model(
            client=client,
            deployment=deployment,
            images=img_list,
            prompt_text=prompts["diff"],
            logger=logger,
            save_full_response=save_full_response,
            max_retries=max_retries,
            reasoning_effort=reasoning_effort,
            step_label="step2",
        )

        step2_payload: dict = {}
        if step2_meta.get("parse", {}).get("json_valid"):
            step2_payload = step2_meta["parse"]["parsed_payload"] or {}

        discard_image = normalize_bool(step2_payload.get("discard_image"))
        if discard_image is None:
            discard_image = True  # safe default
        changes = step2_payload.get("changes", "") or ""
        if not isinstance(changes, str):
            changes = str(changes)

        final_branch = "both_informational_trivial" if discard_image else "both_informational_meaningful"
        final_meta = _assemble_meta(
            discard_image=discard_image, changes=changes, branch=final_branch,
            phase0=p0, step1_meta=step1_meta, step2_meta=step2_meta,
            run_settings=run_settings, status="success",
        )
        return discard_image, changes, final_meta


def _assemble_meta(
    *,
    discard_image: Optional[bool],
    changes: str,
    branch: str,
    phase0: dict,
    step1_meta: Optional[dict],
    step2_meta: Optional[dict],
    run_settings: dict,
    status: str,
) -> dict:
    """Build the final metadata blob stored in {prefix}_metadata."""
    usage_dict, total_rsp_time = aggregate_usage(step1_meta, step2_meta)

    return {
        "status": status,
        "branch": branch,
        "final_output": {
            "discard_image": discard_image,
            "changes": changes,
        },
        # Aggregated usage for eval_human_runs.py compatibility
        "response": {
            "usage": usage_dict,
        },
        "timing": {
            "response_time_seconds": round(total_rsp_time, 3),
        },
        # Per-step detail
        "phase0": phase0,
        "step1": step1_meta,
        "step2": step2_meta,
        "run_settings": run_settings,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_run_settings(args: argparse.Namespace, deployment: str) -> dict:
    return {
        "deployment": deployment,
        "model_name_arg": args.model_name,
        "key_prefix": args.key_prefix,
        "endpoint": args.endpoint,
        "api_version": args.api_version,
        "overwrite": args.overwrite,
        "save_full_response": args.save_full_response,
        "max_retries": args.max_retries,
        "reasoning_effort": args.reasoning_effort,
        "only_doubt": args.only_doubt,
        "start": args.start,
        "end": args.end,
        "limit": args.limit,
        "sleep_seconds": args.sleep_seconds,
        "dry_run": args.dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 multi-step DCA pipeline.")
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--image-pairs-dir", type=Path, default=DEFAULT_IMAGE_PAIRS_DIR)
    parser.add_argument("--deployment", default=os.getenv("MODEL_NAME", "gpt-4.1-mini"))
    parser.add_argument("--model-name", default=None, help="Alias for deployment; overrides --deployment if given")
    parser.add_argument("--key-prefix", required=True,
                        help="Prefix for output keys, e.g. v4_prompt_cr_gpt41. "
                             "Keys written: {prefix}_discard_image, {prefix}_changes, {prefix}_metadata")
    parser.add_argument("--endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT"))
    parser.add_argument("--api-version", default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"))
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-doubt", action="store_true", help="Process only pairs where 'doubt' key is true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-full-response", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reasoning-effort", default=None, choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")

    logger = setup_logger(args.verbose)
    deployment = args.model_name or args.deployment

    key_di = f"{args.key_prefix}_discard_image"
    key_ch = f"{args.key_prefix}_changes"
    key_md = f"{args.key_prefix}_metadata"

    logger.info("Deployment : %s", deployment)
    logger.info("Key prefix : %s", args.key_prefix)
    logger.info("Keys       : %s | %s | %s", key_di, key_ch, key_md)
    logger.info("PIL available: %s", _PIL_AVAILABLE)

    if not args.json_dir.exists():
        raise FileNotFoundError(f"JSON dir not found: {args.json_dir}")
    if not args.image_pairs_dir.exists():
        raise FileNotFoundError(f"Image pairs dir not found: {args.image_pairs_dir}")

    # Load all prompts upfront
    prompts = {
        "1img": load_prompt(PROMPT_1IMG),
        "2img": load_prompt(PROMPT_2IMG),
        "diff": load_prompt(PROMPT_DIFF),
    }
    logger.info("Loaded prompts: %s, %s, %s", PROMPT_1IMG, PROMPT_2IMG, PROMPT_DIFF)

    client = build_client(args.endpoint, args.api_version, logger)

    json_files = sorted(args.json_dir.glob("pair_*.json"))
    if not json_files:
        logger.info("No pair_*.json files in %s", args.json_dir)
        return

    processed = updated = skipped_existing = skipped_no_images = skipped_not_doubt = failed = 0

    for json_path in json_files:
        pair_idx = parse_pair_index(json_path)
        if pair_idx is None:
            continue
        if args.start is not None and pair_idx < args.start:
            continue
        if args.end is not None and pair_idx > args.end:
            continue
        if args.limit is not None and processed >= args.limit:
            break

        processed += 1
        pair_name = f"pair_{pair_idx:03d}"

        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                logger.warning("[%s] JSON root is not an object; skipping", pair_name)
                failed += 1
                continue

            if args.only_doubt and payload.get("doubt") is not True:
                skipped_not_doubt += 1
                logger.debug("[%s] doubt not true; skipping", pair_name)
                continue

            all_keys = key_di in payload and key_ch in payload and key_md in payload
            if not args.overwrite and all_keys:
                skipped_existing += 1
                logger.debug("[%s] Keys already present; skipping", pair_name)
                continue

            pair_dir = args.image_pairs_dir / pair_name
            original = find_image(pair_dir, "original")
            reimage = find_image(pair_dir, "reimage")

            if original is None and reimage is None:
                skipped_no_images += 1
                logger.warning("[%s] No image files found in %s", pair_name, pair_dir)
                continue

            run_settings = build_run_settings(args, deployment)

            discard_image, changes, meta = process_pair(
                pair_name=pair_name,
                original=original,
                reimage=reimage,
                client=client,
                deployment=deployment,
                prompts=prompts,
                logger=logger,
                save_full_response=args.save_full_response,
                max_retries=args.max_retries,
                reasoning_effort=args.reasoning_effort,
                run_settings=run_settings,
            )

            if args.overwrite or key_di not in payload:
                payload[key_di] = discard_image
            if args.overwrite or key_ch not in payload:
                payload[key_ch] = changes
            if args.overwrite or key_md not in payload:
                payload[key_md] = compact_json(meta)

            if args.dry_run:
                logger.info("[%s] DRY RUN: would write keys for prefix '%s' branch=%s discard=%s",
                            pair_name, args.key_prefix, meta.get("branch"), discard_image)
            else:
                json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                logger.info("[%s] Saved keys for prefix '%s' branch=%s discard=%s",
                            pair_name, args.key_prefix, meta.get("branch"), discard_image)
            updated += 1

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

        except ModelCallFailure as exc:
            failed += 1
            meta = exc.metadata
            meta["run_settings"] = build_run_settings(args, deployment)
            meta["status"] = "failed"
            if not args.dry_run:
                payload[key_md] = compact_json(meta)
                json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            logger.error("[%s] ModelCallFailure: %s", pair_name, exc)

        except Exception as exc:
            failed += 1
            logger.error("[%s] Unexpected error: %s", pair_name, exc)

    logger.info("Done. processed=%d updated=%d skipped_existing=%d skipped_no_images=%d "
                "skipped_not_doubt=%d failed=%d",
                processed, updated, skipped_existing, skipped_no_images, skipped_not_doubt, failed)


if __name__ == "__main__":
    main()
