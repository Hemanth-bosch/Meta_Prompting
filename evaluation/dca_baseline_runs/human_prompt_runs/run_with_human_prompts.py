"""
Append prompt outputs to existing DCA pair JSON files using one generic script.

Examples
--------
1) GPT-4.1 style keys:
   python evaluation/dca_baseline_runs/human_prompt_runs/run_with_human_prompts.py \
       --deployment gpt-4.1 \
       --key-prefix v3_prompt_cr_gpt41

2) GPT-5.2 style keys:
   python evaluation/dca_baseline_runs/human_prompt_runs/run_with_human_prompts.py \
       --deployment gpt-5.2 \
       --key-prefix v3_prompt_cr_gpt52

3) GPT-4.1-mini style keys:
   python evaluation/dca_baseline_runs/human_prompt_runs/run_with_human_prompts.py \
       --deployment gpt-4.1-mini \
       --key-prefix v3_prompt_cr_gpt41_mini
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
import dotenv
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI, DefaultHttpxClient
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

dotenv.load_dotenv(dotenv.find_dotenv(), override=True)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DCA_DIR = REPO_ROOT / "data" / "dca_data"
DEFAULT_JSON_DIR = DCA_DIR / "image_pairs_llm_output_exp1"
DEFAULT_IMAGE_PAIRS_DIR = DCA_DIR / "image_pairs"
HUMAN_PROMPTS_DIR = REPO_ROOT / "evaluation" / "dca_baseline_runs" / "human_prompts"


def load_prompt(prompt_name: str) -> str:
    """Load prompt from markdown file in human_prompts directory.

    Args:
        prompt_name: Name of the prompt markdown file (e.g., 'v3_prompt_cr.md')

    Returns:
        Content of the prompt file as a string.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    prompt_path = HUMAN_PROMPTS_DIR / prompt_name
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("append_v3_prompt_generic")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
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
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }
    media_type = ext_to_mime.get(image_path.suffix.lower(), "image/png")
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{media_type};base64,{b64}"


def parse_pair_index(json_path: Path) -> Optional[int]:
    match = re.match(r"^pair_(\d+)\.json$", json_path.name)
    if not match:
        return None
    return int(match.group(1))


def find_image(pair_dir: Path, stem: str) -> Optional[Path]:
    allowed = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
    matches = [path for path in sorted(pair_dir.glob(f"{stem}.*")) if path.suffix.lower() in allowed]
    return matches[0] if matches else None


def normalize_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "dict"):
        try:
            return to_jsonable(value.dict())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "__dict__"):
        try:
            return to_jsonable(vars(value))
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def metadata_to_compact_string(metadata: dict[str, Any]) -> str:
    return json.dumps(to_jsonable(metadata), ensure_ascii=False, separators=(",", ":"))


class ModelCallFailure(Exception):
    def __init__(self, message: str, metadata: dict[str, Any]):
        super().__init__(message)
        self.metadata = metadata


def build_run_settings(args: argparse.Namespace, deployment: str) -> dict[str, Any]:
    return {
        "deployment": deployment,
        "model_name_arg": args.model_name,
        "key_prefix": args.key_prefix,
        "prompt": args.prompt,
        "endpoint": args.endpoint,
        "api_version": args.api_version,
        "overwrite": args.overwrite,
        "save_full_response": args.save_full_response,
        "max_retries": args.max_retries,
        "no_thinking_area": args.no_thinking_area,
        "reasoning_effort": args.reasoning_effort,
        "only_doubt": args.only_doubt,
        "start": args.start,
        "end": args.end,
        "limit": args.limit,
        "sleep_seconds": args.sleep_seconds,
        "dry_run": args.dry_run,
    }


def build_loggable_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    def _strip_base64(content: object) -> object:
        if isinstance(content, list):
            result = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    stripped_url = url[:40] + "...<base64_omitted>" if url.startswith("data:") else url
                    result.append({"type": "image_url", "image_url": {"url": stripped_url}})
                else:
                    result.append(item)
            return result
        return content

    return [
        {"role": message["role"], "content": _strip_base64(message["content"])}
        for message in messages
    ]


def build_failure_metadata(
    deployment: str,
    original_image: Optional[Path],
    reimage: Optional[Path],
    messages_loggable: list[dict[str, object]],
    retry_log: list[dict[str, Any]],
    total_start: float,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    timestamp_start = retry_log[0]["timestamp_start"] if retry_log else None
    timestamp_end = retry_log[-1]["timestamp_end"] if retry_log else None
    return {
        "status": "failed",
        "error_type": error_type,
        "error_message": error_message,
        "request": {
            "deployment": deployment,
            "response_format": {"type": "json_object"},
            "messages": messages_loggable,
            "images": {
                "original": str(original_image) if original_image else None,
                "reimage": str(reimage) if reimage else None,
                "count": int(original_image is not None) + int(reimage is not None),
            },
        },
        "response": {
            "id": None,
            "model": None,
            "created": None,
            "system_fingerprint": None,
            "finish_reason": None,
            "usage": None,
        },
        "timing": {
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "response_time_seconds": None,
            "total_elapsed_seconds": round(time.perf_counter() - total_start, 3),
            "attempts_made": len(retry_log),
            "retry_log": retry_log,
        },
        "parse": {
            "json_valid": False,
            "error": "model_call_failed",
        },
    }


def call_model(
    client: AzureOpenAI,
    deployment: str,
    original_image: Optional[Path],
    reimage: Optional[Path],
    prompt_text: str,
    logger: logging.Logger,
    save_full_response: bool,
    max_retries: int = 3,
    retry_backoff_base: float = 2.0,
    use_thinking_area: bool = True,
    reasoning_effort: Optional[str] = None,
) -> Tuple[Optional[bool], str, Optional[str], dict[str, Any]]:
    user_content: list[dict[str, object]] = []

    if original_image is not None:
        user_content.append({"type": "image_url", "image_url": {"url": image_to_data_url(original_image)}})
    if reimage is not None:
        user_content.append({"type": "image_url", "image_url": {"url": image_to_data_url(reimage)}})
    user_content.append({"type": "text", "text": prompt_text})

    messages = [
        {"role": "system", "content": "You are an expert assistant for comparing document image pairs."},
        {"role": "user", "content": user_content},
    ]
    messages_loggable = build_loggable_messages(messages)

    total_start = time.perf_counter()

    retry_log: list[dict[str, Any]] = []
    last_exc: Optional[Exception] = None
    response = None

    # Waits: rate limit uses Retry-After header or 30s default (Azure 429s need real headroom).
    # Connection/timeout/5xx use exponential backoff from retry_backoff_base (default 2s).
    _RATELIMIT_DEFAULT_WAIT = 30.0

    def _retry_after(exc: RateLimitError) -> float:
        """Read Retry-After header from the 429 response, fall back to default."""
        try:
            val = exc.response.headers.get("Retry-After")
            if val is not None:
                return float(val)
        except Exception:  # noqa: BLE001
            pass
        return _RATELIMIT_DEFAULT_WAIT

    for attempt in range(1, max_retries + 2):  # attempts 1 .. max_retries+1
        ts_attempt_start = datetime.now(timezone.utc).isoformat()
        t_attempt_start = time.perf_counter()
        try:
            extra_kwargs: dict[str, Any] = {}
            if reasoning_effort is not None:
                extra_kwargs["reasoning_effort"] = reasoning_effort
            response = client.chat.completions.create(
                model=deployment,
                messages=messages,
                response_format={"type": "json_object"},
                **extra_kwargs,
            )
            t_attempt_end = time.perf_counter()
            ts_attempt_end = datetime.now(timezone.utc).isoformat()
            attempt_duration = round(t_attempt_end - t_attempt_start, 3)
            last_exc = None
            # Record the successful attempt in retry_log for full traceability
            retry_log.append({
                "attempt": attempt,
                "status": "success",
                "timestamp_start": ts_attempt_start,
                "timestamp_end": ts_attempt_end,
                "duration_seconds": attempt_duration,
            })
            break  # success — exit retry loop
        except RateLimitError as exc:
            last_exc = exc
            t_attempt_end = time.perf_counter()
            ts_attempt_end = datetime.now(timezone.utc).isoformat()
            wait = _retry_after(exc)
            retry_log.append({
                "attempt": attempt,
                "status": "failed",
                "error": "RateLimitError",
                "timestamp_start": ts_attempt_start,
                "timestamp_end": ts_attempt_end,
                "duration_seconds": round(t_attempt_end - t_attempt_start, 3),
                "wait_seconds": wait,
                "detail": str(exc),
            })
            logger.warning("[attempt %d/%d] RateLimitError — waiting %.1fs (Retry-After) before retry", attempt, max_retries + 1, wait)
        except APITimeoutError as exc:
            last_exc = exc
            t_attempt_end = time.perf_counter()
            ts_attempt_end = datetime.now(timezone.utc).isoformat()
            wait = retry_backoff_base ** attempt
            retry_log.append({
                "attempt": attempt,
                "status": "failed",
                "error": "APITimeoutError",
                "timestamp_start": ts_attempt_start,
                "timestamp_end": ts_attempt_end,
                "duration_seconds": round(t_attempt_end - t_attempt_start, 3),
                "wait_seconds": wait,
                "detail": str(exc),
            })
            logger.warning("[attempt %d/%d] APITimeoutError — waiting %.1fs before retry", attempt, max_retries + 1, wait)
        except APIConnectionError as exc:
            last_exc = exc
            t_attempt_end = time.perf_counter()
            ts_attempt_end = datetime.now(timezone.utc).isoformat()
            wait = retry_backoff_base ** attempt
            retry_log.append({
                "attempt": attempt,
                "status": "failed",
                "error": "APIConnectionError",
                "timestamp_start": ts_attempt_start,
                "timestamp_end": ts_attempt_end,
                "duration_seconds": round(t_attempt_end - t_attempt_start, 3),
                "wait_seconds": wait,
                "detail": str(exc),
            })
            logger.warning("[attempt %d/%d] APIConnectionError — waiting %.1fs before retry", attempt, max_retries + 1, wait)
        except APIStatusError as exc:
            last_exc = exc
            t_attempt_end = time.perf_counter()
            ts_attempt_end = datetime.now(timezone.utc).isoformat()
            if exc.status_code in (500, 502, 503, 504):
                wait = retry_backoff_base ** attempt
                retry_log.append({
                    "attempt": attempt,
                    "status": "failed",
                    "error": f"APIStatusError_{exc.status_code}",
                    "timestamp_start": ts_attempt_start,
                    "timestamp_end": ts_attempt_end,
                    "duration_seconds": round(t_attempt_end - t_attempt_start, 3),
                    "wait_seconds": wait,
                    "detail": str(exc),
                })
                logger.warning("[attempt %d/%d] APIStatusError %d — waiting %.1fs before retry", attempt, max_retries + 1, exc.status_code, wait)
            else:
                retry_log.append({
                    "attempt": attempt,
                    "status": "failed",
                    "error": f"APIStatusError_{exc.status_code}",
                    "timestamp_start": ts_attempt_start,
                    "timestamp_end": ts_attempt_end,
                    "duration_seconds": round(t_attempt_end - t_attempt_start, 3),
                    "wait_seconds": 0,
                    "detail": str(exc),
                })
                logger.error("[attempt %d] APIStatusError %d — not retryable, raising", attempt, exc.status_code)
                failure_metadata = build_failure_metadata(
                    deployment=deployment,
                    original_image=original_image,
                    reimage=reimage,
                    messages_loggable=messages_loggable,
                    retry_log=retry_log,
                    total_start=total_start,
                    error_type=f"APIStatusError_{exc.status_code}",
                    error_message=str(exc),
                )
                raise ModelCallFailure(str(exc), failure_metadata) from exc
        if attempt <= max_retries and last_exc is not None:
            time.sleep(wait)
    else:
        # for-loop completed without break → all retries exhausted
        logger.error("All %d retries exhausted. Last error: %s", max_retries, last_exc)
        error_type = type(last_exc).__name__ if last_exc is not None else "UnknownError"
        error_message = str(last_exc) if last_exc is not None else "Model call failed without exception details"
        failure_metadata = build_failure_metadata(
            deployment=deployment,
            original_image=original_image,
            reimage=reimage,
            messages_loggable=messages_loggable,
            retry_log=retry_log,
            total_start=total_start,
            error_type=error_type,
            error_message=error_message,
        )
        raise ModelCallFailure(error_message, failure_metadata)

    total_elapsed = round(time.perf_counter() - total_start, 3)
    # Timing for the successful attempt specifically
    successful = retry_log[-1]
    ts_start = successful["timestamp_start"]
    ts_end = successful["timestamp_end"]
    response_time_seconds = successful["duration_seconds"]

    first_choice = response.choices[0] if response.choices else None
    finish_reason = getattr(first_choice, "finish_reason", None)
    usage = getattr(response, "usage", None)

    metadata: dict[str, Any] = {
        "request": {
            "deployment": deployment,
            "response_format": {"type": "json_object"},
            "messages": messages_loggable,
            "images": {
                "original": str(original_image) if original_image else None,
                "reimage": str(reimage) if reimage else None,
                "count": int(original_image is not None) + int(reimage is not None),
            },
        },
        "response": {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "created": getattr(response, "created", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "finish_reason": finish_reason,
            "usage": to_jsonable(usage),
        },
        "timing": {
            "timestamp_start": ts_start,
            "timestamp_end": ts_end,
            "response_time_seconds": response_time_seconds,
            "total_elapsed_seconds": total_elapsed,
            "attempts_made": len(retry_log),
            "retry_log": retry_log,
        },
    }
    if save_full_response:
        metadata["response"]["full_response"] = to_jsonable(response)

    raw = (first_choice.message.content or "").strip() if first_choice else ""
    metadata["response"]["raw_content"] = raw
    if not raw:
        logger.warning("Model returned empty content")
        metadata["parse"] = {
            "json_valid": False,
            "error": "empty_content",
        }
        return None, "", "", metadata

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Model returned non-JSON response; storing raw content in summary/thinking_area")
        metadata["parse"] = {
            "json_valid": False,
            "error": "json_decode_error",
        }
        return None, raw, raw, metadata

    cls = normalize_bool(payload.get("discard_image"))
    summary = payload.get("changes", "")
    if not isinstance(summary, str):
        summary = str(summary)

    if use_thinking_area:
        thinking_area: Optional[str] = payload.get("thinking_area", "")
        if not isinstance(thinking_area, str):
            thinking_area = str(thinking_area)
    else:
        thinking_area = None

    metadata["parse"] = {
        "json_valid": True,
        "error": None,
        "parsed_payload": to_jsonable(payload),
    }
    metadata["normalized_output"] = {
        "discard_image": cls,
        "changes": summary,
        **(({"thinking_area": thinking_area}) if use_thinking_area else {}),
    }

    return cls, summary, thinking_area, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Append configurable prompt outputs into existing pair JSON files.")
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--image-pairs-dir", type=Path, default=DEFAULT_IMAGE_PAIRS_DIR)
    parser.add_argument("--deployment", default=os.getenv("MODEL_NAME", "gpt-4.1-mini"))
    parser.add_argument("--model-name", default=None, help="Alias for deployment; overrides --deployment if provided")
    parser.add_argument("--key-prefix", required=True, help="Example: v3_prompt_cr_gpt41, v3_prompt_cr_gpt52, v3_prompt_cr_gpt41_mini")
    parser.add_argument("--prompt", default="v3_prompt_cr.md", help="Prompt markdown file name in human_prompts directory")
    parser.add_argument("--endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT", "https://aiq212-openai.openai.azure.com/"))
    parser.add_argument("--api-version", default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"))
    parser.add_argument("--start", type=int, default=None, help="Start pair index, e.g., 1 for pair_001")
    parser.add_argument("--end", type=int, default=None, help="End pair index, e.g., 50 for pair_050")
    parser.add_argument("--limit", type=int, default=None, help="Max number of JSON files to process")
    parser.add_argument("--only-doubt", action="store_true", help="Process only JSON files where key 'doubt' is true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite target keys if present")
    parser.add_argument(
        "--save-full-response",
        action="store_true",
        help="Store full raw ChatCompletion object in metadata (can be large)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write files")
    parser.add_argument("--no-thinking-area", action="store_true", help="Prompt does not produce a thinking_area field; skip writing that key")
    parser.add_argument("--reasoning-effort", default=None, choices=["minimal", "low", "medium", "high"], help="Azure reasoning effort for models that support it (e.g. gpt-5.2, gpt-5-mini). Leave unset for gpt-4.1 / gpt-4.1-mini.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries on transient API errors (rate limit, timeout, 5xx)")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Sleep between API calls")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")

    logger = setup_logger(args.verbose)

    deployment = args.model_name or args.deployment
    key_classification = f"{args.key_prefix}_discard_image"
    key_change_summary = f"{args.key_prefix}_changes"
    key_thinking_area = f"{args.key_prefix}_thinking_area"
    key_metadata = f"{args.key_prefix}_metadata"

    logger.info("Using deployment: %s", deployment)
    logger.info("Using key prefix: %s", args.key_prefix)
    logger.info("Target keys: %s, %s, %s, %s", key_classification, key_change_summary, key_thinking_area, key_metadata)

    if not args.json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {args.json_dir}")
    if not args.image_pairs_dir.exists():
        raise FileNotFoundError(f"Image pairs directory not found: {args.image_pairs_dir}")

    # Load prompt on demand
    try:
        prompt_text = load_prompt(args.prompt)
        logger.info("Loaded prompt from: %s", args.prompt)
    except FileNotFoundError as e:
        logger.error("Failed to load prompt: %s", e)
        raise

    client = build_client(args.endpoint, args.api_version, logger)

    json_files = sorted(args.json_dir.glob("pair_*.json"))
    if not json_files:
        logger.info("No pair_*.json files found in %s", args.json_dir)
        return

    processed = 0
    updated = 0
    skipped_existing = 0
    skipped_missing_images = 0
    skipped_not_doubt = 0
    failed = 0

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
                logger.debug("[%s] 'doubt' is not true; skipping", pair_name)
                continue

            has_cls = key_classification in payload
            has_sum = key_change_summary in payload
            has_thinking = key_thinking_area in payload
            has_metadata = key_metadata in payload
            metadata_is_compact_string = has_metadata and isinstance(payload.get(key_metadata), str)

            # When --no-thinking-area is set, the thinking key is not written and
            # must not be required for the "already complete" skip check.
            all_keys_present = (
                has_cls and has_sum and has_metadata
                and (has_thinking or args.no_thinking_area)
            )

            if (
                not args.overwrite
                and all_keys_present
                and not metadata_is_compact_string
            ):
                try:
                    if isinstance(payload.get(key_metadata), dict):
                        payload[key_metadata] = metadata_to_compact_string(payload[key_metadata])
                    else:
                        payload[key_metadata] = json.dumps(payload.get(key_metadata), ensure_ascii=False, separators=(",", ":"))

                    if args.dry_run:
                        logger.info("[%s] DRY RUN: would normalize metadata to compact string", pair_name)
                    else:
                        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                        logger.info("[%s] Normalized existing metadata to compact string", pair_name)
                    updated += 1
                    continue
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    logger.error("[%s] Failed metadata normalization: %s", pair_name, exc)
                    continue

            if not args.overwrite and all_keys_present:
                skipped_existing += 1
                logger.debug("[%s] Target keys already present; skipping", pair_name)
                continue

            pair_dir = args.image_pairs_dir / pair_name
            original = find_image(pair_dir, "original")
            reimage = find_image(pair_dir, "reimage")
            if original is None and reimage is None:
                skipped_missing_images += 1
                logger.warning("[%s] No original.* or reimage.* found in %s", pair_name, pair_dir)
                continue

            cls, summary, thinking_area, metadata = call_model(
                client=client,
                deployment=deployment,
                original_image=original,
                reimage=reimage,
                prompt_text=prompt_text,
                logger=logger,
                save_full_response=args.save_full_response,
                max_retries=args.max_retries,
                use_thinking_area=not args.no_thinking_area,
                reasoning_effort=args.reasoning_effort,
            )
            metadata["status"] = "success"
            metadata["run_settings"] = build_run_settings(args, deployment)

            changed = False
            if args.overwrite or key_classification not in payload:
                payload[key_classification] = cls
                changed = True
            if args.overwrite or key_change_summary not in payload:
                payload[key_change_summary] = summary
                changed = True
            if not args.no_thinking_area:
                if args.overwrite or key_thinking_area not in payload:
                    payload[key_thinking_area] = thinking_area
                    changed = True
            if args.overwrite or key_metadata not in payload:
                payload[key_metadata] = metadata_to_compact_string(metadata)
                changed = True

            if changed:
                if args.dry_run:
                    logger.info("[%s] DRY RUN: would append keys for prefix '%s'", pair_name, args.key_prefix)
                else:
                    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    logger.info("[%s] Appended keys for prefix '%s'", pair_name, args.key_prefix)
                updated += 1

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

        except ModelCallFailure as exc:
            failed += 1
            metadata = exc.metadata
            metadata["run_settings"] = build_run_settings(args, deployment)

            if args.dry_run:
                logger.error("[%s] DRY RUN: would persist failure metadata for prefix '%s': %s", pair_name, args.key_prefix, exc)
            else:
                payload[key_metadata] = metadata_to_compact_string(metadata)
                json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                logger.error("[%s] Persisted failure metadata for prefix '%s': %s", pair_name, args.key_prefix, exc)
                updated += 1
            continue

        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.error("[%s] Failed: %s", pair_name, exc)

    logger.info("Done")
    logger.info("Processed: %d", processed)
    logger.info("Updated: %d", updated)
    logger.info("Skipped (already had keys): %d", skipped_existing)
    logger.info("Skipped (missing images): %d", skipped_missing_images)
    logger.info("Skipped (doubt not true): %d", skipped_not_doubt)
    logger.info("Failed: %d", failed)


if __name__ == "__main__":
    main()
