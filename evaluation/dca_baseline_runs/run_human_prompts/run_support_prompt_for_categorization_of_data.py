"""
Classify existing human labels into category codes using an LLM prompt.

Reads from each pair_XXX.json:
  - human_label_discard_image
  - human_label_changes

Writes back:
  - human_label_category  (one of A1, A2, B1, B2, B3, B4, B5a, B5b)

Example
-------
python evaluation/dca_baseline_runs/run_human_prompts/run_support_prompt_for_categorization_of_data.py \
    --deployment gpt-4.1-mini \
    --prompt support_prompt_for_categorization_of_data.md \
    --start 1 --end 50 --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import certifi
import dotenv
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI, DefaultHttpxClient
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError


dotenv.load_dotenv(dotenv.find_dotenv(), override=True)

SCRIPT_DIR = Path(__file__).resolve().parent


def _find_repo_root(start_dir: Path) -> Path:
    for candidate in [start_dir, *start_dir.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start_dir.parents[3]


REPO_ROOT = _find_repo_root(SCRIPT_DIR)
DCA_DIR = REPO_ROOT / "data" / "dca_data"
DEFAULT_JSON_DIR = DCA_DIR / "image_pairs_llm_output_exp1"
HUMAN_PROMPTS_DIR = REPO_ROOT / "evaluation" / "dca_baseline_runs" / "human_prompts"
DEFAULT_PROMPT_FILE = "support_prompt_for_categorization_of_data.md"

SOURCE_KEY_DISCARD = "human_label_discard_image"
SOURCE_KEY_CHANGES = "human_label_changes"
TARGET_KEY_CATEGORY = "human_label_category"

CANONICAL_CATEGORY_CODES = {"A1", "A2", "B1", "B2", "B3", "B4", "B5a", "B5b"}


class ModelCallFailure(Exception):
    pass


def load_prompt(prompt_name: str) -> str:
    prompt_path = HUMAN_PROMPTS_DIR / prompt_name
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("run_support_prompt_for_categorization_of_data")
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


def parse_pair_index(json_path: Path) -> Optional[int]:
    match = re.match(r"^pair_(\d+)\.json$", json_path.name)
    if not match:
        return None
    return int(match.group(1))


def normalize_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    if isinstance(value, int):
        return bool(value)
    return None


def normalize_category_code(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw:
        return None

    match = re.match(r"^(A1|A2|B1|B2|B3|B4|B5a|B5b)\b", raw, flags=re.IGNORECASE)
    if match:
        code = match.group(1)
        if code.upper() in {"A1", "A2", "B1", "B2", "B3", "B4"}:
            return code.upper()
        return "B5a" if code.lower() == "b5a" else "B5b"

    lowered = raw.lower()
    compact = re.sub(r"[^a-z0-9]", "", lowered)

    if "couldnotinfer" in compact or "corrupt" in lowered or "unreadable" in lowered or "tootiny" in compact:
        return "B1"
    if "crossedout" in compact or "crossmark" in compact or "strikethrough" in compact or "redaction" in lowered:
        return "B2"
    if "single" in lowered and "non" in lowered and "inform" in lowered:
        return "A1"
    if "nomatchfound" in compact:
        return "A2"
    if "single" in lowered and "informational" in lowered:
        return "A2"
    if "both" in lowered and "non" in lowered and "inform" in lowered:
        return "B3"
    if "one" in lowered and "informational" in lowered and "non" in lowered:
        return "B4"
    if "identical" in lowered or "no changes" in lowered or "trivial" in lowered:
        return "B5a"
    if "meaningful" in lowered or "differences explained" in lowered:
        return "B5b"

    return None


def call_model(
    client: AzureOpenAI,
    deployment: str,
    source_discard: Any,
    source_changes: str,
    prompt_text: str,
    logger: logging.Logger,
    max_retries: int = 3,
    retry_backoff_base: float = 2.0,
    reasoning_effort: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    input_payload = {
        SOURCE_KEY_DISCARD: source_discard,
        SOURCE_KEY_CHANGES: source_changes,
    }

    user_text = (
        f"{prompt_text}\n\n"
        f"Input JSON:\n{json.dumps(input_payload, ensure_ascii=False)}"
    )

    messages = [
        {"role": "system", "content": "You are an expert assistant for categorizing document-comparison labels."},
        {"role": "user", "content": user_text},
    ]

    last_exc: Optional[Exception] = None
    response = None

    _RATELIMIT_DEFAULT_WAIT = 30.0

    def _retry_after(exc: RateLimitError) -> float:
        try:
            val = exc.response.headers.get("Retry-After")
            if val is not None:
                return float(val)
        except Exception:  # noqa: BLE001
            pass
        return _RATELIMIT_DEFAULT_WAIT

    for attempt in range(1, max_retries + 2):
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
            last_exc = None
            break
        except RateLimitError as exc:
            last_exc = exc
            wait = _retry_after(exc)
            logger.warning("[attempt %d/%d] RateLimitError — waiting %.1fs before retry", attempt, max_retries + 1, wait)
        except APITimeoutError as exc:
            last_exc = exc
            wait = retry_backoff_base ** attempt
            logger.warning("[attempt %d/%d] APITimeoutError — waiting %.1fs before retry", attempt, max_retries + 1, wait)
        except APIConnectionError as exc:
            last_exc = exc
            wait = retry_backoff_base ** attempt
            logger.warning("[attempt %d/%d] APIConnectionError — waiting %.1fs before retry", attempt, max_retries + 1, wait)
        except APIStatusError as exc:
            last_exc = exc
            if exc.status_code in (500, 502, 503, 504):
                wait = retry_backoff_base ** attempt
                logger.warning(
                    "[attempt %d/%d] APIStatusError %d — waiting %.1fs before retry",
                    attempt,
                    max_retries + 1,
                    exc.status_code,
                    wait,
                )
            else:
                raise ModelCallFailure(str(exc)) from exc

        if attempt <= max_retries and last_exc is not None:
            time.sleep(wait)
    else:
        error_message = str(last_exc) if last_exc is not None else "Model call failed without exception details"
        raise ModelCallFailure(error_message)

    first_choice = response.choices[0] if response.choices else None
    raw = (first_choice.message.content or "").strip() if first_choice else ""

    if not raw:
        return None, ""

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None, raw

    raw_category = payload.get(TARGET_KEY_CATEGORY)
    normalized_category = normalize_category_code(raw_category)
    return normalized_category, raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign human_label_category codes using an LLM text prompt.")
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--deployment", default=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1"))
    parser.add_argument("--model-name", default=None, help="Alias for deployment; overrides --deployment if provided")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT_FILE, help="Prompt markdown file in human_prompts directory")
    parser.add_argument("--endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT", "https://aiq212-openai.openai.azure.com/"))
    parser.add_argument("--api-version", default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"))
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-doubt", action="store_true", help="Process only JSON files where key 'doubt' is true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite human_label_category if present")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files")
    parser.add_argument("--reasoning-effort", default=None, choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")

    logger = setup_logger(args.verbose)
    deployment = args.model_name or args.deployment

    logger.info("Using deployment: %s", deployment)
    logger.info("Target key: %s", TARGET_KEY_CATEGORY)

    if not args.json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {args.json_dir}")

    prompt_text = load_prompt(args.prompt)
    logger.info("Loaded prompt: %s", args.prompt)

    client = build_client(args.endpoint, args.api_version, logger)

    json_files = sorted(args.json_dir.glob("pair_*.json"))
    if not json_files:
        logger.info("No pair_*.json files found in %s", args.json_dir)
        return

    processed = 0
    updated = 0
    skipped_existing = 0
    skipped_missing_source = 0
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
                continue

            existing_category = payload.get(TARGET_KEY_CATEGORY)
            if (
                not args.overwrite
                and isinstance(existing_category, str)
                and normalize_category_code(existing_category) in CANONICAL_CATEGORY_CODES
            ):
                skipped_existing += 1
                continue

            if SOURCE_KEY_DISCARD not in payload or SOURCE_KEY_CHANGES not in payload:
                skipped_missing_source += 1
                logger.debug("[%s] Missing source keys; skipping", pair_name)
                continue

            source_discard_raw = payload.get(SOURCE_KEY_DISCARD)
            source_discard_norm = normalize_bool(source_discard_raw)
            source_discard_for_model: Any = source_discard_norm if source_discard_norm is not None else source_discard_raw

            source_changes_raw = payload.get(SOURCE_KEY_CHANGES)
            source_changes = source_changes_raw if isinstance(source_changes_raw, str) else ("" if source_changes_raw is None else str(source_changes_raw))

            category_code, raw_output = call_model(
                client=client,
                deployment=deployment,
                source_discard=source_discard_for_model,
                source_changes=source_changes,
                prompt_text=prompt_text,
                logger=logger,
                max_retries=args.max_retries,
                reasoning_effort=args.reasoning_effort,
            )

            if category_code is None:
                failed += 1
                logger.error(
                    "[%s] Invalid model output for category. raw=%s",
                    pair_name,
                    raw_output,
                )
                continue

            payload[TARGET_KEY_CATEGORY] = category_code

            if args.dry_run:
                logger.info("[%s] DRY RUN: would write %s=%s", pair_name, TARGET_KEY_CATEGORY, category_code)
            else:
                json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                logger.info("[%s] Wrote %s=%s", pair_name, TARGET_KEY_CATEGORY, category_code)
            updated += 1

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

        except ModelCallFailure as exc:
            failed += 1
            logger.error("[%s] ModelCallFailure: %s", pair_name, exc)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.error("[%s] Failed: %s", pair_name, exc)

    logger.info("Done")
    logger.info("Processed: %d", processed)
    logger.info("Updated: %d", updated)
    logger.info("Skipped (already had category): %d", skipped_existing)
    logger.info("Skipped (missing source keys): %d", skipped_missing_source)
    logger.info("Skipped (doubt not true): %d", skipped_not_doubt)
    logger.info("Failed: %d", failed)


if __name__ == "__main__":
    main()
