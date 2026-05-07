#!/usr/bin/env python3
"""Mimic the EXACT DSPy prompt structure with direct Azure OpenAI calls (no DSPy).

Supports both:
  - NoCot (Predict) models  — output fields: discard_image, changes
  - CoT (ChainOfThought)    — output fields: reasoning, discard_image, changes

Accepts two JSON formats for --model-json:
  1. Standard DSPy saved model JSON (has a "signature" key)
  2. Client-integration export JSON  (has "static_messages" + "user_content_template")

DSPy NoCot (Predict) constructs a very specific two-message format:

  SYSTEM MESSAGE
  --------------
  Your input fields are:
  1. `image_original` (Union[Image, NoneType]): <description>
  2. `image_reimage` (Union[Image, NoneType]): <description>
  Your output fields are:
  1. `discard_image` (str): <description>
  2. `changes` (str): <description>
  All interactions will be structured in the following way, with the
  appropriate values filled in.

  [[ ## image_original ## ]]
  {image_original}

  [[ ## image_reimage ## ]]
  {image_reimage}

  [[ ## discard_image ## ]]
  {discard_image}

  [[ ## changes ## ]]
  {changes}

  [[ ## completed ## ]]
  In adhering to this structure, your objective is:
  <signature.instructions>

  USER MESSAGE  (multimodal: text + image_url parts)
  ---------------------------------------------------
  [[ ## image_original ## ]]
  <image1 as image_url content part; or literal text "None" if absent>

  [[ ## image_reimage ## ]]
  <image2 as image_url content part; or literal text "None" if absent>

  Respond with the corresponding output fields, starting with the field
  `[[ ## discard_image ## ]]`, then `[[ ## changes ## ]]`, and then
  ending with the marker for `[[ ## completed ## ]]`.

  MODEL RESPONSE (plain text, not json_object)
  --------------------------------------------
  [[ ## discard_image ## ]]
  true

  [[ ## changes ## ]]
  Images are identical - no changes

  [[ ## completed ## ]]

This script loads a DSPy saved model JSON (the .cache/*.json file) to
extract the signature (instructions + field descriptions), reconstructs
these messages faithfully, and evaluates with the same metrics as the
optimized prompt eval.

Usage
-----
    # Evaluate the gpt-4.1-mini optimised model on the val set,
    # running inference with gpt-4.1-mini:
    python evaluation/dca_run_val_test_set_eval/run_dspy_mimic_direct_eval.py \\
        --model-json .cache/azure_gpt-4.1-mini_MIPROv2_DCA_Light_NoCot_dca_optimized_max_6_6_20260402141458.json \\
        --deployment gpt-4.1-mini \\
        --which-set val

    # Run on both sets, overwrite existing result:
    python evaluation/dca_run_val_test_set_eval/run_dspy_mimic_direct_eval.py \\
        --model-json .cache/azure_gpt-4.1-mini_MIPROv2_DCA_Light_NoCot_dca_optimized_max_6_6_20260402141458.json \\
        --deployment gpt-4.1-mini \\
        --which-set both \\
        --overwrite

    # Skip LLM judge (faster):
    python evaluation/dca_run_val_test_set_eval/run_dspy_mimic_direct_eval.py \\
        --model-json .cache/azure_gpt-4.1-mini_MIPROv2_DCA_Light_NoCot_dca_optimized_max_6_6_20260402141458.json \\
        --deployment gpt-4.1-mini \\
        --skip-llm

Output layout
-------------
  eval_results/dspy_mimic_runs/
  ├── eval_dspy_mimic_<run_key>_val.json
  └── eval_dspy_mimic_<run_key>_test.json
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import certifi
import dotenv
import numpy as np
from azure.identity import AzureCliCredential, get_bearer_token_provider
from azure.core.exceptions import ClientAuthenticationError
from openai import AzureOpenAI, DefaultHttpxClient
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

dotenv.load_dotenv(dotenv.find_dotenv(), override=True)

# ---------------------------------------------------------------------------
# Paths & repo root
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent


def _find_repo_root() -> Path:
    for candidate in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return SCRIPT_DIR.parents[1]


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.dca_metric import (
    FIELD_WEIGHT_CLS,
    FIELD_WEIGHT_CHANGE,
    _try_rule_based_match,
    judge_summary_metric,
    compute_classification_score,
    compute_total_score,
)
from evaluation.dca_run_val_test_set_eval.split_utils import resolve_default_split_file

DEFAULT_IMAGE_DIR  = REPO_ROOT / "data" / "dca_data" / "image_pairs"
DEFAULT_GT_DIR     = REPO_ROOT / "data" / "dca_data" / "image_pairs_llm_output_exp1"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "eval_results" / "dspy_mimic_runs"
DEFAULT_SPLIT_FILE = resolve_default_split_file(REPO_ROOT)
DEFAULT_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT")
DEFAULT_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
DEFAULT_TEMPERATURE = 0.1

# ---------------------------------------------------------------------------
# Model config loader — supports both standard DSPy saves and
# client-integration exports (with optional CoT / reasoning field)
# ---------------------------------------------------------------------------

def load_model_config(model_json_path: Path) -> dict:
    """Load model config from either a standard DSPy saved model JSON
    or a client-integration export JSON.

    Returns a dict with keys:
      format:               "signature" | "client_integration"
      is_cot:               bool  (True if a 'reasoning' output field exists)
      output_field_names:   list[str]  (e.g. ["reasoning", "discard_image", "changes"])
      --- signature format only ---
      sig:                  dict  (the raw signature dict from the JSON)
      --- client_integration format only ---
      system_message:       str
      user_content_template: list[dict]
    """
    data = json.loads(model_json_path.read_text(encoding="utf-8"))

    if "signature" in data:
        sig = data["signature"]
        output_names = [
            _field_python_name(f["prefix"])
            for f in sig["fields"]
            if not _field_python_name(f["prefix"]).startswith("image_")
        ]
        return {
            "format": "signature",
            "sig": sig,
            "is_cot": "reasoning" in output_names,
            "output_field_names": output_names,
        }
    elif "static_messages" in data and "user_content_template" in data:
        output_names = list(
            data.get("response_parsing", {}).get("fields", {}).keys()
        )
        return {
            "format": "client_integration",
            "system_message": data["static_messages"][0]["content"],
            "user_content_template": data["user_content_template"],
            "output_field_names": output_names,
            "is_cot": "reasoning" in output_names,
        }
    else:
        raise ValueError(
            f"Unrecognized model JSON format in {model_json_path}. "
            "Expected either 'signature' key (standard DSPy save) or "
            "'static_messages'+'user_content_template' (client-integration export)."
        )


# ---------------------------------------------------------------------------
# DSPy message builder
# ---------------------------------------------------------------------------

def _field_python_name(prefix: str) -> str:
    """Convert a DSPy field prefix like 'Image Original:' to 'image_original'."""
    return prefix.rstrip(":").strip().lower().replace(" ", "_")


def build_dspy_system_message(sig: dict) -> str:
    """Reconstruct the exact DSPy (NoCot/Predict) system message from a signature dict.

    DSPy Predict system message structure (dspy 3.x):

        Your input fields are:
        1. `{name}` ({type}): {description}
        ...
        Your output fields are:
        1. `{name}` ({type}): {description}
        ...
        All interactions will be structured in the following way, with the
        appropriate values filled in.

        [[ ## {input_name1} ## ]]
        {{input_name1}}

        [[ ## {input_name2} ## ]]
        {{input_name2}}

        [[ ## {output_name1} ## ]]
        {{output_name1}}

        [[ ## {output_name2} ## ]]
        {{output_name2}}

        [[ ## completed ## ]]
        In adhering to this structure, your objective is:
        {instructions}
    """
    fields = sig["fields"]
    instructions = sig["instructions"]

    # Separate input fields (no output-like prefix) from output fields.
    # Convention in the JSON: first N fields are inputs, last M are outputs.
    # We detect them by matching the exact fields from the DCA signature:
    # inputs: image_original, image_reimage   outputs: discard_image, changes
    # More generically: inputs are image fields (Union[Image, NoneType]);
    # outputs are the non-image fields.
    # We'll use position: the JSON lists them in order [input..., output...].
    # From the known schema there are 2 inputs and 2 outputs.
    # To be robust we check whether the description looks like an image field.
    input_fields:  list[dict] = []
    output_fields: list[dict] = []
    for f in fields:
        name = _field_python_name(f["prefix"])
        # Input fields are dspy.Image inputs — their python name starts with "image_"
        # e.g. image_original, image_reimage
        # Output fields like "discard_image" contain "image" as a substring but
        # start with "discard_" — must NOT be treated as inputs.
        if name.startswith("image_"):
            input_fields.append(f)
        else:
            output_fields.append(f)

    lines: list[str] = []

    # ── Input field declarations ─────────────────────────────────────────────
    lines.append("Your input fields are:")
    for i, f in enumerate(input_fields, 1):
        name = _field_python_name(f["prefix"])
        lines.append(f"{i}. `{name}` (Union[Image, NoneType]): {f['description']}")

    # ── Output field declarations ────────────────────────────────────────────
    lines.append("Your output fields are:")
    for i, f in enumerate(output_fields, 1):
        name = _field_python_name(f["prefix"])
        lines.append(f"{i}. `{name}` (str): {f['description']}")

    lines.append(
        "All interactions will be structured in the following way, with the "
        "appropriate values filled in."
    )
    lines.append("")

    # ── Template block with markers ──────────────────────────────────────────
    for f in input_fields:
        name = _field_python_name(f["prefix"])
        lines.append(f"[[ ## {name} ## ]]")
        lines.append(f"{{{{{name}}}}}")
        lines.append("")

    for f in output_fields:
        name = _field_python_name(f["prefix"])
        lines.append(f"[[ ## {name} ## ]]")
        lines.append(f"{{{{{name}}}}}")
        lines.append("")

    lines.append("[[ ## completed ## ]]")
    lines.append(f"In adhering to this structure, your objective is: {instructions}")

    return "\n".join(lines)


def build_dspy_user_content(
    sig: dict,
    original_image_path: Optional[Path],
    reimage_path:        Optional[Path],
) -> list[dict]:
    """Build the multimodal user message content list.

    DSPy injects images as image_url parts between the field-marker text blocks.
    The final text part requests output in DSPy marker format.
    """
    fields = sig["fields"]
    input_fields  = [f for f in fields if _field_python_name(f["prefix"]).startswith("image_")]
    output_fields = [f for f in fields if not _field_python_name(f["prefix"]).startswith("image_")]

    # Map field name → image path
    images: dict[str, Optional[Path]] = {}
    for f in input_fields:
        name = _field_python_name(f["prefix"])
        if "original" in name:
            images[name] = original_image_path
        elif "reimage" in name:
            images[name] = reimage_path
        else:
            images[name] = None

    content: list[dict] = []

    for idx, f in enumerate(input_fields):
        name = _field_python_name(f["prefix"])
        img_path = images.get(name)

        # DSPy joins field sections with "\n\n".join(messages) —
        # the first field has no leading separator, subsequent ones get \n\n.
        separator = "" if idx == 0 else "\n\n"
        content.append({"type": "text", "text": f"{separator}[[ ## {name} ## ]]\n"})

        if img_path is not None and img_path.exists():
            data_url = _image_to_data_url(img_path)
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        else:
            # DSPy renders None values as the literal text "None" via str(None).
            # This tells the model explicitly that no image is present.
            content.append({"type": "text", "text": "None"})

    # Build the "Respond with..." instruction
    out_field_names = [_field_python_name(f["prefix"]) for f in output_fields]
    parts = []
    for i, name in enumerate(out_field_names):
        if i == 0:
            parts.append(f"starting with the field `[[ ## {name} ## ]]`")
        else:
            parts.append(f"then `[[ ## {name} ## ]]`")

    respond_instruction = (
        "Respond with the corresponding output fields, "
        + ", ".join(parts)
        + f", and then ending with the marker for `[[ ## completed ## ]]`."
    )
    # DSPy separates the instruction from fields with \n\n (from join)
    content.append({"type": "text", "text": "\n\n" + respond_instruction})

    return content


def _build_user_content_from_template(
    template: list[dict],
    original_image_path: Optional[Path],
    reimage_path:        Optional[Path],
) -> list[dict]:
    """Build user content from a client-integration template.

    Replaces ``<<IMAGE_ORIGINAL>>`` / ``<<IMAGE_REIMAGE>>`` sentinels with
    real base64 data-URLs.  When an image is *None*, the ``image_url`` part
    is replaced with a ``{"type": "text", "text": "None"}`` part — matching
    how DSPy renders absent images.
    """
    content = copy.deepcopy(template)
    result: list[dict] = []
    for part in content:
        if part.get("type") == "image_url":
            url = part["image_url"]["url"]
            if url == "<<IMAGE_ORIGINAL>>":
                if original_image_path and original_image_path.exists():
                    part["image_url"]["url"] = _image_to_data_url(original_image_path)
                    result.append(part)
                else:
                    result.append({"type": "text", "text": "None"})
            elif url == "<<IMAGE_REIMAGE>>":
                if reimage_path and reimage_path.exists():
                    part["image_url"]["url"] = _image_to_data_url(reimage_path)
                    result.append(part)
                else:
                    result.append({"type": "text", "text": "None"})
            else:
                result.append(part)
        else:
            result.append(part)
    return result


# ---------------------------------------------------------------------------
# Response parser — extract DSPy field values from marker-delimited text
# ---------------------------------------------------------------------------

def _extract_dspy_field(text: str, field_name: str, next_field_name: Optional[str] = None) -> str:
    """Extract the value for a DSPy output field from the model response text.

    Looks for   [[ ## field_name ## ]]\\n<value>
    and stops at the next marker or [[ ## completed ## ]].
    """
    pattern = rf"\[\[\s*##\s*{re.escape(field_name)}\s*##\s*\]\]\s*\n?(.*?)(?=\[\[\s*##|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def parse_dspy_response(
    raw: str,
    *,
    sig: dict | None = None,
    output_field_names: list[str] | None = None,
) -> dict[str, str]:
    """Parse a DSPy-format model response into a dict of {field_name: value}.

    Accepts *either* a DSPy signature dict or an explicit list of output
    field names (but not both omitted).
    """
    if output_field_names is not None:
        names = output_field_names
    elif sig is not None:
        names = [
            _field_python_name(f["prefix"])
            for f in sig["fields"]
            if not _field_python_name(f["prefix"]).startswith("image_")
        ]
    else:
        raise ValueError("Must provide either sig or output_field_names")

    result: dict[str, str] = {}
    for i, name in enumerate(names):
        next_name = names[i + 1] if i + 1 < len(names) else "completed"
        result[name] = _extract_dspy_field(raw, name, next_name)
    return result


# ---------------------------------------------------------------------------
# Azure client helpers
# ---------------------------------------------------------------------------

def _build_ca_bundle() -> str:
    for var in ("AOAI_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        val = os.getenv(var)
        if val and Path(val).exists():
            return val
    return certifi.where()


def build_client(endpoint: str, api_version: str) -> AzureOpenAI:
    ca_bundle = _build_ca_bundle()
    api_key   = os.getenv("AZURE_OPENAI_API_KEY")
    if api_key:
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            http_client=DefaultHttpxClient(verify=ca_bundle),
        )
    credential     = AzureCliCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
        http_client=DefaultHttpxClient(verify=ca_bundle),
    )


def _image_to_data_url(path: Path) -> str:
    ext_to_mime = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
        ".tif": "image/tiff", ".tiff": "image/tiff",
    }
    media_type = ext_to_mime.get(path.suffix.lower(), "image/png")
    return f"data:{media_type};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _find_image(pair_dir: Path, stem: str) -> Optional[Path]:
    allowed = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
    matches = [p for p in sorted(pair_dir.glob(f"{stem}.*")) if p.suffix.lower() in allowed]
    return matches[0] if matches else None


def _normalize_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "1"}:  return True
        if v in {"false", "no", "0"}: return False
    if isinstance(value, int):
        return bool(value)
    return None


# ---------------------------------------------------------------------------
# Single-pair model call
# ---------------------------------------------------------------------------

class ModelCallFailure(Exception):
    pass


def call_model_dspy_style(
    client:           AzureOpenAI,
    deployment:       str,
    config:           dict,
    original_image:   Optional[Path],
    reimage:          Optional[Path],
    temperature:      float = 0.1,
    max_retries:      int   = 3,
    reasoning_effort: Optional[str] = None,
    verbose:          bool  = False,
) -> dict:
    """Call the model with the exact DSPy message structure and parse the response."""
    if config["format"] == "signature":
        system_text  = build_dspy_system_message(config["sig"])
        user_content = build_dspy_user_content(config["sig"], original_image, reimage)
    else:  # client_integration
        system_text  = config["system_message"]
        user_content = _build_user_content_from_template(
            config["user_content_template"], original_image, reimage
        )

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_content},
    ]

    extra_kwargs: dict = {}
    if reasoning_effort is not None:
        extra_kwargs["reasoning_effort"] = reasoning_effort
    else:
        extra_kwargs["temperature"] = temperature

    # NOTE: We do NOT set response_format=json_object — DSPy uses plain text
    # with [[ ## field ## ]] markers, not JSON.
    # Match DSPy's max_tokens setting from config.create_lm() (16384 for gpt-4.1-mini).
    extra_kwargs["max_tokens"] = 16384

    _RATELIMIT_WAIT = 30.0
    last_exc: Optional[Exception] = None
    response = None
    t0 = time.time()

    for attempt in range(1, max_retries + 2):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=messages,
                **extra_kwargs,
            )
            last_exc = None
            break
        except RateLimitError as exc:
            last_exc = exc
            wait = _RATELIMIT_WAIT
            try:
                hdr = exc.response.headers.get("Retry-After")
                if hdr:
                    wait = float(hdr)
            except Exception:
                pass
            print(f"    [attempt {attempt}] RateLimitError — waiting {wait:.0f}s …", flush=True)
            if attempt <= max_retries:
                time.sleep(wait)
        except (APITimeoutError, APIConnectionError) as exc:
            last_exc = exc
            wait = 2.0 ** attempt
            print(f"    [attempt {attempt}] {type(exc).__name__} — waiting {wait:.0f}s …", flush=True)
            if attempt <= max_retries:
                time.sleep(wait)
        except APIStatusError as exc:
            last_exc = exc
            if exc.status_code in (500, 502, 503, 504):
                wait = 2.0 ** attempt
                print(f"    [attempt {attempt}] APIStatusError {exc.status_code} — waiting {wait:.0f}s …", flush=True)
                if attempt <= max_retries:
                    time.sleep(wait)
            else:
                raise ModelCallFailure(f"APIStatusError {exc.status_code}: {exc}") from exc
        except (ClientAuthenticationError, Exception) as exc:
            # Catches Azure CLI token expiry mid-run
            exc_name = type(exc).__name__
            if "credential" in exc_name.lower() or "token" in str(exc).lower() or "authentication" in str(exc).lower():
                raise ModelCallFailure(
                    f"Azure credential error — run 'az login' to refresh your token. Detail: {exc}"
                ) from exc
            raise ModelCallFailure(f"Unexpected error: {exc}") from exc

    elapsed = time.time() - t0

    if response is None:
        raise ModelCallFailure(f"All {max_retries + 1} attempts failed. Last: {last_exc}")

    first_choice = response.choices[0] if response.choices else None
    raw = (first_choice.message.content or "").strip() if first_choice else ""

    if verbose:
        print(f"    [RAW] {repr(raw[:300])}{'...' if len(raw) > 300 else ''}")

    # Parse DSPy marker format
    parsed = parse_dspy_response(raw, output_field_names=config["output_field_names"])
    pred_discard_str = parsed.get("discard_image", "")
    pred_changes     = parsed.get("changes", "")

    # Fallback 1: model responded in JSON (e.g. with or without ```json``` fence)
    if not pred_discard_str:
        # Strip markdown fences if present
        raw_stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.DOTALL)
        try:
            payload = json.loads(raw_stripped)
            candidate = str(payload.get("discard_image") or payload.get("Discard Image") or "").strip()
            if candidate:
                pred_discard_str = candidate
                pred_changes     = str(payload.get("changes") or payload.get("Changes") or pred_changes).strip()
                print("    [WARN] Model responded in JSON instead of DSPy markers — parsed OK")
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback 2: look for bare true/false anywhere in the response
    if not pred_discard_str and raw:
        m = re.search(r'\b(true|false)\b', raw, re.IGNORECASE)
        if m:
            pred_discard_str = m.group(1).lower()
            if not pred_changes:
                pred_changes = raw
            print(f"    [WARN] Extracted discard='{pred_discard_str}' from raw text (no markers)")

    # Token usage
    usage = getattr(response, "usage", None)
    input_tokens     = int(getattr(usage, "prompt_tokens",     0) or 0) or None
    output_tokens    = int(getattr(usage, "completion_tokens", 0) or 0) or None
    total_tokens     = int(getattr(usage, "total_tokens",      0) or 0) or None
    details          = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0) or None

    return {
        "pred_discard_str": pred_discard_str,
        "pred_discard":     _normalize_bool(pred_discard_str),
        "pred_changes":     pred_changes,
        "raw_response":     raw,
        "response_time":    elapsed,
        "total_tokens":     total_tokens,
        "input_tokens":     input_tokens,
        "output_tokens":    output_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


# ---------------------------------------------------------------------------
# Azure pre-flight check
# ---------------------------------------------------------------------------

def _preflight_azure_check(n_pairs: int, avg_s: float = 10.0) -> None:
    if os.getenv("AZURE_OPENAI_API_KEY"):
        return
    try:
        token      = AzureCliCredential().get_token("https://cognitiveservices.azure.com/.default")
        expires_in = token.expires_on - time.time()
        estimated  = n_pairs * avg_s
        if expires_in < estimated + 300:
            print(f"  [WARN] Azure CLI token expires in {expires_in/60:.0f} min "
                  f"but run will take ~{estimated/60:.0f} min — consider running 'az login'.")
        else:
            print(f"  Azure CLI token valid for {expires_in/60:.0f} min "
                  f"(est. ~{estimated/60:.0f} min) — OK")
    except Exception as exc:
        print(f"\n[ERROR] Azure CLI credentials not valid: {exc}")
        print("  Please run 'az login' and retry.\n")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Per-set evaluation
# ---------------------------------------------------------------------------

def run_and_evaluate(
    config:           dict,
    pair_names:       list[str],
    image_dir:        Path,
    gt_dir:           Path,
    client:           AzureOpenAI,
    deployment:       str,
    temperature:      float,
    max_retries:      int,
    reasoning_effort: Optional[str],
    use_llm:          bool,
    verbose:          bool = False,
) -> list[dict]:
    gt_records: dict[str, dict] = {}
    for pname in pair_names:
        f = gt_dir / f"{pname}.json"
        if f.exists():
            gt_records[pname] = json.loads(f.read_text(encoding="utf-8"))

    per_pair: list[dict] = []
    n = len(pair_names)

    for i, pair_name in enumerate(pair_names, 1):
        pair_dir  = image_dir / pair_name
        original  = _find_image(pair_dir, "original")
        reimage   = _find_image(pair_dir, "reimage")

        if original is None and reimage is None:
            print(f"  [{i:03d}/{n}] {pair_name}  [SKIP] no images found")
            per_pair.append({"pair_name": pair_name, "missing": True, "note": "no images"})
            continue

        try:
            res = call_model_dspy_style(
                client=client,
                deployment=deployment,
                config=config,
                original_image=original,
                reimage=reimage,
                temperature=temperature,
                max_retries=max_retries,
                reasoning_effort=reasoning_effort,
                verbose=verbose,
            )
        except ModelCallFailure as exc:
            err_str = str(exc)
            print(f"  [{i:03d}/{n}] {pair_name}  [FAIL] {exc}")
            per_pair.append({"pair_name": pair_name, "missing": True, "note": err_str})
            # Abort on credential errors — every subsequent pair will also fail
            if "az login" in err_str or "credential" in err_str.lower():
                print("\n[FATAL] Azure credential error — aborting run. Run 'az login' and retry.")
                break
            continue

        # Scoring
        gt            = gt_records.get(pair_name, {})
        human_discard = _normalize_bool(gt.get("human_label_discard_image"))
        human_changes = str(gt.get("human_label_changes") or "").strip()
        pred_discard  = res["pred_discard"]

        if human_discard is None or pred_discard is None:
            raw_snippet = repr(res.get("raw_response", "")[:200])
            print(f"  [{i:03d}/{n}] {pair_name}  [SKIP] "
                  f"unparseable: human={gt.get('human_label_discard_image')!r}  "
                  f"pred={res['pred_discard_str']!r}")
            if not verbose:
                print(f"    [RAW snippet] {raw_snippet}")
            per_pair.append({
                "pair_name": pair_name, "missing": True,
                "note": "unparseable discard",
                "raw_response": res.get("raw_response", ""),
                **{k: res.get(k) for k in ("response_time", "total_tokens", "input_tokens",
                                            "output_tokens", "reasoning_tokens")},
            })
            continue

        cls_score = compute_classification_score(human_discard, pred_discard)

        change_score, change_source = None, "skipped"
        rb_score, _reason, matched = _try_rule_based_match(human_changes, res["pred_changes"])
        if matched:
            change_score, change_source = rb_score, "rule"
        elif use_llm:
            try:
                change_score  = judge_summary_metric(human_changes, res["pred_changes"])
                change_source = "llm"
            except Exception as exc:
                print(f"    [judge WARN] {pair_name}: {exc}")
                change_source = "llm_error"

        if change_score is not None:
            total_score = compute_total_score(cls_score, change_score)
        elif change_source in ("llm", "llm_error"):
            total_score = compute_total_score(cls_score, 0.0)
        else:
            total_score = None

        tok_str = f"  tok={res['total_tokens']}" if res.get("total_tokens") else ""
        chg_str = f"{change_score:.3f}" if change_score is not None else "N/A"
        print(f"  [{i:03d}/{n}] {pair_name}  "
              f"cls={cls_score:.0f}  chg={chg_str}  src={change_source}  "
              f"t={res['response_time']:.1f}s{tok_str}", flush=True)

        per_pair.append({
            "pair_name":        pair_name,
            "missing":          False,
            "cls_score":        cls_score,
            "change_score":     change_score,
            "total_score":      total_score,
            "change_source":    change_source,
            "response_time":    res["response_time"],
            "total_tokens":     res.get("total_tokens"),
            "input_tokens":     res.get("input_tokens"),
            "output_tokens":    res.get("output_tokens"),
            "reasoning_tokens": res.get("reasoning_tokens"),
            "pred_discard":     res["pred_discard_str"],
            "pred_changes":     res["pred_changes"],
            "raw_response":     res.get("raw_response", ""),
        })

    return per_pair


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate_results(per_pair: list[dict], run_key: str, run_label: str) -> dict:
    valid = [p for p in per_pair if not p.get("missing")]

    cls_scores    = [p["cls_score"]       for p in valid]
    change_scores = [p["change_score"]    for p in valid if p["change_score"]     is not None]
    total_scores  = [p["total_score"]     for p in valid if p["total_score"]      is not None]
    tokens        = [p["total_tokens"]    for p in valid if p.get("total_tokens") is not None]
    in_toks       = [p["input_tokens"]    for p in valid if p.get("input_tokens") is not None]
    out_toks      = [p["output_tokens"]   for p in valid if p.get("output_tokens") is not None]
    rsn_toks      = [p["reasoning_tokens"]for p in valid if p.get("reasoning_tokens") is not None]
    times         = [p["response_time"]   for p in valid if p.get("response_time") is not None]

    source_counts: dict[str, int] = {}
    for p in valid:
        src = p.get("change_source", "skipped")
        source_counts[src] = source_counts.get(src, 0) + 1

    return {
        "run":                  run_key,
        "label":                run_label,
        "n_test_pairs":         len(per_pair),
        "n_evaluated":          len(valid),
        "n_missing":            len(per_pair) - len(valid),
        "avg_cls_score":        float(np.mean(cls_scores))    if cls_scores    else None,
        "avg_change_score":     float(np.mean(change_scores)) if change_scores else None,
        "avg_total_score":      float(np.mean(total_scores))  if total_scores  else None,
        "avg_total_tokens":     float(np.mean(tokens))        if tokens        else None,
        "avg_input_tokens":     float(np.mean(in_toks))       if in_toks       else None,
        "avg_output_tokens":    float(np.mean(out_toks))      if out_toks      else None,
        "avg_reasoning_tokens": float(np.mean(rsn_toks))      if rsn_toks      else None,
        "avg_response_time":    float(np.mean(times))         if times         else None,
        "n_cls":                len(cls_scores),
        "n_change":             len(change_scores),
        "n_total_score":        len(total_scores),
        "change_source_counts": source_counts,
        "per_pair":             per_pair,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_table(results: list[dict], which_set: str, temperature: float) -> None:
    def f(v, fmt=".3f"):
        return f"{v:{fmt}}" if v is not None else "N/A"

    C = (38, 20, 7, 7, 7, 8, 7, 7, 7)
    header = (
        f"{'Run key':<{C[0]}} {'Deployment':<{C[1]}} "
        f"{'Cls':>{C[2]}} {'Chg':>{C[3]}} {'Tot':>{C[4]}} "
        f"{'Time(s)':>{C[5]}} {'InTok':>{C[6]}} {'OutTok':>{C[7]}} {'RsnTok':>{C[8]}}"
    )
    sep = "-" * len(header)
    print()
    print(f"  which-set: {which_set}  |  temperature: {temperature}  |  DSPy-mimic direct API")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        rk = r.get("run", "")
        dep = r.get("deployment", "")
        print(
            f"  {rk:<{C[0]-2}} {dep:<{C[1]}} "
            f"{f(r.get('avg_cls_score')):>{C[2]}} "
            f"{f(r.get('avg_change_score')):>{C[3]}} "
            f"{f(r.get('avg_total_score')):>{C[4]}} "
            f"{f(r.get('avg_response_time'), '.1f'):>{C[5]}} "
            f"{f(r.get('avg_input_tokens'), '.0f'):>{C[6]}} "
            f"{f(r.get('avg_output_tokens'), '.0f'):>{C[7]}} "
            f"{str(r.get('avg_reasoning_tokens') if r.get('avg_reasoning_tokens') is not None else 'N/A'):>{C[8]}}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Evaluate a DSPy optimized model by mimicking its exact prompt structure "
                    "via direct Azure OpenAI API calls (no DSPy runtime required)."
    )
    parser.add_argument(
        "--model-json", type=Path, required=True,
        help="Path to the DSPy saved model .json file (e.g. .cache/azure_*.json)"
    )
    parser.add_argument(
        "--deployment", required=True,
        help="Azure deployment name to use for inference (e.g. gpt-4.1-mini)"
    )
    parser.add_argument("--which-set",   choices=["val", "test", "both"], default="val")
    parser.add_argument("--split-file",  type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--image-dir",   type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--gt-dir",      type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--output-dir",  type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--endpoint",    default=DEFAULT_ENDPOINT)
    parser.add_argument("--api-version", default=DEFAULT_API_VER)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--reasoning-effort", default=None,
                        choices=["low", "medium", "high"],
                        help="Reasoning effort for o-series / gpt-5.x models")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM judge for change score (faster)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print raw model response for every pair (useful for debugging)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-id", default=None,
                        help="Optional suffix to distinguish repeated runs (e.g. rep01)")
    args = parser.parse_args()

    if not args.model_json.exists():
        print(f"[ERROR] Model JSON not found: {args.model_json}")
        sys.exit(1)

    # Load model config (supports both standard DSPy saves and client-integration)
    config = load_model_config(args.model_json)
    cot_tag = " [CoT]" if config["is_cot"] else " [NoCot]"

    # Print the system message that will be used — useful for debugging
    if config["format"] == "signature":
        system_msg = build_dspy_system_message(config["sig"])
    else:
        system_msg = config["system_message"]
    print("=" * 80)
    print(f"DSPy-mimic SYSTEM MESSAGE{cot_tag} (will be sent to every inference call):")
    print("=" * 80)
    print(system_msg)
    print("=" * 80)

    # Derive run key from model filename + deployment
    model_stem = args.model_json.stem  # e.g. azure_gpt-4.1-mini_MIPROv2_DCA_Light_NoCot_...
    dep_slug   = args.deployment.replace("-", "").replace(".", "").replace("/", "")
    temp_sfx   = (f"_temp{args.temperature}"
                  if args.temperature != DEFAULT_TEMPERATURE
                  else str(args.temperature))
    run_key    = f"dspy_mimic_{dep_slug}{temp_sfx}"
    if args.run_id:
        run_key = f"{run_key}_{args.run_id}"
    run_label  = f"DSPy-mimic/{args.model_json.name}_run_{args.deployment}"

    # Load split
    if not args.split_file.exists():
        print(f"[ERROR] Split file not found: {args.split_file}")
        sys.exit(1)
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    seed  = split.get("seed", "?")

    def _get_pair_names(key: str) -> list[str]:
        names = (split.get(key, {}).get("pair_names")
                 or split.get(f"{key}_pair_numbers") or [])
        if names and isinstance(names[0], int):
            names = [f"pair_{n:03d}" for n in names]
        return names

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client   = build_client(args.endpoint, args.api_version)
    use_llm  = not args.skip_llm
    sets     = ["val", "test"] if args.which_set == "both" else [args.which_set]
    summary  = []

    for set_name in sets:
        pair_names  = _get_pair_names(set_name)
        total_pairs = len(pair_names)

        print(f"\nSet: {set_name.upper()}  ({total_pairs} pairs, seed={seed})")
        print(f"Model JSON   : {args.model_json.name}")
        print(f"Deployment   : {args.deployment}  temperature={args.temperature}")
        if args.reasoning_effort:
            print(f"Rsn effort   : {args.reasoning_effort}")
        print(f"LLM judge    : {'ENABLED' if use_llm else 'DISABLED (--skip-llm)'}")
        print(f"Mode         : DSPy-mimic direct API (no DSPy runtime)\n")

        _preflight_azure_check(total_pairs)

        results_path = args.output_dir / f"eval_dspy_mimic_{run_key}_{set_name}.json"
        if not args.overwrite and results_path.exists():
            existing = json.loads(results_path.read_text(encoding="utf-8"))
            if run_key in {r["run"] for r in existing.get("results", [])}:
                print(f"[ERROR] Run '{run_key}' already exists in {results_path.name}")
                print("  Use --overwrite to re-evaluate.")
                sys.exit(1)

        per_pair = run_and_evaluate(
            config=config,
            pair_names=pair_names,
            image_dir=args.image_dir,
            gt_dir=args.gt_dir,
            client=client,
            deployment=args.deployment,
            temperature=args.temperature,
            max_retries=args.max_retries,
            reasoning_effort=args.reasoning_effort,
            use_llm=use_llm,
            verbose=args.verbose,
        )

        result = aggregate_results(per_pair, run_key=run_key, run_label=run_label)
        result["model_json"]   = str(args.model_json)
        result["deployment"]   = args.deployment
        result["temperature"]  = args.temperature
        result["reasoning_effort"] = args.reasoning_effort

        # Merge with any existing JSON
        old_order: list[str]       = []
        existing_map: dict[str, dict] = {}
        if results_path.exists():
            _old = json.loads(results_path.read_text(encoding="utf-8"))
            for r in _old.get("results", []):
                old_order.append(r["run"])
                existing_map[r["run"]] = r
        existing_map[run_key] = result
        merged_order = list(old_order)
        if run_key not in merged_order:
            merged_order.append(run_key)
        merged = [existing_map[r] for r in merged_order if r in existing_map]

        meta = {
            "split_file":            str(args.split_file),
            "which_set":             set_name,
            "n_pairs":               total_pairs,
            "skip_llm":              not use_llm,
            "field_weight_cls":      FIELD_WEIGHT_CLS,
            "field_weight_change":   FIELD_WEIGHT_CHANGE,
            "runs":                  merged_order,
        }
        results_path.write_text(
            json.dumps({"meta": meta, "results": merged}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        c  = f"{result['avg_cls_score']:.3f}"    if result.get("avg_cls_score")    is not None else "N/A"
        ch = f"{result['avg_change_score']:.3f}" if result.get("avg_change_score") is not None else "N/A"
        t  = f"{result['avg_total_score']:.3f}"  if result.get("avg_total_score")  is not None else "N/A"
        print(f"\nResults -> {results_path}")
        print(f"  cls={c}  chg={ch}  tot={t}")

        if set_name == sets[-1]:
            summary.append(result)

    print(f"\n{'='*90}")
    print("FINAL TABLE")
    print(f"{'='*90}")
    print_table(summary, which_set=args.which_set, temperature=args.temperature)
    print("\nDone.")


if __name__ == "__main__":
    main()
