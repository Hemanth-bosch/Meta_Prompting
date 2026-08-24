"""LM / runtime configuration for the DCA pipeline.

Builds the DSPy language models used across DCA optimization and evaluation:
  - ``create_lm(model_name, ...)`` — a provider-agnostic ``dspy.LM`` (Azure /
    Vertex / OpenAI / Anthropic, chosen by the model-name prefix). Azure uses an
    API key when ``AZURE_OPENAI_API_KEY`` is set, otherwise an ``az login``
    bearer token. Handles per-model ``max_tokens`` caps, gpt-5 temperature
    auto-config, response caching, and the corporate CA bundle.
  - ``reconfigure_lm(model_name, ...)`` — swap the globally-configured student LM.
  - ``reflectin_lm`` — the separate reflection LM used by GEPA.

The DCA metric's LLM judge builds its own Azure client here via
``_get_azure_client`` / ``_build_ca_bundle``.
"""


import dspy
from typing import Optional
import dotenv
import os
from datetime import datetime
import json
from pathlib import Path
import subprocess
import shutil
import sys
from threading import Lock
import time

import certifi
from openai import AzureOpenAI, DefaultHttpxClient

# Load environment variables
dotenv.load_dotenv()

# Load configuration from .env
_dotenv_file = dotenv.find_dotenv()

# Azure OpenAI credentials

AZURE_API_KEY = dotenv.get_key(_dotenv_file, "AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = dotenv.get_key(_dotenv_file, "AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT", "https://aiq212-openai.openai.azure.com/")
AZURE_API_VERSION = dotenv.get_key(_dotenv_file, "AZURE_OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

# Vertex AI credentials

VERTEX_PROJECT = dotenv.get_key(_dotenv_file, "VERTEX_PROJECT")
VERTEX_LOCATION = dotenv.get_key(_dotenv_file, "VERTEX_LOCATION")

# Direct API keys (for non-Azure usage)

OPENAI_API_KEY = dotenv.get_key(_dotenv_file, "OPENAI_API_KEY")
ANTHROPIC_API_KEY = dotenv.get_key(_dotenv_file, "ANTHROPIC_API_KEY")

# Optional CA bundle override for corporate TLS-inspecting environments
REQUESTS_CA_BUNDLE = dotenv.get_key(_dotenv_file, "REQUESTS_CA_BUNDLE") or os.getenv("REQUESTS_CA_BUNDLE")

# Model names
DEFAULT_MODEL_NAME = dotenv.get_key(_dotenv_file, "MODEL_NAME")
REFLECTION_MODEL_NAME = dotenv.get_key(_dotenv_file, "REFLECTION_LM_NAME")

DCA_JUDGE_MODEL_NAME = "gpt-4.1"  # LLM judge used in dca_metric.judge_summary_metric



def _build_ca_bundle() -> str:
    explicit = os.getenv("AOAI_CA_BUNDLE") or os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")
    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.exists():
            return str(explicit_path)
        print(f"Warning: CA bundle path from env var not found: {explicit}. Falling back to certifi/combined bundle.")

    exported_dir = Path(os.getenv("TEMP", "")) / "bosch-ca"
    exported_pems = sorted(exported_dir.glob("*.pem")) if exported_dir.is_dir() else []

    if not exported_pems:
        return certifi.where()

    out_path = exported_dir / "combined-ca-bundle.pem"
    certifi_path = Path(certifi.where())

    parts: list[Path] = []
    if certifi_path.exists():
        parts.append(certifi_path)
    parts.extend(exported_pems)

    combined = "\n".join(p.read_text(encoding="utf-8", errors="ignore").strip() for p in parts if p.exists())
    combined = combined.strip() + "\n"
    out_path.write_text(combined, encoding="utf-8")
    return str(out_path)



_azure_fallback_client = None
_azure_cli_token_cache: str | None = None
_azure_cli_token_expiry = 0.0
_azure_cli_token_lock = Lock()



def _find_azure_cli_entrypoint() -> str:
    az_path = shutil.which("az") or shutil.which("az.cmd") or shutil.which("az.bat")
    if not az_path:
        raise RuntimeError(
            "Azure CLI executable 'az' was not found on PATH. "
            "Install Azure CLI system-wide or run with AZURE_OPENAI_API_KEY configured."
        )
    az_path_obj = Path(az_path)
    if os.name == "nt" and az_path_obj.suffix.lower() not in {".cmd", ".bat", ".exe"}:
        for suffix in (".cmd", ".bat", ".exe"):
            candidate = az_path_obj.with_suffix(suffix)
            if candidate.exists():
                return str(candidate)
    return az_path



def _build_azure_cli_command() -> tuple[list[str], dict[str, str]]:
    az_path = _find_azure_cli_entrypoint()
    env = os.environ.copy()

    python_dir = str(Path(sys.executable).resolve().parent)
    path_parts = env.get("PATH", "").split(os.pathsep) if env.get("PATH") else []
    if not any(os.path.normcase(part) == os.path.normcase(python_dir) for part in path_parts if part):
        env["PATH"] = os.pathsep.join([python_dir, *[part for part in path_parts if part]])

    env.pop("PYTHONPATH", None)


    command = [
        az_path,
        "account",
        "get-access-token",
        "--resource",
        "https://cognitiveservices.azure.com",
        "--output",
        "json",
    ]

    return command, env



def _parse_azure_cli_expiry(payload: dict) -> float:
    expires_on = payload.get("expires_on") or payload.get("expiresOn")
    if expires_on is None:
        return 0.0

    if isinstance(expires_on, (int, float)):
        return float(expires_on)

    if isinstance(expires_on, str):
        stripped = expires_on.strip()
        if stripped.isdigit():
            return float(stripped)

        for fmt in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
        ):
            try:
                return datetime.strptime(stripped, fmt).timestamp()
            except ValueError:
                continue

    return 0.0



def _fetch_azure_cli_token() -> tuple[str, float]:
    command, env = _build_azure_cli_command()
    result = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details = stderr or stdout or f"Azure CLI exited with code {result.returncode}."
        raise RuntimeError(
            "Azure CLI token retrieval failed. "
            "Run `az login` manually or configure AZURE_OPENAI_API_KEY. "
            f"Details: {details}"
        )

    payload = json.loads(result.stdout)
    token = payload.get("accessToken") or payload.get("access_token")
    if not token:
        raise RuntimeError("Azure CLI token retrieval returned no access token.")

    return token, _parse_azure_cli_expiry(payload)



def _azure_cli_token_provider() -> str:
    """Return Azure AD bearer token for Azure OpenAI.

    This callback is intentionally module-level (picklable-friendly) so DSPy
    can deepcopy LM instances during optimizers like MIPROv2.
    """
    global _azure_cli_token_cache, _azure_cli_token_expiry

    with _azure_cli_token_lock:
        now = time.time()
        if _azure_cli_token_cache and now < (_azure_cli_token_expiry - 300):
            return _azure_cli_token_cache

        token, expires_at = _fetch_azure_cli_token()
        _azure_cli_token_cache = token
        _azure_cli_token_expiry = expires_at or (now + 300)
        return token



def _get_azure_client(auth_mode: Optional[str] = None):
    global _azure_fallback_client
    if _azure_fallback_client is not None:
        return _azure_fallback_client

    # Auto-select auth: prefer the API key when it's set (works on any machine),
    # otherwise fall back to the az-login bearer token. Callers can still force a
    # mode explicitly by passing auth_mode="key" / "cli".
    if auth_mode is None:
        auth_mode = "key" if AZURE_API_KEY else "cli"

    # Always pass an explicit http_client so httpx doesn't call certifi.where()
    # internally. Use the CA bundle if one exists (corporate TLS proxy), otherwise
    # verify=False (encrypted but unverified — fine for trusted Azure endpoints).
    ca_bundle = _build_ca_bundle()
    ca_bundle_valid = Path(ca_bundle).exists()
    http_client = DefaultHttpxClient(verify=ca_bundle if ca_bundle_valid else False)
    if ca_bundle_valid:
        print(f"Using CA bundle: {ca_bundle}")

    if auth_mode == "key":
        print("Creating Azure OpenAI client with direct API Key...")
        _azure_fallback_client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
            http_client=http_client,
        )
    else:                                                  # az-login token path
        print("Initializing Azure CLI credential...")
        print("Creating token provider...")
        def token_provider():
            return _azure_cli_token_provider()
        print("Creating Azure OpenAI client with token provider...")
        _azure_fallback_client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            azure_ad_token_provider=token_provider,
            api_version=AZURE_API_VERSION,
            http_client=http_client,
        )

    print("Azure Client created successfully.")
    return _azure_fallback_client



def _get_provider(model_name: str) -> str:
    """Extract the provider from a model name prefix."""
    if "/" in model_name:
        return model_name.split("/")[0]

    if OPENAI_API_KEY:
        return "openai"
    if AZURE_API_KEY or AZURE_ENDPOINT:
        return "azure"
    return "openai"



def _get_provider_params(model_name: str) -> dict:
    """
    Return provider-specific parameters for dspy.LM based on the model name prefix.

    Supported providers:
        - azure/*       : Azure OpenAI (api_key, api_base, api_version)
        - vertex_ai/*   : Vertex AI — Gemini & Claude (vertex_project, vertex_location)
        - openai/*      : OpenAI direct (api_key)
        - anthropic/*   : Anthropic direct (api_key)
        - (no prefix)   : Falls back to OpenAI direct
    """
    provider = _get_provider(model_name)

    if provider == "azure":
        return {
            "api_key": AZURE_API_KEY,
            "api_base": AZURE_ENDPOINT,
            "api_version": AZURE_API_VERSION,
        }
    elif provider == "vertex_ai":
        params = {}
        if VERTEX_PROJECT:
            params["vertex_project"] = VERTEX_PROJECT
        if VERTEX_LOCATION:
            params["vertex_location"] = VERTEX_LOCATION
        return params
    elif provider == "openai":
        params = {}
        if OPENAI_API_KEY:
            params["api_key"] = OPENAI_API_KEY
        return params
    elif provider == "anthropic":
        params = {}
        if ANTHROPIC_API_KEY:
            params["api_key"] = ANTHROPIC_API_KEY
        return params
    else:                                                  # unknown provider
        # Unknown provider — pass no extra params, let litellm handle it
        return {}



def _get_default_max_tokens(model_name: str) -> int:
    """Return a safe default max_tokens for the given model, staying within provider limits."""
    name_lower = model_name.lower()
    # Strip provider prefix for matching (e.g. "azure/gpt-4o-mini" → "gpt-4o-mini")
    base = name_lower.split("/", 1)[-1] if "/" in name_lower else name_lower

    # Optional STUDENT/WORKER output cap for the DCA tournament A/B against the agentic
    # loop: set by run_tournament's --worker-max-tokens (exported as DCA_WORKER_MAX_TOKENS).
    # Applies ONLY to the gpt-4.1 / gpt-4o student worker (matching the agentic loop's
    # --worker-max-tokens 2048), bounding repetition blow-ups so the token comparison is
    # fair. gpt-5.2 seed/reflection/teacher helpers keep their full budget. No-op (and
    # byte-identical) when the env var is unset. Only affects calls that don't pass an
    # explicit max_tokens (all the student create_lm sites do exactly that).
    _wcap = os.environ.get("DCA_WORKER_MAX_TOKENS")
    if _wcap and ("gpt-4.1" in base or "gpt-4o" in base):
        try:
            return int(_wcap)
        except (TypeError, ValueError):
            pass

    # Model-specific limits (output token limits from provider docs)

    if "gpt-5" in base:
        return 20000
    if "gpt-4.1" in base:
        return 16384
    if "gpt-4o-mini" in base:
        return 16384
    if "gpt-4o" in base:
        return 16384
    if "o3" in base or "o4" in base:
        return 16384  # reasoning models
    if "gemini" in base:
        return 8192
    if "claude" in base:
        return 8192
    # Conservative fallback
    return 8192



def create_lm(model_name: Optional[str] = None,
              temperature: Optional[float] = None,
              max_tokens: Optional[int] = None,
              cache: bool = True,
              **kwargs) -> dspy.LM:
    """
    Create a DSPy LM instance with the correct credentials for the given provider.

    The provider is detected from the model name prefix:
        azure/gpt-4o-mini          → Azure OpenAI
        vertex_ai/gemini-2.0-flash → Vertex AI (Gemini)
        vertex_ai/claude-sonnet-4-20250514 → Vertex AI (Claude)
        gpt-4o              → OpenAI direct
        anthropic/claude-sonnet-4-20250514 → Anthropic direct

    Args:
        model_name: Model name with provider prefix (defaults to MODEL_NAME from .env)
        temperature: Temperature setting (auto-set to 1.0 for gpt-5 if not specified)
        max_tokens: Max tokens (auto-derived per model if not specified)
        cache: Enable/disable response caching (default: True)
        **kwargs: Additional parameters for dspy.LM

    Returns:
        dspy.LM instance configured with the appropriate credentials
    """
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME

    provider = _get_provider(model_name)

    if "/" not in model_name:
        if provider == "azure":
            model_name = f"azure/{model_name}"
        elif provider == "openai":
            model_name = f"openai/{model_name}"

    # Auto-select max_tokens based on model limits if not explicitly provided
    if max_tokens is None:
        max_tokens = _get_default_max_tokens(model_name)

    # Auto-configure gpt-5 models with special settings
    is_gpt5 = "gpt-5" in model_name.lower()
    if is_gpt5:
        if temperature is None:
            temperature = 1.0

    # Build LM parameters: provider-specific credentials + common params
    lm_params = {
        "model": model_name,
        "cache": cache,
        "num_retries": 4,
        "timeout": 300,
        **kwargs,
    }

    if provider == "azure":
        if AZURE_API_KEY:
            lm_params.update({
                "api_key": AZURE_API_KEY,
                "api_base": AZURE_ENDPOINT,
                "api_version": AZURE_API_VERSION,
                "cache": cache,
            })
        else:
            ca_bundle = _build_ca_bundle()
            os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)
            os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
            lm_params.update({
                "api_base": AZURE_ENDPOINT,
                "api_version": AZURE_API_VERSION,
                "azure_ad_token_provider": _azure_cli_token_provider,
                "cache": False,
            })
    else:
        lm_params.update(_get_provider_params(model_name))

    if temperature is not None:
        lm_params["temperature"] = temperature
    if max_tokens is not None:
        lm_params["max_tokens"] = max_tokens

    return dspy.LM(**lm_params)


# Initialize main language model
lm = create_lm(DEFAULT_MODEL_NAME)

# Initialize reflection language model (used by GEPA and similar optimizers)

# traces and rewrites instructions during optimization; temperature=1.0 for diverse proposals.

reflectin_lm = create_lm(REFLECTION_MODEL_NAME, temperature=1.0)

# Configure DSPy with the main LM
dspy.configure(lm=lm)



def reconfigure_lm(model_name: str, **kwargs) -> dspy.LM:
    """
    Reconfigure DSPy with a different model name.

    Args:
        model_name: Name of the model to use (e.g., 'azure/gpt-4o', 'vertex_ai/gemini-2.0-flash')
        **kwargs: Additional parameters for the LM (temperature, max_tokens, etc.)

    Returns:
        The newly configured LM instance
    """
    global lm

    lm = create_lm(model_name, **kwargs)
    dspy.configure(lm=lm)

    return lm