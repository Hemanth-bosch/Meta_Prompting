import dspy
from typing import Optional
import dotenv
import os
from pathlib import Path

import certifi
from azure.identity import AzureCliCredential
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

# TODO 
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
_azure_cli_credential = None


def _azure_cli_token_provider() -> str:
    """Return Azure AD bearer token for Azure OpenAI.

    This callback is intentionally module-level (picklable-friendly) so DSPy
    can deepcopy LM instances during optimizers like MIPROv2.
    """
    global _azure_cli_credential
    if _azure_cli_credential is None:
        _azure_cli_credential = AzureCliCredential()
    token = _azure_cli_credential.get_token("https://cognitiveservices.azure.com/.default")
    return token.token


def _get_azure_client(auth_mode: str = "cli"):
    global _azure_fallback_client
    if _azure_fallback_client is not None:
        return _azure_fallback_client

    ca_bundle = _build_ca_bundle()
    print(f"Using CA bundle: {ca_bundle}")

    if auth_mode == "key":
        print("Creating Azure OpenAI client with direct API Key...")
        _azure_fallback_client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
            http_client=DefaultHttpxClient(verify=ca_bundle),
        )
    else:
        print("Initializing Azure CLI credential...")
        credential = AzureCliCredential()

        print("Creating token provider...")
        try:
            from azure.identity import get_bearer_token_provider
            token_provider = get_bearer_token_provider(
                credential,
                "https://cognitiveservices.azure.com/.default",
            )
        except ImportError:
            def token_provider():
                token = credential.get_token("https://cognitiveservices.azure.com/.default")
                return token.token

        print("Creating Azure OpenAI client with token provider...")
        _azure_fallback_client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            azure_ad_token_provider=token_provider,
            api_version=AZURE_API_VERSION,
            http_client=DefaultHttpxClient(verify=ca_bundle),
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
    else:
        # Unknown provider — pass no extra params, let litellm handle it
        return {}


def _get_default_max_tokens(model_name: str) -> int:
    """Return a safe default max_tokens for the given model, staying within provider limits."""
    name_lower = model_name.lower()
    # Strip provider prefix for matching (e.g. "azure/gpt-4o-mini" → "gpt-4o-mini")
    base = name_lower.split("/", 1)[-1] if "/" in name_lower else name_lower

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
        **kwargs,
    }

    if provider == "azure":
        if AZURE_API_KEY:
            lm_params.update({
                "api_key": AZURE_API_KEY,
                "api_base": AZURE_ENDPOINT,
                "api_version": AZURE_API_VERSION,
                "cache": False,
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