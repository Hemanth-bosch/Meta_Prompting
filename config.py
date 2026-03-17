import dspy
from typing import Optional
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Load configuration from .env
_dotenv_file = dotenv.find_dotenv()

# Azure OpenAI credentials
AZURE_API_KEY = dotenv.get_key(_dotenv_file, "AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = dotenv.get_key(_dotenv_file, "AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = dotenv.get_key(_dotenv_file, "AZURE_OPENAI_API_VERSION")

# Vertex AI credentials
VERTEX_PROJECT = dotenv.get_key(_dotenv_file, "VERTEX_PROJECT")
VERTEX_LOCATION = dotenv.get_key(_dotenv_file, "VERTEX_LOCATION")

# Direct API keys (for non-Azure usage)
OPENAI_API_KEY = dotenv.get_key(_dotenv_file, "OPENAI_API_KEY")
ANTHROPIC_API_KEY = dotenv.get_key(_dotenv_file, "ANTHROPIC_API_KEY")

# Model names
DEFAULT_MODEL_NAME = dotenv.get_key(_dotenv_file, "MODEL_NAME")
REFLECTION_MODEL_NAME = dotenv.get_key(_dotenv_file, "REFLECTION_LM_NAME")


def _get_provider(model_name: str) -> str:
    """Extract the provider from a model name prefix."""
    if "/" in model_name:
        return model_name.split("/")[0]
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
        max_tokens: Max tokens (auto-set to 20000 for gpt-5 if not specified)
        cache: Enable/disable response caching (default: True)
        **kwargs: Additional parameters for dspy.LM

    Returns:
        dspy.LM instance configured with the appropriate credentials
    """
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME

    # Auto-configure gpt-5 models with special settings
    is_gpt5 = "gpt-5" in model_name.lower()
    if is_gpt5:
        if temperature is None:
            temperature = 1.0
        if max_tokens is None:
            max_tokens = 20000

    # Build LM parameters: provider-specific credentials + common params
    lm_params = {
        "model": model_name,
        "cache": cache,
        **_get_provider_params(model_name),
        **kwargs,
    }

    if temperature is not None:
        lm_params["temperature"] = temperature
    if max_tokens is not None:
        lm_params["max_tokens"] = max_tokens

    return dspy.LM(**lm_params)


# Initialize main language model
lm = create_lm(DEFAULT_MODEL_NAME)

# Initialize reflection language model (used by GEPA and similar optimizers)
reflectin_lm = create_lm(REFLECTION_MODEL_NAME, temperature=1.0, max_tokens=20000)

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