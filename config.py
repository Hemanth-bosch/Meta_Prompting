import dspy
from typing import Optional
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Load Azure OpenAI credentials once
_dotenv_file = dotenv.find_dotenv()
AZURE_API_KEY = dotenv.get_key(_dotenv_file, "AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = dotenv.get_key(_dotenv_file, "AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = dotenv.get_key(_dotenv_file, "AZURE_OPENAI_API_VERSION")
DEFAULT_MODEL_NAME = dotenv.get_key(_dotenv_file, "MODEL_NAME")
REFLECTION_MODEL_NAME = dotenv.get_key(_dotenv_file, "REFLECTION_LM_NAME")


def create_lm(model_name: Optional[str] = None,
              temperature: Optional[float] = None,
              max_tokens: Optional[int] = None,
              cache: bool = True,
              **kwargs) -> dspy.LM:
    """
    Create a DSPy LM instance with Azure OpenAI credentials.

    Args:
        model_name: Model name (defaults to MODEL_NAME from .env)
        temperature: Temperature setting (auto-set to 1.0 for gpt-5 if not specified)
        max_tokens: Max tokens (auto-set to 20000 for gpt-5 if not specified)
        cache: Enable/disable response caching (default: True, matches dspy.LM default)
        **kwargs: Additional parameters for dspy.LM

    Returns:
        dspy.LM instance configured with Azure OpenAI

    Examples:
        >>> lm = create_lm()  # Uses MODEL_NAME from .env with cache enabled
        >>> lm = create_lm("gpt-4o")  # Custom model
        >>> lm = create_lm("gpt-4o", temperature=0.7)  # Custom params
        >>> lm = create_lm("gpt-4o", cache=False)  # Disable caching for debugging
    """
    # Use default model if not specified
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME

    # Auto-configure gpt-5 models with special settings
    is_gpt5 = "gpt-5" in model_name.lower()
    if is_gpt5:
        if temperature is None:
            temperature = 1.0
        if max_tokens is None:
            max_tokens = 20000

    # Build LM parameters
    lm_params = {
        "model": model_name,
        "api_key": AZURE_API_KEY,
        "api_base": AZURE_ENDPOINT,
        "api_version": AZURE_API_VERSION,
        "cache": cache,
        **kwargs
    }

    # Add optional parameters if specified
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
        model_name: Name of the model to use (e.g., 'gpt-4o', 'gpt-4-turbo')
        **kwargs: Additional parameters for the LM (temperature, max_tokens, etc.)

    Returns:
        The newly configured LM instance

    Examples:
        >>> reconfigure_lm("gpt-4o")
        >>> reconfigure_lm("gpt-4-turbo", temperature=0.5)
    """
    global lm

    # Create new LM with specified model
    lm = create_lm(model_name, **kwargs)

    # Update DSPy configuration
    dspy.configure(lm=lm)

    return lm
