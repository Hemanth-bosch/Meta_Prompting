"""[wrapper-only] LM / runtime configuration -- no structural run.py analog.

RELATIONSHIP TO run.py (MBD/weakspot_code_to_run_2/):
    This module conceptually parallels the original MIDAS MBD pair
    llm_client.py + config.py (the Azure client, az-login auth, CA bundle,
    temperature 0.0 defaults), but there is NO one-to-one run.py analog:
    in run.py the pipeline OWNS the transport (llm_client._make_azure_client
    builds an AsyncAzureOpenAI via AzureCliCredential and call_llm hardcodes
    temperature=0.0 with no response_format), whereas HERE DSPy owns the LM.

    (corresponds-to) run.py/config.py::_Config
        AZURE_ENDPOINT        (~) cfg.AZURE_OPENAI_ENDPOINT
        AZURE_API_VERSION     (~) cfg.AZURE_OPENAI_API_VERSION
        REQUESTS_CA_BUNDLE    (~) cfg.REQUESTS_CA_BUNDLE
        (run.py's AZURE_OPENAI_DEPLOYMENT is the model/deployment name; here
         that role is played by DEFAULT_MODEL_NAME / the model_name argument.)

    (corresponds-to) run.py/llm_client.py::_make_azure_client
        _get_azure_client + _azure_cli_token_provider (~) AzureCliCredential
        token provider; _build_ca_bundle (~) llm_client's certifi/CA verify.

    The DSPy LM built via create_lm() / reconfigure_lm() REPLACES run.py's
    per-call `resp = await call_llm(prompt)` transport so that GEPA can drive
    and trace each predictor. The pipeline flow itself (Phase 1/2 precompute,
    Phase 3a/3b/4a + post-process) lives in extract_mbd_full.py, not here;
    the frozen run.py prompt templates live in signatures/mbd.py.

    [wrapper-only] additions with no run.py analog also include: multi-provider
    dispatch (azure / vertex_ai / openai / anthropic), per-model max_tokens
    caps, gpt-5 temperature auto-config, DSPy response caching, and the
    separate reflection LM used by GEPA-style optimizers.
"""

# [wrapper-only] runtime deps: dspy owns the LM (replaces run.py's llm_client.call_llm); rest support az-login/CA-bundle plumbing.
import dspy
from typing import Optional
import dotenv                                                                     # [wrapper-only] .env loader (~) config.py:25 from dotenv import load_dotenv
import os
from datetime import datetime                                                    # [wrapper-only] used by _parse_azure_cli_expiry (token TTL parse)
import json
from pathlib import Path
import subprocess                                                                # [wrapper-only] shells out to `az` CLI (run.py uses AzureCliCredential in-process)
import shutil                                                                     # [wrapper-only] shutil.which() to locate the `az` entrypoint
import sys
from threading import Lock                                                        # [wrapper-only] guards the module-level az-token cache
import time                                                                       # [wrapper-only] token expiry clock for the cache

import certifi                                                                    # [wrapper-only] default CA bundle (~) llm_client.py:77 import certifi
from openai import AzureOpenAI, DefaultHttpxClient                                # [wrapper-only] raw fallback client (~) llm_client.py:33 AsyncAzureOpenAI (DSPy normally owns transport)

# Load environment variables
dotenv.load_dotenv()                                                             # (~) config.py:31-32 load_dotenv() — populate os.environ from .env

# Load configuration from .env
_dotenv_file = dotenv.find_dotenv()                                              # [wrapper-only] locate .env for get_key() reads below

# Azure OpenAI credentials
# (corresponds-to) run.py/config.py:44-48 _Config Azure block (endpoint/version).
AZURE_API_KEY = dotenv.get_key(_dotenv_file, "AZURE_OPENAI_API_KEY")             # [wrapper-only] api-key path; run.py uses az-login only (no key var)
AZURE_ENDPOINT = dotenv.get_key(_dotenv_file, "AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT", "https://aiq212-openai.openai.azure.com/")  # (~) config.py:45 AZURE_OPENAI_ENDPOINT
AZURE_API_VERSION = dotenv.get_key(_dotenv_file, "AZURE_OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")  # (~) config.py:46 AZURE_OPENAI_API_VERSION (same default "2024-08-01-preview")

# Vertex AI credentials
# [wrapper-only] multi-provider support — no run.py analog (run.py is Azure-only).
VERTEX_PROJECT = dotenv.get_key(_dotenv_file, "VERTEX_PROJECT")                  # [wrapper-only] Vertex AI project id
VERTEX_LOCATION = dotenv.get_key(_dotenv_file, "VERTEX_LOCATION")                # [wrapper-only] Vertex AI region

# Direct API keys (for non-Azure usage)
# [wrapper-only] direct OpenAI/Anthropic paths — no run.py analog (run.py is Azure-only).
OPENAI_API_KEY = dotenv.get_key(_dotenv_file, "OPENAI_API_KEY")                  # [wrapper-only] OpenAI-direct key
ANTHROPIC_API_KEY = dotenv.get_key(_dotenv_file, "ANTHROPIC_API_KEY")            # [wrapper-only] Anthropic-direct key

# Optional CA bundle override for corporate TLS-inspecting environments
REQUESTS_CA_BUNDLE = dotenv.get_key(_dotenv_file, "REQUESTS_CA_BUNDLE") or os.getenv("REQUESTS_CA_BUNDLE")  # (~) config.py:48 REQUESTS_CA_BUNDLE (corporate TLS bundle)

# Model names
DEFAULT_MODEL_NAME = dotenv.get_key(_dotenv_file, "MODEL_NAME")                  # (~) config.py:47 AZURE_OPENAI_DEPLOYMENT role — the model/deployment name
REFLECTION_MODEL_NAME = dotenv.get_key(_dotenv_file, "REFLECTION_LM_NAME")       # [wrapper-only] separate reflection LM for GEPA — no run.py analog

DCA_JUDGE_MODEL_NAME = "gpt-4.1"  # LLM judge used in dca_metric.judge_summary_metric


# [wrapper-only] CA bundle resolution -- (~) llm_client._make_azure_client's certifi/REQUESTS_CA_BUNDLE verify; no run.py analog for the combined-bundle export.
def _build_ca_bundle() -> str:
    explicit = os.getenv("AOAI_CA_BUNDLE") or os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")  # (~) llm_client.py:79 cfg.REQUESTS_CA_BUNDLE — honor an explicit override first
    if explicit:                                           # (~) llm_client.py:80 use ca if it exists...
        explicit_path = Path(explicit)                     # [wrapper-only] resolve the override to a Path
        if explicit_path.exists():                         # (~) llm_client.py:80 Path(ca).exists() gate
            return str(explicit_path)                      # [wrapper-only] override is valid → use it verbatim
        print(f"Warning: CA bundle path from env var not found: {explicit}. Falling back to certifi/combined bundle.")  # [wrapper-only] warn + fall through to certifi

    exported_dir = Path(os.getenv("TEMP", "")) / "bosch-ca"  # [wrapper-only] Bosch-specific dir of exported corporate .pem certs
    exported_pems = sorted(exported_dir.glob("*.pem")) if exported_dir.is_dir() else []  # [wrapper-only] gather any exported corporate certs

    if not exported_pems:                                  # (~) llm_client.py:80 else branch → certifi.where()
        return certifi.where()                             # (~) llm_client.py:80 certifi.where() default bundle

    out_path = exported_dir / "combined-ca-bundle.pem"     # [wrapper-only] where the merged bundle is written
    certifi_path = Path(certifi.where())                   # [wrapper-only] base certifi bundle to prepend

    parts: list[Path] = []                                 # [wrapper-only] accumulate bundle pieces in order
    if certifi_path.exists():                              # [wrapper-only] include certifi base if present
        parts.append(certifi_path)                         # [wrapper-only] certifi first
    parts.extend(exported_pems)                            # [wrapper-only] then the corporate certs

    combined = "\n".join(p.read_text(encoding="utf-8", errors="ignore").strip() for p in parts if p.exists())  # [wrapper-only] concat all PEM texts (no run.py analog — run.py never merges certs)
    combined = combined.strip() + "\n"                     # [wrapper-only] normalize trailing newline
    out_path.write_text(combined, encoding="utf-8")        # [wrapper-only] persist the merged bundle for httpx verify=
    return str(out_path)                                   # [wrapper-only] hand the merged bundle path back to the caller


# [wrapper-only] module-level az-login token cache + fallback client — no run.py analog (AzureCliCredential caches internally).
_azure_fallback_client = None                                                    # [wrapper-only] lazily-built raw AzureOpenAI (fallback path)
_azure_cli_token_cache: str | None = None                                        # [wrapper-only] last bearer token (reused until near expiry)
_azure_cli_token_expiry = 0.0                                                    # [wrapper-only] epoch seconds when the cached token expires
_azure_cli_token_lock = Lock()                                                   # [wrapper-only] serialize token refresh across DSPy worker threads


# [wrapper-only] locate the `az` CLI on PATH -- helper for the az-login token flow; no run.py analog (run.py uses AzureCliCredential in-process).
def _find_azure_cli_entrypoint() -> str:
    az_path = shutil.which("az") or shutil.which("az.cmd") or shutil.which("az.bat")  # [wrapper-only] resolve `az` on PATH (multiple Windows shims)
    if not az_path:                                        # [wrapper-only] `az` missing → cannot mint a token
        raise RuntimeError(                                # [wrapper-only] fail loud with install guidance
            "Azure CLI executable 'az' was not found on PATH. "
            "Install Azure CLI system-wide or run with AZURE_OPENAI_API_KEY configured."
        )
    az_path_obj = Path(az_path)                            # [wrapper-only] wrap for suffix inspection
    if os.name == "nt" and az_path_obj.suffix.lower() not in {".cmd", ".bat", ".exe"}:  # [wrapper-only] on Windows, `which` may return an extension-less shim
        for suffix in (".cmd", ".bat", ".exe"):            # [wrapper-only] try the real launcher extensions
            candidate = az_path_obj.with_suffix(suffix)    # [wrapper-only] candidate with a runnable extension
            if candidate.exists():                         # [wrapper-only] first existing launcher wins
                return str(candidate)                      # [wrapper-only] use the resolved launcher
    return az_path                                         # [wrapper-only] non-Windows / already-runnable path


# [wrapper-only] build the `az account get-access-token` command + env -- no run.py analog.
def _build_azure_cli_command() -> tuple[list[str], dict[str, str]]:
    az_path = _find_azure_cli_entrypoint()                 # [wrapper-only] resolved `az` launcher
    env = os.environ.copy()                                # [wrapper-only] child env to tweak before spawning `az`

    python_dir = str(Path(sys.executable).resolve().parent)  # [wrapper-only] this venv's dir (az needs a python on PATH)
    path_parts = env.get("PATH", "").split(os.pathsep) if env.get("PATH") else []  # [wrapper-only] split current PATH entries
    if not any(os.path.normcase(part) == os.path.normcase(python_dir) for part in path_parts if part):  # [wrapper-only] only prepend if not already present
        env["PATH"] = os.pathsep.join([python_dir, *[part for part in path_parts if part]])  # [wrapper-only] prepend venv python so `az` finds an interpreter

    env.pop("PYTHONPATH", None)                             # [wrapper-only] drop PYTHONPATH so az's bundled python isn't polluted

    # [wrapper-only] `az account get-access-token` for the cognitiveservices resource (mints the Azure OpenAI bearer token).
    command = [
        az_path,
        "account",
        "get-access-token",
        "--resource",
        "https://cognitiveservices.azure.com",             # (~) llm_client.py:85 credential.get_token("https://cognitiveservices.azure.com/.default")
        "--output",
        "json",
    ]

    return command, env                                    # [wrapper-only] hand argv + env to the subprocess runner


# [wrapper-only] parse the token expiry timestamp from the `az` CLI JSON payload -- no run.py analog.
def _parse_azure_cli_expiry(payload: dict) -> float:
    expires_on = payload.get("expires_on") or payload.get("expiresOn")  # [wrapper-only] az emits either snake- or camel-case key
    if expires_on is None:                                 # [wrapper-only] no expiry field → caller falls back to now+300
        return 0.0

    if isinstance(expires_on, (int, float)):               # [wrapper-only] already an epoch timestamp
        return float(expires_on)

    if isinstance(expires_on, str):                        # [wrapper-only] string form needs parsing
        stripped = expires_on.strip()                      # [wrapper-only] trim whitespace
        if stripped.isdigit():                             # [wrapper-only] all-digits → epoch seconds as text
            return float(stripped)
        # [wrapper-only] try each known az datetime format until one parses.
        for fmt in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
        ):
            try:                                           # [wrapper-only] attempt this format
                return datetime.strptime(stripped, fmt).timestamp()  # [wrapper-only] parsed → epoch seconds
            except ValueError:                             # [wrapper-only] format mismatch → try next
                continue

    return 0.0                                             # [wrapper-only] unparseable → signal "unknown expiry"


# [wrapper-only] run the `az` CLI subprocess and extract the bearer token -- no run.py analog.
def _fetch_azure_cli_token() -> tuple[str, float]:
    command, env = _build_azure_cli_command()              # [wrapper-only] argv + env for `az account get-access-token`
    result = subprocess.run(command, capture_output=True, text=True, env=env, check=False)  # [wrapper-only] run `az` out-of-process (run.py calls AzureCliCredential in-process instead)
    if result.returncode != 0:                             # [wrapper-only] az failed (not logged in, etc.)
        stderr = (result.stderr or "").strip()             # [wrapper-only] capture stderr for the message
        stdout = (result.stdout or "").strip()             # [wrapper-only] capture stdout as fallback detail
        details = stderr or stdout or f"Azure CLI exited with code {result.returncode}."  # [wrapper-only] best available error detail
        raise RuntimeError(                                # [wrapper-only] surface az failure with `az login` guidance
            "Azure CLI token retrieval failed. "
            "Run `az login` manually or configure AZURE_OPENAI_API_KEY. "
            f"Details: {details}"
        )

    payload = json.loads(result.stdout)                    # [wrapper-only] parse the JSON token blob from `az`
    token = payload.get("accessToken") or payload.get("access_token")  # [wrapper-only] bearer token (~) llm_client.py:86 token.token; az uses camelCase
    if not token:                                          # [wrapper-only] defensive: az returned no token field
        raise RuntimeError("Azure CLI token retrieval returned no access token.")  # [wrapper-only] fail loud

    return token, _parse_azure_cli_expiry(payload)         # [wrapper-only] token + parsed expiry for the cache


# [wrapper-only] cached az-login bearer-token callback -- (~) llm_client._token_provider (AzureCliCredential.get_token); here it feeds dspy.LM's azure_ad_token_provider instead of AsyncAzureOpenAI.
def _azure_cli_token_provider() -> str:
    """Return Azure AD bearer token for Azure OpenAI.

    This callback is intentionally module-level (picklable-friendly) so DSPy
    can deepcopy LM instances during optimizers like MIPROv2.
    """
    global _azure_cli_token_cache, _azure_cli_token_expiry  # [wrapper-only] mutate the module-level token cache

    with _azure_cli_token_lock:                            # [wrapper-only] serialize refresh across DSPy worker threads
        now = time.time()                                  # [wrapper-only] current clock for the expiry check
        if _azure_cli_token_cache and now < (_azure_cli_token_expiry - 300):  # [wrapper-only] reuse cached token until 5 min before expiry
            return _azure_cli_token_cache                  # [wrapper-only] cache hit — skip the az subprocess

        token, expires_at = _fetch_azure_cli_token()       # [wrapper-only] cache miss → mint a fresh token via `az`
        _azure_cli_token_cache = token                     # [wrapper-only] store the new token
        _azure_cli_token_expiry = expires_at or (now + 300)  # [wrapper-only] use parsed expiry, else a 5-min TTL
        return token                                       # [wrapper-only] hand the fresh token to dspy.LM's azure_ad_token_provider


# [wrapper-only] build a raw AzureOpenAI client (key or az-login token) -- (~) llm_client._make_azure_client, but DSPy normally owns transport so this is a fallback path only.
def _get_azure_client(auth_mode: str = "cli"):            # (~) llm_client.py:74 _make_azure_client() — but a fallback, DSPy owns transport
    global _azure_fallback_client                          # [wrapper-only] memoize the built client module-wide
    if _azure_fallback_client is not None:                 # [wrapper-only] return the cached client if already built
        return _azure_fallback_client

    ca_bundle = _build_ca_bundle()                         # (~) llm_client.py:79-80 resolve the CA bundle for httpx verify=
    print(f"Using CA bundle: {ca_bundle}")                 # [wrapper-only] trace which bundle was chosen

    if auth_mode == "key":                                 # [wrapper-only] api-key auth path (run.py has no key path — az-login only)
        print("Creating Azure OpenAI client with direct API Key...")  # [wrapper-only] trace
        # [wrapper-only] key-auth AzureOpenAI (~) llm_client.py:88-93 AsyncAzureOpenAI, but sync + api_key instead of token provider.
        _azure_fallback_client = AzureOpenAI(
            api_key=AZURE_API_KEY,                          # [wrapper-only] direct key (no llm_client analog)
            azure_endpoint=AZURE_ENDPOINT,                 # (~) llm_client.py:89 azure_endpoint=cfg.AZURE_OPENAI_ENDPOINT
            api_version=AZURE_API_VERSION,                  # (~) llm_client.py:90 api_version=cfg.AZURE_OPENAI_API_VERSION
            http_client=DefaultHttpxClient(verify=ca_bundle),  # (~) llm_client.py:92 http_client=httpx client with verify=ca
        )
    else:                                                  # az-login token path
        print("Initializing Azure CLI credential...")      # [wrapper-only] trace

        print("Creating token provider...")                # [wrapper-only] trace
        def token_provider():                              # (~) llm_client.py:84 _token_provider() nested callback
            return _azure_cli_token_provider()             # [wrapper-only] delegate to the cached az-login provider

        print("Creating Azure OpenAI client with token provider...")  # [wrapper-only] trace
        # [wrapper-only] token-auth AzureOpenAI (~) llm_client.py:88-93 AsyncAzureOpenAI (sync here; DSPy normally owns transport).
        _azure_fallback_client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,                 # (~) llm_client.py:89 azure_endpoint=cfg.AZURE_OPENAI_ENDPOINT
            azure_ad_token_provider=token_provider,        # (~) llm_client.py:91 azure_ad_token_provider=_token_provider
            api_version=AZURE_API_VERSION,                 # (~) llm_client.py:90 api_version=cfg.AZURE_OPENAI_API_VERSION
            http_client=DefaultHttpxClient(verify=ca_bundle),  # (~) llm_client.py:92 http_client with verify=ca
        )

    print("Azure Client created successfully.")            # [wrapper-only] trace
    return _azure_fallback_client                          # [wrapper-only] hand back (and cache) the raw fallback client


# [wrapper-only] multi-provider dispatch from the model-name prefix -- no run.py analog (run.py is Azure-only via AZURE_OPENAI_DEPLOYMENT).
def _get_provider(model_name: str) -> str:
    """Extract the provider from a model name prefix."""
    if "/" in model_name:                                  # [wrapper-only] "azure/gpt-4o" style prefix present
        return model_name.split("/")[0]                    # [wrapper-only] provider == text before the first "/"

    if OPENAI_API_KEY:                                     # [wrapper-only] no prefix → infer from which creds are set
        return "openai"                                    # [wrapper-only] OpenAI key present → OpenAI direct
    if AZURE_API_KEY or AZURE_ENDPOINT:                    # [wrapper-only] Azure creds present (run.py is Azure-only)
        return "azure"                                     # [wrapper-only] default to Azure when only Azure is configured
    return "openai"                                        # [wrapper-only] last-resort default provider


# [wrapper-only] per-provider credential params for dspy.LM -- no run.py analog (only azure/* mirrors run.py's endpoint/version).
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
    provider = _get_provider(model_name)                  # [wrapper-only] pick a branch from the model-name prefix

    if provider == "azure":                               # azure/* branch mirrors run.py's Azure creds
        return {                                           # (~) llm_client.py:88-90 endpoint/version params (run.py has no api_key path)
            "api_key": AZURE_API_KEY,                      # [wrapper-only] direct key (no llm_client analog — run.py uses az-login)
            "api_base": AZURE_ENDPOINT,                    # (~) llm_client.py:89 azure_endpoint=cfg.AZURE_OPENAI_ENDPOINT
            "api_version": AZURE_API_VERSION,              # (~) llm_client.py:90 api_version=cfg.AZURE_OPENAI_API_VERSION
        }
    elif provider == "vertex_ai":                         # [wrapper-only] Vertex AI (Gemini/Claude) — no run.py analog
        params = {}                                        # [wrapper-only] collect only the creds that are set
        if VERTEX_PROJECT:                                 # [wrapper-only] optional GCP project id
            params["vertex_project"] = VERTEX_PROJECT      # [wrapper-only] litellm vertex_project param
        if VERTEX_LOCATION:                                # [wrapper-only] optional GCP region
            params["vertex_location"] = VERTEX_LOCATION    # [wrapper-only] litellm vertex_location param
        return params                                      # [wrapper-only] hand vertex creds to dspy.LM
    elif provider == "openai":                             # [wrapper-only] OpenAI direct — no run.py analog
        params = {}                                        # [wrapper-only] optional key only
        if OPENAI_API_KEY:                                 # [wrapper-only] pass key when configured
            params["api_key"] = OPENAI_API_KEY             # [wrapper-only] OpenAI api_key
        return params                                      # [wrapper-only] hand OpenAI creds to dspy.LM
    elif provider == "anthropic":                          # [wrapper-only] Anthropic direct — no run.py analog
        params = {}                                        # [wrapper-only] optional key only
        if ANTHROPIC_API_KEY:                              # [wrapper-only] pass key when configured
            params["api_key"] = ANTHROPIC_API_KEY          # [wrapper-only] Anthropic api_key
        return params                                      # [wrapper-only] hand Anthropic creds to dspy.LM
    else:                                                  # unknown provider
        # Unknown provider — pass no extra params, let litellm handle it
        return {}                                          # [wrapper-only] empty dict → litellm resolves creds itself


# [wrapper-only] per-model max_tokens caps -- no run.py analog (run.py call_llm sets no max_tokens).
def _get_default_max_tokens(model_name: str) -> int:
    """Return a safe default max_tokens for the given model, staying within provider limits."""
    name_lower = model_name.lower()                       # [wrapper-only] case-insensitive matching
    # Strip provider prefix for matching (e.g. "azure/gpt-4o-mini" → "gpt-4o-mini")
    base = name_lower.split("/", 1)[-1] if "/" in name_lower else name_lower  # [wrapper-only] drop provider prefix before matching

    # Model-specific limits (output token limits from provider docs)
    # [wrapper-only] per-model output caps; run.py's call_llm sets NO max_tokens
    if "gpt-5" in base:                                   # [wrapper-only] gpt-5 output cap
        return 20000                                       # [wrapper-only] 20k tokens
    if "gpt-4.1" in base:                                 # [wrapper-only] gpt-4.1 output cap
        return 16384                                       # [wrapper-only] 16k tokens
    if "gpt-4o-mini" in base:                             # [wrapper-only] gpt-4o-mini output cap
        return 16384                                       # [wrapper-only] 16k tokens
    if "gpt-4o" in base:                                  # [wrapper-only] gpt-4o output cap
        return 16384                                       # [wrapper-only] 16k tokens
    if "o3" in base or "o4" in base:                      # [wrapper-only] o3/o4 reasoning models
        return 16384  # reasoning models
    if "gemini" in base:                                  # [wrapper-only] Gemini output cap
        return 8192                                        # [wrapper-only] 8k tokens
    if "claude" in base:                                  # [wrapper-only] Claude output cap
        return 8192                                        # [wrapper-only] 8k tokens
    # Conservative fallback
    return 8192                                            # [wrapper-only] safe default when no model matched


# [wrapper-only] build the dspy.LM that REPLACES run.py's per-call `await call_llm(prompt)` transport (so GEPA can drive/trace predictors); azure/* branch (~) llm_client._make_azure_client (endpoint/version + az-login token or api_key). No run.py analog.
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
    if model_name is None:                                # [wrapper-only] default to .env MODEL_NAME (run.py role: AZURE_OPENAI_DEPLOYMENT)
        model_name = DEFAULT_MODEL_NAME                    # [wrapper-only] fall back to the configured default

    provider = _get_provider(model_name)                  # [wrapper-only] resolve provider from prefix/creds

    if "/" not in model_name:                             # [wrapper-only] add a litellm provider prefix if bare
        if provider == "azure":                           # [wrapper-only] azure/* is litellm's Azure route
            model_name = f"azure/{model_name}"            # [wrapper-only] prefix the deployment name
        elif provider == "openai":                        # [wrapper-only] openai/* is litellm's OpenAI route
            model_name = f"openai/{model_name}"           # [wrapper-only] prefix the model name

    # Auto-select max_tokens based on model limits if not explicitly provided
    if max_tokens is None:                                # [wrapper-only] no run.py analog (call_llm sets no max_tokens)
        max_tokens = _get_default_max_tokens(model_name)  # [wrapper-only] derive a safe per-model cap

    # Auto-configure gpt-5 models with special settings
    is_gpt5 = "gpt-5" in model_name.lower()               # [wrapper-only] detect gpt-5 (needs temperature==1.0)
    if is_gpt5:                                            # [wrapper-only] gpt-5 only supports the default temperature
        if temperature is None:                           # [wrapper-only] only override when caller left it unset
            temperature = 1.0                             # [wrapper-only] gpt-5 requires temperature=1.0

    # Build LM parameters: provider-specific credentials + common params
    lm_params = {                                          # [wrapper-only] kwargs for dspy.LM (replaces run.py call_llm transport)
        "model": model_name,                              # [wrapper-only] litellm model id (~) cfg.AZURE_OPENAI_DEPLOYMENT
        "cache": cache,                                    # [wrapper-only] DSPy response cache toggle (no run.py analog)
        "num_retries": 4,                                 # [wrapper-only] retry transient Azure ConnectTimeout (WinError 10060) instead of failing the example
        "timeout": 300,                                   # [wrapper-only] per-request timeout (s) — a stalled connection fails fast then retries
        **kwargs,                                          # [wrapper-only] pass through any extra caller params (can override num_retries/timeout)
    }

    if provider == "azure":                               # azure/* branch (~) llm_client._make_azure_client
        if AZURE_API_KEY:                                 # [wrapper-only] api-key auth (run.py has no key path)
            lm_params.update({                            # [wrapper-only] api-key credentials for dspy.LM
                "api_key": AZURE_API_KEY,                 # [wrapper-only] direct key (no llm_client analog)
                "api_base": AZURE_ENDPOINT,               # (~) llm_client.py:89 azure_endpoint=cfg.AZURE_OPENAI_ENDPOINT
                "api_version": AZURE_API_VERSION,         # (~) llm_client.py:90 api_version=cfg.AZURE_OPENAI_API_VERSION
                "cache": cache,                           # [wrapper-only] respect the cache arg — key auth uses a string key (dspy ignores it in the cache key), so caching is safe (no unhashable az-login token provider)
            })
        else:                                             # az-login token path (~) llm_client.py:82-91 AzureCliCredential
            ca_bundle = _build_ca_bundle()                # (~) llm_client.py:79-80 resolve the CA bundle for verify=
            os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)  # [wrapper-only] point litellm/httpx at the CA bundle
            os.environ.setdefault("SSL_CERT_FILE", ca_bundle)       # [wrapper-only] OpenSSL CA path for the same bundle
            lm_params.update({                            # [wrapper-only] az-login token credentials for dspy.LM
                "api_base": AZURE_ENDPOINT,               # (~) llm_client.py:89 azure_endpoint=cfg.AZURE_OPENAI_ENDPOINT
                "api_version": AZURE_API_VERSION,         # (~) llm_client.py:90 api_version=cfg.AZURE_OPENAI_API_VERSION
                "azure_ad_token_provider": _azure_cli_token_provider,  # (~) llm_client.py:91 azure_ad_token_provider=_token_provider
                "cache": False,                           # [wrapper-only] force cache off on the token path
            })
    else:                                                 # non-Azure provider (no run.py analog)
        lm_params.update(_get_provider_params(model_name))  # [wrapper-only] merge vertex/openai/anthropic creds

    if temperature is not None:                           # [wrapper-only] run.py call_llm hardcodes temperature=0.0; here it's per-LM
        lm_params["temperature"] = temperature            # [wrapper-only] apply resolved temperature
    if max_tokens is not None:                            # [wrapper-only] apply the per-model output cap
        lm_params["max_tokens"] = max_tokens              # [wrapper-only] set dspy.LM max_tokens

    return dspy.LM(**lm_params)                            # [wrapper-only] the LM that REPLACES run.py's per-call call_llm(prompt)


# Initialize main language model
lm = create_lm(DEFAULT_MODEL_NAME)                        # [wrapper-only] the LM that drives every predictor — REPLACES run.py's call_llm(prompt) transport

# Initialize reflection language model (used by GEPA and similar optimizers)
# [wrapper-only] `reflectin_lm` (real, misspelled name) is GEPA's REFLECTION LM: it reads
# traces and rewrites instructions during optimization; temperature=1.0 for diverse proposals.
# No run.py analog — run.py never optimizes prompts, so it has no reflection model.
reflectin_lm = create_lm(REFLECTION_MODEL_NAME, temperature=1.0)  # [wrapper-only] GEPA reflection LM (temp 1.0 for proposal diversity)

# Configure DSPy with the main LM
dspy.configure(lm=lm)                                     # [wrapper-only] register the main LM as DSPy's global default (no run.py analog)


# [wrapper-only] swap the active DSPy LM at runtime (dspy.configure) -- no run.py analog; run.py has no notion of a globally-configured optimizable LM.
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