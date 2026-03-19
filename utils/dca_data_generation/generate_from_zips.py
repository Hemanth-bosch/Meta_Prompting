"""
generate_dca_dataset.py
=======================
DCA Dataset Generation Pipeline
--------------------------------
Reads image pairs from image_matches_*.txt files inside extracted zip archives,
sends each pair to Azure OpenAI GPT-4.1 for visual comparison, and saves
structured JSON outputs alongside the copied images.

All outputs (extracted zips, images, JSONs, prompt, log) live under dca_data/.

txt file line formats
----------------------
Two images:   sessions/.../v1/img.png, sessions/.../v2/img.png
Old only:     sessions/.../v1/img.png, 0      ← image was deleted
New only:     0, sessions/.../v2/img.png       ← image was added

When one side is '0', the single real image is still sent to the LLM.
The prompt handles this case with its "If only one image is provided..." rules.

Usage
------
  1. Create a .env file in the same directory as this script:
       AZURE_OPENAI_ENDPOINT=https://<your-endpoint>.openai.azure.com/
       AZURE_OPENAI_API_VERSION=2024-08-01-preview
       AZURE_OPENAI_DEPLOYMENT=gpt-4.1
       # Optional: AOAI_CA_BUNDLE=C:/path/to/bundle.pem
  2. Log in via Azure CLI BEFORE activating the venv:
       deactivate
       az login
       .venv\Scripts\activate
       python generate_dca_dataset.py

Dependencies
-------------
  pip install openai azure-identity certifi python-dotenv
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────────────────────────────────────
import base64
import json
import logging
import os
import re
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Third-party  (pip install openai azure-identity certifi python-dotenv)
# ─────────────────────────────────────────────────────────────────────────────
import certifi
from azure.identity import AzureCliCredential, get_bearer_token_provider
from azure.identity._exceptions import CredentialUnavailableError
from dotenv import load_dotenv
from openai import AzureOpenAI, DefaultHttpxClient

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# PATH CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DATA_DIR: Path = Path(__file__).resolve().parents[2] / "data"

DCA_DATA_DIR:    Path = BASE_DATA_DIR / "dca_data"
IMAGE_PAIRS_DIR: Path = DCA_DATA_DIR / "image_pairs"
PROMPT_FILE:     Path = DCA_DATA_DIR / "comparison_prompt.md"
LOG_FILE:        Path = DCA_DATA_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# ── Experiment & model tag (used in LLM output folder name) ──────────────────
EXPERIMENT_NAME: str = "exp1"
MODEL_NAME:      str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
LLM_OUTPUT_DIR:  Path = DCA_DATA_DIR / f"image_pairs_llm_output_{MODEL_NAME}_{EXPERIMENT_NAME}"


# ─────────────────────────────────────────────────────────────────────────────
# ★ CONFIGURABLE ZIP LIST
# ─────────────────────────────────────────────────────────────────────────────
# Edit this list to control which zip files are processed and in what order.
# Each entry is the zip filename (relative to DCA_DATA_DIR).
# Examples:
#   ZIP_FILES = ["pdf_1.zip"]                       # single zip
#   ZIP_FILES = ["pdf_1.zip", "pdf_3.zip"]          # cherry-pick
#   ZIP_FILES = [f"pdf_{i}.zip" for i in range(1, 7)]  # all six

ZIP_FILES: List[str] = [
    # "pdf_1.zip",
    # "pdf_2.zip",
    # "pdf_3.zip",
    "pdf_4.zip",
    # "pdf_5.zip",
    "pdf_6.zip",
]


# Value used in txt file to indicate "no image on this side of the pair"
NO_IMAGE_SENTINEL: str = "0"


# ─────────────────────────────────────────────────────────────────────────────
# LLM COMPARISON PROMPT
# ─────────────────────────────────────────────────────────────────────────────

COMPARISON_PROMPT: str = """\
You are given two images. Compare them pixel-by-pixel and report ONLY the actual visual differences you can see.

CRITICAL ANTI-HALLUCINATION RULES (READ FIRST)
1) Do NOT invent changes. If images look the same, they ARE the same.
2) Do NOT assume spelling corrections or intent. If a word/spelling appears the SAME in both images, report NO change.
3) Report WHAT changed only if you can CLEARLY see different characters/words in the same location.
4) If you cannot unambiguously see a difference, default to NO CHANGE.

VERIFICATION CHECKLIST (must pass BEFORE reporting a change)
- Locate the SAME region in both images.
- Read the text EXACTLY from image 1 and EXACTLY from image 2.
- If the strings are identical → NO CHANGE.
- If they differ → quote BOTH strings verbatim in your answer.

STRICT OUTPUT POLICY
- Default to: {"discard_image": true, "changes": "Images are identical - no changes"} unless a difference is unequivocal.
- When describing text differences, use: "Text changed from '[exact original]' to '[exact new]'". No opinions.
- For punctuation-only differences, output: {"discard_image": true, "changes": "NoMatchFound"}.

Other Rules
- If only one image is provided, determine if it contains a watermark or a logo (full or partial).
- If the image is a watermark or a logo (even partially), set "discard_image" to true. Otherwise, set it to false.
- If the image is just a block/rectangle with no information, or only a line, set "discard_image" to true.
- If the image is just a table with tabular content, set "discard_image" to true.
- If only one image is provided and it is not a watermark or logo, set "changes" to "NoMatchFound" and "discard_image" to false.
- If any of the images seem like a section number (e.g., 5.2.1 or 1.2.4), set "discard_image" to true and "changes" to "NoMatchFound".
- If the image just has plain text and nothing else, set "discard_image" to true and "changes" to "NoMatchFound".
- Ignore differences in resolution, size, quality, or watermark presence.
- Image 1 is the original version. Image 2 is the updated version.
- If image 2 has a cross mark indicating deletion, mention it in the changes.

Generate ONLY JSON in the following format:
{"discard_image": true/false, "changes": "detailed explanation here"}
"""

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dca_pipeline")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger: logging.Logger = setup_logging(LOG_FILE)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT FILE
# ─────────────────────────────────────────────────────────────────────────────

def save_prompt_file() -> None:
    PROMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_FILE.write_text(
        f"# Image Comparison Prompt\n\nUsed verbatim in every LLM call.\n\n```\n{COMPARISON_PROMPT}\n```\n",
        encoding="utf-8",
    )
    logger.info(f"Prompt file saved → {PROMPT_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# CA BUNDLE
# ─────────────────────────────────────────────────────────────────────────────

def _build_ca_bundle() -> str:
    for var in ("AOAI_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        val = os.getenv(var)
        if val:
            p = Path(val)
            if p.exists():
                logger.debug(f"CA bundle from {var}: {p}")
                return str(p)
            logger.warning(f"CA bundle path in {var} not found: {val}. Falling back.")

    bosch_dir = Path(os.getenv("TEMP", "")) / "bosch-ca"
    pems: List[Path] = sorted(bosch_dir.glob("*.pem")) if bosch_dir.is_dir() else []
    if not pems:
        return certifi.where()

    combined_path = bosch_dir / "combined-ca-bundle.pem"
    parts = [Path(certifi.where())] + pems
    combined_path.write_text(
        "\n".join(p.read_text(encoding="utf-8", errors="ignore").strip() for p in parts if p.exists()) + "\n",
        encoding="utf-8",
    )
    logger.debug(f"Combined CA bundle → {combined_path}")
    return str(combined_path)


# ─────────────────────────────────────────────────────────────────────────────
# AZURE CLI CREDENTIAL — VENV-SAFE
# ─────────────────────────────────────────────────────────────────────────────

def _make_az_credential() -> AzureCliCredential:
    """
    Return an AzureCliCredential that uses the SYSTEM az CLI, not the venv one.

    When a venv is active, Python installs an `az.BAT` shim into
    `.venv/Scripts/` which shadows the real system `az.exe`.  We temporarily
    remove the venv Scripts directory from PATH before creating the credential.
    """
    venv = os.environ.get("VIRTUAL_ENV", "")
    if not venv:
        return AzureCliCredential()

    venv_scripts: str = os.path.join(venv, "Scripts")
    original_path: str = os.environ.get("PATH", "")

    filtered_path: str = os.pathsep.join(
        segment
        for segment in original_path.split(os.pathsep)
        if os.path.normcase(segment) != os.path.normcase(venv_scripts)
    )

    logger.debug(f"Temporarily removing venv Scripts from PATH for AzureCliCredential: {venv_scripts}")
    os.environ["PATH"] = filtered_path
    try:
        credential = AzureCliCredential()
    finally:
        os.environ["PATH"] = original_path
    logger.debug("PATH restored after AzureCliCredential creation.")
    return credential


# ─────────────────────────────────────────────────────────────────────────────
# AZ LOGIN PREFLIGHT CHECK
# ─────────────────────────────────────────────────────────────────────────────

def verify_az_login() -> None:
    """Abort early with a clear message if az login is missing."""
    logger.info("Verifying Azure CLI login (preflight check) …")
    credential = _make_az_credential()
    try:
        token = credential.get_token("https://cognitiveservices.azure.com/.default")
        expires_str = datetime.fromtimestamp(token.expires_on).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Azure CLI login verified ✓  Token expires: {expires_str}")
    except CredentialUnavailableError as exc:
        logger.error(
            "\n"
            "╔══════════════════════════════════════════════════════════════════╗\n"
            "║            AZURE CLI LOGIN REQUIRED — ABORTING                   ║\n"
            "╠══════════════════════════════════════════════════════════════════╣\n"
            "║  Run az login BEFORE activating the venv:                        ║\n"
            "║                                                                  ║\n"
            "║    deactivate                                                    ║\n"
            "║    az login                                                      ║\n"
            "║    .venv\\Scripts\\activate                                        ║\n"
            "║    python generate_dca_dataset.py                                ║\n"
            "╚══════════════════════════════════════════════════════════════════╝\n"
            f"\nUnderlying error: {exc}"
        )
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Unexpected error during Azure login check: {exc}", exc_info=True)
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# AZURE OPENAI CLIENT
# ─────────────────────────────────────────────────────────────────────────────

def build_openai_client() -> Tuple[AzureOpenAI, str, str, str]:
    """Returns (client, deployment, api_version, endpoint)."""
    endpoint    = os.getenv("AZURE_OPENAI_ENDPOINT",    "https://aiq212-openai.openai.azure.com/")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    deployment  = os.getenv("AZURE_OPENAI_DEPLOYMENT",  "gpt-4.1")

    logger.info(f"Building Azure OpenAI client | endpoint={endpoint} | deployment={deployment}")

    credential     = _make_az_credential()
    token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
    ca_bundle      = _build_ca_bundle()

    logger.info(f"Using CA bundle: {ca_bundle}")

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
        http_client=DefaultHttpxClient(verify=ca_bundle),
    )
    logger.info("Azure OpenAI client ready.")
    return client, deployment, api_version, endpoint


# ─────────────────────────────────────────────────────────────────────────────
# WINDOWS LONG-PATH HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _to_longpath(path: Path) -> str:
    """Prefix with \\\\?\\ so Windows accepts paths longer than 260 characters."""
    if os.name != "nt":
        return str(path)
    abs_str = os.path.abspath(str(path))
    if abs_str.startswith("\\\\?\\"):
        return abs_str
    if abs_str.startswith("\\\\"):
        return "\\\\?\\UNC\\" + abs_str[2:]
    return "\\\\?\\" + abs_str


# ─────────────────────────────────────────────────────────────────────────────
# ZIP EXTRACTION (long-path safe)
# ─────────────────────────────────────────────────────────────────────────────

def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    """
    Extract zip_path into extract_to/ using manual entry iteration so that
    every target path is prefixed with \\\\?\\ before any I/O call.
    """
    extract_to.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {zip_path.name} → {extract_to} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        entries = zf.infolist()
        logger.debug(f"  Archive has {len(entries)} entries.")
        for entry in entries:
            target = extract_to / Path(entry.filename)
            if entry.is_dir():
                os.makedirs(_to_longpath(target), exist_ok=True)
            else:
                os.makedirs(_to_longpath(target.parent), exist_ok=True)
                with open(_to_longpath(target), "wb") as fh:
                    fh.write(zf.read(entry.filename))
    logger.info(f"Extraction complete: {extract_to}")
    return extract_to


# ─────────────────────────────────────────────────────────────────────────────
# MATCHES FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def find_matches_file(folder: Path) -> Optional[Path]:
    candidates = list(folder.glob("image_matches_*.txt")) or list(folder.rglob("image_matches_*.txt"))
    if not candidates:
        logger.warning(f"[WARN] No image_matches_*.txt found in: {folder}")
        return None
    if len(candidates) > 1:
        logger.warning(f"[WARN] Multiple matches files found; using: {candidates[0]}")
    logger.info(f"Found matches file: {candidates[0]}")
    return candidates[0]


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PATH RESOLVER (WITH FALLBACK)
# ─────────────────────────────────────────────────────────────────────────────
def resolve_image_path(extracted_folder: Path, raw_path: str, line_no: int, side: str) -> Optional[Path]:
    """
    Attempts to find the file using the original path.
    If it fails, it strips the first folder (the session ID) and tries again.
    """
    if raw_path == NO_IMAGE_SENTINEL:
        return None

    # Strip 'sessions/' prefix if present
    clean_path = re.sub(r"^sessions/", "", raw_path.strip().replace("\\", "/"))

    # Attempt 1: Full path
    primary_path = extracted_folder / clean_path
    if os.path.exists(_to_longpath(primary_path)):
        return primary_path

    # Attempt 2: Fallback (remove the session ID / first directory segment)
    parts = clean_path.split("/")
    if len(parts) > 1:
        fallback_clean_path = "/".join(parts[1:])
        fallback_path = extracted_folder / fallback_clean_path

        # Also trim any leading spaces that might have carried over (e.g. from " SFS-119...")
        fallback_path_trimmed = extracted_folder / fallback_clean_path.lstrip()

        if os.path.exists(_to_longpath(fallback_path)):
            return fallback_path
        if os.path.exists(_to_longpath(fallback_path_trimmed)):
            return fallback_path_trimmed

    # If both fail, log a detailed warning
    logger.warning(
        f"[LINE {line_no}] {side} file not found.\n"
        f"   Attempted Primary:  {primary_path}\n"
        f"   Attempted Fallback: {fallback_path if len(parts) > 1 else 'N/A'}"
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PAIR PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_image_pairs(
    txt_file: Path,
    extracted_folder: Path,
) -> List[Tuple[Optional[Path], Optional[Path]]]:
    """
    Parse image pairs from txt_file.

    Returns list of (old_path, new_path) where either side may be None when
    the txt file has '0' as a sentinel (image added or deleted with no pair).
    """
    # Fix for Long Path FileNotFoundError applied here:
    with open(_to_longpath(txt_file), "r", encoding="utf-8") as f:
        raw_lines = f.read().splitlines()

    logger.info(f"Parsing {len(raw_lines)} raw lines from {txt_file.name} …")

    pairs: List[Tuple[Optional[Path], Optional[Path]]] = []
    sentinel_count = 0

    for line_no, line in enumerate(raw_lines, start=1):
        line = line.strip()
        if not line:
            continue

        # Smart split: the second column always starts with 'sessions/' or is '0'.
        m = re.search(r",\s*(sessions/|0\s*$)", line)
        if m:
            raw_old = line[:m.start()].strip()
            raw_new = line[m.start() + 1:].strip()
        else:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                logger.warning(
                    f"[LINE {line_no}] Cannot split into 2 path columns "
                    f"({len(parts)} comma tokens). Line: '{line[:120]}'"
                )
                continue
            raw_old, raw_new = parts

        old_is_sentinel = (raw_old == NO_IMAGE_SENTINEL)
        new_is_sentinel = (raw_new == NO_IMAGE_SENTINEL)

        # Smartly resolve paths using our new fallback logic
        old_path = resolve_image_path(extracted_folder, raw_old, line_no, "OLD") if not old_is_sentinel else None
        new_path = resolve_image_path(extracted_folder, raw_new, line_no, "NEW") if not new_is_sentinel else None

        if old_path is None and new_path is None:
            logger.warning(f"[LINE {line_no}] Both sides missing or invalid — skipping.")
            continue

        if old_is_sentinel or new_is_sentinel:
            sentinel_count += 1
            side = "OLD" if old_is_sentinel else "NEW"
            logger.debug(f"[LINE {line_no}] Single-image entry ({side}=sentinel). Will run LLM with 1 image.")

        logger.debug(
            f"[LINE {line_no}] Pair: "
            f"OLD={old_path.name if old_path else 'None'}  ↔  "
            f"NEW={new_path.name if new_path else 'None'}"
        )
        pairs.append((old_path, new_path))

    logger.info(
        f"Parsed {txt_file.name}: "
        f"{len(pairs)} pair(s) "
        f"({sentinel_count} single-image / sentinel entries) "
        f"from {len(raw_lines)} line(s)."
    )
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE ENCODING  (★ NO resizing — raw bytes at original quality)
# ─────────────────────────────────────────────────────────────────────────────

def image_to_base64(image_path: Path) -> Tuple[str, str]:
    """
    Read image file and return (base64_string, mime_type).
    """
    ext_to_mime = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
        ".tiff": "image/tiff", ".tif": "image/tiff",
    }
    media_type = ext_to_mime.get(image_path.suffix.lower(), "image/png")
    with open(_to_longpath(image_path), "rb") as fh:
        raw_bytes = fh.read()
    b64 = base64.b64encode(raw_bytes).decode("utf-8")
    logger.debug(
        f"Encoded {image_path.name}: {len(raw_bytes):,} bytes → "
        f"base64 ({len(b64):,} chars), mime={media_type}"
    )
    return b64, media_type


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALL  (images sent at original size/quality — no preprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def call_llm_compare(
    client: AzureOpenAI,
    deployment: str,
    old_image_path: Optional[Path],
    new_image_path: Optional[Path],
) -> dict:
    """
    Send one or two images to GPT-4.1 with COMPARISON_PROMPT.
    Matches the specific production JSON payload structure exactly.
    """
    # 1. Build user content array (Images FIRST, Text LAST based on production example)
    user_content: list = []

    if old_image_path is not None:
        b64, mt = image_to_base64(old_image_path)
        user_content.append({"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}})

    if new_image_path is not None:
        b64, mt = image_to_base64(new_image_path)
        user_content.append({"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}})

    # Text goes last
    user_content.append({"type": "text", "text": f"\n{COMPARISON_PROMPT}"})

    img_count = (old_image_path is not None) + (new_image_path is not None)
    old_name = old_image_path.name if old_image_path else "NONE"
    new_name = new_image_path.name if new_image_path else "NONE"
    logger.debug(
        f"LLM call ({img_count} image(s)) | "
        f"old={old_name} | new={new_name}"
    )

    # 2. Call the API with the strict payload structure and parameters
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": "You are an assistant expertly trained to analyze and compare images."
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        max_tokens=2048,
        top_p=1,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content.strip()
    logger.debug(f"LLM raw response: {raw}")

    # Because we use response_format={"type": "json_object"},
    # the response should always be clean JSON, but we keep the try/except as a safety net.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback to regex extraction just in case the API acts up
        logger.warning(f"LLM response not valid JSON. Raw (first 500): {raw[:500]}")
        fence = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
        if fence:
            try:
                return json.loads(fence.group(1).strip())
            except json.JSONDecodeError:
                pass
        return {"discard_image": None, "changes": raw}


# ─────────────────────────────────────────────────────────────────────────────
# PAIR OUTPUT WRITER
# ─────────────────────────────────────────────────────────────────────────────

def save_pair_output(
    pair_id: int,
    old_image_path: Optional[Path],
    new_image_path: Optional[Path],
    llm_result: dict,
    zip_name: str,
    model_name: str,
    api_version: str,
    endpoint: str,
) -> None:
    """
    Save images + JSON for one pair.  Either side may be None (sentinel).
    Only the non-None image(s) are copied at original quality (shutil.copy2).
    """
    pair_name = f"pair_{pair_id:03d}"

    # ── Image pair directory ──────────────────────────────────────────────────
    pair_img_dir = IMAGE_PAIRS_DIR / pair_name
    pair_img_dir.mkdir(parents=True, exist_ok=True)

    if old_image_path is not None:
        shutil.copy2(_to_longpath(old_image_path), str(pair_img_dir / f"original{old_image_path.suffix}"))
    if new_image_path is not None:
        shutil.copy2(_to_longpath(new_image_path), str(pair_img_dir / f"reimage{new_image_path.suffix}"))

    logger.info(f"[{pair_name}] Images copied → {pair_img_dir}")

    # ── Traceability (source_paths.txt in image dir) ─────────────────────────
    source_txt = (
        f"zip_file:     {zip_name}\n"
        f"original:     {old_image_path if old_image_path else 'N/A (sentinel)'}\n"
        f"reimage:      {new_image_path if new_image_path else 'N/A (sentinel)'}\n"
    )
    (pair_img_dir / "source_paths.txt").write_text(source_txt, encoding="utf-8")

    # ── LLM settings block (included in the LLM output traceability file) ────
    llm_settings_txt = (
        f"zip_file:     {zip_name}\n"
        f"original:     {old_image_path if old_image_path else 'N/A (sentinel)'}\n"
        f"reimage:      {new_image_path if new_image_path else 'N/A (sentinel)'}\n"
        f"\n"
        f"--- LLM Settings ---\n"
        f"model:        {model_name}\n"
        f"max_tokens:   2048\n"
        f"top_p:        1\n"
        f"response_fmt: json_object\n"
        f"api_version:  {api_version}\n"
        f"endpoint:     {endpoint}\n"
        f"image_detail: auto (no override — original size/quality)\n"
        f"timestamp:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    # ── JSON output ───────────────────────────────────────────────────────────
    LLM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "discard_image":                       llm_result.get("discard_image"),
        "changes":                             llm_result.get("changes", ""),
        "human_evaluation_for_classification": "",
        "human_evaluation_for_change_summary": "",
    }
    json_path = LLM_OUTPUT_DIR / f"{pair_name}.json"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"[{pair_name}] JSON saved → {json_path}")

    # ── Traceability mirror (source_paths.txt in LLM output dir) ─────────────
    (LLM_OUTPUT_DIR / f"{pair_name}_source_paths.txt").write_text(llm_settings_txt, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# RESUME STATE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def get_latest_pair_id(image_pairs_dir: Path) -> int:
    """
    Scans the output directory to find the highest existing pair ID.
    Returns that integer so the next run can pick up exactly where it left off.
    """
    max_id = 0
    if not image_pairs_dir.exists():
        return max_id

    # Looks for folders named exactly like "pair_001", "pair_105", etc.
    pattern = re.compile(r"^pair_(\d+)$")
    for path in image_pairs_dir.iterdir():
        if path.is_dir():
            match = pattern.match(path.name)
            if match:
                current_id = int(match.group(1))
                if current_id > max_id:
                    max_id = current_id

    return max_id


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 70)
    logger.info("DCA Dataset Generation Pipeline  —  START")
    logger.info(f"Log file        : {LOG_FILE}")
    logger.info(f"DCA data dir    : {DCA_DATA_DIR}")
    logger.info(f"Image pairs dir : {IMAGE_PAIRS_DIR}")
    logger.info(f"LLM output dir  : {LLM_OUTPUT_DIR}")
    logger.info(f"Zip files       : {ZIP_FILES}")
    logger.info("=" * 70)

    # ── Step 0: abort early if az login is missing ────────────────────────────
    verify_az_login()

    IMAGE_PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    LLM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_prompt_file()

    client, deployment, api_version, endpoint = build_openai_client()

    # ── Detect latest run to prevent overwriting ──────────────────────────────
    pair_counter: int = get_latest_pair_id(IMAGE_PAIRS_DIR)
    if pair_counter > 0:
        logger.info(f"Found existing data up to pair_{pair_counter:03d}. Resuming from there...")

    for zip_idx, zip_name in enumerate(ZIP_FILES, start=1):
        zip_path = DCA_DATA_DIR / zip_name

        logger.info(f"\n{'─' * 60}")
        logger.info(f"Processing ZIP {zip_idx}/{len(ZIP_FILES)}: {zip_name}")
        logger.info(f"{'─' * 60}")

        if not zip_path.exists():
            logger.warning(f"[SKIP] Zip not found: {zip_path}")
            continue

        # Derive extraction folder name from zip stem (e.g., "pdf_1.zip" → "pdf_1")
        extract_dir = DCA_DATA_DIR / zip_path.stem
        extract_zip(zip_path, extract_dir)

        matches_file = find_matches_file(extract_dir)
        if matches_file is None:
            logger.warning(f"[SKIP] {zip_name} has no matches file.")
            continue

        pairs = parse_image_pairs(matches_file, extract_dir)
        if not pairs:
            logger.warning(f"[SKIP] No valid pairs in {matches_file.name}.")
            continue

        for old_img, new_img in pairs:
            pair_counter += 1
            pair_label = f"pair_{pair_counter:03d}"

            if old_img is None:
                mode = "SINGLE-IMAGE (added)"
                img_label = f"NEW={new_img.name}"  # type: ignore[union-attr]
            elif new_img is None:
                mode = "SINGLE-IMAGE (deleted)"
                img_label = f"OLD={old_img.name}"
            else:
                mode = "TWO-IMAGE"
                img_label = f"OLD={old_img.name}  |  NEW={new_img.name}"

            logger.info(f"\n[{pair_label}] {mode} | {img_label}")

            try:
                llm_result = call_llm_compare(client, deployment, old_img, new_img)
                logger.info(
                    f"[{pair_label}] LLM → "
                    f"discard_image={llm_result.get('discard_image')}  |  "
                    f"changes: {str(llm_result.get('changes', ''))[:100]}"
                )
                save_pair_output(pair_counter, old_img, new_img, llm_result, zip_name,
                                 deployment, api_version, endpoint)

            except Exception as exc:
                logger.error(f"[{pair_label}] Unexpected error: {exc}", exc_info=True)
                try:
                    save_pair_output(
                        pair_counter, old_img, new_img,
                        {"discard_image": None, "changes": f"PIPELINE_ERROR: {exc}"},
                        zip_name, deployment, api_version, endpoint,
                    )
                except Exception as save_exc:
                    logger.error(f"[{pair_label}] Could not write error sentinel: {save_exc}", exc_info=True)

    logger.info("\n" + "=" * 70)
    logger.info(f"Pipeline COMPLETE  —  Latest pair processed: {pair_counter}")
    logger.info(f"Prompt file     : {PROMPT_FILE}")
    logger.info(f"Images saved in : {IMAGE_PAIRS_DIR}")
    logger.info(f"JSON output in  : {LLM_OUTPUT_DIR}")
    logger.info(f"Log file        : {LOG_FILE}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
