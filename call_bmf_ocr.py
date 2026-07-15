"""Call Mistral OCR via the Bosch Model Farm (BMF) — Vertex AI publisher endpoint."""

import base64
import json
import os
import sys
import requests

BMF_OCR_URL = "https://aoai-farm.bosch-temp.com/api/providers/mistral/azure/ocr"

api_key = os.environ.get("BMF_API_KEY")
if not api_key:
    raise EnvironmentError("BMF_API_KEY environment variable is not set.")

# --- input file ---
pdf_path = sys.argv[1] if len(sys.argv) > 1 else "document.pdf"
with open(pdf_path, "rb") as f:
    pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": "mistral-document-ai-2512",
    "document": {
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{pdf_b64}",
    },
    "include_image_base64": False,
}

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

response = requests.post(BMF_OCR_URL, headers=headers, json=payload, timeout=120)
if not response.ok:
    print(f"HTTP {response.status_code}: {response.text}", file=sys.stderr)
    sys.exit(1)

result = response.json()
print(json.dumps(result, indent=2, ensure_ascii=False))
