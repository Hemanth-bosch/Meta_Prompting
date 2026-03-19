You are given one or two images from a document comparison (old version vs new version).
Image 1 = ORIGINAL (old). Image 2 = UPDATED (new).

Your objective is to report meaningful changes between old and new documents.

Your Task:
Analyze step-by-step using the decision tree below, then return ONLY a raw JSON object — no markdown, no explanation, no extra keys.

Output schema:
{
  "discard_image": true/false,
  "changes": "<result based on matched branch>"
}

═══════════════════════════════════════════════════
IMAGE INPUT RULES (READ FIRST)
═══════════════════════════════════════════════════
- Images are provided as separate inputs: Image 1 (original/old) and Image 2 (updated/new).
- You will ALWAYS receive the images as distinct attachments. Count the number of image attachments you received — do NOT judge by content.
- A watermark, logo, white image, blank image or non-informational image still counts as ONE image. If you receive two image attachments, you have TWO images — even if one appears to be only a watermark/logo/white/blank.
- NEVER dismiss or skip an image because it looks "empty" or "non-informational." Acknowledge every image attachment you received.

═══════════════════════════════════════════════════
DEFINITIONS
═══════════════════════════════════════════════════

NON-INFORMATIONAL IMAGE — a blank/rectangle/block with no content, a single horizontal/ vertical line or divider, a watermark or logo or brand mark or graphic (no meaningful surrounding text/ data/ information), a standalone section number (e.g. 5.2.1)

INFORMATIONAL IMAGE means - Meaningful sentences, paragraphs, labels, tables with data or text in cells, Diagrams, flowcharts, charts, or illustrations with labels, any image combining a logo/watermark WITH other meaningful content (e.g., a watermarked page with text), any image that contains readable text or diagram or data or charts or flow etc that conveys MEANINGFUL information, even if it also contains a logo/watermark.

TRIVIAL DIFFERENCES include: layout shifts, resizing, resolution or quality differences, punctuation-only differences, spelling mistakes, watermark additions/removals, minor color shade, compression differences with no content impact.

MEANINGFUL CHANGES include: Any change that alters the information conveyed e.g. Words added, removed, or replaced, Numbers, dates, or values changed that conveys meaningful impact • Sentences reworded or restructured with meaningful impact • Rows/columns added or removed in a table • Cell content changed in a table • Diagram elements added, removed, or reconnected or text in diagrams being changed • Flow changed in a flowchart ETC. — any change that alters the information conveyed by the image • Labels, captions, or headings or text changed meaningfully in a way that alters the information conveyed • One of the many diagrams in an image being crossed out partially is also a meaningful change.

═══════════════════════════════════════════════════
DECISION TREE — Follow top to bottom.
═══════════════════════════════════════════════════

─────────────────────────────────────
CASE A: SINGLE IMAGE (only one image attachment received)
─────────────────────────────────────

  A1. IF the single image is NON-INFORMATIONAL:
      → discard_image = true
      → changes = "Only one image which is non-informational and it contains logo (or watermark / blank block / single line / standalone section number / etc.)"
      (STOP)

  A2. IF the single image is INFORMATIONAL:
      → discard_image = false
      → changes = "NoMatchFound"
      (STOP)

─────────────────────────────────────
CASE B: TWO IMAGES (two image attachments received)
─────────────────────────────────────

  B1. IF either image is corrupted, unreadable, or too tiny to interpret:
      → discard_image = false
      → changes = "Could not infer"
      (STOP)

  B2. IF either image has a throughout "X"/ strikethrough/ redaction/ cross mark indicating deletion of the whole image (all diagrams crossed inside that image):
      → discard_image = false
      → changes = "Image * has been crossed out"
      (STOP)

  B3. IF BOTH images are NON-INFORMATIONAL:
      → discard_image = true
      → changes = "Both images are non informational and contain only logos / etc.."
      (STOP)

  B4. IF ONE image is NON-INFORMATIONAL and the OTHER is INFORMATIONAL:
      → discard_image = false
      → changes = "<describe what the informational image contains and state that only one image is informational>"
      (STOP)

  B5. IF BOTH images are INFORMATIONAL:

      B5a. IF all differences between the two images are only TRIVIAL DIFFERENCES:
           → discard_image = true
           → changes = "Images are identical - no changes"
           (STOP)

      B5b. IF at least one MEANINGFUL difference exists:
           → Perform DETAILED CHANGE ANALYSIS (see below).
           → discard_image = false
           → changes = "<list ALL meaningful changes found as per below 'CHANGES FIELD TEMPLATE'>"
           (STOP)

═══════════════════════════════════════════════════
CHANGES FIELD TEMPLATE
═══════════════════════════════════════════════════

- When describing changes/ differences in output JSON, use below CHANGES FIELD TEMPLATE (use this structure when meaningful changes exist):
    "OVERVIEW: <few sentences summary of what broadly changed between Input Image 1 and Image 2.> | DETAILS (Each Change): 1) <where changed - Area or Location or section name or block name etc; if location or area or region defined can be mentioned>: <what changed, quoting original (image1/ old) and new (image2/ new) - describing each added/ new content/ removed/ modified in detail; 2) <where AND what changed AND describing each added/ new content/ removed/ modified in detail>; 3) ...; 4) ..."

Verification: After analysis, double-check — are the changes you found REAL and clearly visible, or might you be hallucinating? If uncertain, default to discard_image=true, changes="Images are identical - no changes".

═══════════════════════════════════════════════════
OUTPUT RULES
═══════════════════════════════════════════════════
- Follow the decision tree IN ORDER.
- Output raw JSON only. No markdown fences, no commentary outside the JSON.
- Use EXACTLY these two keys: "discard_image", "changes". No other keys.
