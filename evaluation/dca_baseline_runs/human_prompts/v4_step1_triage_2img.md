You are given TWO images (Image 1 = ORIGINAL, Image 2 = UPDATED). Check conditions in order and stop at the first match.

NON-INFORMATIONAL IMAGE — a blank/rectangle/block with no content, a single horizontal/ vertical line or divider, a watermark or logo or brand mark or graphic (no meaningful surrounding text/ data/ information), a standalone section number (e.g. 5.2.1).

INFORMATIONAL IMAGE means - Meaningful sentences, paragraphs, labels, tables with data or text in cells, Diagrams, flowcharts, charts, or illustrations with labels, any image combining a logo/watermark with meaningful content (e.g., a watermarked page with text), any image that contains readable text or diagram or data or charts or flow etc that conveys MEANINGFUL information, even if it also contains a logo/watermark.

B1. Is either image corrupted, unreadable, or too tiny to interpret?
    → {"branch": "B1", "changes": ""}

B2. Does either image have a full-image X / strikethrough / cross mark across the ENTIRE image (not just one element but across all diagrams within the image)?
    → {"branch": "B2", "changes": "Image <1 or 2> has been crossed out"}

B3. Are BOTH images non-informational?
    → {"branch": "B3", "changes": ""}

B4. Is exactly ONE image non-informational and the other informational?
    → {"branch": "B4", "changes": "<briefly describe what the informational image contains; state that only one image is informational>"}

ELSE — both images are informational:
    → {"branch": "both_informational", "changes": ""}

Output raw JSON only. No markdown fences.
{"branch": "...", "changes": "..."}
