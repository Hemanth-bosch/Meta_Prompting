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
