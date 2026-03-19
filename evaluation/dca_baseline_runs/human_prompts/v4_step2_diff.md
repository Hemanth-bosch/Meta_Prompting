You are given TWO informational images (Image 1 = ORIGINAL/old, Image 2 = UPDATED/new).

Your ONLY task: determine whether meaningful changes exist between them, and if so describe them.

TRIVIAL — ignore these: layout shifts, resizing, resolution/quality differences, punctuation-only differences, spelling mistakes, watermark additions/removals, minor color shade, compression artifacts.

MEANINGFUL — report these: Any change that alters the information conveyed including words added/removed/replaced with meaningful impact, numbers/dates/values changed with impact, sentences reworded or restructured meaningfully, table rows/columns added or removed, table cell content changed, diagram elements added/removed/reconnected, text in diagrams changed, flowchart flow changed, labels/captions/headings changed meaningfully, one diagram within a multi-diagram image being partially crossed out.

If images are identical or differ only in TRIVIAL ways:
→ {"discard_image": true, "changes": "Images are identical - no changes"}

If at least one meaningful change exists:
→ {"discard_image": false, "changes": "OVERVIEW: <few sentences summary of what broadly changed between Input Image 1 and Image 2.> | DETAILS (Each Change): 1) <where changed - Area or Location or section name or block name etc>: <what changed, quoting original (image1/ old) and new (image2/ new) - describing each added/ new content/ removed/ modified in detail; 2) ...; 3) ..."}

Double-check — are the changes real and clearly visible, or might you be hallucinating? If uncertain or trivial changes, default to discard_image=true, changes="Images are identical - no changes".

Output raw JSON only. No markdown fences.
{"discard_image": true/false, "changes": "..."}
