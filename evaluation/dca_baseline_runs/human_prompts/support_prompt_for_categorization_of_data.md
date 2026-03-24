You are given two human-labeled fields from a document-comparison dataset:

- human_label_discard_image: boolean (true/false)
- human_label_changes: string

Your task is to map this pair into EXACTLY ONE category code from:
A1, A2, B1, B2, B3, B4, B5a, B5b

Return ONLY raw JSON (no markdown, no explanation):
{
  "human_label_category": "<one of A1|A2|B1|B2|B3|B4|B5a|B5b>"
}

═══════════════════════════════════════════════════
CATEGORY DEFINITIONS
═══════════════════════════════════════════════════
A1  Single Image; Non-informational (human_label_changes will contain = Single image and non-informational)
A2  Single Image; Informational (human_label_changes will contain = NoMatchFound)
B1  Either Image is Corrupted, unreadable or too tiny (human_label_changes will contain = could not infer)
B2  Either Image has throughout cross mark/ X/ strikethough/redaction indicating deletion of whole image (not just one diagram)
B3  Both images non-informational (human_label_changes will contain = Both images are non-informational)
B4  One informational describing what it contains; one non-informational (human_label_changes will contain = Description about image which is informational and mention that the other is non-informational)
B5a Differences exists but are trivial/ So -> Both images are identical and no changes exist (human_label_changes will contain = Both images are identical and no changes)
B5b Differences explained in changes output (meaningful) (human_label_changes will contain = Detailed explanation of differences which are meaningful)

═══════════════════════════════════════════════════
DECISION RULES (IN ORDER - THINK DEEP)
═══════════════════════════════════════════════════
1) Normalize the text mentally to lowercase and ignore punctuation/style differences.
2) If changes indicates inability to infer (examples: "could not infer", unreadable/corrupted/tiny), classify B1.
3) If changes indicates fully crossed-out whole image (examples: "fully crossed out", "fully cross mark" etc.), classify B2. IMPORTANT NOTE: There are cases, where one diagram in either image in crossed out, none of the images is fully crossed. In such cases, where none of the whole image is fully crossed out, and the human_label_changes explains the differences and mentions that one diagram or certain area is crossed out -> here the correct category would be B5b, not B2.
4) If changes indicates one image only + non-informational, classify A1.
5) If changes is "NoMatchFound" (or equivalent only-one-image informational wording), classify A2.
6) If changes indicates both images non-informational, classify B3.
7) If changes indicates one informational and one non-informational, classify B4.
8) If changes indicates identical/no changes/trivial-only differences, classify B5a.
9) Otherwise, classify B5b (detailed/meaningful differences present).

═══════════════════════════════════════════════════
CONSISTENCY CHECKS WITH human_label_discard_image
═══════════════════════════════════════════════════
- If discard_image is true, final category must be one of: A1, B3, B5a.
- If discard_image is false, final category must be one of: A2, B1, B2, B4, B5b.
- If text signal and boolean conflict, use the strongest explicit text signal first (B1/B2/B3/B4/A1/A2), then enforce the closest consistent category.

═══════════════════════════════════════════════════
IMPORTANT OUTPUT RULES
═══════════════════════════════════════════════════
- Output EXACTLY one key: "human_label_category"
- Value must be EXACTLY one code from: A1, A2, B1, B2, B3, B4, B5a, B5b
- Do not output any label descriptions, only the code
- No extra keys, no comments, no markdown fences