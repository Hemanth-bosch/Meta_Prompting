Compare Image 1 (original) and Image 2 (updated) and output:
- discard_image: 'true' or 'false'
- changes: branch-consistent explanation string

Decision policy (aligned with current DCA rules):

Non-informational image examples:
- logo/watermark/brand mark only
- blank block / rectangle / line / divider only
- standalone section number (e.g., 5.2.1)

Informational image examples:
- meaningful readable text, labels, tables with data/text, diagrams/charts/flow content
- logo/watermark + meaningful content still counts as informational

Set discard_image='true' when:
- both images are informational but only trivial differences exist
    (layout/resize/resolution/quality/watermark/punctuation-only/no content impact)
- both images are non-informational
- only one image is present and that image is non-informational

Set discard_image='false' when:
- any meaningful content change exists (text/value/label/table/diagram/flow/content removal)
- either image is crossed out / redacted indicating deletion
- either image is corrupted/unreadable/too tiny to interpret
- only one image is present and that image is informational

Preferred canonical changes outputs for special cases:
- identical/trivial: "Images are identical - no changes"
- one informational image only: "NoMatchFound"
- crossed out: "Image has been crossed out"
- unreadable/tiny: "Could not infer"
- both non-informational: "Both images are non informational and contain only logo/watermark etc. no meaningful content"
- single non-informational image: "Only one image and it is non informational"

For meaningful differences, describe observable changes faithfully and avoid hallucinations.
Use this format for meaningful changes:
"OVERVIEW: <brief summary of what changed between Image 1 and Image 2> | DETAILS (Each Change): 1) <location/area/section>: <what changed, quoting original and updated content>; 2) <location>: <what changed>; ..."

Respond ONLY with a raw JSON object with exactly these two keys:
{
  "discard_image": "true" or "false",
  "changes": "<canonical string as described above>"
}
No markdown fences, no extra keys, no commentary outside the JSON.