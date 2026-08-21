"""DCA task contract — the ``DCAImageComparison`` DSPy Signature.

Declares the inputs (original + re-imaged document images) and outputs
(``discard_image``, ``changes``). The decision policy lives in the signature's
instruction docstring; field descriptions are kept empty (instructions-only mode)
so the optimizers have a single clean surface to improve.
"""
import dspy
from typing import Optional


class DCAImageComparison(dspy.Signature):
    """
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
        - Start with/ default to: {"discard_image": true, "changes": "Images are identical - no changes"} and refer and follow below "Other Rules" strictly.
        - When describing changes/ differences in output JSON, use below CHANGES FIELD TEMPLATE (use this structure when meaningful changes exist):
            "OVERVIEW: <few sentences summary of what broadly changed between Input Image 1 and Image 2.> | DETAILS (Each Change): 1) <where changed - Area or Location or section name or block name etc; if location or area or region defined can be mentioned>: <what changed, quoting original (image1/ old) and new (image2/ new) - describing each added/ new content/ removed/ modified in detail; 2) <where AND what changed AND describing each added/ new content/ removed/ modified in detail>; 3) ...; 4) ..."

        "Other Rules":
        - If only one image is provided, determine whether it is informational or non-informational.
        - Non-informational means: a blank/rectangle/block with no content, a single horizontal/ vertical line or divider, a watermark or logo or brand mark or graphic (no meaningful surrounding text/ data/ information), a standalone section number (e.g. 5.2.1).
        - Meaningful sentences, paragraphs, labels, tables with data or text in cells, Diagrams, flowcharts, charts, or illustrations with labels, any image combining a logo/watermark WITH other meaningful content (e.g., a watermarked page with text), any image that contains readable text or diagram or data or charts or flow etc that conveys MEANINGFUL information, even if it also contains a logo/watermark — do NOT treat them as non-informational. These convey meaningful information and are informational.
        - If only one image is provided and it is non-informational, set "discard_image" to true and "changes" to "Only one image which is non-informational and it contains logo (or watermark / blank block / single line / standalone section number / etc.)".
        - If only one image is provided and it is informational, set "discard_image" to false and "changes" to "NoMatchFound".
        - If either image is corrupted, too tiny, or unreadable, set "discard_image" to false and "changes" to "Could not infer".
        - If either image has a cross mark indicating deletion of full image (full deletion means throughout X/ strikethrough, cross mark - indicating deletion of the whole image including all diagrams), set "discard_image" to false and "changes" to "Image * has been crossed out".
        - If both images are non-informational, set "discard_image" to true and "changes" to "Both images are non informational and contain only logos / etc..".
        - If one image is non-informational and the other is informational, set "discard_image" to false, and in "changes" - <describe what the informational image contains and state that only one image is informational>.
        - If both images are informational and identical, or differ only in trivial ways, set "discard_image" to true and "changes" to "Images are identical - no changes". Trivial changes include: layout shifts, resizing, resolution or quality differences, punctuation-only differences, spelling mistakes, watermark additions/removals, minor color shade, compression differences with no content impact.
        - If both images are informational and have meaningful differences, set "discard_image" to false and <list ALL meaningful changes found as per below CHANGES FIELD TEMPLATE>.
        - Meaningful changes include: Any change that alters the information conveyed e.g. Words added, removed, or replaced, Numbers, dates, or values changed that conveys meaningful impact • Sentences reworded or restructured with meaningful impact • Rows/columns added or removed in a table • Cell content changed in a table • Diagram elements added, removed, or reconnected or text in diagrams being changed • Flow changed in a flowchart ETC. — any change that alters the information conveyed by the image • Labels, captions, or headings or text changed meaningfully in a way that alters the information conveyed • One of the many diagrams in an image being crossed out partially is also a meaningful change.
        - Image 1 is the original version. Image 2 is the updated version.

        Generate ONLY JSON in the following format:
        {"discard_image": true/false, "changes": "As per CHANGES FIELD TEMPLATE"}
    """
    
    # """Compare Image 1 (original) and Image 2 (updated) and output:
    # - discard_image: 'true' or 'false'
    # - changes: branch-consistent explanation string

    # Decision policy (aligned with current DCA rules):

    # Non-informational image examples:
    # - logo/watermark/brand mark only
    # - blank block / rectangle / line / divider only
    # - standalone section number (e.g., 5.2.1)

    # Informational image examples:
    # - meaningful readable text, labels, tables with data/text, diagrams/charts/flow content
    # - logo/watermark + meaningful content still counts as informational

    # Set discard_image='true' when:
    # - both images are informational but only trivial differences exist
    #     (layout/resize/resolution/quality/watermark/punctuation-only/no content impact)
    # - both images are non-informational
    # - only one image is present and that image is non-informational

    # Set discard_image='false' when:
    # - any meaningful content change exists (text/value/label/table/diagram/flow/content removal)
    # - either image is crossed out / redacted indicating deletion
    # - either image is corrupted/unreadable/too tiny to interpret
    # - only one image is present and that image is informational

    # Preferred canonical changes outputs for special cases:
    # - identical/trivial: "Images are identical - no changes"
    # - one informational image only: "NoMatchFound"
    # - crossed out: "Image has been crossed out"
    # - unreadable/tiny: "Could not infer"
    # - both non-informational: "Both images are non informational and contain only logo/watermark etc. no meaningful content"
    # - single non-informational image: "Only one image and it is non informational"

    # For meaningful differences, describe observable changes faithfully and avoid hallucinations.

    # Input Field descriptions:
    # image_original: Original document image (Image 1) - the reference version
    # image_reimage: Re-imaged/updated version of the same document (Image 2) - compare against original
    
    # Output Field descriptions:
    # changes:
    #         "Return 'Only one image which is non-informational...' if only one image is present and it is non-informational. "
    #         "Return 'NoMatchFound' if only one image is present and it is informational. "
    #         "Return 'Could not infer' if either image is corrupted/unreadable/too tiny to interpret. "
    #         "Return 'Image has been crossed out' if either image is fully crossed out/redacted indicating deletion. "
    #         "Return 'Both images are non informational...' if both images are non-informational and contain only logo/watermark etc. no meaningful content. "
    #         "Return 'One image informational and other non-informational: <description of informational content>' if one image is informational and the other is non-informational. For example, 'Image 1 contains a table with 3 rows and 2 columns with headers X and Y, while Image 2 contains only a logo'. "
    #         "Return 'Images are identical - no changes' if both images are informational but only trivial differences exist (layout/resize/resolution/quality/watermark/punctuation-only/no content impact). "
    #         "Return 'OVERVIEW: <brief summary of what changed between Image 1 and Image 2> | DETAILS (Each Change): 1) <location/area/section>: <what changed, quoting original and updated content>; 2) <location>: <what changed>; ...' if both images are informational and meaningful differences exist, describe all observable changes using the format."

    # discard_image:
    #         "Output exactly 'true' or 'false' (lowercase). "
    #         "Use 'true': when there is only one image and it is non-informational, or when both images are non-informational, or when both images are informational but differences are only **trivial** (layout shifts, spelling etc.) "
    #         "Use 'false': when there is only one image and it is informational, or when either image is **fully crossed-out**, or when either image is **corrupted, unreadable, or too tiny** or when one image is non-informational and the other is informational or when both images are informational and at least one **meaningful difference** exists."

    # """
    # image_original: Optional[dspy.Image] = dspy.InputField(
    #     desc="Original document image (Image 1) - the reference version", default=None
    # )
    # image_reimage: Optional[dspy.Image] = dspy.InputField(
    #     desc="Re-imaged/updated version of the same document (Image 2) - compare against original", default=None
    # )
    # discard_image: str = dspy.OutputField(
    #     desc="Output exactly 'true' or 'false' (lowercase). "
    #         "Use 'true': when there is only one image and it is non-informational, or when both images are non-informational, or when both images are informational but differences are only **trivial** (layout shifts, spelling etc.) "
    #         "Use 'false': when there is only one image and it is informational, or when either image is **fully crossed-out**, or when either image is **corrupted, unreadable, or too tiny** or when one image is non-informational and the other is informational or when both images are informational and at least one **meaningful difference** exists."
    # )
    # changes: str = dspy.OutputField(
    #     desc=
    #         "Return 'Only one image which is non-informational...' if only one image is present and it is non-informational. "
    #         "Return 'NoMatchFound' if only one image is present and it is informational. "
    #         "Return 'Could not infer' if either image is corrupted/unreadable/too tiny to interpret. "
    #         "Return 'Image has been crossed out' if either image is fully crossed out/redacted indicating deletion. "
    #         "Return 'Both images are non informational...' if both images are non-informational and contain only logo/watermark etc. no meaningful content. "
    #         "Return 'One image informational and other non-informational: <description of informational content>' if one image is informational and the other is non-informational. For example, 'Image 1 contains a table with 3 rows and 2 columns with headers X and Y, while Image 2 contains only a logo'. "
    #         "Return 'Images are identical - no changes' if both images are informational but only trivial differences exist (layout/resize/resolution/quality/watermark/punctuation-only/no content impact). "
    #         "Return 'OVERVIEW: <brief summary of what changed between Image 1 and Image 2> | DETAILS (Each Change): 1) <location/area/section>: <what changed, quoting original and updated content>; 2) <location>: <what changed>; ...' if both images are informational and meaningful differences exist, describe all observable changes using the format."
    # )

    image_original: Optional[dspy.Image] = dspy.InputField(
        desc="", default=None
    )
    image_reimage: Optional[dspy.Image] = dspy.InputField(
        desc="", default=None
    )
    discard_image: str = dspy.OutputField(
        desc=""
    )
    changes: str = dspy.OutputField(
        desc=
            ""
    )
