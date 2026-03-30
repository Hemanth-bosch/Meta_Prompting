import dspy
from typing import Optional


class DCAImageComparison(dspy.Signature):
        """Compare Image 1 (original) and Image 2 (updated) and output:
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
        """
        image_original: Optional[dspy.Image] = dspy.InputField(
            desc="Original document image (Image 1) - the reference version", default=None
        )
        image_reimage: Optional[dspy.Image] = dspy.InputField(
            desc="Re-imaged/updated version of the same document (Image 2) - compare against original", default=None
        )
        discard_image: str = dspy.OutputField(
            desc="Output exactly 'true' or 'false' (lowercase). "
                "Use 'true': when there is only one image and it is non-informational, or when both images are non-informational, or when both images are informational but differences are only **trivial** (layout shifts, spelling etc.) "
                "use 'false': when these is only one image and it is informational, or when either image is **fully crossed-out**, or when either image is **corrupted, unreadable, or too tiny** or when one image is non-informational and the other is informational or when both images are informational and at least one **meaningful difference** exists."
        )
        changes: str = dspy.OutputField(
            desc=
                "Return 'Only one image which is non-informational...' if only one image is present and it is non-informational. "
                "Return 'NoMatchFound' if only one image is present and it is informational. "
                "Return 'Could not infer' if either image is corrupted/unreadable/too tiny to interpret. "
                "Return 'Image has been crossed out' if either image is fully crossed out/redacted indicating deletion. "
                "Return 'Both images are non informational...' if both images are non-informational and contain only logo/watermark etc. no meaningful content. "
                "Return 'One image information and other non-informational and Description of the informational content' if one image is informational and the other is non-informational. For example, 'Image 1 contains a table with 3 rows and 2 columns with headers X and Y, while Image 2 contains only a logo'. "
                "Return 'Images are identical - no changes' if both images are informational but only trivial differences exist (layout/resize/resolution/quality/watermark/punctuation-only/no content impact). "
                "Return 'OVERVIEW: <few sentences summary of what broadly changed between Input Image 1 and Image 2.> | DETAILS (Each Change): 1) <where changed>; <what changed in detail>', if both images are informational and atleast one meaningful difference exists between them."
        )
