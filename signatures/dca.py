import dspy


class DCAImageComparison(dspy.Signature):
    """Compare two document images pixel-by-pixel and determine if the re-image
    should be discarded or kept based on visual differences.

    Set discard_image to 'true' if:
    - The images are visually identical (no meaningful differences)
    - Differences are only trivial (resolution, watermark presence, minor border)
    - The image is just a watermark, logo, blank block, line, section number, or plain text
    - The image is a table with only tabular content
    - Punctuation-only differences

    Set discard_image to 'false' if:
    - There are meaningful text changes (words added, removed, or modified)
    - Diagram or structural content has changed
    - Labels, flowchart elements, or annotations have been modified
    - Image 2 has a cross mark indicating deletion
    - Content has been removed between versions (e.g., text blocks deleted)
    - Only one image is informational and it is NOT a watermark or logo

    Ignore differences in resolution, size, quality, or watermark presence.
    Image 1 is the original version. Image 2 is the updated version."""

    image_original: dspy.Image = dspy.InputField(
        desc="Original document image (Image 1) - the reference version"
    )
    image_reimage: dspy.Image = dspy.InputField(
        desc="Re-imaged/updated version of the same document (Image 2) - compare against original"
    )
    discard_image: str = dspy.OutputField(
        desc="Output exactly 'true' if the images are identical or differences are trivial "
             "(watermark, resolution, border). Output exactly 'false' if there are meaningful "
             "visual differences such as changed text, removed content, modified diagrams, "
             "cross-out marks, or structural changes. Must be 'true' or 'false'."
    )
    changes: str = dspy.OutputField(
        desc="If discard_image is true: 'Images are identical - no changes' or 'NoMatchFound'. "
             "If discard_image is false: Describe the specific changes using format "
             "'Text changed from [exact original] to [exact new]' for text differences. "
             "For content removal, list what was removed. No opinions, only observable facts."
    )
