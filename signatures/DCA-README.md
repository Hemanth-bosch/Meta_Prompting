# Signatures

This folder contains DSPy Signature definitions for all extraction and classification tasks in the repository.

---

## What is a DSPy Signature?

A DSPy Signature is the task definition injected into every DSPy program and optimizer. It specifies:

- **Input fields** — what the model receives (images, text, etc.)
- **Output fields** — what the model must produce
- **Docstring** — the instruction block that tells the model what to do

Optimizers like MIPROv2 and GEPA can rewrite the docstring and field descriptions to improve performance. Bootstrap can select and synthesize few-shot examples that demonstrate correct behavior.

---

## Files

| File | Task | Key Fields |
|------|------|------------|
| `dca.py` | DCA image-pair comparison | `image_original`, `image_reimage` → `discard_image`, `changes` |
| `jo.py` | Japanese Order extraction | Order document image → 13 structured fields |
| `yanase.py` | YANASE parts order extraction | Order document image → 16 structured fields |

---

## DCA Signature — `dca.py`

### Class: `DCAImageComparison(dspy.Signature)`

Compares two document images (original and updated) and produces a classification decision and a change summary.

### Input Fields

| Field | Type | Description |
|-------|------|-------------|
| `image_original` | `Optional[dspy.Image]` | Image 1 — the reference version of the document |
| `image_reimage` | `Optional[dspy.Image]` | Image 2 — the updated/re-imaged version to compare against |

### Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `discard_image` | `str` | `"true"` or `"false"` — whether this image pair should be discarded |
| `changes` | `str` | Description of what changed between Image 1 and Image 2 |

### Decision Policy (docstring)

The docstring encodes the full DCA business logic:

**Discard (`discard_image="true"`) when:**
- Both images are informational but only trivial differences exist (layout, resolution, watermark, punctuation-only changes)
- Both images are non-informational
- Only one image is present and it is non-informational

**Keep (`discard_image="false"`) when:**
- Any meaningful content change exists (text, value, label, table, diagram, flow)
- Either image is crossed out or redacted (indicating deletion)
- Either image is corrupted, unreadable, or too tiny to interpret
- Only one image is present and it is informational
- One image is informational and the other is non-informational

**Canonical `changes` outputs for special cases:**

| Situation | Required output |
|-----------|----------------|
| Identical or trivial difference | `Images are identical - no changes` |
| One informational image only | `NoMatchFound` |
| Either image crossed out | `Image has been crossed out` |
| Either image corrupted/unreadable/tiny | `Could not infer` |
| Both non-informational | `Both images are non informational and contain only logo/watermark etc. no meaningful content` |
| Single non-informational image | `Only one image and it is non informational` |
| Meaningful differences present | `OVERVIEW: <summary> \| DETAILS (Each Change): 1) <location>: <what changed>; ...` |

### DCA Categories

The 8 ground-truth category codes used for stratified splitting, per-category scoring, and data augmentation:

| Code | Description |
|------|-------------|
| `A1` | Both images non-informational → discard |
| `A2` | Both informational, trivial differences only → discard |
| `B1` | Either image corrupted/unreadable/too tiny → keep |
| `B2` | Either image crossed out → keep |
| `B3` | One informational + one image only → keep (NoMatchFound) |
| `B4` | One informational + one non-informational → keep |
| `B5a` | Both informational, meaningful differences → keep |
| `B5b` | Both informational, meaningful differences (sub-type) → keep |

### P1 Mode (Instructions-Only)

The live active version uses `desc=""` for all input and output fields. All guidance lives in the docstring.

This is enforced via `instructions_only=True` in `TournamentConfig` and `extract_dca.py`. When an optimizer writes guidance into field descriptions, the pipeline absorbs it back into the instruction docstring and clears the field descs before saving.

The commented-out version with full field `desc=` values is preserved in `dca.py` for reference. Switching back is a matter of commenting/uncommenting those lines and setting `instructions_only=False`.

### Usage

```python
import dspy
from signatures.dca import DCAImageComparison

# Plain Predict (no chain-of-thought)
program = dspy.Predict(DCAImageComparison)

# Chain-of-Thought
program = dspy.ChainOfThought(DCAImageComparison)

# Run inference
result = program(
    image_original=dspy.Image.from_file("pair_001/original.png"),
    image_reimage=dspy.Image.from_file("pair_001/reimage.png"),
)
print(result.discard_image)   # "true" or "false"
print(result.changes)         # change description string
```
