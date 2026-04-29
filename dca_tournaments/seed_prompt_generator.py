"""Stage 0 — LLM-written seed prompt generation.

Uses a strong LLM (e.g. gpt-5.2) to generate N candidate instructions,
evaluates each on the val set (bare + with demos variants), tracks token
usage, and returns all candidates as CandidateEntry objects.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import dspy

from config import create_lm
from signatures.dca import DCAImageComparison
from evaluation.dca_metric import dca_comparison_metric
from dca_tournaments.leaderboard import CandidateEntry, OptimizationLeaderboard
from dca_tournaments.tour_utils import (
    evaluate_candidate,
    save_candidate,
    get_program_info,
    find_worst_performing_examples,
)

logger = logging.getLogger("tournament.seed")


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

# ── Category reference (injected into meta-prompt) ───────────────
_CATEGORY_REFERENCE = """
Category definitions for DCA (Document Change Analysis):

Case A — Single image present:
  A1: Single NON-INFORMATIONAL image (blank/logo/watermark/divider) → discard=true
  A2: Single INFORMATIONAL image (meaningful text/tables/diagrams)  → discard=false, changes="NoMatchFound"

Case B — Two images present:
  B1: Either image is CORRUPTED / UNREADABLE / TOO TINY              → discard=false, changes="Could not infer"
  B2: Either image has a FULL STRIKETHROUGH / CROSS-OUT               → discard=false, changes="Image crossed out"
  B3: BOTH images are NON-INFORMATIONAL                               → discard=true,  changes="Both non informational"
  B4: ONE informational + ONE non-informational                        → discard=false, describe the informational image
  B5a: Both informational, TRIVIAL differences only                    → discard=true,  changes="Images are identical - no changes"
  B5b: Both informational, MEANINGFUL differences exist                → discard=false, detailed change description
""".strip()


class SeedPromptGenerator:
    """Generate LLM-written seed prompts for Stage 0.

    1. Create baseline module with default DCAImageComparison signature
    2. Run baseline on train set to find failing examples
    3. Ask a strong LLM to write improved instructions (+ optional field descs)
    4. Evaluate each candidate bare, and optionally with demos variant
    5. Track token usage per candidate
    6. Return all as CandidateEntry objects
    """

    def __init__(
        self,
        seed_model_name: str = "azure/gpt-5.2",
        student_model_name: str = "azure/gpt-4.1-mini",
        n_candidates: int = 3,
        max_context_examples: int = 6,
        use_cot: bool = False,
        seed_temperature: float = 1.0,
        seed_eval_temperature: float = 0.1,
        eval_num_threads: int = 4,
        instructions_only: bool = True,
        seed_reasoning_effort: str | None = "low",
        seed_add_demos_variant: bool = True,
        seed_parallel_evaluations: int = 4,
        track_tokens: bool = True,
        seed_context_mode: str = "worst",  # "worst" | "best"
    ):
        self.seed_model_name = seed_model_name
        self.student_model_name = student_model_name
        self.n_candidates = n_candidates
        self.max_context_examples = max_context_examples
        self.use_cot = use_cot
        self.seed_temperature = seed_temperature
        self.seed_eval_temperature = seed_eval_temperature
        self.eval_num_threads = eval_num_threads
        self.instructions_only = instructions_only
        self.seed_reasoning_effort = seed_reasoning_effort
        self.seed_add_demos_variant = seed_add_demos_variant
        self.seed_parallel_evaluations = seed_parallel_evaluations
        self.track_tokens = track_tokens
        self.seed_context_mode = seed_context_mode  # "worst" → show failures; "best" → show gold examples

    def generate(
        self,
        train_set: list,
        val_set: list,
        metric=None,
        leaderboard: OptimizationLeaderboard | None = None,
        output_dir: str | Path = ".",
        log_dir: str | Path | None = None,
    ) -> list[CandidateEntry]:
        """Run seed prompt generation. Returns list of CandidateEntry.

        ``train_set`` and ``val_set`` must be the MINI subsets (mini_train and
        mini_val).  The full val set and test set are never touched here — they
        are only used in the final re-evaluation step after all stages complete.
        Every candidate score stored on a CandidateEntry refers to mini_val.
        """
        if metric is None:
            metric = dca_comparison_metric

        output_dir = Path(output_dir) / "stage_0_seed"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = Path(log_dir) if log_dir is not None else None

        print(f"\n{'='*70}")
        print(f" Stage 0: LLM-Written Seed Prompt Generation")
        print(f"{'='*70}")
        print(f"  Seed LM           : {self.seed_model_name}")
        print(f"  Student LM        : {self.student_model_name}")
        print(f"  N candidates      : {self.n_candidates}")
        print(f"  Context examples  : {self.max_context_examples} ({self.seed_context_mode})")
        print(f"  Instructions only : {self.instructions_only}")
        print(f"  Reasoning effort  : {self.seed_reasoning_effort or 'default'}")
        print(f"  Add demos variant : {self.seed_add_demos_variant}")
        print(f"  Parallel evals    : {self.seed_parallel_evaluations}")
        print(f"  Seed eval temp    : {self.seed_eval_temperature}")
        print(f"  Track tokens      : {self.track_tokens}")
        print(f"  Output            : {output_dir}")
        print(f"{'='*70}\n")

        # ── Step 0: Enforce P1 (instructions-only) ────────────────────
        # If any field descriptions have crept in (e.g. from a prior run or
        # manual edit), absorb them into the instruction docstring so that the
        # fields remain empty — this is the contract for all downstream stages.
        _absorbed_descs: list[str] = []
        for fname, finfo in DCAImageComparison.model_fields.items():
            extra = finfo.json_schema_extra or {}
            desc = extra.get("desc", "") if isinstance(extra, dict) else ""
            if desc:
                _absorbed_descs.append(f"- {fname}: {desc}")
                logger.warning("Field '%s' has non-empty desc — absorbing into instruction.", fname)
        if _absorbed_descs:
            _merged = "\n".join(_absorbed_descs)
            _old_doc = DCAImageComparison.__doc__ or ""
            DCAImageComparison.__doc__ = (
                f"{_old_doc}\n\nField guidance (absorbed from field descs):\n{_merged}"
            )
            # Clear the field descs so downstream sees empty strings
            for fname in DCAImageComparison.model_fields:
                DCAImageComparison.model_fields[fname].json_schema_extra = {"desc": "", "prefix": ""}
            print(f"[{_ts()}] P1: absorbed {len(_absorbed_descs)} non-empty field desc(s) into instruction.")
        else:
            print(f"[{_ts()}] P1 check passed: all DCAImageComparison field descs are already empty.")

        # ── Step 1: Baseline run + example mining (worst or best) ───────
        _select_best = self.seed_context_mode == "best"
        _mode_label = "best-performing" if _select_best else "worst-performing"
        print(f"\n[{_ts()}] Step 1/4: Baseline eval on train set ({len(train_set)} examples) — mining {_mode_label} examples...")
        step1_t0 = time.time()
        student_lm = create_lm(
            self.student_model_name,
            temperature=self.seed_eval_temperature,
            cache=False,
        )
        dspy.configure(lm=student_lm)

        if self.use_cot:
            baseline = dspy.ChainOfThought(DCAImageComparison)
        else:
            baseline = dspy.Predict(DCAImageComparison)

        # Single-pass: score all examples, collect worst failures AND compute
        # aggregate metrics in one shot (avoids a redundant second evaluation).
        print(f"  [{_ts()}] Running single-pass baseline scoring + worst-example mining...")
        failures, baseline_eval = find_worst_performing_examples(
            baseline, train_set, metric,
            n=self.max_context_examples,
            per_category_limit=1,
            num_threads=self.eval_num_threads,
            return_eval_result=True,
            select_best=_select_best,
        )
        baseline_info = get_program_info(baseline)
        baseline_dir = output_dir / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = baseline_dir / "seed_baseline.json"
        save_candidate(baseline, baseline_path, metadata={
            "stage": "seed_baseline",
            "config_name": "seed_baseline",
            "student_model": self.student_model_name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "note": "Baseline run on TRAIN set (diagnostic only, excluded from Pareto)",
            "eval_result": baseline_eval,
        })
        # NOTE: seed_baseline is evaluated on train_set (for worst-example mining),
        # NOT val_set, so scores are NOT comparable and val_avg_total_tokens is N/A.
        # It is excluded from Pareto selection via exclude_stages=["seed_baseline"].
        baseline_entry = CandidateEntry(
            program_path=str(baseline_path),
            stage="seed_baseline",
            config_name="seed_baseline",
            parent_path=None,
            val_cls_score=baseline_eval["cls_score"],
            val_change_score=baseline_eval["change_score"],
            val_total_score=baseline_eval["total_score"],
            per_category_scores=baseline_eval["per_category_scores"],
            n_demos=baseline_info["n_demos"],
            instruction_length=baseline_info["instruction_length"],
            timestamp=datetime.now().isoformat(timespec="seconds"),
            metadata={
                "stage": "seed_baseline",
                "config_name": "seed_baseline",
                "model": self.student_model_name,
                "student_model": self.student_model_name,
                "eval_result": baseline_eval,
            },
        )
        if leaderboard:
            leaderboard.add(baseline_entry)

        print(f"  [{_ts()}] Baseline: total={baseline_eval['total_score']:.4f}  "
              f"cls={baseline_eval['cls_score']:.4f}  chg={baseline_eval['change_score']:.4f}")

        print(f"  [{_ts()}] Found {len(failures)} {_mode_label} examples (took {time.time() - step1_t0:.1f}s):")
        for i, f in enumerate(failures):
            cat = f["category"]
            score = f["score"]
            pair = getattr(f["example"], "pair_name", "?")
            print(f"    {i+1}. [{cat}] pair={pair}  score={score:.3f}")

        # ── Step 2: Build meta-prompt ────────────────────────────────
        print("\nStep 2/4: Building meta-prompt...")

        current_instruction = DCAImageComparison.__doc__ or ""
        current_fields = {}
        for fname, finfo in DCAImageComparison.model_fields.items():
            extra = finfo.json_schema_extra or {}
            desc = extra.get("desc", "") if isinstance(extra, dict) else ""
            current_fields[fname] = desc

        # Format failure examples (text portion)
        failure_text = ""
        # Collect image content parts for multimodal meta-prompt (up to 3 examples = 6 images)
        _max_image_examples = min(3, len(failures))
        _failure_image_parts: list[dict] = []
        for i, f in enumerate(failures):
            ex = f["example"]
            pred = f["prediction"]
            expected_discard = getattr(ex, "expected_discard_image", "?")
            expected_changes = getattr(ex, "expected_changes", "?")
            pred_discard = getattr(pred, "discard_image", "?") if pred else "ERROR"
            pred_changes = getattr(pred, "changes", "?") if pred else "ERROR"

            failure_text += f"""
Example {i+1} (Category: {f['category']}, Score: {f['score']:.3f}):
  Expected discard_image: {expected_discard}
  Predicted discard_image: {pred_discard}
  Expected changes: {str(expected_changes)[:300]}
  Predicted changes: {str(pred_changes)[:300]}
"""

            # Attach images for the first N failing examples so gpt-5.2 can SEE
            # what the model failed on and write better visual rules.
            if i < _max_image_examples:
                img_orig = getattr(ex, "image_original", None)
                img_reimg = getattr(ex, "image_reimage", None)
                _failure_image_parts.append({
                    "type": "text",
                    "text": f"\n--- Failing Example {i+1} images (Category: {f['category']}) ---",
                })
                if img_orig is not None and hasattr(img_orig, "url") and img_orig.url:
                    _failure_image_parts.append({
                        "type": "image_url",
                        "image_url": {"url": img_orig.url, "detail": "low"},
                    })
                else:
                    _failure_image_parts.append({"type": "text", "text": "  [Image 1 (original): MISSING / None]"})
                if img_reimg is not None and hasattr(img_reimg, "url") and img_reimg.url:
                    _failure_image_parts.append({
                        "type": "image_url",
                        "image_url": {"url": img_reimg.url, "detail": "low"},
                    })
                else:
                    _failure_image_parts.append({"type": "text", "text": "  [Image 2 (reimage): MISSING / None]"})

        # Build the output format section based on instructions_only
        # NOTE: A JSON list IS valid JSON — the output format is correct.
        # We request a JSON list (array) of N instruction strings.
        if self.instructions_only:
                        output_format = f"""Return ONLY a valid JSON list with exactly {self.n_candidates} items.

Each item must be a single instruction string.

Example:
[
    "Instruction candidate 1...",
    "Instruction candidate 2..."
]"""
        else:
                        output_format = f"""Return ONLY a valid JSON list with exactly {self.n_candidates} items.

Each item must be an object with exactly these keys:
[
    {{
        "instruction": "Your improved system instruction text here...",
        "field_descs": {{
            "discard_image": "Improved description for the discard_image output field...",
            "changes": "Improved description for the changes output field..."
        }}
    }}
]"""

        # Build per-category score block so gpt-5.2 knows where to focus
        per_cat_block = ""
        if baseline_eval and baseline_eval.get("per_category_scores"):
            per_cat = baseline_eval["per_category_scores"]
            sorted_cats = sorted(per_cat.items(), key=lambda x: x[1])  # worst first
            per_cat_block = "BASELINE PER-CATEGORY SCORES (worst first — focus improvement here):\n"
            for cat, sc in sorted_cats:
                bar = "█" * int(sc * 20)
                status = "(CRITICAL — needs most attention)" if sc < 0.3 else "(needs improvement)" if sc < 0.7 else "(good — preserve)"
                per_cat_block += f"  {cat}: {sc:.3f}  {bar:<20}  {status}\n"

        # Canonical output strings — always show explicitly since field descs are empty (P1)
        canonical_outputs = """REQUIRED CANONICAL OUTPUT STRINGS for 'changes' field (exact strings the model must use):
  - Single image, informational        → changes = "NoMatchFound"
  - Single image, non-informational    → changes = "Only one image and it is non informational"
  - Either image corrupted/tiny        → changes = "Could not infer"
  - Either image fully crossed out     → changes = "Image has been crossed out"
  - Both images non-informational      → changes = "Both images are non informational and contain only logo/watermark etc. no meaningful content"
  - Both informational, trivial diffs  → changes = "Images are identical - no changes"
  - One info + one non-info            → changes = "One image informational and other non-informational: <describe informational image>"
  - Both informational, meaningful     → changes = "OVERVIEW: <summary> | DETAILS (Each Change): 1) <location>: <what changed>; 2) ..."""

        meta_prompt = f"""You are a prompt engineering expert specializing in vision-language model instructions.

TASK: IMPROVE the existing system instruction for a vision-language model that compares pairs of document images and produces:
1. discard_image: 'true' or 'false'
2. changes: a descriptive string

IMPORTANT: Do NOT rewrite from scratch. Improve and extend the existing instruction.
Keep everything that already works well. Only strengthen weak areas.
The instruction is injected as the system-level docstring of a DSPy Signature class.

CURRENT INSTRUCTION (the model currently uses this — DO NOT lose any rules from it):
---
{current_instruction[:8000]}
---

{canonical_outputs}

{per_cat_block}
{_CATEGORY_REFERENCE}

FAILING EXAMPLES (from training data — model got these wrong, image pairs not included):
{failure_text}

GUIDELINES FOR YOUR {self.n_candidates} IMPROVED VARIANT(S):
1. Keep all 8 category rules — do NOT simplify or remove any
2. The canonical 'changes' output strings above are MANDATORY — the model must output them exactly for special cases
3. Fix the specific failure patterns shown above without breaking well-performing categories
4. Focus your variants on different angles:
   - Variant A: More explicit rules for the lowest-scoring categories (B2, B1)
   - Variant B: Stronger emphasis on canonical output string compliance
   - Variant C: Clearer decision tree / stepwise reasoning guide
5. Each variant must be complete and self-contained (no references to "the above")
6. Do NOT shorten the instruction — completeness is more important than brevity

{output_format}"""

        # ── Step 3: Generate N candidates ────────────────────────────
        print(f"\n[{_ts()}] Step 3/4: Generating {self.n_candidates} candidate prompts via {self.seed_model_name}...")
        step3_t0 = time.time()

        seed_lm_kwargs = dict(
            temperature=self.seed_temperature,
            cache=False,
        )
        if self.seed_reasoning_effort:
            seed_lm_kwargs["reasoning_effort"] = self.seed_reasoning_effort

        seed_lm = create_lm(self.seed_model_name, **seed_lm_kwargs)

        candidates = []
        print(f"\n  [{_ts()}] Generating all candidates in a single meta-call"
              f" (with {len(_failure_image_parts)} image parts from {_max_image_examples} failing examples)...")
        try:
            # Build multimodal content: text prompt + failing-example images
            # so gpt-5.2 can SEE what the student model got wrong.
            _content_parts: list = [{"type": "text", "text": meta_prompt}]
            if _failure_image_parts:
                _content_parts.append({"type": "text", "text": "\n\nBELOW ARE THE ACTUAL IMAGE PAIRS the model FAILED on (low-res). Study them to understand what visual patterns need better rules:"})
                _content_parts.extend(_failure_image_parts)

            response = seed_lm(
                messages=[{"role": "user", "content": _content_parts}],
            )
            text = response[0] if isinstance(response, list) else str(response)

            text = text.strip()
            if text.startswith("```"):
                text = text.split("```", 2)[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            parsed = json.loads(text)
            if isinstance(parsed, dict) and "candidates" in parsed:
                parsed = parsed["candidates"]
            if not isinstance(parsed, list):
                raise ValueError("Expected a JSON list of seed candidates")

            if len(parsed) != self.n_candidates:
                print(f"    [WARN] Requested {self.n_candidates} candidates, received {len(parsed)}")

            for i, item in enumerate(parsed[:self.n_candidates]):
                if self.instructions_only:
                    instruction = str(item).strip() if item is not None else ""
                    field_descs = {}
                else:
                    if not isinstance(item, dict):
                        print(f"    [WARN] Candidate {i+1} is not an object, skipping")
                        continue
                    instruction = str(item.get("instruction", "")).strip()
                    field_descs = item.get("field_descs", {}) or {}

                if not instruction:
                    print(f"    [WARN] Empty instruction for candidate {i+1}, skipping")
                    continue

                candidates.append({
                    "index": i,
                    "instruction": instruction,
                    "field_descs": field_descs,
                })
                print(f"    [{_ts()}] Candidate {i+1}: instruction_len={len(instruction)} "
                      f"field_descs={list(field_descs.keys())}")
                print(f"    Instruction preview: {instruction[:150]}...")

        except Exception as exc:
            print(f"    [ERROR] Failed to generate seed candidate list: {exc}")

        if not candidates:
            print("\n  [ERROR] No valid candidates generated. Falling back to baseline.")
            # Save baseline as the only seed
            baseline_path = output_dir / "seed_candidate_0_baseline.json"
            save_candidate(baseline, baseline_path, metadata={
                "stage": "seed", "config_name": "baseline_fallback",
                "student_model": self.student_model_name,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "note": "No LLM candidates generated, using default signature",
            })
            info = get_program_info(baseline)
            eval_result = evaluate_candidate(baseline, val_set, metric, self.eval_num_threads)
            entry = CandidateEntry(
                program_path=str(baseline_path),
                stage="seed",
                config_name="baseline_fallback",
                parent_path=None,
                val_cls_score=eval_result["cls_score"],
                val_change_score=eval_result["change_score"],
                val_total_score=eval_result["total_score"],
                per_category_scores=eval_result["per_category_scores"],
                n_demos=info["n_demos"],
                instruction_length=info["instruction_length"],
                timestamp=datetime.now().isoformat(timespec="seconds"),
                val_avg_total_tokens=eval_result.get("avg_total_tokens"),
                val_avg_input_tokens=eval_result.get("avg_input_tokens"),
                val_avg_output_tokens=eval_result.get("avg_output_tokens"),
                val_avg_reasoning_tokens=eval_result.get("avg_reasoning_tokens"),
                val_eval_time_seconds=eval_result.get("eval_time_seconds"),
                metadata={
                    "stage": "seed",
                    "config_name": "baseline_fallback",
                    "model": self.student_model_name,
                    "seed_model": self.seed_model_name,
                    "student_model": self.student_model_name,
                    "eval_result": eval_result,
                    "log_dir": str(log_dir) if log_dir else None,
                },
            )
            if leaderboard:
                leaderboard.add(entry)
            return [entry]

        # ── Step 4: Evaluate each candidate ──────────────────────────
        print(f"\n[{_ts()}] Step 3 done in {time.time() - step3_t0:.1f}s. "
              f"Generated {len(candidates)}/{self.n_candidates} valid candidates.")

        n_variants = len(candidates) * (2 if self.seed_add_demos_variant else 1)
        print(f"\n[{_ts()}] Step 4/4: Evaluating {n_variants} variants on val set ({len(val_set)} examples)...")
        step4_t0 = time.time()

        # Build evaluation work items: bare + optionally with demos
        eval_items = []
        for cand in candidates:
            eval_items.append({**cand, "variant": "bare"})
            if self.seed_add_demos_variant:
                eval_items.append({**cand, "variant": "with_demos"})

        entries: list[CandidateEntry] = []

        def _eval_variant(item):
            """Evaluate a single seed variant (bare or with demos)."""
            idx = item["index"]
            instruction = item["instruction"]
            field_descs = item.get("field_descs", {})
            variant = item["variant"]

            # Each worker gets its own LM via thread-local context
            worker_lm = create_lm(
                self.student_model_name,
                temperature=self.seed_eval_temperature,
                cache=False,
            )
            with dspy.settings.context(lm=worker_lm):
                variant_t0 = time.time()

                # Create module with injected instruction
                if self.use_cot:
                    module = dspy.ChainOfThought(DCAImageComparison)
                else:
                    module = dspy.Predict(DCAImageComparison)

                module.signature = module.signature.with_instructions(instruction)

                # Inject field descriptions (if not instructions_only)
                if field_descs:
                    for fname, desc in field_descs.items():
                        if fname in module.signature.fields and desc:
                            module.signature = module.signature.with_updated_fields(fname, desc=desc)

                # Add demos for the "with_demos" variant.
                # NOTE: intentionally omit image_original/image_reimage from the
                # demo Example object.  Including raw image objects as demo inputs
                # injects many images into the LM context, which reliably causes
                # structured-output parsing failures (60/60 errors, score=0.0).
                # Keeping only the gold label fields (discard_image, changes) gives
                # the model output-format guidance without the image payload.
                if variant == "with_demos" and failures:
                    demos = []
                    for f in failures[:self.max_context_examples]:
                        ex = f["example"]
                        demo = dspy.Example(
                            discard_image=str(getattr(ex, "expected_discard_image", "")),
                            changes=str(getattr(ex, "expected_changes", "")),
                        ).with_inputs("image_original", "image_reimage")
                        demos.append(demo)
                    module.demos = demos

                config_name = f"seed_candidate_{idx}{'_with_demos' if variant == 'with_demos' else ''}"
                print(f"\n  [{_ts()}] Evaluating {config_name} ({len(instruction)} char instruction)...")

                # evaluate_candidate() diffs lm.history to extract per-sample
                # avg token stats (input/output/reasoning) from the LM API JSON.
                # Pass worker_lm explicitly so inner ThreadPoolExecutor threads
                # use it (ContextVars don't propagate to pool threads < 3.12).
                eval_result = evaluate_candidate(
                    module, val_set, metric, self.eval_num_threads, lm=worker_lm,
                )

                info = get_program_info(module)
                program_path = output_dir / f"{config_name}.json"

                save_candidate(module, program_path, metadata={
                    "stage": "seed",
                    "config_name": config_name,
                    "variant": variant,
                    "seed_model": self.seed_model_name,
                    "student_model": self.student_model_name,
                    "model": self.student_model_name,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "instruction_preview": instruction[:300],
                    "eval_result": eval_result,
                })

                elapsed = time.time() - variant_t0
                avg_tok = eval_result.get("avg_total_tokens")
                tok_str = f"  avg_tok/smp={avg_tok:.0f}" if avg_tok is not None else ""
                print(f"    [{_ts()}] total={eval_result['total_score']:.4f}  "
                      f"cls={eval_result['cls_score']:.4f}  "
                      f"chg={eval_result['change_score']:.4f}  "
                      f"instruction_len={info['instruction_length']}  "
                      f"demos={info['n_demos']}{tok_str}  "
                      f"({elapsed:.1f}s)")

                return CandidateEntry(
                    program_path=str(program_path),
                    stage="seed",
                    config_name=config_name,
                    parent_path=None,
                    val_cls_score=eval_result["cls_score"],
                    val_change_score=eval_result["change_score"],
                    val_total_score=eval_result["total_score"],
                    per_category_scores=eval_result["per_category_scores"],
                    n_demos=info["n_demos"],
                    instruction_length=info["instruction_length"],
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    val_avg_total_tokens=eval_result.get("avg_total_tokens"),
                    val_avg_input_tokens=eval_result.get("avg_input_tokens"),
                    val_avg_output_tokens=eval_result.get("avg_output_tokens"),
                    val_avg_reasoning_tokens=eval_result.get("avg_reasoning_tokens"),
                    val_eval_time_seconds=eval_result.get("eval_time_seconds"),
                    metadata={
                        "stage": "seed",
                        "config_name": config_name,
                        "variant": variant,
                        "model": self.student_model_name,
                        "seed_model": self.seed_model_name,
                        "student_model": self.student_model_name,
                        "n_context_examples": len(failures),
                        "eval_result": eval_result,
                        "log_dir": str(log_dir) if log_dir else None,
                    },
                )

        # Run evaluations in parallel
        max_workers = min(len(eval_items), max(1, self.seed_parallel_evaluations))
        if max_workers <= 1:
            for item in eval_items:
                try:
                    entry = _eval_variant(item)
                    entries.append(entry)
                    if leaderboard:
                        leaderboard.add(entry)
                except Exception as exc:
                    print(f"    [ERROR] seed candidate {item['index']} ({item['variant']}): {exc}")
                    logger.exception("Seed eval failed")
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_item = {}
                for item in eval_items:
                    fut = pool.submit(_eval_variant, item)
                    future_to_item[fut] = item

                for fut in as_completed(future_to_item):
                    item = future_to_item[fut]
                    try:
                        entry = fut.result()
                        entries.append(entry)
                        if leaderboard:
                            leaderboard.add(entry)
                    except Exception as exc:
                        print(f"    [ERROR] seed candidate {item['index']} ({item['variant']}): {exc}")
                        logger.exception("Seed eval failed")

        # ── Include baseline candidates as safety-net seeds ─────────
        # The LLM-generated seeds might all be worse than the original
        # signature.  By including the baseline (bare) and optionally
        # baseline+demos as competing candidates, the top-K selection
        # guarantees we never regress below the starting point.
        print(f"\n  [{_ts()}] Adding baseline candidates as safety-net seeds...")

        # Baseline bare — evaluate on val_set (not train_set) for fair comparison
        baseline_for_val = dspy.ChainOfThought(DCAImageComparison) if self.use_cot else dspy.Predict(DCAImageComparison)
        worker_lm_bl = create_lm(self.student_model_name, temperature=self.seed_eval_temperature, cache=False)
        with dspy.settings.context(lm=worker_lm_bl):
            baseline_val_eval = evaluate_candidate(baseline_for_val, val_set, metric, self.eval_num_threads, lm=worker_lm_bl)

        baseline_val_info = get_program_info(baseline_for_val)
        baseline_val_path = output_dir / "seed_baseline_bare.json"
        save_candidate(baseline_for_val, baseline_val_path, metadata={
            "stage": "seed", "config_name": "seed_baseline_bare",
            "variant": "bare", "student_model": self.student_model_name,
            "model": self.student_model_name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "note": "Baseline (original signature) evaluated on val_set as safety-net seed",
            "eval_result": baseline_val_eval,
        })
        baseline_val_entry = CandidateEntry(
            program_path=str(baseline_val_path), stage="seed",
            config_name="seed_baseline_bare", parent_path=None,
            val_cls_score=baseline_val_eval["cls_score"],
            val_change_score=baseline_val_eval["change_score"],
            val_total_score=baseline_val_eval["total_score"],
            per_category_scores=baseline_val_eval["per_category_scores"],
            n_demos=baseline_val_info["n_demos"],
            instruction_length=baseline_val_info["instruction_length"],
            timestamp=datetime.now().isoformat(timespec="seconds"),
            val_avg_total_tokens=baseline_val_eval.get("avg_total_tokens"),
            val_avg_input_tokens=baseline_val_eval.get("avg_input_tokens"),
            val_avg_output_tokens=baseline_val_eval.get("avg_output_tokens"),
            val_avg_reasoning_tokens=baseline_val_eval.get("avg_reasoning_tokens"),
            val_eval_time_seconds=baseline_val_eval.get("eval_time_seconds"),
            metadata={"stage": "seed", "config_name": "seed_baseline_bare",
                      "variant": "bare", "model": self.student_model_name,
                      "eval_result": baseline_val_eval},
        )
        entries.append(baseline_val_entry)
        if leaderboard:
            leaderboard.add(baseline_val_entry)
        print(f"    [{_ts()}] Baseline bare: total={baseline_val_eval['total_score']:.4f}  "
              f"cls={baseline_val_eval['cls_score']:.4f}  chg={baseline_val_eval['change_score']:.4f}")

        # Baseline with demos (if demos variant enabled)
        if self.seed_add_demos_variant and failures:
            baseline_demos = dspy.ChainOfThought(DCAImageComparison) if self.use_cot else dspy.Predict(DCAImageComparison)
            demos_list = []
            for f_item in failures[:self.max_context_examples]:
                ex = f_item["example"]
                # Same image-stripping as _eval_variant: omit image fields from
                # the demo to avoid injecting large image payloads into LM context.
                demo = dspy.Example(
                    discard_image=str(getattr(ex, "expected_discard_image", "")),
                    changes=str(getattr(ex, "expected_changes", "")),
                ).with_inputs("image_original", "image_reimage")
                demos_list.append(demo)
            baseline_demos.demos = demos_list

            with dspy.settings.context(lm=worker_lm_bl):
                bl_demos_eval = evaluate_candidate(baseline_demos, val_set, metric, self.eval_num_threads, lm=worker_lm_bl)

            bl_demos_info = get_program_info(baseline_demos)
            bl_demos_path = output_dir / "seed_baseline_with_demos.json"
            save_candidate(baseline_demos, bl_demos_path, metadata={
                "stage": "seed", "config_name": "seed_baseline_with_demos",
                "variant": "with_demos", "student_model": self.student_model_name,
                "model": self.student_model_name,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "note": "Baseline + worst-example demos as safety-net seed",
                "eval_result": bl_demos_eval,
            })
            bl_demos_entry = CandidateEntry(
                program_path=str(bl_demos_path), stage="seed",
                config_name="seed_baseline_with_demos", parent_path=None,
                val_cls_score=bl_demos_eval["cls_score"],
                val_change_score=bl_demos_eval["change_score"],
                val_total_score=bl_demos_eval["total_score"],
                per_category_scores=bl_demos_eval["per_category_scores"],
                n_demos=bl_demos_info["n_demos"],
                instruction_length=bl_demos_info["instruction_length"],
                timestamp=datetime.now().isoformat(timespec="seconds"),
                val_avg_total_tokens=bl_demos_eval.get("avg_total_tokens"),
                val_avg_input_tokens=bl_demos_eval.get("avg_input_tokens"),
                val_avg_output_tokens=bl_demos_eval.get("avg_output_tokens"),
                val_avg_reasoning_tokens=bl_demos_eval.get("avg_reasoning_tokens"),
                val_eval_time_seconds=bl_demos_eval.get("eval_time_seconds"),
                metadata={"stage": "seed", "config_name": "seed_baseline_with_demos",
                          "variant": "with_demos", "model": self.student_model_name,
                          "eval_result": bl_demos_eval},
            )
            entries.append(bl_demos_entry)
            if leaderboard:
                leaderboard.add(bl_demos_entry)
            print(f"    [{_ts()}] Baseline with_demos: total={bl_demos_eval['total_score']:.4f}  "
                  f"cls={bl_demos_eval['cls_score']:.4f}  chg={bl_demos_eval['change_score']:.4f}")

        # Sort by score and print summary
        entries.sort(key=lambda e: e.val_total_score, reverse=True)
        step4_elapsed = time.time() - step4_t0

        print(f"\n[{_ts()}] Step 4 done in {step4_elapsed:.1f}s")
        print(f"\n{'='*70}")
        print(f" Stage 0 Results ({len(entries)} candidates, incl. baselines)")
        print(f"{'='*70}")
        for i, e in enumerate(entries):
            marker = " ★" if i == 0 else ""
            tok_str = f"  avg_tok/smp={e.val_avg_total_tokens:.0f}" if e.val_avg_total_tokens is not None else ""
            is_bl = " [BASELINE]" if "baseline" in e.config_name else ""
            print(f"  {i+1}. {e.config_name}  total={e.val_total_score:.4f}  "
                  f"cls={e.val_cls_score:.4f}  chg={e.val_change_score:.4f}  "
                  f"demos={e.n_demos}{tok_str}{is_bl}{marker}")

        # Log seed candidate details to JSON for later debugging
        seed_details = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "n_llm_candidates": len(candidates),
            "n_total_entries": len(entries),
            "n_baseline_entries": sum(1 for e in entries if "baseline" in e.config_name),
            "entries": [
                {
                    "rank": i + 1,
                    "config_name": e.config_name,
                    "stage": e.stage,
                    "val_total_score": round(e.val_total_score, 4),
                    "val_cls_score": round(e.val_cls_score, 4),
                    "val_change_score": round(e.val_change_score, 4),
                    "n_demos": e.n_demos,
                    "instruction_length": e.instruction_length,
                    "val_avg_total_tokens": e.val_avg_total_tokens,
                    "program_path": e.program_path,
                }
                for i, e in enumerate(entries)
            ],
        }
        seed_details_path = output_dir / "seed_stage_details.json"
        with open(seed_details_path, "w", encoding="utf-8") as f:
            json.dump(seed_details, f, indent=2, ensure_ascii=False)
        print(f"  Seed details saved → {seed_details_path}")
        print(f"{'='*70}\n")

        return entries
