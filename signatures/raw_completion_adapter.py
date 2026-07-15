"""RawCompletionAdapter — a run.py-faithful DSPy adapter for the MBD-UTA pipeline.

RELATIONSHIP TO run.py
----------------------
[wrapper-only] This file has NO run.py analog. run.py never used a DSPy adapter at all: it called the
LLM directly through llm_client.call_llm -> AsyncAzureOpenAI.chat.completions.create (llm_client.py:119)
in PLAIN TEXT (temperature=0.0, NO response_format). This adapter exists solely to RESTORE that
plain-text transport under DSPy predictors, so the model still emits the ```json fence and run.py's
fence-only `extract_json_from_text_dll` (equation_solver.py:65; regex ``re.findall(r"```json\\s*\\n(.*?)\\n```")``
at equation_solver.py:75/79) sees the fenced JSON array exactly as it did in the original pipeline.
It changes only the LLM transport layer — the verbatim prompt seeds (signatures/mbd.py) and every
run.py-faithful pipeline step are untouched.

WHY THIS EXISTS
---------------
The MBD pipeline's Prompt-4 DLL solver is designed (in the original MIDAS run.py) to reply with
"step-by-step explanation + a ```json [...]``` fenced array", which run.py parses via
`extract_json_from_text_dll` (pipeline/equation_solver.py) — a FENCE-ONLY regex search
(``re.findall(r"```json\\s*\\n(.*?)\\n```")``). run.py works because it calls the LLM in PLAIN TEXT
(`chat.completions.create`, no `response_format`), so the model emits the fence and the regex finds it.

The DSPy wrappers use predictors (needed so GEPA can optimize the prompt instructions), and DSPy's
built-in adapters DESTROY that fence:
  * JSONAdapter forces a structured JSON object -> no fenced array -> ~81% of solve attempts return
    no parseable JSON (measured by the solver-JSON-health counter) -> M7/M8 pinned, retries wasted.
  * ChatAdapter demands ``[[ ## field ## ]]`` markers and, when the marker-less solver reply fails its
    parse (chat_adapter.py: ChatAdapter.parse raises AdapterParseError), ChatAdapter.__call__ CATCHES
    the error and re-routes to JSONAdapter -> same mangling + extra round-trips.

THE FIX
-------
`RawCompletionAdapter` reuses ChatAdapter's message formatting but:
  1. KILLS the JSONAdapter fallback — it runs the base `Adapter` call path directly (ChatAdapter's
     `__call__`/`acall` is the only place that catches parse errors and re-routes to JSONAdapter).
  2. Parses TOLERANTLY — if the model used ``[[ ## markers ## ]]`` we parse normally; otherwise we put
     the RAW completion into the (real) output field, exactly like run.py hands the raw text to
     `extract_json_from_text_dll`. Every MBD predictor has one substantive output field
     (context_analysis / test_cases / variable_assignments; ChainOfThought prepends `reasoning`), and
     downstream code only reads that field, so "raw completion -> last output field" is faithful.

Net: the solver's ```json fence survives to `extract_json_from_text_dll`, Prompt-2's fence-free
`[{...}]` regex still works, and Prompt-1 free text is unaffected — i.e. the LLM transport now matches
run.py. This is a transport-layer change only; the verbatim prompt seeds are untouched, so GEPA still
optimizes the same instructions.
"""
from __future__ import annotations  # [wrapper-only] PEP 563 lazy annotations (dict[str,Any] hint at line 69/79); no run.py analog

from typing import Any  # [wrapper-only] type hint for the tolerant-parse return dict; run.py has no adapter/typing here

# [wrapper-only] DSPy adapter machinery — run.py had NO adapter (called AsyncAzureOpenAI directly, llm_client.py:119)
from dspy.adapters.base import Adapter  # [wrapper-only] base __call__/acall path we call directly to SKIP ChatAdapter's JSON fallback
from dspy.adapters.chat_adapter import ChatAdapter  # [wrapper-only] subclassed for its message formatting + [[ ## field ## ]] parse
from dspy.signatures.signature import Signature  # [wrapper-only] type of the signature arg to parse() (output_fields access at line 78)


class RawCompletionAdapter(ChatAdapter):  # [wrapper-only] restores run.py's plain-text transport (no response_format) so the ```json fence survives to extract_json_from_text_dll (equation_solver.py:75)
    """ChatAdapter minus the JSONAdapter fallback, with a raw-completion parse fallback (run.py style)."""

    # ── 1) Kill the JSONAdapter fallback ─────────────────────────────────────────
    # ChatAdapter.__call__/acall wrap super().__call__ in a try/except that re-routes ANY failure to
    # JSONAdapter. Calling the base Adapter path directly runs format -> lm() -> _call_postprocess ->
    # self.parse (our tolerant parse below) with NO fallback.
    def __call__(self, lm, lm_kwargs, signature, demos, inputs):  # [wrapper-only] override ChatAdapter.__call__ (sync predictor entry point)
        return Adapter.__call__(self, lm, lm_kwargs, signature, demos, inputs)  # [wrapper-only] BASE path (format→lm()→parse), skipping ChatAdapter's except→JSONAdapter re-route

    async def acall(self, lm, lm_kwargs, signature, demos, inputs):  # [wrapper-only] override ChatAdapter.acall (async predictor entry point, e.g. GEPA)
        return await Adapter.acall(self, lm, lm_kwargs, signature, demos, inputs)  # [wrapper-only] BASE async path, same reason — no JSONAdapter fallback

    # ── 2) Tolerant parse (markers if present, else raw completion) ──────────────
    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:  # [wrapper-only] override ChatAdapter.parse; called by base _call_postprocess with NO fallback wrapper
        try:  # [wrapper-only] first try strict marker parse; fall through to raw-completion on any failure
            # Model used [[ ## field ## ]] markers -> parse exactly like ChatAdapter.
            return super().parse(signature, completion)  # [wrapper-only] ChatAdapter.parse: splits on [[ ## field ## ]] markers (raises AdapterParseError if absent)
        except Exception:  # [wrapper-only] marker-less reply (the run.py solver case) — ChatAdapter would re-route to JSONAdapter; we DON'T
            # No markers (the run.py case for the solver): hand the RAW completion to the real output
            # field (the LAST one; ChainOfThought prepends `reasoning`, which downstream ignores). The
            # per-prompt regex (extract_json_from_text_dll / the [{...}] test-case regex) then extracts
            # from it, exactly as run.py extracts from its raw LLM text.
            names = list(signature.output_fields)  # [wrapper-only] declared output field names (e.g. reasoning, test_cases); order preserved
            out: dict[str, Any] = {name: "" for name in names}  # [wrapper-only] init all output fields empty so DSPy Prediction has every declared key
            if names:  # [wrapper-only] guard against a signature with zero output fields
                out[names[-1]] = completion  # [wrapper-only] raw text → LAST field == run.py handing raw LLM text to extract_json_from_text_dll (equation_solver.py:75)
            return out  # [wrapper-only] dict consumed by DSPy to build the Prediction (matches ChatAdapter.parse return shape)
