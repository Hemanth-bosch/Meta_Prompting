"""Word-level comparison of the SAME LLM call's input between run.py and extract.
Reports tokens present in one prompt but ENTIRELY ABSENT in the other (real content
divergences), separate from packaging. Run with .venv (stdlib only)."""
import re, sys
from collections import Counter

RP = "apples_compare/run_py_egssig/run_20260706_200555/EGSSig_LamSymcp_1_6_0"
D = "apples_compare/extract_EGSSig_LamSymcp_1_6_0_ws2"

# DSPy structural / packaging tokens we EXPECT only on the extract side (not content).
PACKAGING = {
    "system", "user", "assistant", "role", "message", "reasoning", "respond",
    "objective", "adhering", "inputs", "outputs", "output", "input", "format",
    "structure", "field", "fields", "marker", "markers", "completed", "then",
    "produce", "your", "the", "with", "will", "be", "a", "an", "of", "to", "and",
}


def tok(t):
    return re.findall(r"[A-Za-z0-9_\$\.]+", t.lower())


def compare(a_path, b_path, label):
    a = open(a_path, encoding="utf-8", errors="replace").read()
    b = open(b_path, encoding="utf-8", errors="replace").read()
    sa, sb = set(tok(a)), set(tok(b))
    only_a = sorted(sa - sb)
    only_b = sorted(sb - sa)
    # filter packaging noise from the extract-only list to expose REAL content adds
    only_b_content = [w for w in only_b if w not in PACKAGING]
    print(f"\n{'#'*70}\n# {label}\n{'#'*70}")
    print(f"run.py : {len(a):>7} chars, {len(sa):>4} unique tokens")
    print(f"extract: {len(b):>7} chars, {len(sb):>4} unique tokens")
    print(f"\n--- tokens in run.py but ABSENT from extract ({len(only_a)}) ---")
    print("  " + ", ".join(only_a) if only_a else "  (none)")
    print(f"\n--- tokens in extract but ABSENT from run.py ({len(only_b)}; content-ish after pkg filter: {len(only_b_content)}) ---")
    print("  " + ", ".join(only_b) if only_b else "  (none)")


compare(f"{RP}/prompts/prompt_1_context.txt", f"{D}/llm_call_01_prompt.txt", "PROMPT 1 — Context Extraction")
compare(f"{RP}/prompts/prompt_2_testcase.txt", f"{D}/llm_call_02_prompt.txt", "PROMPT 2 — Test Case Generation")
