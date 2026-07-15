"""DSPy Signatures for MBD Unit Test Assist pipeline.

Pipeline:
  1. MBDContextExtraction    — Analyze weakspot equation graph, identify inputs/outputs/scenarios
  2. MBDTestCaseGeneration   — Generate structured JSON test cases from context
  3. MBDEquationSolvingDLL   — Primary: LLM proposes assignments, DLL verifies
     MBDEquationSolverRetry  — Retry prompt (separate, GEPA-optimized) on DLL mismatch
     MBDEquationSolvingLLM   — Fallback: LLM-only solving when no DLL is available

ONE-TO-ONE MAP — each signature replaces ONE run.py `prompt = generate_*() ; await call_llm(prompt)`:
  signature                run.py call site                   original prompt template
  ─────────────────────    ────────────────────────────────   ─────────────────────────────────────
  MBDContextExtraction  ≙  _step_3a_get_context   (run.py:445)  prompts/context_prompt.py
  MBDTestCaseGeneration ≙  _step_3b_generate_tc   (run.py:475)  prompts/testcase_prompt.py
  MBDEquationSolvingDLL ≙  _step_4a 1st attempt   (run.py:550)  prompts/equation_solution_prompt_dll.py
  MBDEquationSolverRetry≙  _step_4a retry         (run.py:614)  build_retry_instruction()+ DLL template
  MBDEquationSolvingLLM ≙  _step_4b (UNUSED here) (run.py:692)  prompts/equation_solution_prompt.py
  TestCoverageJudge     =  wrapper-only (M4 metric LLM judge; no run.py counterpart)

FIDELITY NOTE
-------------
The instruction docstrings below are the **verbatim original MIDAS MBD-UTA prompts**
(from MBD/weakspot_code_to_run_2/prompts/*.py), with the only change being that the
data the original f-strings interpolated inline is now supplied through the DSPy
**input fields** (whose `desc` is intentionally left empty — all guidance lives in the
instruction). This makes the GEPA *baseline* the true original prompt, and the
instruction text is what GEPA optimizes. The retry wording (originally
build_retry_instruction()) is migrated into its own signature so GEPA can optimize it.

EMPTY-desc CONVENTION (input AND output fields)
-----------------------------------------------
For the four run.py-faithful signatures (MBDContextExtraction, MBDTestCaseGeneration,
MBDEquationSolvingDLL, MBDEquationSolverRetry) every InputField AND OutputField carries
`desc=""` **on purpose**: run.py's original prompt strings contain 100% of the task
guidance, so putting any hint in a field `desc` would (a) leak text run.py never sent and
(b) create a second, un-optimized instruction surface outside GEPA's reach. The output is
therefore identified by name only; each OutputField's per-line comment records which
run.py local (`context`, `raw_response`, `solve_response`/`simulated_response`) it maps to.
Only the two [wrapper-only] signatures below (MBDEquationSolvingLLM fallback, and the M4
TestCoverageJudge) use populated `desc` strings — they have no verbatim run.py template to
be faithful to, so their guidance lives in the field descriptions instead.

FLOW MAP CONTEXT (run.py -> this wrapper)
-----------------------------------------
run.py run_pipeline() drives per-weakspot: Phase 1 _locate_weakspot (called 843),
Phase 2 _build_equation_context (848) — both PRECOMPUTED into each dspy.Example by
utils/mbd_data.py; then Phase 3a _step_3a_get_context (851), Phase 3b
_step_3b_generate_testcases (856), Phase 4 _step_4a_solve_equations_dll (878-881, DLL
path used by this wrapper) — all three INLINED in
extract_mbd_full.MBDFullPipelineModule.forward(), which calls the DSPy predictors built
from the signatures below. Each signature therefore replaces exactly one run.py
`prompt = generate_*(...); resp = await call_llm(prompt)` pair (see table above).
"""

import dspy  # [wrapper-only] DSPy framework; run.py builds prompts as plain f-strings (no DSPy) — signatures below replace those templates


# ════════════════════════════════════════════════════════════════════════════
# VERBATIM ORIGINAL PROMPT TEXT (data interpolations replaced by input-field refs)
# Source (verified 2026-07-06): MBD/weakspot_code_to_run_2/prompts/
#   context_prompt.py::generate_context_prompt (line 40),
#   testcase_prompt.py::generate_testcase_prompt (line 46),
#   equation_solution_prompt_dll.py::generate_equation_solution_prompt_dll (line 49)
#     + build_retry_instruction (line 131),
#   equation_solution_prompt.py::generate_equation_solution_prompt (line 40, fallback).
# FIDELITY: these strings are the MIDAS prompt bodies verbatim; only the inline
# f-string data slots were lifted out into the DSPy input fields declared per class.
# ════════════════════════════════════════════════════════════════════════════

# Prompt-1 seed (verbatim body of context_prompt.py:40 generate_context_prompt; data slots→input fields). LIVE GEPA text — string below is NOT to be edited.
_PROMPT1_CONTEXT_INSTRUCTIONS = """\
You are provided with the following information (supplied in the input fields below):

Below is the data you should use:
- **Equation JSON**: provided in the `equation_json` input field
- **Constant Values**: provided in the `const_values` input field
- **Weakspot Element**: provided in the `weakspot_operation` input field
- **Weakspot Context**: provided in the `weakspot_context` input field

### 1. Equation JSON
This JSON describes a complex equation broken down into multiple elements. Each element is identified by a unique key and includes properties such as:
- **instance_name** -- the name of the variable or component represented by the element.
- **function_name** -- the function or operation that this element performs.
- **arg_names** -- a list of names for the input arguments used by the element.
- **args** -- a list of references (by key) to other elements that supply the input values for this element.

### 2. Constant Values
A dictionary of constant values that are used within the equation.

### 3. Weakspot Information
- **Weakspot Element**: This is the unique key of the weakspot element in the equation.
- **Weakspot Context**: A text description providing details on how the weakspot element is intended to work (its behavior, role, and impact).

**Note:** The parameter representing the raster sample time (e.g. "dT") should be considered fixed and must not be listed among the inputs that can be altered.
---

### Task:
Using the provided data, generate a detailed context for the identified weakspot element. Your response should include:

1. **Overview of the Weakspot Element:**
   - Identify the weakspot element by its unique key.
   - Describe its role using its "instance_name" and "function_name".
   - Describe in full detail, the working of the weakspot element.

2. **Inputs That Can Be Altered:**
   - Extract and list the names of the input parameters for the weakspot element (from its "arg_names" field).
   - For each input, specify the corresponding reference from the "args" list that provides its value.
   - Get the type information about these input operations from the equations json dictionary.
   - **Exclude** any parameter that is fixed (for example, the sample time parameter or Literals or System Constants).

3. **Outputs That Can Be Checked:**
   - Identify any output variables that are directly produced by or depend on the weakspot element.
   - Describe how these outputs are connected to the weakspot element.
   - Only Local or Variable type output values can be checked.

4. **Possible Testcases for Weakspot Element:**
   - Provide a brief overview of the types of test cases that can be generated for the weakspot element.
   - Include any specific conditions or scenarios that should be tested to validate the weakspot element.
   - Keep the number of testcases to a minimum, checking the full working with the least number of testcases.

Your answer should be clear and well organized, using headings, bullet points, and tables as needed. Generate the detailed context now.
"""


# Prompt-2 seed (verbatim body of testcase_prompt.py:46 generate_testcase_prompt; data slots→input fields). LIVE GEPA text — string below is NOT to be edited.
_PROMPT2_TESTCASE_INSTRUCTIONS = """\
Generate **test cases** for the identified weakspot.

### Output Requirements:
First give the "Detailed Explanation" of the testcase.
The output should have single JSON for all the test cases and should have the following structure:
```json
[
    {
        "Pre-Condition": ["...", "...", ...],
        "Steps": [
            {
                "Actions": ["...", "...", ...],
                "Expected Results": ["...", "...", ...]
            },
            ...
        ],
        "Detailed Explanation": "..."
    },
    ...
]
```

### Formatting Guidelines:
1. **Detailed Explanation**:
   - Provide a detailed explanation of the test case, including why the steps are necessary and how they validate the weakspot.

2. **Pre-Condition**:
   - Only include value assignments in the format: "<value_name> = <value>".
   - Do not include any variables which are part of present in Action.

3. **Steps**:
   - Each step should consist of:
     - Actions: Only use **value assignments** ("<value_name> = <value>") and **wait time information** ("Wait <wait time>").
     - Expected Results: Only use **value checks** ("<value_name> = <value>") and **wait time information** ("Wait <wait time>").

4. **String Types**:
   - Only use the following types of strings in the output:
     - **Value Assignment**: "<value_name> = <value>"   (value can only be a number, do not use any equation expansion or variable names)
     - **Wait Time**: "Wait <wait time>" (time can only be a number, variable, or `'@'` for 1 raster. !!Time can only be in multiples of the dT value!!)
     - **Value Check**: "<value_name> = <value>"  (value can only be a number, comparision can only be equals operation)

### Additional Constraints:
- **NOTE** Pre-Condition and Expected Results **cannot be same variables**
- **NOTE** Pre-Condition and Actions **cannot be same variables**
- Do not add new constant values that are not part of the provided input.
- DO NOT ADD ANY VALUES from constant values in the precondition or action or result. It is only for reference.
- Ensure all steps are logically consistent with the input equation and constants.
- Do not leave any placeholder values. Fill all placeholders with a value or variable. Never use the "$opX" representation for the variables.

EXAMPLE:
(An example test-case in the exact expected format is provided in the `example_explanation_output` input field.)

### Input Details:

**Constant Values**: provided in the `const_values` input field
**Weakspot Operation**: provided in the `weakspot_operation` input field
**Weakspot Context**: provided in the `context_analysis` input field
**use this Expected Results**: provided in the `instance_names` input field

From the context, check each input type and DO NOT consider any **Literal** or **SystemConstant** as input that can be altered in the testcases.
Here, "dT" is the raster sample time and the functions can be called for every raster and values updated accordingly.
**DO NOT** generate values for dT in the testcases.

Using the provided JSON data, generate the test cases as specified.

(The equation JSON is provided in the `equation_json` input field.)

### Task:
Using the provided JSON data and context, generate detailed test cases for possible scenarios following the structure and guidelines above.
Ensure that the test cases are unique and tailored to the specific weakspot without duplicating the example provided.
Keep the number of testcases to a **MINIMUM**. Two or three are ideal and have four or five testcases at max.

Please ensure the response has the specified JSON format and adheres strictly to the outlined instructions.
The output prompt text will include the example explanation text as desired. Adjust any of the literal values or formatting as needed for your specific use case.
"""


# Prompt-4 DLL seed (verbatim body of equation_solution_prompt_dll.py:49 generate_equation_solution_prompt_dll; data slots→input fields). LIVE GEPA text — string below is NOT to be edited.
_PROMPT4_DLL_INSTRUCTIONS = """\
You are provided with a nested equation representation that computes an output value. The equation is given below (in the `equation_chain` input field).

This expression involves various components such as parameters, system constants, and variables. The nested structure may include constructs like Variable(...), Operation(...), Local(...), Parameter(...), and SystemConstant(...), which combine in different ways to produce the final output.

Assume the final output value (LHS) is known to be the value given in the `target_value` input field.

Additionally, you have the following constant values (dictionary) in the `const_values` input field:

- If a system constant from this dictionary appears in the equation, **use that exact numeric value** as the assignment.
- If a system constant does **not** appear in the equation, do not assign it.

Your task is:

1. **Only** assign **integer** numeric values to all direct inputs of the system:
   - Variable(...)
   - Parameter(...)
   - SystemConstant(...)

   and **never** assign values to Local(...), Operation(...), or Function(...).
   If you need a numeric value for something inside an Operation(...), break it down into separate lines
   that assign numeric values to the underlying Variable(...), Parameter(...), or SystemConstant(...).

2. Assign **always same values** to the Parameter(...) for every call - use same value only **not different values**.

3. **Ensure** all Variable(...), SystemConstant(...), and Parameter(...) that appear anywhere in the equation
   (even inside nested Operation(...)) are assigned a numeric value:
   - No direct input should be omitted. For instance, if SystemConstant(DSMGENN_SC) is present, it must be assigned.
   - Do not add any system constant from the dictionary if it isn't actually in the equation.

4. **MOST IMPORTANT**: You MUST format your output exactly as shown below:
   - First provide your step-by-step explanation
   - Then provide your final solution in a JSON array format within a code block
   - Each item in the array must use EXACTLY this format: "<type>(<name>) = <integer>"
   - No other text or formatting in the JSON array

## Example of Correct Output Format:
Your step-by-step explanation here...

```json
[
    "SystemConstant(SomeSystemConstant) = 10",
    "Variable(SomeVariable) = 42",
    "Parameter(SomeParameter) = -1"
]
```

## Additional Constraints:
- Do not assign values to Local(...), Operation(...), or Function(...).
- No lines like `Operation(...) = 10` or `Local(...) = 5` or `Function(...)= 1` or `SystemConstant(...) = 10`.
- If the equation references any `Operation(...)`, break down the relevant portion into the underlying variables/parameters/constants.
- If neither type is defined than it cannot be **global input**
- The final output should strictly list only the direct assignments.
- Make sure LHS = RHS holds true with your final numeric values.


Please **derive integer values** for any `Variable(...)`, `Parameter(...)`, or `SystemConstant(...)` so that **when substituted**, the equation yields the `target_value`.
"""


# FIDELITY: the ORIGINAL retry call is NOT a standalone prompt — run.py builds it by injecting
# build_retry_instruction()'s text into the locked_param_block slot of the FULL Prompt-4 DLL
# template (run.py:614-620:
#   extra  = build_retry_instruction(expected, actual, try_fc, eq_list, attempt.solved_equations)
#   prompt = generate_equation_solution_prompt_dll(eq_to_solve, lhs_value,
#                                                  filtered_const_values, locked_param_block + extra))
# So the retry LLM sees: [retry diagnostic] + [the entire Prompt-4 DLL solving rules]. To make the
# GEPA *baseline* the true original retry prompt, we compose the same two pieces. The retry is still
# its OWN signature, so GEPA optimizes it independently of the first attempt.
# STRICT-VERBATIM (2026-06-22): _PROMPT4_RETRY_DIAGNOSTIC is the FULL build_retry_instruction body
# (data→field), INCLUDING its trailing "**Output** For example given below ..." JSON block. That
# block is reproduced exactly as the original — note the leading space before "**Output**" and the
# ```json fence the original never closes (its f-string ends right after the `]`). Do NOT "tidy" the
# unclosed fence: that is literally what run.py sends, and the appended DLL rules follow it just as
# the DLL template body follows the injected retry text in run.py:617-620.
_PROMPT4_RETRY_DIAGNOSTIC = """\
The previously calculated solution does not match the expected result:

- **Expected Output:** provided in the `expected` input field
- **DLL Execution Output:** provided in the `actual` input field (obtained using DLL calls: provided in the `dll_trace` input field)

Given the equations evaluated:
(provided in the `eq_list` input field)

Your previously computed intermediate values were:
(provided in the `solved_equations` input field)

Think Step by step, And in `solved_equations` ony change the value for the variable for which verification is failing and generate the value in similar units as in **Expected Output** and keeping the Parameter value same as before, and from Please re-calculate and correct the values in these equations, ensuring that after evaluation, the final DLL verification matches the expected output (provided in the `expected` input field). Provide a clear and accurate solution.

 **Output** For example given below -- no extra text, no comments, just the JSON array:

```json
[
    "SystemConstant(SomeSystemConstant) = 10",
    "Variable(SomeVariable) = 42",
    "Parameter(SomeParameter) = -1"
]
"""

# retry seed = the verbatim retry diagnostic, then the verbatim Prompt-4 DLL rules (which carry the
# JSON-array output format + the "assign only Variable/Parameter/SystemConstant" rules the original
# retry call also includes via the DLL template).
_PROMPT4_RETRY_INSTRUCTIONS = _PROMPT4_RETRY_DIAGNOSTIC + "\n" + _PROMPT4_DLL_INSTRUCTIONS


# ════════════════════════════════════════════════════════════════════════════
# SIGNATURES
# ════════════════════════════════════════════════════════════════════════════

class MBDContextExtraction(dspy.Signature):   # Prompt 1 == run.py _step_3a_get_context (438-458); template prompts/context_prompt.py
    # Instruction = verbatim original Prompt #1 (generate_context_prompt). GEPA-optimized.
    __doc__ = _PROMPT1_CONTEXT_INSTRUCTIONS    # the frozen Prompt-1 text (GEPA seed) — DO NOT edit the string
                                               # input fields == run.py generate_context_prompt(filtered_dict) data:
    equation_json: str = dspy.InputField(desc="")        # run.py: filtered_dict["equation"]
    const_values: str = dspy.InputField(desc="")         # run.py: filtered_dict["const_values"]
    weakspot_operation: str = dspy.InputField(desc="")   # run.py: filtered_dict["weakspot_operation"] ($opN)
    weakspot_context: str = dspy.InputField(desc="")     # run.py:445 Prompt-1 ctx (context_prompt.py:55/65 get_weakspot_context)
    kb_guidelines: str = dspy.InputField(                # [wrapper-only] OPT-IN Bosch-BU test guidelines — NO run.py analog. Resolved NEW KB -> OLD KB fallback -> "" (empty) if neither has the block.
        desc="MANDATORY Bosch BU test guidelines (may be empty). If NON-EMPTY, your context analysis and "
             "the test scenarios you identify MUST reflect and require EVERY guideline listed here — treat "
             "them as hard, non-negotiable BU requirements; none may be skipped. If EMPTY, ignore this field.")

    # desc="" is intentional: all guidance is in the frozen instruction; GEPA optimizes the instruction, not field descs.
    context_analysis: str = dspy.OutputField(desc="")    # run.py:454 `context = await call_llm(prompt)` (Prompt-1 free-text output)


class MBDTestCaseGeneration(dspy.Signature):   # Prompt 2 == run.py _step_3b_generate_testcases (465-501); template prompts/testcase_prompt.py
    # Instruction = verbatim original Prompt #2 (generate_testcase_prompt). GEPA-optimized.
    __doc__ = _PROMPT2_TESTCASE_INSTRUCTIONS   # the frozen Prompt-2 text (GEPA seed) — DO NOT edit the string
                                               # input fields == run.py generate_testcase_prompt(...) args:
    context_analysis: str = dspy.InputField(desc="")   # run.py:476 `context` (Prompt-1 output)
    equation_json: str = dspy.InputField(desc="")      # run.py:476 filtered_dict["equation"]
    const_values: str = dspy.InputField(desc="")       # run.py: filtered_dict["const_values"] (carries dT)
    weakspot_operation: str = dspy.InputField(desc="") # run.py: filtered_dict["weakspot_operation"]
    instance_names: str = dspy.InputField(desc="")     # run.py:476 eq_ctx.instance_names ("use this Expected Results")
    example_explanation_output: str = dspy.InputField(desc="")  # run.py:472-474 get_weakspot_example_output(filtered_dict, weakspot_op_key) [pipeline/context_helpers.py:112]
    kb_guidelines: str = dspy.InputField(                # [wrapper-only] OPT-IN Bosch-BU test guidelines — NO run.py analog. Resolved NEW KB -> OLD KB fallback -> "" (empty) if neither has the block.
        desc="MANDATORY Bosch BU test guidelines (may be empty). If NON-EMPTY, the test cases you generate "
             "MUST satisfy EVERY guideline listed here: at minimum each guideline must be exercised by at least "
             "one test case, and NONE may be skipped — treat them as hard BU requirements (extra cases are fine). "
             "If EMPTY, ignore this field.")
    # FIDELITY: the original Prompt 2 (generate_testcase_prompt, testcase_prompt.py:46-51) takes
    # NO raster_time argument — the raster sample time dT travels inside `const_values` (and the
    # static text already references dT). So there is no `raster_time` input field here. It remains
    # an Example METADATA field used only by the metric (M3 Wait-multiple / M8 tick math).

    # desc="" is intentional: all guidance is in the frozen instruction; GEPA optimizes the instruction, not field descs.
    test_cases: str = dspy.OutputField(desc="")        # run.py:486 `raw_response` (Prompt-2 raw JSON test cases, before update_test_cases/extract_and_replace_equations at 489-490)


class MBDEquationSolvingDLL(dspy.Signature):   # Prompt 4 (1st attempt) == run.py _step_4a per-equation (550-564); template prompts/equation_solution_prompt_dll.py
    # Instruction = verbatim original Prompt #4 first-attempt (generate_equation_solution_prompt_dll).
    # STRICT-VERBATIM: no explanatory `locked_param_block` preamble sentence — the original template
    # interpolates {locked_param_block} at the top with no explanation; that content arrives here via
    # the `locked_param_block` InputField (DSPy renders it as its own labelled field block).
    __doc__ = _PROMPT4_DLL_INSTRUCTIONS        # the frozen Prompt-4 DLL text (GEPA seed) — DO NOT edit the string
                                               # input fields == run.py generate_equation_solution_prompt_dll(...) args:
    equation_chain: str = dspy.InputField(desc="")       # run.py:550 eq_to_solve (reversed assignment chain)
    target_value: str = dspy.InputField(desc="")         # run.py:551 lhs_value (RHS target)
    const_values: str = dspy.InputField(desc="")         # run.py:551 eq_ctx.filtered_const_values
    locked_param_block: str = dspy.InputField(desc="")   # run.py:551 locked_param_block (locked Parameter values)

    # desc="" is intentional: all guidance is in the frozen instruction; GEPA optimizes the instruction, not field descs.
    variable_assignments: str = dspy.OutputField(desc="")  # run.py:561 `solve_response = await call_llm(prompt)` → 564 extract_json_from_text_dll → `simulated_response`


class MBDEquationSolverRetry(dspy.Signature):   # Prompt 4 (retry) == run.py _step_4a retry branch (614-624); = build_retry_instruction()+ DLL template
    # Instruction = the original retry wording (build_retry_instruction), migrated into
    # its own signature so GEPA optimizes the retry prompt independently. The diagnostic
    # data the original interpolated is supplied via the input fields below.
    __doc__ = _PROMPT4_RETRY_INSTRUCTIONS      # frozen retry text (= retry diagnostic + DLL rules; GEPA seed) — DO NOT edit
                                               # input fields == run.py:617-620 generate_equation_solution_prompt_dll + build_retry_instruction args:
    equation_chain: str = dspy.InputField(desc="")       # run.py:617 eq_to_solve
    target_value: str = dspy.InputField(desc="")         # run.py:618 lhs_value
    const_values: str = dspy.InputField(desc="")         # run.py:618 filtered_const_values
    locked_param_block: str = dspy.InputField(desc="")   # run.py:619 locked_param_block (run.py concatenates +extra)
    expected: str = dspy.InputField(desc="")             # run.py:614 build_retry_instruction(expected, ...)
    actual: str = dspy.InputField(desc="")               # run.py:614 build_retry_instruction(..., actual, ...)
    dll_trace: str = dspy.InputField(desc="")            # run.py:614 build_retry_instruction(..., try_fc, ...)
    eq_list: str = dspy.InputField(desc="")              # run.py:614 build_retry_instruction(..., eq_list, ...)
    solved_equations: str = dspy.InputField(desc="")     # run.py:615 build_retry_instruction(..., attempt.solved_equations)

    # desc="" is intentional: all guidance is in the frozen instruction; GEPA optimizes the instruction, not field descs.
    variable_assignments: str = dspy.OutputField(desc="")  # run.py:624 `solve_response = await call_llm(prompt)` → 627 extract_json_from_text_dll → `simulated_response`


class MBDEquationSolvingLLM(dspy.Signature):   # Prompt 3 == run.py _step_4b_solve_equations_llm (676-720) LLM-only path — UNUSED here (wrapper takes the DLL path via _step_4a); template prompts/equation_solution_prompt.py::generate_equation_solution_prompt
    # NOTE: run.py's executed call is `generate_equation_solution_prompt(...)` at run.py:705 (the
    #   copy at 692 is only the dry-run print branch). This wrapper is DLL-only, so this signature is
    #   present for completeness but is not exercised by extract_mbd_full.MBDFullPipelineModule.forward().
    # Unlike the four faithful signatures above, the field `desc`s below ARE populated: there is no
    #   verbatim run.py template being tracked here, so guidance lives in the descriptions (no fidelity
    #   constraint to keep them empty). GEPA still optimizes the docstring instruction.
    """FALLBACK equation solving path (used only when DLL is unavailable).
    LLM derives numeric values for Variable/Parameter nodes so that substituting them
    back into the equation yields the target output. No external verification.

    RULES:
    - ONLY assign to Variable(...) and Parameter(...) nodes.
    - NEVER assign to Local(...), Operation(...), SystemConstant(...), Literal(...) or Function(...).
    - If the equation has a single direct input, assign that value directly.
    - Derive values step-by-step so that LHS == RHS holds with the chosen values.
    """
    equation_chain: str = dspy.InputField(   # run.py:705-706 equation_str (parts[0].strip()) arg of generate_equation_solution_prompt
        desc="Nested equation representation showing the computation chain"
    )
    target_value: str = dspy.InputField(   # run.py:705-706 lhs_value (parts[1].strip()) arg [equation_solution_prompt.py:42]
        desc="Expected final output value (numeric)"
    )
    const_values: str = dspy.InputField(   # run.py:705-706 const_values arg (const_values_dict) [equation_solution_prompt.py:43]
        desc="JSON dict of system constant values — read-only, use exact values for SystemConstants"
    )

    variable_assignments: str = dspy.OutputField(   # run.py:710 `response = await call_llm(...)` → 713 extract_json_from_text (LLM-only path, UNUSED here)
        desc="Step-by-step explanation followed by JSON array of assignments: "
             '["Variable(SomeVariable) = <value>", "Parameter(SomeParameter) = <value>"]. '
             "No SystemConstant/Local/Operation/Function assignments."
    )


class TestCoverageJudge(dspy.Signature):   # [wrapper-only] M4 coverage LLM-judge — NO run.py counterpart (run.py computes no score)
    # [wrapper-only] This signature is part of the M1..M8 metric suite in evaluation/mbd_metric.py,
    #   which has no run.py analog (run.py only emits the XLSM, it never grades). It is NOT one of the
    #   four run.py prompt seeds; its instruction and field descs are original to this wrapper, so the
    #   populated `desc`s below are intentional (no verbatim template to preserve).
    """Rigorously judge which required testing guidelines are ACTUALLY exercised by the generated
    MBD test cases for a specific ASCET block type.

    You are given the FULL generated test cases — each with its Pre-Conditions, ordered Steps
    (Actions / Wait / Expected Results) and a Detailed Explanation. Judge each numbered guideline
    INDEPENDENTLY and STRICTLY against the ACTUAL test-case content (the concrete input values,
    preconditions, steps and Expected Results), not merely the prose explanation. A guideline counts
    as COVERED only if at least one test case EXPLICITLY exercises that exact scenario — i.e. it
    provides BOTH (a) the concrete stimulus the guideline calls for (the specific input value / edge /
    state / boundary / branch / parameter setting, visible in its Pre-Conditions/Actions) AND (b) the
    expected resulting behavior or output for that scenario (visible in its Expected Results and/or
    explanation). Be conservative:
      • Merely naming the block, or generically "testing"/"reading"/"using" it, does NOT count.
      • Partial, tangential, implied, or restated-guideline mentions do NOT count.
      • A scenario without a concrete stimulus AND an expected result does NOT count.
      • When in doubt, mark it NOT covered.
    One test case may cover several guidelines only if it genuinely exercises each distinct scenario.
    Distinct scenarios must be distinguished — do not mark two different guidelines covered from a
    single generic test case that fits neither specifically.
    """
    block_type: str = dspy.InputField(   # [wrapper-only] M4 judge input; no run.py analog (fed by evaluation/mbd_metric.py resolve_guidelines)
        desc="ASCET block type name (e.g., 'Srv_Debounce', 'Srv_RampSwitch', 'LowpassT')"
    )
    guidelines: str = dspy.InputField(   # [wrapper-only] M4 judge input: the block's required KB guidelines (no run.py analog)
        desc="Numbered list of required test guidelines for this block type. "
             "Format: '1: <guideline text>\\n2: <guideline text>\\n...'"
    )
    test_cases: str = dspy.InputField(   # [wrapper-only] M4 judge input: the FULL generated test cases (preconds + steps + expected + explanation); no run.py analog
        desc="The full generated test cases to judge. Each test case includes its Pre-Conditions, its "
             "ordered Steps (Actions / Wait / Expected Results), and a Detailed Explanation. "
             "Format: 'TC1:\\n  Pre-Condition: ...\\n  Step 1: Actions=..., Expected Results=...\\n  "
             "Explanation: ...\\nTC2: ...'"
    )

    coverage_reasoning: str = dspy.OutputField(   # [wrapper-only] M4 judge explanation (surfaced in M4 detail + GEPA feedback; no run.py analog)
        desc="Per-guideline reasoning. For EACH guideline number (1..N) state COVERED or NOT COVERED, and "
             "for covered ones name which test case (TC#) exercises it and how (the concrete stimulus + "
             "expected result); for NOT-covered ones briefly say what test case would be needed. Keep it "
             "concise — one short line per guideline."
    )

    covered_indices: str = dspy.OutputField(   # [wrapper-only] M4 judge verdict: which guideline numbers are covered (drives coverage score; no run.py analog)
        desc="Comma-separated list of guideline numbers that are covered by at least "
             "one test case. Example: '1,3,4' or '1,2,3,4,5'. "
             "Empty string '' if no guidelines are covered. "
             "Only output the numbers, nothing else."
    )
