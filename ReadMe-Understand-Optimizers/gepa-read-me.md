# GEPA — The Complete Story: Reflective Prompt Evolution

> A ground-up walkthrough of every concept, data structure, code path, and parameter in GEPA.
> **Source files:** `dspy/teleprompt/gepa/gepa.py` and `dspy/teleprompt/gepa/gepa_utils.py`

---

## Table of Contents

1. [What Problem Does GEPA Solve?](#1-what-problem-does-gepa-solve)
2. [The Core Philosophy — Why Reflection Works](#2-the-core-philosophy--why-reflection-works)
3. [Key Vocabulary](#3-key-vocabulary)
4. [The Metric — GEPA's Most Important Contract](#4-the-metric--gepas-most-important-contract)
5. [ScoreWithFeedback and DSPyTrace](#5-scorewithfeedback-and-dspytrace)
6. [The DspyAdapter — The Bridge Layer](#6-the-dspyadapter--the-bridge-layer)
7. [The Evolutionary Loop — Every Step Explained](#7-the-evolutionary-loop--every-step-explained)
8. [Merge / Crossover](#8-merge--crossover)
9. [The Budget System](#9-the-budget-system)
10. [compile() — Full Setup Walk-Through](#10-compile--full-setup-walk-through)
11. [DspyGEPAResult — Reading the Results](#11-dspygepaResult--reading-the-results)
12. [Complete Parameter Reference](#12-complete-parameter-reference)
13. [Practical Usage Patterns](#13-practical-usage-patterns)
14. [Common Pitfalls and Tips](#14-common-pitfalls-and-tips)
15. [GEPA vs MIPROv2 — When to Use Which](#15-gepa-vs-miprov2--when-to-use-which)
16. [File Reference Map](#16-file-reference-map)

---

## 1. What Problem Does GEPA Solve?

### The Story

Imagine you've built a DSPy program that answers science questions. It's scoring 0.61 on your validation set. You know the problem — the default instruction is essentially a placeholder:

```
Given the fields `context`, `question`, produce the fields `answer`.
```

What you *actually* want is something like:

```
You are a precise scientific assistant. Read the context carefully.
Answer the question in one complete sentence using ONLY information
present in the context. Do not speculate beyond the given text.
```

The gap between these two instructions is obvious to a human who has seen the failures. But how do you get there automatically?

**Option 1 — Write it yourself.** You'd need to manually review dozens of failures, find patterns, and craft instructions. Doable, but slow and requires deep involvement every time your data changes.

**Option 2 — Random/grid search.** Generate 20 variations, score all 20, keep the best. This works but wastes enormous evaluation budget because most of those 20 are blind guesses that don't specifically address any observed failure.

**Option 3 — MIPROv2.** Generates N candidate instructions up front using a "proposer" LLM, then searches over them with Bayesian optimization. Good general optimizer, but it generates its candidates before seeing failure patterns.

**Option 4 — GEPA.** Runs your program, watches it fail on specific examples, reads the failure reasons, and asks a strong LLM: *"Given these exact failures and this feedback, what should the instruction say instead?"* Each new candidate is *designed* to fix observed problems rather than randomly hoping for improvement.

This is the **reflective paradigm**: generate one informed candidate at a time, each one a targeted response to observed failure modes.

### What GEPA Can and Cannot Do

| ✅ GEPA does | ❌ GEPA does not |
|---|---|
| Evolve instruction text per predictor | Add few-shot demos |
| Use natural-language failure feedback | Work without a 5-arg metric |
| Resume from checkpoint | Support a `teacher` argument |
| Merge successful variants | Tune LM temperature or other config |
| Track Pareto-optimal candidates | Handle per-predictor different valsets |

---

## 2. The Core Philosophy — Why Reflection Works

### Prompts Are Editable Policies

Think of an instruction as a behavior policy: it tells the LLM *how to behave* on this class of inputs. When a policy produces bad behavior, the right response isn't to randomly swap it for another policy — it's to understand *why* the behavior was bad and adjust accordingly.

GEPA treats every failure as a signal. A wrong answer isn't just a 0 on a scorecard; it's evidence that something in the instruction is ambiguous, missing, or actively misleading.

### Failures Have Structure

In practice, failures cluster. Your model doesn't randomly fail on everything — it fails *systematically* on a specific kind of input or because of a specific misunderstanding baked into the current instruction. For example:

- "Too verbose" → the instruction doesn't constrain output length
- "Ignored units" → the instruction never mentioned including units
- "Used outside knowledge" → the instruction never said to stay within the context

Each of these patterns, once spotted, maps directly to a targeted instruction fix.

### LLMs Can Diagnose LLMs

The key insight that makes GEPA practical: a strong LLM (the `reflection_lm`) can read a set of failure examples with feedback annotations, understand the pattern, and propose an instruction that specifically addresses those failures. This is exactly what a skilled prompt engineer does manually — GEPA automates it.

### Diversity via Pareto Thinking

Not every improved instruction will be better at *everything*. One instruction might be better at precise factual extraction while another is better at multi-step reasoning. GEPA keeps *both* alive on a Pareto frontier rather than discarding one just because its average score is slightly lower. Both become potential parents for future evolution, maintaining a diverse gene pool of prompt strategies.

---

## 3. Key Vocabulary

### Candidate

A **candidate** is a Python dictionary mapping each predictor name to an instruction string. This is the atomic unit GEPA creates, evaluates, and evolves.

```python
# The starting candidate — extracted from your original student program
candidate_0 = {
    "answer": "Given the fields `context`, `question`, produce the fields `answer`."
}

# After first reflection — targeted fix for verbosity failures
candidate_1 = {
    "answer": "Read the context carefully and answer the question in one sentence."
}

# After second reflection — targeted fix for outside-knowledge failures
candidate_2 = {
    "answer": "You are a precise assistant. Use ONLY text from the context to answer."
}
```

GEPA never directly edits your DSPy module. It builds a live program from a candidate on demand using `build_program()`, runs it, observes results, then uses those observations to propose the next candidate.

### Reflective Dataset

The **reflective dataset** is the bridge between observation and improvement. It's a structured collection of failure records, one per predictor, that the `reflection_lm` reads before proposing a new instruction.

```python
reflective_dataset = {
    "answer": [
        {
            "Inputs": {
                "context": "France is a country in Europe. Its capital is Paris.",
                "question": "What is the capital of France?"
            },
            "Generated Outputs": {
                "answer": "The capital of France is Paris, situated along the Seine..."
            },
            "Feedback": "Answer was too lengthy. Expected just 'Paris'. Be concise."
        },
        {
            "Inputs": {
                "context": "Water freezes at 0°C and boils at 100°C.",
                "question": "At what temperature does water freeze?"
            },
            "Generated Outputs": {
                "answer": "Zero degrees"
            },
            "Feedback": "The answer should include the unit: '0°C', not 'Zero degrees'."
        }
    ]
}
```

The quality of this dataset directly determines the quality of the evolved instruction. Rich, specific feedback → targeted improvements. Vague feedback like "wrong answer" → near-random mutation.

### Pareto Frontier

The **Pareto frontier** is the set of candidates where no single candidate dominates all others across all valset instances. Think of it as GEPA's "hall of promising candidates."

The key insight is that a higher *average* score doesn't mean a candidate is better in every way. Candidate A might score higher on formal reasoning questions while Candidate B scores higher on factual extraction. A simple average would make them look equal, but they're actually complementary specialists. GEPA keeps both alive because either could be the right parent for the next evolution step.

```
Example: 3 candidates evaluated on 4 valset examples

           val[0]  val[1]  val[2]  val[3]  mean
Cand 0:     0.8     0.6     0.4     1.0     0.70
Cand 1:     0.9     0.5     0.7     0.8     0.725
Cand 2:     0.7     0.9     0.6     0.9     0.775

Cand 2 has the best mean, but:
  - Cand 0 still beats it on val[0] and val[3]
  - Cand 1 still beats it on val[0]
→ None dominates the others on ALL dimensions → all three stay on the frontier
→ GEPA picks from them stochastically as parents
```

### Predictor

A **predictor** is any optimizable module in your DSPy program — `dspy.Predict`, `dspy.ChainOfThought`, `dspy.ReAct`, etc. Each predictor has its own instruction, and GEPA can evolve each one independently. A program with three predictors has three separate instruction slots, each with its own failure patterns and its own evolution trajectory.

---

## 4. The Metric — GEPA's Most Important Contract

### Why GEPA Needs a Different Kind of Metric

A standard DSPy metric is a judge: it reads a prediction and returns a score. That's enough for most optimizers, which just rank candidates by average score.

GEPA needs more. It needs a *doctor*, not just a judge. When something goes wrong, GEPA doesn't just want to know *that* it went wrong — it needs to know *why*, in natural language, so it can explain the failure to the `reflection_lm`. That's what the feedback string is for.

This is the single biggest design difference between GEPA and every other DSPy optimizer: **your metric is also the mechanism that teaches GEPA what to fix.**

### The 5-Argument Signature

Your metric must accept exactly five arguments:

```python
def my_metric(
    gold,        # dspy.Example      — the labeled ground-truth row
    pred,        # dspy.Prediction   — the full program output
    trace,       # Optional[list]    — full execution trace of the program
    pred_name,   # Optional[str]     — name of the specific predictor being diagnosed
    pred_trace,  # Optional[list]    — just this predictor's trace step
) -> Union[float, ScoreWithFeedback]:
    ...
```

This signature is validated at `__init__` time (`gepa.py L258-L263`). If your metric doesn't accept 5 arguments, GEPA will refuse to initialize.

### The Two Calling Modes

GEPA calls your metric in two distinct situations, and you need to handle both:

**Mode 1 — Overall scoring** (during valset evaluation and initial trace capture):
```python
# pred_name = None, pred_trace = None
# GEPA just wants a number: how good was the overall output?
metric(gold_example, full_prediction, captured_trace, None, None)
# → return a float
```

**Mode 2 — Predictor-level diagnosis** (during reflection dataset construction):
```python
# pred_name = "answer", pred_trace = [(...)]
# GEPA wants to know: why did THIS specific predictor fail?
metric(gold_example, full_prediction, captured_trace, "answer", trace_for_answer)
# → return ScoreWithFeedback(score=0.0, feedback="Expected 'Paris', got 'Paris, France'. ...")
```

The pattern is: check if `pred_name is not None`, and if so, return a `ScoreWithFeedback`. Otherwise return a float.

### A Complete Metric Example

```python
from dspy.teleprompt.gepa import ScoreWithFeedback

def my_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    correct = pred.answer.strip().lower() == gold.answer.strip().lower()
    score = 1.0 if correct else 0.0

    if pred_name is not None:
        # GEPA is asking: "why did this predictor fail?"
        if correct:
            return ScoreWithFeedback(score=score, feedback="Correct.")
        else:
            return ScoreWithFeedback(
                score=score,
                feedback=(
                    f"Expected '{gold.answer}' but got '{pred.answer}'. "
                    f"The answer should be taken directly from the context "
                    f"without paraphrasing."
                )
            )

    # Normal scoring call — just return the number
    return score
```

### The `pred_trace` Argument in Detail

When GEPA calls your metric in Mode 2, the `pred_trace` argument is a synthetic single-step trace built specifically for the predictor being diagnosed:

```python
pred_trace = [
    (
        <ChainOfThought: answer>,          # the predictor object
        {                                   # inputs that went INTO this predictor
            "context": "France is a country in Europe. Its capital is Paris.",
            "question": "What is the capital of France?"
        },
        Prediction(                         # what came OUT
            reasoning="France is in Europe. Paris is its capital city...",
            answer="The capital of France is Paris, situated along the Seine..."
        )
    )
]
```

Use `pred_trace[0][1]` for inputs and `pred_trace[0][2]` for outputs. This lets your metric make feedback much more specific — for example, checking whether the correct answer was actually present in the context, or whether the model's reasoning steps were logically sound.

### A Rich Feedback Example

The richer your feedback, the more surgical the reflection LM's proposed fix will be:

```python
def rag_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    gold_ans = gold.answer.strip().lower()
    pred_ans = pred.answer.strip().lower()

    if gold_ans == pred_ans:
        score = 1.0
    elif gold_ans in pred_ans or pred_ans in gold_ans:
        score = 0.5
    else:
        score = 0.0

    if pred_name is not None and pred_trace is not None:
        pred_inputs = pred_trace[0][1]
        context = pred_inputs.get("context", "")

        if score == 1.0:
            return ScoreWithFeedback(score=score, feedback="Perfect answer.")

        elif score == 0.5:
            return ScoreWithFeedback(
                score=score,
                feedback=(
                    f"Partial match. Expected: '{gold.answer}'. Got: '{pred.answer}'. "
                    f"The answer has the right info but wrong format. "
                    f"Specify exact format in the instruction."
                )
            )
        else:
            if gold.answer.lower() in context.lower():
                return ScoreWithFeedback(
                    score=score,
                    feedback=(
                        f"The correct answer '{gold.answer}' IS present in the context, "
                        f"but the model said '{pred.answer}'. Instruct the model to "
                        f"extract information verbatim from the context."
                    )
                )
            else:
                return ScoreWithFeedback(
                    score=score,
                    feedback=(
                        f"Expected '{gold.answer}', got '{pred.answer}'. "
                        f"The model may be using outside knowledge instead of the context."
                    )
                )

    return score
```

### What Happens Without Good Feedback

If `pred_name is not None` but you return just a float, GEPA generates this feedback automatically:

```python
feedback = f"This trajectory got a score of {score}."
# e.g. "This trajectory got a score of 0.0."
```

GEPA still runs, but the reflection LM receives no meaningful signal. The evolved instructions will be near-random rather than targeted. Think of it like telling a doctor "the patient is sick" without any symptoms — technically true, but unhelpful for diagnosis.

---

## 5. ScoreWithFeedback and DSPyTrace

### ScoreWithFeedback

`ScoreWithFeedback` is a `dspy.Prediction` subclass with two fields (`gepa_utils.py L22-L27`):

```python
class ScoreWithFeedback(Prediction):
    score: float    # numeric score (same scale as your metric)
    feedback: str   # natural language explanation of what went wrong
```

You can construct it like any `Prediction`:

```python
result = ScoreWithFeedback(score=0.0, feedback="Expected 'Paris', got 'London'.")
result.score     # → 0.0
result.feedback  # → "Expected 'Paris', got 'London'."
```

### DSPyTrace

A `DSPyTrace` is the full execution record of one forward pass through your program:

```python
DSPyTrace = list[tuple[Any, dict[str, Any], Prediction]]
# Each entry: (predictor_object, inputs_dict, output_Prediction)
# One entry per LLM call made during forward()
```

For a two-predictor program:

```python
trace = [
    # Step 1: "generate" predictor ran
    (
        <Predict: generate>,
        {"context": "France is in Europe. Its capital is Paris.",
         "question": "What is the capital of France?"},
        Prediction(answer="Paris, France")
    ),
    # Step 2: "judge" predictor ran
    (
        <ChainOfThought: judge>,
        {"answer": "Paris, France"},
        Prediction(
            reasoning="The answer 'Paris, France' is correct per the context.",
            verdict="correct"
        )
    )
]
```

GEPA uses this trace to match each LLM call to its predictor (by comparing signatures, not object identity, since `build_program` creates deep copies).

---

## 6. The DspyAdapter — The Bridge Layer

### The Story

The external `gepa` package (from PyPI) is a generic evolutionary optimizer — it knows nothing about DSPy modules, `dspy.Predict`, signatures, or `dspy.Evaluate`. DSPy's `DspyAdapter` is the translation layer that maps GEPA's generic operations ("build a program from this dict", "score this batch", "get feedback for this failure") onto DSPy's concrete machinery.

Think of the adapter as a contract: GEPA says "I need these three operations", and `DspyAdapter` implements them using DSPy internals.

### `build_program(candidate)` — `gepa_utils.py L76-L79`

This is how a candidate dictionary becomes a live, runnable DSPy module:

```python
def build_program(self, candidate: dict[str, str]) -> dspy.Module:
    new_prog = self.student.deepcopy()   # fresh independent copy every time
    for name, pred in new_prog.named_predictors():
        if name in candidate:
            pred.signature = pred.signature.with_instructions(candidate[name])
    return new_prog
```

Every call produces a completely independent copy of your program. This isolation is critical — without deep copying, multiple candidates would share state and corrupt each other's evaluations. The method installs candidate instructions using `.with_instructions()`, which replaces only the instruction text while keeping the rest of the signature (field names, descriptions, output format) intact.

```python
# Example: installing a candidate
candidate = {"generate": "Read the context and answer concisely in one sentence."}
prog = adapter.build_program(candidate)

prog.generate.signature.instructions
# → "Read the context and answer concisely in one sentence."

student.generate.signature.instructions   # original untouched
# → "Given the fields context, question, produce the fields answer."
```

### `evaluate(batch, candidate, capture_traces=False)` — `gepa_utils.py L80-L122`

This method has two very different code paths depending on whether traces are needed:

**Fast path (`capture_traces=False`) — for valset scoring:**

The story here is speed. Valset evaluation happens after every single reflection cycle, so it needs to be as fast as possible. GEPA uses `dspy.Evaluate` with full parallelism, no trace recording, just scores.

```python
evaluator = Evaluate(
    devset=batch,
    metric=self.metric_fn,
    num_threads=self.num_threads,
    return_all_scores=True,
    failure_score=self.failure_score,
)
res = evaluator(program)
scores = [r[2] for r in res.results]
return EvaluationBatch(outputs=outputs, scores=scores, trajectories=None)
```

**Reflection path (`capture_traces=True`) — for minibatch introspection:**

The story here is depth. Before GEPA can construct a reflective dataset, it needs to know exactly what happened inside the program — not just the final output but every intermediate LLM call. This uses `bootstrap_trace_data`, which runs the program sequentially (no parallelism) and records the full execution trace for each example.

```python
trajs = bootstrap_trace_data(
    program=program,
    dataset=batch,
    metric=self.metric_fn,
    capture_failed_parses=True,
    failure_score=self.failure_score,
)
# Each trajectory contains: trace, example, prediction, score
return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajs)
```

**Note on `FailedPrediction`:** When the LLM's response can't be parsed into the expected output fields, DSPy wraps it as a `FailedPrediction` instead of a normal `Prediction`. If `add_format_failure_as_feedback=True`, these are included in the reflective dataset with feedback about the expected output format — useful if your program frequently produces malformed outputs.

### `make_reflective_dataset(candidate, eval_batch, components_to_update)` — `gepa_utils.py L124-L220`

This is where captured traces get transformed into the structured failure records the `reflection_lm` will read.

The process for each predictor being updated:

1. For each trajectory in the minibatch, find all trace entries belonging to this predictor (matched by signature equality, not object identity — see note above about deep copies).
2. Select which trace instance to use: prefer `FailedPrediction` (parse errors need special attention), otherwise pick randomly.
3. Format inputs and outputs into readable strings (with special handling for `History` fields).
4. Call the predictor's feedback closure to get the score and feedback text.
5. Build the record `{"Inputs": ..., "Generated Outputs": ..., "Feedback": ...}`.

The feedback closure is what connects back to your metric — it calls `metric(gold, pred, trace, pred_name, pred_trace)` with the predictor-specific context, extracting the feedback string from the `ScoreWithFeedback` you return.

---

## 7. The Evolutionary Loop — Every Step Explained

### The Big Picture

The loop is the heart of GEPA. Think of it like a research cycle: observe → hypothesize → test → update knowledge → repeat. Each iteration proposes one new instruction variant, tests it, and updates the running understanding of what makes a good instruction for your task.

### INIT — Evaluate the Seed Candidate

Before the loop starts, GEPA evaluates your original program on the full validation set. This establishes the baseline score vector — one score per valset example — and initializes the Pareto frontier.

```python
val_result = adapter.evaluate(valset, seed_candidate, capture_traces=False)
# seed_candidate = {"generate": "Given the fields context, question, produce answer.", ...}
# scores per valset example: [0.8, 0.6, 0.4, 1.0, 0.3, 0.9, 0.7, 0.5]

frontier = {seed_candidate: val_result.scores}
```

Budget cost: `len(valset)` metric calls.

---

### STEP 1 — Select a Parent from the Pareto Frontier

Every new candidate is a *mutation of a parent*. The parent is chosen from the current Pareto frontier.

**Pareto strategy (default):** Randomly sample from the frontier. This maintains diversity — a parent that scores lower overall but excels on hard examples is just as likely to be chosen as the current best, giving it a chance to pass its specialized knowledge to children.

```
Frontier candidates and their mean scores:
  Cand A: mean=0.700  (specialist: good at math questions)
  Cand B: mean=0.725  (specialist: good at factual extraction)
  Cand C: mean=0.775  (generalist: good across the board)

Pareto selection: any of A, B, C can be chosen.
Why keep A and B if C is better? Because neither A nor B is dominated by C on ALL examples.
If future evolution needs to improve math questions specifically, A is the right parent.
```

**current_best strategy:** Always pick the candidate with the highest mean score. More exploitative — faster convergence but risks missing specialized improvements.

---

### STEP 2 — Sample a Reflection Minibatch

```python
minibatch = rng.sample(trainset, reflection_minibatch_size)
# Default: 3 examples
```

Why only 3? Because you don't need to observe every failure — you need to observe *enough* failures to understand the pattern. Three well-chosen failure examples, with good feedback, are more useful to the reflection LM than 30 mediocre ones. The small size also keeps trace capture fast, enabling more reflection cycles within the same budget.

---

### STEP 3 — Evaluate the Minibatch With Trace Capture

```python
eval_batch = adapter.evaluate(minibatch, parent_candidate, capture_traces=True)
```

This is the slow step, but it's worth it. GEPA runs the parent candidate on each minibatch example sequentially, recording every LLM call. The result is a rich dataset of what happened, not just whether the output was correct.

For a three-example minibatch, you might see:
```
scores = [0.0, 1.0, 0.0]   → 2 failures, 1 success
trajectories[0]: score=0.0, answer="Paris, France" (too verbose)
trajectories[1]: score=1.0, answer="0°C"           (correct)
trajectories[2]: score=0.0, answer="A molecule"    (too vague)
```

---

### STEP 4 — Decide Which Predictors to Update

```python
# If skip_perfect_score=True (default):
components_to_update = [
    name for name in predictor_names
    if not all(scores are perfect for this predictor on the minibatch)
]
```

If every minibatch example scored perfectly for a given predictor, there's nothing to improve there — skip it. This saves both reflection LM calls and the overhead of building records for a predictor that's already doing well.

---

### STEP 5 — Build the Reflective Dataset

For each predictor being updated, GEPA collects the failure records and calls your metric's feedback mode:

```python
reflective_dataset = adapter.make_reflective_dataset(
    candidate=parent_candidate,
    eval_batch=eval_batch,
    components_to_update=["generate"]
)

# Result:
# {
#   "generate": [
#     {
#       "Inputs": {"context": "France is in Europe...", "question": "Capital?"},
#       "Generated Outputs": {"answer": "The capital of France is Paris, which..."},
#       "Feedback": "Expected 'Paris' exactly. Too verbose. Return only the entity name."
#     },
#     {
#       "Inputs": {"context": "DNA is a double helix...", "question": "What is DNA?"},
#       "Generated Outputs": {"answer": "A molecule"},
#       "Feedback": "Too vague. A double helix molecule carrying genetic information."
#     }
#   ]
# }
```

Note that the `score=1.0` example is excluded — only failures (or marginal scores) are informative here.

---

### STEP 6 — Ask the `reflection_lm` to Propose a New Instruction

This is the creative step. The `reflection_lm` receives the current instruction and the failure records, and proposes an improved instruction that would address the observed patterns.

The prompt looks roughly like:

```
Current instruction:
  "Given the fields context, question, produce the fields answer."

Failed examples with expert feedback:

  Example 1:
    Inputs:
      context: "France is a country in Europe. Its capital is Paris."
      question: "What is the capital of France?"
    Generated Outputs:
      answer: "The capital of France is Paris, which is a major European city..."
    Feedback: "Expected 'Paris' exactly. Too verbose. Return only the entity name."

  Example 2:
    Inputs:
      context: "DNA is a double helix molecule carrying genetic information."
      question: "What is DNA?"
    Generated Outputs:
      answer: "A molecule"
    Feedback: "Too vague. Expected a complete one-sentence definition."

---
Propose an improved instruction:
```

→ `reflection_lm` output: *"Answer the question using ONLY text from the context. Give a concise, complete answer — not too brief, not too long. Use the exact wording from the context where possible."*

The quality of this step depends entirely on (a) the richness of your feedback strings and (b) the capability of the `reflection_lm`. A strong model like GPT-4o reading specific feedback will propose targeted fixes. A weak model reading "wrong answer" will propose vague improvements.

---

### STEP 7 — Form the New Candidate

```python
# Inherit all predictor instructions from parent
new_candidate = dict(parent_candidate)
# Replace only the predictor(s) that were updated
new_candidate["generate"] = "Answer the question using ONLY text from the context..."
# "judge" instruction stays unchanged if it wasn't in components_to_update
```

---

### STEP 8 — Evaluate on the Full Valset

```python
new_val_result = adapter.evaluate(valset, new_candidate, capture_traces=False)
# Fast parallel evaluation — this is where GEPA spends most of its budget
# new_val_result.scores = [0.9, 0.7, 0.6, 1.0, 0.5, 0.9, 0.8, 0.7]
```

This per-example score vector is what gets added to the Pareto frontier. Budget cost: `len(valset)` metric calls.

---

### STEP 9 — Update the Pareto Frontier

The new candidate's score vector is compared against every existing frontier member:

```
New candidate scores:  [0.9, 0.7, 0.6, 1.0, 0.5, 0.9, 0.8, 0.7]
Seed candidate scores: [0.8, 0.6, 0.4, 1.0, 0.3, 0.9, 0.7, 0.5]

Difference: [+0.1, +0.1, +0.2, 0.0, +0.2, 0.0, +0.1, +0.2]
All ≥ 0 and some > 0 → new candidate DOMINATES seed
→ Seed is removed from frontier; new candidate takes its place
```

If the new candidate doesn't dominate every existing member, both coexist:

```
New candidate: [0.9, 0.7, 0.6, 0.8, 0.7, 0.9, 0.8, 0.7]  (dropped on val[3])
Seed:          [0.8, 0.6, 0.4, 1.0, 0.3, 0.9, 0.7, 0.5]  (still better on val[3])
→ Neither dominates → both remain on frontier
```

---

### Loop Termination

The loop runs until `max_metric_calls` is exhausted. The final best candidate is whichever had the highest aggregate (mean) score across all candidates ever evaluated — not just the frontier, but all of them.

---

## 8. Merge / Crossover

### The Story

Sometimes GEPA discovers two successful instructions through independent evolution paths, and the best possible instruction would combine the strengths of both. Merge is GEPA's mechanism for this.

If `use_merge=True` (default), when at least two distinct candidates exist on the frontier, GEPA periodically asks the `reflection_lm` to combine them:

```
Parent A: "Use ONLY text from the context. Be exactly precise."
Parent B: "Give a complete one-sentence answer using key facts from the context."

Merge prompt:
  Combine these two successful instructions into a single superior one:
  ...

→ Merged: "Give a complete, precise one-sentence answer using ONLY key facts verbatim from the context."
```

The merged candidate is evaluated on the full valset and added to the frontier if it's not dominated. `max_merge_invocations` (default: 5) controls how many times this can happen within a run.

---

## 9. The Budget System

### The Story

Every metric call costs time and money. GEPA's budget system lets you think about cost at whatever level of abstraction you prefer: "I want a light experiment" (`auto="light"`), "I want 20 complete evaluation passes" (`max_full_evals=20`), or "I want exactly 500 metric calls" (`max_metric_calls=500`). Exactly one must be specified.

### Budget Modes

| Parameter | How it works | When to use |
|---|---|---|
| `auto="light"` | `auto_budget()` with N=6 | Quick experiments to test your metric setup |
| `auto="medium"` | N=12 | Standard runs — good default |
| `auto="heavy"` | N=18 | Thorough optimization when quality matters most |
| `max_full_evals=20` | `20 × (len(trainset) + len(valset))` metric calls | When you think in "full dataset passes" |
| `max_metric_calls=500` | Exactly 500 calls to your metric | Precise budget control |

### `auto_budget()` Formula — `gepa.py L337-L373`

```python
def auto_budget(self, num_preds, num_candidates, valset_size, minibatch_size=35, full_eval_steps=5):
    # Trial count: same formula as MIPROv2
    num_trials = int(max(
        2 * (num_preds * 2) * log2(num_candidates),
        1.5 * num_candidates
    ))

    total  = valset_size                          # initial full eval
    total += num_candidates * 5                   # bootstrap cost (approx 5 evals per candidate)
    total += num_trials * minibatch_size          # N reflection trials × minibatch size
    periodic_fulls = (num_trials + 1) // full_eval_steps + 1
    total += periodic_fulls * valset_size         # periodic full evals

    return total
```

**Worked example** — `auto="light"`, 2 predictors, valset=80:
```
num_trials = max(2×4×log2(6), 1.5×6) = max(20.7, 9) = 21 trials

total = 80          (initial full eval)
      + 30          (6 candidates × 5)
      + 735         (21 trials × 35 minibatch)
      + 400         (5 periodic full evals × 80)
      = 1245 metric calls
```

---

## 10. compile() — Full Setup Walk-Through

`compile()` is the entry point. Here's what happens before the loop starts:

**`gepa.py L395-L502`**

### Step 1: Calculate Budget

```python
if self.auto is not None:
    self.max_metric_calls = self.auto_budget(
        num_preds=len(student.predictors()),
        num_candidates=AUTO_RUN_SETTINGS[self.auto]["n"],
        valset_size=len(valset or trainset),
    )
elif self.max_full_evals is not None:
    self.max_metric_calls = self.max_full_evals * (len(trainset) + len(valset or []))
# else: max_metric_calls used directly
```

### Step 2: Finalize Valset

```python
valset = valset or trainset
```

If you don't pass a separate valset, the trainset serves double duty. This is fine for small datasets but means GEPA is simultaneously learning from and measuring against the same data — a mild form of overfitting risk.

### Step 3: Build the Feedback Map

```python
feedback_map = {
    name: feedback_fn_creator(name, pred)
    for name, pred in student.named_predictors()
}
```

One closure is created per predictor. Each closure captures `pred_name` and calls your metric with the right predictor context. This is the mechanism that bridges your metric to the reflective dataset construction.

### Step 4: Build the Adapter

```python
adapter = DspyAdapter(
    student_module=student,
    metric_fn=self.metric_fn,
    feedback_map=feedback_map,
    failure_score=self.failure_score,
    num_threads=self.num_threads,
    add_format_failure_as_feedback=self.add_format_failure_as_feedback,
    rng=rng,
)
```

### Step 5: Extract the Seed Candidate

```python
base_program = {
    name: pred.signature.instructions
    for name, pred in student.named_predictors()
}
# → {"generate": "Given the fields context, question, produce answer.",
#     "judge":    "Given the fields answer, produce verdict."}
```

This is candidate 0 — the starting point. Every evolved candidate is a variation of these original instructions.

### Step 6: Call `gepa.optimize()`

The external GEPA engine takes over, driving the evolutionary loop using the adapter. GEPA never directly sees your DSPy module — everything goes through the adapter's three methods.

### Step 7: Build the Final Program

```python
new_prog = adapter.build_program(gepa_result.best_candidate)
if self.track_stats:
    new_prog.detailed_results = DspyGEPAResult.from_gepa_result(gepa_result, adapter)
return new_prog
```

---

## 11. DspyGEPAResult — Reading the Results

When `track_stats=True`, `best_program.detailed_results` is a `DspyGEPAResult` dataclass containing the full history of the optimization run.

**`gepa.py L57-L145`:**

```python
@dataclass(frozen=True)
class DspyGEPAResult:
    candidates:                    list[Module]        # all proposed programs as live dspy.Modules
    parents:                       list[list[int]]     # lineage: parents[i] = parent indices of candidate i
    val_aggregate_scores:          list[float]         # mean valset score per candidate
    val_subscores:                 list[list[float]]   # val_subscores[i][j] = candidate i's score on valset[j]
    per_val_instance_best_candidates: list[set[int]]   # which candidates were best on each valset instance
    discovery_eval_counts:         list[int]           # budget consumed at time of each candidate's discovery
    best_outputs_valset:           list | None         # best outputs per val instance (needs track_best_outputs=True)
    total_metric_calls:            int | None
    num_full_val_evals:            int | None
    log_dir:                       str | None
    seed:                          int | None
```

### Reading Results in Practice

```python
optimizer = GEPA(metric=my_metric, reflection_lm=reflection_lm,
                 auto="light", track_stats=True)
best = optimizer.compile(student=MyProgram(), trainset=trainset, valset=valset)

r = best.detailed_results

# How many candidates were proposed?
print(len(r.candidates))   # e.g. 7

# Aggregate scores over evolution
for i, score in enumerate(r.val_aggregate_scores):
    print(f"Candidate {i}: {score:.3f}")
# Candidate 0: 0.650  (seed)
# Candidate 1: 0.712
# Candidate 2: 0.741
# Candidate 3: 0.695
# Candidate 4: 0.768
# Candidate 5: 0.755
# Candidate 6: 0.780  (best — merged from 4 and 5)

# Lineage: who were the parents?
print(r.parents[6])    # [4, 5] → merged from candidates 4 and 5

# Which valset instances were still hard for the best candidate?
for j, score in enumerate(r.val_subscores[r.best_idx]):
    if score < 0.5:
        print(f"Valset[{j}]: score={score:.2f} → hard instance")

# Budget consumed over time
for i, count in enumerate(r.discovery_eval_counts):
    print(f"Candidate {i} discovered at {count} metric calls")

# Total resource usage
print(f"Total metric calls: {r.total_metric_calls}")
print(f"Full valset evaluations: {r.num_full_val_evals}")
```

---

## 12. Complete Parameter Reference

### `GEPA.__init__()` Parameters

| Parameter | Type | Default | What it controls |
|---|---|---|---|
| `metric` | `GEPAFeedbackMetric` | **required** | 5-arg callable returning `float` or `ScoreWithFeedback`. Validated at init. |
| `auto` | `"light"\|"medium"\|"heavy"\|None` | `None` | Budget preset. Exactly one of `auto`, `max_full_evals`, `max_metric_calls` must be set. |
| `max_full_evals` | `int\|None` | `None` | Budget as N complete passes: `N × (len(trainset) + len(valset))`. |
| `max_metric_calls` | `int\|None` | `None` | Direct total budget control. |
| `reflection_minibatch_size` | `int` | `3` | Examples per reflection iteration. Keep small (3–5). |
| `candidate_selection_strategy` | `"pareto"\|"current_best"` | `"pareto"` | Parent selection: diverse vs greedy. |
| `reflection_lm` | `LM` | **required** | LM that reads failures and proposes new instructions. Use your strongest model. |
| `skip_perfect_score` | `bool` | `True` | Skip predictors that scored perfectly on the minibatch. |
| `add_format_failure_as_feedback` | `bool` | `False` | Include DSPy parse failures in reflective dataset. |
| `use_merge` | `bool` | `True` | Allow crossover between two frontier candidates. |
| `max_merge_invocations` | `int` | `5` | Max merge operations per run. |
| `num_threads` | `int\|None` | `None` | Parallelism for valset evaluation (not trace capture). |
| `failure_score` | `float` | `0.0` | Score for crashed/exception examples. |
| `perfect_score` | `float` | `1.0` | Upper bound of metric range. Adjust if your metric uses a different scale. |
| `log_dir` | `str\|None` | `None` | Save all candidates and logs. Re-running with same path **resumes** from last checkpoint. |
| `track_stats` | `bool` | `False` | Attach `DspyGEPAResult` to `program.detailed_results`. |
| `use_wandb` | `bool` | `False` | Log to Weights & Biases. |
| `wandb_api_key` | `str\|None` | `None` | W&B API key. Falls back to `WANDB_API_KEY` env var. |
| `wandb_init_kwargs` | `dict\|None` | `None` | Extra kwargs for `wandb.init`. |
| `track_best_outputs` | `bool` | `False` | Store best per-task outputs. Requires `track_stats=True`. |
| `seed` | `int\|None` | `0` | RNG seed for reproducibility. |

### `GEPA.compile()` Parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `student` | `dspy.Module` | **required** | Program to optimize. |
| `trainset` | `list[Example]` | **required** | Labeled examples for reflection minibatches. |
| `teacher` | `Module\|None` | `None` | **Not supported** — raises `AssertionError`. |
| `valset` | `list[Example]\|None` | `None` | For Pareto scoring. If `None`, trainset used for both. |

### Tuning Advice

**`reflection_lm`** — Use your strongest model. The reflection step is the bottleneck for candidate quality. If you're using GPT-4o-mini here, upgrade to GPT-4o or better. Set `temperature=1.0` for more diverse proposal variety.

**`reflection_minibatch_size`** — The default of 3 is intentionally small. Increase to 5–8 if your failures are highly diverse and a few examples don't capture the full pattern. Decrease (or keep at 3) if examples are expensive to run.

**`candidate_selection_strategy`** — Keep `"pareto"` unless you have a specific reason to be greedy. The diversity it maintains often leads to better final results through unexpected recombination.

**`use_merge`** — Leave enabled. Merge costs are modest and crossover can discover improvements unreachable by single-parent evolution.

**`log_dir`** — Always set this. If your run gets interrupted (rate limit, timeout, crash), you can resume without losing progress by running again with the same `log_dir`.

---

## 13. Practical Usage Patterns

### Minimal Setup

```python
import dspy
from dspy.teleprompt.gepa import GEPA, ScoreWithFeedback

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class QuestionAnswerer(dspy.Module):
    def __init__(self):
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, context, question):
        return self.answer(context=context, question=question)

def qa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    correct = pred.answer.strip().lower() == gold.answer.strip().lower()
    score = 1.0 if correct else 0.0

    if pred_name is not None:
        return ScoreWithFeedback(
            score=score,
            feedback=(
                "Correct!" if correct else
                f"Expected '{gold.answer}'. Got '{pred.answer}'. "
                f"Extract the answer directly from the context without paraphrasing."
            )
        )
    return score

optimizer = GEPA(
    metric=qa_metric,
    reflection_lm=dspy.LM("openai/gpt-4o", temperature=1.0, max_tokens=4096),
    auto="light",
    log_dir="./gepa_qa_run",
)

best = optimizer.compile(
    student=QuestionAnswerer(),
    trainset=trainset,
    valset=valset,
)
```

### Multi-Predictor Program with Differentiated Feedback

```python
class RAGWithJudge(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.Predict("context, question -> answer")
        self.judge    = dspy.ChainOfThought("question, answer -> verdict, confidence")

    def forward(self, question):
        context = "\n".join(self.retrieve(question).passages)
        answer  = self.generate(context=context, question=question).answer
        result  = self.judge(question=question, answer=answer)
        return dspy.Prediction(answer=answer, verdict=result.verdict,
                               confidence=result.confidence)

def rag_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    correct = pred.answer.strip().lower() == gold.answer.strip().lower()
    score = 1.0 if correct else 0.0

    if pred_name == "generate":
        pred_inputs = pred_trace[0][1] if pred_trace else {}
        context = pred_inputs.get("context", "")
        if correct:
            return ScoreWithFeedback(score=score, feedback="Correct.")
        if gold.answer.lower() in context.lower():
            return ScoreWithFeedback(
                score=score,
                feedback=(
                    f"The answer '{gold.answer}' IS in the context, "
                    f"but the model said '{pred.answer}'. Instruct it to extract verbatim."
                )
            )
        return ScoreWithFeedback(
            score=score,
            feedback=f"Expected '{gold.answer}', got '{pred.answer}'."
        )

    elif pred_name == "judge":
        verdict_correct = (pred.verdict == gold.verdict) if hasattr(gold, "verdict") else True
        return ScoreWithFeedback(
            score=score,
            feedback="Verdict matches." if verdict_correct else
                     f"Verdict should be '{gold.verdict}' not '{pred.verdict}'."
        )

    return score

optimizer = GEPA(
    metric=rag_metric,
    reflection_lm=dspy.LM("openai/gpt-4o", temperature=1.0, max_tokens=8000),
    auto="medium",
    reflection_minibatch_size=5,
    use_merge=True,
    track_stats=True,
    log_dir="./gepa_rag_run",
)

best = optimizer.compile(student=RAGWithJudge(), trainset=trainset, valset=valset)
```

### Resuming an Interrupted Run

```python
# Run 1 — may be interrupted (rate limit, timeout, etc.)
optimizer = GEPA(
    metric=my_metric,
    reflection_lm=reflection_lm,
    auto="heavy",
    log_dir="./experiment_42",   # all checkpoints saved here
)
best = optimizer.compile(student=MyProgram(), trainset=trainset, valset=valset)

# Run 2 — same log_dir = automatic resume from last checkpoint
optimizer2 = GEPA(
    metric=my_metric,
    reflection_lm=reflection_lm,
    auto="heavy",
    log_dir="./experiment_42",   # GEPA reads existing checkpoints and continues
)
best = optimizer2.compile(student=MyProgram(), trainset=trainset, valset=valset)
```

### Inspecting the Full Evolution History

```python
optimizer = GEPA(metric=my_metric, reflection_lm=reflection_lm,
                 auto="light", track_stats=True)
best = optimizer.compile(student=MyProgram(), trainset=trainset, valset=valset)

# Print all candidates sorted best to worst
result = best.detailed_results
for score, program in sorted(
    zip(result.val_aggregate_scores, result.candidates),
    reverse=True
):
    print(f"Score: {score:.3f}")
    for name, pred in program.named_predictors():
        print(f"  [{name}]: {pred.signature.instructions[:80]}...")
```

---

## 14. Common Pitfalls and Tips

### Pitfall 1: Metric Not Accepting 5 Arguments

```python
# WRONG — crashes at __init__ time
def bad_metric(example, prediction, trace=None):
    ...
# → TypeError: GEPA metric must accept five arguments

# CORRECT — even if you don't use pred_name/pred_trace, declare them
def good_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    ...
```

### Pitfall 2: Vague Feedback

```python
# WEAK — reflection LM has almost nothing to work with
feedback = f"The score was {score}."

# STRONG — specific, actionable, tells the reflection LM exactly what to change
feedback = (
    f"The model answered '{pred.answer}' but the correct answer is '{gold.answer}'. "
    f"The context explicitly states: '{relevant_sentence}'. "
    f"The instruction should tell the model to identify the specific numerical value "
    f"mentioned in the context and include its unit."
)
```

### Pitfall 3: Forgetting `valset` → Overfitting Risk

```python
# Risky: trainset used for both reflection and Pareto scoring
optimizer.compile(student=prog, trainset=trainset)

# Safer: separate held-out valset for honest Pareto tracking
optimizer.compile(student=prog, trainset=trainset, valset=valset)
```

### Pitfall 4: Always Computing Feedback (Minor Inefficiency)

```python
# Slightly inefficient: builds feedback string even for valset scoring calls
def slightly_wasteful(gold, pred, trace=None, pred_name=None, pred_trace=None):
    score = compute_score(gold, pred)
    feedback = compute_expensive_feedback(gold, pred)  # called even when pred_name is None
    return ScoreWithFeedback(score=score, feedback=feedback)

# Optimal: only build feedback when GEPA actually asks for it
def optimal(gold, pred, trace=None, pred_name=None, pred_trace=None):
    score = compute_score(gold, pred)
    if pred_name is not None:
        feedback = compute_expensive_feedback(gold, pred, pred_trace)
        return ScoreWithFeedback(score=score, feedback=feedback)
    return score   # fast path for valset evaluation
```

### Tip: Test With a Tiny Budget First

Before committing to a full `auto="medium"` run, verify your setup works:

```python
optimizer = GEPA(
    metric=my_metric,
    reflection_lm=reflection_lm,
    max_metric_calls=50,    # just enough for 1–2 reflection cycles
)
best = optimizer.compile(
    student=MyProgram(),
    trainset=trainset[:10],
    valset=valset[:10],
)
```

This validates that your metric, feedback, and program all work together before spending real budget.

### Tip: Always Set `log_dir`

```python
optimizer = GEPA(
    metric=my_metric,
    reflection_lm=reflection_lm,
    auto="medium",
    log_dir=f"./gepa_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
```

You can always discard the directory if the run succeeds. You can't recover from a crash without it.

---

## 15. GEPA vs MIPROv2 — When to Use Which

| | MIPROv2 | GEPA |
|---|---|---|
| **Optimizes** | Instructions **+** few-shot demos | Instructions only |
| **Candidate generation** | N candidates upfront via proposer LLM | One at a time via reflection |
| **Search strategy** | Bayesian (Optuna TPE) | Evolutionary (Pareto frontier) |
| **Metric signature** | `(example, pred, trace)` — 3 args | `(gold, pred, trace, pred_name, pred_trace)` — 5 args |
| **Failure feedback** | Not used | Core signal — drives every new candidate |
| **Demos bootstrapped** | Yes | No |
| **Resume support** | No | Yes (`log_dir`) |
| **Merge/crossover** | No | Yes (`use_merge=True`) |

**Use GEPA when:**
- You can write natural-language feedback explaining *why* failures happen
- Your prompts are the main bottleneck (not few-shot demos)
- Your program has multiple predictors with distinct failure modes
- You want resumable runs
- You want to understand the evolution history of your prompts

**Use MIPROv2 when:**
- You need few-shot examples in the prompt (GEPA cannot add demos)
- Your metric is purely binary (0/1) with no diagnostic information
- You have a very tight budget — MIPROv2 has lower overhead per trial
- You want a simpler metric setup (3-arg vs 5-arg)

**Rule of thumb:** If your metric can answer "why was this wrong?", use GEPA. If it can only answer "was this wrong?", use MIPROv2.

---

## 16. File Reference Map

| What you want to find | File | Lines |
|---|---|---|
| `GEPA` class definition, `__init__`, `compile` | `dspy/teleprompt/gepa/gepa.py` | L144–L502 |
| `GEPAFeedbackMetric` protocol | `gepa.py` | L26–L55 |
| `DspyGEPAResult` dataclass + properties | `gepa.py` | L57–L145 |
| Metric 5-arg validation | `gepa.py` | L258–L263 |
| `auto_budget()` formula | `gepa.py` | L337–L373 |
| `feedback_fn_creator` closure | `gepa.py` | L420–L458 |
| `ScoreWithFeedback` class | `dspy/teleprompt/gepa/gepa_utils.py` | L22–L27 |
| `DSPyTrace` type alias | `gepa_utils.py` | L22 |
| `PredictorFeedbackFn` protocol | `gepa_utils.py` | L28–L48 |
| `DspyAdapter.__init__` | `gepa_utils.py` | L56–L76 |
| `build_program()` | `gepa_utils.py` | L76–L79 |
| `evaluate()` — both modes | `gepa_utils.py` | L80–L122 |
| `make_reflective_dataset()` — full | `gepa_utils.py` | L124–L220 |