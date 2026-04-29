# BootstrapFewShot — Complete Deep Dive

> **One-line pitch**: BootstrapFewShot is the foundational DSPy teleprompter that automatically generates *in-context demonstrations* for every predictor in your program by running a teacher program on your training data, keeping only the traces that pass the metric, and injecting them as few-shot examples into the student prompts.

---

## B-0 — Why Few-Shot Examples Matter (and Why Manual Ones Are Hard)

### 🗣️ The Story

Imagine you're training a new employee to answer customer support tickets. You could hand them a policy manual and say "figure it out." But the faster, more reliable approach is to show them a stack of *solved tickets*: "Here's a complaint we received, and here's how we responded." Those example tickets — demonstrations — teach tone, format, and reasoning style far more effectively than abstract instructions.

LLMs work the same way. You can describe a task with a system prompt, but dropping a handful of real *input → output* examples directly into the prompt (called "few-shot demonstrations") dramatically sharpens accuracy. The model thinks: *"Oh, I see the pattern — here's what a good answer looks like."*

The catch is that writing those demonstrations by hand is genuinely painful, especially when your program has multiple chained steps. You'd need to write demos for *each* LLM call in the chain, in exactly the format DSPy uses internally (matching field names, ordering, etc.). One wrong field name and the demos become noise. And the demos you hand-craft might not even be the ones the model finds most helpful — you're essentially guessing.

BootstrapFewShot's insight is: **don't write the demos yourself — let the program generate its own.** Run your program on your training data, observe what it produces, and keep the examples where the output was correct. Those become your demonstrations. The program teaches itself.

---

Every `Predict` step in a DSPy program wraps a call to the LLM. The prompt contains the signature (instructions + field names) followed by optional *demonstrations*:

```
[SYSTEM / INSTRUCTIONS]

[FEW-SHOT DEMO 1]
question: What is 2+2?
answer: 4

[FEW-SHOT DEMO 2]
question: What is the capital of France?
answer: Paris

[LIVE QUERY]
question: Explain black holes.
answer:
```

Good demonstrations dramatically improve accuracy — but writing them by hand is:
- **Tedious**: you need demos for each predictor in a multi-hop chain
- **Brittle**: they must match the exact field names and format DSPy uses
- **Sub-optimal**: hand-picked demos may not represent what the LLM actually needs

BootstrapFewShot solves all three by *running the program itself* on training data and collecting the intermediate LLM inputs/outputs as demos.

---

## B-1 — The Four-Phase Pipeline

### 🗣️ The Story

Think of BootstrapFewShot as a four-act play.

**Act 1 — Setup:** You create two versions of your program. The *student* is a blank slate with no examples at all — this is the program you'll ultimately deploy. The *teacher* is a copy that gets seeded with raw labeled data so it has some context to work from. They start from the same codebase but the teacher has a head start.

**Act 2 — Bootstrap (the core act):** You run the teacher on each training example, watching it work through the problem step by step. Every LLM call the teacher makes — every intermediate thought, every retrieved passage, every reasoning step — gets recorded in a *trace*. After the teacher finishes, you check: did it get the right answer? If yes, all those recorded intermediate steps become gold-standard demonstrations. If no, you throw the trace away and move on.

**Act 3 — Train (demo injection):** Now you take the best traces you collected and inject them as demonstrations into the student. The student is no longer blank — it has real, verified examples of good intermediate reasoning baked into every step of its prompt.

**Act 4 — Ship:** You return the newly-equipped student, ready to perform better than it ever could from scratch.

---

```
trainset
    |
    v
[1] PREPARE
    student = student.reset_copy()         <- blank slate (no demos)
    teacher = student.deepcopy()           <- pre-loaded with raw labels via LabeledFewShot
    |
    v
[2] BOOTSTRAP  (_bootstrap)
    For each example in trainset:
        run teacher(**example.inputs())    <- full program forward pass
        collect dspy.settings.trace        <- every predictor call captured
        evaluate metric(example, pred, trace)
        if passes -> store trace steps as augmented demos
    |
    v
[3] TRAIN  (_train)
    For each predictor in student:
        demos = augmented_demos[:max_bootstrapped_demos]
                + random sample of raw_demos up to max_labeled_demos budget
        predictor.demos = demos
    |
    v
[4] RETURN student   (student._compiled = True)
```

---

## B-2 — Constructor Parameters Explained

### 🗣️ The Story

When you instantiate `BootstrapFewShot`, you're essentially configuring *how picky* and *how generous* the compilation process will be.

The `metric` is your quality gate. Think of it as the grader sitting at the end of each teacher run. Without a metric, every trace gets accepted regardless of correctness — useful for quickly checking that the machinery works, but not for real quality. The `metric_threshold` lets you accept "good enough" answers when your metric returns a float score rather than a binary pass/fail.

The `max_bootstrapped_demos` and `max_labeled_demos` parameters control the *shape* of each predictor's demo list. Bootstrapped demos are richer — they include intermediate reasoning steps — but they also make prompts longer and more expensive. Raw labeled demos are leaner. The two parameters let you find the right balance for your budget and task.

`max_rounds` is a patience setting. If the teacher fails a particular training example on the first attempt, do you give up and move on, or do you try again with some randomness injected (higher temperature)? Setting this to 2 or 3 is a cheap way to squeeze more bootstrapped demos out of hard examples.

`teacher_settings` is where the distillation trick lives — you can point the teacher at a more powerful (and expensive) model like GPT-4o while the student uses a cheap model like GPT-4o-mini. The teacher does the heavy lifting once at compile time; the student reaps the benefits forever.

---

```python
BootstrapFewShot(
    metric=None,               # callable(example, prediction, trace) -> bool/float
    metric_threshold=None,     # if float metric: accept if metric_val >= threshold
    teacher_settings={},       # extra dspy.settings context for teacher LM
    max_bootstrapped_demos=4,  # hard cap on augmented (LLM-traced) demos per predictor
    max_labeled_demos=16,      # total demos per predictor (augmented + raw labels)
    max_rounds=1,              # retry attempts per example if metric fails
    max_errors=None,           # crash after this many exceptions
)
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_bootstrapped_demos` | 4 | How many *augmented* (traced) demos per predictor. More = richer context, larger prompt. |
| `max_labeled_demos` | 16 | Total demo budget. Remaining slots after augmented are filled with raw labels. |
| `max_rounds` | 1 | If teacher fails metric on example N, retry with `temperature=1.0` up to this many times. |
| `metric_threshold` | None | For float metrics: only accept demos where metric_val >= threshold. |
| `teacher_settings` | {} | Pass `{"lm": stronger_model}` to use a better LM for tracing. |

---

## B-3 — Phase 1: Prepare (Source Walkthrough)

### 🗣️ The Story

Before any learning happens, BootstrapFewShot needs to set the stage carefully. This phase is about creating two separate, independent copies of your program that will play different roles.

Think of it like preparing for a rehearsal. You have the *original script* (your program). You make one copy for the understudy (the student) — this copy is wiped clean, no notes in the margins, no prior performance history. Then you make another copy for the experienced lead actor (the teacher) — this copy gets annotated with a few worked examples from the training data so the teacher has some context before going on stage.

Why do they need to be separate objects in memory? Because during bootstrapping, the teacher will have demonstrations loaded into its predictors. If student and teacher were the same object, every time you updated the teacher's demos you'd accidentally change the student too. Separate objects = clean isolation.

The teacher's head start (via `LabeledFewShot`) is important: it loads up to `max_labeled_demos` raw examples into the teacher before the tracing begins. A teacher with zero examples is likely to produce mediocre traces. A teacher with 10–16 plain `input → label` examples already in its context will generate much better intermediate reasoning — and those better traces are exactly what gets injected into the student.

---

```python
# bootstrap.py  L91-L101
def _prepare_student_and_teacher(self, student, teacher):
    self.student = student.reset_copy()          # (A) blank student

    self.teacher = teacher.deepcopy() if teacher is not None else student.deepcopy()  # (B)

    assert getattr(self.student, "_compiled", False) is False, "Student must be uncompiled."

    if self.max_labeled_demos and getattr(self.teacher, "_compiled", False) is False:
        teleprompter = LabeledFewShot(k=self.max_labeled_demos)   # (C)
        self.teacher = teleprompter.compile(self.teacher.reset_copy(), trainset=self.trainset)
```

**(A) `reset_copy()`** — creates a fresh copy with all predictor `.demos = []`. This is the program being trained.

**(B) `deepcopy()`** — teacher is a full deep copy of the original (or the passed teacher). If `teacher=None`, the student is deep-copied. They are separate objects.

**(C) `LabeledFewShot`** — before the teacher starts tracing, it is pre-loaded with up to `k=max_labeled_demos` raw labeled examples from the trainset. This gives it good raw context for generating quality traces.

```
original program
        |
        +--reset_copy()--> student  (demos=[])
        |
        +--deepcopy()---> teacher  (demos=raw labels via LabeledFewShot)
```

---

## B-4 — Phase 2: Bootstrap Loop (Source Walkthrough)

### 🗣️ The Story

This is the heart of the whole operation. Imagine a film director watching rehearsals and keeping only the takes that feel right.

For each training example, BootstrapFewShot does something deceptively simple: it runs the *teacher* program on that example as if it were a real inference call. The teacher works through the full pipeline — retrieval, reasoning, generation — and every single LLM call along the way is silently recorded into a trace log. It's like having a court reporter present for every step of the teacher's thought process.

Once the teacher finishes, the metric gets called: was the final answer correct? If yes, the director shouts "print it!" — all the intermediate steps from that trace get saved as high-quality demonstrations. If no, the director shouts "cut!" — that trace is discarded, it's not good enough to teach from.

There's one clever guard built in: before running the teacher on a given example, BootstrapFewShot checks if that example is already in the teacher's demo list. If so, it temporarily removes it. Why? Because if the teacher literally has the answer in front of it (as a demonstration), it would trivially copy it — producing a trace that appears correct but was essentially "cheating." That demo would then contaminate the student with a false sense of how to reason through hard problems. By stripping self-contaminating examples first, every trace the teacher produces represents genuine reasoning.

The retry logic (`max_rounds`) gives the teacher a second chance on difficult examples. On a retry, temperature gets bumped to 1.0 and the cache is bypassed, forcing a fresh generation. This is especially useful for tricky examples where the first attempt was unlucky rather than truly beyond the model's ability.

---

### Outer loop

```python
# bootstrap.py  L153-L183
def _bootstrap(self, *, max_bootstraps=None):
    max_bootstraps = max_bootstraps or self.max_bootstrapped_demos
    bootstrapped = {}
    self.name2traces = {name: [] for name in self.name2predictor}

    for example_idx, example in enumerate(tqdm.tqdm(self.trainset)):
        if len(bootstrapped) >= max_bootstraps:   # (A) early exit
            break

        for round_idx in range(self.max_rounds):  # (B) retry loop
            if self._bootstrap_one_example(example, round_idx):
                bootstrapped[example_idx] = True
                break
```

**(A)** Once `max_bootstrapped_demos` successes are collected, iteration stops. BootstrapFewShot is **linear** — it does not try the entire trainset if the budget is met early.

**(B)** On `round_idx > 0`, the LM is copied with `temperature=1.0` and a fresh `rollout_id` to bypass the cache and get a new generation.

### One example

```python
# bootstrap.py  L185-L265  (simplified)
def _bootstrap_one_example(self, example, round_idx=0):
    with dspy.settings.context(trace=[], **self.teacher_settings):
        lm = dspy.settings.lm
        lm = lm.copy(rollout_id=round_idx, temperature=1.0) if round_idx > 0 else lm

        # Remove current example from teacher demos (self-contamination guard)
        for name, predictor in teacher.named_predictors():
            predictor.demos = [x for x in predictor.demos if x != example]  # (A)

        prediction = teacher(**example.inputs())   # (B) full forward pass
        trace = dspy.settings.trace                # (C) list of (predictor, inputs, outputs)

        for name, predictor in teacher.named_predictors():
            predictor.demos = predictor_cache[name]   # (D) restore

    if self.metric:
        metric_val = self.metric(example, prediction, trace)
        success = metric_val >= self.metric_threshold if self.metric_threshold else metric_val
    else:
        success = True   # (E) no metric -> accept all

    if success:
        for step in trace:
            predictor, inputs, outputs = step
            demo = dspy.Example(augmented=True, **inputs, **outputs)  # (F)
            name2traces[predictor_name].append(demo)

    return success
```

**(A) Self-contamination guard** — if the current example is already in the teacher's demo list, it is stripped before inference so the teacher cannot copy the answer.

**(B) Full forward pass** — runs the entire multi-hop pipeline. Every `Predict.__call__` appends `(predictor, inputs_dict, outputs_dict)` to `dspy.settings.trace`.

**(C) trace** — `List[Tuple[predictor_obj, inputs_dict, outputs_dict]]`. A 3-predictor chain produces 3 steps per example.

**(D) Restore** — teacher's demo list is restored after inference so the next example runs with full context.

**(E) metric=None** — accepts all examples. Useful for testing or when quality filtering is not desired.

**(F) augmented=True** — marks this demo as traced. DSPy uses this flag to render intermediate fields (e.g., `reasoning` from ChainOfThought) when building the prompt.

---

## B-5 — The Trace Data Structure

### 🗣️ The Story

The trace is the secret ingredient — the thing that makes BootstrapFewShot different from just shuffling labeled data around.

When you run a DSPy program normally, you see only the final output. The trace captures *everything in between*: every predictor that was called, the exact inputs it received at that moment in the pipeline, and the exact output it produced. It's the difference between seeing someone's finished essay and seeing their entire drafting process — scratch notes, crossed-out sentences, restructured paragraphs, and all.

For a multi-step program like `Retrieve → Generate`, the trace records two entries: one for the retrieval step (what query went in, what passages came out) and one for the generation step (what context plus question went in, what answer came out). Each of those entries will eventually become a demonstration for its respective predictor — giving the student concrete, contextually-grounded examples of exactly what each step in the chain is supposed to do.

The `augmented=True` flag on each saved demo is critical. It tells DSPy's prompt renderer: "this demo has intermediate fields (like `reasoning`) — render them too." Without it, only top-level inputs and outputs would be shown, and all that rich intermediate thinking would be silently dropped.

---

```python
dspy.settings.trace  # List[Tuple[predictor_obj, inputs_dict, outputs_dict]]
```

Example for a `Retrieve -> Generate` program on one training example:

```python
[
    (
        <Predict: retrieve>,
        {"question": "What causes thunder?"},
        {"passages": ["Sound from lightning...", "Thunder is..."]}
    ),
    (
        <Predict: generate>,
        {"question": "What causes thunder?", "passages": "Sound from lightning..."},
        {"answer": "Thunder is caused by the rapid expansion of air heated by lightning."}
    )
]
```

Each tuple becomes one `augmented=True` demo injected into that predictor's `.demos` list in the student.

---

## B-6 — Phase 3: Train (Source Walkthrough)

### 🗣️ The Story

By the time `_train()` runs, the hard work is done. You have a collection of high-quality traced demonstrations — the "augmented" demos — for each predictor. Now it's just a matter of packaging them into the student.

Think of it like assembling a textbook. The front pages get the rich, worked examples with step-by-step reasoning (the augmented demos). The remaining pages get filled with simpler reference examples — just input and correct answer, no reasoning (the raw labeled demos). Together they give the student model both deep guidance (the augmented traces) and broad coverage (the raw demos).

The raw demos come from whichever training examples *didn't* produce successful traces — either they failed the metric or the bootstrap loop hit its budget cap before reaching them. They're not useless; they still provide valid `input → label` pairs that give the LLM basic familiarity with the task format.

One subtle but important detail: the random seed is fixed at 0. This means every time you compile the same program with the same data, you get the same raw demo selection. Reproducibility is a first-class concern here — if the raw sample were random, debugging compilation problems would be a nightmare.

---

```python
# bootstrap.py  L257-L266
def _train(self):
    rng = random.Random(0)        # deterministic, fixed seed
    raw_demos = self.validation   # trainset examples NOT successfully bootstrapped

    for name, predictor in self.student.named_predictors():
        augmented_demos = self.name2traces[name][: self.max_bootstrapped_demos]

        sample_size = min(self.max_labeled_demos - len(augmented_demos), len(raw_demos))
        sample_size = max(0, sample_size)

        raw_demos_sample = rng.sample(raw_demos, sample_size)
        predictor.demos = augmented_demos + raw_demos_sample

    return self.student
```

### Worked example

```
max_bootstrapped_demos = 4
max_labeled_demos      = 16

augmented_demos["generate"] = [d1, d2, d3, d4]   # 4 traces passed metric
remaining budget = 16 - 4 = 12
raw_demos pool = 30 examples (failed or were never reached)

-> sample 12 from 30 (deterministic via Random(0))
-> predictor.demos = [d1, d2, d3, d4, r1, ..., r12]   # 16 total
```

`self.validation` = trainset examples NOT in `bootstrapped` dict. They provide plain `input->label` context with no intermediate reasoning.

---

## B-7 — Augmented vs Raw Demos in the Rendered Prompt

### 🗣️ The Story

Once the student is compiled, you can actually look at what goes into the LLM's prompt and see the difference between the two types of demonstrations.

A **raw demo** is like a flashcard. Front: the question. Back: the answer. No explanation of how you got there, no intermediate thinking. It's compact and useful for establishing "here's what an answer looks like," but it doesn't teach *how to think*.

An **augmented demo** is like a tutor's worked example. The tutor shows the question, then walks through their reasoning out loud, *then* gives the answer. When a student sees this in a test, they're not just reminded of the correct answer format — they're shown the mental process that leads to it. For tasks requiring chain-of-thought reasoning, this is enormously more powerful.

The `augmented=True` flag on a demo is the signal DSPy uses to decide which treatment to apply when rendering the prompt. It's a small boolean with a big effect on prompt quality.

---

**Raw labeled demo:**
```
---

[[ ## question ## ]]
What causes thunder?

[[ ## answer ## ]]
The expansion of air heated by lightning.
```

**Augmented (bootstrapped) demo** — same structure, but `augmented=True` causes DSPy to also render intermediate reasoning fields captured from ChainOfThought:
```
---

[[ ## question ## ]]
What causes thunder?

[[ ## reasoning ## ]]
Thunder follows lightning because the bolt superheats the surrounding air,
which rapidly expands, creating a shockwave we hear as thunder.

[[ ## answer ## ]]
The expansion of air heated by lightning.
```

The `reasoning` field only appears in augmented demos. Raw demos have no intermediate fields — only the top-level input and label.

---

## B-8 — Teacher/Student Split: When and Why

### 🗣️ The Story

The teacher/student architecture is one of the most powerful ideas in BootstrapFewShot, and it unlocks three very different use cases depending on how you configure it.

**The default case** is the simplest: teacher and student are copies of the same program. The teacher gets a head start with some labeled examples, generates traces, and teaches itself. It's bootstrapping in the literal sense — pulling yourself up by your own bootstraps. This works surprisingly well for many tasks because even a lightly-informed copy of your program can generate better demonstrations than you'd write by hand.

**The distillation case** is where things get exciting. What if you could use a powerful, expensive model (say GPT-4o) to *generate* the demonstrations, but then have a cheap model (GPT-4o-mini) *learn from* those demonstrations at inference time? You'd pay the expensive model's cost only once (at compile time), and then deploy the cheap model that now carries the distilled wisdom of the expensive one in its prompts. This is model distillation without any gradient updates — pure prompt engineering.

**The pre-compiled teacher case** is for teams that have already spent optimization budget (e.g., with MIPROv2) and want to bootstrap a fresh student from those hard-won optimized traces rather than starting from scratch.

---

### Default (no explicit teacher)
```python
bs = BootstrapFewShot(metric=my_metric, max_bootstrapped_demos=4)
compiled = bs.compile(student=my_program, trainset=train)
# teacher = deep copy of my_program with raw labeled demos pre-loaded
```

### Distillation: stronger teacher -> cheaper student
```python
bs = BootstrapFewShot(
    metric=my_metric,
    max_bootstrapped_demos=4,
    teacher_settings={"lm": dspy.LM("openai/gpt-4o")}  # teacher uses GPT-4o
)
# student uses dspy.settings.lm (e.g., gpt-4o-mini)
compiled_student = bs.compile(student=MyProgram(), trainset=train)
```

GPT-4o generates high-quality traced reasoning -> those traces become few-shot demos for the cheaper student.

### Pre-compiled teacher
```python
pre_compiled = mipro.compile(MyProgram(), trainset=train, ...)   # MIPROv2-optimized

bs = BootstrapFewShot(metric=my_metric)
compiled_student = bs.compile(
    student=MyProgram(),           # fresh instance
    teacher=pre_compiled,          # provides optimized traces
    trainset=train
)
```

---

## B-9 — Multi-Predictor Programs: Trace Propagation

### 🗣️ The Story

One of BootstrapFewShot's most elegant properties is how it handles multi-step pipelines without any special configuration.

Imagine your program has three predictors chained together: a `Retriever`, a `Reasoner`, and a `Summarizer`. When the teacher runs one training example, all three predictors fire, and all three calls get recorded in the trace. When that trace passes the metric, *each predictor gets its own demo* — the retriever gets a demo showing what it produced for that question, the reasoner gets a demo showing how it transformed retrieved passages into a structured argument, and the summarizer gets a demo showing how it turned that argument into a final answer.

This is crucial: the demos injected into each predictor are *contextually appropriate* for that predictor's specific role. The retriever's demo doesn't contain reasoning steps — it contains queries and retrieved passages. The reasoner's demo contains passages and structured arguments. The demos naturally match what each predictor actually needs to see.

The `max_bootstrapped_demos` budget applies *independently per predictor*, so a 3-predictor chain with `max_bootstrapped_demos=4` can ultimately inject up to 4 rich demos into each of the three predictors — not 4 shared across all three.

---

For a 3-predictor chain (`Retriever -> Reasoner -> Summarizer`), one successful bootstrap on example E appends one entry to each predictor's trace list:

```
name2traces["retriever"].append(trace_step_1_from_E)
name2traces["reasoner"].append(trace_step_2_from_E)
name2traces["summarizer"].append(trace_step_3_from_E)
```

The `max_bootstrapped_demos=4` cap applies independently per predictor:

```
student.retriever.demos  = [r_E1, r_E2, r_E3, r_E4] + raw_sample
student.reasoner.demos   = [re_E1, re_E2, re_E3, re_E4] + raw_sample
student.summarizer.demos = [s_E1, s_E2, s_E3, s_E4] + raw_sample
```

The raw demos pool is shared across all predictors. Raw demos have no intermediate fields — only the top-level `example.inputs()` and label.

---

## B-10 — Complete End-to-End Example

### 🗣️ The Story

Let's walk through what happens step by step when you actually run this.

You define a program — a `CoTQA` that takes a question and produces an answer using chain-of-thought reasoning. You have six labeled examples covering simple factual questions. You define a strict exact-match metric (the predicted answer, lowercased and stripped, must exactly match the label).

When you call `bs.compile(CoTQA(), trainset=trainset)`, here's what plays out behind the scenes:

1. A blank student and a teacher with raw demos loaded are created.
2. The teacher is run on the first training example: "What is 2+2?" It generates a reasoning step ("Adding 2 and 2 gives 4") and the answer "4". The metric passes — exact match. That trace gets saved.
3. This repeats for each training example until three successes are recorded (matching `max_bootstrapped_demos=3`). The three successful traces become augmented demos.
4. The remaining three training examples (that were either skipped or failed) become raw demos — three plain `question → answer` pairs with no reasoning.
5. All six demos get assigned to `compiled.cot.demos`, and the student is returned.

When you inspect the resulting prompts, you'll see a beautiful difference: the first three demos in the prompt will show the full chain-of-thought reasoning, while the last three will show just the question and answer. Together, they give the model both process guidance and broad coverage.

---

```python
import dspy
from dspy.teleprompt import BootstrapFewShot

# 1. Define program
class CoTQA(dspy.Module):
    def __init__(self):
        self.cot = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.cot(question=question)

# 2. Build trainset
trainset = [
    dspy.Example(question="What is 2+2?",              answer="4").with_inputs("question"),
    dspy.Example(question="Tallest mountain?",          answer="Mount Everest").with_inputs("question"),
    dspy.Example(question="Gas plants absorb?",         answer="Carbon dioxide").with_inputs("question"),
    dspy.Example(question="Speed of light (km/s)?",     answer="299,792").with_inputs("question"),
    dspy.Example(question="Author of Hamlet?",          answer="Shakespeare").with_inputs("question"),
    dspy.Example(question="Boiling point of water C?",  answer="100").with_inputs("question"),
]

# 3. Metric
def exact_match(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()

# 4. Configure LM
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# 5. Compile
bs = BootstrapFewShot(
    metric=exact_match,
    max_bootstrapped_demos=3,
    max_labeled_demos=6,
)
compiled = bs.compile(CoTQA(), trainset=trainset)

# 6. Inspect injected demos
for demo in compiled.cot.demos:
    aug = demo.get("augmented", False)
    print(f"{'[AUG]' if aug else '[RAW]'} Q={demo.question!r}")
# [AUG] Q='What is 2+2?'
# [AUG] Q='Gas plants absorb?'
# [AUG] Q='Speed of light (km/s)?'
# [RAW] Q='Tallest mountain?'
# [RAW] Q='Author of Hamlet?'
# [RAW] Q='Boiling point of water C?'

# 7. Use the compiled program
result = compiled(question="Who invented the telephone?")
print(result.answer)
```

### What the final prompt looks like

```
Given the fields `question`, produce the fields `answer`.

Follow the following format:

Question: ${question}
Reasoning: Let's think step by step ...
Answer: ${answer}

---

[Augmented demo 1 - has Reasoning]
Question: What is 2+2?
Reasoning: Adding 2 and 2 gives 4.
Answer: 4

---

[Augmented demo 2 - has Reasoning]
Question: Gas plants absorb?
Reasoning: Plants use CO2 for photosynthesis.
Answer: Carbon dioxide

---

[Augmented demo 3 - has Reasoning]
Question: Speed of light (km/s)?
Reasoning: The speed of light is approximately 299,792 km/s.
Answer: 299,792

---

[Raw demo - no Reasoning]
Question: Tallest mountain?
Answer: Mount Everest

---

[Raw demo - no Reasoning]
Question: Author of Hamlet?
Answer: Shakespeare

---

[Raw demo - no Reasoning]
Question: Boiling point of water C?
Answer: 100

---

Question: Who invented the telephone?
Reasoning:
```

---

## B-11 — BootstrapFewShot vs BootstrapFewShotWithRandomSearch vs MIPROv2

### 🗣️ The Story

These three optimizers exist on a spectrum of *how much optimization work they do* — and therefore how much they cost in API calls and time.

**BootstrapFewShot** is the entry point. It makes one pass, collects whatever demos it can, and calls it done. No searching, no comparing, no iterating. Think of it as "compile once and ship." It's fast, cheap, and often good enough for a strong baseline.

**BootstrapFewShotWithRandomSearch** does the same bootstrap process multiple times (say 10 times), producing 10 different demo sets with different random seeds. It then evaluates all 10 on a held-out validation set and picks the best one. You're essentially running a small tournament — more API calls, but a meaningfully better chance of landing on a demo configuration that generalizes well.

**MIPROv2** is the full optimizer. It not only searches over different demo sets (using BootstrapFewShot internally as a building block), but also proposes and tests different instruction phrasings for each predictor using a Bayesian search strategy (Optuna TPE). It can tune both *what examples to show* and *how to phrase the task instructions*. It costs the most but can yield the largest improvements, especially on complex programs.

The key insight: BootstrapFewShot is not competing with MIPROv2 — it's literally used *inside* MIPROv2. Understanding BootstrapFewShot deeply means you understand the foundation that all DSPy optimization builds on.

---

| Feature | BootstrapFewShot | BootstrapFewShotWithRandomSearch | MIPROv2 |
|--------|-----------------|--------------------------------|---------|
| Demo generation | YES | YES (multiple runs) | YES |
| Instruction optimization | NO | NO | YES |
| Search strategy | single pass | random N programs | Bayesian TPE |
| Eval budget | low | medium (N x trainset) | configurable |
| Best for | quick baseline / distillation | better demos on small budget | full optimization |
| Validation set selection | NO | YES (picks best of N) | YES (Optuna objective) |

---

## B-12 — Common Pitfalls

### 🗣️ The Story

Most bugs with BootstrapFewShot fall into predictable patterns. Here are the ones that trip people up most often, with the intuition for why they happen.

**Zero bootstrapped demos** is the most alarming-looking failure — the progress bar runs, nothing passes, and you end up with only raw demos. Nine times out of ten this is a metric bug: either the metric is too strict (checking for exact string equality when the model produces slightly different formatting), or there's an exception happening silently inside the teacher. The fastest debug: temporarily set `metric=None` and check that the teacher can run at all.

**Inflated prompts** happen when you set `max_bootstrapped_demos` too high. Each augmented demo for a ChainOfThought predictor can be 5–10 lines of text. With 10 augmented demos plus 10 raw demos, you're burning 100+ tokens per predictor just on examples. For a 3-predictor chain, that's 300+ overhead tokens before the actual query. This adds cost and — counterintuitively — can hurt accuracy by crowding out the actual question.

**Accidentally compiling an already-compiled student** triggers an assertion error. DSPy enforces this because re-compiling a compiled student would stack demos on top of demos, producing nonsensical prompts. Always pass `MyProgram()` (a fresh instance), never a previously-compiled program object.

**Missing `.with_inputs()`** is silent and devastating. Without it, `example.inputs()` returns an empty dict, the teacher receives no input fields, and whatever it produces is random noise. Every training example in DSPy must have `.with_inputs()` called with the names of the input fields.

---

### Pitfall 1: Bootstrap produces 0 demos
```
Bootstrapped 0 full traces after 30 examples for up to 1 rounds...
```
Cause: metric never returns True, or teacher throws exceptions.
Fix: add print to metric, try metric=None temporarily, or lower metric_threshold.

### Pitfall 2: max_bootstrapped_demos too high -> inflated prompts
10 augmented CoT demos = context overflow and high inference cost.
Fix: start at 3-4.

### Pitfall 3: max_labeled_demos=0 removes raw demos
sample_size = max(0, 0 - len(augmented)) = 0. Only augmented demos survive. Check this is intentional.

### Pitfall 4: Trainset too small
Need 4 bootstraps but only 5 examples -> may get only 2 augmented. Use 3-5x trainset size relative to max_bootstrapped_demos.

### Pitfall 5: Compiling an already-compiled student
```python
bs.compile(compiled, trainset=train)
# AssertionError: Student must be uncompiled.
```
Always pass a fresh instance: `bs.compile(MyProgram(), trainset=train)`.

### Pitfall 6: Forgetting .with_inputs()
```python
# WRONG
dspy.Example(question="...", answer="...")

# CORRECT
dspy.Example(question="...", answer="...").with_inputs("question")
```
Without .with_inputs(), example.inputs() returns {} and the teacher receives nothing.

---

## B-13 — Saving and Loading a Compiled Program

### 🗣️ The Story

Compilation is expensive — it involves real API calls to the LLM for every training example that gets traced. You don't want to redo that work every time you restart your script or deploy to a new environment.

DSPy handles this simply: `compiled.save("path.json")` serializes the entire program state, including every predictor's demo list, to a JSON file. `loaded.load("path.json")` restores it. The JSON is human-readable, so you can inspect exactly which demos ended up in each predictor, diff them between runs, or even manually tweak them if you know what you're doing.

A compiled program is effectively a "snapshot" of your optimization result. Treat it like a build artifact — commit it alongside your code, version it, and reload it rather than recompiling from scratch in production.

---

```python
# Save after compiling
compiled.save("compiled_qa.json")

# Load later (no need to re-bootstrap)
loaded = CoTQA()
loaded.load("compiled_qa.json")
# loaded.cot.demos is fully restored
```

The JSON stores each predictor's demos list. Rerunning bootstrap is unnecessary once you have a saved compile.

---

## B-14 — BootstrapFewShot as a Sub-Component of MIPROv2

### 🗣️ The Story

Understanding BootstrapFewShot fully means understanding that it isn't a standalone optimizer competing with the others — it's more like a *primitive* that more sophisticated optimizers are built from.

MIPROv2's job is to search for the best *combination* of instruction phrasing and demo set for each predictor in your program. To generate the demo set candidates that it will search over, it calls BootstrapFewShot repeatedly — say 10 times with different random seeds — producing 10 different candidate demo configurations. Optuna (a Bayesian optimizer) then figures out which demo configuration, paired with which instruction variant, works best on a held-out validation set.

So when you use MIPROv2, BootstrapFewShot is running silently in the background, doing exactly what this document describes, multiple times over. The difference is that MIPROv2 doesn't just accept the first compiled result — it treats each BootstrapFewShot compilation as a candidate to be evaluated and compared.

This means everything you learned in this document about how traces are collected, how metrics gate quality, how teacher/student splitting works — all of it applies inside MIPROv2 as well. You're not learning isolated facts about a single class; you're learning the vocabulary that the entire DSPy optimization framework is built on.

---

```python
# mipro_optimizer_v2.py (simplified)
demo_candidates = create_n_fewshot_demo_sets(
    student=student,
    num_candidate_sets=num_candidate_programs,   # e.g. 10
    trainset=trainset,
    max_bootstrapped_demos=self.max_bootstrapped_demos,
    max_labeled_demos=self.max_labeled_demos,
    ...
)
```

`create_n_fewshot_demo_sets` runs BootstrapFewShot.compile() N times with different random seeds, producing N different demo set candidates. Optuna TPE then searches over:

- Which instruction candidate (from GroundedProposer)
- Which demo set candidate (from N BootstrapFewShot runs)

...is the best combination for each predictor.

BootstrapFewShot is a *building block* used inside both BootstrapFewShotWithRandomSearch and MIPROv2 — not a competing optimizer.

---

## B-15 — Quick Reference Card

### 🗣️ The Story

When you've internalized the full mental model, BootstrapFewShot boils down to a single sentence: *run the teacher, keep the good traces, inject them as demos into the student.* Everything else is parameterization of that core loop. The quick reference below is the cheat sheet you'll want when you're in the middle of a project and need to remember which knob does what.

The demo slot layout diagram at the bottom is worth memorizing — it's the clearest mental picture of what `predictor.demos` actually looks like after compilation and why the two parameters (`max_bootstrapped_demos` and `max_labeled_demos`) interact the way they do.

---

```python
from dspy.teleprompt import BootstrapFewShot

compiled = BootstrapFewShot(
    metric=my_metric,             # required for quality filtering
    metric_threshold=0.5,         # optional float threshold
    teacher_settings={            # optional: stronger LM for teacher
        "lm": dspy.LM("openai/gpt-4o")
    },
    max_bootstrapped_demos=4,     # augmented (traced) demos per predictor
    max_labeled_demos=16,         # total demos per predictor
    max_rounds=1,                 # retries per example
    max_errors=5,                 # exception tolerance
).compile(
    student=MyProgram(),          # always a fresh (uncompiled) instance
    teacher=optional_teacher,     # None -> student.deepcopy() used
    trainset=my_trainset,         # list[dspy.Example] with .with_inputs()
)
```

### Demo slot layout per predictor after compile

```
predictor.demos = [
    augmented_1, ..., augmented_K,    # K <= max_bootstrapped_demos  (traced, have reasoning)
    raw_1,       ..., raw_M           # M = max_labeled_demos - K    (raw labels, no reasoning)
]
```

### File reference

| Symbol | File | Lines |
|--------|------|-------|
| BootstrapFewShot class | dspy/teleprompt/bootstrap.py | L36-L266 |
| _prepare_student_and_teacher | bootstrap.py | L91-L101 |
| _bootstrap (outer loop) | bootstrap.py | L153-L183 |
| _bootstrap_one_example | bootstrap.py | L185-L265 |
| _train (demo assignment) | bootstrap.py | L257-L266 |
| LabeledFewShot (used for teacher init) | dspy/teleprompt/vanilla.py | - |
| create_n_fewshot_demo_sets | dspy/teleprompt/utils.py | - |