# MIPROv2 — The Complete Story: From Zero to Optimized Prompt

> A ground-up, code-anchored walkthrough of every concept, phase, and parameter.
> Written for someone reading the DSPy source for the first time.
> Now with detailed natural-language explanations alongside every concept and code block.

---

## Chapter 0 — The Problem MIPROv2 Solves

### The Story

Picture yourself building a question-answering system. You've wired everything up, the LLM is responding, the code runs — but the answers are mediocre. Sometimes the model ignores your context entirely. Sometimes it's too verbose. Sometimes it hallucinates facts you never provided.

You suspect the problem isn't the model itself — it's *how you're talking to it*. The instruction at the top of the prompt is bland and uninformative. The examples you've given it don't quite capture the style you want. But there are thousands of possible instruction phrasings and millions of combinations of examples. You can't try them all by hand.

This is exactly the problem MIPROv2 was built to solve. Think of it as an automated prompt engineer: it systematically explores the space of possible instructions and example sets for every LLM call in your program, runs them against real data, and finds the combination that actually scores best on your metric. No gut feeling, no guessing — just evidence-based optimization.

The insight at the heart of MIPROv2 is that a prompt has two independently tunable levers. The first is the **instruction** — the sentence at the very top that tells the model its role and how to behave. The second is the **few-shot demonstrations** — example input/output pairs shown just before the real query, which teach the model the exact format and reasoning style you want. MIPROv2 searches over both simultaneously using a principled probabilistic technique called Bayesian optimization.

---

```python
class MyQA(dspy.Module):
    def __init__(self):
        self.generate = dspy.Predict("context, question -> answer")

    def forward(self, question, context):
        return self.generate(context=context, question=question)
```

When this runs, the prompt sent to the LLM looks like this:

```
Given the fields `context`, `question`, produce the fields `answer`.

---

Follow the following format.

Context: ${context}
Question: ${question}
Answer: ${answer}

---

Context: The Eiffel Tower is in Paris, France.
Question: Where is the Eiffel Tower?
Answer:
```

Two things shape how well the LLM answers:

1. **The instruction** — that first line: *"Given the fields context, question, produce the fields answer."*  
   A better instruction might be: *"You are a careful reader. Use only information from the context to answer concisely."*

2. **The few-shot demos** — example context/question/answer pairs shown before the real question.  
   Good examples teach the LLM the exact format and reasoning style you want.

**MIPROv2's job:** automatically find the best instruction text and the best set of few-shot
examples for every LLM call in your program, using Bayesian optimization.

---

## Chapter 1 — Key Vocabulary

### The Story

Before diving into the algorithm, it helps to understand the three building blocks MIPROv2 thinks in terms of. Once these click, every subsequent phase will feel intuitive.

**Predictors** are the individual LLM calls inside your program. If your RAG system does "retrieve → generate → judge", then `generate` and `judge` are two separate predictors — each with its own prompt, its own instruction, and its own set of demonstrations. MIPROv2 optimizes all of them independently but simultaneously, because the best instruction for your generator might depend on what kind of examples you're showing your judge.

**Signatures** are the structured contracts attached to each predictor. They declare what inputs the predictor expects, what outputs it should produce, and what instruction text should appear at the top of its prompt. When MIPROv2 replaces an instruction, it does so by calling `.with_instructions(...)` on the signature — leaving everything else (field names, format rules) intact.

**Demos** are the few-shot examples stored inside each predictor. They are `dspy.Example` objects — essentially dictionaries with field names and values. When a predictor renders its prompt, it writes out all its demos first, then your actual input. The model sees the examples as a pattern to follow, not just context. Getting the right examples into the right predictor is half the battle.

---

### Predictor

Every `dspy.Predict(...)` or `dspy.ChainOfThought(...)` in your module is a **Predictor** —
one LLM call with its own instruction and its own list of demos.

```python
class MyRAGProgram(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)          # NOT a predictor (no LLM)
        self.generate = dspy.Predict(               # ← Predictor 0, name="generate"
            "context, question -> answer"
        )
        self.judge = dspy.ChainOfThought(           # ← Predictor 1, name="judge"
            "answer -> verdict"
        )

    def forward(self, question):
        context = self.retrieve(question).passages
        answer  = self.generate(context=context, question=question).answer
        verdict = self.judge(answer=answer).verdict
        return dspy.Prediction(answer=answer, verdict=verdict)
```

**How DSPy discovers predictors** — `module.py L87-L93`:
```python
def named_predictors(self):
    from dspy.predict.predict import Predict
    # Walk all attributes; keep only Predict instances
    return [(name, param) for name, param in self.named_parameters()
            if isinstance(param, Predict)]

def predictors(self):
    return [param for _, param in self.named_predictors()]
```

```python
prog = MyRAGProgram()
prog.predictors()        # → [<Predict: generate>, <Predict: judge>]
prog.named_predictors()  # → [("generate", <Predict>), ("judge", <Predict>)]
```

### Signature
Each predictor has a **Signature** — structured definition of inputs, outputs, and the instruction:

```python
# The instruction is the text at the top of the prompt
predictor.signature.instructions
# e.g. "Given the fields context, question, produce the fields answer."

# Change the instruction:
predictor.signature = predictor.signature.with_instructions("You are a careful reader...")
```

### Demos
Each predictor also has `predictor.demos` — a `list[dspy.Example]` of few-shot examples
shown inside the prompt before the actual input:

```python
predictor.demos = [
    dspy.Example(
        context="Water freezes at 0°C.",
        question="At what temperature does water freeze?",
        answer="0°C"
    ),
    dspy.Example(
        context="The sun is a star at the center of our solar system.",
        question="What is the sun?",
        answer="A star at the center of our solar system."
    ),
]
```

---

## Chapter 2 — The Three-Phase Pipeline

### The Story

MIPROv2's work happens in three distinct phases, each building on the last. Think of it as three separate jobs that must be done in order before you can answer the question: "which prompt is best?"

The **first job** is to gather raw material — specifically, candidate sets of few-shot examples. You can't just pick examples randomly from your dataset and hope for the best. MIPROv2 actually *runs your program* on training data and collects the traces of what happened: what the model was given, what it produced, and whether the result was any good. Good traces become high-quality demo candidates. This is Phase 1: Bootstrap.

The **second job** is to generate candidate instructions. Rather than relying on you to brainstorm twenty phrasings of "answer questions from context", MIPROv2 prompts a powerful language model to write instructions for you — armed with information about your dataset, your program structure, and example inputs/outputs. This is Phase 2: Propose.

The **third job** is the actual search: taking all the candidate instructions and candidate demo sets from phases 1 and 2, and finding the combination that performs best on your validation data. There are potentially thousands of combinations, so MIPROv2 uses Bayesian optimization — a smart algorithm that learns from each trial and focuses its budget on promising areas of the search space. This is Phase 3: Search.

---

```
MIPROv2.compile(student_program, trainset, valset)
         │
         ├─ Phase 1: BOOTSTRAP
         │   Produce N candidate demo-sets per predictor
         │   (N=6 for auto="light")
         │
         ├─ Phase 2: PROPOSE
         │   Produce N candidate instruction strings per predictor
         │
         └─ Phase 3: SEARCH (Optuna Bayesian Optimization)
             Try combinations of (instruction, demos) per predictor
             Pick the combo with the highest validation score
```

---

## Chapter 3 — Phase 1: Bootstrap Few-Shot Demos

### The Story

Imagine you're trying to teach a student by showing them worked examples. The quality of those worked examples matters enormously. If your examples are sloppy, incomplete, or wrong, the student will learn the wrong patterns. So where do the examples for your LLM come from?

MIPROv2 takes a clever approach: it uses a second, better-prepared copy of your program — called the **teacher** — to *generate* the examples. The teacher runs through your training data, solves each problem, and its solutions (along with the inputs) become the demonstration examples for your actual program. The key insight is that if the teacher produces a good answer (as judged by your metric), then that teacher's solution is worth showing to the student as an example of how to behave.

This is called "bootstrapping" because you're pulling yourself up by your bootstraps — using your own program (or a better version of it) to create the training signal that will make your program better.

#### Why Have a Separate Teacher at All?

You might ask: why not just use the student program to generate its own demos? The answer is that the student starts completely blank — no instructions, no examples. Ask a blank program to answer questions and it will produce garbage, which you definitely don't want as your demonstration examples. The teacher, by contrast, is pre-seeded with some raw labeled examples from your dataset, so it gives decent answers and produces useful traces.

#### What Exactly Gets Recorded?

When the teacher runs on a training example, DSPy records a **trace** — a complete log of every predictor call that happened, including what was passed in and what came out. If the teacher's final answer scores above your `metric_threshold`, every step of that trace gets saved as a demo. This matters because in a multi-step pipeline (retrieve → generate → judge), you get demos for *all* the intermediate steps, not just the final output.

#### From Traces to Demo Sets — Diversity by Design

Rather than running bootstrap once and using those demos forever, MIPROv2 runs the bootstrap process **N times** with different random seeds and configurations. This produces N distinct candidate demo sets per predictor. Some sets will have purely raw labeled examples. Some will have richly traced LLM-generated examples. Some will have nothing at all (zero-shot). This diversity is intentional: it ensures that the subsequent search phase has real choices to evaluate, not just N nearly-identical demo lists.

---

### The Teacher–Student Distinction

Before collecting demos, Bootstrap creates two copies of your program:

| Copy | Role | Starts with |
|------|------|-------------|
| **Teacher** | Runs on training examples; its outputs are recorded as demos | `max_labeled_demos` raw gold examples (so it gives decent answers) |
| **Student** | Gets optimized and returned to you | Empty — no demos, no special instructions |

**`bootstrap.py L91-L101`:**
```python
def _prepare_student_and_teacher(self, student, teacher):
    self.student = student.reset_copy()   # blank slate

    # If no teacher passed, the student itself (deep copy) becomes the teacher
    self.teacher = teacher.deepcopy() if teacher is not None else student.deepcopy()

    # Give the teacher some labeled examples first so it answers well
    if self.max_labeled_demos and not self.teacher._compiled:
        teleprompter = LabeledFewShot(k=self.max_labeled_demos)
        self.teacher = teleprompter.compile(
            self.teacher.reset_copy(), trainset=self.trainset
        )
```

Why not just use the student? Because it starts blank — it would hallucinate,
producing useless traces. The teacher, pre-seeded with some examples,
gives better outputs → better traces → better demos for the student.

---

### Running the Teacher and Capturing Traces

**`bootstrap.py L173-L237`:**
```python
def _bootstrap_one_example(self, example, round_idx=0):
    with dspy.settings.context(trace=[], **self.teacher_settings):
        # ↑ "trace=[]" tells DSPy: record every LLM call into this list

        # Safety: remove current example from teacher's own demos (prevent leakage)
        for name, predictor in teacher.named_predictors():
            predictor.demos = [x for x in predictor.demos if x != example]

        prediction = teacher(**example.inputs())   # run the teacher
        trace = dspy.settings.trace
        # trace = [
        #   (predictor_generate,
        #    {"context": "Water freezes at 0°C.", "question": "At what T?"},
        #    Prediction(answer="0°C")),
        #
        #   (predictor_judge,
        #    {"answer": "0°C"},
        #    Prediction(verdict="Correct")),
        # ]

    metric_val = self.metric(example, prediction, trace)
    # e.g. 0.9  (how good was the teacher's answer?)

    success = metric_val >= self.metric_threshold   # e.g. 0.9 >= 0.6 → True
```

If `success = True`, all trace steps become demos:
```python
for step in trace:
    predictor, inputs, outputs = step
    demo = dspy.Example(augmented=True, **inputs, **outputs)
    # dspy.Example(
    #   augmented=True,         ← marks this as an LLM-traced example
    #   context="Water...",     ← input (from teacher's run)
    #   question="At what T?",  ← input
    #   answer="0°C",           ← LLM-generated output (teacher's output, NOT gold label)
    # )
    name2traces["generate"].append(demo)
```

> **Important:** `augmented=True` demos contain the **teacher's LLM-generated output**,
> not necessarily the gold label from your dataset. They capture the reasoning *style*
> and *format* that produced a high-scoring answer.

---

### How Augmented and Raw Demos Are Mixed

#### The Story

Not every training example will produce a usable trace. The teacher might score below your threshold, might fail entirely, or you might simply not have enough training data to fill your demo budget. MIPROv2 handles this gracefully with a simple priority rule: **augmented demos come first, raw labeled examples fill the gaps.**

Think of it like seating at a dinner party. You have a budget of four seats per predictor. You seat your VIP guests (augmented demos — the ones where the teacher actually demonstrated good reasoning) first. Whatever seats remain go to general guests (raw labeled examples — just dataset rows with no special properties). If no VIPs show up, regular guests fill all the seats. This means you always have *something* to show the model, even when bootstrapping fails.

**`bootstrap.py L239-L251`:**
```python
def _train(self):
    rng = random.Random(0)
    # self.validation = examples where the teacher FAILED (no trace was saved)
    raw_demos = self.validation

    for name, predictor in self.student.named_predictors():

        # FIRST: take up to max_bootstrapped_demos augmented (traced) examples
        augmented_demos = self.name2traces[name][:self.max_bootstrapped_demos]
        # e.g. max_bootstrapped_demos=3, got 5 traces → take first 3

        # SECOND: fill remaining budget with raw labeled examples
        # budget = max_labeled_demos - how many augmented we already have
        sample_size = min(
            self.max_labeled_demos - len(augmented_demos),  # remaining slots
            len(raw_demos)
        )
        sample_size = max(0, sample_size)
        raw_demos_sampled = rng.sample(raw_demos, sample_size)

        # RESULT: augmented first, then raw
        predictor.demos = augmented_demos + raw_demos_sampled
```

**Concrete example** — `max_bootstrapped_demos=3`, `max_labeled_demos=4`, 2 traces found:

```
augmented_demos    = [aug_ex1, aug_ex2]       # 2 augmented traces (metric passed)
remaining budget   = 4 - 2 = 2
raw_demos_sampled  = [raw_ex7, raw_ex12]      # 2 raw fallback examples

predictor.demos = [aug_ex1, aug_ex2, raw_ex7, raw_ex12]
                   ←  augmented  →  ←  raw labeled  →
```

**What if no traces are found?** `augmented_demos=[]`, budget=4 raw examples:
```
predictor.demos = [raw_ex3, raw_ex9, raw_ex1, raw_ex7]
                   ←      all raw labeled      →
```

---

### N Candidate Demo Sets — `create_n_fewshot_demo_sets()` (utils.py L328)

#### The Story

Here's where MIPROv2 gets creative. Instead of running bootstrap once and calling it done, it runs the whole process N times, each time with a slightly different configuration. The result is N different candidate demo sets per predictor — a menu of options that the later search phase can choose from.

Three of the N sets are always "special baseline" sets added without any bootstrapping:
- One set is **completely empty** (zero-shot). This tests whether your predictor is actually *hurt* by examples rather than helped — sometimes simpler really is better.
- One set uses **only raw gold labels** from your dataset, with no LLM tracing at all. This is a clean baseline for comparing traced vs. non-traced examples.
- One set uses the **unshuffled bootstrap** — the same training data in its original order, with the standard bootstrapping process.

The remaining sets are produced by **shuffling** the training data and using a **random number of bootstrapped demos** (anywhere from 1 to `max_bootstrapped_demos`). This randomness is intentional: different shuffles expose the bootstrap to different training examples first, and different demo counts produce sets with different amounts of LLM-traced vs. raw content. The search phase will figure out which mix actually matters for your task.

```python
# utils.py L354-L415 (simplified)

num_candidate_sets -= 3   # 3 special sets always added, so total stays = N

for seed in range(-3, num_candidate_sets):

    if seed == -3:
        # ZERO-SHOT: no demos at all
        program2 = student.reset_copy()

    elif seed == -2:
        # LABELS ONLY: raw gold examples, no LLM tracing
        program2 = LabeledFewShot(k=max_labeled_demos).compile(
            student, trainset=trainset_copy, sample=True
        )

    elif seed == -1:
        # UNSHUFFLED BOOTSTRAP: trainset in original order
        program2 = BootstrapFewShot(
            max_bootstrapped_demos=max_bootstrapped_demos,  # fixed value e.g. 3
            max_labeled_demos=max_labeled_demos,
        ).compile(student, teacher=teacher, trainset=trainset_copy)

    else:  # seed = 0, 1, 2, ...
        # SHUFFLED BOOTSTRAP: randomize order and random demo count
        rng.shuffle(trainset_copy)                          # different order each time
        size = rng.randint(1, max_bootstrapped_demos)       # random: e.g. 1, 3, 2

        program2 = BootstrapFewShot(
            max_bootstrapped_demos=size,    # RANDOM count!
            max_labeled_demos=max_labeled_demos,
        ).compile(student, teacher=teacher, trainset=trainset_copy)

    # Collect what each predictor ended up with
    for i, _ in enumerate(student.predictors()):
        demo_candidates[i].append(program2.predictors()[i].demos)
```

**`LabeledFewShot` (seed==-2) — `vanilla.py L6-L25`:**
```python
class LabeledFewShot(Teleprompter):
    def compile(self, student, *, trainset, sample=True):
        self.student = student.reset_copy()
        for predictor in self.student.predictors():
            if sample:
                predictor.demos = rng.sample(trainset, min(self.k, len(trainset)))
            else:
                predictor.demos = trainset[:min(self.k, len(trainset))]
        return self.student
```

Raw gold examples look like:
```python
dspy.Example(
    context="The Eiffel Tower is in Paris.",
    question="Where is the Eiffel Tower?",
    answer="Paris, France"
    # NO augmented=True flag
    # This is just a dataset row, not an LLM trace
)
```

---

### The Full Picture After Phase 1

With `auto="light"` (N=6), 2 predictors, `max_bootstrapped_demos=3`, `max_labeled_demos=4`:

```
demo_candidates = {
    0: [                                              # for predictor "generate"
        [],                                           # set 0: seed=-3, ZERO-SHOT
        [raw_ex3, raw_ex9, raw_ex1, raw_ex7],         # set 1: seed=-2, LABELS ONLY (4 raw)
        [aug_ex5, aug_ex11, raw_ex0, raw_ex3],        # set 2: seed=-1, UNSHUFFLED (2 aug + 2 raw)
        [aug_ex9, aug_ex3, raw_ex7, raw_ex2],         # set 3: seed=0,  size=2 aug + 2 raw
        [aug_ex7, aug_ex1, aug_ex4, raw_ex12],        # set 4: seed=1,  size=3 aug + 1 raw
        [aug_ex2, raw_ex6, raw_ex9, raw_ex1],         # set 5: seed=2,  size=1 aug + 3 raw
    ],
    1: [                                              # for predictor "judge" — same runs, judge's trace
        [],
        [raw_ex3, raw_ex9, raw_ex1, raw_ex7],
        [aug_ex5, aug_ex11, raw_ex0, raw_ex3],
        ...
    ]
}
```

Each set has a different number and type of demos — this diversity is intentional.
Optuna will figure out which set actually helps your metric the most.

---

## Chapter 4 — Phase 2: Propose Instruction Candidates

### The Story

Writing a good instruction for an LLM is harder than it looks. You might try "Answer the question using the context" and get mediocre results. Someone else might write "You are a meticulous research assistant. Read the context carefully and answer only based on what is explicitly stated, without adding your own knowledge." That second instruction can produce dramatically better results — but arriving at it through intuition and trial-and-error takes significant effort.

Phase 2 automates this process. Rather than asking you to brainstorm instructions, MIPROv2 prompts a language model to write instructions *for* you. And critically, it doesn't just ask the proposer to write an instruction blindly — it gives the proposer rich context to work with. The proposer gets to see what your dataset looks like, what your overall program does, what the specific predictor is responsible for, examples of real inputs and outputs, and even a creative nudge in the form of a random "tip". The more context the proposer has, the more targeted and useful the instructions it writes tend to be.

One important design choice: **the original default instruction is always kept as index 0**. This guarantees that the baseline is always part of the search space. If none of the LLM-proposed instructions turn out to be better, the optimizer can always fall back to the original. You never inadvertently throw away a working baseline.

---

### Goal
For each predictor, produce N candidate instruction strings.

### Output
```python
instruction_candidates = {
    0: [   # for predictor "generate"
        "Given the fields context, question, produce the fields answer.",   # index 0 = ALWAYS original
        "You are a careful reader. Use ONLY information from the context.",  # LLM-proposed
        "Extract the most relevant fact and answer in one sentence.",
        # ... up to N
    ],
    1: [   # for predictor "judge"
        "Given the fields answer, produce the fields verdict.",  # index 0 = ALWAYS original
        "Critically evaluate whether the answer is factually grounded.",
        # ...
    ],
}
```

**Index 0 is always the original** — guaranteed by `mipro_optimizer_v2.py L408`:
```python
instruction_candidates[i][0] = get_signature(pred).instructions
```

This ensures the baseline is always included in the search space.

---

### The Four Pieces of Context Given to the Proposer

#### The Story

Each of the four context pieces addresses a different blind spot the proposer would otherwise have. Without a dataset summary, the proposer doesn't know whether it's writing for a coding task or a medical Q&A task. Without program and module descriptions, it doesn't know what role this particular predictor plays in the larger pipeline. Without bootstrapped examples, it can't see what real inputs and outputs look like. And without a creativity tip, all its proposals might converge on the same generic phrasing. Together, these four inputs push the proposer toward instructions that are specific, contextual, and diverse.

#### 1. Dataset Summary (`data_aware_proposer=True`)

The proposer scans your training data in batches and summarizes it. This gives the proposer a sense of the domain, difficulty, and nature of the task before it tries to write any instructions.

- Calls `dspy.Predict(DatasetDescriptor)` on the first batch → initial observations
- Calls `dspy.Predict(DatasetDescriptorWithPriorObservations)` on subsequent batches → refines
- Calls `dspy.Predict(ObservationSummarizer)` at the end → 2-3 sentence summary

Result:
```
"The dataset contains factual question-answer pairs about science and geography.
 Questions are direct and answers are typically one to two sentences.
 Some examples require multi-step reasoning from context."
```

#### 2. Program + Module Description (`program_aware_proposer=True`)

The proposer reads your program's source code and generates descriptions. This is important in multi-predictor pipelines: the proposer needs to know that the "generate" step is different from the "judge" step, and should write instructions appropriate to each role.

```
program_description:
  "This RAG program retrieves relevant passages and uses them to generate
   grounded answers, then a judge verifies factual accuracy."

module_description (for predictor "generate"):
  "This module takes retrieved context and a user question and generates
   a precise, context-grounded answer in one to two sentences."
```

#### 3. Bootstrapped Examples (`fewshot_aware_proposer=True`)

Up to `BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT=3` examples from `demo_candidates[i][j]`
are shown to the proposer so it can see real inputs/outputs. This is crucial — an abstract description of your task is far less useful to the proposer than seeing actual concrete examples of what the predictor receives and what it should produce.

```
Example 1:
  Context: "Water boils at 100°C at sea level."
  Question: "At what temperature does water boil?"
  Answer: "100°C"

Example 2:
  Context: "Newton's first law states objects at rest stay at rest..."
  Question: "What does Newton's first law describe?"
  Answer: "The tendency of objects to resist changes in motion."
```

#### 4. Random Creativity Tip (`tip_aware_proposer=True`)

A random tip is selected from the `TIPS` dict. These tips serve as a gentle creative nudge — they encourage the proposer to approach the instruction from different angles across multiple calls, rather than always producing similar phrasings. Since MIPROv2 calls the proposer N times per predictor, having different tips for different calls helps produce genuinely diverse instruction candidates.

- `"Think step by step."`
- `"Be creative and use novel reasoning."`
- `"Keep the instruction simple and direct."`
- `"Write as if for high-stakes decision making."`
- `"Describe what you want in the style of a specific expert persona."`

---

### The Full Proposer Prompt (what the LLM actually sees)

```
SYSTEM: You are an expert at writing instructions for LLM modules.

Dataset summary:
  "The dataset contains factual question-answer pairs about science..."

Program description:
  "This RAG program retrieves relevant passages..."

Module description (for the module you are writing an instruction for):
  "This module takes context and a question and generates a precise answer..."

Instruction history (previous proposals and their scores):
  1. "Given the fields context, question, produce answer."  → score: 0.62

Demonstrations of the module's task:
  Context: "Water boils at 100°C."
  Question: "At what temperature does water boil?"
  Answer: "100°C"
  ---
  [2 more examples]

Tip: Think step by step.

---
Propose a new instruction for this module that improves upon the history above:
```

→ LLM output: `"You are a precise scientific assistant. Read the provided context carefully and answer the question using only the information given, in one clear sentence."`

This is called N times per predictor, giving N candidate instructions.

---

## Chapter 5 — Phase 3: Bayesian Search with Optuna

### The Story

Now comes the part where everything comes together. You have N candidate instruction strings per predictor and N candidate demo sets per predictor. The question is: which combination actually works best for your task?

The naïve approach would be to try every possible combination. But if you have 2 predictors and 6 candidates each, that's already 6⁴ = 1,296 combinations (since each predictor has both an instruction choice and a demo-set choice). With 4 predictors it becomes 6⁸ = over 1.6 million. Evaluating each combination on your validation set is expensive — every trial costs real LLM calls and real time. You need a smarter approach.

MIPROv2 uses **Bayesian optimization via Optuna** — specifically, the Tree-structured Parzen Estimator (TPE) algorithm. The idea is elegant: instead of trying combinations randomly or exhaustively, you build a probabilistic model of which parameter choices tend to lead to high scores, and use that model to guide your next trial. Each trial makes the model more accurate, which makes the next trial's suggestion more informed. Over time, the optimizer zeroes in on promising regions of the search space rather than wasting trials on combinations it's already learned are bad.

The analogy is something like wine tasting. After your first few sips you start to notice patterns — "Bordeaux varieties from warm years score higher." You don't need to taste every bottle; your developing model of the space helps you pick your next bottle more intelligently. Bayesian optimization does the same thing with parameter combinations.

There's an additional subtlety: MIPROv2 uses `multivariate=True` in Optuna, which means the optimizer models **interactions between parameters**. It can learn "instruction 3 for predictor 0 works especially well when paired with demo set 4 for predictor 1" — a correlation that a simpler optimizer treating each parameter independently would completely miss.

#### The Minibatch Trick — Speed Without Sacrificing Accuracy

There's a practical tension in the search phase: you want to evaluate many trials (for better optimization), but each trial is expensive (because it runs your program on validation examples). MIPROv2's solution is a two-speed evaluation strategy.

For most trials, it evaluates only on a small **minibatch** of 35 validation examples — fast and cheap, though noisy. Every sixth trial or so, it identifies the configuration with the best average minibatch score and runs it on the **full validation set** for an accurate, definitive score. This full-eval result is injected back into Optuna as a training signal, so the optimizer learns from high-quality data even though most of its trials are approximate. The net effect is that you get the exploration benefits of many cheap trials *and* the accuracy of full evaluations for the promising candidates.

---

### The Search Space

For a program with 2 predictors and N=6 candidates,
Optuna has 4 categorical parameters to search:

```
"0_predictor_instruction"  → choose from [0, 1, 2, 3, 4, 5]   (which instruction for pred 0)
"0_predictor_demos"        → choose from [0, 1, 2, 3, 4, 5]   (which demo set for pred 0)
"1_predictor_instruction"  → choose from [0, 1, 2, 3, 4, 5]
"1_predictor_demos"        → choose from [0, 1, 2, 3, 4, 5]

Total combinations: 6^4 = 1296
```

### Why Number of Predictors Matters for Trial Count

The more predictors your program has, the larger the search space. MIPROv2 automatically scales the number of Optuna trials to account for this. The formula ensures that programs with many predictors get enough trials to meaningfully explore their larger search space, while simple single-predictor programs don't waste budget on trials that yield diminishing returns.

**`mipro_optimizer_v2.py L218-L225`:**
```python
def _set_num_trials_from_num_candidates(self, program, zeroshot_opt, num_candidates):
    num_vars = len(program.predictors())   # M = number of predictors
    if not zeroshot_opt:
        num_vars *= 2   # ×2 because each predictor has 2 variables (instruction + demos)

    # N = num_candidates
    num_trials = int(max(
        2 * num_vars * np.log2(num_candidates),   # scale with M and log(N)
        1.5 * num_candidates                       # minimum floor
    ))
    return num_trials
```

$$\text{num\_trials} = \max\!\Bigl(2 \times M \times \log_2{N},\;\; 1.5 \times N\Bigr)$$

**Worked examples:**

| Program | Predictors (P) | M = P×2 | N (`auto="light"`) | num_trials |
|---------|---------------|---------|-------------------|------------|
| Simple QA | 1 | 2 | 6 | **10** |
| RAG (gen+judge) | 2 | 4 | 6 | **21** |
| Multi-hop chain | 4 | 8 | 6 | **42** |
| Simple QA zero-shot | 1 | 1 | 6 | **9** |

More predictors → larger search space → more Optuna trials needed to converge.

---

### The Objective Function — What Happens Each Trial

Every time Optuna runs a trial, it calls this objective function. The function takes the parameter choices Optuna suggests, assembles a candidate program with those exact instructions and demos, runs it against a minibatch of validation data, and returns the score. That score is what Optuna uses to update its model of the search space. Notice that the base program is always **deep-copied** before modifications — the original program is never touched, which means each trial is fully independent.

**`mipro_optimizer_v2.py L465-L564`:**

```python
def objective(trial):
    # Step 1: Optuna (TPE) suggests which index to try for each parameter
    #         It uses ALL past trial results to make smart suggestions
    inst_idx_0  = trial.suggest_categorical("0_predictor_instruction", range(N))  # e.g. → 3
    demos_idx_0 = trial.suggest_categorical("0_predictor_demos",        range(N))  # e.g. → 1
    inst_idx_1  = trial.suggest_categorical("1_predictor_instruction", range(N))  # e.g. → 0
    demos_idx_1 = trial.suggest_categorical("1_predictor_demos",        range(N))  # e.g. → 4

    # Step 2: Deep copy the base program (never modify the original)
    candidate_program = program.deepcopy()

    # Step 3: Install the chosen instruction + demos into the copy
    pred0 = candidate_program.predictors()[0]
    pred0.signature = pred0.signature.with_instructions(
        instruction_candidates[0][3]
        # = "You are a precise scientific assistant. Read the context carefully..."
    )
    pred0.demos = demo_candidates[0][1]
    # = [raw_ex3, raw_ex9, raw_ex1, raw_ex7]   (labels-only set)

    pred1 = candidate_program.predictors()[1]
    pred1.signature = pred1.signature.with_instructions(
        instruction_candidates[1][0]
        # = "Given the fields answer, produce the fields verdict."  (original baseline)
    )
    pred1.demos = demo_candidates[1][4]
    # = [aug_ex7, aug_ex1, aug_ex4, raw_ex12]  (shuffled bootstrap set)

    # Step 4: Evaluate on a minibatch of the validation set
    score = eval_candidate_program(35, valset, candidate_program, evaluate, rng).score
    # e.g. → 0.74

    # Optuna records this score and updates its internal TPE model
    return score
```

---

### Minibatch Strategy — Speed vs Accuracy

Evaluating the full valset every trial is expensive.
MIPROv2 uses a two-speed approach:

- **Most trials:** evaluate on `minibatch_size=35` random valset examples → fast, approximate
- **Every `minibatch_full_eval_steps+1 = 6`th trial:** find the combo with the
  best **average** minibatch score and run it on the **full** valset

**`mipro_optimizer_v2.py L543-L562`:**
```python
if minibatch and (trial_num % (minibatch_full_eval_steps + 1) == 0):
    best_score, best_program, total_eval_calls = self._perform_full_evaluation(
        ...param_score_dict,         # contains all (score, program) per combo
        fully_evaled_param_combos,   # tracks which combos already got a full eval
        ...
    )
    # Inject full eval result BACK into Optuna as a new trial
    # So TPE learns from accurate full-eval data, not just noisy minibatch data
```

**Visual timeline:**
```
Trial 1:   FULL EVAL (baseline, all params=0)         → 0.62
Trial 2:   minibatch 35   inst=[1,0] demos=[4,3]      → 0.71  (fast estimate)
Trial 3:   minibatch 35   inst=[2,1] demos=[2,5]      → 0.68
Trial 4:   minibatch 35   inst=[1,0] demos=[4,2]      → 0.75
Trial 5:   minibatch 35   inst=[0,2] demos=[1,4]      → 0.65
Trial 6:   minibatch 35   inst=[1,1] demos=[4,3]      → 0.73
Trial 7:   FULL EVAL (best avg was combo [1,0,4,2])   → 0.79  ← NEW BEST
Trial 8:   minibatch 35   ...
Trial 13:  FULL EVAL ...
...
Final:     return program with inst=[1,0], demos=[4,2] → score 0.79
```

---

### Optuna TPE with `multivariate=True`

```python
sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
study = optuna.create_study(direction="maximize", sampler=sampler)
```

`multivariate=True` means Optuna models **correlations between parameters**. For example, it
can learn:

> *"Instruction 3 for predictor 0 works best specifically when demo set 4 is used for predictor 1."*

A non-multivariate TPE treats each parameter independently and would miss this.

---

## Chapter 6 — What the Final Program Looks Like

### The Story

After all the trials conclude, MIPROv2 takes the winning combination — the specific instruction index and demo-set index for each predictor — and permanently installs them into a final copy of your student program. This is the program you get back from `.compile()`. It carries all the learned improvements baked in: every predictor now has a carefully crafted instruction and a curated set of demonstrations that were empirically validated to produce the best scores on your metric.

One interesting thing to observe is that the "winner" isn't always one of the AI-generated instruction candidates. Sometimes the original default instruction turns out to score best — and that's fine. The whole point is to find the empirically best option, not to assume that AI-generated is always better. Similarly, the zero-shot demo set (no examples at all) might win for some predictors if the model is capable enough that few-shot examples actually hurt performance by adding noise.

You can inspect what the optimizer found by reading back the instructions and demos from the returned program. You can also look at `best_program.score` to see the validation score, and `best_program.candidate_programs` for a full log of every configuration that was fully evaluated — useful for understanding the optimization landscape.

---

After Phase 3, the winning `(instruction, demos)` combo is installed into a copy of `student`:

```python
# For predictor "generate":
best_program.generate.signature.instructions
# = "You are a precise scientific assistant. Read the provided context carefully..."

best_program.generate.demos
# = [aug_ex7, aug_ex1, aug_ex4, raw_ex12]

# For predictor "judge":
best_program.judge.signature.instructions
# = "Given the fields answer, produce the fields verdict."  (original was best!)

best_program.judge.demos
# = [aug_ex9, aug_ex3, raw_ex7, raw_ex2]
```

When you call `best_program(question="What is DNA?", context="DNA is...")`, the
actual prompt that goes to the LLM now looks like:

```
You are a precise scientific assistant. Read the provided context carefully
and answer the question using only the information given, in one clear sentence.

---

Context: Water boils at 100°C at sea level.
Question: At what temperature does water boil?
Answer: 100°C

Context: Newton's first law states objects at rest stay at rest...
Question: What does Newton's first law describe?
Answer: The tendency of objects to resist changes in motion.

Context: ATP stands for adenosine triphosphate, the energy currency of cells.
Question: What is ATP?
Answer: The primary energy-carrying molecule in cells.

---

Context: DNA is a double helix molecule that carries genetic information...
Question: What is DNA?
Answer:
```

---

## Chapter 7 — Complete Parameter Reference

### The Story

MIPROv2 has a lot of parameters, but they fall into a small number of conceptual categories. Understanding the *why* behind each parameter makes them much easier to remember than reading a flat list.

The **metric** is the north star of the entire process — every bootstrap decision, every eval trial, every full evaluation is judged against this function. Get the metric wrong and you'll optimize toward the wrong thing, so it deserves careful design.

The **budget parameters** (`auto`, `num_candidates`, `num_trials`) control the tradeoff between optimization quality and computation cost. Higher budgets mean more candidate instructions, more diverse demo sets, and more Optuna trials, which generally means better results — but also more LLM calls and more time. The `auto` presets give you sensible defaults without requiring you to think through all the formulas manually.

The **demo parameters** (`max_bootstrapped_demos`, `max_labeled_demos`) control the shape of each individual demo set. They determine the maximum number of LLM-traced examples and the total demo budget. Setting these too high can make prompts very long and expensive; setting them too low gives the model too little to learn from. Four of each is usually a reasonable starting point.

The **proposer flags** (`program_aware_proposer`, `data_aware_proposer`, etc.) control how much context the instruction-writing LLM gets. They're all on by default because more context generally produces better instructions — but you can turn them off if you want faster proposal generation or if you have reason to believe the added context is misleading the proposer.

The **minibatch parameters** balance trial speed against eval accuracy. Smaller minibatch sizes mean faster trials but noisier scores; the `minibatch_full_eval_steps` parameter controls how often you pay the cost of a full validation run to calibrate the optimizer.

---

### `MIPROv2.__init__()` Parameters

| Parameter | Type | Default | What it controls |
|-----------|------|---------|-----------------|
| `metric` | `Callable` | **required** | `(example, prediction, trace) → float\|bool` — your task scorer |
| `prompt_model` | LM | `None` → `dspy.settings.lm` | LM that **writes instructions** (proposal phase) |
| `task_model` | LM | `None` → `dspy.settings.lm` | LM that **runs your program** during eval |
| `teacher_settings` | `dict` | `{}` | Extra `dspy.settings` overrides when running the teacher |
| `max_bootstrapped_demos` | `int` | `4` | Max **augmented** (LLM-traced) demos per predictor per demo set |
| `max_labeled_demos` | `int` | `4` | Total demo budget; raw examples fill slots after augmented |
| `auto` | `str\|None` | `"light"` | Budget preset: `"light"` / `"medium"` / `"heavy"` / `None` |
| `num_candidates` | `int\|None` | `None` | N candidates when `auto=None` |
| `num_threads` | `int\|None` | `None` | Parallelism for `Evaluate` |
| `max_errors` | `int\|None` | `None` | Max tolerated evaluation errors before crash |
| `seed` | `int` | `9` | Controls ALL randomness: shuffling, sampling, instruction generation |
| `init_temperature` | `float` | `1.0` | Temperature for the proposer LM (higher = more diverse proposals) |
| `verbose` | `bool` | `False` | Print the full candidate program before each evaluation |
| `track_stats` | `bool` | `True` | Attach `trial_logs`, `score`, `candidate_programs` to returned program |
| `log_dir` | `str\|None` | `None` | Save JSON of every evaluated program here |
| `metric_threshold` | `float\|None` | `None` | Only keep bootstrap traces where `metric >= threshold` |

### `auto` Budget Presets

> A quick word on picking a preset: start with `"light"` to verify your setup works end-to-end and get a rough sense of what optimized performance looks like. Move to `"medium"` when you want a more thorough search and can afford more compute. Use `"heavy"` for final optimization runs where you're preparing a production-quality system and cost is secondary to quality.

| `auto` | N candidates | valset max | num_trials (2 pred) | num_instruct_candidates |
|--------|-------------|------------|---------------------|------------------------|
| `"light"` | 6 | 100 | ~21 | 3 (= N//2) |
| `"medium"` | 12 | 300 | ~36 | 6 (= N//2) |
| `"heavy"` | 18 | 1000 | ~48 | 9 (= N//2) |

> When few-shot demos are enabled, `num_instruct_candidates = N//2`.
> Reason: demos empirically matter more, so more budget goes to demo diversity.

### `MIPROv2.compile()` Parameters

| Parameter | Type | Default | What it controls |
|-----------|------|---------|-----------------|
| `student` | `dspy.Module` | **required** | Your program to optimize |
| `trainset` | `list[Example]` | **required** | Used for bootstrap + instruction proposal |
| `teacher` | `dspy.Module\|None` | `None` | Pre-compiled program for better traces |
| `valset` | `list[Example]\|None` | `None` | Eval set; if None → 80% of trainset auto-carved |
| `num_trials` | `int\|None` | `None` | Required if `auto=None`; total Optuna trials |
| `max_bootstrapped_demos` | `int\|None` | `None` | Per-run override of `__init__` value |
| `max_labeled_demos` | `int\|None` | `None` | Per-run override of `__init__` value |
| `seed` | `int\|None` | `None` | Per-run seed override |
| `minibatch` | `bool` | `True` | Use minibatch evals; auto-disabled if valset ≤ 50 |
| `minibatch_size` | `int` | `35` | Examples per minibatch trial |
| `minibatch_full_eval_steps` | `int` | `5` | Full eval every N+1 trials |
| `program_aware_proposer` | `bool` | `True` | Show program source code to proposer LM |
| `data_aware_proposer` | `bool` | `True` | Generate and show dataset summary to proposer LM |
| `view_data_batch_size` | `int` | `10` | Trainset examples per batch when building dataset summary |
| `tip_aware_proposer` | `bool` | `True` | Include a random creativity tip in each instruction proposal |
| `fewshot_aware_proposer` | `bool` | `True` | Show bootstrapped demos to proposer LM |
| `provide_traceback` | `bool\|None` | `None` | Forward to `Evaluate` for error reporting |

---

## Chapter 8 — Practical Usage Patterns

### The Story

The patterns below reflect the most common situations you'll encounter. The minimal setup works for most first-time experiments — just give MIPROv2 your metric, your program, and your training data, and let it figure out the rest. 

The **separate prompt and task model** pattern is worth knowing about because it unlocks a powerful cost-optimization strategy: you use a powerful, expensive model (like GPT-4) to write the instructions once during optimization, but a cheap, fast model (like GPT-4o-mini) to actually run your program during all the evaluation trials. Since evaluation is where most of the LLM calls happen, this can cut costs dramatically without sacrificing instruction quality.

The **custom teacher** pattern matters when you already have a partially-optimized program lying around. Rather than bootstrapping from scratch with a blank teacher, you give MIPROv2 your existing compiled program as the teacher. Its already-tuned instructions and demos will produce better-quality traces, which leads to better demo candidates, which leads to a better final result.

The **zero-shot optimization** pattern is useful when your prompts are hitting token limits, or when you've found empirically that few-shot examples hurt rather than help your particular model and task. Setting both demo parameters to zero tells MIPROv2 to only search over instructions, skipping the entire bootstrap phase.

---

### Minimal Setup
```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

def my_metric(example, prediction, trace=None):
    return float(example.answer.lower() in prediction.answer.lower())

optimizer = dspy.MIPROv2(metric=my_metric, auto="light")

best = optimizer.compile(
    student=MyProgram(),
    trainset=train_examples,
)
```

### Separate Prompt and Task Models
```python
optimizer = dspy.MIPROv2(
    metric=my_metric,
    prompt_model=dspy.LM("openai/gpt-4o"),       # powerful model writes instructions
    task_model=dspy.LM("openai/gpt-4o-mini"),    # cheap model runs your program
    auto="medium",
)
```

### Custom Teacher
```python
# If you already have a compiled program, use it as teacher for better traces
existing_compiled = dspy.load("my_baseline_program.json")

best = optimizer.compile(
    student=MyProgram(),
    trainset=train_examples,
    teacher=existing_compiled,   # teacher gives better answers → better traces
)
```

### Zero-Shot Optimization (instructions only, no demos)
```python
optimizer = dspy.MIPROv2(
    metric=my_metric,
    max_bootstrapped_demos=0,
    max_labeled_demos=0,
    auto="light",
)
# demo_candidates is discarded; Optuna only searches over instructions
```

### Manual Budget Control
```python
optimizer = dspy.MIPROv2(
    metric=my_metric,
    auto=None,               # disable auto presets
    num_candidates=10,       # 10 instruction + 10 demo candidates
)
best = optimizer.compile(
    student=MyProgram(),
    trainset=trainset,
    valset=valset,
    num_trials=40,           # must be provided when auto=None
)
```

### Inspecting Results
```python
best = optimizer.compile(student=MyProgram(), trainset=trainset)

# Best score across all full evaluations
print(best.score)

# All programs that were fully evaluated, sorted by score
for entry in best.candidate_programs:
    print(entry["score"], entry["program"])

# What instruction did predictor 0 get?
print(best.predictors()[0].signature.instructions)

# What demos does predictor 0 have?
for demo in best.predictors()[0].demos:
    print(demo)
```

---

## Chapter 9 — Complete Worked Example

### The Story

Let's walk through a complete run with concrete numbers so every piece of the algorithm snaps into place. Imagine you have a RAG program with two predictors (`generate` and `judge`), 100 training examples, and you're using `auto="light"`. MIPROv2 automatically sets aside 80 of those examples as your validation set and uses the remaining 20 for bootstrapping. With `auto="light"`, you get N=6 candidate sets, and the optimizer will run approximately 21 trials.

Watch how the three phases interact: Phase 1 produces 6 distinct demo sets per predictor (including zero-shot, labels-only, and several bootstrap variants with different degrees of augmentation). Phase 2 proposes 3 instruction candidates per predictor (half of N, because demos are empirically more impactful). Phase 3 then runs the Bayesian search — alternating between cheap minibatch evaluations and occasional full-dataset evaluations to zero in on the best configuration.

In the result below, notice that the `judge` predictor ends up with its original instruction — the optimizer discovered that none of the AI-proposed alternatives were actually better. This is a completely normal outcome and illustrates why keeping the baseline as index 0 matters.

---

```python
dspy.configure(lm=dspy.LM("azure/gpt-4.1"))

optimizer = MIPROv2(
    metric=my_metric,
    auto="light",                  # N=6, valset≤100, ~21 trials
    max_bootstrapped_demos=3,
    max_labeled_demos=4,
    metric_threshold=0.6,
)

best_program = optimizer.compile(
    student=MyRAGProgram(),
    trainset=trainset,             # 100 examples → 80 valset, 20 trainset
    program_aware_proposer=True,
    data_aware_proposer=True,
    minibatch_size=25,
)
```

**Phase 1 — Bootstrap (N=6 sets for each of 2 predictors):**
```
Set 0 (seed=-3): pred0.demos=[], pred1.demos=[]
Set 1 (seed=-2): pred0.demos=[raw×4], pred1.demos=[raw×4]
Set 2 (seed=-1): pred0.demos=[aug×2 + raw×2], pred1.demos=[aug×2 + raw×2]
Set 3 (seed=0):  pred0.demos=[aug×2 + raw×2], pred1.demos=[aug×1 + raw×3]  (size=2)
Set 4 (seed=1):  pred0.demos=[aug×3 + raw×1], pred1.demos=[aug×3 + raw×1]  (size=3)
Set 5 (seed=2):  pred0.demos=[aug×1 + raw×3], pred1.demos=[aug×1 + raw×3]  (size=1)
```

**Phase 2 — Propose (num_instruct=3 per predictor):**
```
pred0: [0]="Given fields...", [1]="You are a careful reader...", [2]="Extract key fact..."
pred1: [0]="Given fields...", [1]="Critically evaluate...",      [2]="As expert judge..."
```

**Phase 3 — Optuna search (4 params, ~21 trials):**
```
Trial 1:  FULL EVAL  [0,0,0,0] → 0.62
Trial 2:  minibatch  [1,0,4,3] → 0.71
Trial 3:  minibatch  [2,1,2,5] → 0.68
Trial 4:  minibatch  [1,0,4,2] → 0.75  ← best avg so far
Trial 5:  minibatch  [0,2,1,4] → 0.65
Trial 6:  minibatch  [1,1,4,3] → 0.73
Trial 7:  FULL EVAL  [1,0,4,2] → 0.79  ← NEW BEST
...
```

**Result:**
```python
best_program.generate.signature.instructions
# = "You are a careful reader. Use ONLY information from the context."
best_program.generate.demos
# = [aug_ex7, aug_ex1, aug_ex4, raw_ex12]   (demo set 4)

best_program.judge.signature.instructions
# = "Given the fields answer, produce the fields verdict."  (original was best!)
best_program.judge.demos
# = [aug_ex9, aug_ex3, raw_ex7, raw_ex2]    (demo set 3)

best_program.score  # → 0.79
```

---

## Chapter 10 — File Reference Map

| What you want to find | File | Lines |
|----------------------|------|-------|
| MIPROv2 class, all 3 phases | `dspy/teleprompt/mipro_optimizer_v2.py` | L46–L650 |
| `num_trials` formula | `mipro_optimizer_v2.py` | L218–L225 |
| Bootstrap N sets outer loop | `dspy/teleprompt/utils.py` | L328–L417 |
| `BootstrapFewShot` class | `dspy/teleprompt/bootstrap.py` | L36–L251 |
| Teacher prep | `bootstrap.py` | L91–L101 |
| Trace capture per example | `bootstrap.py` | L173–L237 |
| `_train()` — aug + raw demo mix | `bootstrap.py` | L239–L251 |
| `LabeledFewShot` (seed=-2) | `dspy/teleprompt/vanilla.py` | L6–L25 |
| `predictors()` / `named_predictors()` | `dspy/primitives/module.py` | L87–L93 |
| Instruction proposer | `dspy/propose/grounded_proposer.py` | L270+ |
| Dataset summary generator | `dspy/propose/dataset_summary_generator.py` | L42+ |

---

## Quick Reference Card

```
auto="light"   → N=6,  valset≤100,  ~21 trials (2 predictors)
auto="medium"  → N=12, valset≤300,  ~36 trials
auto="heavy"   → N=18, valset≤1000, ~48 trials

demo_candidates[predictor_idx][demo_set_idx] → list[dspy.Example]
instruction_candidates[predictor_idx][inst_idx] → str

seed=-3  → zero-shot (demos=[])
seed=-2  → raw gold labels (no LLM trace, no augmented=True)
seed=-1  → unshuffled bootstrap (augmented + raw fill)
seed≥ 0  → shuffled bootstrap, random demo count between 1 and max_bootstrapped_demos

predictor.demos = augmented[:max_bootstrapped_demos] + raw[:remaining_budget]
                  where remaining_budget = max_labeled_demos - len(augmented)

num_trials = max(2 × num_predictors × 2 × log2(N), 1.5 × N)
             (×2 because each predictor has 2 Optuna variables: instruction + demos)
```
