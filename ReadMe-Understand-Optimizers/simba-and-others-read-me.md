# DSPy Optimizers — Complete Deep Dive

> A guide that pairs every technical detail with a plain-English story of what is actually happening and why.

---

# Part I — SIMBA

> **Full name**: Stochastic Introspective Mini-Batch Ascent
> **One-line pitch**: SIMBA is a DSPy optimizer that repeatedly samples mini-batches, runs your program multiple times per example with stochastic temperatures, identifies the *highest-variability* examples (where small changes produce big score swings), then uses the LLM to either generate a reflective *rule* that improves the predictor's instructions, or inject a successful *demo* — all without a validation set.

---

## S-0 — The Core Idea

### The Story

Imagine you are a coach trying to improve a student's performance on an exam. One approach is to pick questions at random, mark them, and write generic study notes. A smarter approach is to find the questions where the student sometimes gets it right and sometimes gets it wrong — those are the questions where they are teetering on the boundary of understanding. Once you find those boundary questions, you do two things: you show them a worked example of a correct answer, and you write them a targeted piece of advice explaining *why* the correct approach works.

That is exactly what SIMBA does for an LLM program. Instead of a student, you have a DSPy module. Instead of exam questions, you have training examples. SIMBA runs the same program multiple times on the same example (with slight randomness each time) and watches for examples where the scores scatter wildly — sometimes 1.0, sometimes 0.0. Those are the "boundary" examples. Then it either injects a winning trace as a demonstration or asks the LLM to contrast the winning and losing runs and write a behavioral rule. Over many steps, the program accumulates both targeted demos and targeted rules.

The key philosophical difference from other optimizers: **SIMBA does not need a separate validation set**. It scores everything on the same trainset it is learning from, using mini-batches to keep each step cheap.

### How It Compares

```
MIPROv2:
  trainset → bootstrap demos → generate instructions → Bayesian search over
  (instruction × demo) combinations → return best config

SIMBA:
  trainset
     │
     for step in max_steps:
         sample mini-batch of bsize examples
         run program num_candidates times per example (stochastic LMs)
         find high-variability examples (max_score − avg_score is large)
         for those: LLM reflects on why one run beat another
         apply reflection as a rule (instruction update) OR inject best trace as demo
         register new program, keep growing a pool
     │
     evaluate pool on full trainset → return best
```

### Two Strategies

| Strategy | What It Does | When Used |
|---|---|---|
| `append_a_demo` | Takes the best-scoring trace from the high-variability bucket and injects it as a `demos` entry | When `max_demos > 0` |
| `append_a_rule` | Asks the LLM to compare good vs bad trajectories and produce per-module advice; appends advice to instructions | Always available |

SIMBA randomly picks between these strategies at each bucket, so the program learns from both demonstration and reflection simultaneously.

---

## S-1 — Constructor Parameters

### The Story

Think of the constructor as dialing in how aggressive and how thorough you want SIMBA to be. `bsize` controls how many examples you look at per step — bigger batches give more reliable signal but cost more LLM calls. `num_candidates` controls how many parallel "attempts" you make at improvement in each step — more attempts mean more chances to find a good mutation. `max_steps` is simply how many rounds you want to run. `max_demos` is a cap on how many worked examples accumulate inside a predictor — without a cap, the prompt would grow unboundedly and eventually break.

The two temperature parameters control how greedily or exploratively SIMBA samples from its growing pool of candidate programs. A low temperature means "almost always pick the best program seen so far." A high temperature means "sample more broadly, including weaker programs." Most of the time you want low temperature — exploit what is already working.

```python
SIMBA(
    metric,                          # callable(example, prediction) -> float  (2 args!)
    bsize=32,                        # mini-batch size per step
    num_candidates=6,                # new candidate programs produced per step
    max_steps=8,                     # total optimization steps
    max_demos=4,                     # max demos before random dropping kicks in
    demo_input_field_maxlen=100_000, # truncate long input fields in demos
    num_threads=None,                # parallelism for dspy.Parallel
    temperature_for_sampling=0.2,    # softmax temp when sampling programs for trajectories
    temperature_for_candidates=0.2,  # softmax temp when picking source program for new candidates
)
```

### Critical: SIMBA's metric takes exactly 2 arguments

This is a common breaking difference when migrating from other DSPy optimizers:

```python
# MIPROv2 / BootstrapFewShot metric:  3 args
def metric(example, prediction, trace=None): ...

# SIMBA metric:  2 args — no trace parameter
def metric(example, prediction): ...
```

SIMBA calls `metric(example, prediction)` directly inside `wrap_program`. Passing a 3-arg metric will throw a `TypeError`.

### Parameter Intuition

| Parameter | Low value | High value |
|---|---|---|
| `bsize` | Faster per step, noisier score estimates | Slower, more reliable score estimates |
| `num_candidates` | Fewer programs explored per step | More programs, more chances to improve |
| `max_steps` | Quick run, less optimization | More thorough search |
| `max_demos` | Fewer examples in prompt | More examples, bigger prompts |
| `temperature_for_sampling` | Greedy: always pick highest-scoring program | Exploratory: sample proportionally from all programs |
| `temperature_for_candidates` | Exploit best programs | Explore all programs when building candidates |

---

## S-2 — The Program Pool and Scoring

### The Story

SIMBA is not a single-program optimizer — it maintains a living *population* of programs that grows over time. Think of it like a talent pool. Every time SIMBA creates a new variant of a program (by adding a demo or a rule), it registers that variant into the pool and keeps track of how well it has scored on the mini-batches it has seen. Programs that consistently score high become more "attractive" — future steps preferentially start from them when building the next generation of candidates. Programs that score low are still kept (they might improve later or serve as contrast), but they are rarely chosen as starting points.

Attractiveness is computed via softmax over average scores. At low temperature the distribution is nearly a point mass on the best program. At high temperature it spreads out, allowing weaker programs a chance to contribute.

```python
programs = [student]           # index 0: the original unoptimized baseline
program_scores = {0: []}       # maps simba_idx → list of mini-batch scores

def calc_average_score(prog_idx):
    scores = program_scores.get(prog_idx, [])
    return sum(scores) / len(scores) if scores else 0.0

def softmax_sample(rng, program_idxs, temperature):
    scores = [calc_average_score(idx) for idx in program_idxs]
    exps = [np.exp(s / temperature) for s in scores]
    probs = [e / sum(exps) for e in exps]
    return rng.choices(program_idxs, weights=probs)[0]
```

---

## S-3 — One Complete Optimization Step

### The Story

Picture one step of SIMBA as a mini-experiment conducted inside a laboratory. You pull out 32 specimens (training examples). You run the same experiment (program forward pass) 6 times on each specimen using slightly different conditions (randomized LM temperature and rollout seeds). You then look at each specimen's 6 results and ask: "which specimens showed the most inconsistency?" Those inconsistent specimens are your most valuable data — the program is right on the edge of figuring out how to handle them. You pick those specimens, study the best and worst runs, and use that contrast to either write a new behavioral guideline (a rule) or pull out a worked example (a demo). You test your modified program on the same batch, record the score, and move on to the next step.

### Sub-step 1: Sample the mini-batch

32 examples are drawn sequentially from a shuffled trainset index. When the trainset is exhausted, the indices are re-shuffled — so examples cycle through evenly over many steps.

```python
batch_indices = data_indices[instance_idx : instance_idx + self.bsize]
batch = [trainset[i] for i in batch_indices]
instance_idx += self.bsize
```

### Sub-step 2: Build execution pairs

Each of `num_candidates` LMs (with different rollout IDs and temperature=1.0) is paired with each example in the batch. The different LMs ensure stochastic variation — the same program run twice on the same input will produce different outputs.

```python
models = prepare_models_for_resampling(programs[0], self.num_candidates)
# Returns num_candidates LMs with different rollout_ids and temperature=1.0

for model in models:             # 6 models
    for example in batch:        # 32 examples
        chosen_prog_idx = softmax_sample(rng, top_programs, temperature_for_sampling)
        candidate_system = programs[chosen_prog_idx].deepcopy()
        candidate_system.set_lm(model)
        exec_pairs.append((wrap_program(candidate_system, metric), example))
# Total: num_candidates × bsize = 6 × 32 = 192 forward passes
```

### Sub-step 3: Execute in parallel

```python
outputs = run_parallel(exec_pairs)
# Each output: {"prediction": ..., "trace": [...], "score": float, "example": ...}
```

### Sub-step 4: Build and sort buckets

Outputs are grouped by example into **buckets** of size `num_candidates`. Each bucket contains 6 outcomes for the same input. The buckets are then sorted by variability — high gap between best and worst score comes first.

```python
for idx, example in enumerate(batch):
    bucket = [outputs[i] for i in range(idx, len(outputs), bsize)]
    bucket.sort(key=lambda x: x["score"], reverse=True)

    max_score = bucket[0]["score"]
    avg_score = mean(scores)
    max_to_avg_gap = max_score - avg_score
    # sort all buckets: highest variability first
```

**Why variability?** A bucket where all 6 runs score 1.0 means the program already handles this example well — no learning needed. A bucket where all 6 score 0.0 means the program is completely lost — no good example to learn from. A bucket where scores range from 0.0 to 1.0 is the sweet spot: there exists a winning trajectory AND a losing trajectory on the exact same input, making the contrast maximally informative.

### Sub-step 5: Apply strategies to build new candidate programs

```python
for bucket, bucket_stats in buckets:
    src_prog_idx = softmax_sample(rng, top_k_plus_baseline, temperature_for_candidates)
    system_candidate = programs[src_prog_idx].deepcopy()

    # Drop some demos first (Poisson-distributed count) — prevents runaway accumulation
    num_demos_to_drop = max(poisson(num_demos / max_demos), int(num_demos >= max_demos))

    # Pick and apply a strategy
    strategy = rng.choice(self.strategies)   # append_a_demo OR append_a_rule
    strategy(bucket, system_candidate, ...)
    system_candidates.append(system_candidate)
```

### Sub-step 6: Evaluate new candidates on the same batch

Each of the newly created candidate programs is scored on the same 32-example batch. This gives an apples-to-apples comparison: the same data for all candidates.

### Sub-step 7: Record the best, register all

```python
best_idx = candidate_scores.index(max(candidate_scores))
winning_programs.append(system_candidates[best_idx].deepcopy())

for cand_sys in system_candidates:
    register_new_program(cand_sys, sys_scores)
```

`winning_programs` grows by exactly 1 per step — it is a timestamped checkpoint of "best program seen at the end of step N."

---

## S-4 — Strategy 1: `append_a_demo`

### The Story

When a high-variability bucket is selected and SIMBA rolls "demo injection," it asks a simple question: of the 6 runs on this example, which one produced the best outcome? It takes the full execution trace of that winning run — every intermediate input and output from every predictor — and packages it as a worked example. That worked example is then appended to the predictor's `demos` list, so that future runs of the program will see this successful trace in their few-shot prompt, effectively showing them "here is how to handle a situation like this."

```python
def append_a_demo_(bucket, system, **kwargs):
    trace = bucket[0]["trace"]     # best-scoring run's full trace

    for step in trace:
        predictor, _inputs, _outputs = step

        # Truncate very long input fields to avoid context overflow
        for k, v in _inputs.items():
            if len(str(v)) > demo_input_field_maxlen:
                _inputs[k] = f"{str(v)[:demo_input_field_maxlen]}\n\t\t... <TRUNCATED>"

        demo = dspy.Example(augmented=True, **_inputs, **_outputs)
        name2demo[predictor_name] = demo

    for name, demo in name2demo.items():
        name2predictor[name].demos.append(demo)
```

Key detail: `augmented=True` tells DSPy to render intermediate fields like `reasoning` when this demo appears in a future prompt. Without it, only the final answer would be shown.

---

## S-5 — Strategy 2: `append_a_rule`

### The Story

This is SIMBA's most intellectually interesting move. When it rolls "append a rule," it takes the best run and the worst run for the same example and hands both to the LLM with a simple brief: "these two runs received the same input but got very different scores — why? And what should each module do differently?"

The LLM does a kind of post-mortem analysis. It reads the full trajectory of the winning run, the full trajectory of the losing run, and the program's own source code. It then writes a concrete, targeted piece of advice for each predictor involved — not generic platitudes, but specific behavioral instructions like "when the question asks for a year, extract the four-digit number directly from the context rather than paraphrasing." That advice is then appended directly to the predictor's instruction string. Instructions grow over time as rules accumulate.

The quality gate before this step is important: if even the "best" run on this example is not good enough (below the 10th percentile batch score), or if even the "worst" run is already acceptable (above the 90th percentile), then the contrast between them is not meaningful and the step is skipped.

```python
def append_a_rule(bucket, system, **kwargs):
    batch_10p_score, batch_90p_score = kwargs["batch_10p_score"], kwargs["batch_90p_score"]

    good, bad = bucket[0], bucket[-1]

    # Quality gate: skip unhelpful contrasts
    if good["score"] < batch_10p_score or bad["score"] > batch_90p_score:
        return False

    # Serialize both trajectories for the LLM
    better_trajectory = [
        {"module_name": name, "inputs": i, "outputs": dict(o)}
        for name, i, o in good["trace"]
    ]
    worse_trajectory = [...]   # same for bad["trace"]

    # Ask the LLM to reflect and generate advice
    advice = dspy.Predict(OfferFeedback)(
        program_code=inspect.getsource(system.__class__),
        better_program_trajectory=better_trajectory,
        worse_program_trajectory=worse_trajectory,
        oracle_metadata={**example.labels()},
        module_names=[name for name, _ in system.named_predictors()],
    ).module_advice   # dict[str, str]

    # Append advice to each predictor's instructions
    for name, predictor in system.named_predictors():
        if name in advice:
            new_instructions = predictor.signature.instructions + "\n\n" + advice[name]
            predictor.signature = predictor.signature.with_instructions(new_instructions)
```

---

## S-6 — The `OfferFeedback` Signature

### The Story

`OfferFeedback` is the LLM "judge" at the heart of `append_a_rule`. It receives both trajectories side-by-side, the program's own source code so it understands the intended architecture, the correct answer (oracle metadata) so it knows what success looks like, and the list of module names. It is asked to first *discuss blame* — where in the execution did things go wrong? — and then output a concrete dictionary mapping each module name to a specific behavioral guideline.

The docstring is carefully crafted: "The module won't see its own history. It will rely on your advice balancing being concrete and being generalizable." This constraint forces the LLM to write rules that work across many inputs, not just for this one example.

```python
class OfferFeedback(dspy.Signature):
    """
    You will be given two trajectories of an LLM-driven program's execution.
    The module won't see its own history. It will rely on your advice balancing
    being concrete and being generalizable.

    In your advice:
    - Avoid boilerplate. Offer advice that would change the module's behavior.
    - Ensure advice to module M is specific to M's sub-task.
    - Rely on contrasting the worse trajectory against the better trajectory.
    - Ensure each unique module name appears exactly once as a key in the advice dict.
    """
    program_code: str = InputField(...)
    modules_defn: str = InputField(...)
    program_inputs: str = InputField(...)
    oracle_metadata: str = InputField(...)
    worse_program_trajectory: str = InputField(...)
    better_program_trajectory: str = InputField(...)
    better_reward_value: float = InputField(...)
    worse_reward_value: float = InputField(...)
    module_names: list[str] = InputField(...)

    discussion: str = OutputField(desc="Discussing blame: where did each module go wrong?")
    module_advice: dict[str, str] = OutputField(
        desc="For each module: If input is X, then module should Y."
    )
```

---

## S-7 — Demo Dropping: Poisson-Based Eviction

### The Story

Left unchecked, `append_a_demo` would keep adding worked examples to a predictor's demo list until the prompt overflowed the context window. SIMBA prevents this with a neat probabilistic trick: before applying any strategy to a candidate program, it randomly evicts some existing demos using a Poisson-distributed drop count.

The Poisson distribution is parameterized by `num_demos / max_demos`. When you have very few demos, the expected number of drops is near zero — almost nothing is dropped. As demos accumulate toward `max_demos`, the expected drop count rises toward 1. Once you hit `max_demos`, at least one demo is always dropped (floor is forced to 1). This creates a soft carrying capacity: the list is allowed to grow, but gets regularly pruned so the program never over-fits to a fixed set of examples.

```python
num_demos = max(len(predictor.demos) for predictor in ...)
num_demos_to_drop = max(
    rng_np.poisson(num_demos / max_demos),   # expected drops = current_count / max_demos
    int(num_demos >= max_demos)              # force ≥1 drop when at capacity
)
num_demos_to_drop = min(num_demos_to_drop, num_demos)
demos_to_drop = [rng.randrange(num_demos) for _ in range(num_demos_to_drop)]

for _, predictor in name2predictor.items():
    predictor.demos = [d for i, d in enumerate(predictor.demos) if i not in demos_to_drop]
```

---

## S-8 — Final Selection: Evenly-Spaced Winning Programs

### The Story

After all steps complete, SIMBA has a list of checkpoint programs — one saved after each step. But which one is truly best? Early-step programs have simpler, more general instructions. Late-step programs have richer, more specialized instructions that might generalize less well. Rather than just taking the last checkpoint, SIMBA samples the full timeline evenly — a few checkpoints from early, middle, and late in the run — and evaluates them all on the *entire* trainset (not just a mini-batch). This is the one real full-data evaluation in the whole process, and it is used exclusively for the final selection.

```python
M = len(winning_programs) - 1
N = self.num_candidates + 1

# Evenly-spaced indices across winning_programs[0..M]
program_idxs = [round(i * M / (N - 1)) for i in range(N)]
candidate_programs = [winning_programs[i].deepcopy() for i in program_idxs]

# Evaluate all on full trainset
exec_pairs = [(wrap_program(sys, metric), ex) for sys in candidate_programs for ex in trainset]
outputs = run_parallel(exec_pairs)

scores = [mean(output_scores) for each candidate]
best_program = candidate_programs[argmax(scores)]
best_program.candidate_programs = sorted_candidates
return best_program
```

---

## S-9 — What Accumulated Instructions Look Like Over Time

### The Story

One of the most human-readable artifacts of a SIMBA run is the evolution of a predictor's instruction string. It starts as a bare DSPy default and grows organically as `append_a_rule` fires on high-variability examples. Each appended rule is a specific, task-grounded behavioral guideline — not generic prompt engineering advice, but lessons the LLM drew directly from examining real failures on real training examples.

**Step 0 (baseline):**
```
Given the fields `question` and `context`, produce the fields `answer`.
```

**After step 2 (one rule added):**
```
Given the fields `question` and `context`, produce the fields `answer`.

If the question asks for a specific year or date, extract it directly from the context
without paraphrasing. Prefer exact quoted values over approximations when the context
provides them.
```

**After step 5 (two more rules added):**
```
Given the fields `question` and `context`, produce the fields `answer`.

If the question asks for a specific year or date, extract it directly from the context
without paraphrasing. Prefer exact quoted values over approximations when the context
provides them.

When the question asks for a person's name (e.g., "who wrote X"), return the full name
as it appears in the context, not an abbreviated or informal form.

If the context contains multiple plausible answers, prefer the one that directly addresses
the syntactic structure of the question (e.g., "who" → a person, "when" → a date).
```

---

## S-10 — Complete End-to-End Example

```python
import dspy
from dspy.teleprompt import SIMBA

# 1. Define program
class MultiHopQA(dspy.Module):
    def __init__(self):
        self.reason = dspy.ChainOfThought("question, context -> answer")

    def forward(self, question, context):
        return self.reason(question=question, context=context)

# 2. Build trainset (at least bsize examples recommended)
trainset = [
    dspy.Example(
        question="What year was Python created?",
        context="Python is a high-level programming language created by Guido van Rossum.",
        answer="1991"
    ).with_inputs("question", "context"),
    # ... more examples
]

# 3. Metric — 2 args, no trace
def answer_exact_match(example, prediction):
    if prediction is None:
        return 0.0
    return float(str(prediction.answer).strip().lower() == str(example.answer).strip().lower())

# 4. Configure LM
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# 5. Compile
simba = SIMBA(
    metric=answer_exact_match,
    bsize=16,
    num_candidates=4,
    max_steps=5,
    max_demos=3,
)
compiled = simba.compile(MultiHopQA(), trainset=trainset, seed=42)

# 6. Use
result = compiled(question="What does CPU stand for?", context="A CPU (Central Processing Unit) is...")
print(result.answer)

# 7. Inspect candidates
for item in compiled.candidate_programs:
    print(f"score={item['score']:.3f}, demos={len(item['program'].reason.demos)}")

# 8. Inspect accumulated rules
print(compiled.reason.signature.instructions)
```

---

## S-11 — Common Pitfalls

### Pitfall 1: Trainset smaller than bsize
```python
assert len(trainset) >= self.bsize
# AssertionError: Trainset too small: 20 < 32
```
Fix: lower `bsize` to match your trainset size, or add more examples.

### Pitfall 2: 3-argument metric
SIMBA calls `metric(example, prediction)`. Passing a 3-arg metric raises `TypeError`.
Fix: ensure your metric signature is exactly `(example, prediction)`.

### Pitfall 3: Instructions growing too long
After many steps with `append_a_rule`, instructions can become very long (many appended paragraphs), increasing token cost and confusing the LLM.
Fix: lower `max_steps`, or inspect `predictor.signature.instructions` periodically.

### Pitfall 4: Zero variability across runs
If every run on every example gets the same exact score, no bucket has variability and SIMBA produces weak candidates. This happens when the task is trivially easy (metric always 1.0), trivially hard (metric always 0.0), or `bsize` is too small to see variation.
Fix: verify your metric returns a range of values on the trainset before running SIMBA.

### Pitfall 5: `append_a_rule` always skips
If `good["score"] < batch_10p_score` for most buckets, the rule strategy is always skipped and SIMBA degenerates into demo-only optimization.
Fix: check that your metric is calibrated and the LM produces varied-quality outputs across runs.

### Pitfall 6: Forgetting `.with_inputs()`
`example.inputs()` returns `{}` without `.with_inputs()`. The program will be called with no arguments and fail.

---

## S-12 — Saving and Loading

Both the demos and the accumulated instruction rules are serialized to JSON:

```python
compiled.save("simba_compiled.json")

loaded = MultiHopQA()
loaded.load("simba_compiled.json")
# loaded.reason.demos and loaded.reason.signature.instructions are both restored
```

---

## S-13 — SIMBA vs MIPROv2 vs BootstrapFewShot

| Feature | SIMBA | MIPROv2 | BootstrapFewShot |
|---|---|---|---|
| Demo optimization | YES (append_a_demo) | YES (bootstrap) | YES (bootstrap) |
| Instruction optimization | YES (LLM reflection) | YES (GroundedProposer + Optuna) | NO |
| Search strategy | Stochastic mini-batch ascent | Bayesian TPE (Optuna) | Single pass |
| Validation set required | NO | YES (recommended) | NO |
| Instruction generation style | Self-reflective contrast (good vs bad trace) | Proposed by meta-LM from data summary | N/A |
| Instruction style | Additive rule append | Discrete proposals | N/A |
| Best for | Task-specific behavioral rules + demos | Full optimization with explicit search budget | Fast baseline / distillation |

---

## S-14 — Quick Reference Card

```python
compiled = SIMBA(
    metric=my_2arg_metric,          # (example, prediction) -> float
    bsize=32,
    num_candidates=6,
    max_steps=8,
    max_demos=4,
    demo_input_field_maxlen=100_000,
    num_threads=None,
    temperature_for_sampling=0.2,
    temperature_for_candidates=0.2,
).compile(
    student=MyProgram(),
    trainset=my_trainset,           # list[dspy.Example] with .with_inputs()
    seed=0,
)
```

**What SIMBA optimizes per predictor:**
- `predictor.demos` ← grows via `append_a_demo`
- `predictor.signature.instructions` ← grows via `append_a_rule`

**Per-step inference budget (approximate):**
- Trajectory sampling: `bsize × num_candidates` calls (e.g., 192)
- Candidate evaluation: `num_candidates × bsize` calls (e.g., 224)
- Rule generation: up to `num_candidates` OfferFeedback calls
- Final validation: `num_candidate_programs × len(trainset)`

---
---

# Part II — All Other DSPy Optimizers

---

## Optimizer Map

| Class | What It Optimizes | Core Technique |
|---|---|---|
| `BootstrapFewShotWithRandomSearch` | Demos | N random bootstrap runs → valset selects best |
| `COPRO` | Instructions only | Iterative proposal + coordinate-wise hill-climb |
| `InferRules` | Instructions (as rules) | Bootstrap + LLM induces natural-language rules |
| `KNNFewShot` | Demos (dynamic at inference time) | KNN retrieval per query |
| `AvatarOptimizer` | Instructions for tool-use programs | Positive/negative contrast → feedback → rewrite |
| `Ensemble` | N/A (post-hoc combination) | Runs N programs, reduces with majority or custom fn |
| `BootstrapFinetune` | LM weights | Bootstrap traces → fine-tune the LM itself |
| `GRPO` | LM weights | Group Relative Policy Optimization |
| `BetterTogether` | Prompts + weights | Alternating prompt and weight optimization |

---

## OPT-1 — BootstrapFewShotWithRandomSearch

### The Story

The simplest upgrade from `BootstrapFewShot` is to run it multiple times with different random seeds and pick the winner. That is exactly what this optimizer does. It runs N independent bootstrap passes, each with a slightly different random seed that changes which training examples get selected and how many demos are used per predictor (ranging from zero-shot up to `max_bootstrapped_demos`). All N candidate programs are then evaluated on a held-out validation set, and the one with the highest score is returned.

The zero-shot run (0 bootstrapped demos) is always included as candidate #0, acting as a baseline. This means if bootstrapping actually hurts (sometimes it does, especially on easy tasks), the optimizer will correctly fall back to zero-shot.

```python
BootstrapFewShotWithRandomSearch(
    metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
    num_candidate_programs=16,   # number of random bootstrap runs
    num_threads=None,
    stop_at_score=None,          # early exit if this score is reached
    metric_threshold=None,
)
```

**Algorithm in plain English:**
1. Run `BootstrapFewShot` with 0 demos → candidate 0 (zero-shot baseline)
2. For each of N remaining seeds: run `BootstrapFewShot` with random demo count (1 to max) → candidate
3. Evaluate all candidates on valset in parallel
4. Return the candidate with the highest valset score

**Key difference from MIPROv2:** This optimizer only searches over demo sets. It does not generate or optimize instructions. If you need instruction optimization, use MIPROv2 or SIMBA instead.

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

compiled = BootstrapFewShotWithRandomSearch(
    metric=my_metric,
    max_bootstrapped_demos=3,
    max_labeled_demos=8,
    num_candidate_programs=10,
    num_threads=8,
).compile(
    student=MyProgram(),
    trainset=trainset,
    valset=valset,
)
```

**When to use:** You want better demos than a single bootstrap run, you do not need instruction optimization, and you have a modest budget (10–16x a single bootstrap run cost).

---

## OPT-2 — COPRO

### The Story

COPRO treats instruction optimization as a coordinate-wise hill-climbing problem. It works on one predictor at a time. For that predictor it generates `breadth` candidate instructions, evaluates them all, and keeps the best. Then in the next round it shows the LLM the full history of instructions already tried — sorted from worst to best, so the best are most recently in context — and asks it to propose `breadth` more. This continues for `depth` rounds. Because the LLM sees what has already been attempted and what scored well, its proposals tend to improve over rounds rather than just sampling randomly.

After COPRO finishes all rounds for one predictor, it locks in the best instruction found and moves to the next predictor. This coordinate-wise approach avoids the exponential cost of searching all predictor instructions jointly. The trade-off is that it might miss interaction effects between predictors.

```python
COPRO(
    prompt_model=None,    # LM used to generate instruction proposals
    metric=None,
    breadth=10,           # instructions proposed per round
    depth=3,              # iterative refinement rounds
    init_temperature=1.4, # temperature for the proposer LM
    track_stats=False,
)
```

**Algorithm:**

```
For each predictor P in the program:
    Round 0: generate breadth instructions from the baseline via BasicGenerateInstruction
    Rounds 1..depth:
        Evaluate all candidates on trainset
        Sort by score, best last (so best are most recent for the LLM)
        Generate breadth new proposals using GenerateInstructionGivenAttempts
    Pick the highest-scoring instruction for P
```

**What COPRO does NOT do:** It does not touch demos. Only `predictor.signature.instructions` (and optionally the output prefix) are changed. Combine with BootstrapFewShot first if you want both.

```python
from dspy.teleprompt import COPRO

compiled = COPRO(
    prompt_model=dspy.LM("openai/gpt-4o"),
    metric=my_metric,
    breadth=10,
    depth=3,
    init_temperature=1.4,
).compile(
    student=MyProgram(),
    trainset=trainset[:50],
    eval_kwargs={"num_threads": 8, "display_progress": True, "display_table": 0},
)
```

**COPRO vs MIPROv2:**

| | COPRO | MIPROv2 |
|---|---|---|
| Instructions | Iterative hill-climb per predictor | Global search (Optuna TPE) |
| Demos | NOT optimized | YES (bootstrapped) |
| Proposal method | LM sees history of attempts | GroundedProposer from dataset summary |
| Validation | Per-predictor, per-round | Full program eval per trial |

---

## OPT-3 — InferRules

### The Story

InferRules is the middle ground between BootstrapFewShot (which only adds demos) and MIPROv2 (which runs a full Bayesian search). It first bootstraps demos the normal way, but then takes an extra step: it looks at all the bootstrapped examples for each predictor and asks the LLM to *induce natural-language rules* that explain the patterns in those examples. These rules are appended to the predictor's instructions, giving the program both worked examples and an explicit behavioral summary of what the examples are trying to teach.

Multiple candidates are generated (different rule sets) and evaluated on a held-out split, with the best-scoring candidate returned.

```python
InferRules(
    num_candidates=10,    # rule-instruction candidate programs to evaluate
    num_rules=10,         # rules to induce per predictor
    # + all BootstrapFewShot kwargs
)
```

**Algorithm:**
1. Split trainset 50/50 into train and validation
2. Run `BootstrapFewShot` on the train split → student with demos
3. For each candidate index: call `RulesInductionProgram` to analyze bootstrapped demos and output N rules; append rules to predictor instructions; evaluate on validation split
4. Return best-scoring candidate

Example rules that might be induced:
```
1. When the input contains a unit conversion, always include the unit in the final answer.
2. For multi-part questions, answer each part explicitly.
3. Default to conservative estimates when numerical precision is unclear.
```

**When to use:** You want instruction augmentation on top of bootstrapped demos but cannot afford the full Bayesian search cost of MIPROv2.

---

## OPT-4 — KNNFewShot

### The Story

Most few-shot optimizers choose demos at compile time and lock them into the prompt. The problem is that the best demo for a math question is different from the best demo for a geography question. KNNFewShot solves this by not choosing demos at compile time at all — instead it patches the program's `forward` method so that at inference time, for every new query, it performs a vector similarity search over the entire trainset and retrieves the k most similar examples. Those k retrieved examples are then bootstrapped into demos on the fly.

The result is a program whose demonstrations are always tailored to the specific input being processed, not just the average input in the training distribution.

```python
KNNFewShot(
    k=3,                      # neighbors to retrieve
    trainset=my_trainset,
    vectorizer=my_embedder,   # dspy.Embedder instance
    # + any BootstrapFewShot kwargs
)
```

```python
import dspy
from dspy.teleprompt import KNNFewShot
from sentence_transformers import SentenceTransformer

knn = KNNFewShot(
    k=3,
    trainset=trainset,
    vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode),
    metric=my_metric,
    max_bootstrapped_demos=2,
)
compiled = knn.compile(qa_program)
```

**Trade-offs:**

| | Benefit | Cost |
|---|---|---|
| Relevance | Demos always match current query domain | Per-query embedding lookup overhead |
| Coverage | Large trainsets provide diverse coverage | Re-runs BootstrapFewShot per call |
| Flexibility | Works on any task | Requires an embedding model |

**When to use:** High input diversity (e.g., math, geography, code all in one dataset), large enough trainset that fixed demos give poor coverage, and you can afford per-query embedding latency.

---

## OPT-5 — AvatarOptimizer

### The Story

AvatarOptimizer is built for a specific kind of DSPy program: an `Avatar` agent that uses tools (search, calculator, code execution, etc.). The core challenge for tool-using agents is that the instruction guiding the actor — the text that tells it when to use which tool and how to formulate tool calls — is crucial and hard to write by hand. AvatarOptimizer automates this.

It evaluates the agent on the trainset and splits the results into positives (ran well, score above threshold) and negatives (ran poorly, score below threshold). It shows both sets to a `Comparator` LLM that writes concrete behavioral feedback: "in these positive cases, the agent correctly used the search tool before answering; in the negative cases, it attempted to answer from memory." Then a `FeedbackBasedInstruction` LLM rewrites the actor's instruction incorporating that feedback. If the new instruction scores better, it is kept; otherwise the old instruction is restored.

```python
AvatarOptimizer(
    metric,
    max_iters=10,
    lower_bound=0,      # threshold below which a result counts as negative
    upper_bound=1,      # threshold above which a result counts as positive
    max_positive_inputs=None,
    max_negative_inputs=None,
)
```

**Algorithm:**
```
For iter in range(max_iters):
    Evaluate program on trainset
    Split into positive (score ≥ upper_bound) and negative (score ≤ lower_bound) examples
    Comparator(pos_inputs, neg_inputs) → feedback
    FeedbackBasedInstruction(prev_instruction, feedback) → new_instruction
    Update actor instruction; evaluate; keep if improved
```

**When to use:** You are using DSPy's `Avatar` module, you have a clear positive/negative outcome split, and you want to improve the natural-language instruction governing tool selection.

---

## OPT-6 — Ensemble

### The Story

Ensemble is not really an optimizer in the learning sense — it does not touch prompts or weights. Instead, it is a post-hoc combiner. You compile several independently-optimized programs (using any optimizer, any seed), then wrap them all in an `EnsembledProgram` that runs all of them on every query and reduces their outputs using a function you specify, most commonly majority vote.

The value of ensembling comes from variance reduction. If each program makes different mistakes, the majority is often right even when individual programs are wrong. It is the same logic that makes random forests more accurate than individual decision trees.

```python
Ensemble(
    reduce_fn=None,        # e.g., dspy.majority
    size=None,             # if set, randomly sample `size` programs per call
)
```

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, Ensemble

programs = []
for seed in range(5):
    compiled = BootstrapFewShotWithRandomSearch(
        metric=my_metric, num_candidate_programs=8
    ).compile(MyProgram(), trainset=trainset, valset=valset)
    programs.append(compiled)

ensemble_program = Ensemble(reduce_fn=dspy.majority).compile(programs)
result = ensemble_program(question="Who invented the telephone?")
```

**When to use:** You have compiled multiple programs with different optimizers or seeds; majority voting improves robustness on noisy or ambiguous tasks; you want variance reduction without re-running optimization.

---

## OPT-7 — BootstrapFinetune

### The Story

All the optimizers discussed so far work by modifying what goes into the LLM's prompt — the instructions and the few-shot examples. `BootstrapFinetune` takes a fundamentally different approach: it modifies the LLM's *weights*. It first generates bootstrapped traces (the same way `BootstrapFewShot` does), but instead of injecting them as demos, it converts them into fine-tuning data and submits a fine-tuning job to the LM provider. The resulting fine-tuned model checkpoint is then plugged back into the student program.

The practical consequence is that a smaller, cheaper model can be taught to behave like a larger one on the specific task. At inference time you pay less per call, and your prompts stay short because the model has internalized the behavior.

```python
BootstrapFinetune(
    metric=None,
    multitask=True,        # fine-tune all predictors jointly
    train_kwargs=None,     # e.g., {"epochs": 3}
    exclude_demos=False,
)
```

**Critical differences:**

| | BootstrapFinetune | All other optimizers |
|---|---|---|
| What changes | LM weights | Prompts (instructions + demos) |
| Reversible | No (new model checkpoint) | Yes (just in-memory objects) |
| Inference cost | Lower (smaller model can work) | Same LM, larger prompts |
| Requires LM fine-tuning API | YES | NO |

**When to use:** You have hundreds of training examples; you want to distill prompt-based performance into a smaller/cheaper model; your LM provider supports fine-tuning (OpenAI, together.ai, etc.); deployment cost matters more than optimization cost.

---

## OPT-8 — GRPO

### The Story

GRPO (Group Relative Policy Optimization) is the DSPy integration of the training algorithm popularized by DeepSeek-R1. Where `BootstrapFinetune` is supervised learning (show the model correct answers), GRPO is reinforcement learning (let the model explore, then score what it produces and nudge it toward the high-scoring outputs).

For each training example, GRPO generates a *group* of G rollouts from the current model. It scores every rollout with your metric function. Then it normalizes the rewards within the group — a rollout that scores higher than the group average gets a positive advantage, one that scores lower gets a negative advantage. These normalized advantages drive a policy gradient update that pushes the model's weights to produce more of the good rollouts and fewer of the bad ones.

This requires a local LM with gradient access — a hosted API that only supports inference is not sufficient.

```python
GRPO(
    metric=None,
    num_train_steps=100,
    num_rollouts_per_grpo_step=1,
    num_dspy_examples_per_grpo_step=1,
    use_train_as_val=False,
    failure_score=0,
    format_failure_score=-1,
)
```

**GRPO vs BootstrapFinetune:**

| | GRPO | BootstrapFinetune |
|---|---|---|
| Training signal | RL reward (group-relative) | Supervised (teacher traces) |
| Exploration | YES (multiple rollouts) | NO (teacher-determined outputs) |
| Requires gradient access | YES | Depends on provider |
| Output | RL-trained weights | SFT-trained weights |

**When to use:** You have GPU compute for RL-based training; your task requires behavioral improvement that prompting alone cannot achieve; you are training a small local model to match a larger one's capabilities.

---

## OPT-9 — BetterTogether (Experimental)

### The Story

Prompting and fine-tuning are complementary strategies: good prompts help a fine-tuned model generalize, and a fine-tuned model needs fewer (or simpler) prompts to perform well. `BetterTogether` alternates between the two in a user-specified sequence, passing the output of each step as the input to the next. The simplest useful strategy is `"p → w → p"`: first optimize prompts for the baseline model, then fine-tune the prompted model, then re-optimize prompts for the fine-tuned model (which now needs different guidance than the baseline).

```python
BetterTogether(
    metric,
    prompt_optimizer=None,   # default: BootstrapFewShotWithRandomSearch
    weight_optimizer=None,   # default: BootstrapFinetune
    seed=None,
)

# Requires: dspy.settings.experimental = True

compiled = bt.compile(
    student=MyProgram(),
    trainset=trainset,
    strategy="p -> w -> p",
    valset_ratio=0.1,
)
```

**Strategy tokens:**
- `p` = run prompt optimizer
- `w` = run weight optimizer

**When to use:** You want the best of both worlds; your deployment target is a fine-tunable model; you can afford 3x the optimization cost of a single optimizer run.

---

## Full Optimizer Decision Tree

```
Do you want to fine-tune LM weights?
    YES →
        RL-style training?             → GRPO
        Supervised distillation?       → BootstrapFinetune
        Both prompts and weights?      → BetterTogether (experimental)
    NO →
        Program uses tools / Avatar?   → AvatarOptimizer
        Need dynamic per-query demos?  → KNNFewShot
        Want to ensemble N programs?   → Ensemble (post-hoc)
        Want instructions + demos?
            Fast baseline?             → BootstrapFewShot
            Best demo set only?        → BootstrapFewShotWithRandomSearch
            Instruction rules too?     → InferRules
            Full instruction search?
                Iterative refinement?  → COPRO
                Bayesian search?       → MIPROv2
                Mini-batch stochastic? → SIMBA
                Evolutionary?          → GEPA
```

---

## Summary Table

| Optimizer | Demos | Instructions | Weights | Search Strategy |
|---|---|---|---|---|
| `BootstrapFewShot` | YES | NO | NO | Single pass |
| `BootstrapFewShotWithRandomSearch` | YES | NO | NO | Random N runs |
| `COPRO` | NO | YES | NO | Hill-climb per predictor |
| `InferRules` | YES | YES (rules) | NO | Bootstrap + induction |
| `KNNFewShot` | YES (dynamic) | NO | NO | KNN per query at runtime |
| `AvatarOptimizer` | NO | YES (actor) | NO | Feedback loop |
| `Ensemble` | N/A | N/A | N/A | None (post-hoc) |
| `MIPROv2` | YES | YES | NO | Bayesian TPE (Optuna) |
| `SIMBA` | YES | YES (rules) | NO | Mini-batch stochastic |
| `GEPA` | NO | YES | NO | Pareto + reflection |
| `BootstrapFinetune` | NO | NO | YES (SFT) | N/A |
| `GRPO` | NO | NO | YES (RL) | N/A |
| `BetterTogether` | YES | YES | YES | Prompt + weight alternating |