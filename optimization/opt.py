"""
Wrapper-only GEPA/DSPy multi-prompt optimization orchestration -- NO run.py analog.

RELATIONSHIP TO run.py (MBD/weakspot_code_to_run_2/run.py):
    run.py has NO optimization stage. Its run_pipeline() (~837-897) runs each
    weakspot (FC) exactly ONCE: Phase 1 _locate_weakspot -> Phase 2
    _build_equation_context -> Phase 3a PROMPT 1 (_step_3a_get_context) ->
    Phase 3b PROMPT 2 (_step_3b_generate_testcases) -> Phase 4 PROMPT 4
    (_step_4a_solve_equations_dll) -> XLSM. The four prompt TEMPLATES are fixed
    strings; run.py never rewrites them and computes NO score.

    This module is the [wrapper-only] layer that DOES NOT exist in run.py: it
    drives a DSPy teleprompter (GEPA / MIPROv2 / SIMBA / Bootstrap) that calls
    optimizer.compile() to OPTIMIZE the four prompt-instruction strings. Those
    instruction strings are the frozen __doc__ seeds in signatures/mbd.py
    (VERBATIM run.py templates); GEPA reflectively rewrites them to raise the
    M1..M8 feedback metric (evaluation/mbd_metric.py) -- which itself is a
    wrapper-only addition, since run.py produces no metric.

    Data flow feeding this optimizer is precomputed elsewhere and has no analog
    here: Phase 1/2 are serialized into each dspy.Example by utils/mbd_data.py
    (which imports run.py's _locate_weakspot + _build_equation_context), and
    Phase 3a/3b/4a + post-process are inlined in
    extract_mbd_full.MBDFullPipelineModule.forward(). This file only configures
    the optimizer, evaluates a baseline, runs compile(), and reports the best
    program -- none of which run.py does.

    [wrapper-only] Everything below (OptimizationConfig, OptimizationRunner, the
    create_*_strategy factories) is pure optimization scaffolding with no
    run.py correspondence.
"""
from dataclasses import dataclass, field  # [wrapper-only] declarative OptimizationConfig; run.py has no config dataclass
from datetime import datetime  # [wrapper-only] wall-clock stamps for run/strategy summaries (run.py records no timings)
import inspect  # [wrapper-only] used to feature-detect GEPA kwargs (reflection_lm vs prompt_model) across DSPy versions
import time  # [wrapper-only] duration measurement for baseline/strategy summaries; no run.py analog
from typing import List, Optional, Literal, Dict, Any  # [wrapper-only] type hints for the optimizer scaffolding
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, MIPROv2, SIMBA  # [wrapper-only] DSPy teleprompters; run.py never optimizes prompts
from dspy import GEPA  # [wrapper-only] reflective prompt optimizer (rewrites the 4 mbd.py __doc__ seeds); absent from run.py


# [wrapper-only] error raised when a DSPy optimizer strategy fails -- no run.py analog.
class StrategyExecutionError(RuntimeError):
    """Optimization strategy failed after the underlying optimizer exhausted its retries."""

    def __init__(self, strategy_name: str, optimizer_type: str, cause: Exception):
        self.strategy_name = strategy_name  # [wrapper-only] which OptimizationConfig.name failed
        self.optimizer_type = optimizer_type  # [wrapper-only] "bootstrap"/"gepa"/"miprov2"/"simba"
        self.cause = cause  # [wrapper-only] original optimizer exception, chained below via `from exc`
        super().__init__(  # [wrapper-only] compose human-readable message for the run summary
            f"Strategy '{strategy_name}' ({optimizer_type}) failed: {cause}"
        )


# [wrapper-only] declarative config for one DSPy optimizer strategy -- no run.py analog.
@dataclass
class OptimizationConfig:
    """Flexible configuration for optimization strategies with DSPy-aligned defaults"""
    # Core identification
    name: str = "Default"  # [wrapper-only] label shown in prints / run summary
    description: str = "Default optimization strategy"  # [wrapper-only] human blurb for the strategy
    optimizer_type: Literal["bootstrap", "gepa", "miprov2", "simba"] = "bootstrap"  # [wrapper-only] selects DSPy teleprompter in _run_strategy

    # Common parameters - aligned with DSPy defaults where applicable
    num_threads: Optional[int] = None  # [wrapper-only] DSPy parallelism knob; None => optimizer default
    max_labeled_demos: Optional[int] = None  # [wrapper-only] few-shot labeled demo cap; filled in __post_init__
    max_bootstrapped_demos: Optional[int] = None  # [wrapper-only] few-shot bootstrapped demo cap; filled in __post_init__

    # Flexible optimizer-specific parameters
    optimizer_params: Dict[str, Any] = field(default_factory=dict)  # [wrapper-only] per-optimizer kwargs bag (populated from defaults)

    def __post_init__(self):  # [wrapper-only] backfill defaults per optimizer_type after dataclass init
        if not self.optimizer_params:  # [wrapper-only] no explicit params => use built-in defaults
            self.optimizer_params = self._get_default_params()  # [wrapper-only] optimizer-specific default kwargs
        if self.max_labeled_demos is None:  # [wrapper-only] unset => apply DSPy-aligned default
            self.max_labeled_demos = self._get_dspy_labeled_demos_default()  # [wrapper-only]
        if self.max_bootstrapped_demos is None:  # [wrapper-only] unset => apply DSPy-aligned default
            self.max_bootstrapped_demos = self._get_dspy_bootstrapped_demos_default()  # [wrapper-only]

    def _get_dspy_labeled_demos_default(self) -> int:  # [wrapper-only] per-optimizer labeled-demo default
        defaults = {  # [wrapper-only] bootstrap wants more labeled demos than the others
            "bootstrap": 16,
            "gepa": 4,
            "miprov2": 4,
            "simba": 4,
        }
        return defaults.get(self.optimizer_type, 4)  # [wrapper-only] fall back to 4 for unknown types

    def _get_dspy_bootstrapped_demos_default(self) -> int:  # [wrapper-only] per-optimizer bootstrapped-demo default
        defaults = {  # [wrapper-only] all optimizers default to 4 bootstrapped demos
            "bootstrap": 4,
            "gepa": 4,
            "miprov2": 4,
            "simba": 4,
        }
        return defaults.get(self.optimizer_type, 4)  # [wrapper-only] fall back to 4 for unknown types

    def _get_default_params(self) -> Dict[str, Any]:  # [wrapper-only] optimizer-specific kwargs passed to the DSPy teleprompter
        defaults = {  # [wrapper-only] mirror each DSPy teleprompter's own defaults
            "bootstrap": {  # [wrapper-only] BootstrapFewShotWithRandomSearch knobs
                "num_candidate_programs": 16,
                "max_rounds": 1,
            },
            "gepa": {},  # [wrapper-only] GEPA kwargs supplied at call site (auto/reflection_lm), not here
            "miprov2": {  # [wrapper-only] MIPROv2 budget preset
                "auto": "light",
            },
            "simba": {  # [wrapper-only] SIMBA step/batch/candidate knobs
                "max_steps": 8,
                "bsize": 32,
                "num_candidates": 6,
            },
        }
        return defaults.get(self.optimizer_type, {})  # [wrapper-only] empty dict for unknown types

    def get_param(self, key: str, default=None):  # [wrapper-only] convenience read of optimizer_params
        return self.optimizer_params.get(key, default)  # [wrapper-only]

    def set_param(self, key: str, value: Any):  # [wrapper-only] fluent single-key setter
        self.optimizer_params[key] = value  # [wrapper-only] mutate the kwargs bag
        return self  # [wrapper-only] return self for chaining

    def update_params(self, **kwargs):  # [wrapper-only] fluent bulk setter
        self.optimizer_params.update(kwargs)  # [wrapper-only] merge kwargs into the bag
        return self  # [wrapper-only] return self for chaining


# [wrapper-only] runs optimizer.compile() with the M1..M8 feedback metric to optimize the
# 4 prompt instructions -- no run.py analog (run.py runs each FC once, never optimizes).
class OptimizationRunner:
    """Configurable optimization runner supporting Bootstrap, MIPROv2, GEPA, and SimBA strategies"""

    def __init__(self,
                 metric,
                 student_model,
                 train_set,
                 val_set,
                 evaluate_fn,
                 metric_with_feedback=None,
                 reflection_lm=None):
        self.metric = metric  # [wrapper-only] scalar M1..M8 metric (evaluation/mbd_metric.py) for scoring/optimizers; run.py computes no score
        self.metric_with_feedback = metric_with_feedback or metric  # [wrapper-only] GEPA needs (score, textual feedback); fall back to scalar metric
        self.student_model = student_model  # [wrapper-only] the DSPy program being optimized (extract_mbd_full.MBDFullPipelineModule)
        self.train_set = train_set  # [wrapper-only] dspy.Examples from utils/mbd_data.py (Phase 1/2 precomputed); used by optimizers to compile
        self.val_set = val_set  # [wrapper-only] held-out Examples for baseline + candidate scoring
        self.evaluate_fn = evaluate_fn  # [wrapper-only] dspy.Evaluate instance used for baseline/candidate evaluation
        self.reflection_lm = reflection_lm  # [wrapper-only] LM GEPA uses to reflectively rewrite the 4 prompt instructions

        self.default_strategies = []  # [wrapper-only] used when run_optimization is called with no explicit strategies
        self.run_summary = {}  # [wrapper-only] JSON-serializable record of the whole run (timings/scores); no run.py analog

    def run_optimization(self, strategies: Optional[List[OptimizationConfig]] = None) -> tuple:
        """Run optimization with configurable strategies"""
        if strategies is None:  # [wrapper-only] caller passed nothing => use preconfigured defaults
            strategies = self.default_strategies  # [wrapper-only]

        print("Starting Optimization...")  # [wrapper-only] progress print (run.py has no optimization phase)

        run_started_at = datetime.now().isoformat(timespec="seconds")  # [wrapper-only] run-level start stamp for summary
        run_t0 = time.time()  # [wrapper-only] monotonic start for total-duration measurement

        baseline_started_at = datetime.now().isoformat(timespec="seconds")  # [wrapper-only] baseline start stamp
        baseline_t0 = time.time()  # [wrapper-only] baseline timing start
        # [wrapper-only] GEPA measures its OWN iteration-0 seed baseline under the SAME conditions as its best
        # score. A separate wrapper baseline pass here would be a redundant, differently-drawn eval (val n small
        # + cache off → ~10-pt swings → fake "improvement"). So for an all-GEPA run, DEFER: adopt GEPA's internal
        # baseline after compile(). Non-GEPA optimizers still get exactly one pre-eval here.
        _defer_baseline = bool(strategies) and all(getattr(c, "optimizer_type", "") == "gepa" for c in strategies)  # [wrapper-only]
        baseline_score = None if _defer_baseline else self._evaluate_baseline()  # [wrapper-only] ONE baseline only; GEPA supplies its own
        if _defer_baseline:  # [wrapper-only]
            print("Baseline: using GEPA's own iteration-0 seed eval (no redundant wrapper pass).")  # [wrapper-only]
        baseline_completed_at = datetime.now().isoformat(timespec="seconds")  # [wrapper-only] baseline end stamp
        baseline_duration_seconds = round(time.time() - baseline_t0, 1)  # [wrapper-only] baseline elapsed seconds

        best_extractor = None  # [wrapper-only] best optimized program found so far (None => keep baseline)
        best_score = baseline_score if baseline_score is not None else float("-inf")  # [wrapper-only] GEPA fills the baseline after compile; -inf so its first result is adopted
        best_strategy_name = "Baseline"  # [wrapper-only] name of the currently-winning strategy
        strategy_summaries = []  # [wrapper-only] accumulates per-strategy result dicts

        for i, config in enumerate(strategies, 1):  # [wrapper-only] run each strategy in order (1-indexed for prints)
            print(f"\nStrategy {i}: {config.name}...")  # [wrapper-only] progress print
            print(f"   {config.description}")  # [wrapper-only] progress print

            strategy_started_at = datetime.now().isoformat(timespec="seconds")  # [wrapper-only] per-strategy start stamp
            strategy_t0 = time.time()  # [wrapper-only] per-strategy timing start
            try:  # [wrapper-only] isolate one strategy's failure so it can be summarized/re-raised
                extractor, score = self._run_strategy(config)  # [wrapper-only] compile() this strategy -> (optimized program, val score)
            except Exception as exc:  # [wrapper-only] any optimizer error => record + re-raise as StrategyExecutionError
                strategy_completed_at = datetime.now().isoformat(timespec="seconds")  # [wrapper-only] failure end stamp
                strategy_duration_seconds = round(time.time() - strategy_t0, 1)  # [wrapper-only] elapsed until failure
                failed_summary = {  # [wrapper-only] structured record of the failed strategy for run_summary
                    "index": i,
                    "name": config.name,
                    "optimizer_type": config.optimizer_type,
                    "description": config.description,
                    "started_at": strategy_started_at,
                    "completed_at": strategy_completed_at,
                    "duration_seconds": strategy_duration_seconds,
                    "status": "failed",
                    "error": str(exc),
                    "num_threads": config.num_threads,
                    "max_labeled_demos": config.max_labeled_demos,
                    "max_bootstrapped_demos": config.max_bootstrapped_demos,
                    "optimizer_params": dict(config.optimizer_params or {}),
                }
                strategy_summaries.append(failed_summary)  # [wrapper-only] add failure to the per-strategy list
                self.run_summary = {  # [wrapper-only] persist a "failed" run summary before raising
                    "started_at": run_started_at,
                    "completed_at": strategy_completed_at,
                    "total_duration_seconds": round(time.time() - run_t0, 1),
                    "baseline": {
                        "started_at": baseline_started_at,
                        "completed_at": baseline_completed_at,
                        "duration_seconds": baseline_duration_seconds,
                        "score": baseline_score,
                    },
                    "best_strategy_name": best_strategy_name,
                    "best_score": best_score,
                    "failed_strategy": failed_summary,
                    "strategies": strategy_summaries,
                    "status": "failed",
                }
                raise StrategyExecutionError(config.name, config.optimizer_type, exc) from exc  # [wrapper-only] surface failure with chained cause
            strategy_completed_at = datetime.now().isoformat(timespec="seconds")  # [wrapper-only] success end stamp
            strategy_duration_seconds = round(time.time() - strategy_t0, 1)  # [wrapper-only] strategy elapsed seconds
            if baseline_score is None:  # [wrapper-only] GEPA path: adopt GEPA's own iteration-0 seed eval as THE baseline
                baseline_score = getattr(self, "gepa_baseline_score", score)  # [wrapper-only] same-conditions baseline (0 improvement when seed kept)
                if best_score == float("-inf"):  # [wrapper-only] seed the gate now that we have the baseline
                    best_score = baseline_score  # [wrapper-only]
            improvement = score - baseline_score  # [wrapper-only] delta of this strategy over the baseline score (honest, apples-to-apples for GEPA)

            print(f"Strategy {i} Results:")  # [wrapper-only] progress print
            print(f"  Accuracy: {score:.1f}%")  # [wrapper-only] progress print
            print(f"  Improvement: {improvement:+.1f}%")  # [wrapper-only] progress print

            if score >= best_score:  # [wrapper-only] >= so ties adopt the later strategy as the new best
                best_extractor = extractor  # [wrapper-only] keep this optimized program
                best_score = score  # [wrapper-only] raise the running best score
                best_strategy_name = config.name  # [wrapper-only] record the winning strategy name
                print(f"  Updated best extractor (score: {score:.1f}%)")  # [wrapper-only] progress print
            else:  # [wrapper-only] strategy did not beat current best
                print(f"  Score not better than current best ({best_score:.1f}%)")  # [wrapper-only] progress print

            strategy_summaries.append({  # [wrapper-only] record a "completed" per-strategy summary
                "index": i,
                "name": config.name,
                "optimizer_type": config.optimizer_type,
                "description": config.description,
                "started_at": strategy_started_at,
                "completed_at": strategy_completed_at,
                "duration_seconds": strategy_duration_seconds,
                "status": "completed",
                "score": score,
                "improvement_over_baseline": round(improvement, 4),
                "num_threads": config.num_threads,
                "max_labeled_demos": config.max_labeled_demos,
                "max_bootstrapped_demos": config.max_bootstrapped_demos,
                "optimizer_params": dict(config.optimizer_params or {}),
            })

        self._print_final_results(best_strategy_name, baseline_score, best_score)  # [wrapper-only] print baseline-vs-best comparison
        self.best_strategy_name = best_strategy_name  # [wrapper-only] expose winner name to caller
        run_completed_at = datetime.now().isoformat(timespec="seconds")  # [wrapper-only] run-level end stamp
        total_duration_seconds = round(time.time() - run_t0, 1)  # [wrapper-only] whole-run elapsed seconds
        for strategy_summary in strategy_summaries:  # [wrapper-only] flag which summary corresponds to the overall best
            strategy_summary["is_best"] = strategy_summary["name"] == best_strategy_name  # [wrapper-only] mark best strategy
        self.run_summary = {  # [wrapper-only] final "completed" run summary (timings + best score)
            "started_at": run_started_at,
            "completed_at": run_completed_at,
            "total_duration_seconds": total_duration_seconds,
            "status": "completed",
            "baseline": {
                "started_at": baseline_started_at,
                "completed_at": baseline_completed_at,
                "duration_seconds": baseline_duration_seconds,
                "score": baseline_score,
            },
            "best_strategy_name": best_strategy_name,
            "best_score": best_score,
            "strategies": strategy_summaries,
        }
        return best_extractor, best_score  # [wrapper-only] hand the winning optimized program + its score back to the caller (run.py returns no program/score)

    def _evaluate_baseline(self) -> float:
        """Evaluate baseline model performance"""
        print("Evaluating baseline model...")  # [wrapper-only] progress print; run.py has no baseline-scoring step
        self.evaluate_fn.devset = self.val_set  # [wrapper-only] point dspy.Evaluate at the val_set before scoring
        baseline_result = self.evaluate_fn(self.student_model)  # [wrapper-only] run the UNoptimized program (run.py-faithful seed prompts) through the M1..M8 metric
        baseline_score = baseline_result.score  # [wrapper-only] extract the 0-100 percentage score
        print(f"Baseline accuracy: {baseline_score:.1f}%")  # [wrapper-only] progress print
        return baseline_score  # [wrapper-only] baseline used as the initial best_score gate in run_optimization

    def _run_strategy(self, config: OptimizationConfig) -> tuple:
        """Run optimization with given configuration"""
        # [wrapper-only] whole method has no run.py analog: run.py never instantiates a DSPy
        # teleprompter nor calls compile(); it runs each FC once with fixed prompt templates.
        if config.optimizer_type == "bootstrap":  # [wrapper-only] branch: build a BootstrapFewShotWithRandomSearch optimizer
            bootstrap_kwargs = {  # [wrapper-only] assemble ctor kwargs from the config
                "metric": self.metric,  # [wrapper-only] scalar M1..M8 metric drives demo selection
                "max_labeled_demos": config.max_labeled_demos,  # [wrapper-only] few-shot labeled-demo cap
                "max_bootstrapped_demos": config.max_bootstrapped_demos,  # [wrapper-only] few-shot bootstrapped-demo cap
                "num_candidate_programs": config.get_param("num_candidate_programs", 16),  # [wrapper-only] random-search candidate count
                "max_rounds": config.get_param("max_rounds", 1),  # [wrapper-only] bootstrapping rounds
            }
            if config.num_threads is not None:  # [wrapper-only] only pass num_threads when the config set it
                bootstrap_kwargs["num_threads"] = config.num_threads  # [wrapper-only] DSPy parallelism knob
            if config.get_param("teacher_lm") is not None:  # teacher LM override for demo generation (else student LM)
                bootstrap_kwargs["teacher_settings"] = {"lm": config.get_param("teacher_lm")}
            for param_name, param_value in config.optimizer_params.items():  # [wrapper-only] fold in any extra non-None optimizer_params
                if param_name in ("teacher_lm", "prompt_lm", "reflection_lm"):  # aux LM objects handled explicitly; not ctor kwargs
                    continue
                if param_name not in bootstrap_kwargs and param_value is not None:  # [wrapper-only] don't clobber explicit kwargs / skip Nones
                    bootstrap_kwargs[param_name] = param_value  # [wrapper-only] passthrough extra kwarg
            bootstrap_kwargs = self._filter_supported_kwargs(  # [wrapper-only] drop kwargs the installed DSPy version no longer accepts
                BootstrapFewShotWithRandomSearch,
                bootstrap_kwargs,
                optimizer_name=config.name,
            )
            optimizer = BootstrapFewShotWithRandomSearch(**bootstrap_kwargs)  # [wrapper-only] instantiate the DSPy teleprompter

        elif config.optimizer_type == "gepa":  # [wrapper-only] branch: build the GEPA reflective prompt optimizer
            reflection_lm = config.get_param("reflection_lm") or self.reflection_lm  # [wrapper-only] LM GEPA uses to rewrite the 4 mbd.py __doc__ seeds
            gepa_kwargs = {  # [wrapper-only] assemble GEPA ctor kwargs
                "metric": self.metric_with_feedback,  # [wrapper-only] GEPA needs (score, textual feedback) not just a scalar
                "track_stats": config.get_param("track_stats", False),  # [wrapper-only] enables detailed_results used later for the in-pass best score
                "use_merge": config.get_param("use_merge", True),  # [wrapper-only] whether GEPA merges candidate programs
            }
            # Budget: GEPA requires EXACTLY ONE of {auto, max_metric_calls, max_full_evals}.
            # An explicit call budget (for "lighter than light") takes precedence over auto.
            _mmc = config.get_param("max_metric_calls")  # [wrapper-only] explicit metric-call budget (optional)
            _mfe = config.get_param("max_full_evals")  # [wrapper-only] explicit full-eval budget (optional)
            if _mmc is not None:  # [wrapper-only] prefer an explicit metric-call budget when given
                gepa_kwargs["max_metric_calls"] = _mmc  # [wrapper-only]
            elif _mfe is not None:  # [wrapper-only] else prefer an explicit full-eval budget
                gepa_kwargs["max_full_evals"] = _mfe  # [wrapper-only]
            else:  # [wrapper-only] fall back to the "auto" budget preset
                gepa_kwargs["auto"] = config.get_param("auto")  # [wrapper-only]
            if config.num_threads is not None:  # [wrapper-only] only pass num_threads when configured
                gepa_kwargs["num_threads"] = config.num_threads  # [wrapper-only] DSPy parallelism knob
            if reflection_lm is not None:  # [wrapper-only] only pass reflection_lm when one is available
                gepa_kwargs["reflection_lm"] = reflection_lm  # [wrapper-only]
            # Optional passthrough params
            # [wrapper-only] copy each optional GEPA knob from the config only when explicitly set
            for _key in ("log_dir", "seed", "max_merge_invocations",
                         "reflection_minibatch_size", "skip_perfect_score",
                         "candidate_selection_strategy", "failure_score",
                         "perfect_score", "track_best_outputs"):
                _val = config.get_param(_key)  # [wrapper-only] read the optional knob
                if _val is not None:  # [wrapper-only] skip unset knobs so GEPA uses its own defaults
                    gepa_kwargs[_key] = _val  # [wrapper-only] passthrough
            optimizer = GEPA(**gepa_kwargs)  # [wrapper-only] instantiate the GEPA teleprompter

        elif config.optimizer_type == "miprov2":  # [wrapper-only] branch: delegate MIPROv2 construction to a helper
            optimizer = self._create_miprov2_optimizer(config)  # [wrapper-only] build the MIPROv2 teleprompter

        elif config.optimizer_type == "simba":  # [wrapper-only] branch: build the SIMBA optimizer
            simba_kwargs = {  # [wrapper-only] assemble SIMBA ctor kwargs
                "metric": self.metric,  # [wrapper-only] scalar M1..M8 metric
                "max_steps": config.get_param("max_steps", 8),  # [wrapper-only] SIMBA optimization steps
                "max_demos": config.max_labeled_demos,  # [wrapper-only] demo cap (SIMBA reuses the labeled-demo config)
                "bsize": config.get_param("bsize", 32),  # [wrapper-only] minibatch size
                "num_candidates": config.get_param("num_candidates", 6),  # [wrapper-only] candidates per step
            }
            if config.num_threads is not None:  # [wrapper-only] only pass num_threads when configured
                simba_kwargs["num_threads"] = config.num_threads  # [wrapper-only] DSPy parallelism knob
            for param_name, param_value in config.optimizer_params.items():  # [wrapper-only] fold in extra non-None optimizer_params
                if param_name not in simba_kwargs and param_value is not None:  # [wrapper-only] don't clobber explicit kwargs / skip Nones
                    simba_kwargs[param_name] = param_value  # [wrapper-only] passthrough extra kwarg
            optimizer = SIMBA(**simba_kwargs)  # [wrapper-only] instantiate the SIMBA teleprompter

        else:  # [wrapper-only] unknown optimizer_type => programmer error
            raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")  # [wrapper-only] fail fast on a bad config

        print(f"Compiling {config.name}...")  # [wrapper-only] progress print (run.py never "compiles" a program)
        compile_kwargs = {  # [wrapper-only] args for the teleprompter's compile()
            "student": self.student_model,  # [wrapper-only] the program to optimize (extract_mbd_full.MBDFullPipelineModule)
            "trainset": self.train_set,  # [wrapper-only] Phase-1/2-precomputed dspy.Examples the optimizer learns from
        }
        # SIMBA works entirely on trainset; it does not accept valset
        if config.optimizer_type != "simba":  # [wrapper-only] SIMBA rejects valset; all others use it for candidate scoring
            compile_kwargs["valset"] = self.val_set  # [wrapper-only] held-out Examples for candidate evaluation

        # [wrapper-only] THE optimization run: compile() drives the LM to rewrite the 4 prompt
        # instruction seeds. NOTE (see project memory): compile() is itself launched inside an
        # isolated subprocess (optimizer_subprocess.py) by the caller because nested DSPy thread
        # pools zero-scored in-process; this method only assembles the optimizer + kwargs.
        optimized_extractor = optimizer.compile(**compile_kwargs)  # [wrapper-only] returns the best optimized program

        # Prefer the optimizer's OWN best valset score (already measured during compile())
        # over a fresh re-evaluation. Re-evaluating here is redundant work AND, with caching
        # off + a non-deterministic pipeline, an independent noisy draw that can discard a
        # genuine improvement (GEPA measured its winner at 60.7% in-pass; a re-eval here drew
        # 52.9% and the gate in run_optimization() then rejected it). GEPA attaches
        # detailed_results when track_stats=True; its val_aggregate_scores are fractions
        # (0-1), so scale by 100 to match dspy.Evaluate's score (a 0-100 percentage:
        # round(100 * ncorrect / ntotal, 2)). Fall back to a single eval only when no internal
        # score exists, so other optimizers / configs without one keep their prior behavior.
        print(f"Evaluating {config.name}...")  # [wrapper-only] progress print
        gepa_res = getattr(optimized_extractor, "detailed_results", None)  # [wrapper-only] GEPA attaches detailed_results when track_stats=True; None otherwise
        if gepa_res is not None and getattr(gepa_res, "val_aggregate_scores", None):  # [wrapper-only] use GEPA's own in-pass best val score when available
            _bi = getattr(gepa_res, "best_idx", 0)  # [wrapper-only] index of GEPA's best program (0 == seed retained, no predictor improved)
            score = gepa_res.val_aggregate_scores[_bi] * 100.0  # [wrapper-only] fractions (0-1) → 0-100 to match dspy.Evaluate's percentage
            # [wrapper-only] GEPA's iteration-0 seed eval IS the baseline (index 0), measured under the SAME
            # conditions as `score` — so improvement is apples-to-apples (and exactly 0 when the seed is kept).
            # This lets run_optimization DROP its own redundant, differently-drawn baseline pass for GEPA.
            self.gepa_baseline_score = gepa_res.val_aggregate_scores[0] * 100.0  # [wrapper-only] THE baseline (GEPA's own seed eval)
            self.gepa_seed_retained = (_bi == 0)  # [wrapper-only] True => GEPA found no candidate that beat the seed
        else:  # [wrapper-only] no internal score (non-GEPA, or track_stats off)
            score = self.evaluate_fn(optimized_extractor, devset=self.val_set).score  # [wrapper-only] fresh single eval on the val_set
        return optimized_extractor, score  # [wrapper-only] hand (optimized program, val score) back to run_optimization

    @staticmethod  # [wrapper-only] pure helper; no run.py analog
    def _filter_supported_kwargs(optimizer_cls, kwargs: Dict[str, Any], optimizer_name: str) -> Dict[str, Any]:
        """Drop legacy optimizer kwargs that the installed DSPy version no longer accepts."""
        allowed = set(inspect.signature(optimizer_cls.__init__).parameters)  # [wrapper-only] introspect the optimizer ctor's accepted params
        filtered = {key: value for key, value in kwargs.items() if key in allowed}  # [wrapper-only] keep only ctor-accepted kwargs
        dropped = sorted(key for key in kwargs if key not in filtered)  # [wrapper-only] names of the discarded legacy kwargs
        if dropped:  # [wrapper-only] warn when anything was dropped so config drift is visible
            print(  # [wrapper-only] progress print listing dropped kwargs
                f"Skipping unsupported {optimizer_cls.__name__} kwargs for {optimizer_name}: "
                f"{', '.join(dropped)}"
            )
        return filtered  # [wrapper-only] version-safe kwargs for the ctor

    def _create_miprov2_optimizer(self, config: OptimizationConfig):  # [wrapper-only] MIPROv2 ctor helper; no run.py analog
        """Create MIPROv2 optimizer with DSPy-aligned parameter handling"""
        miprov2_params = {  # [wrapper-only] assemble MIPROv2 ctor kwargs
            "metric": self.metric,  # [wrapper-only] scalar M1..M8 metric
            "auto": config.get_param("auto", "light"),  # [wrapper-only] MIPROv2 budget preset (default "light")
            "max_labeled_demos": config.max_labeled_demos,  # [wrapper-only] few-shot labeled-demo cap
            "max_bootstrapped_demos": config.max_bootstrapped_demos,  # [wrapper-only] few-shot bootstrapped-demo cap
        }
        if config.num_threads is not None:  # [wrapper-only] only pass num_threads when configured
            miprov2_params["num_threads"] = config.num_threads  # [wrapper-only] DSPy parallelism knob
        if config.get_param("teacher_lm") is not None:  # teacher LM override for demo generation (else student LM)
            miprov2_params["teacher_settings"] = {"lm": config.get_param("teacher_lm")}
        if config.get_param("prompt_lm") is not None:  # proposer LM override for instruction generation (else student LM)
            miprov2_params["prompt_model"] = config.get_param("prompt_lm")
        for param_name, param_value in config.optimizer_params.items():  # [wrapper-only] fold in extra non-None optimizer_params
            if param_name in ("teacher_lm", "prompt_lm", "reflection_lm"):  # aux LM objects handled explicitly; not ctor kwargs
                continue
            if param_name not in miprov2_params and param_value is not None:  # [wrapper-only] don't clobber explicit kwargs / skip Nones
                miprov2_params[param_name] = param_value  # [wrapper-only] passthrough extra kwarg
        print(f"MIPROv2 parameters: {miprov2_params}")  # [wrapper-only] progress print of the resolved kwargs
        return MIPROv2(**miprov2_params)  # [wrapper-only] instantiate the MIPROv2 teleprompter

    def _print_final_results(self, best_strategy: str, baseline_score: float, best_score: float):  # [wrapper-only] summary print; no run.py analog
        # [wrapper-only] print the baseline-vs-best comparison (run.py computes no scores to compare)
        print(f"\nOptimization Complete!")
        print(f"Best Strategy: {best_strategy}")
        print(f"Baseline Accuracy: {baseline_score:.1f}%")
        print(f"Best Accuracy: {best_score:.1f}%")
        print(f"Total Improvement: {best_score - baseline_score:+.1f}%")


# [wrapper-only] factory for an arbitrary-optimizer strategy config -- no run.py analog.
def create_custom_strategy(name: str, description: str, optimizer_type: str = "bootstrap",
                           num_threads: int = 4, max_labeled_demos: int = 4,
                           max_bootstrapped_demos: int = 4, **optimizer_params) -> OptimizationConfig:
    """Helper function to create custom optimization strategies with flexible parameters"""
    # [wrapper-only] pack the caller's args into an OptimizationConfig (**optimizer_params as-is)
    return OptimizationConfig(
        name=name,
        description=description,
        optimizer_type=optimizer_type,
        num_threads=num_threads,
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,
        optimizer_params=optimizer_params,
    )


# [wrapper-only] factory for a GEPA prompt-optimization strategy config -- no run.py analog.
def create_gepa_strategy(name: str, description: str, auto=None,
                         num_threads=None, max_labeled_demos=None, max_bootstrapped_demos=None,
                         **gepa_params) -> OptimizationConfig:
    """Helper function to create GEPA optimization strategies."""
    optimizer_params = {}  # [wrapper-only] start an empty GEPA kwargs bag
    if auto is not None:  # [wrapper-only] only set the budget preset when the caller gave one
        optimizer_params["auto"] = auto  # [wrapper-only] GEPA budget preset
    optimizer_params.update(gepa_params)  # [wrapper-only] merge in any extra GEPA kwargs
    # [wrapper-only] pack into a GEPA-typed OptimizationConfig
    return OptimizationConfig(
        name=name,
        description=description,
        optimizer_type="gepa",
        num_threads=num_threads,
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,
        optimizer_params=optimizer_params,
    )


# [wrapper-only] factory for a Bootstrap few-shot strategy config -- no run.py analog.
def create_bootstrap_strategy(name: str, description: str, max_labeled_demos=None,
                            max_bootstrapped_demos=None, num_threads=None, num_candidate_traces=None,
                            **bootstrap_params) -> OptimizationConfig:
    """
    Helper function to create Bootstrap optimization strategies with flexible interface.
    """
    # Add any additional Bootstrap-specific parameters
    optimizer_params = {}  # [wrapper-only] start an empty Bootstrap kwargs bag
    optimizer_params.update(bootstrap_params)  # [wrapper-only] merge in any extra Bootstrap kwargs

    if num_candidate_traces is not None:  # [wrapper-only] legacy param: only stash it when explicitly passed
        # Legacy compatibility only. Older repo configs still pass this,
        # but current DSPy BootstrapFewShotWithRandomSearch no longer accepts it.
        optimizer_params["num_candidate_traces"] = num_candidate_traces  # [wrapper-only] kept for old configs; _filter_supported_kwargs drops it later

    # [wrapper-only] pack into a bootstrap-typed OptimizationConfig
    return OptimizationConfig(
        name=name,
        description=description,
        optimizer_type="bootstrap",
        num_threads=num_threads,
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,
        optimizer_params=optimizer_params,
    )


# [wrapper-only] factory for a MIPROv2 strategy config -- no run.py analog.
def create_miprov2_strategy(name: str, description: str, auto="light",
                             num_threads=4, max_labeled_demos=4, max_bootstrapped_demos=4,
                             **miprov2_params) -> OptimizationConfig:
    """Helper function to create MIPROv2 optimization strategies."""
    optimizer_params = {"auto": auto}  # [wrapper-only] seed the kwargs bag with the budget preset
    optimizer_params.update(miprov2_params)  # [wrapper-only] merge in any extra MIPROv2 kwargs
    # [wrapper-only] pack into a miprov2-typed OptimizationConfig
    return OptimizationConfig(
        name=name,
        description=description,
        optimizer_type="miprov2",
        num_threads=num_threads,
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,
        optimizer_params=optimizer_params,
    )


# [wrapper-only] factory for a SIMBA strategy config -- no run.py analog.
def create_simba_strategy(name: str, description: str, max_steps=8,
                          max_labeled_demos=4, max_bootstrapped_demos=None,
                          num_threads=None, bsize=32, num_candidates=6,
                          **simba_params) -> OptimizationConfig:
    """Helper function to create SIMBA optimization strategies."""
    optimizer_params = {  # [wrapper-only] seed the kwargs bag with SIMBA's step/batch/candidate knobs
        "max_steps": max_steps,
        "bsize": bsize,
        "num_candidates": num_candidates,
    }
    optimizer_params.update(simba_params)  # [wrapper-only] merge in any extra SIMBA kwargs
    # [wrapper-only] pack into a simba-typed OptimizationConfig
    return OptimizationConfig(
        name=name,
        description=description,
        optimizer_type="simba",
        num_threads=num_threads,
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,
        optimizer_params=optimizer_params,
    )
