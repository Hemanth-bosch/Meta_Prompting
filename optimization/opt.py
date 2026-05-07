from dataclasses import dataclass, field
from datetime import datetime
import inspect
import time
from typing import List, Optional, Literal, Dict, Any
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, MIPROv2, SIMBA
from dspy import GEPA


class StrategyExecutionError(RuntimeError):
    """Optimization strategy failed after the underlying optimizer exhausted its retries."""

    def __init__(self, strategy_name: str, optimizer_type: str, cause: Exception):
        self.strategy_name = strategy_name
        self.optimizer_type = optimizer_type
        self.cause = cause
        super().__init__(
            f"Strategy '{strategy_name}' ({optimizer_type}) failed: {cause}"
        )


@dataclass
class OptimizationConfig:
    """Flexible configuration for optimization strategies with DSPy-aligned defaults"""
    # Core identification
    name: str = "Default"
    description: str = "Default optimization strategy"
    optimizer_type: Literal["bootstrap", "gepa", "miprov2", "simba"] = "bootstrap"

    # Common parameters - aligned with DSPy defaults where applicable
    num_threads: Optional[int] = None
    max_labeled_demos: Optional[int] = None
    max_bootstrapped_demos: Optional[int] = None

    # Flexible optimizer-specific parameters
    optimizer_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.optimizer_params:
            self.optimizer_params = self._get_default_params()
        if self.max_labeled_demos is None:
            self.max_labeled_demos = self._get_dspy_labeled_demos_default()
        if self.max_bootstrapped_demos is None:
            self.max_bootstrapped_demos = self._get_dspy_bootstrapped_demos_default()

    def _get_dspy_labeled_demos_default(self) -> int:
        defaults = {
            "bootstrap": 16,
            "gepa": 4,
            "miprov2": 4,
            "simba": 4,
        }
        return defaults.get(self.optimizer_type, 4)

    def _get_dspy_bootstrapped_demos_default(self) -> int:
        defaults = {
            "bootstrap": 4,
            "gepa": 4,
            "miprov2": 4,
            "simba": 4,
        }
        return defaults.get(self.optimizer_type, 4)

    def _get_default_params(self) -> Dict[str, Any]:
        defaults = {
            "bootstrap": {
                "num_candidate_programs": 16,
                "max_rounds": 1,
            },
            "gepa": {},
            "miprov2": {
                "auto": "light",
            },
            "simba": {
                "max_steps": 8,
                "bsize": 32,
                "num_candidates": 6,
            },
        }
        return defaults.get(self.optimizer_type, {})

    def get_param(self, key: str, default=None):
        return self.optimizer_params.get(key, default)

    def set_param(self, key: str, value: Any):
        self.optimizer_params[key] = value
        return self

    def update_params(self, **kwargs):
        self.optimizer_params.update(kwargs)
        return self


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
        self.metric = metric
        self.metric_with_feedback = metric_with_feedback or metric
        self.student_model = student_model
        self.train_set = train_set
        self.val_set = val_set
        self.evaluate_fn = evaluate_fn
        self.reflection_lm = reflection_lm

        self.default_strategies = []
        self.run_summary = {}

    def run_optimization(self, strategies: Optional[List[OptimizationConfig]] = None) -> tuple:
        """Run optimization with configurable strategies"""
        if strategies is None:
            strategies = self.default_strategies

        print("Starting Optimization...")

        run_started_at = datetime.now().isoformat(timespec="seconds")
        run_t0 = time.time()

        baseline_started_at = datetime.now().isoformat(timespec="seconds")
        baseline_t0 = time.time()
        baseline_score = self._evaluate_baseline()
        baseline_completed_at = datetime.now().isoformat(timespec="seconds")
        baseline_duration_seconds = round(time.time() - baseline_t0, 1)

        best_extractor = None
        best_score = baseline_score
        best_strategy_name = "Baseline"
        strategy_summaries = []

        for i, config in enumerate(strategies, 1):
            print(f"\nStrategy {i}: {config.name}...")
            print(f"   {config.description}")

            strategy_started_at = datetime.now().isoformat(timespec="seconds")
            strategy_t0 = time.time()
            try:
                extractor, score = self._run_strategy(config)
            except Exception as exc:
                strategy_completed_at = datetime.now().isoformat(timespec="seconds")
                strategy_duration_seconds = round(time.time() - strategy_t0, 1)
                failed_summary = {
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
                strategy_summaries.append(failed_summary)
                self.run_summary = {
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
                raise StrategyExecutionError(config.name, config.optimizer_type, exc) from exc
            strategy_completed_at = datetime.now().isoformat(timespec="seconds")
            strategy_duration_seconds = round(time.time() - strategy_t0, 1)
            improvement = score - baseline_score

            print(f"Strategy {i} Results:")
            print(f"  Accuracy: {score:.1f}%")
            print(f"  Improvement: {improvement:+.1f}%")

            if score >= best_score:
                best_extractor = extractor
                best_score = score
                best_strategy_name = config.name
                print(f"  Updated best extractor (score: {score:.1f}%)")
            else:
                print(f"  Score not better than current best ({best_score:.1f}%)")

            strategy_summaries.append({
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

        self._print_final_results(best_strategy_name, baseline_score, best_score)
        self.best_strategy_name = best_strategy_name
        run_completed_at = datetime.now().isoformat(timespec="seconds")
        total_duration_seconds = round(time.time() - run_t0, 1)
        for strategy_summary in strategy_summaries:
            strategy_summary["is_best"] = strategy_summary["name"] == best_strategy_name
        self.run_summary = {
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
        return best_extractor, best_score

    def _evaluate_baseline(self) -> float:
        """Evaluate baseline model performance"""
        print("Evaluating baseline model...")
        self.evaluate_fn.devset = self.val_set
        baseline_result = self.evaluate_fn(self.student_model)
        baseline_score = baseline_result.score
        print(f"Baseline accuracy: {baseline_score:.1f}%")
        return baseline_score

    def _run_strategy(self, config: OptimizationConfig) -> tuple:
        """Run optimization with given configuration"""
        if config.optimizer_type == "bootstrap":
            bootstrap_kwargs = {
                "metric": self.metric,
                "max_labeled_demos": config.max_labeled_demos,
                "max_bootstrapped_demos": config.max_bootstrapped_demos,
                "num_candidate_programs": config.get_param("num_candidate_programs", 16),
                "max_rounds": config.get_param("max_rounds", 1),
            }
            if config.num_threads is not None:
                bootstrap_kwargs["num_threads"] = config.num_threads
            for param_name, param_value in config.optimizer_params.items():
                if param_name not in bootstrap_kwargs and param_value is not None:
                    bootstrap_kwargs[param_name] = param_value
            bootstrap_kwargs = self._filter_supported_kwargs(
                BootstrapFewShotWithRandomSearch,
                bootstrap_kwargs,
                optimizer_name=config.name,
            )
            optimizer = BootstrapFewShotWithRandomSearch(**bootstrap_kwargs)

        elif config.optimizer_type == "gepa":
            reflection_lm = config.get_param("reflection_lm") or self.reflection_lm
            gepa_kwargs = {
                "metric": self.metric_with_feedback,
                "auto": config.get_param("auto"),
                "track_stats": config.get_param("track_stats", False),
                "use_merge": config.get_param("use_merge", True),
            }
            if config.num_threads is not None:
                gepa_kwargs["num_threads"] = config.num_threads
            if reflection_lm is not None:
                gepa_kwargs["reflection_lm"] = reflection_lm
            optimizer = GEPA(**gepa_kwargs)

        elif config.optimizer_type == "miprov2":
            optimizer = self._create_miprov2_optimizer(config)

        elif config.optimizer_type == "simba":
            simba_kwargs = {
                "metric": self.metric,
                "max_steps": config.get_param("max_steps", 8),
                "max_demos": config.max_labeled_demos,
                "bsize": config.get_param("bsize", 32),
                "num_candidates": config.get_param("num_candidates", 6),
            }
            if config.num_threads is not None:
                simba_kwargs["num_threads"] = config.num_threads
            for param_name, param_value in config.optimizer_params.items():
                if param_name not in simba_kwargs and param_value is not None:
                    simba_kwargs[param_name] = param_value
            optimizer = SIMBA(**simba_kwargs)

        else:
            raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")

        print(f"Compiling {config.name}...")
        compile_kwargs = {
            "student": self.student_model,
            "trainset": self.train_set,
        }
        # SIMBA works entirely on trainset; it does not accept valset
        if config.optimizer_type != "simba":
            compile_kwargs["valset"] = self.val_set
        optimized_extractor = optimizer.compile(**compile_kwargs)

        print(f"Evaluating {config.name}...")
        result = self.evaluate_fn(optimized_extractor, devset=self.val_set)
        return optimized_extractor, result.score

    @staticmethod
    def _filter_supported_kwargs(optimizer_cls, kwargs: Dict[str, Any], optimizer_name: str) -> Dict[str, Any]:
        """Drop legacy optimizer kwargs that the installed DSPy version no longer accepts."""
        allowed = set(inspect.signature(optimizer_cls.__init__).parameters)
        filtered = {key: value for key, value in kwargs.items() if key in allowed}
        dropped = sorted(key for key in kwargs if key not in filtered)
        if dropped:
            print(
                f"Skipping unsupported {optimizer_cls.__name__} kwargs for {optimizer_name}: "
                f"{', '.join(dropped)}"
            )
        return filtered

    def _create_miprov2_optimizer(self, config: OptimizationConfig):
        """Create MIPROv2 optimizer with DSPy-aligned parameter handling"""
        miprov2_params = {
            "metric": self.metric,
            "auto": config.get_param("auto", "light"),
            "max_labeled_demos": config.max_labeled_demos,
            "max_bootstrapped_demos": config.max_bootstrapped_demos,
        }
        if config.num_threads is not None:
            miprov2_params["num_threads"] = config.num_threads
        for param_name, param_value in config.optimizer_params.items():
            if param_name not in miprov2_params and param_value is not None:
                miprov2_params[param_name] = param_value
        print(f"MIPROv2 parameters: {miprov2_params}")
        return MIPROv2(**miprov2_params)

    def _print_final_results(self, best_strategy: str, baseline_score: float, best_score: float):
        print(f"\nOptimization Complete!")
        print(f"Best Strategy: {best_strategy}")
        print(f"Baseline Accuracy: {baseline_score:.1f}%")
        print(f"Best Accuracy: {best_score:.1f}%")
        print(f"Total Improvement: {best_score - baseline_score:+.1f}%")


def create_custom_strategy(name: str, description: str, optimizer_type: str = "bootstrap",
                           num_threads: int = 4, max_labeled_demos: int = 4,
                           max_bootstrapped_demos: int = 4, **optimizer_params) -> OptimizationConfig:
    """Helper function to create custom optimization strategies with flexible parameters"""
    return OptimizationConfig(
        name=name,
        description=description,
        optimizer_type=optimizer_type,
        num_threads=num_threads,
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,
        optimizer_params=optimizer_params,
    )


def create_gepa_strategy(name: str, description: str, auto=None,
                         num_threads=None, max_labeled_demos=None, max_bootstrapped_demos=None,
                         **gepa_params) -> OptimizationConfig:
    """Helper function to create GEPA optimization strategies."""
    optimizer_params = {}
    if auto is not None:
        optimizer_params["auto"] = auto
    optimizer_params.update(gepa_params)
    return OptimizationConfig(
        name=name,
        description=description,
        optimizer_type="gepa",
        num_threads=num_threads,
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,
        optimizer_params=optimizer_params,
    )


def create_bootstrap_strategy(name: str, description: str, max_labeled_demos=None,
                            max_bootstrapped_demos=None, num_threads=None, num_candidate_traces=None,
                            **bootstrap_params) -> OptimizationConfig:
    """
    Helper function to create Bootstrap optimization strategies with flexible interface.
    """
    # Add any additional Bootstrap-specific parameters
    optimizer_params = {}
    optimizer_params.update(bootstrap_params)

    if num_candidate_traces is not None:
        # Legacy compatibility only. Older repo configs still pass this,
        # but current DSPy BootstrapFewShotWithRandomSearch no longer accepts it.
        optimizer_params["num_candidate_traces"] = num_candidate_traces
    
    return OptimizationConfig(
        name=name,
        description=description,
        optimizer_type="bootstrap",
        num_threads=num_threads,
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,
        optimizer_params=optimizer_params,
    )


def create_miprov2_strategy(name: str, description: str, auto="light",
                             num_threads=4, max_labeled_demos=4, max_bootstrapped_demos=4,
                             **miprov2_params) -> OptimizationConfig:
    """Helper function to create MIPROv2 optimization strategies."""
    optimizer_params = {"auto": auto}
    optimizer_params.update(miprov2_params)
    return OptimizationConfig(
        name=name,
        description=description,
        optimizer_type="miprov2",
        num_threads=num_threads,
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,
        optimizer_params=optimizer_params,
    )


def create_simba_strategy(name: str, description: str, max_steps=8,
                          max_labeled_demos=4, max_bootstrapped_demos=None,
                          num_threads=None, bsize=32, num_candidates=6,
                          **simba_params) -> OptimizationConfig:
    """Helper function to create SIMBA optimization strategies."""
    optimizer_params = {
        "max_steps": max_steps,
        "bsize": bsize,
        "num_candidates": num_candidates,
    }
    optimizer_params.update(simba_params)
    return OptimizationConfig(
        name=name,
        description=description,
        optimizer_type="simba",
        num_threads=num_threads,
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,
        optimizer_params=optimizer_params,
    )
