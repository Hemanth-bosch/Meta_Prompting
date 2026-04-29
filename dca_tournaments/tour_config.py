"""TournamentConfig — central configuration for the cascaded optimization pipeline.

Every hyperparameter is collected here so that ``run_tournament.py`` has a
single source of truth.  Factory class-methods provide convenient presets,
and ``from_cli_args()`` bridges the argparse namespace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config import create_lm


# ── Default bootstrap configurations (Stage 1) ──────────────────
# Each config is one BootstrapFewShotWithRandomSearch run.
# Fields: name, n (num_candidate_programs), max_rounds,
#         max_labeled_demos, max_bootstrapped_demos
_DEFAULT_BOOTSTRAP_CONFIGS: list[dict] = [
    # {"name": "n16_d3b3", "n": 16, "max_rounds": 2, "max_labeled_demos": 3, "max_bootstrapped_demos": 3},
    # {"name": "n16_d5b3", "n": 16, "max_rounds": 2, "max_labeled_demos": 5, "max_bootstrapped_demos": 3},
    # {"name": "n32_d3b3", "n": 32, "max_rounds": 2, "max_labeled_demos": 3, "max_bootstrapped_demos": 3},
    # {"name": "n32_d5b3", "n": 32, "max_rounds": 2, "max_labeled_demos": 5, "max_bootstrapped_demos": 3},
    # {"name": "n16_d3b5", "n": 16, "max_rounds": 2, "max_labeled_demos": 3, "max_bootstrapped_demos": 5},
    {"name": "n32_d3b5", "n": 4, "max_rounds": 2, "max_labeled_demos": 3, "max_bootstrapped_demos": 5},
]

# ── Default GEPA configurations (Stage 3) ───────────────────────
_DEFAULT_GEPA_CONFIGS: list[dict] = [
    # {"name": "gepa_medium", "auto": "medium", "use_merge": True, "track_stats": True},
    {"name": "gepa_light", "auto": "light", "use_merge": True, "track_stats": True},
]

# ── Valid stage names (in canonical execution order) ─────────────
VALID_STAGES = ("seed", "bootstrap", "miprov2", "gepa")


@dataclass
class TournamentConfig:
    """All hyperparameters for the sequential cascaded DCA optimization."""

    # ── Stage control ──────────────────────────────────────────────
    stages: List[str] = field(default_factory=lambda: ["seed", "bootstrap", "miprov2", "gepa"])
    top_k: int = 2
    skip_final_tournament: bool = False
    run_test_eval: bool = True

    # ── Model configuration ────────────────────────────────────────
    student_model_name: str = "azure/gpt-4.1-mini"
    seed_model_name: str = "azure/gpt-5.2"
    reflection_model_name: Optional[str] = "azure/gpt-5.2"  # fallback to REFLECTION_MODEL_NAME from .env
    use_cot: bool = False

    # ── Per-stage model overrides (None → use student_model_name) ──
    bootstrap_model_name: Optional[str] = None
    miprov2_model_name: Optional[str] = None
    gepa_model_name: Optional[str] = None

    # ── Temperature strategy ───────────────────────────────────────
    optimization_temperature: Optional[float] = None  # None → dspy default 0.0
    eval_temperature: float = 0.1 # VAL SET and Test set eval
    seed_generation_temperature: float = 1.0
    seed_eval_temperature: float = 0.1  # temp for baseline eval in Stage 0

    # ── Mini dataset sizes ─────────────────────────────────────────
    # Set to None to use full datasets.
    mini_train_size: Optional[int] = 120
    mini_val_size: Optional[int] = 60

    # ── Instructions vs field descriptions ─────────────────────────
    instructions_only: bool = True

    # ── Stage 0: Seed prompt generation ────────────────────────────
    n_seed_candidates: int = 3
    max_context_examples: int = 6  # examples shown to seed LLM per category
    seed_context_mode: str = "best"  # "worst" → failure examples; "best" → gold examples
    seed_from_file: Optional[str] = None  # skip Stage 0 — use this saved program
    seed_reasoning_effort: Optional[str] = "low"  # passed to seed LLM create_lm()
    seed_add_demos_variant: bool = True  # eval each candidate bare + with demos
    seed_parallel_evaluations: int = 4  # max concurrent Stage-0 seed eval variants

    # ── Stage 1: Bootstrap ─────────────────────────────────────────
    bootstrap_configs: List[dict] = field(default_factory=lambda: list(_DEFAULT_BOOTSTRAP_CONFIGS))
    bootstrap_num_threads: int = 4
    bootstrap_stop_at_score: Optional[float] = None  # early stop threshold
    bootstrap_teacher_model: Optional[str] = "azure/gpt-5.2"  # teacher_settings lm for Bootstrap

    # ── Stage 2: MIPROv2 ──────────────────────────────────────────
    miprov2_intensities: List[str] = field(default_factory=lambda: ["light", 
                                                                    # "medium"
                                                                    ]
                                                                    )
    miprov2_models: List[str] = field(default_factory=lambda: ["azure/gpt-4.1-mini"])
    miprov2_demo_configs: List[dict] = field(default_factory=lambda: [
        {"max_labeled_demos": 3, "max_bootstrapped_demos": 3},
        # {"max_labeled_demos": 6, "max_bootstrapped_demos": 6},
        # {"max_labeled_demos": 3, "max_bootstrapped_demos": 5},
        # {"max_labeled_demos": 5, "max_bootstrapped_demos": 3},
    ])
    # P1 enforcement: field descs are always empty when instructions_only=True.
    # _load_student() enforces this on every stage entry.  After compile(),
    # _normalize_compiled_field_descs absorbs any stray descs into the docstring.
    # This flag only controls whether descs are explicitly cleared BEFORE
    # MIPROv2 runs — effectively a no-op since they're already empty on load.
    miprov2_clear_field_descs: bool = False
    miprov2_num_threads: int = 4
    miprov2_prompt_model: Optional[str] = "azure/gpt-5.2"  # prompt_model for MIPROv2 instruction generation
    miprov2_prompt_model_reasoning_effort: Optional[str] = "low"  # reasoning_effort for prompt_model
    miprov2_minibatch_size: Optional[int] = None  # None → DSPy auto (trainset-based default)
    miprov2_metric_threshold: Optional[float] = None  # filter bootstrap traces with score < threshold
    miprov2_teacher_model: Optional[str] = "azure/gpt-5.2"  # teacher_settings lm for MIPROv2

    # ── Stage 3: GEPA ─────────────────────────────────────────────
    gepa_configs: List[dict] = field(default_factory=lambda: list(_DEFAULT_GEPA_CONFIGS))
    gepa_num_threads: int = 4
    gepa_reflection_images: bool = True
    gepa_reflection_image_detail: str = "auto"  # "low" | "high" | "auto"
    gepa_reflection_max_images: int = 6
    gepa_reflection_model: Optional[str] = "azure/gpt-5.2"
    gepa_reflection_minibatch_size: int = 5  # GEPA reflection_minibatch_size param
    gepa_train_size: Optional[int] = None    # per-GEPA train subset (None → use global)
    gepa_val_size: Optional[int] = None      # per-GEPA val subset (None → use global)
    gepa_add_format_failure_as_feedback: bool = True  # include format failures in GEPA reflection
    gepa_perfect_score: Optional[float] = 1.0   # maps to GEPA's perfect_score param
    gepa_lowest_score: Optional[float] = 0.0    # maps to GEPA's failure_score param (floor)
    gepa_candidate_selection_strategy: Optional[str] = "pareto"  # "pareto" | "current_best" (or "greedy" → mapped to "current_best")
    gepa_teacher_model: Optional[str] = "azure/gpt-5.2"  # stored in metadata only; GEPA has no teacher_settings param

    # ── Selection criteria ─────────────────────────────────────────
    selection_criteria: str = "score_only"  # "score_only" | "pareto" | "tokens_only"
    cross_stage_selection: bool = True
    track_tokens: bool = True

    # ── Hybrid metric (optional prompt-token efficiency penalty during optimization) ──
    # When enabled, the per-example metric seen by all optimizers becomes:
    #   hybrid = hybrid_score_weight * dca_score
    #          + hybrid_length_weight * max(0, 1 - prompt_tokens / hybrid_max_prompt_tokens)
    # Prompt token count is read from lm.history (real API-reported input tokens),
    # so multimodal prompts (images, long contexts) are counted correctly.
    # During plain val/test evaluation (trace=None) the raw DCA score is returned
    # unchanged, so leaderboard scores stay on the true metric.
    # Default off — existing runs are unaffected.
    use_hybrid_metric: bool = False
    hybrid_score_weight: float = 0.5
    hybrid_length_weight: float = 0.5
    hybrid_max_prompt_tokens: int = 4000  # input tokens at which length_score → 0

    # ── Dataset split ratios ────────────────────────────────────────
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    # ── Paths ──────────────────────────────────────────────────────
    output_base_dir: str = ".cache/tournament"
    log_base_dir: str = "log/tournament"
    pairs_dir: str = "data/dca_data/image_pairs"
    gt_dir: str = "data/dca_data/image_pairs_llm_output_exp1"
    split_seed: int = 42

    # ── Resolved at runtime (set by __post_init__ or run_tournament.py) ──
    timestamp: str = ""
    output_dir: str = ""
    log_dir: str = ""

    # ── Evaluation ─────────────────────────────────────────────────
    eval_num_threads: int = 4

    # ── Parallelism ────────────────────────────────────────────────
    max_parallel_optimizations: int = 1

    def __post_init__(self):
        for s in self.stages:
            if s not in VALID_STAGES:
                raise ValueError(f"Invalid stage '{s}'. Must be one of {VALID_STAGES}")
        # Note: pareto/tokens_only selection now uses val_avg_total_tokens
        # extracted from lm.history (not dspy.track_usage), so track_tokens
        # is no longer required for these selection criteria.
        if self.seed_context_mode not in {"worst", "best"}:
            raise ValueError(
                f"seed_context_mode='{self.seed_context_mode}' is invalid. Must be 'worst' or 'best'."
            )
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        if not self.output_dir:
            self.output_dir = str(Path(self.output_base_dir) / f"tournament_{self.timestamp}")
        if not self.log_dir:
            self.log_dir = str(Path(self.log_base_dir) / f"tournament_{self.timestamp}")

    def get_stage_model(self, stage: str) -> str:
        """Return the model to use for a given stage, with per-stage override."""
        overrides = {
            "bootstrap": self.bootstrap_model_name,
            "miprov2": self.miprov2_model_name,
            "gepa": self.gepa_model_name,
        }
        return overrides.get(stage) or self.student_model_name

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path | None = None) -> Path:
        """Save config to JSON.  Defaults to ``output_dir/config.json``."""
        if path is None:
            path = Path(self.output_dir) / "config.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TournamentConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    # ── Factory presets ────────────────────────────────────────────

    @classmethod
    def default(cls, **overrides) -> "TournamentConfig":
        """Full default: seed → bootstrap → miprov2."""
        return cls(**overrides)

    @classmethod
    def quick_test(cls, **overrides) -> "TournamentConfig":
        """Minimal preset for smoke-testing the pipeline."""
        defaults = dict(
            stages=["seed"],
            n_seed_candidates=1,
            top_k=1,
            run_test_eval=False,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def bootstrap_only(cls, **overrides) -> "TournamentConfig":
        """Run only the bootstrap stage (requires --seed-from-file)."""
        defaults = dict(stages=["bootstrap"], top_k=1)
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def full_pipeline(cls, **overrides) -> "TournamentConfig":
        """All four stages: seed → bootstrap → miprov2 → gepa."""
        defaults = dict(stages=["seed", "bootstrap", "miprov2", "gepa"])
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def from_cli_args(cls, args) -> "TournamentConfig":
        """Build a TournamentConfig from an argparse Namespace."""
        kwargs: dict = {}

        # Stage control
        if args.stages:
            kwargs["stages"] = [s.strip() for s in args.stages.split(",")]
        if args.top_k is not None:
            kwargs["top_k"] = args.top_k
        if getattr(args, "skip_final_tournament", False):
            kwargs["skip_final_tournament"] = True
        if getattr(args, "skip_test_eval", False):
            kwargs["run_test_eval"] = False

        # Models
        if args.student_model:
            kwargs["student_model_name"] = args.student_model
        if args.seed_model:
            kwargs["seed_model_name"] = args.seed_model
        if getattr(args, "reflection_model", None):
            kwargs["reflection_model_name"] = args.reflection_model
        if getattr(args, "cot", False):
            kwargs["use_cot"] = True
        if getattr(args, "no_cot", False):
            kwargs["use_cot"] = False

        # Per-stage model overrides
        if getattr(args, "bootstrap_model", None):
            kwargs["bootstrap_model_name"] = args.bootstrap_model
        if getattr(args, "miprov2_model", None):
            kwargs["miprov2_model_name"] = args.miprov2_model
        if getattr(args, "gepa_model", None):
            kwargs["gepa_model_name"] = args.gepa_model

        # Temperature
        if getattr(args, "optimization_temperature", None) is not None:
            kwargs["optimization_temperature"] = args.optimization_temperature
        if getattr(args, "eval_temperature", None) is not None:
            kwargs["eval_temperature"] = args.eval_temperature
        if getattr(args, "seed_eval_temperature", None) is not None:
            kwargs["seed_eval_temperature"] = args.seed_eval_temperature

        # Mini datasets
        if getattr(args, "mini_train_size", None) is not None:
            kwargs["mini_train_size"] = args.mini_train_size if args.mini_train_size > 0 else None
        if getattr(args, "mini_val_size", None) is not None:
            kwargs["mini_val_size"] = args.mini_val_size if args.mini_val_size > 0 else None

        # Instructions only
        if getattr(args, "instructions_only", False):
            kwargs["instructions_only"] = True
        if getattr(args, "with_field_descs", False):
            kwargs["instructions_only"] = False

        # Stage 0 — Seed
        if getattr(args, "n_seed_candidates", None) is not None:
            kwargs["n_seed_candidates"] = args.n_seed_candidates
        if getattr(args, "max_context_examples", None) is not None:
            kwargs["max_context_examples"] = args.max_context_examples
        if getattr(args, "seed_context_mode", None):
            kwargs["seed_context_mode"] = args.seed_context_mode
        if getattr(args, "seed_from_file", None):
            kwargs["seed_from_file"] = args.seed_from_file
        if getattr(args, "seed_reasoning_effort", None):
            kwargs["seed_reasoning_effort"] = args.seed_reasoning_effort
        if getattr(args, "seed_add_demos", False):
            kwargs["seed_add_demos_variant"] = True
        if getattr(args, "no_seed_add_demos", False):
            kwargs["seed_add_demos_variant"] = False
        if getattr(args, "seed_parallel_evaluations", None) is not None:
            kwargs["seed_parallel_evaluations"] = args.seed_parallel_evaluations

        # Stage 1 — Bootstrap
        if getattr(args, "bootstrap_configs", None):
            kwargs["bootstrap_configs"] = json.loads(args.bootstrap_configs)
        if getattr(args, "bootstrap_stop_at_score", None) is not None:
            kwargs["bootstrap_stop_at_score"] = args.bootstrap_stop_at_score
        if getattr(args, "bootstrap_num_threads", None) is not None:
            kwargs["bootstrap_num_threads"] = args.bootstrap_num_threads
        if getattr(args, "bootstrap_teacher_model", None):
            kwargs["bootstrap_teacher_model"] = args.bootstrap_teacher_model

        # Stage 2 — MIPROv2
        if getattr(args, "miprov2_auto", None):
            kwargs["miprov2_intensities"] = [s.strip() for s in args.miprov2_auto.split(",")]
        if getattr(args, "miprov2_models", None):
            kwargs["miprov2_models"] = [s.strip() for s in args.miprov2_models.split(",")]
        if getattr(args, "miprov2_max_labeled", None) is not None or \
                getattr(args, "miprov2_max_bootstrapped", None) is not None:
            labeled = args.miprov2_max_labeled if getattr(args, "miprov2_max_labeled", None) is not None else 6
            bootstrapped = args.miprov2_max_bootstrapped if getattr(args, "miprov2_max_bootstrapped", None) is not None else 6
            kwargs["miprov2_demo_configs"] = [
                {"max_labeled_demos": labeled, "max_bootstrapped_demos": bootstrapped}
            ]
        if getattr(args, "miprov2_clear_descs", False):
            kwargs["miprov2_clear_field_descs"] = True
        if getattr(args, "miprov2_num_threads", None) is not None:
            kwargs["miprov2_num_threads"] = args.miprov2_num_threads
        if getattr(args, "miprov2_prompt_model", None):
            kwargs["miprov2_prompt_model"] = args.miprov2_prompt_model
        if getattr(args, "miprov2_prompt_reasoning_effort", None):
            kwargs["miprov2_prompt_model_reasoning_effort"] = args.miprov2_prompt_reasoning_effort
        if getattr(args, "miprov2_minibatch_size", None) is not None:
            kwargs["miprov2_minibatch_size"] = args.miprov2_minibatch_size
        if getattr(args, "miprov2_metric_threshold", None) is not None:
            kwargs["miprov2_metric_threshold"] = args.miprov2_metric_threshold
        if getattr(args, "miprov2_teacher_model", None):
            kwargs["miprov2_teacher_model"] = args.miprov2_teacher_model

        # Stage 3 — GEPA
        if getattr(args, "gepa_auto", None):
            kwargs["gepa_configs"] = [{"name": f"gepa_{args.gepa_auto}", "auto": args.gepa_auto,
                                       "use_merge": True, "track_stats": True}]
        if getattr(args, "gepa_reflection_images", False):
            kwargs["gepa_reflection_images"] = True
        if getattr(args, "gepa_reflection_detail", None):
            kwargs["gepa_reflection_image_detail"] = args.gepa_reflection_detail
        if getattr(args, "gepa_reflection_max_images", None) is not None:
            kwargs["gepa_reflection_max_images"] = args.gepa_reflection_max_images
        if getattr(args, "gepa_reflection_model_name", None):
            kwargs["gepa_reflection_model"] = args.gepa_reflection_model_name
        if getattr(args, "gepa_minibatch_size", None) is not None:
            kwargs["gepa_reflection_minibatch_size"] = args.gepa_minibatch_size
        if getattr(args, "gepa_train_size", None) is not None:
            kwargs["gepa_train_size"] = args.gepa_train_size if args.gepa_train_size > 0 else None
        if getattr(args, "gepa_val_size", None) is not None:
            kwargs["gepa_val_size"] = args.gepa_val_size if args.gepa_val_size > 0 else None
        if getattr(args, "gepa_num_threads", None) is not None:
            kwargs["gepa_num_threads"] = args.gepa_num_threads
        if getattr(args, "gepa_teacher_model", None):
            kwargs["gepa_teacher_model"] = args.gepa_teacher_model
        if getattr(args, "gepa_add_format_failure_feedback", False):
            kwargs["gepa_add_format_failure_as_feedback"] = True
        if getattr(args, "gepa_no_format_failure_feedback", False):
            kwargs["gepa_add_format_failure_as_feedback"] = False
        if getattr(args, "gepa_perfect_score", None) is not None:
            kwargs["gepa_perfect_score"] = args.gepa_perfect_score
        if getattr(args, "gepa_lowest_score", None) is not None:
            kwargs["gepa_lowest_score"] = args.gepa_lowest_score
        if getattr(args, "gepa_candidate_selection", None):
            kwargs["gepa_candidate_selection_strategy"] = args.gepa_candidate_selection

        # Selection criteria
        if getattr(args, "selection_criteria", None):
            kwargs["selection_criteria"] = args.selection_criteria
        if getattr(args, "cross_stage_selection", False):
            kwargs["cross_stage_selection"] = True
        if getattr(args, "no_cross_stage_selection", False):
            kwargs["cross_stage_selection"] = False

        # Token tracking
        if getattr(args, "track_tokens", False):
            kwargs["track_tokens"] = True
        if getattr(args, "no_track_tokens", False):
            kwargs["track_tokens"] = False

        # Hybrid metric
        if getattr(args, "hybrid_metric", False):
            kwargs["use_hybrid_metric"] = True
        if getattr(args, "hybrid_score_weight", None) is not None:
            kwargs["hybrid_score_weight"] = args.hybrid_score_weight
        if getattr(args, "hybrid_length_weight", None) is not None:
            kwargs["hybrid_length_weight"] = args.hybrid_length_weight
        if getattr(args, "hybrid_max_prompt_tokens", None) is not None:
            kwargs["hybrid_max_prompt_tokens"] = args.hybrid_max_prompt_tokens

        # Seed generation temperature
        if getattr(args, "seed_generation_temperature", None) is not None:
            kwargs["seed_generation_temperature"] = args.seed_generation_temperature

        # Dataset split ratios
        if getattr(args, "train_ratio", None) is not None:
            kwargs["train_ratio"] = args.train_ratio
        if getattr(args, "val_ratio", None) is not None:
            kwargs["val_ratio"] = args.val_ratio
        if getattr(args, "test_ratio", None) is not None:
            kwargs["test_ratio"] = args.test_ratio

        # Paths & parallelism
        if getattr(args, "output_dir", None):
            kwargs["output_base_dir"] = args.output_dir
        if getattr(args, "log_dir", None):
            kwargs["log_base_dir"] = args.log_dir
        if getattr(args, "pairs_dir", None):
            kwargs["pairs_dir"] = args.pairs_dir
        if getattr(args, "gt_dir", None):
            kwargs["gt_dir"] = args.gt_dir
        if getattr(args, "split_seed", None) is not None:
            kwargs["split_seed"] = args.split_seed
        if getattr(args, "eval_num_threads", None) is not None:
            kwargs["eval_num_threads"] = args.eval_num_threads
        if getattr(args, "max_parallel", None) is not None:
            kwargs["max_parallel_optimizations"] = args.max_parallel
        if getattr(args, "resume_from", None):
            kwargs["output_dir"] = args.resume_from
            dirname = Path(args.resume_from).name
            if dirname.startswith("tournament_"):
                kwargs["timestamp"] = dirname.replace("tournament_", "")

        return cls(**kwargs)

    # ── Display ────────────────────────────────────────────────────

    def print_summary(self):
        """Print a human-readable summary of the config."""
        print(f"\n{'='*80}")
        print(f" Tournament Configuration  (ID: tournament_{self.timestamp})")
        print(f"{'='*80}")
        print(f"  Stages           : {' → '.join(self.stages)}")
        print(f"  Top-K            : {self.top_k}")
        print(f"  Selection        : {self.selection_criteria}")
        print(f"  Cross-stage sel. : {'Yes' if self.cross_stage_selection else 'No (per-stage)'}")
        print(f"  Student LM       : {self.student_model_name}")
        print(f"  Seed LM          : {self.seed_model_name}")
        print(f"  Reflection LM    : {self.reflection_model_name or '(from .env)'}")
        print(f"  CoT              : {'Yes' if self.use_cot else 'No (Predict)'}")
        print(f"  Opt temp         : {self.optimization_temperature or '0.0 (dspy default)'}")
        print(f"  Eval temp        : {self.eval_temperature}")
        print(f"  Seed gen temp    : {self.seed_generation_temperature}")
        print(f"  Seed eval temp   : {self.seed_eval_temperature}")
        print(f"  Instructions only: {'Yes' if self.instructions_only else 'No (+ field descs)'}")
        print(f"  Track tokens     : {'Yes' if self.track_tokens else 'No'}")
        if self.use_hybrid_metric:
            print(f"  Hybrid metric    : score*{self.hybrid_score_weight} + length*{self.hybrid_length_weight} "
                  f"(max_tokens={self.hybrid_max_prompt_tokens})")
        print(f"  Parallelism      : {self.max_parallel_optimizations} concurrent per stage")
        print(f"  Eval threads     : {self.eval_num_threads}")
        print(f"  Split ratios     : train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}")
        print(f"  Split seed       : {self.split_seed}")
        print(f"  Final re-eval    : {'Skip' if self.skip_final_tournament else 'Yes'}")
        print(f"  Test eval        : {'Yes' if self.run_test_eval else 'No'}")

        # Mini dataset
        mt = self.mini_train_size or "full"
        mv = self.mini_val_size or "full"
        print(f"  Mini train size  : {mt}")
        print(f"  Mini val size    : {mv}")

        # Per-stage model overrides
        overrides = []
        if self.bootstrap_model_name:
            overrides.append(f"bootstrap={self.bootstrap_model_name}")
        if self.miprov2_model_name:
            overrides.append(f"miprov2={self.miprov2_model_name}")
        if self.gepa_model_name:
            overrides.append(f"gepa={self.gepa_model_name}")
        if overrides:
            print(f"  Model overrides  : {', '.join(overrides)}")

        if "seed" in self.stages:
            print(f"\n  Stage 0 — Seed Generation:")
            if self.seed_from_file:
                print(f"    Seed from file   : {self.seed_from_file}")
            else:
                print(f"    N candidates     : {self.n_seed_candidates}")
                print(f"    Context examples : {self.max_context_examples} ({self.seed_context_mode})")
                print(f"    Reasoning effort : {self.seed_reasoning_effort or 'default'}")
                print(f"    Add demos variant: {'Yes (×2 evals)' if self.seed_add_demos_variant else 'No'}")
                print(f"    Parallel evals   : {self.seed_parallel_evaluations}")

        if "bootstrap" in self.stages:
            print(f"\n  Stage 1 — Bootstrap:")
            for i, bc in enumerate(self.bootstrap_configs):
                print(f"    Config {i}: {bc['name']}  "
                      f"(n={bc.get('n', '?')}, rounds={bc.get('max_rounds', '?')}, "
                      f"labeled={bc['max_labeled_demos']}, bootstrapped={bc['max_bootstrapped_demos']})")
            if self.bootstrap_stop_at_score is not None:
                print(f"    Stop at score  : {self.bootstrap_stop_at_score}")
            if self.bootstrap_teacher_model:
                print(f"    Teacher model  : {self.bootstrap_teacher_model}")

        if "miprov2" in self.stages:
            print(f"\n  Stage 2 — MIPROv2:")
            print(f"    Intensities : {', '.join(self.miprov2_intensities)}")
            effective_models = [self.miprov2_model_name] if self.miprov2_model_name else self.miprov2_models
            print(f"    Models      : {', '.join(effective_models)}")
            for dc in self.miprov2_demo_configs:
                print(f"    Demo config : labeled={dc['max_labeled_demos']}, bootstrapped={dc['max_bootstrapped_demos']}")
            print(f"    Clear descs : {self.miprov2_clear_field_descs}")
            if self.miprov2_prompt_model:
                print(f"    Prompt model: {self.miprov2_prompt_model} (effort={self.miprov2_prompt_model_reasoning_effort or 'default'})")
            if self.miprov2_minibatch_size is not None:
                print(f"    Minibatch sz: {self.miprov2_minibatch_size}")
            if self.miprov2_metric_threshold is not None:
                print(f"    Metric thr. : {self.miprov2_metric_threshold}")
            if self.miprov2_teacher_model:
                print(f"    Teacher mdl : {self.miprov2_teacher_model}")

        if "gepa" in self.stages:
            print(f"\n  Stage 3 — GEPA:")
            for i, gc in enumerate(self.gepa_configs):
                print(f"    Config {i}: {gc.get('name', '?')}  (auto={gc.get('auto', '?')}, "
                      f"merge={gc.get('use_merge', True)})")
            print(f"    Minibatch size    : {self.gepa_reflection_minibatch_size}")
            gt = self.gepa_train_size or "global"
            gv = self.gepa_val_size or "global"
            print(f"    GEPA train size   : {gt}")
            print(f"    GEPA val size     : {gv}")
            print(f"    Reflection images : {self.gepa_reflection_images}")
            if self.gepa_reflection_images:
                print(f"    Image detail      : {self.gepa_reflection_image_detail}")
                print(f"    Max images        : {self.gepa_reflection_max_images}")
            print(f"    Format feedback   : {self.gepa_add_format_failure_as_feedback}")
            print(f"    Candidate select  : {self.gepa_candidate_selection_strategy or 'default'}")
            if self.gepa_perfect_score is not None:
                print(f"    Perfect score     : {self.gepa_perfect_score}")
            if self.gepa_lowest_score is not None:
                print(f"    Lowest score      : {self.gepa_lowest_score}")
            if self.gepa_teacher_model:
                print(f"    Teacher model     : {self.gepa_teacher_model}")

        # Estimate run count
        n_seeds_initial = 1 if self.seed_from_file else self.n_seed_candidates
        total_runs = 0
        stage_runs = {}
        prev_k = n_seeds_initial

        for stage in self.stages:
            if stage == "seed":
                r = self.n_seed_candidates if not self.seed_from_file else 0
                # LLM-generated eval variants + baseline bare + baseline with_demos
                n_llm_evals = r * (2 if self.seed_add_demos_variant else 1)
                n_baseline_evals = 1 + (1 if self.seed_add_demos_variant else 0)
                n_evals = n_llm_evals + n_baseline_evals
                stage_runs[stage] = r
                stage_runs["seed_evals"] = n_evals
                prev_k = min(self.top_k, max(n_evals, 1))
            elif stage == "bootstrap":
                r = prev_k * len(self.bootstrap_configs)
                stage_runs[stage] = r
                prev_k = min(self.top_k, r) if r > 0 else prev_k
            elif stage == "miprov2":
                eff_models = [self.miprov2_model_name] if self.miprov2_model_name else self.miprov2_models
                n_configs = len(self.miprov2_intensities) * len(eff_models) * len(self.miprov2_demo_configs)
                r = prev_k * n_configs
                stage_runs[stage] = r
                prev_k = min(self.top_k, r) if r > 0 else prev_k
            elif stage == "gepa":
                r = prev_k * len(self.gepa_configs)
                stage_runs[stage] = r
                prev_k = min(self.top_k, r) if r > 0 else prev_k
            total_runs += stage_runs.get(stage, 0)

        print(f"\n  Estimated runs:")
        for stage in self.stages:
            extra = ""
            if stage == "seed" and "seed_evals" in stage_runs:
                extra = (f" ({stage_runs['seed_evals']} evals: "
                         f"{stage_runs.get(stage, 0)} LLM-generated × "
                         f"{'2 variants' if self.seed_add_demos_variant else '1 variant'}"
                         f" + baselines)")
            print(f"    {stage:12s}: {stage_runs.get(stage, '?')} optimization runs{extra}")
        print(f"    {'TOTAL':12s}: {total_runs}")

        print(f"\n  Output dir    : {self.output_dir}")
        print(f"  Log dir       : {self.log_dir}")
        print(f"{'='*80}\n")


