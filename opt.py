from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, MIPROv2
from dspy import GEPA


@dataclass
class OptimizationConfig:
    """Flexible configuration for optimization strategies with DSPy-aligned defaults"""
    # Core identification
    name: str = "Default"
    description: str = "Default optimization strategy"
    optimizer_type: Literal["bootstrap", "gepa", "miprov2"] = "bootstrap"
    
    # Common parameters - aligned with DSPy defaults where applicable
    num_threads: int = None  # DSPy default: None (uses system default)
    max_labeled_demos: int = None  # Will use DSPy defaults per optimizer
    max_bootstrapped_demos: int = None  # Will use DSPy defaults per optimizer
    
    # Flexible optimizer-specific parameters
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default parameters based on optimizer type if not provided"""
        if not self.optimizer_params:
            self.optimizer_params = self._get_default_params()
        
        # Set DSPy-aligned defaults for demo counts if None
        if self.max_labeled_demos is None:
            self.max_labeled_demos = self._get_dspy_labeled_demos_default()
        if self.max_bootstrapped_demos is None:
            self.max_bootstrapped_demos = self._get_dspy_bootstrapped_demos_default()
    
    def _get_dspy_labeled_demos_default(self) -> int:
        """Get DSPy default for max_labeled_demos by optimizer type"""
        defaults = {
            "bootstrap": 16,  # DSPy default
            "gepa": 4,        # Common default
            "miprov2": 4      # DSPy default
        }
        return defaults.get(self.optimizer_type, 4)
    
    def _get_dspy_bootstrapped_demos_default(self) -> int:
        """Get DSPy default for max_bootstrapped_demos by optimizer type"""
        defaults = {
            "bootstrap": 4,   # DSPy default
            "gepa": 4,        # Common default  
            "miprov2": 4      # DSPy default
        }
        return defaults.get(self.optimizer_type, 4)
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for each optimizer type aligned with DSPy defaults"""
        defaults = {
            "bootstrap": {
                # DSPy defaults: max_labeled_demos=16, max_bootstrapped_demos=4,
                # num_candidate_programs=16, max_rounds=1
                "num_candidate_programs": 16,
                "max_rounds": 1
            },
            "gepa": {
                # DSPy defaults: auto=None, use_merge=True, track_stats=False
                # Only override if we need different behavior
            },
            "miprov2": {
                # DSPy defaults: auto='light', max_labeled_demos=4, 
                # max_bootstrapped_demos=4, seed=9
                "auto": "light"
            }
        }
        return defaults.get(self.optimizer_type, {})
    
    def get_param(self, key: str, default=None):
        """Get a parameter with fallback to default"""
        return self.optimizer_params.get(key, default)
    
    def set_param(self, key: str, value: Any):
        """Set an optimizer-specific parameter"""
        self.optimizer_params[key] = value
        return self
    
    def update_params(self, **kwargs):
        """Update multiple optimizer-specific parameters"""
        self.optimizer_params.update(kwargs)
        return self


class OptimizationRunner:
    """Configurable optimization runner for order extraction"""
    
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
        
        # Default optimization strategies using DSPy-aligned minimal parameters
        self.default_strategies = [
        ]
    
    def run_optimization(self, strategies: Optional[List[OptimizationConfig]] = None) -> tuple:
        """Run optimization with configurable strategies"""
        if strategies is None:
            strategies = self.default_strategies
        
        print("🚀 Starting Configurable Order Extractor Optimization...")
        
        # Evaluate baseline
        baseline_score = self._evaluate_baseline()
        
        best_extractor = None
        best_score = baseline_score
        best_strategy_name = "Baseline"
        
        # Test each strategy
        for i, config in enumerate(strategies, 1):
            print(f"\n⚡ Strategy {i}: {config.name}...")
            print(f"   {config.description}")
            
            extractor, score = self._run_strategy(config)
            improvement = score - baseline_score
            
            print(f"Strategy {i} Results:")
            print(f"  Accuracy: {score:.1f}%")
            print(f"  Improvement: {improvement:+.1f}%")
            
            if score >= best_score:  # Use >= instead of > to handle equal scores
                best_extractor = extractor
                best_score = score
                best_strategy_name = config.name
                print(f"  ✅ Updated best extractor (score: {score:.1f}%)")
            else:
                print(f"  ⚠️  Score not better than current best ({best_score:.1f}%)")
        
        self._print_final_results(best_strategy_name, baseline_score, best_score)
        return best_extractor, best_score
    
    def _evaluate_baseline(self) -> float:
        """Evaluate baseline model performance"""
        print("📊 Evaluating baseline model...")
        self.evaluate_fn.devset = self.val_set
        baseline_result = self.evaluate_fn(self.student_model)
        baseline_score = baseline_result.score
        print(f"Baseline accuracy: {baseline_score:.1f}%")
        return baseline_score
    
    def _run_strategy(self, config: OptimizationConfig) -> tuple:
        """Run optimization with given configuration using clean parameter system"""
        if config.optimizer_type == "bootstrap":
            bootstrap_kwargs = {
                "metric": self.metric,
                "max_labeled_demos": config.max_labeled_demos,
                "max_bootstrapped_demos": config.max_bootstrapped_demos,
                "num_candidate_programs": config.get_param("num_candidate_programs", 16),
                "max_rounds": config.get_param("max_rounds", 1)  # DSPy default
            }
            # Only add num_threads if it's not None (let DSPy handle default)
            if config.num_threads is not None:
                bootstrap_kwargs["num_threads"] = config.num_threads
            optimizer = BootstrapFewShotWithRandomSearch(**bootstrap_kwargs)
            
        elif config.optimizer_type == "gepa":
            # Use reflection_lm from config params or fallback to self.reflection_lm
            reflection_lm = config.get_param("reflection_lm") or self.reflection_lm
            gepa_kwargs = {
                "metric": self.metric_with_feedback,
                "auto": config.get_param("auto"),  # DSPy default: None
                "track_stats": config.get_param("track_stats", False),  # DSPy default
                "use_merge": config.get_param("use_merge", True),  # DSPy default
            }
            # Only add optional parameters if they're not None
            if config.num_threads is not None:
                gepa_kwargs["num_threads"] = config.num_threads
            if reflection_lm is not None:
                gepa_kwargs["reflection_lm"] = reflection_lm
            optimizer = GEPA(**gepa_kwargs)
            
        elif config.optimizer_type == "miprov2":
            optimizer = self._create_miprov2_optimizer(config)
            
        else:
            raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")
        
        print(f"🔧 Compiling {config.name}...")
        optimized_extractor = optimizer.compile(
            student=self.student_model,
            trainset=self.train_set,
            valset=self.val_set
        )
        
        print(f"🔍 {config.optimizer_type.upper()} compilation result:")
        print(f"   Type: {type(optimized_extractor)}")
        print(f"   Is None: {optimized_extractor is None}")
        if optimized_extractor is not None:
            print(f"   Has forward method: {hasattr(optimized_extractor, 'forward')}")
            print(f"   Has __call__ method: {hasattr(optimized_extractor, '__call__')}")
        
        print(f"📈 Evaluating {config.name}...")
        result = self.evaluate_fn(optimized_extractor, devset=self.val_set)
        return optimized_extractor, result.score
    
    def _create_miprov2_optimizer(self, config: OptimizationConfig):
        """Create MIPROv2 optimizer with DSPy-aligned parameter handling"""
        # Start with required parameters
        miprov2_params = {
            "metric": self.metric,
            "auto": config.get_param("auto", "light")  # DSPy default
        }
        
        # Add common demo parameters from config
        miprov2_params["max_labeled_demos"] = config.max_labeled_demos
        miprov2_params["max_bootstrapped_demos"] = config.max_bootstrapped_demos
        
        # Add num_threads only if specified (let DSPy handle None default)
        if config.num_threads is not None:
            miprov2_params["num_threads"] = config.num_threads
        
        # Add all other parameters directly from optimizer_params
        # This allows users to set ANY MIPROv2 parameter without hardcoding them
        for param_name, param_value in config.optimizer_params.items():
            if param_name not in miprov2_params and param_value is not None:
                miprov2_params[param_name] = param_value
        
        print(f"MIPROv2 parameters: {miprov2_params}")
        return MIPROv2(**miprov2_params)
    
    def _print_final_results(self, best_strategy: str, baseline_score: float, best_score: float):
        """Print final optimization results"""
        print(f"\n🎉 Optimization Complete!")
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
        optimizer_params=optimizer_params
    )


def create_gepa_strategy(name: str, description: str, auto=None,
                        num_threads=None, max_labeled_demos=None, max_bootstrapped_demos=None,
                        **gepa_params) -> OptimizationConfig:
    """
    Helper function to create GEPA optimization strategies with flexible interface.
    """
    # Set basic GEPA parameters in optimizer_params
    optimizer_params = {}
    if auto is not None:
        optimizer_params["auto"] = auto
    
    # Add any additional GEPA-specific parameters
    optimizer_params.update(gepa_params)
    
    return OptimizationConfig(
        name=name,
        description=description,
        optimizer_type="gepa",
        num_threads=num_threads,
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,
        optimizer_params=optimizer_params
    )


def create_bootstrap_strategy(name: str, description: str, max_labeled_demos=None,
                            max_bootstrapped_demos=None, num_threads=None,
                            **bootstrap_params) -> OptimizationConfig:
    """
    Helper function to create Bootstrap optimization strategies with flexible interface.
    """
    # Add any additional Bootstrap-specific parameters
    optimizer_params = {}
    optimizer_params.update(bootstrap_params)
    
    return OptimizationConfig(
        name=name,
        description=description,
        optimizer_type="bootstrap",
        num_threads=num_threads,
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,
        optimizer_params=optimizer_params
    )


def create_miprov2_strategy(name: str, description: str, auto="light",
                           num_threads=4, max_labeled_demos=4, max_bootstrapped_demos=4,
                           **miprov2_params) -> OptimizationConfig:
    """
    Helper function to create MIPROv2 optimization strategies with intuitive interface.
    """
    # Set auto parameter in optimizer_params
    optimizer_params = {"auto": auto}
    
    # Add any additional MIPROv2-specific parameters
    optimizer_params.update(miprov2_params)
    
    return OptimizationConfig(
        name=name,
        description=description,
        optimizer_type="miprov2",
        num_threads=num_threads,
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,
        optimizer_params=optimizer_params
    )


# Example usage with custom strategies:
if __name__ == "__main__":
    
    # Create custom Bootstrap strategies
    bootstrap_strategies = [
        # create_bootstrap_strategy(
        #     name="High Exploration Bootstrap",
        #     description="High candidate exploration with moderate demos",
        #     max_labeled_demos=5,
        #     max_bootstrapped_demos=3,
        #     num_candidate_programs=30,
        #     num_threads=6,
        #     max_rounds=1
        # ),
        # create_bootstrap_strategy(
        #     name="Deep Refinement Bootstrap",
        #     description="Multiple rounds of refinement with fewer candidates",
        #     max_labeled_demos=3,
        #     max_bootstrapped_demos=5,
        #     num_candidate_programs=10,
        #     num_threads=4,
        #     max_rounds=3
        # )
    ]
    
    # Create custom GEPA strategies for Japanese order extraction
    gepa_strategies = [
        # create_gepa_strategy(
        #     name="GEPA Light Japanese",
        #     description="GEPA light optimization for Japanese order extraction with fast iteration",
        #     auto="light",
        #     num_threads=4,
        #     track_stats=True,
        #     use_merge=False
        # )
    ]
    
    # Create custom MIPROv2 strategies for Japanese order extraction
    miprov2_strategies = [
        create_miprov2_strategy(
            name="MIPROv2 Fast Japanese",
            description="MIPROv2 light optimization with fast iteration",
            # auto="light",
            # minibatch=True,
            # minibatch_size=25,
            # view_data_batch_size=5
        ),
        # create_miprov2_strategy(
        #     name="MIPROv2 Balanced Japanese", 
        #     description="MIPROv2 medium optimization with balanced performance",
        #     auto="medium",
        #     minibatch=True,
        #     minibatch_size=50,
        #     view_data_batch_size=10,
        #     num_trials=15
        # ),
        # create_miprov2_strategy(
        #     name="MIPROv2 Comprehensive Japanese",
        #     description="MIPROv2 heavy optimization with comprehensive search",
        #     auto="heavy",
        #     num_threads=6,
        #     max_labeled_demos=6,
        #     max_bootstrapped_demos=6,
        #     minibatch=True,
        #     minibatch_size=75,
        #     view_data_batch_size=15,
        #     num_trials=25,
        #     seed=42  # For reproducibility
        # )
    ]
    
    # Combine all strategies
    all_strategies = bootstrap_strategies + gepa_strategies + miprov2_strategies
    
    # Example: Use only GEPA strategies for Japanese order extraction
    japanese_order_gepa_strategies = gepa_strategies
    
    # Example: Use only MIPROv2 strategies for Japanese order extraction
    japanese_order_miprov2_strategies = miprov2_strategies
    
    print("Available optimization strategies for Japanese order extraction:")
    for strategy in all_strategies:
        print(f"  - {strategy.name}: {strategy.description} (Type: {strategy.optimizer_type})")
        print(strategy)
    