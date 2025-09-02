from dataclasses import dataclass
from typing import List, Optional
from dspy.teleprompt import BootstrapFewShotWithRandomSearch


@dataclass
class OptimizationConfig:
    """Configuration for optimization strategies"""
    max_labeled_demos: int = 4
    max_bootstrapped_demos: int = 4
    num_candidate_programs: int = 16
    num_threads: int = 4
    max_rounds: int = 2
    name: str = "Default"
    description: str = "Default optimization strategy"


class OptimizationRunner:
    """Configurable optimization runner for order extraction"""
    
    def __init__(self, 
                 metric,
                 student_model,
                 train_set,
                 val_set,
                 evaluate_fn):
        self.metric = metric
        self.student_model = student_model
        self.train_set = train_set
        self.val_set = val_set
        self.evaluate_fn = evaluate_fn
        
        # Default optimization strategies
        self.default_strategies = [
            OptimizationConfig(
                max_labeled_demos=4,
                max_bootstrapped_demos=4,
                num_candidate_programs=16,
                num_threads=4,
                max_rounds=2,
                name="Enhanced Bootstrap",
                description="Enhanced BootstrapFewShotWithRandomSearch with balanced parameters"
            ),
            OptimizationConfig(
                max_labeled_demos=6,
                max_bootstrapped_demos=2,
                num_candidate_programs=20,
                num_threads=4,
                max_rounds=1,
                name="Conservative Bootstrap",
                description="Conservative approach with more labeled examples, fewer bootstrapped"
            )
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
            
            if score > best_score:
                best_extractor = extractor
                best_score = score
                best_strategy_name = config.name
        
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
        """Run optimization with given configuration"""
        optimizer = BootstrapFewShotWithRandomSearch(
            metric=self.metric,
            max_labeled_demos=config.max_labeled_demos,
            max_bootstrapped_demos=config.max_bootstrapped_demos,
            num_candidate_programs=config.num_candidate_programs,
            num_threads=config.num_threads,
            max_rounds=config.max_rounds
        )
        
        optimized_extractor = optimizer.compile(
            student=self.student_model,
            trainset=self.train_set,
            valset=self.val_set
        )
        
        print(f"📈 Evaluating {config.name}...")
        result = self.evaluate_fn(optimized_extractor, devset=self.val_set)
        return optimized_extractor, result.score
    
    def _print_final_results(self, best_strategy: str, baseline_score: float, best_score: float):
        """Print final optimization results"""
        print(f"\n🎉 Optimization Complete!")
        print(f"Best Strategy: {best_strategy}")
        print(f"Baseline Accuracy: {baseline_score:.1f}%")
        print(f"Best Accuracy: {best_score:.1f}%")
        print(f"Total Improvement: {best_score - baseline_score:+.1f}%")
        
        # print(f"\n📊 Normalized Results (0-100%):")
        # print(f"Baseline: {baseline_score:.1f}%")
        # print(f"Best: {best_score:.1f}%") 
        # print(f"Improvement: {(best_score - baseline_score):+.1f}%")


def create_custom_strategy(name: str, description: str, **params) -> OptimizationConfig:
    """Helper function to create custom optimization strategies"""
    return OptimizationConfig(name=name, description=description, **params)


# Example usage with custom strategies:
if __name__ == "__main__":
    
    # Or create custom strategies
    custom_strategies = [
        create_custom_strategy(
            name="High Exploration",
            description="High candidate exploration with moderate demos",
            max_labeled_demos=5,
            max_bootstrapped_demos=3,
            num_candidate_programs=30,
            num_threads=6,
            max_rounds=1
        ),
        create_custom_strategy(
            name="Deep Refinement",
            description="Multiple rounds of refinement with fewer candidates",
            max_labeled_demos=3,
            max_bootstrapped_demos=5,
            num_candidate_programs=10,
            num_threads=4,
            max_rounds=3
        )
    ]
    