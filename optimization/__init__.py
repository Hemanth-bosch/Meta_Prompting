"""DCA prompt-optimization package.

Exports ``OptimizationRunner`` and the Bootstrap / MIPROv2 / GEPA / SIMBA
strategy factories (defined in ``optimization/opt.py``) used by
``extract_dca.py`` to optimize the DCA prompt with a single optimizer.
"""
from optimization.opt import (
    OptimizationConfig,
    OptimizationRunner,
    create_custom_strategy,
    create_gepa_strategy,
    create_bootstrap_strategy,
    create_miprov2_strategy,
    create_simba_strategy,
)
