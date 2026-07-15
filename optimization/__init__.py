"""Wrapper-only GEPA/DSPy optimization package (OptimizationRunner + strategy factories). No run.py analog."""
# [wrapper-only] -- no run.py analog. GEPA/DSPy optimization is a pure-wrapper addition.
# run.py (MBD/weakspot_code_to_run_2/run.py) runs a fixed per-weakspot pipeline
# (run_pipeline -> _locate_weakspot / _build_equation_context -> _step_3a_get_context ->
# _step_3b_generate_testcases -> _step_4a_solve_equations_dll) that computes NO score and
# performs NO instruction optimization. This package instead drives DSPy optimizers over
# the frozen run.py prompt seeds in signatures/mbd.py (the 4 VERBATIM run.py templates),
# scored by the M1..M8 metric suite in evaluation/mbd_metric.py.
# Re-exports from optimization/opt.py (all [wrapper-only], no run.py analog):
#   OptimizationConfig         -- optimization/opt.py:56   optimizer/run config dataclass (@dataclass at :55)
#   OptimizationRunner         -- optimization/opt.py:129  builds + runs the DSPy optimizer
#   create_custom_strategy     -- optimization/opt.py:421  strategy factory (custom)
#   create_gepa_strategy       -- optimization/opt.py:437  strategy factory (GEPA)
#   create_bootstrap_strategy  -- optimization/opt.py:457  strategy factory (BootstrapFewShot)
#   create_miprov2_strategy    -- optimization/opt.py:484  strategy factory (MIPROv2)
#   create_simba_strategy      -- optimization/opt.py:502  strategy factory (SIMBA)
from optimization.opt import (
    OptimizationConfig,
    OptimizationRunner,
    create_custom_strategy,
    create_gepa_strategy,
    create_bootstrap_strategy,
    create_miprov2_strategy,
    create_simba_strategy,
)
