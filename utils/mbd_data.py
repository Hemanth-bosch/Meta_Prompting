"""Dataset loader for MBD weakspot tasks → dspy.Example objects.

This is the DATASET-BUILD half of the MBD-UTA wrapper. It reproduces run.py's
run_pipeline() input-prep + per-weakspot Phase 1/Phase 2, but stops BEFORE the LLM
prompts (those run in extract_mbd_full.MBDFullPipelineModule.forward()). Mapping:

    load_mbd_dataset()            ≙ run.py run_pipeline() inputs + graph prep
        load_equations/...        → run.py:754-756
        extract_step_time         → run.py:784
        get_const_default_values  → run.py:795-796  (+ None-filter)
        get_sequence_by_name/     → run.py:829-831
          fix_operation_instance_names/build_global_ops
    _build_example_for_weakspot() ≙ run.py Phase 1 + Phase 2
        _locate_weakspot          → run.py:843 / run.py:297-313  (IMPORTED from run.py)
        _build_equation_context   → run.py:848 / run.py:320-431  (IMPORTED from run.py)

SINGLE SOURCE OF TRUTH — IMPORTED, NOT RE-CODED (the core fidelity invariant of this file):
  _build_example_for_weakspot() does NOT contain its own copy of the Phase-1/Phase-2 graph
  math. `_locate_weakspot` and `_build_equation_context` are the ACTUAL run.py functions,
  pulled off a uniquely-named `mbd_run` module (loaded at import, lines 108-113) under the same
  MBD-config swap that the pipeline.* imports use. run_pipeline() invokes them identically
  (run.py:843 selection = _locate_weakspot(data, ws_id); run.py:848 eq_ctx =
  _build_equation_context(data, global_ops, selection, const_values)). Consequence: if MIDAS
  edits either function, this wrapper's Phase 1/2 follows automatically — there is no shadow
  reimplementation to drift. The ONLY wrapper work in _build_example_for_weakspot() is
  SERIALIZATION of eq_ctx's fields into dspy.Example strings (which must be pickle/deep-copy
  safe for GEPA's subprocess + DSPy's thread pool) plus front-loading two prompt inputs that
  run.py builds later at render time (see that function's docstring).

[wrapper-only] GEPA train/val/test SPLIT — NO run.py analog (run.py runs every configured FC
exactly once, computes no score, and never partitions its inputs). These helpers exist solely
to feed GEPA optimization + the M1..M8 metric suite and are all tagged [wrapper-only] inline:
    stratified_split_mbd_dataset()  → GEPA train/val/test partition (global or per-FC)
    _example_is_arithmetic()        → weakspot has a solver-runnable (non-empty) assignment chain
    _rebalance_arith_into_train()   → keep arithmetic weakspots in train for solver-reflection
    _example_block_type()           → weakspot op function/method name (stratification key)
    _example_is_bit()               → bit-operation weakspot flag (must span train AND val)
    _ensure_group_on_both_sides()   → size-preserving swap so a weakspot group spans both sides
    _default_fc_entries() / all_fc_entries() / _hardcoded_all_fc_entries()
                                    → task lists; all_fc_entries reads MBD/.../run_config.json,
                                      the SAME fc_entries source run.py:230 consumes.

The Phase 1/Phase 2 graph math is IMPORTED from run.py (not re-coded) so there is a
single source of truth; the wrapper here only SERIALIZES the result into dspy.Example
fields (which must be pickle/deep-copy safe for GEPA's subprocess + DSPy's thread pool).

Each Example contains:
  - Input fields (passed to DSPy Module): equation_json, const_values,
    weakspot_context, instance_names, raster_time
  - Metadata fields (used by metric, not passed to LLM): signalsheet_data,
    model_interface_data, fc_name, weakspot_id, dll_path
  - Precomputed Phase-2 fields (consumed by forward()): relations_json,
    equation_results_json, op_map_json, weakspot_op_key, weakspot_operation,
    example_explanation_output
"""

from __future__ import annotations

import json                              # run.py:9 import json (config + eq_ctx serialization)
import logging                           # run.py:10 import logging (root-handler neutralize below)
import os                                # run.py:11 import os (fc_dir/dll_path joins)
import random                            # [wrapper-only] seeded RNG for GEPA train/val split (run.py never partitions)
import sys                               # run.py:14 import sys (sys.path append + config-swap in sys.modules)
from pathlib import Path                 # [wrapper-only] type import (Path unused at runtime here)
from typing import Any, List, Tuple      # [wrapper-only] type hints for Example lists/split tuples

import dspy                              # [wrapper-only] Example container (run.py has no dspy — runs pipeline inline)

# Dataset-build messages (FC/weakspot drops) go through this logger at WARNING level —
# they are EXPECTED filtering of degenerate weakspots, not errors.
logger = logging.getLogger("mbd.data")

# Add MBD pipeline to path for imports (append to avoid shadowing repo-level config.py)
# [wrapper-only] locate MBD root relative to THIS file (run.py:95 uses its own __file__ dir instead)
_MBD_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "MBD", "weakspot_code_to_run_2"
)
_MBD_ROOT = os.path.normpath(_MBD_ROOT)   # [wrapper-only] canonicalize the ../MBD/... path
if _MBD_ROOT not in sys.path:             # [wrapper-only] APPEND (run.py:95 sys.path.insert(0,...); we append to avoid shadowing repo config.py)
    sys.path.append(_MBD_ROOT)            # [wrapper-only] make pipeline.* importable

# We need to temporarily make MBD config importable as the pipeline expects it.
# Save repo-level config module, import pipeline, then restore.
import importlib                         # [wrapper-only] used to load the MBD config.py by file path
_repo_config = sys.modules.get("config")  # [wrapper-only] stash repo config.py so we can restore it after the pipeline imports

# Force the MBD config to be loadable for pipeline imports
_mbd_config_path = os.path.join(_MBD_ROOT, "config.py")  # [wrapper-only] MBD/.../config.py (the cfg every pipeline.* does `from config import cfg`, run.py:97)
_mbd_config_spec = importlib.util.spec_from_file_location("mbd_config", _mbd_config_path)  # [wrapper-only] build import spec for MBD config
_mbd_config_mod = importlib.util.module_from_spec(_mbd_config_spec)  # [wrapper-only] materialize the module object

# Temporarily inject MBD config as "config" so pipeline imports work
sys.modules["config"] = _mbd_config_mod   # [wrapper-only] masquerade MBD config as top-level `config` so pipeline.* `from config import cfg` resolves to it
_mbd_config_spec.loader.exec_module(_mbd_config_mod)  # [wrapper-only] execute the MBD config module

# Used by load_mbd_dataset() — mirrors run.py:101-111 (inputs) and run.py:829-831 (graph prep).
from pipeline.input_loader import (
    load_equations,          # run.py:754  eqn_dict = load_equations(fc_dir)
    load_signal_sheet,       # run.py:755  signal_sheet_df = load_signal_sheet(fc_dir)
    load_model_interface,    # run.py:756  model_interface_data = load_model_interface(fc_dir)
    get_filtered_dfs,        # run.py:781  _, _, const_df, _ = get_filtered_dfs(signal_sheet_df)
    get_const_default_values,# run.py:795  const_values_dict = get_const_default_values(...)
    extract_step_time,       # run.py:784  step_time = extract_step_time(leaf_node)
)
# After the import refactor (below) the Phase-1/Phase-2 graph math lives ENTIRELY in run.py's
# _build_equation_context, so the wrapper only needs the three graph helpers run_pipeline() itself
# calls to PREPARE that input (run.py:829-831) plus build_op_mapping to serialize the op_map.
from pipeline.graph import (
    build_global_ops,            # run.py:831  global_ops = build_global_ops(data)
    fix_operation_instance_names,# run.py:830  fix_operation_instance_names(data)
    get_sequence_by_name,        # run.py:829  data = get_sequence_by_name(eqn_dict, leaf_node)
)
from pipeline.tree_to_string import (
    build_op_mapping,            # run.py:662  op_map = build_op_mapping(eq_ctx.dependency_mapping)
)
# Front-loaded prompt context (run.py builds these at prompt-render time in _step_3a/_step_3b;
# the wrapper precomputes them into Example fields): get_weakspot_context feeds Prompt 1's
# "Weakspot Context"; get_weakspot_example_output feeds Prompt 2's "# !!!EXAMPLE ONLY!!!" block.
from pipeline.context_helpers import get_weakspot_context, get_weakspot_example_output
from pipeline.types import EquationContext, EquationResult  # noqa: F401  (documents eq_ctx contract)

# ── Import run.py's Phase 1 + Phase 2 DIRECTLY (single source of truth) ───────────────────────
# _locate_weakspot (run.py:297-313) and _build_equation_context (run.py:320-431) are defined in
# run.py itself (NOT in pipeline/*), so we load run.py as a uniquely-named module under the same
# config swap. This CONFINES the Phase-1/Phase-2 graph math to ONE place: if MIDAS changes
# _build_equation_context, the wrapper follows automatically (no silent divergence). run.py's
# `if __name__ == "__main__"` guard (run.py:947) keeps main() from running on import.
# SIDE EFFECT to neutralize: run.py calls logging.basicConfig() at import (run.py:189-192), which
# would attach a root handler before extract_mbd_full.main() configures logging. Capture root
# logging state and undo any handler the import adds, restoring the level.
_root_logger = logging.getLogger()          # [wrapper-only] capture root logger before importing run.py (basicConfig side effect at run.py:189)
_pre_run_handlers = list(_root_logger.handlers)  # [wrapper-only] snapshot pre-import handlers so we can remove only the one run.py adds
_pre_run_level = _root_logger.level         # [wrapper-only] snapshot pre-import level to restore after run.py:189 basicConfig

_mbd_run_path = os.path.join(_MBD_ROOT, "run.py")  # [wrapper-only] path to the ORIGINAL run.py (source of _locate_weakspot/_build_equation_context)
_mbd_run_spec = importlib.util.spec_from_file_location("mbd_run", _mbd_run_path)  # [wrapper-only] unique module name so run.py's `if __name__=="__main__"` guard (run.py:947) skips main()
_mbd_run_mod = importlib.util.module_from_spec(_mbd_run_spec)  # [wrapper-only] materialize the run.py module object
_mbd_run_spec.loader.exec_module(_mbd_run_mod)  # [wrapper-only] execute run.py (triggers run.py:189 basicConfig side effect, undone below)
_locate_weakspot = _mbd_run_mod._locate_weakspot              # run.py:297-313 (Phase 1)
_build_equation_context = _mbd_run_mod._build_equation_context  # run.py:320-431 (Phase 2)

# Undo run.py's logging.basicConfig() side effect (remove only handlers it added; restore level).
for _h in list(_root_logger.handlers):      # [wrapper-only] scan current root handlers after import
    if _h not in _pre_run_handlers:         # [wrapper-only] identify the handler run.py:189 basicConfig added
        _root_logger.removeHandler(_h)      # [wrapper-only] remove it (leave logging config to extract_mbd_full.main())
_root_logger.setLevel(_pre_run_level)       # [wrapper-only] restore pre-import root level (run.py:190 set it to INFO)

# Restore repo-level config
if _repo_config is not None:                # [wrapper-only] repo had a real config.py loaded → put it back
    sys.modules["config"] = _repo_config    # [wrapper-only] undo the MBD-config masquerade injected above
else:                                       # [wrapper-only]
    del sys.modules["config"]               # [wrapper-only] no prior config → drop the temporary MBD-config entry

MBD_DATA_ROOT = os.path.join(_MBD_ROOT, "data")  # [wrapper-only] default fc_dir root (data/ dir of FC folders); run.py:239 takes this root from run_config.json's "fc_dir"


def load_mbd_dataset(
    data_root: str = None,
    fc_entries: List[dict] = None,
) -> List[dspy.Example]:
    """Load MBD weakspot tasks as dspy.Example objects.

    Each Example has input fields only (no expected outputs — metric is reference-free).

    Args:
        data_root: Path to data/ directory containing FC folders
        fc_entries: List of dicts with keys: fc_name, leaf_node, weakspot_ids, raster_time

    Returns:
        List of dspy.Example with input fields set
    """
    if data_root is None:                          # [wrapper-only] default to MBD/.../data (run.py takes fc_dir from config)
        data_root = MBD_DATA_ROOT                   # [wrapper-only]
                                                    #
    if fc_entries is None:                          # [wrapper-only] default task list (run.py reads run_config.json fc_entries, 230)
        fc_entries = _default_fc_entries()          # [wrapper-only]
                                                    #
    dataset = []                                    # [wrapper-only] accumulate dspy.Example list (run.py accumulates all_outputs:835)
                                                    #
    for entry in fc_entries:                        # run.py:940 for i, args in enumerate(configs)  (one FC entry per config)
        fc_name = entry["fc_name"]                  # run.py:234 fc_name = entry.get("fc_name")
        fc_dir = os.path.join(data_root, fc_name)   # run.py:239/728 fc_dir = join(fc_dir_root, fc_name) → abspath
                                                    #
        if not os.path.isdir(fc_dir):               # [wrapper-only] skip missing FC dir (run.py assumes it exists)
            logger.warning("FC directory not found: %s, skipping", fc_dir)  # [wrapper-only] trace
            continue                                # [wrapper-only]
                                                    #
        # Load raw data
        try:                                        # [wrapper-only] guard (run.py loads unconditionally)
            equations_data = load_equations(fc_dir) # run.py:754 eqn_dict = load_equations(fc_dir)
        except Exception as e:                      # [wrapper-only] skip unreadable FC
            logger.warning("Could not load equations for %s: %s", fc_name, e)  # [wrapper-only] trace
            continue                                # [wrapper-only]
                                                    #
        # Load signalsheet as raw CSV text (for metric use)
        signalsheet_path = os.path.join(fc_dir, "signalsheet.csv")  # run.py:755 (run.py loads it as a DataFrame; wrapper keeps raw text for the metric)
        signalsheet_data = ""                       # [wrapper-only] Example.signalsheet_data (metric input M1/M2/M3/M6)
        if os.path.exists(signalsheet_path):        # [wrapper-only]
            with open(signalsheet_path, "r", encoding="utf-8") as f:  # [wrapper-only] read raw CSV
                signalsheet_data = f.read()         # [wrapper-only]
                                                    #
        # Load model_interface as raw CSV text (for metric use)
        model_interface_path = os.path.join(fc_dir, "model_interface.csv")  # run.py:756 (run.py: load_model_interface() list-of-dicts; wrapper keeps raw text)
        model_interface_data = ""                   # [wrapper-only] Example.model_interface_data (metric/DLL input)
        if os.path.exists(model_interface_path):    # [wrapper-only]
            with open(model_interface_path, "r", encoding="utf-8") as f:  # [wrapper-only] read raw CSV
                model_interface_data = f.read()     # [wrapper-only]
                                                    #
        # Process leaf node and step time (needed for const_values).
        # FIDELITY (run.py:784): step_time is derived from the LEAF-NODE NAME via
        # extract_step_time(), NOT from the config entry's raster_time (which run.py
        # ignores entirely). e.g. '_Proc' has no 'Nms' token -> default 20.0 -> dT=20000.
        leaf_node = entry["leaf_node"]              # run.py:761 leaf_node = args.leaf_node or leaf_nodes[0]
        step_time = extract_step_time(leaf_node)  # ms (matches run.py)  # run.py:784 step_time = extract_step_time(leaf_node)

        # Get signalsheet DataFrame for const_values extraction
        try:                                        # [wrapper-only] guard (run.py loads unconditionally)
            signal_df = load_signal_sheet(fc_dir)   # run.py:755 signal_sheet_df = load_signal_sheet(fc_dir)
            _, _, const_df, _ = get_filtered_dfs(signal_df)  # run.py:781 _, _, const_df, _ = get_filtered_dfs(signal_sheet_df)
            model_interface = load_model_interface(fc_dir)   # run.py:756 model_interface_data = load_model_interface(fc_dir)
            const_values = get_const_default_values(const_df, step_time, model_interface)  # run.py:795 const_values_dict = get_const_default_values(const_df, step_time, model_interface_data)
            # FIDELITY (run.py:796): drop None-valued constants.
            const_values = {k: v for k, v in const_values.items() if v is not None}  # run.py:796 const_values = {k: v for k, v in const_values_dict.items() if v is not None}
        except Exception as e:                      # [wrapper-only] degrade to empty consts on load error
            logger.warning("Could not load signalsheet/const for %s: %s", fc_name, e)  # [wrapper-only] trace
            const_values = {}                       # [wrapper-only]

        # Filter to specified sequence.
        # FIDELITY (run.py:829-831): run.py uses the get_sequence_by_name() result DIRECTLY:
        #     data = get_sequence_by_name(eqn_dict, leaf_node)
        #     fix_operation_instance_names(data); global_ops = build_global_ops(data)
        # get_sequence_by_name() returns ONLY {"Sequence": <filtered>} (graph.py:46-58) — it
        # drops every other top-level key. We name the result `data` (matching run.py) to keep
        # variable names aligned and the flow trivially mappable.
        data = get_sequence_by_name(equations_data, leaf_node)  # run.py:829 data = get_sequence_by_name(eqn_dict, leaf_node)
        if not data.get("Sequence"):                # [wrapper-only] skip leaf-not-found (run.py:762-764 errors/returns)
            logger.warning("Leaf node '%s' not found in %s, skipping", leaf_node, fc_name)  # [wrapper-only] trace
            continue                                # [wrapper-only]

        # TODO(resolved 2026-06-18): the prior code did
        #     data_full = {**equations_data, "Sequence": data_filtered["Sequence"]}
        # run.py does NOT do this — and it was a LATENT fidelity divergence. build_global_ops()
        # scans not just "Sequence" but also "Output Signal"/"Weakspot Element" sections
        # (graph.py:71-76). Merging the full equations.json back in would feed those sections to
        # build_global_ops while run.py (which passes the bare get_sequence_by_name result) would
        # not — silently inflating global_ops. It happens to be a NO-OP for the current dataset
        # (verified 2026-06-18: all 16 equations.json files have ONLY the "Sequence" top-level
        # key, so {**equations_data, "Sequence": X} == {"Sequence": X}), but a future FC with
        # extra sections would have diverged. Using `data` directly is byte-faithful to run.py.
        fix_operation_instance_names(data)          # run.py:830 fix_operation_instance_names(data)
        global_ops = build_global_ops(data)         # run.py:831 global_ops = build_global_ops(data)

        # DLL path
        dll_dir = os.path.join(fc_dir, "env_libs")  # run.py:800 dll_path = join(fc_dir, "env_libs", f"{fcname}.dll")
        dll_path = ""                               # [wrapper-only] Example.dll_path (metric M7/M8 + forward _get_function_caller)
        if os.path.isdir(dll_dir):                  # [wrapper-only] run.py builds path from fcname; here we scan env_libs/
            dll_files = [f for f in os.listdir(dll_dir) if f.endswith(".dll")]  # [wrapper-only] find the .dll
            if dll_files:                           # [wrapper-only]
                dll_path = os.path.join(dll_dir, dll_files[0])  # [wrapper-only] first .dll found

        # Process each weakspot
        for ws_id in entry["weakspot_ids"]:         # run.py:837 for ws_id in weakspot_selection
            example = _build_example_for_weakspot(  # run.py:843+848 Phase 1 (_locate_weakspot) + Phase 2 (_build_equation_context)
                data, global_ops, const_values, ws_id,  # run.py:843/848 args (data, global_ops, selection/const_values)
                fc_name=fc_name,                    # [wrapper-only] metadata
                leaf_node=leaf_node,                # [wrapper-only] metadata
                step_time_ms=step_time,             # run.py:784 step_time → Example.raster_time (×1000)
                signalsheet_data=signalsheet_data,  # [wrapper-only] metric metadata
                model_interface_data=model_interface_data,  # [wrapper-only] metric/DLL metadata
                dll_path=dll_path,                  # [wrapper-only] metric/DLL metadata
            )
            if example is not None:                 # run.py:844-845 if selection is None: continue (degenerate skip)
                dataset.append(example)             # [wrapper-only] collect Example (run.py appends WeakspotTestCaseOutput:897)

    print(f"Loaded {len(dataset)} MBD weakspot tasks from {len(fc_entries)} FC entries")  # [wrapper-only] trace
    return dataset                                  # [wrapper-only] return Example list (run.py has no dataset — runs inline)


def _build_example_for_weakspot(
    data: dict,
    global_ops: dict,
    const_values: dict,
    ws_id: int,
    *,
    fc_name: str,
    leaf_node: str,
    step_time_ms: float,
    signalsheet_data: str,
    model_interface_data: str,
    dll_path: str,
) -> dspy.Example | None:
    """Build a single dspy.Example for one weakspot task.

    ── 1:1 MAPPING TO run.py (Phase 1 + Phase 2) ────────────────────────────────────────
    Phase 1 + Phase 2 are NOT re-implemented here — they are IMPORTED from run.py and called
    directly, so the graph math has exactly ONE home (eliminating the silent-divergence risk):
        _locate_weakspot(data, ws_id)                        → run.py:297-313  (Phase 1)
        _build_equation_context(data, global_ops, sel, cv)   → run.py:320-431  (Phase 2)
    run_pipeline() calls these the SAME way (run.py:843 and run.py:848). If MIDAS changes either
    function, this wrapper follows automatically.

    ── WHY THERE IS STILL WRAPPER CODE HERE (serialization, not math) ────────────────────
    _build_equation_context() returns an in-memory EquationContext dataclass (run.py:418) whose
    fields are not all JSON-serializable, but dspy.Example fields MUST survive pickling/deep-copy
    across GEPA's optimizer subprocess and DSPy's eval thread pool. So the only wrapper-specific
    work is a SERIALIZATION step: pull the needed fields off eq_ctx and json.dumps() them. We also
    FRONT-LOAD two prompt inputs that run.py builds later at render time (run.py:445
    generate_context_prompt → Prompt 1's weakspot context; run.py:472-474
    get_weakspot_example_output → Prompt 2's example block), because the wrapper's DSPy signatures
    take them as explicit input fields.
    """
    # ── Phase 1: Locate weakspot — run.py:843  selection = _locate_weakspot(data, ws_id) ──
    selection = _locate_weakspot(data, ws_id)   # → WeakspotSelection | None (run.py:297-313)  # run.py:843 selection = _locate_weakspot(data, ws_id)  [IMPORTED from run.py]
    if selection is None:                       # run.py:844 if selection is None
        # run.py:844-845 `if selection is None: continue`. _locate_weakspot already logged the
        # specific reason (instance-not-found vs op-not-found) via run.py's own logger.
        logger.warning("Weakspot %s not found / op missing in %s, skipping", ws_id, fc_name)  # [wrapper-only] trace
        return None                             # run.py:845 continue (skip this weakspot)

    # ── Phase 2: Build equation context — run.py:848  eq_ctx = _build_equation_context(...) ──
    try:                                         # [wrapper-only] guard (run.py calls it directly)
        eq_ctx = _build_equation_context(data, global_ops, selection, const_values)  # run.py:848 eq_ctx = _build_equation_context(data, global_ops, selection, const_values)  [IMPORTED]
    except Exception as e:                       # [wrapper-only] skip weakspots whose Phase-2 graph build fails
        logger.warning("Error processing weakspot %s in %s: %s", ws_id, fc_name, e)  # [wrapper-only] trace
        return None                              # [wrapper-only] skip

    ws_key         = eq_ctx.weakspot_op_key   # run.py:419  # run.py:419 weakspot_op_key
    instance_names = eq_ctx.instance_names    # run.py:425 (output instance names)  # run.py:425 instance_names
    filtered_dict  = eq_ctx.filtered_dict     # run.py:423 (filtered_dict["equation"] = the prompt graph)  # run.py:423 filtered_dict

    # Front-loaded Prompt 1 weakspot context — run.py:445 generate_context_prompt(eq_ctx.filtered_dict)
    # internally renders get_weakspot_context(filtered_dict["equation"], weakspot_op_key).
    weakspot_context_parts = get_weakspot_context(filtered_dict["equation"], ws_key)  # context_prompt.py:55 get_weakspot_context(...) — rendered inside run.py:445 Prompt 1; precomputed here
    weakspot_context_text = ""                   # [wrapper-only] flatten to a string field
    if weakspot_context_parts and isinstance(weakspot_context_parts, list):  # context_prompt.py:65 weakspot_context[0]["text"]
        weakspot_context_text = weakspot_context_parts[0].get("text", "")    # context_prompt.py:65 take [0]["text"]
    elif isinstance(weakspot_context_parts, str):  # [wrapper-only] tolerate str return
        weakspot_context_text = weakspot_context_parts  # [wrapper-only]

    # Front-loaded Prompt 2 example block — run.py:472-474 get_weakspot_example_output(eq_ctx.filtered_dict, ws_key).
    try:                                         # [wrapper-only] guard
        example_explanation_output = get_weakspot_example_output(filtered_dict, ws_key)  # run.py:472-474 get_weakspot_example_output(eq_ctx.filtered_dict, eq_ctx.weakspot_op_key)
    except Exception:                            # [wrapper-only] empty example on failure
        example_explanation_output = ""          # [wrapper-only]

    # ── Serialize the Phase-2 results the DSPy module/metric consume (JSON, deep-copy safe) ──
    # All values come straight off eq_ctx — NO re-computation of the graph math:
    #   eq_ctx.relations        → run.py:429  (Phase-2 var → solvable equation; feeds update_test_cases)
    #   eq_ctx.equation_results → run.py:430  (list[EquationResult]; → merged_equation_map in forward())
    #   build_op_mapping(eq_ctx.dependency_mapping) → run.py:662 (op_map for 2nd compute_unselected_mux_variables;
    #     identical to run.py since eq_ctx.dependency_mapping is exactly run.py:400-402's).
    try:                                         # [wrapper-only] serialize eq_ctx → JSON (run.py keeps eq_ctx live in-process)
        relations_json_val = json.dumps(eq_ctx.relations)  # run.py:429 eq_ctx.relations → JSON (forward() feeds update_test_cases)
        equation_results_json_val = json.dumps([   # run.py:430 eq_ctx.equation_results (list[EquationResult]) → JSON
            {                                       # [wrapper-only] one dict per EquationResult
                "original_equation": er.original_equation,            # run.py:413 EquationResult.original_equation
                "assignment_chain": er.assignment_chain,              # run.py:414 EquationResult.assignment_chain
                "final_substituted_equation": er.final_substituted_equation,  # run.py:415 EquationResult.final_substituted_equation
            }
            for er in eq_ctx.equation_results       # run.py:430 iterate eq_ctx.equation_results
        ])
        op_map_json_val = json.dumps(build_op_mapping(eq_ctx.dependency_mapping))  # run.py:662 build_op_mapping(eq_ctx.dependency_mapping) → JSON (forward 2nd compute_unselected_mux)
        weakspot_op_key_val = ws_key                # run.py:419 weakspot_op_key
    except Exception as e_ser:                      # [wrapper-only] serialization failure → empty fields
        logger.warning("Phase 2 serialization failed for %s in %s: %s", ws_id, fc_name, e_ser)  # [wrapper-only] trace
        relations_json_val = "{}"                   # [wrapper-only]
        equation_results_json_val = "[]"            # [wrapper-only]
        op_map_json_val = "{}"                       # [wrapper-only]
        weakspot_op_key_val = ""                     # [wrapper-only]

    # Phase A3: exclude weakspots with NO solvable equations (degenerate tasks).
    # run.py no-ops these at LLM time (run.py:876 `if len(equation_dict) != 0`); they
    # are not real optimization targets and only dilute the averaged metric.
    if not json.loads(equation_results_json_val):   # [wrapper-only] drop no-equation weakspots (run.py:876 no-ops them at LLM time)
        logger.warning("Weakspot %s in %s has no solvable equations, skipping", ws_id, fc_name)  # [wrapper-only] trace
        return None                                 # [wrapper-only] skip degenerate task

    # Build the dspy.Example
    equation_json_str = json.dumps(filtered_dict["equation"], ensure_ascii=False)  # run.py:423 filtered_dict["equation"] → JSON (Prompt-1/2 equation_json)
    const_values_str = json.dumps(const_values, ensure_ascii=False)  # run.py:796 const_values → JSON (== filtered_dict["const_values"])

    example = dspy.Example(                         # [wrapper-only] package run.py's Phase-1/2 result as a dspy.Example (run.py runs inline, no Example)
        # Identification (metadata)
        fc_name=fc_name,                            # [wrapper-only] metadata
        weakspot_id=ws_id,                          # run.py:837 ws_id (the weakspot being processed)
        leaf_node=leaf_node,                        # run.py:761 leaf_node
        # Inputs for DSPy Module
        equation_json=equation_json_str,            # run.py:423 filtered_dict["equation"] (Prompt-1/2 input)
        const_values=const_values_str,              # run.py:796 const_values (Prompt-1/2 input)
        weakspot_context=weakspot_context_text,     # run.py:445 Prompt-1 weakspot context (via context_prompt.py:55/65)
        instance_names=",".join(instance_names) if instance_names else fc_name,  # run.py:425 eq_ctx.instance_names (Prompt-2 "use this Expected Results")
        raster_time=str(int(step_time_ms * 1000)),  # ms → microseconds (matches run.py dT)  # run.py:784/795 step_time→dT µs (metric metadata; NOT a Prompt-2 arg)
        # FIDELITY: original Prompt 1/2 inputs the wrapper previously omitted
        weakspot_operation=weakspot_op_key_val,            # op key (e.g. "$op39")  # run.py:419 weakspot_op_key (Prompt-1/2 input)
        example_explanation_output=example_explanation_output,  # Prompt 2 example block  # run.py:472-474 get_weakspot_example_output(...)
        # Metadata for metric (not inputs to LLM)
        signalsheet_data=signalsheet_data,          # [wrapper-only] metric input (no run.py equivalent — run.py has no metric)
        model_interface_data=model_interface_data,  # [wrapper-only] metric/DLL input
        dll_path=dll_path,                          # [wrapper-only] metric/DLL input (M7/M8 + forward _get_function_caller)
        # Phase 2 precomputed data for MBDFullPipelineModule
        relations_json=relations_json_val,          # run.py:429 eq_ctx.relations (consumed by forward update_test_cases)
        equation_results_json=equation_results_json_val,  # run.py:430 eq_ctx.equation_results (→ merged_equation_map in forward)
        op_map_json=op_map_json_val,                # run.py:662 build_op_mapping(dependency_mapping) (forward 2nd compute_unselected_mux)
        weakspot_op_key=weakspot_op_key_val,        # run.py:419 weakspot_op_key (metric M4/M7/M8)
    ).with_inputs(                                  # [wrapper-only] mark the 5 DSPy input fields (run.py has no with_inputs)
        "equation_json", "const_values", "weakspot_context",  # [wrapper-only] DSPy input fields (1/3)
        "instance_names", "raster_time"             # [wrapper-only] DSPy input fields (2/3)
        # with_inputs() UNCHANGED — do not add dll_path etc. here
        # (would break MBDTestGenerator which has exactly 5 params, no **kwargs)
    )

    return example                                  # [wrapper-only] one Example == one run.py per-weakspot task


def _example_is_arithmetic(ex: dspy.Example) -> bool:  # [wrapper-only] GEPA split helper — no run.py analog (run.py has no train/val)
    """True if this weakspot has at least one dependency equation with a NON-EMPTY assignment_chain
    — i.e. the equation-solver / retry predictors can actually RUN on it. Function-call weakspots
    (Srv_TrnOnDly / Srv_TrnOffDly / Srv_RampSwitch / getBit / …) carry only empty-chain arg
    relations and return False. GEPA reflects on the solver predictors ONLY via arithmetic examples,
    so the train split must keep some (else solver-reflection turns raise "No valid predictions
    found for any module" and are wasted)."""
    try:                                            # [wrapper-only] guard malformed serialized field
        ers = json.loads(getattr(ex, "equation_results_json", "[]") or "[]")  # [wrapper-only] deserialize Example.equation_results_json (run.py:430 eq_ctx.equation_results, JSON here)
    except (json.JSONDecodeError, TypeError):       # [wrapper-only] bad/missing field → not arithmetic
        return False                                # [wrapper-only]
    return any((er.get("assignment_chain") or "").strip() for er in ers if isinstance(er, dict))  # [wrapper-only] arithmetic ⇔ any non-empty assignment_chain (run.py:414 EquationResult.assignment_chain)


def _rebalance_arith_into_train(train, val, rng, target: int = 2):  # [wrapper-only] keep arithmetic weakspots in train for GEPA solver-reflection — no run.py analog
    """Deterministically swap arithmetic weakspots from val into train (exchanging non-arith ones
    back) until train holds up to `target` of them, or none remain. Size-preserving and seeded, so
    the split stays reproducible — only membership shifts to guarantee solver-reflection signal."""
    is_arith = _example_is_arithmetic               # [wrapper-only] alias the arithmetic-predicate helper (above)
    n_arith_train = sum(1 for e in train if is_arith(e))  # [wrapper-only] arithmetic weakspots already in train
    total_arith = n_arith_train + sum(1 for e in val if is_arith(e))  # [wrapper-only] arithmetic weakspots across both sides
    want = min(target, total_arith)                 # [wrapper-only] can't want more than exist
    if n_arith_train >= want:                       # [wrapper-only] train already satisfies target → no swap
        return train, val                           # [wrapper-only]
    val_arith_idx = [i for i, e in enumerate(val) if is_arith(e)]  # [wrapper-only] candidate arithmetic examples to move val→train
    train_nonarith_idx = [i for i, e in enumerate(train) if not is_arith(e)]  # [wrapper-only] non-arith examples to send back train→val (size-preserving)
    rng.shuffle(val_arith_idx)                      # [wrapper-only] deterministic (seeded rng) pick
    rng.shuffle(train_nonarith_idx)                 # [wrapper-only] deterministic (seeded rng) pick
    k = min(want - n_arith_train, len(val_arith_idx), len(train_nonarith_idx))  # [wrapper-only] #swaps = min(deficit, available on each side)
    train, val = list(train), list(val)             # [wrapper-only] copy so we mutate locally, not caller's lists
    for j in range(k):                              # [wrapper-only] perform the k size-preserving swaps
        ti, vi = train_nonarith_idx[j], val_arith_idx[j]  # [wrapper-only] indices to exchange
        train[ti], val[vi] = val[vi], train[ti]     # [wrapper-only] swap arith into train, non-arith into val
    return train, val                               # [wrapper-only] rebalanced split


def _example_block_type(ex: dspy.Example) -> str:  # [wrapper-only] weakspot block type for stratification — no run.py analog
    """The weakspot op's function/method name (e.g. 'Srv_RampSwitch_out', 'getBit_unsigned_getBit'),
    used to spread DUPLICATED weakspot types across train/val so neither side is blind to a type."""
    try:                                            # [wrapper-only] guard malformed serialized field
        eqj = json.loads(getattr(ex, "equation_json", "{}") or "{}")  # [wrapper-only] deserialize Example.equation_json (run.py:423 filtered_dict["equation"])
        op = eqj.get(getattr(ex, "weakspot_op_key", ""), {}) or {}  # [wrapper-only] weakspot op entry (run.py:419 weakspot_op_key; same lookup as metric M4:810)
        return op.get("function_name") or op.get("class_method_name") or op.get("method_name") or ""  # [wrapper-only] op's fn/method name = stratification key
    except (json.JSONDecodeError, TypeError):       # [wrapper-only] bad/missing field → empty type
        return ""                                   # [wrapper-only]


def _example_is_bit(ex: dspy.Example) -> bool:  # [wrapper-only] bit-operation weakspot flag — no run.py analog
    """True for bit-operation weakspots (getBit/putBit/bitmask). Requirement: bit weakspots must
    appear in BOTH train and val — never all clustered on one side."""
    return "bit" in _example_block_type(ex).lower()  # [wrapper-only] bit-op ⇔ "bit" in block type (getBit/putBit/bitmask)


def _ensure_group_on_both_sides(train, val, pred, rng):  # [wrapper-only] guarantee a weakspot group spans train AND val — no run.py analog
    """Size-preserving: if every example matching `pred` sits on ONE side (and ≥2 exist total),
    swap one matching example to the other side (exchanging a non-matching example back). No-op when
    the group already spans both sides, or has <2 total members (can't span both)."""
    train, val = list(train), list(val)             # [wrapper-only] copy so we mutate locally, not caller's lists
    t_match = [i for i, e in enumerate(train) if pred(e)]  # [wrapper-only] train indices matching the group predicate
    v_match = [i for i, e in enumerate(val) if pred(e)]    # [wrapper-only] val indices matching the group predicate
    if len(t_match) + len(v_match) < 2:             # [wrapper-only] <2 total members → can't span both sides → no-op
        return train, val                           # [wrapper-only]
    if not t_match:              # all matches in val → pull one into train  # [wrapper-only] group absent from train
        t_non = [i for i, e in enumerate(train) if not pred(e)]  # [wrapper-only] non-matching train examples to send back (size-preserving)
        if t_non:                                   # [wrapper-only] need a non-match to exchange
            rng.shuffle(t_non); rng.shuffle(v_match)  # [wrapper-only] deterministic (seeded rng) pick of the swap pair
            ti, vi = t_non[0], v_match[0]           # [wrapper-only] indices to exchange
            train[ti], val[vi] = val[vi], train[ti]  # [wrapper-only] pull one match into train, push a non-match to val
    elif not v_match:            # all matches in train → push one into val  # [wrapper-only] group absent from val
        v_non = [i for i, e in enumerate(val) if not pred(e)]  # [wrapper-only] non-matching val examples to send back (size-preserving)
        if v_non:                                   # [wrapper-only] need a non-match to exchange
            rng.shuffle(v_non); rng.shuffle(t_match)  # [wrapper-only] deterministic (seeded rng) pick of the swap pair
            vi, ti = v_non[0], t_match[0]           # [wrapper-only] indices to exchange
            val[vi], train[ti] = train[ti], val[vi]  # [wrapper-only] push one match into val, pull a non-match to train
    return train, val                               # [wrapper-only] group now spans both sides (or was unfixable)


# [wrapper-only] Human-validated examples PINNED to VAL (PS-GB expert-reviewed TPT specs). Pinning puts
# ground-truth-checked cases in the val set. Rail_TempLim_50.8.0#ws2 has an expert "ok" remark (a CLEAN,
# validated case) and moving it from train makes train/val 5/5. Format: {(fc_name, str(weakspot_id))}.
# Empty set = default stratified split (no pinning). Applied AFTER the arith/bit balancing guarantees.
_FORCE_VAL_EXAMPLES = {("Rail_TempLim_50.8.0", "2")}


def stratified_split_mbd_dataset(   # [wrapper-only] train/val/test split for GEPA — no run.py analog (run.py runs every FC once)
    dataset: List[dspy.Example],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_seed: int = 42,
) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    """Split MBD dataset.

    - test_ratio == 0 → GLOBAL train/val split honoring the exact ratio (e.g. 70/30). With
      many single-example FCs, per-FC stratification would force ~80/20 (every singleton goes
      to train); a global shuffle gives the requested ratio. No test set is created.
    - test_ratio > 0 → stratified by fc_name (each FC has ≥1 example in train).
    """
    rng = random.Random(random_seed)                # [wrapper-only] seeded RNG for a reproducible split (run.py never partitions)

    # ── Global split (no stratification) when no test set is wanted ──────────
    if test_ratio == 0:                             # [wrapper-only] no test set → honor exact train/val ratio globally
        pool = list(dataset)                        # [wrapper-only] copy of all Examples to shuffle
        rng.shuffle(pool)                           # [wrapper-only] seeded global shuffle
        n = len(pool)                               # [wrapper-only] total example count
        if n <= 1:                                  # [wrapper-only] 0/1 examples → all to train, empty val
            train, val = pool, []                   # [wrapper-only]
        else:                                       # [wrapper-only] ≥2 examples → split by ratio, keeping ≥1 each side
            n_train = max(1, min(n - 1, round(n * train_ratio)))  # keep ≥1 in each  # [wrapper-only] clamp to [1, n-1]
            train, val = pool[:n_train], pool[n_train:]  # [wrapper-only] slice into train/val
            # (1) Keep solver-exercising (arithmetic) weakspots in train so GEPA's equation-solver /
            #     retry reflection turns have traces (else "No valid predictions found for any module").
            train, val = _rebalance_arith_into_train(train, val, rng)  # [wrapper-only] guarantee (1)
            # (2) Spread every DUPLICATED weakspot type across BOTH sides (e.g. the two LowpassT,
            #     the two getBit) so neither train nor val is blind to a type it could learn/validate.
            dup_types = {t for t in (_example_block_type(e) for e in pool)  # [wrapper-only] block types that appear ≥2× in the pool
                         if t and sum(1 for e in pool if _example_block_type(e) == t) >= 2}  # [wrapper-only] keep only duplicated (non-empty) types
            for _t in sorted(dup_types):            # [wrapper-only] deterministic order over duplicated types
                train, val = _ensure_group_on_both_sides(  # [wrapper-only] guarantee (2): this type spans both sides
                    train, val, lambda e, t=_t: _example_block_type(e) == t, rng)  # [wrapper-only] predicate binds _t via default arg
            # (3) Guarantee bit-operation weakspots appear in BOTH train and val (never all on one side).
            train, val = _ensure_group_on_both_sides(train, val, _example_is_bit, rng)  # [wrapper-only] guarantee (3)
            # (4) Re-assert the arith-in-train guarantee in case (2)/(3) shifted an arithmetic example.
            train, val = _rebalance_arith_into_train(train, val, rng)  # [wrapper-only] re-assert guarantee (1)
            # (5) Pin human-validated examples to VAL (applied LAST so it overrides the balancing above).
            if _FORCE_VAL_EXAMPLES:                 # [wrapper-only] optional ground-truth pin (e.g. Rail#2 -> val for 5/5)
                _keep, _move = [], []               # [wrapper-only] partition train into stay vs move-to-val
                for _e in train:                    # [wrapper-only] scan train for pinned (fc, ws)
                    (_move if (_e.fc_name, str(_e.weakspot_id)) in _FORCE_VAL_EXAMPLES else _keep).append(_e)  # [wrapper-only]
                if _move:                           # [wrapper-only] only if something matched
                    train, val = _keep, val + _move  # [wrapper-only] move pinned examples train -> val
                    print(f"Split: pinned {[(e.fc_name, e.weakspot_id) for e in _move]} to VAL (human-validated)")  # [wrapper-only] trace
        n_arith_tr = sum(1 for e in train if _example_is_arithmetic(e))  # [wrapper-only] diagnostic counts for the trace below
        n_bit_tr = sum(1 for e in train if _example_is_bit(e))  # [wrapper-only] diagnostic count
        n_bit_va = sum(1 for e in val if _example_is_bit(e))  # [wrapper-only] diagnostic count
        print(f"Split (global, test_ratio=0): train={len(train)} (arith={n_arith_tr}, bit={n_bit_tr}), "  # [wrapper-only] trace
              f"val={len(val)} (bit={n_bit_va}), test=0")  # [wrapper-only] trace
        return train, val, []                       # [wrapper-only] empty test set for the global-split path

    # Group by FC
    by_fc: dict[str, list] = {}                     # [wrapper-only] bucket Examples per fc_name for stratification
    for ex in dataset:                              # [wrapper-only] iterate all Examples
        fc = ex.fc_name                             # [wrapper-only] stratification key = FC name
        by_fc.setdefault(fc, []).append(ex)         # [wrapper-only] append to that FC's bucket

    train, val, test = [], [], []                   # [wrapper-only] accumulate the three splits

    for fc, examples in sorted(by_fc.items()):      # [wrapper-only] deterministic order over FCs
        rng.shuffle(examples)                       # [wrapper-only] seeded shuffle within each FC bucket
        n = len(examples)                           # [wrapper-only] examples for this FC

        if n == 1:                                  # [wrapper-only] singleton FC → train only
            # Single example → train (will be duplicated to val if needed)
            train.extend(examples)                  # [wrapper-only]
        elif n == 2:                                # [wrapper-only] pair FC → one train, one val
            # Two examples → train + val
            train.append(examples[0])               # [wrapper-only]
            val.append(examples[1])                 # [wrapper-only]
        else:                                       # [wrapper-only] ≥3 → ratio split into train/val/test
            n_train = max(1, int(n * train_ratio))  # [wrapper-only] ≥1 train from this FC
            n_val = max(1, int(n * val_ratio))      # [wrapper-only] ≥1 val from this FC
            train.extend(examples[:n_train])        # [wrapper-only] first slice → train
            val.extend(examples[n_train:n_train + n_val])  # [wrapper-only] next slice → val
            test.extend(examples[n_train + n_val:])  # [wrapper-only] remainder → test

    # If test set is empty and we have enough data, move some from val
    if not test and len(val) >= 3:                  # [wrapper-only] ensure a non-empty test set when possible
        rng.shuffle(val)                            # [wrapper-only] seeded pick
        test.append(val.pop())                      # [wrapper-only] move one val example to test

    # If val is still empty, duplicate a few from train
    if not val and train:                           # [wrapper-only] never leave val empty (metric needs a val set)
        val.append(train[-1])                       # [wrapper-only] duplicate one train example into val

    rng.shuffle(train)                              # [wrapper-only] final seeded shuffle of each split
    rng.shuffle(val)                                # [wrapper-only]
    rng.shuffle(test)                               # [wrapper-only]

    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")  # [wrapper-only] trace
    return train, val, test                         # [wrapper-only] the three GEPA splits

# NOTE (resolved 2026-06-18): the per-entry "raster_time" key below is VESTIGIAL — NOTHING reads
# it. load_mbd_dataset() consumes only fc_name / leaf_node / weakspot_ids; the real step time is
# derived from the LEAF-NODE NAME via extract_step_time(leaf_node) (input_loader.py:233-239 —
# regex r"(\d+)ms", default 20.0), exactly as run.py does (run.py:784). So for leaf_node
# "Proc_10ms" the effective step time is 10 ms no matter what "raster_time" says here, and a leaf
# with no "<N>ms" token (e.g. "Proc", "_Proc") yields the 20.0 default. The values are kept only
# as human-readable annotations; the obviously-wrong "Proc_10ms": 100 ones were corrected to 10 so
# the annotation isn't misleading (no runtime effect either way).
def _default_fc_entries() -> List[dict]:  # [wrapper-only] default task subset ≙ a curated run_config.json fc_entries (run.py:230)
    """Default subset of 10-15 tasks for initial development.

    Selected for diversity of weakspot types and manageable size.
    """
    return [                                         # [wrapper-only] curated fc_entries subset (shape ≙ run_config.json fc_entries, run.py:230-233)
        {"fc_name": "InjRly_PwrCtrl_1_1_0", "leaf_node": "Proc_20ms", "weakspot_ids": [0], "raster_time": 20},
        {"fc_name": "GbxFanCoorn_Sp_1_0_2", "leaf_node": "Proc", "weakspot_ids": [2], "raster_time": 100},
        {"fc_name": "oilsepactr_coorr", "leaf_node": "Proc_100ms", "weakspot_ids": [20, 22], "raster_time": 100},
        {"fc_name": "EGSSig_LamSymcp_1_6_0", "leaf_node": "Proc_10ms", "weakspot_ids": [2, 5], "raster_time": 10},
        {"fc_name": "Rail_TempLim_50.8.0", "leaf_node": "_Proc", "weakspot_ids": [0, 1, 2], "raster_time": 100},
        {"fc_name": "moftqptdveh_fil", "leaf_node": "Proc", "weakspot_ids": [0], "raster_time": 100},
        {"fc_name": "AltIO_DD", "leaf_node": "Proc_20ms", "weakspot_ids": [0], "raster_time": 20},
        {"fc_name": "a2mamg", "leaf_node": "Proc_20ms", "weakspot_ids": [0], "raster_time": 20},
        {"fc_name": "GearLvrPosn_Vd", "leaf_node": "Proc_10ms", "weakspot_ids": [0], "raster_time": 10},
        {"fc_name": "IT1A_OdoCalc_1.5.0", "leaf_node": "Proc_1000ms", "weakspot_ids": [0], "raster_time": 1000},
    ]


def all_fc_entries() -> List[dict]:  # [wrapper-only] reads MBD/.../run_config.json fc_entries (the SAME source run.py:230 uses)
    """Dataset weakspots — read from MIDAS's run_config.json (single source of truth).

    Uses exactly the `fc_entries` (FC, leaf, weakspot_ids) defined in
    MBD/weakspot_code_to_run_2/run_config.json, so editing that config updates the
    dataset. Degenerate weakspots (op-not-found / no-equations / leaf-not-found) are
    dropped by the loader, so the valid count is typically smaller than the configured
    count. Falls back to the hardcoded list if run_config.json can't be read.
    """
    cfg_path = os.path.join(_MBD_ROOT, "run_config.json")  # [wrapper-only] the SAME config run.py loads (run.py:230 reads fc_entries from it)
    try:                                            # [wrapper-only] fall back to hardcoded list on any read/parse error
        with open(cfg_path, "r", encoding="utf-8") as f:  # [wrapper-only] open run_config.json
            entries = (json.load(f).get("fc_entries") or [])  # run.py:230 fc_entries = raw.get("fc_entries") (same key/source)
        if entries:                                 # [wrapper-only] non-empty config → use it as the dataset task list
            n_ws = sum(len(e.get("weakspot_ids", [])) for e in entries)  # [wrapper-only] count weakspot_ids for the trace
            print(f"all_fc_entries: {len(entries)} entries / {n_ws} weakspot_ids from run_config.json")  # [wrapper-only] trace
            return entries                          # [wrapper-only] fc_entries straight from run_config.json
    except Exception as e:                          # [wrapper-only] unreadable/invalid config
        print(f"WARNING: could not read run_config.json ({e}); using hardcoded fallback")  # [wrapper-only] trace
    return _hardcoded_all_fc_entries()              # [wrapper-only] fallback task list (below)


# NOTE (resolved 2026-06-18): same as _default_fc_entries — "raster_time" here is VESTIGIAL and
# unread; step time comes from extract_step_time(leaf_node) (input_loader.py:233-239), matching
# run.py:784. This list is only a fallback for when run_config.json can't be read; the live path
# uses all_fc_entries() → run_config.json. "Proc_10ms": 100 annotations corrected to 10 (no effect).
def _hardcoded_all_fc_entries() -> List[dict]:  # [wrapper-only] fallback task list if run_config.json unreadable — no run.py analog
    """Fallback list (kept in sync with run_config.json) if the config can't be read."""
    return [                                         # [wrapper-only] hardcoded fc_entries fallback (shape ≙ run_config.json / run.py:230)
        {"fc_name": "EGSSig_LamSymcp_1_6_0", "leaf_node": "Proc_10ms", "weakspot_ids": [2, 5, 8, 11], "raster_time": 10},
        {"fc_name": "ExhCo_InhbnEngOffCust", "leaf_node": "Proc_10ms", "weakspot_ids": [0, 1, 2, 3], "raster_time": 10},
        {"fc_name": "PFltMon_TBasd_1_3_0", "leaf_node": "Proc", "weakspot_ids": [5, 6, 9, 10], "raster_time": 100},
        {"fc_name": "GbxFanCoorn_Sp_1_0_2", "leaf_node": "Proc", "weakspot_ids": [2], "raster_time": 100},
        {"fc_name": "oilsepactr_coorr", "leaf_node": "Proc_100ms", "weakspot_ids": [20, 22, 23], "raster_time": 100},
        {"fc_name": "gevctl_phaena", "leaf_node": "Proc_10ms", "weakspot_ids": [26, 27, 30], "raster_time": 10},
        {"fc_name": "InjRly_PwrCtrl_1_1_0", "leaf_node": "Proc_20ms", "weakspot_ids": [0], "raster_time": 20},
        {"fc_name": "AltIO_DD", "leaf_node": "Proc_20ms", "weakspot_ids": [0, 1], "raster_time": 20},
        {"fc_name": "moftqptdveh_fil", "leaf_node": "Proc", "weakspot_ids": [0], "raster_time": 100},
        {"fc_name": "Rail_TempLim_50.8.0", "leaf_node": "_Proc", "weakspot_ids": [0, 1, 2], "raster_time": 100},
        {"fc_name": "a2mamg", "leaf_node": "Proc_20ms", "weakspot_ids": [0], "raster_time": 20},
        {"fc_name": "a2mamg", "leaf_node": "Proc_10ms", "weakspot_ids": [0], "raster_time": 10},
        {"fc_name": "CoESS_Dem_Prj", "leaf_node": "Proc_20ms", "weakspot_ids": [2, 5, 11], "raster_time": 20},
        {"fc_name": "IT1A_OdoCalc_1.5.0", "leaf_node": "Proc_1000ms", "weakspot_ids": [0, 1], "raster_time": 1000},
        {"fc_name": "Fans_Trq_P", "leaf_node": "Fans_Trq_Proc", "weakspot_ids": [7], "raster_time": 100},
        {"fc_name": "GbxFanCoorn_Sp_1_0_0", "leaf_node": "proc", "weakspot_ids": [3], "raster_time": 100},
        {"fc_name": "GearLvrPosn_Vd", "leaf_node": "Proc_10ms", "weakspot_ids": [0], "raster_time": 10},
    ]
