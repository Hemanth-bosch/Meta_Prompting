"""StageRunner \u2014 generic stage execution engine for the tournament pipeline.

ROLE IN THE PIPELINE
--------------------
StageRunner is called once per stage (bootstrap / miprov2 / gepa) by
run_tournament.py.  It takes a list of seed CandidateEntry objects (the
top-K from the previous stage) and a stage name, then:

  1. Expands seeds \xd7 optimizer_configs into a work-item list.
     Example: 2 seeds \xd7 1 bootstrap config = 2 work items.

  2. For each work item that hasn\u2019t already been completed (resume-safe),
     spawns an isolated subprocess running optimizer_subprocess.py.
     The subprocess writes the compiled program to disk and exits.

  3. After the subprocess exits, evaluates the saved program in-process
     using evaluate_candidate() which extracts per-sample token stats
     from lm.history (the LM\u2019s own API-reported token counts).

  4. Creates a CandidateEntry and adds it to the shared leaderboard.

  5. Returns all new CandidateEntry objects produced by the stage.

WHY SUBPROCESSES?
-----------------
DSPy optimizers use ThreadPoolExecutors internally.  On Windows/Azure,
nested thread pools (parent process \u2192 optimizer \u2192 Azure HTTP client) cause
race conditions and near-zero metric scores.  Spawning each optimizer as a
fresh subprocess with no inherited state completely eliminates this bug.

WORK ITEM NAMING
----------------
Config names are built as \"{stage}_{cfg_name}_from_{seed_name}\" where
seed_name is the full file stem of the seed program.  With cross-stage
selection enabled, cascaded stages can produce long names like:
  gepa_medium_from_miprov2_medium_..._from_bootstrap_..._from_seed_candidate_0
On Windows (MAX_PATH=260), very long paths may cause FileNotFoundError.
Workaround: keep the tournament output folder path short.

RESUME SAFETY
-------------
The stage_dir contains one _subprocess_config.json and one .json result per
work item.  If a work item\u2019s result file already exists on disk it is reused
(no re-optimization).  Combined with stage_completed.json this means an
interrupted run can be resumed at zero cost for completed work items.
"""

from __future__ import annotations

import copy
import json
import logging
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import create_lm
from evaluation.dca_metric import dca_comparison_metric, dca_comparison_metric_with_feedback
from dca_tournaments.tour_config import TournamentConfig
from dca_tournaments.leaderboard import CandidateEntry, OptimizationLeaderboard
from dca_tournaments.tour_utils import (
    evaluate_candidate,
    save_candidate,
    load_candidate,
    get_program_info,
    TeeLogger,
)

logger = logging.getLogger("dca_tournaments.stage_runner")

# Lock for serialising subprocess stdout streaming so lines from
# concurrent subprocesses don't interleave mid-line in the log.
_output_lock = threading.Lock()


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


class StageExecutionError(RuntimeError):
    """Raised when all optimizer runs in a stage fail.

    Partial failure (some runs succeed, some fail) is tolerated and
    logged as a warning.  Only raised when *every* config-seed pair
    fails and there are no new entries to add to the leaderboard.

    Attributes
    ----------
    stage           Name of the failed stage ("bootstrap", etc.)
    failed_configs  Config names whose subprocesses returned exit-code != 0
    completed_runs  Number of work items that DID succeed
    total_runs      Total work items attempted in this stage
    """

    def __init__(self, stage: str, failed_configs: list[str], completed_runs: int, total_runs: int):
        self.stage = stage
        self.failed_configs = failed_configs
        self.completed_runs = completed_runs
        self.total_runs = total_runs
        failed_preview = ", ".join(failed_configs[:5])
        if len(failed_configs) > 5:
            failed_preview += ", ..."
        super().__init__(
            f"Stage '{stage}' failed for {len(failed_configs)}/{total_runs} optimizer run(s). "
            f"Completed {completed_runs}. Failed configs: {failed_preview}"
        )


class StageRunner:
    """Run all optimization trials for a single stage.

    Parameters
    ----------
    config : TournamentConfig
    train_set : list
        Training examples (used by optimizer.compile).
    val_set : list
        Validation examples (used by optimizer.compile + our post-eval).
    metric : callable
        Per-example metric, defaults to dca_comparison_metric.
    leaderboard : OptimizationLeaderboard
        Shared mutable leaderboard that accumulates results.
    """

    def __init__(
        self,
        config: TournamentConfig,
        train_set: list,
        val_set: list,
        metric=None,
        metric_with_feedback=None,
        leaderboard: OptimizationLeaderboard | None = None,
    ):
        # IMPORTANT: train_set and val_set must be the MINI subsets
        # (mini_train / mini_val from create_mini_dataset).
        # The full val and test sets are NEVER used here \u2014 they are only used
        # in the final re-evaluation step in run_tournament.py after all
        # stages complete.  Using minis keeps each subprocess fast (~minutes)
        # rather than hours.
        self.config = config
        self.train_set = train_set
        self.val_set = val_set
        self.metric = metric or dca_comparison_metric
        self.metric_with_feedback = metric_with_feedback or dca_comparison_metric_with_feedback
        self.leaderboard = leaderboard or OptimizationLeaderboard()

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def get_stage_dir_name(stage: str) -> str:
        stage_order = {
            "seed": 0,
            "bootstrap": 1,
            "miprov2": 2,
            "gepa": 3,
        }
        stage_idx = stage_order.get(stage)
        if stage_idx is None:
            return f"stage_{stage}"
        return f"stage_{stage_idx}_{stage}"

    def run_stage(
        self,
        stage: str,
        seed_entries: list[CandidateEntry],
    ) -> list[CandidateEntry]:
        """Execute one full optimization stage and return new CandidateEntry objects.

        Steps
        -----
        1. Build the list of optimizer configs for this stage.
        2. For each (seed x config) pair:
           a. Check if a completed result already exists on disk (resume-safe).
           b. If not, call _spawn_optimizer() to run the subprocess.
           c. After subprocess exits, load + evaluate the compiled program.
           d. Add CandidateEntry to leaderboard.
        3. Raise StageExecutionError if all work items failed.
        4. Return the new entries produced by this stage only.

        Parameters
        ----------
        stage         One of "bootstrap", "miprov2", "gepa".
        seed_entries  Top-K entries from previous stage used as starting points.
        """
        stage_dir = Path(self.config.output_dir) / self.get_stage_dir_name(stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        log_dir = Path(self.config.log_dir)

        print(f"\n{'='*70}")
        print(f" [{_ts()}] Stage: {stage.upper()}")
        print(f"{'='*70}")
        print(f"  Seeds    : {len(seed_entries)}")
        print(f"  Stage LM : {self.config.get_stage_model(stage)}")
        print(f"  Output   : {stage_dir}")
        if self.config.use_hybrid_metric:
            print(f"  Compile metric : HYBRID ({self.config.hybrid_penalty_mode})")
        else:
            print(f"  Compile metric : raw DCA")
        print(f"  Eval metric    : raw DCA (leaderboard scores)")

        start = time.time()

        with TeeLogger(log_dir, f"stage_{stage}"):
            if stage == "bootstrap":
                entries = self._run_bootstrap_stage(seed_entries, stage_dir)
            elif stage == "miprov2":
                entries = self._run_miprov2_stage(seed_entries, stage_dir)
            elif stage == "gepa":
                entries = self._run_gepa_stage(seed_entries, stage_dir)
            else:
                raise ValueError(f"StageRunner does not handle stage '{stage}'")

        elapsed = time.time() - start

        print(f"\n{'='*70}")
        print(f" Stage {stage.upper()} complete  ({elapsed:.0f}s, {len(entries)} candidates)")
        print(f"{'='*70}")
        if entries:
            entries.sort(key=lambda e: e.val_total_score, reverse=True)
            best = entries[0]
            seed_best = max((e.val_total_score for e in seed_entries), default=None)
            print(f"  Best: {best.config_name}  total={best.val_total_score:.4f}")
            if seed_best is not None:
                print(f"  Improvement over best seed: {best.val_total_score - seed_best:+.4f}")

            # Token summary (per-sample inference cost)
            valid_tok = [e for e in entries if e.val_avg_total_tokens and e.val_avg_total_tokens > 0]
            if valid_tok:
                cheapest = min(valid_tok, key=lambda e: e.val_avg_total_tokens)
                print(f"  Cheapest (avg tok/sample): {cheapest.config_name}  "
                      f"avg_tok/smp={cheapest.val_avg_total_tokens:.0f}  total={cheapest.val_total_score:.4f}")

            print("\n  Stage summary table:")
            for idx, entry in enumerate(entries, 1):
                parent_name = Path(entry.parent_path).stem if entry.parent_path else "—"
                tok_str = f"  avg_tok/smp={entry.val_avg_total_tokens:.0f}" if entry.val_avg_total_tokens and entry.val_avg_total_tokens > 0 else ""
                print(
                    f"    {idx:2d}. {entry.config_name} from {parent_name}  "
                    f"total={entry.val_total_score:.4f}  "
                    f"cls={entry.val_cls_score:.4f}  "
                    f"chg={entry.val_change_score:.4f}{tok_str}"
                )
            print(f"\n  Top candidate per-category: {OptimizationLeaderboard.format_per_category(best)}")

            # Save stage detail JSON for later debugging
            try:
                import json as _json
                from dataclasses import asdict as _asdict
                _detail = {
                    "stage": stage,
                    "elapsed_seconds": round(elapsed, 1),
                    "n_candidates": len(entries),
                    "n_seeds": len(seed_entries),
                    "best_score": round(best.val_total_score, 4),
                    "seed_best_score": round(seed_best, 4) if seed_best is not None else None,
                    "entries": [
                        {
                            "rank": i + 1,
                            "config_name": e.config_name,
                            "val_total_score": round(e.val_total_score, 4),
                            "val_cls_score": round(e.val_cls_score, 4),
                            "val_change_score": round(e.val_change_score, 4),
                            "n_demos": e.n_demos,
                            "instruction_length": e.instruction_length,
                            "val_avg_total_tokens": e.val_avg_total_tokens,
                            "val_avg_input_tokens": e.val_avg_input_tokens,
                            "val_avg_output_tokens": e.val_avg_output_tokens,
                            "val_avg_reasoning_tokens": e.val_avg_reasoning_tokens,
                            "optimization_started_at": (e.metadata or {}).get("optimization_started_at"),
                            "optimization_completed_at": (e.metadata or {}).get("optimization_completed_at"),
                            "optimization_duration_seconds": (e.metadata or {}).get("optimization_duration_seconds"),
                            "compile_duration_seconds": (e.metadata or {}).get("compile_duration_seconds"),
                            "evaluation_duration_seconds": (e.metadata or {}).get("evaluation_duration_seconds", e.val_eval_time_seconds),
                            "parent": e.parent_path,
                            "program_path": e.program_path,
                        }
                        for i, e in enumerate(entries)
                    ],
                }
                _stage_detail_path = stage_dir / f"stage_{stage}_detail.json"
                with open(_stage_detail_path, "w", encoding="utf-8") as _f:
                    _json.dump(_detail, _f, indent=2, ensure_ascii=False, default=str)
                print(f"  Stage detail saved → {_stage_detail_path}")
            except Exception as _e:
                logger.warning("Failed to save stage detail JSON: %s", _e)
        print()

        return self.leaderboard.top_k(
            self.config.top_k, stage=stage,
            criteria=self.config.selection_criteria,
        )

    def _build_bootstrap_configs(self) -> list[dict]:
        """Return normalized Bootstrap config variants (each has its own 'n')."""
        return list(self.config.bootstrap_configs)

    def _build_miprov2_configs(self) -> list[dict]:
        """Return normalized MIPROv2 intensity × model × demo_config combos.

        If ``miprov2_model_name`` is set, it overrides the per-combo model list
        so that every MIPROv2 run uses that single model.
        """
        model_list = self.config.get_miprov2_models()
        return [
            {
                "auto": intensity,
                "model_name": model_name,
                "max_labeled_demos": dc["max_labeled_demos"],
                "max_bootstrapped_demos": dc["max_bootstrapped_demos"],
                "clear_field_descs": self.config.miprov2_clear_field_descs,
            }
            for intensity in self.config.miprov2_intensities
            for model_name in model_list
            for dc in self.config.miprov2_demo_configs
        ]

    def _build_gepa_configs(self) -> list[dict]:
        """Return normalized GEPA config variants from config.gepa_configs."""
        return list(self.config.gepa_configs)

    # ─────────────────────────────────────────────────────────────
    # Subprocess dispatch
    # ─────────────────────────────────────────────────────────────

    def _spawn_optimizer(self, config_dict: dict, stage_dir: Path, config_name: str) -> dict | None:
        """Run optimizer.compile() in a fully isolated subprocess.

        Writes a config JSON, spawns ``optimizer_subprocess.py``, streams its
        output to the current stdout (which is already tee'd to the stage log),
        and returns the optimization-token usage dict when the subprocess
        exits with code 0.

        Benefits of subprocess isolation:
        - Fresh Python interpreter: no inherited DSPy ContextVar state
        - No nested ThreadPoolExecutors: DSPy's internal metric eval works correctly
        - No Azure client race conditions from concurrent parent threads
        - All DSPy internal scores are valid, so Bootstrap picks the best of N,
          MIPROv2 finds real improvements, and GEPA converges normally
        """
        config_path = stage_dir / f"{config_name}_subprocess_config.json"
        token_usage_path = stage_dir / f"{config_name}_optimization_tokens.json"
        payload = dict(config_dict)
        payload["track_tokens"] = self.config.track_tokens
        payload["token_usage_path"] = str(token_usage_path)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        worker_script = Path(__file__).parent / "optimizer_subprocess.py"
        cmd = [sys.executable, str(worker_script), "--config", str(config_path)]

        print(f"    [{_ts()}] Spawning subprocess: {worker_script.name} --config {config_path.name}")

        # Force UTF-8 stdout in the subprocess so emoji/unicode chars
        # (e.g. 📊 in stratified_split_dca_dataset verbose output) don't
        # crash with UnicodeEncodeError on Windows cp1252.
        env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        # Stream subprocess output line-by-line into parent stdout
        # (parent stdout is already tee'd to the stage log file).
        # The lock prevents lines from concurrent subprocesses interleaving.
        for line in proc.stdout:
            with _output_lock:
                sys.stdout.write(f"    | {line}")
                sys.stdout.flush()

        proc.wait()

        if proc.returncode != 0:
            print(f"    [{_ts()}] Subprocess FAILED (exit {proc.returncode}) for {config_name}")
            return None

        print(f"    [{_ts()}] Subprocess OK for {config_name}")
        if token_usage_path.exists():
            try:
                with open(token_usage_path, "r", encoding="utf-8") as f:
                    token_usage = json.load(f)
                if isinstance(token_usage, dict):
                    return {
                        "prompt_tokens": int(token_usage.get("prompt_tokens", 0) or 0),
                        "completion_tokens": int(token_usage.get("completion_tokens", 0) or 0),
                        "total_tokens": int(token_usage.get("total_tokens", 0) or 0),
                    }
            except Exception:
                logger.exception("Failed to read optimizer token usage for %s", config_name)
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _existing_stage_entry_map(self, stage: str) -> dict[str, CandidateEntry]:
        return {
            entry.config_name: entry
            for entry in self.leaderboard.stage_entries(stage)
        }

    # ─────────────────────────────────────────────────────────────
    # Bootstrap (Stage 1)
    # ─────────────────────────────────────────────────────────────

    def _run_bootstrap_stage(
        self,
        seed_entries: list[CandidateEntry],
        stage_dir: Path,
    ) -> list[CandidateEntry]:
        """Run Bootstrap with multiple demo configs × seeds (parallel)."""
        cfg = self.config
        entries: list[CandidateEntry] = []

        configs = self._build_bootstrap_configs()
        max_workers = cfg.max_parallel_optimizations
        stage_model = cfg.get_stage_model("bootstrap")
        print(f"  [{_ts()}] Bootstrap configs : {len(configs)}")
        print(f"  Stage model       : {stage_model}")
        if cfg.bootstrap_stop_at_score is not None:
            print(f"  Stop at score     : {cfg.bootstrap_stop_at_score}")
        print(f"  Parallel workers  : {max_workers}")
        print(f"  Work items        : {len(configs)} configs × {len(seed_entries)} seeds")

        # Build work items
        work_items = []
        existing_entries = self._existing_stage_entry_map("bootstrap")
        for seed_entry in seed_entries:
            seed_name = Path(seed_entry.program_path).stem
            for bc in configs:
                config_name = f"bootstrap_{bc['name']}_from_{seed_name}"
                if config_name in existing_entries:
                    print(f"    [{_ts()}] Reusing existing bootstrap result: {config_name}")
                    entries.append(existing_entries[config_name])
                    continue
                work_items.append((seed_entry, bc, config_name))

        failed_configs: list[str] = []

        def _bootstrap_worker(seed_entry, bc, config_name):
            """Single bootstrap run — compile() in isolated subprocess, eval in-process."""
            print(f"\n  [{_ts()}] ── {config_name} ──")
            worker_started_at = datetime.now().isoformat(timespec="seconds")
            worker_t0 = time.time()

            output_path = stage_dir / f"{config_name}_compiled.json"

            worker_cfg = {
                "stage": "bootstrap",
                "seed_path": seed_entry.program_path,
                "output_path": str(output_path),
                "use_cot": cfg.use_cot,
                "instructions_only": cfg.instructions_only,
                "pairs_dir": cfg.pairs_dir,
                "gt_dir": cfg.gt_dir,
                "split_seed": cfg.split_seed,
                "train_ratio": cfg.train_ratio,
                "val_ratio": cfg.val_ratio,
                "test_ratio": cfg.test_ratio,
                "mini_train_size": cfg.mini_train_size,
                "mini_val_size": cfg.mini_val_size,
                "student_model_name": stage_model,
                "optimization_temperature": cfg.optimization_temperature,
                "bootstrap_cfg": bc,
                "bootstrap_num_threads": cfg.bootstrap_num_threads,
                "bootstrap_teacher_model": cfg.bootstrap_teacher_model,
                "bootstrap_stop_at_score": cfg.bootstrap_stop_at_score,
                # Hybrid metric config (applied inside compile() only)
                "use_hybrid_metric": cfg.use_hybrid_metric,
                "hybrid_score_weight": cfg.hybrid_score_weight,
                "hybrid_length_weight": cfg.hybrid_length_weight,
                "hybrid_max_prompt_tokens": cfg.hybrid_max_prompt_tokens,
                "hybrid_penalty_mode": cfg.hybrid_penalty_mode,
                "hybrid_target_tokens": cfg.hybrid_target_tokens,
                "hybrid_penalty_steepness": cfg.hybrid_penalty_steepness,
            }

            token_data = self._spawn_optimizer(worker_cfg, stage_dir, config_name)
            if token_data is None:
                raise RuntimeError(f"Bootstrap subprocess failed for {config_name}")

            compiled = load_candidate(str(output_path), use_cot=cfg.use_cot)
            print(f"    [{_ts()}] compile() subprocess done in {time.time() - worker_t0:.1f}s")

            entry = self._evaluate_and_save(
                compiled, config_name, "bootstrap",
                seed_entry.program_path, stage_dir,
                metadata={"model": stage_model, **bc},
                token_data=token_data,
                timing_data={
                    "optimization_started_at": worker_started_at,
                    "compile_duration_seconds": round(time.time() - worker_t0, 1),
                    "optimizer_subprocess_completed_at": datetime.now().isoformat(timespec="seconds"),
                    "_optimization_started_monotonic": worker_t0,
                },
            )
            return entry

        if max_workers <= 1:
            # Sequential fallback
            for seed_entry, bc, config_name in work_items:
                try:
                    entry = _bootstrap_worker(seed_entry, bc, config_name)
                    entries.append(entry)
                except Exception as exc:
                    print(f"    [ERROR] {config_name}: {exc}")
                    logger.exception("Bootstrap run failed: %s", config_name)
                    failed_configs.append(config_name)
        else:
            # Parallel execution — uses a continuous pool (all items submitted at once,
            # max_workers controls concurrency). This is intentionally NOT batch-wait-batch
            # because continuous pooling has better throughput (no idle time between batches).
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_name = {}
                for seed_entry, bc, config_name in work_items:
                    fut = pool.submit(_bootstrap_worker, seed_entry, bc, config_name)
                    future_to_name[fut] = config_name

                for fut in as_completed(future_to_name):
                    config_name = future_to_name[fut]
                    try:
                        entry = fut.result()
                        entries.append(entry)
                    except Exception as exc:
                        print(f"    [ERROR] {config_name}: {exc}")
                        logger.exception("Bootstrap run failed: %s", config_name)
                        failed_configs.append(config_name)

        if failed_configs:
            raise StageExecutionError(
                "bootstrap",
                failed_configs,
                completed_runs=len(entries),
                total_runs=len(entries) + len(failed_configs),
            )

        return entries

    # ─────────────────────────────────────────────────────────────
    # MIPROv2 (Stage 2)
    # ─────────────────────────────────────────────────────────────

    def _run_miprov2_stage(
        self,
        seed_entries: list[CandidateEntry],
        stage_dir: Path,
    ) -> list[CandidateEntry]:
        """Run MIPROv2 with cross-product of intensities × models × seeds (parallel)."""
        cfg = self.config
        entries: list[CandidateEntry] = []

        combos = self._build_miprov2_configs()
        max_workers = cfg.max_parallel_optimizations
        stage_model = cfg.get_stage_model("miprov2")
        print(f"  [{_ts()}] MIPROv2 combos: {len(combos)} (intensity × model × demo_config)")
        print(f"  Stage model       : {stage_model}")
        for combo in combos:
            print(f"    - {combo['auto']} / {combo['model_name']}  d{combo['max_labeled_demos']}b{combo['max_bootstrapped_demos']}")
        print(f"  Parallel workers  : {max_workers}")
        print(f"  Work items        : {len(combos)} combos × {len(seed_entries)} seeds = {len(combos) * len(seed_entries)} runs")

        # Build work items
        work_items = []
        existing_entries = self._existing_stage_entry_map("miprov2")
        for seed_entry in seed_entries:
            seed_name = Path(seed_entry.program_path).stem
            for combo in combos:
                intensity = combo["auto"]
                model_name = combo["model_name"]
                max_labeled = combo["max_labeled_demos"]
                max_bootstrapped = combo["max_bootstrapped_demos"]
                model_tag = model_name.split("/")[-1]
                config_name = f"miprov2_{intensity}_{model_tag}_d{max_labeled}b{max_bootstrapped}_from_{seed_name}"
                if config_name in existing_entries:
                    print(f"    [{_ts()}] Reusing existing MIPROv2 result: {config_name}")
                    entries.append(existing_entries[config_name])
                    continue
                work_items.append((seed_entry, combo, config_name))

        failed_configs: list[str] = []

        def _miprov2_worker(seed_entry, combo, config_name):
            """Single MIPROv2 run — compile() in isolated subprocess, eval in-process."""
            intensity = combo["auto"]
            model_name = combo["model_name"]
            max_labeled = combo["max_labeled_demos"]
            max_bootstrapped = combo["max_bootstrapped_demos"]

            print(f"\n  [{_ts()}] ── {config_name} ──")
            worker_started_at = datetime.now().isoformat(timespec="seconds")
            worker_t0 = time.time()

            output_path = stage_dir / f"{config_name}_compiled.json"

            worker_cfg = {
                "stage": "miprov2",
                "seed_path": seed_entry.program_path,
                "output_path": str(output_path),
                "use_cot": cfg.use_cot,
                "instructions_only": cfg.instructions_only,
                "pairs_dir": cfg.pairs_dir,
                "gt_dir": cfg.gt_dir,
                "split_seed": cfg.split_seed,
                "train_ratio": cfg.train_ratio,
                "val_ratio": cfg.val_ratio,
                "test_ratio": cfg.test_ratio,
                "mini_train_size": cfg.mini_train_size,
                "mini_val_size": cfg.mini_val_size,
                "student_model_name": model_name,
                "optimization_temperature": cfg.optimization_temperature,
                "miprov2_cfg": combo,
                "miprov2_num_threads": cfg.miprov2_num_threads,
                "miprov2_prompt_model": cfg.miprov2_prompt_model,
                "miprov2_prompt_model_reasoning_effort": cfg.miprov2_prompt_model_reasoning_effort,
                "miprov2_minibatch_size": cfg.miprov2_minibatch_size,
                "miprov2_metric_threshold": cfg.miprov2_metric_threshold,
                "miprov2_teacher_model": cfg.miprov2_teacher_model,
                # Hybrid metric config (applied inside compile() only)
                "use_hybrid_metric": cfg.use_hybrid_metric,
                "hybrid_score_weight": cfg.hybrid_score_weight,
                "hybrid_length_weight": cfg.hybrid_length_weight,
                "hybrid_max_prompt_tokens": cfg.hybrid_max_prompt_tokens,
                "hybrid_penalty_mode": cfg.hybrid_penalty_mode,
                "hybrid_target_tokens": cfg.hybrid_target_tokens,
                "hybrid_penalty_steepness": cfg.hybrid_penalty_steepness,
            }

            token_data = self._spawn_optimizer(worker_cfg, stage_dir, config_name)
            if token_data is None:
                raise RuntimeError(f"MIPROv2 subprocess failed for {config_name}")

            compiled = load_candidate(str(output_path), use_cot=cfg.use_cot)
            print(f"    [{_ts()}] compile() subprocess done in {time.time() - worker_t0:.1f}s")

            entry = self._evaluate_and_save(
                compiled, config_name, "miprov2",
                seed_entry.program_path, stage_dir,
                metadata={"intensity": intensity, "model": model_name,
                          "max_labeled_demos": max_labeled,
                          "max_bootstrapped_demos": max_bootstrapped},
                token_data=token_data,
                timing_data={
                    "optimization_started_at": worker_started_at,
                    "compile_duration_seconds": round(time.time() - worker_t0, 1),
                    "optimizer_subprocess_completed_at": datetime.now().isoformat(timespec="seconds"),
                    "_optimization_started_monotonic": worker_t0,
                },
            )
            return entry

        if max_workers <= 1:
            # Sequential fallback
            for seed_entry, combo, config_name in work_items:
                try:
                    entry = _miprov2_worker(seed_entry, combo, config_name)
                    entries.append(entry)
                except Exception as exc:
                    print(f"    [ERROR] {config_name}: {exc}")
                    logger.exception("MIPROv2 run failed: %s", config_name)
                    failed_configs.append(config_name)
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_name = {}
                for seed_entry, combo, config_name in work_items:
                    fut = pool.submit(_miprov2_worker, seed_entry, combo, config_name)
                    future_to_name[fut] = config_name

                for fut in as_completed(future_to_name):
                    config_name = future_to_name[fut]
                    try:
                        entry = fut.result()
                        entries.append(entry)
                    except Exception as exc:
                        print(f"    [ERROR] {config_name}: {exc}")
                        logger.exception("MIPROv2 run failed: %s", config_name)
                        failed_configs.append(config_name)

        if failed_configs:
            raise StageExecutionError(
                "miprov2",
                failed_configs,
                completed_runs=len(entries),
                total_runs=len(entries) + len(failed_configs),
            )

        return entries

    # ─────────────────────────────────────────────────────────────
    # GEPA (Stage 3)
    # ─────────────────────────────────────────────────────────────

    def _run_gepa_stage(
        self,
        seed_entries: list[CandidateEntry],
        stage_dir: Path,
    ) -> list[CandidateEntry]:
        """Run GEPA (instruction optimization) for each seed × GEPA config (parallel)."""
        cfg = self.config
        entries: list[CandidateEntry] = []

        configs = self._build_gepa_configs()
        stage_model = cfg.get_stage_model("gepa")
        max_workers = cfg.max_parallel_optimizations

        print(f"  [{_ts()}] GEPA configs   : {len(configs)}")
        print(f"  Stage model       : {stage_model}")
        print(f"  Reflection images : {cfg.gepa_reflection_images}")
        print(f"  Parallel workers  : {max_workers}")
        print(f"  Work items        : {len(configs)} configs × {len(seed_entries)} seeds = {len(configs) * len(seed_entries)} runs")

        # Note: gepa_utils patch is applied inside the subprocess (optimizer_subprocess.py)

        # Build work items
        work_items = []
        existing_entries = self._existing_stage_entry_map("gepa")
        for seed_entry in seed_entries:
            seed_name = Path(seed_entry.program_path).stem
            for gepa_cfg in configs:
                config_name = f"gepa_{gepa_cfg.get('name', gepa_cfg['auto'])}_from_{seed_name}"
                if config_name in existing_entries:
                    print(f"    [{_ts()}] Reusing existing GEPA result: {config_name}")
                    entries.append(existing_entries[config_name])
                    continue
                work_items.append((seed_entry, gepa_cfg, config_name))

        failed_configs: list[str] = []

        def _gepa_worker(seed_entry, gepa_cfg, config_name):
            """Single GEPA run — compile() in isolated subprocess, eval in-process."""
            print(f"\n  [{_ts()}] ── {config_name} ──")
            worker_started_at = datetime.now().isoformat(timespec="seconds")
            worker_t0 = time.time()

            output_path = stage_dir / f"{config_name}_compiled.json"

            worker_cfg = {
                "stage": "gepa",
                "seed_path": seed_entry.program_path,
                "output_path": str(output_path),
                "use_cot": cfg.use_cot,
                "instructions_only": cfg.instructions_only,
                "pairs_dir": cfg.pairs_dir,
                "gt_dir": cfg.gt_dir,
                "split_seed": cfg.split_seed,
                "train_ratio": cfg.train_ratio,
                "val_ratio": cfg.val_ratio,
                "test_ratio": cfg.test_ratio,
                "mini_train_size": cfg.mini_train_size,
                "mini_val_size": cfg.mini_val_size,
                "student_model_name": stage_model,
                "optimization_temperature": cfg.optimization_temperature,
                "gepa_cfg": gepa_cfg,
                "gepa_num_threads": cfg.gepa_num_threads,
                "gepa_reflection_model": cfg.gepa_reflection_model,
                "gepa_reflection_images": cfg.gepa_reflection_images,
                "gepa_reflection_image_detail": cfg.gepa_reflection_image_detail,
                "gepa_reflection_max_images": cfg.gepa_reflection_max_images,
                "gepa_reflection_minibatch_size": cfg.gepa_reflection_minibatch_size,
                "gepa_add_format_failure_as_feedback": cfg.gepa_add_format_failure_as_feedback,
                "gepa_perfect_score": cfg.gepa_perfect_score,
                "gepa_lowest_score": cfg.gepa_lowest_score,
                "gepa_candidate_selection_strategy": cfg.gepa_candidate_selection_strategy,
                "gepa_train_size": cfg.gepa_train_size,
                "gepa_val_size": cfg.gepa_val_size,
                # Hybrid metric config (applied inside compile() only)
                "use_hybrid_metric": cfg.use_hybrid_metric,
                "hybrid_score_weight": cfg.hybrid_score_weight,
                "hybrid_length_weight": cfg.hybrid_length_weight,
                "hybrid_max_prompt_tokens": cfg.hybrid_max_prompt_tokens,
                "hybrid_penalty_mode": cfg.hybrid_penalty_mode,
                "hybrid_target_tokens": cfg.hybrid_target_tokens,
                "hybrid_penalty_steepness": cfg.hybrid_penalty_steepness,
            }

            token_data = self._spawn_optimizer(worker_cfg, stage_dir, config_name)
            if token_data is None:
                raise RuntimeError(f"GEPA subprocess failed for {config_name}")

            compiled = load_candidate(str(output_path), use_cot=cfg.use_cot)
            print(f"    [{_ts()}] compile() subprocess done in {time.time() - worker_t0:.1f}s")

            entry = self._evaluate_and_save(
                compiled, config_name, "gepa",
                seed_entry.program_path, stage_dir,
                metadata={"model": stage_model, **gepa_cfg},
                token_data=token_data,
                timing_data={
                    "optimization_started_at": worker_started_at,
                    "compile_duration_seconds": round(time.time() - worker_t0, 1),
                    "optimizer_subprocess_completed_at": datetime.now().isoformat(timespec="seconds"),
                    "_optimization_started_monotonic": worker_t0,
                },
            )
            return entry

        if max_workers <= 1:
            for seed_entry, gepa_cfg, config_name in work_items:
                try:
                    entry = _gepa_worker(seed_entry, gepa_cfg, config_name)
                    entries.append(entry)
                except Exception as exc:
                    print(f"    [ERROR] {config_name}: {exc}")
                    logger.exception("GEPA run failed: %s", config_name)
                    failed_configs.append(config_name)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_name = {}
                for seed_entry, gepa_cfg, config_name in work_items:
                    fut = pool.submit(_gepa_worker, seed_entry, gepa_cfg, config_name)
                    future_to_name[fut] = config_name

                for fut in as_completed(future_to_name):
                    config_name = future_to_name[fut]
                    try:
                        entry = fut.result()
                        entries.append(entry)
                    except Exception as exc:
                        print(f"    [ERROR] {config_name}: {exc}")
                        logger.exception("GEPA run failed: %s", config_name)
                        failed_configs.append(config_name)

        if failed_configs:
            raise StageExecutionError(
                "gepa",
                failed_configs,
                completed_runs=len(entries),
                total_runs=len(entries) + len(failed_configs),
            )

        return entries

    # ─────────────────────────────────────────────────────────────
    # Shared helpers
    # ─────────────────────────────────────────────────────────────

    def _load_student(self, seed_entry: CandidateEntry):
        """Load and deep-copy a student program from a seed entry.

        When ``instructions_only=True``, enforces P1 on load: any non-empty
        field descriptions are absorbed into the instruction docstring and
        cleared.  This guarantees every optimizer receives a student where
        ALL guidance lives in the instruction — no matter what the saved
        program contained.
        """
        program = load_candidate(
            seed_entry.program_path,
            use_cot=self.config.use_cot,
        )
        student = copy.deepcopy(program)

        # ── P1 enforcement on load ──────────────────────────────
        if self.config.instructions_only:
            for predictor in student.predictors():
                sig = predictor.signature
                desc_parts = []
                for fname, finfo in sig.fields.items():
                    desc = ""
                    if finfo.json_schema_extra and isinstance(finfo.json_schema_extra, dict):
                        desc = finfo.json_schema_extra.get("desc", "")
                    if desc:
                        desc_parts.append(f"- {fname}: {desc}")
                if desc_parts:
                    merged = "\n".join(desc_parts)
                    current_doc = sig.__doc__ or ""
                    new_doc = (
                        f"{current_doc}\n\nField guidance:\n{merged}"
                        if current_doc
                        else f"Field guidance:\n{merged}"
                    )
                    predictor.signature = sig.with_instructions(new_doc)
                    logger.info(
                        "P1 enforced on load (%s): absorbed %d non-empty field desc(s) into instruction",
                        seed_entry.config_name, len(desc_parts),
                    )
                # Always clear field descs — even if none were non-empty,
                # this is a no-op safety net.
                for fname in list(predictor.signature.fields.keys()):
                    predictor.signature = predictor.signature.with_updated_fields(fname, desc="")

        return student

    def _get_reflection_lm(self):
        """Create the reflection LM (for GEPA) if configured."""
        model_name = (
            self.config.gepa_reflection_model
            or self.config.reflection_model_name
            or self.config.student_model_name
        )
        refl_kwargs = {"cache": False}
        if self.config.optimization_temperature is not None:
            refl_kwargs["temperature"] = self.config.optimization_temperature
        return create_lm(model_name, **refl_kwargs)

    @staticmethod
    def _field_descs_for_signature(signature) -> dict[str, str]:
        return {
            fname: ((finfo.json_schema_extra or {}).get("desc", ""))
            for fname, finfo in signature.fields.items()
        }

    def _snapshot_field_descs(self, program) -> list[dict[str, str]]:
        return [self._field_descs_for_signature(predictor.signature) for predictor in program.predictors()]

    def _normalize_compiled_field_descs(
        self,
        compiled_program,
        original_field_descs: list[dict[str, str]],
        stage_name: str,
    ) -> None:
        predictors = list(compiled_program.predictors())

        # When instructions_only=True, any field descriptions that the optimizer
        # wrote are merged INTO the instruction docstring and then cleared.
        # This ensures all guidance lives in the docstring per P1.
        if self.config.instructions_only:
            for predictor in predictors:
                sig = predictor.signature
                field_desc_parts = []
                for fname, finfo in sig.fields.items():
                    desc = finfo.json_schema_extra.get("desc", "") if finfo.json_schema_extra else ""
                    if desc:
                        field_desc_parts.append(f"- {fname}: {desc}")
                if field_desc_parts:
                    merged = "\n".join(field_desc_parts)
                    current_doc = sig.__doc__ or ""
                    new_doc = (
                        f"{current_doc}\n\nField guidance (from {stage_name.upper()}):\n{merged}"
                        if current_doc
                        else f"Field guidance (from {stage_name.upper()}):\n{merged}"
                    )
                    predictor.signature = sig.with_instructions(new_doc)
                for fname in list(predictor.signature.fields.keys()):
                    predictor.signature = predictor.signature.with_updated_fields(fname, desc="")
            return

        restored_count = 0
        for predictor, snapshot in zip(predictors, original_field_descs):
            current_descs = self._field_descs_for_signature(predictor.signature)
            for fname, original_desc in snapshot.items():
                if current_descs.get(fname, "") != original_desc:
                    predictor.signature = predictor.signature.with_updated_fields(fname, desc=original_desc)
                    restored_count += 1
        if restored_count:
            print(f"    [DESC] Restored {restored_count} original field desc(s) after {stage_name.upper()}")

    def _evaluate_and_save(
        self,
        compiled_program,
        config_name: str,
        stage: str,
        parent_path: str,
        stage_dir: Path,
        metadata: dict | None = None,
        token_data: dict | None = None,
        timing_data: dict | None = None,
    ) -> CandidateEntry:
        """Evaluate a compiled program, save it, and create a CandidateEntry.

        Token accounting
        ~~~~~~~~~~~~~~~~
        ``token_data`` = tokens consumed by the **optimization** step.
        This method tracks the **evaluation** tokens separately.

        On the ``CandidateEntry`` we store **eval-only** tokens (the per-
        inference cost that is relevant for Pareto selection: score vs cost).
        Both are saved in the metadata sidecar so nothing is lost.
        """
        print(f"    [{_ts()}] Evaluating {config_name}...")
        eval_t0 = time.time()

        optimization_td = dict(token_data or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        timing_md = dict(timing_data or {})
        optimization_started_monotonic = timing_md.pop("_optimization_started_monotonic", None)

        # Create a dedicated LM for this evaluation so that concurrent
        # _evaluate_and_save calls (max_parallel > 1) each have their own
        # lm.history — preventing interleaved entries and wrong token counts.
        # Stage-to-stage promotion is always judged on the overall student model.
        # Post-stage evaluation always uses raw DCA metric (no hybrid penalty).
        # Hybrid penalty is only applied during optimizer compile() in the subprocess.
        eval_lm = create_lm(self.config.student_model_name, cache=False)
        eval_result = evaluate_candidate(
            compiled_program, self.val_set, dca_comparison_metric,
            self.config.eval_num_threads, lm=eval_lm,
        )
        info = get_program_info(compiled_program)

        # Per-sample avg token stats from lm.history (input+output+reasoning)
        avg_total_tok     = eval_result.get("avg_total_tokens")
        avg_input_tok     = eval_result.get("avg_input_tokens")
        avg_output_tok    = eval_result.get("avg_output_tokens")
        avg_reasoning_tok = eval_result.get("avg_reasoning_tokens")
        eval_time_s       = eval_result.get("eval_time_seconds")
        eval_duration_seconds = round(time.time() - eval_t0, 1)
        optimization_completed_at = datetime.now().isoformat(timespec="seconds")
        optimization_duration_seconds = (
            round(time.time() - optimization_started_monotonic, 1)
            if optimization_started_monotonic is not None
            else None
        )
        timing_md.update({
            "optimization_completed_at": optimization_completed_at,
            "optimization_duration_seconds": optimization_duration_seconds,
            "evaluation_duration_seconds": eval_duration_seconds,
            "stage_eval_model": self.config.student_model_name,
        })

        program_path = stage_dir / f"{config_name}.json"
        save_candidate(compiled_program, program_path, metadata={
            "stage": stage,
            "config_name": config_name,
            "parent": parent_path,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "eval_result": eval_result,
            "optimization_token_usage": optimization_td,
            **timing_md,
            **(metadata or {}),
        })

        tok_str = (f"  avg_tok/sample={avg_total_tok:.0f}"
                   f"(in={avg_input_tok:.0f} out={avg_output_tok:.0f})"
                   if avg_total_tok is not None else "")
        opt_tok_str = f"  opt_tokens={optimization_td['total_tokens']:,}" if optimization_td.get("total_tokens", 0) > 0 else ""
        print(f"    [{_ts()}] total={eval_result['total_score']:.4f}  "
              f"cls={eval_result['cls_score']:.4f}  "
              f"chg={eval_result['change_score']:.4f}  "
              f"demos={info['n_demos']}  "
              f"instr_len={info['instruction_length']}{tok_str}{opt_tok_str}  "
              f"(eval took {time.time() - eval_t0:.1f}s)")

        # CandidateEntry stores per-sample avg inference tokens for Pareto selection.
        # val_avg_total_tokens = avg(input + output + reasoning) per val-set sample —
        # this is what the end user pays per inference call in production.
        entry = CandidateEntry(
            program_path=str(program_path),
            stage=stage,
            config_name=config_name,
            parent_path=parent_path,
            val_cls_score=eval_result["cls_score"],
            val_change_score=eval_result["change_score"],
            val_total_score=eval_result["total_score"],
            per_category_scores=eval_result["per_category_scores"],
            n_demos=info["n_demos"],
            instruction_length=info["instruction_length"],
            timestamp=datetime.now().isoformat(timespec="seconds"),
            # Optimization-phase cost — kept for diagnostic/budget logging only
            prompt_tokens=optimization_td.get("prompt_tokens", 0),
            completion_tokens=optimization_td.get("completion_tokens", 0),
            total_tokens=optimization_td.get("total_tokens", 0),
            # Per-sample inference cost — used for Pareto selection
            val_avg_total_tokens=avg_total_tok,
            val_avg_input_tokens=avg_input_tok,
            val_avg_output_tokens=avg_output_tok,
            val_avg_reasoning_tokens=avg_reasoning_tok,
            val_eval_time_seconds=eval_time_s,
            metadata={
                "stage": stage,
                "config_name": config_name,
                "parent": parent_path,
                "eval_result": eval_result,
                "optimization_token_usage": optimization_td,
                **timing_md,
                **(metadata or {}),
            },
        )

        self.leaderboard.add(entry)
        return entry
