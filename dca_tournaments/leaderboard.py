"""OptimizationLeaderboard \u2014 tracks all candidate programs across stages.

ROLE IN THE PIPELINE
--------------------
The leaderboard is the central registry of every prompt candidate produced
during a tournament run.  It is:
  - Created once in run_tournament.py and passed to every StageRunner
  - Thread-safe: guarded by a reentrant lock (multiple eval threads can add
    entries concurrently during a stage)
  - Auto-saved to leaderboard.json after every insertion so progress is never
    lost on crash / Ctrl+C
  - Reloaded from disk when resuming an interrupted run

WHAT IS STORED (CandidateEntry)
--------------------------------
For every compiled program produced by an optimizer:
  - Path to the saved .json program on disk
  - Stage name and config name (e.g. \"miprov2_medium_gpt-4.1-mini_d3b5\")
  - Val set scores: total, cls, change, per-category
  - Token usage stats: avg input/output/reasoning tokens per sample
  - Metadata: model name, timestamps, optimizer params

SELECTION CRITERIA
------------------
  score_only   \u2014 rank by val_total_score descending (default)
  pareto       \u2014 Pareto-optimal front (maximize score, minimize tokens),
                 then fill with next-best by score
  tokens_only  \u2014 rank by val_avg_total_tokens ascending (cheapest first)

CROSS-STAGE SELECTION (--cross-stage-selection)
------------------------------------------------
  When enabled, top_k() draws from ALL stages combined, not just the last.
  This means a bootstrap result can \"seed\" GEPA even after MIPROv2 if the
  bootstrap score is still higher than any MIPROv2 output.

EXAMPLE USAGE
-------------
  lb = OptimizationLeaderboard(save_path=\".cache/tournament/t_1/leaderboard.json\")
  lb.add(entry)             # inserts, re-sorts, auto-saves
  best = lb.best()          # top entry by score
  top2 = lb.top_k(2)        # top-2 entries
  lb.print_table()          # console table
  lb.export_to_excel(path)  # Excel file for offline analysis
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Optional, List

# Canonical execution order for stages.  Used by get_stages() so that
# improvement_over_prev in get_stage_summary() always compares in the
# correct direction (not score order).
_CANONICAL_STAGE_ORDER = (
    "seed", "seed_baseline", "bootstrap", "miprov2", "gepa", "external"
)


@dataclass
class CandidateEntry:
    """A single candidate program and all its evaluation results.

    Created by StageRunner after each optimizer subprocess completes and its
    output is evaluated on mini_val.  Also created by SeedPromptGenerator
    (stage 0) and by run_tournament.py for the seed-from-file path.

    Fields
    ------
    program_path          Path to the saved DSPy .json state on disk.
    stage                 Which pipeline stage produced this:
                          seed / bootstrap / miprov2 / gepa
    config_name           Human-readable name encoding stage + model + demo
                          counts, e.g. \"miprov2_medium_gpt-4.1-mini_d3b5_from_seed_candidate_0\"
    parent_path           program_path of the seed that was optimized to
                          produce this entry.  None for Stage-0 seeds.
    val_total_score       0.6*cls + 0.4*change on mini_val.
                          Updated to full-val score during final re-evaluation.
    val_avg_total_tokens  Average (input+output) tokens per val sample.
                          Used for Pareto/token selection and pareto_plot.png.
    metadata              Dict with optimizer params, model name, timestamps.
    """

    program_path: str  # path to saved .json (dump_state)
    stage: str  # "seed" / "bootstrap" / "miprov2" / "gepa"
    config_name: str  # e.g. "bootstrap_n16_d3b3", "miprov2_light_gpt41mini"
    parent_path: Optional[str]  # seed that produced this candidate (None for Stage 0)
    val_cls_score: float
    val_change_score: float
    val_total_score: float
    per_category_scores: dict = field(default_factory=dict)  # {A1: 0.95, ...}
    n_demos: int = 0
    instruction_length: int = 0
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)  # optimizer params, model, etc.

    # Token usage tracking (optimization phase — from dspy.track_usage())
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Final val-set evaluation stats (per-sample averages over full val set)
    val_avg_total_tokens:     Optional[float] = None
    val_avg_input_tokens:     Optional[float] = None
    val_avg_output_tokens:    Optional[float] = None
    val_avg_reasoning_tokens: Optional[float] = None
    val_eval_time_seconds:    Optional[float] = None

    def short_label(self) -> str:
        parent = Path(self.parent_path).stem if self.parent_path else "—"
        return f"[{self.stage}] {self.config_name}  from={parent}"

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateEntry":
        """Construct from a dict, ignoring unknown keys (Layer 2).

        This replaces raw ``CandidateEntry(**d)`` which crashes on extra keys
        (e.g. 'rank' written for display but not part of the dataclass).
        """
        _known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in _known})


class OptimizationLeaderboard:
    """Thread-safe, auto-persisting registry of candidate programs.

    Maintains all candidates sorted by ``val_total_score`` descending.  After
    every insertion the registry is serialised to JSON at ``save_path`` so
    that a crashed run can be resumed by calling ``from_file()``.

    Thread safety
    -------------
    An ``RLock`` (reentrant lock) guards all mutations and complex queries.
    Methods like ``top_k`` can internally call ``pareto_front`` which also
    acquires the lock — the reentrant lock prevents deadlock.

    Typical usage
    -------------
    >>> lb = OptimizationLeaderboard(save_path=".cache/tournament/t_001/leaderboard.json")
    >>> lb.add(entry)                 # inserts, sorts, saves
    >>> seeds = lb.top_k(2)           # [best, 2nd-best]
    >>> pareto = lb.pareto_front()    # Pareto-optimal (score, tokens)
    >>> lb.export_to_excel("out.xlsx")
    """

    def __init__(self, save_path: str | Path = ""):
        self.save_path = Path(save_path) if save_path else None
        self.entries: List[CandidateEntry] = []
        self._lock = threading.RLock()  # RLock: top_k→_pareto_top_k→pareto_front re-enters

    # ── Mutation ────────────────────────────────────────────────────

    def add(self, entry: CandidateEntry) -> None:
        """Insert an entry, re-sort descending by val_total_score, auto-save.

        Thread-safe — guarded by an internal lock.
        """
        with self._lock:
            # Idempotent: update existing entry with same config_name (Layer 4)
            for i, existing in enumerate(self.entries):
                if existing.config_name == entry.config_name:
                    self.entries[i] = entry
                    self.entries.sort(key=lambda e: e.val_total_score, reverse=True)
                    self.save()
                    return
            self.entries.append(entry)
            self.entries.sort(key=lambda e: e.val_total_score, reverse=True)
            self.save()

    # ── Queries ─────────────────────────────────────────────────────

    def top_k(self, k: int, stage: str | None = None,
              criteria: str = "score_only",
              exclude_stages: list | None = None) -> List[CandidateEntry]:
        """Return the best *k* entries.

        Parameters
        ----------
        k : int
            Number of entries to return.
        stage : str or None
            If set, filter to this stage only.
        criteria : str
            "score_only" — rank by val_total_score descending (default).
            "pareto"     — Pareto front (maximize score, minimize val_avg_total_tokens),
                           then fill with next-best by score if needed.
            "tokens_only"— rank by val_avg_total_tokens ascending.
        exclude_stages : list of str or None
            Stages to exclude from the pool before selection.  Applied
            BEFORE ``_require_token_data`` so that diagnostic-only stages
            (e.g. "seed_baseline", which is scored on train_set and has no
            meaningful token data) never cause token-tracking checks to fail.
        """
        with self._lock:
            if stage is not None:
                pool = [e for e in self.entries if e.stage == stage]
            else:
                pool = list(self.entries)

            if exclude_stages:
                _excl = set(exclude_stages)
                pool = [e for e in pool if e.stage not in _excl]

            if criteria == "pareto":
                self._require_token_data(pool, criteria)
                return self._pareto_top_k(pool, k)
            elif criteria == "tokens_only":
                self._require_token_data(pool, criteria)
                pool.sort(key=lambda e: (e.val_avg_total_tokens, -e.val_total_score))
                return pool[:k]
            else:  # score_only
                pool.sort(key=lambda e: e.val_total_score, reverse=True)
                return pool[:k]

    def _require_token_data(self, pool: List[CandidateEntry], criteria: str) -> None:
        """Verify every entry in *pool* has valid per-sample token data.

        Raises ``ValueError`` if any entry is missing ``val_avg_total_tokens``.
        This is intentionally a hard error — callers (seed generator, stage
        runner) must always pass an explicit ``lm`` to ``evaluate_candidate()``
        so that token tracking works regardless of ContextVar propagation.
        """
        if not pool:
            return
        missing = [e for e in pool if e.val_avg_total_tokens is None or e.val_avg_total_tokens <= 0]
        if missing:
            sample = ", ".join(f"{e.stage}:{e.config_name}" for e in missing[:5])
            raise ValueError(
                f"Selection criteria '{criteria}' requires val_avg_total_tokens for every candidate. "
                f"Missing token data for {len(missing)} candidate(s): {sample}"
            )

    def pareto_front(self, pool: List[CandidateEntry] | None = None) -> List[CandidateEntry]:
        """Return the Pareto-optimal subset of *pool* (or all entries).

        An entry A dominates B when:
          A.val_total_score  >= B.val_total_score   (higher is better)
          A.val_avg_total_tokens <= B.val_avg_total_tokens  (lower is better)
          at least one of the inequalities is strict.

        Returns entries sorted by (score DESC, tokens ASC) for readability.
        """
        with self._lock:
            entries = list(pool) if pool is not None else list(self.entries)

        self._require_token_data(entries, "pareto")

        def dominates(a: CandidateEntry, b: CandidateEntry) -> bool:
            a_better_score = a.val_total_score >= b.val_total_score
            a_better_tokens = a.val_avg_total_tokens <= b.val_avg_total_tokens
            a_strictly = (a.val_total_score > b.val_total_score) or (a.val_avg_total_tokens < b.val_avg_total_tokens)
            return a_better_score and a_better_tokens and a_strictly

        pareto = []
        for candidate in entries:
            is_dominated = False
            for other in entries:
                if other is candidate:
                    continue
                if dominates(other, candidate):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto.append(candidate)

        pareto.sort(key=lambda e: (e.val_total_score, -e.val_avg_total_tokens), reverse=True)
        return pareto

    def _pareto_top_k(self, pool: List[CandidateEntry], k: int) -> List[CandidateEntry]:
        """Select up to *k* entries preferring the Pareto front, filling with non-Pareto by score.

        Objectives: maximize val_total_score, minimize val_avg_total_tokens (per-sample inference cost).

        Selection strategy:
          - From Pareto front: ceil(k/2) score leaders + floor(k/2) efficiency leaders
          - If Pareto front yields fewer than k unique entries, remaining slots are filled
            with the best non-Pareto candidates by score (so we always pass k seeds forward
            even when one entry dominates all others on both dimensions).
        """
        if not pool:
            return []

        pareto = self.pareto_front(pool)

        if not pareto:
            return []

        n_score = math.ceil(k / 2)
        n_tokens = math.floor(k / 2)

        # Score leaders: top n_score by val_total_score descending
        by_score = sorted(pareto, key=lambda e: e.val_total_score, reverse=True)
        score_picks = by_score[:n_score]

        # Efficiency leaders: top n_tokens by val_avg_total_tokens ascending (per-sample inference cost)
        by_tokens = sorted(pareto, key=lambda e: e.val_avg_total_tokens)
        token_picks = by_tokens[:n_tokens]

        # Merge, de-duplicate (preserve order: score leaders first)
        seen = set()
        selected = []
        for e in score_picks + token_picks:
            if id(e) not in seen:
                seen.add(id(e))
                selected.append(e)

        # Fill remaining slots from non-Pareto candidates by score
        # (handles the case where one entry dominates all others so pareto front = 1)
        if len(selected) < k:
            pareto_ids = {id(e) for e in pareto}
            non_pareto = sorted(
                [e for e in pool if id(e) not in pareto_ids],
                key=lambda e: e.val_total_score,
                reverse=True,
            )
            for e in non_pareto:
                if len(selected) >= k:
                    break
                if id(e) not in seen:
                    seen.add(id(e))
                    selected.append(e)

        selected.sort(key=lambda e: e.val_total_score, reverse=True)
        return selected

    def best(self, stage: str | None = None,
             criteria: str = "score_only",
             exclude_stages: list | None = None) -> Optional[CandidateEntry]:
        """Return the single best entry, or None if the leaderboard is empty."""
        top = self.top_k(1, stage=stage, criteria=criteria, exclude_stages=exclude_stages)
        return top[0] if top else None

    def get_stages(self) -> List[str]:
        """Return distinct stages in canonical execution order.

        ``self.entries`` is sorted by score, so iterating it directly would
        return stages in score-order (best stage first).  This method uses
        ``_CANONICAL_STAGE_ORDER`` so that ``get_stage_summary``'s
        ``improvement_over_prev`` always compares stages in execution order.
        Non-canonical stage names (e.g. user-defined) are appended in
        first-seen order after the canonical ones.
        """
        present = {e.stage for e in self.entries}
        ordered = [s for s in _CANONICAL_STAGE_ORDER if s in present]
        # Append any non-canonical stage names in first-seen order
        for e in self.entries:
            if e.stage not in _CANONICAL_STAGE_ORDER and e.stage not in ordered:
                ordered.append(e.stage)
        return ordered

    def stage_entries(self, stage: str) -> List[CandidateEntry]:
        """All entries produced by *stage*, sorted by score descending."""
        return [e for e in self.entries if e.stage == stage]

    # ── Summaries ───────────────────────────────────────────────────

    def get_stage_summary(self, stage: str) -> dict:
        """Return best / worst / mean for a given stage + improvement over previous."""
        ents = sorted(
            self.stage_entries(stage),
            key=lambda e: e.val_total_score,
            reverse=True,
        )
        if not ents:
            return {"stage": stage, "n_candidates": 0}
        scores = [e.val_total_score for e in ents]
        best_entry = ents[0]
        worst_entry = ents[-1]

        # Compute improvement over the best entry from the PREVIOUS stage
        stages = self.get_stages()
        idx = stages.index(stage) if stage in stages else -1
        prev_best_score = None
        if idx > 0:
            prev_stage = stages[idx - 1]
            prev_best = self.best(prev_stage)
            if prev_best:
                prev_best_score = prev_best.val_total_score

        improvement = (best_entry.val_total_score - prev_best_score) if prev_best_score is not None else None

        return {
            "stage": stage,
            "n_candidates": len(ents),
            "best_score": best_entry.val_total_score,
            "best_cls": best_entry.val_cls_score,
            "best_change": best_entry.val_change_score,
            "best_candidate": best_entry.config_name,
            "best_program_path": best_entry.program_path,
            "worst_score": worst_entry.val_total_score,
            "mean_score": sum(scores) / len(scores),
            "improvement_over_prev": improvement,
        }

    def get_full_summary(
        self,
        selection_criteria: str = "score_only",
        exclude_stages: list | None = None,
    ) -> dict:
        """Aggregate summary across all stages."""
        excluded = set(exclude_stages or [])
        stages = [s for s in self.get_stages() if s not in excluded]
        per_stage = {s: self.get_stage_summary(s) for s in stages}
        score_leader = self.best(criteria="score_only", exclude_stages=exclude_stages)
        selection_leader = self.best(
            criteria=selection_criteria,
            exclude_stages=exclude_stages,
        )
        candidate_pool = [
            e for e in self.entries
            if e.stage not in excluded
        ]
        return {
            "n_total_candidates": len(candidate_pool),
            "stages": stages,
            "per_stage": per_stage,
            "score_leader": asdict(score_leader) if score_leader else None,
            "selection_criteria": selection_criteria,
            "selection_leader": asdict(selection_leader) if selection_leader else None,
        }

    # ── Display ────────────────────────────────────────────────────

    def print_table(self, file=None, max_rows: int | None = None):
        """Print a formatted leaderboard table (includes per-sample token column)."""
        header = (
            f"{'#':>3}  {'Stage':<10} {'Config':<35} "
            f"{'Total':>7} {'Cls':>7} {'Chg':>7} "
            f"{'Demos':>5} {'InstLen':>7} {'AvgTok/Smp':>11}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]

        entries = self.entries[:max_rows] if max_rows else self.entries
        for i, e in enumerate(entries, 1):
            tok_str = (f"{e.val_avg_total_tokens:>11.0f}"
                       if e.val_avg_total_tokens is not None else f"{'—':>11}")
            lines.append(
                f"{i:3d}  {e.stage:<10} {e.config_name:<35} "
                f"{e.val_total_score:7.4f} {e.val_cls_score:7.4f} {e.val_change_score:7.4f} "
                f"{e.n_demos:5d} {e.instruction_length:7d} {tok_str}"
            )
        lines.append(sep)
        text = "\n".join(lines)
        print(text, file=file)
        return text

    def print_stage_summary(self, stage: str, file=None):
        """Print a summary block for a single stage."""
        s = self.get_stage_summary(stage)
        if s["n_candidates"] == 0:
            print(f"  Stage '{stage}': no candidates.", file=file)
            return
        imp = f"{s['improvement_over_prev']:+.4f}" if s["improvement_over_prev"] is not None else "N/A"
        lines = [
            f"  Stage '{stage}' summary ({s['n_candidates']} candidates):",
            f"    Best  : {s['best_score']:.4f}  (cls={s['best_cls']:.4f}, chg={s['best_change']:.4f})  — {s['best_candidate']}",
            f"    Worst : {s['worst_score']:.4f}",
            f"    Mean  : {s['mean_score']:.4f}",
            f"    Improvement over prev stage: {imp}",
        ]
        text = "\n".join(lines)
        print(text, file=file)

    # ── Per-category detail for a single entry ──────────────────────

    @staticmethod
    def format_per_category(entry: CandidateEntry) -> str:
        """Format per-category scores for a single entry."""
        if not entry.per_category_scores:
            return "  (no per-category data)"
        cats = sorted(entry.per_category_scores.keys())
        parts = [f"{c}={entry.per_category_scores[c]:.3f}" for c in cats]
        return "  Per-category: " + "  ".join(parts)

    # ── Persistence ────────────────────────────────────────────────

    def save(self, path: str | Path | None = None):
        """Serialise all entries to JSON.  Called automatically after every add().

        Writes to *path* if given, otherwise to ``self.save_path``.
        Silently skips when no path is configured (in-memory-only mode).
        """
        path = Path(path) if path else self.save_path
        if not path:
            return  # no save_path configured — skip silently
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "n_entries": len(self.entries),
            "entries": [asdict(e) for e in self.entries],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: str | Path | None = None) -> "OptimizationLeaderboard":
        """Load entries from JSON on disk, replacing in-memory state.

        Returns *self* for chaining.  Used by ``from_file()`` and internally
        when resuming an interrupted tournament run.
        """
        path = Path(path) if path else self.save_path
        if not path or not path.exists():
            return self
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _known = {f.name for f in fields(CandidateEntry)}
        self.entries = [
            CandidateEntry.from_dict(e)
            for e in data.get("entries", [])
        ]
        self.entries.sort(key=lambda e: e.val_total_score, reverse=True)
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> "OptimizationLeaderboard":
        """Create a new leaderboard by loading from a saved JSON file."""
        lb = cls(save_path=str(path))
        lb.load(path)
        return lb

    # ── Export ──────────────────────────────────────────────────────

    def export_to_excel(self, path: str | Path, stage_execution_history: dict | None = None):
        """Write leaderboard to an Excel file for comparison with dca_experiments.xlsx."""
        try:
            import pandas as pd
        except ImportError:
            print(f"  [WARN] pandas not available — skipping Excel export to {path}")
            return

        rows = []
        for i, e in enumerate(self.entries, 1):
            metadata = e.metadata or {}
            row = {
                "Rank": i,
                "Stage": e.stage,
                "Config": e.config_name,
                "Parent": Path(e.parent_path).stem if e.parent_path else "",
                "Total": e.val_total_score,
                "Cls": e.val_cls_score,
                "Change": e.val_change_score,
                "Demos": e.n_demos,
                "InstructionLen": e.instruction_length,
                "ValAvgTotalTok": e.val_avg_total_tokens,
                "ValAvgInputTok": e.val_avg_input_tokens,
                "ValAvgOutputTok": e.val_avg_output_tokens,
                "ValAvgReasoningTok": e.val_avg_reasoning_tokens,
                "ValEvalTimeSec": e.val_eval_time_seconds,
                "OptimizationStartedAt": metadata.get("optimization_started_at"),
                "OptimizationCompletedAt": metadata.get("optimization_completed_at"),
                "OptimizationDurationSec": metadata.get("optimization_duration_seconds"),
                "CompileDurationSec": metadata.get("compile_duration_seconds"),
                "EvaluationDurationSec": metadata.get("evaluation_duration_seconds", e.val_eval_time_seconds),
                "Timestamp": e.timestamp,
                "ProgramPath": e.program_path,
            }
            # Add per-category columns
            for cat in sorted(e.per_category_scores.keys()):
                row[f"Cat_{cat}"] = e.per_category_scores[cat]
            rows.append(row)

        df = pd.DataFrame(rows)
        stage_rows = []
        if stage_execution_history:
            for stage in self.get_stages():
                execution = dict(stage_execution_history.get(stage) or {})
                if not execution:
                    continue
                stage_rows.append({
                    "Stage": stage,
                    "Status": execution.get("status"),
                    "StartedAt": execution.get("started_at"),
                    "CompletedAt": execution.get("completed_at"),
                    "DurationSec": execution.get("duration_seconds"),
                })

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(str(path)) as writer:
            df.to_excel(writer, index=False, sheet_name="Candidates")
            if stage_rows:
                pd.DataFrame(stage_rows).to_excel(writer, index=False, sheet_name="StageTimings")
        print(f"  Leaderboard exported to: {path}")
