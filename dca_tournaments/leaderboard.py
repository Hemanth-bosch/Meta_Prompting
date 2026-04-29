"""OptimizationLeaderboard — tracks all candidate programs across stages.

Maintains a sorted list of ``CandidateEntry`` objects, auto-saves after
every insertion, and provides stage summaries, top-K selection, and export
to Excel / formatted console table.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field, asdict
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
    """A single candidate program and its evaluation results."""

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


class OptimizationLeaderboard:
    """Sorted collection of CandidateEntry objects with persistence."""

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

    def best(self, stage: str | None = None) -> Optional[CandidateEntry]:
        top = self.top_k(1, stage=stage)
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

    def get_full_summary(self) -> dict:
        """Aggregate summary across all stages."""
        stages = self.get_stages()
        per_stage = {s: self.get_stage_summary(s) for s in stages}
        global_best = self.best()
        return {
            "n_total_candidates": len(self.entries),
            "stages": stages,
            "per_stage": per_stage,
            "global_best": asdict(global_best) if global_best else None,
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
        path = Path(path) if path else self.save_path
        if not path or not path.exists():
            return self
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.entries = [CandidateEntry(**e) for e in data.get("entries", [])]
        self.entries.sort(key=lambda e: e.val_total_score, reverse=True)
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> "OptimizationLeaderboard":
        """Create a new leaderboard by loading from a saved JSON file."""
        lb = cls(save_path=str(path))
        lb.load(path)
        return lb

    # ── Export ──────────────────────────────────────────────────────

    def export_to_excel(self, path: str | Path):
        """Write leaderboard to an Excel file for comparison with dca_experiments.xlsx."""
        try:
            import pandas as pd
        except ImportError:
            print(f"  [WARN] pandas not available — skipping Excel export to {path}")
            return

        rows = []
        for i, e in enumerate(self.entries, 1):
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
                "Timestamp": e.timestamp,
                "ProgramPath": e.program_path,
            }
            # Add per-category columns
            for cat in sorted(e.per_category_scores.keys()):
                row[f"Cat_{cat}"] = e.per_category_scores[cat]
            rows.append(row)

        df = pd.DataFrame(rows)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(str(path), index=False)
        print(f"  Leaderboard exported to: {path}")
