"""DCA scatter-plus-mean plots (individual runs + mean marker only — no box, no
median) from results/plots/dca_datapoints.csv.

For every (inference model x optimizer-seed-source) it draws the same x-axis:
    MiDAS pipeline (v1, non-DSPy) | v1 human (DSPy) | v2 human (DSPy) |
    bootstrap | miprov2 | gepa | tournament
Circle = No-CoT run, triangle = CoT run; green diamond = mean. Categories with no
data are kept on the axis and marked "no run".

Run:  python results/plots/plot_dca_scatter.py     (build_datapoints.py first)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV = os.path.join("results", "plots", "dca_datapoints.csv")
OUTDIR = os.path.join("results", "plots", "figures")
DOT, MEAN = "#FF8C00", "#2E8B57"
D = pd.read_csv(CSV)


def sub(**cond):
    m = pd.Series(True, index=D.index)
    for k, v in cond.items():
        m &= (D[k] == v)
    return D.loc[m]


def human_dspy(model, ver):
    """DSPy-run human prompt; fall back to the single unlabeled run if no DSPy row."""
    s = sub(optimizer="human", prompt_version=ver, prompt_type="dspy", model=model)
    return s if len(s) else sub(optimizer="human", prompt_version=ver, prompt_type="single", model=model)


def scatter_mean(groups, title, outfile, ylabel="DCA Total Score (val set)"):
    """groups: list of (label, DataFrame with 'score' and 'cot' columns)."""
    fig, ax = plt.subplots(figsize=(max(9.5, 1.55 * len(groups) + 2), 5.6))
    allv = [s for _, df in groups for s in df["score"]]
    lo, hi = (min(allv) - 0.03, max(allv) + 0.03) if allv else (0.45, 0.95)
    rng = np.random.default_rng(42)
    for i, (lbl, df) in enumerate(groups):
        vals = df["score"].to_numpy(float)
        if len(vals) == 0:
            ax.annotate("no run\nyet", (i, (lo + hi) / 2), ha="center", va="center",
                        fontsize=8, style="italic", color="#c0392b",
                        bbox=dict(boxstyle="round", fc="#fdecec", ec="#e74c3c", alpha=0.9))
            continue
        is_cot = (df["cot"] == "cot").to_numpy()
        for mask, marker in [(~is_cot, "o"), (is_cot, "^")]:
            v = vals[mask]
            if len(v) == 0:
                continue
            jit = (rng.random(len(v)) - 0.5) * (0.14 if len(v) > 1 else 0.0)
            ax.scatter(np.full(len(v), i) + jit, v, marker=marker, s=44, facecolor=DOT,
                       edgecolor="#7a4400", linewidth=0.5, zorder=3)
        m = float(vals.mean())
        ax.scatter([i], [m], marker="D", s=95, color=MEAN, edgecolor="black",
                   linewidth=0.6, zorder=4)
        ax.annotate(f"{m*100:.1f}", (i, m), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=9, fontweight="bold", color=MEAN)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Run", fontsize=12)
    ax.set_title(title, fontsize=10.5)
    ax.set_ylim(lo, hi)
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.margins(x=0.06)
    handles = [
        Line2D([], [], marker="o", ls="", mfc=DOT, mec="#7a4400", label="Individual run — No-CoT"),
        Line2D([], [], marker="^", ls="", mfc=DOT, mec="#7a4400", label="Individual run — CoT"),
        Line2D([], [], marker="D", ls="", mfc=MEAN, mec="black", label="Mean"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(os.path.join(OUTDIR, outfile), dpi=150)
    plt.close(fig)
    print(f"  wrote {outfile}  ({sum(len(df) for _, df in groups)} points, {len(groups)} categories)")


def main():
    models = [("gpt-4.1-mini", "gpt41mini"), ("gpt-4.1", "gpt41")]
    seeds = [("cr-default", "CR-default signature", "crseed"),
             ("v1", "v1 (MiDAS) prompt", "v1seed"),
             ("v2", "v2 (CR) prompt", "v2seed")]
    for model, mtag in models:
        for seed, seed_lbl, stag in seeds:
            groups = [
                ("MiDAS pipeline\n(v1, non-DSPy)", sub(optimizer="human", prompt_version="v1", prompt_type="midas", model=model)),
                ("v1 human\n(DSPy)", human_dspy(model, "v1")),
                ("v2 human\n(DSPy)", human_dspy(model, "v2")),
                ("bootstrap", sub(optimizer="bootstrap", model=model, seed_source=seed)),
                ("miprov2", sub(optimizer="miprov2", model=model, seed_source=seed)),
                ("gepa", sub(optimizer="gepa", model=model, seed_source=seed)),
                ("tournament", sub(optimizer="tournament", model=model, seed_source=seed)),
            ]
            scatter_mean(
                groups,
                f"DCA — human baselines vs optimizers (inference: {model} · optimizer seed: {seed_lbl} · val)",
                f"dca_{mtag}_{stag}.png")


if __name__ == "__main__":
    main()
