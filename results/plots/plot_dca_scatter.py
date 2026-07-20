"""Generate DCA scatter-plus-mean plots (individual runs + mean marker only —
no box, no median) from results/plots/dca_datapoints.csv.

Left axis: DCA total score (val set). X axis: run categories. Each dot is one
run; the green diamond is the mean. Produces one PNG per slice.

Run:  python results/plots/plot_dca_scatter.py   (run build_datapoints.py first)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = os.path.join("results", "plots", "dca_datapoints.csv")
OUTDIR = os.path.join("results", "plots", "figures")
DOT = "#FF8C00"
MEAN = "#2E8B57"


def scatter_mean(groups, title, outfile, ylabel="DCA Total Score (val set)"):
    """groups: list of (label, list-of-scores). Draw dots + mean diamond per group."""
    groups = [(lbl, list(v)) for lbl, v in groups if len(v) > 0]
    if not groups:
        print(f"  skip (no data): {outfile}")
        return
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(groups) + 2), 5.2))
    for i, (lbl, vals) in enumerate(groups):
        vals = np.asarray(vals, float)
        jit = (rng.random(len(vals)) - 0.5) * (0.12 if len(vals) > 1 else 0.0)
        ax.scatter(np.full(len(vals), i) + jit, vals, s=42, facecolor=DOT,
                   edgecolor="#7a4400", linewidth=0.5, zorder=3,
                   label="Individual run" if i == 0 else None)
        m = float(vals.mean())
        ax.scatter([i], [m], marker="D", s=95, color=MEAN, edgecolor="black",
                   linewidth=0.6, zorder=4, label="Mean" if i == 0 else None)
        ax.annotate(f"{m*100:.1f}", (i, m), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=9, fontweight="bold", color=MEAN)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Run", fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.margins(x=0.08)
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, outfile)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    n = sum(len(v) for _, v in groups)
    print(f"  wrote {path}  ({len(groups)} categories, {n} points)")


def main():
    d = pd.read_csv(CSV)

    def sc(**cond):
        m = pd.Series(True, index=d.index)
        for k, v in cond.items():
            m &= (d[k] == v)
        return d.loc[m, "score"].tolist()

    # 1 & 2 — full comparison per model, CR-default seed
    for model, tag in [("gpt-4.1-mini", "gpt41mini"), ("gpt-4.1", "gpt41")]:
        groups = [
            ("human v1", sc(optimizer="human", prompt_version="v1", model=model)),
            ("human v2", sc(optimizer="human", prompt_version="v2", model=model)),
            ("bootstrap", sc(optimizer="bootstrap", model=model, seed_source="cr-default")),
            ("miprov2", sc(optimizer="miprov2", model=model, seed_source="cr-default")),
            ("gepa", sc(optimizer="gepa", model=model, seed_source="cr-default")),
            ("tournament", sc(optimizer="tournament", model=model)),
        ]
        scatter_mean(groups, f"DCA — human baselines vs optimizers ({model}, CR-default seed, val)",
                     f"dca_cr_default_{tag}.png")

    # 3 — seed source comparison (MIPROv2, gpt-4.1-mini)
    scatter_mean(
        [("CR-default seed", sc(optimizer="miprov2", model="gpt-4.1-mini", seed_source="cr-default")),
         ("v1 seed", sc(optimizer="miprov2", model="gpt-4.1-mini", seed_source="v1")),
         ("v2 seed", sc(optimizer="miprov2", model="gpt-4.1-mini", seed_source="v2"))],
        "DCA — MIPROv2 by starting-seed prompt (gpt-4.1-mini, val)",
        "dca_seed_source_miprov2_gpt41mini.png")

    # 4 — model comparison (gpt-4.1-mini vs gpt-4.1) per optimizer, CR-default
    groups = []
    for opt in ["bootstrap", "miprov2", "gepa", "tournament"]:
        for model, ml in [("gpt-4.1-mini", "mini"), ("gpt-4.1", "4.1")]:
            cond = dict(optimizer=opt, model=model)
            if opt != "tournament":
                cond["seed_source"] = "cr-default"
            groups.append((f"{opt}\n{ml}", sc(**cond)))
    scatter_mean(groups, "DCA — gpt-4.1-mini vs gpt-4.1 per optimizer (CR-default seed, val)",
                 "dca_model_compare_cr_default.png")

    # 5 — CoT vs No-CoT (MIPROv2, gpt-4.1-mini, CR-default)
    scatter_mean(
        [("CoT", sc(optimizer="miprov2", model="gpt-4.1-mini", seed_source="cr-default", cot="cot")),
         ("No-CoT", sc(optimizer="miprov2", model="gpt-4.1-mini", seed_source="cr-default", cot="nocot"))],
        "DCA — CoT vs No-CoT (MIPROv2, gpt-4.1-mini, val)",
        "dca_cot_vs_nocot_miprov2_gpt41mini.png")

    # 6 — human v2 across models
    scatter_mean(
        [(m, sc(optimizer="human", prompt_version="v2", model=m))
         for m in ["gpt-4.1-mini", "gpt-4.1", "gpt-5-mini", "gpt-5.2"]],
        "DCA — v2 (CR) human prompt across inference models (val)",
        "dca_human_v2_by_model.png")


if __name__ == "__main__":
    main()
