"""
Optimization Cost vs Performance Gain — DCA Tournament Analysis
Generates charts for the slide: "Is prompt optimization worth it?"
"""

import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(r"C:\Users\BHS1RNG\OneDrive - Bosch Group\AIQ-212\Code\meta-prompting\.cache\tournament")
OUT_DIR  = Path(r"C:\Users\BHS1RNG\OneDrive - Bosch Group\AIQ-212\Code\meta-prompting\analysis_output")
OUT_DIR.mkdir(exist_ok=True)

PYTHON_EXE = r"C:\Users\BHS1RNG\OneDrive - Bosch Group\AIQ-212\Code\meta-prompting\.meta-prompting\Scripts\python.exe"

# GPT-4.1 / GPT-4.1-mini Azure pricing (per 1M tokens, approximate)
PRICE_INPUT  = {"gpt-4.1": 2.00, "gpt-4.1-mini": 0.40}   # $/1M input tokens
PRICE_OUTPUT = {"gpt-4.1": 8.00, "gpt-4.1-mini": 1.60}   # $/1M output tokens
# Assume ~90% of tokens are prompt, 10% completion for optimization
OPT_PROMPT_RATIO = 0.90

V1_MIDAS_TOURNAMENTS = {"t_4953"}

CR_TOURNAMENTS = {
    "tournament_20260504235001",
    "tournament_20260505160645",
    "tournament_20260507113820",
    "tournament_20260507113839",
    "tournament_20260506120822",
    "tournament_20260506120758",
    "tournament_20260506090846",
    "tournament_20260506090909",
    "tournament_20260505172652",
    "tournament_20260505153708",
    "tournament_20260507121046",
    "tournament_20260504233802",
}

ALL_TOURNAMENTS = V1_MIDAS_TOURNAMENTS | CR_TOURNAMENTS


# ── Data collection ──────────────────────────────────────────────────────────
def load_tournament(t_dir: Path):
    lb_path = t_dir / "leaderboard.json"
    if not lb_path.exists():
        return None
    lb = json.loads(lb_path.read_text(encoding="utf-8"))
    entries = lb.get("entries", [])
    if not entries:
        return None

    # Sum opt tokens per stage
    stage_tokens = {"bootstrap": 0, "miprov2": 0, "gepa": 0}
    for f in t_dir.rglob("*optimization_tokens*.json"):
        tok = json.loads(f.read_text(encoding="utf-8"))
        sf = f.parent.name
        if "bootstrap" in sf:
            stage_tokens["bootstrap"] += tok.get("total_tokens", 0)
        elif "miprov2" in sf:
            stage_tokens["miprov2"]   += tok.get("total_tokens", 0)
        elif "gepa" in sf:
            stage_tokens["gepa"]      += tok.get("total_tokens", 0)

    total_opt = sum(stage_tokens.values())
    if total_opt == 0:
        return None

    # Separate prompt vs completion for cost calc (use files if available)
    stage_prompt = {"bootstrap": 0, "miprov2": 0, "gepa": 0}
    stage_comp   = {"bootstrap": 0, "miprov2": 0, "gepa": 0}
    for f in t_dir.rglob("*optimization_tokens*.json"):
        tok = json.loads(f.read_text(encoding="utf-8"))
        sf = f.parent.name
        key = "bootstrap" if "bootstrap" in sf else ("miprov2" if "miprov2" in sf else "gepa")
        stage_prompt[key] += tok.get("prompt_tokens", 0)
        stage_comp[key]   += tok.get("completion_tokens", 0)

    # Baseline
    baseline = next((e for e in entries if e.get("config_name") == "seed_baseline_bare"), None)
    base_score   = baseline["val_total_score"] if baseline else None
    base_inf_tok = baseline.get("val_avg_total_tokens") or 0

    # Best non-baseline
    non_base = [e for e in entries if "baseline" not in e.get("config_name", "")]
    if not non_base:
        return None
    best = max(non_base, key=lambda e: e.get("val_total_score", 0))
    best_score   = best["val_total_score"]
    best_stage   = best.get("stage", "")
    best_inf_tok = best.get("val_avg_total_tokens") or 0

    # Model
    model = ""
    sum_path = t_dir / "summary.json"
    if sum_path.exists():
        s = json.loads(sum_path.read_text(encoding="utf-8"))
        model = s.get("config", {}).get("student_model_name", "")
    model_short = "gpt-4.1" if "gpt-4.1" in model and "mini" not in model else "gpt-4.1-mini"

    return {
        "id":             t_dir.name,
        "baseline_type":  "v1 MIDAS" if t_dir.name in V1_MIDAS_TOURNAMENTS else "CR Signature",
        "model":          model_short,
        "stage_tokens":   stage_tokens,
        "stage_prompt":   stage_prompt,
        "stage_comp":     stage_comp,
        "total_opt_tok":  total_opt,
        "base_score":     base_score * 100 if base_score else None,
        "best_score":     best_score * 100,
        "gain_pp":        (best_score - base_score) * 100 if base_score else None,
        "best_stage":     best_stage,
        "base_inf_tok":   base_inf_tok,
        "best_inf_tok":   best_inf_tok,
        "inf_tok_saved":  base_inf_tok - best_inf_tok,
    }


records = []
for t_name in sorted(ALL_TOURNAMENTS):
    t_dir = BASE_DIR / t_name
    if not t_dir.exists():
        print(f"  [MISSING] {t_name}")
        continue
    rec = load_tournament(t_dir)
    if rec:
        records.append(rec)
    else:
        print(f"  [SKIP] {t_name} – no opt token data")

print(f"\nLoaded {len(records)} tournaments")
for r in records:
    print(f"  {r['id'][-13:]}  {r['baseline_type']:12s}  {r['model']:12s}  "
          f"opt={r['total_opt_tok']/1e6:.1f}M  base={r['base_score']:.1f}%  "
          f"best={r['best_score']:.1f}%  gain={r['gain_pp']:+.1f}pp  "
          f"inf_saved={r['inf_tok_saved']:+.0f}")


# ── Pricing helper ───────────────────────────────────────────────────────────
def opt_cost_usd(rec):
    total_p = sum(rec["stage_prompt"].values())
    total_c = sum(rec["stage_comp"].values())
    if total_p == 0 and total_c == 0:
        # Estimate from total using 90/10 split
        total_p = rec["total_opt_tok"] * 0.90
        total_c = rec["total_opt_tok"] * 0.10
    m = rec["model"]
    pi = PRICE_INPUT.get(m, 0.40)
    po = PRICE_OUTPUT.get(m, 1.60)
    return (total_p * pi + total_c * po) / 1e6


# ── Colour / marker helpers ──────────────────────────────────────────────────
COLOUR = {"v1 MIDAS": "#E05A2B", "CR Signature": "#2B72C8"}
MARKER = {"gpt-4.1": "D", "gpt-4.1-mini": "o"}

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 – Scatter: Total Opt Tokens vs Gain (pp), by baseline type + model
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor("#F7F9FC")
ax.set_facecolor("#F7F9FC")

for r in records:
    if r["gain_pp"] is None:
        continue
    x = r["total_opt_tok"] / 1e6
    y = r["gain_pp"]
    c = COLOUR[r["baseline_type"]]
    mk = MARKER[r["model"]]
    ax.scatter(x, y, color=c, marker=mk, s=110, edgecolors="white", linewidths=0.8, zorder=3)
    ax.annotate(
        f"{r['best_score']:.1f}%",
        (x, y), textcoords="offset points", xytext=(6, 3),
        fontsize=7.5, color="#333333"
    )

# Mean lines per group
for btype, col in COLOUR.items():
    grp = [r for r in records if r["baseline_type"] == btype and r["gain_pp"] is not None]
    if grp:
        mx = np.mean([r["total_opt_tok"]/1e6 for r in grp])
        my = np.mean([r["gain_pp"] for r in grp])
        ax.axhline(my, color=col, lw=1.2, ls="--", alpha=0.45)
        ax.text(0.35, my + 0.2, f"avg gain {btype}: {my:.1f} pp", color=col,
                fontsize=8, va="bottom")

ax.set_xlabel("Total Optimization Tokens (millions, one-time cost)", fontsize=10)
ax.set_ylabel("DCA Score Gain over Baseline (pp)", fontsize=10)
ax.set_title("Optimization Cost vs Performance Gain — DCA Tournaments", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, ls=":")
ax.set_xlim(left=0)

# Legend
patches = [mpatches.Patch(color=v, label=k) for k, v in COLOUR.items()]
mlines  = [plt.Line2D([0],[0], marker=mk, color="gray", markersize=7, ls="none",
                       label=f"Student: {m}") for m, mk in MARKER.items()]
ax.legend(handles=patches + mlines, fontsize=8, loc="lower right", framealpha=0.8)

plt.tight_layout()
out1 = OUT_DIR / "chart1_scatter_opt_tokens_vs_gain.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out1}")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 – Stacked bar per tournament: stage breakdown of opt tokens
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor("#F7F9FC")
ax.set_facecolor("#F7F9FC")

recs_sorted = sorted(records, key=lambda r: r["total_opt_tok"])
labels = []
for r in recs_sorted:
    short = r["id"][-8:]  # last 8 chars of timestamp
    labels.append(f"{short}\n{r['model'].replace('azure/','')}\n({r['baseline_type']})")

x = np.arange(len(recs_sorted))
w = 0.6

b_boot = [r["stage_tokens"]["bootstrap"] / 1e6 for r in recs_sorted]
b_mipr = [r["stage_tokens"]["miprov2"]   / 1e6 for r in recs_sorted]
b_gepa = [r["stage_tokens"]["gepa"]       / 1e6 for r in recs_sorted]

bars1 = ax.bar(x, b_boot, w, label="Bootstrap",  color="#5B9BD5", edgecolor="white")
bars2 = ax.bar(x, b_mipr, w, bottom=b_boot, label="MIPROv2",   color="#ED7D31", edgecolor="white")
bars3 = ax.bar(x, b_gepa, w,
               bottom=[b_boot[i]+b_mipr[i] for i in range(len(x))],
               label="GEPA",       color="#70AD47", edgecolor="white")

# Annotate gain on top of each bar
for i, r in enumerate(recs_sorted):
    total_m = r["total_opt_tok"] / 1e6
    gain = r["gain_pp"]
    label_str = f"+{gain:.1f}pp" if gain is not None else ""
    ax.text(i, total_m + 0.3, label_str, ha="center", va="bottom", fontsize=8, fontweight="bold",
            color="#C00000")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("Optimization Tokens (M)", fontsize=10)
ax.set_title("Stage-by-Stage Optimization Token Cost\n(labelled with DCA score gain over baseline)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.grid(axis="y", alpha=0.3, ls=":")
plt.tight_layout()
out2 = OUT_DIR / "chart2_stacked_bar_opt_tokens_by_stage.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 – USD Cost vs Gain scatter (cost estimated from token counts + pricing)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
fig.patch.set_facecolor("#F7F9FC")
ax.set_facecolor("#F7F9FC")

for r in records:
    if r["gain_pp"] is None:
        continue
    cost = opt_cost_usd(r)
    y = r["gain_pp"]
    c = COLOUR[r["baseline_type"]]
    mk = MARKER[r["model"]]
    ax.scatter(cost, y, color=c, marker=mk, s=120, edgecolors="white", linewidths=0.8, zorder=3)
    ax.annotate(
        f"${cost:.0f}\n+{y:.1f}pp",
        (cost, y), textcoords="offset points", xytext=(6, 2),
        fontsize=7.5, color="#333333"
    )

ax.set_xlabel("Estimated One-Time Optimization Cost (USD)", fontsize=10)
ax.set_ylabel("DCA Score Gain over Baseline (pp)", fontsize=10)
ax.set_title("Optimization Cost (USD) vs Performance Gain\nGPT-4.1: $2/$8 per 1M in/out  |  GPT-4.1-mini: $0.40/$1.60 per 1M in/out",
             fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3, ls=":")

patches = [mpatches.Patch(color=v, label=k) for k, v in COLOUR.items()]
mlines  = [plt.Line2D([0],[0], marker=mk, color="gray", markersize=7, ls="none",
                       label=f"Student: {m}") for m, mk in MARKER.items()]
ax.legend(handles=patches + mlines, fontsize=8, loc="lower right", framealpha=0.8)
plt.tight_layout()
out3 = OUT_DIR / "chart3_usd_cost_vs_gain.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 – Total cost (opt + ongoing inference overhead) vs call volume
# Optimized prompts cost SLIGHTLY MORE per inference (longer instructions)
# but deliver higher quality. Show cumulative cost delta vs baseline prompt.
# ─────────────────────────────────────────────────────────────────────────────
# Pick representative tournaments: best gpt-4.1 and best gpt-4.1-mini
rep_41     = sorted([r for r in records if r["model"] == "gpt-4.1"      and r["gain_pp"] is not None],
                    key=lambda r: r["gain_pp"], reverse=True)[:2]
rep_41mini = sorted([r for r in records if r["model"] == "gpt-4.1-mini" and r["gain_pp"] is not None],
                    key=lambda r: r["gain_pp"], reverse=True)[:2]
representative = rep_41 + rep_41mini

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor("#F7F9FC")
ax.set_facecolor("#F7F9FC")

n_calls = np.linspace(0, 15000, 500)

for r in representative:
    if r["gain_pp"] is None:
        continue
    cost_opt = opt_cost_usd(r)
    m = r["model"]
    pi = PRICE_INPUT.get(m, 0.40)
    # Additional inference cost per call (optimized uses more tokens than baseline)
    extra_tok_per_call = -r["inf_tok_saved"]          # positive when opt uses MORE
    extra_inf_cost_per_call = extra_tok_per_call * pi / 1e6  # $ extra per call

    # Cumulative extra spend = opt_cost + extra_inf_cost × N_calls
    cumulative_extra = cost_opt + extra_inf_cost_per_call * n_calls

    col  = COLOUR[r["baseline_type"]]
    mk   = MARKER[r["model"]]
    lbl  = (f"{r['id'][-8:]} | {r['model'].replace('azure/','')}"
            f" | +{r['gain_pp']:.1f} pp gain"
            f" | opt cost ${cost_opt:.0f}"
            f" | +{int(extra_tok_per_call):,} tok/call")
    ax.plot(n_calls, cumulative_extra, color=col, lw=2, label=lbl)

    # Annotate at 15k calls
    y_end = cost_opt + extra_inf_cost_per_call * 15000
    ax.annotate(f"+{r['gain_pp']:.1f} pp",
                xy=(15000, y_end), xytext=(-45, 8), textcoords="offset points",
                fontsize=8, color=col, fontweight="bold")

ax.axhline(0, color="#888", lw=1, ls="--", alpha=0.5)

# Reference: gpt-4.1 cost for 10k calls with baseline tokens ~4800
ref_41_cost_per_call = 4800 * PRICE_INPUT["gpt-4.1"] / 1e6
ref_cum = ref_41_cost_per_call * n_calls
ax.plot(n_calls, ref_cum, color="#AAAAAA", lw=1.5, ls=":", label="Ref: 10k × gpt-4.1 baseline inference cost")

ax.set_xlabel("Production Inference Calls (cumulative)", fontsize=10)
ax.set_ylabel("Cumulative Extra Cost vs. Using Unoptimized Prompt (USD)", fontsize=9)
ax.set_title(
    "True Cost of Optimization: One-Time Opt + Extra Inference Overhead\n"
    "(Optimized prompts are longer → cost slightly more per call, but gain quality)",
    fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85)
ax.grid(True, alpha=0.25, ls=":")
ax.set_xlim(0, 15000)
plt.tight_layout()
out4 = OUT_DIR / "chart4_cumulative_cost.png"
plt.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out4}")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 5 – Summary: avg gain and avg cost per baseline type (grouped bars)
# ─────────────────────────────────────────────────────────────────────────────
groups = ["v1 MIDAS", "CR Signature"]
avg_gain  = []
avg_cost  = []
max_gain  = []
n_runs    = []

for g in groups:
    grp = [r for r in records if r["baseline_type"] == g and r["gain_pp"] is not None]
    avg_gain.append(np.mean([r["gain_pp"] for r in grp]) if grp else 0)
    max_gain.append(max([r["gain_pp"] for r in grp]) if grp else 0)
    avg_cost.append(np.mean([opt_cost_usd(r) for r in grp]) if grp else 0)
    n_runs.append(len(grp))

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4.5))
fig.patch.set_facecolor("#F7F9FC")
for ax_ in [ax_l, ax_r]:
    ax_.set_facecolor("#F7F9FC")

x = np.arange(len(groups))
cols = [COLOUR[g] for g in groups]

# Left: Gain
bars_avg = ax_l.bar(x - 0.2, avg_gain, 0.35, label="Avg Gain (pp)", color=cols, alpha=0.85, edgecolor="white")
bars_max = ax_l.bar(x + 0.2, max_gain, 0.35, label="Max Gain (pp)", color=cols, alpha=0.45, edgecolor="white", hatch="//")
for i, (av, mx) in enumerate(zip(avg_gain, max_gain)):
    ax_l.text(i - 0.2, av + 0.1, f"{av:.1f}pp", ha="center", fontsize=9, fontweight="bold")
    ax_l.text(i + 0.2, mx + 0.1, f"{mx:.1f}pp", ha="center", fontsize=9)
ax_l.set_xticks(x)
ax_l.set_xticklabels([f"{g}\n(n={n_runs[i]})" for i, g in enumerate(groups)], fontsize=10)
ax_l.set_ylabel("DCA Score Gain (pp)", fontsize=10)
ax_l.set_title("Performance Gain\nBaseline → Best Optimized", fontsize=10, fontweight="bold")
ax_l.legend(fontsize=8)
ax_l.grid(axis="y", alpha=0.3, ls=":")
ax_l.set_ylim(0, max(max_gain) * 1.3)

# Right: Cost
bars_cost = ax_r.bar(x, avg_cost, 0.5, color=cols, alpha=0.85, edgecolor="white")
for i, c in enumerate(avg_cost):
    ax_r.text(i, c + 0.3, f"${c:.0f}", ha="center", fontsize=10, fontweight="bold")
ax_r.set_xticks(x)
ax_r.set_xticklabels([f"{g}\n(n={n_runs[i]})" for i, g in enumerate(groups)], fontsize=10)
ax_r.set_ylabel("Avg Optimization Cost (USD, one-time)", fontsize=10)
ax_r.set_title("Avg One-Time Optimization Cost (USD)\nper full tournament run", fontsize=10, fontweight="bold")
ax_r.grid(axis="y", alpha=0.3, ls=":")
ax_r.set_ylim(0, max(avg_cost) * 1.4)

plt.suptitle("Optimization Investment vs Gain — v1 MIDAS vs CR Signature Baseline",
             fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout()
out5 = OUT_DIR / "chart5_summary_gain_vs_cost.png"
plt.savefig(out5, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out5}")


# ── Print summary table ──────────────────────────────────────────────────────
print("\n" + "="*90)
print(f"{'Tournament':<22} {'Type':<14} {'Model':<14} {'OptTok(M)':>10} {'Cost($)':>8} "
      f"{'Base%':>7} {'Best%':>7} {'Gain pp':>8} {'InfSaved':>9}")
print("="*90)
for r in sorted(records, key=lambda r: r["baseline_type"]):
    cost = opt_cost_usd(r)
    print(f"{r['id'][-22:]:<22} {r['baseline_type']:<14} {r['model']:<14} "
          f"{r['total_opt_tok']/1e6:>10.1f} {cost:>8.1f} "
          f"{(r['base_score'] or 0):>7.2f} {r['best_score']:>7.2f} "
          f"{(r['gain_pp'] or 0):>+8.2f} {r['inf_tok_saved']:>+9.0f}")
print("="*90)

print(f"\nOutputs saved to: {OUT_DIR}")
print("Files:")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  {f.name}")
