"""
Standalone right panel:
  gpt-4.1 | v1 MIDAS | total optimization token budget vs performance gain
  Prompt tokens and completion tokens shown separately.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

OUT_DIR  = Path(r"C:\Users\BHS1RNG\OneDrive - Bosch Group\AIQ-212\Code\meta-prompting\analysis_output")
T_DIR    = Path(r"C:\Users\BHS1RNG\OneDrive - Bosch Group\AIQ-212\Code\meta-prompting\.cache\tournament\t_4953")
OUT_DIR.mkdir(exist_ok=True)

# ── load counts + config from summary.json ────────────────────────────────────
import json
summary  = json.loads((T_DIR / "summary.json").read_text(encoding="utf-8"))
cfg      = summary["config"]
per_stg  = summary["per_stage"]

N_BOOT   = per_stg["bootstrap"]["n_candidates"]     # 2
N_MIPR   = per_stg["miprov2"]["n_candidates"]       # 2
N_GEPA   = per_stg["gepa"]["n_candidates"]          # 1

# Bootstrap config details
bc       = cfg["bootstrap_configs"][0]              # single config
BOOT_N   = bc["n"]                                  # 16 examples bootstrapped
BOOT_RDS = bc["max_rounds"]                         # 2 rounds

# MIPROv2 intensity
MIPR_INT = cfg["miprov2_intensities"][0]            # "medium"

# GEPA config
gc       = cfg["gepa_configs"][0]
GEPA_AUTO = gc["auto"]                              # "medium"
GEPA_DUR_H = per_stg["gepa"]["top_k_entries"][0]["optimization_duration_seconds"] / 3600

# ── gpt-4.1 v1 MIDAS token data (from tournament t_4953) ──────────────────────
BOOT_IN   = 10.7   # M — Bootstrap prompt tokens   (tracked)
BOOT_OUT  = 0.1    # M — Bootstrap completion tokens (tracked)
MIPR_IN   = 11.5   # M — MIPROv2  prompt tokens   (tracked)
MIPR_OUT  = 0.1    # M — MIPROv2  completion tokens (tracked)
# GEPA: dspy.track_usage() incompatible with subprocess ParallelExecutor → all zeros
# Estimated: 30 iter × (40 val×4k + 5 reflect×6k) × 1.5 merge overhead ≈ 8M total
# GEPA token estimate — derived from actual log data:
#   895 student rollouts × avg 8,633 input tok  = 7.73M student prompt tokens
#   895 student rollouts × avg 46   output tok  = 0.04M student completion tokens
#   + teacher/reflection (gpt-5.2): 46 iter × (5 fail×~8k + instruct ~3k) ≈ 1.8M prompt + 0.3M completion
# Total: ~9.5M prompt  +  ~0.35M completion  ≈  9.85M  (token tracking disabled in subprocess)
GEPA_IN   = 9.5    # M — GEPA prompt tokens   (log-derived estimate)
GEPA_OUT  = 0.35   # M — GEPA completion tokens (log-derived estimate)

TOTAL_IN  = BOOT_IN  + MIPR_IN  + GEPA_IN
TOTAL_OUT = BOOT_OUT + MIPR_OUT + GEPA_OUT
TOTAL     = TOTAL_IN + TOTAL_OUT

BASE_SCORE = 72.3   # %
BEST_SCORE = 92.3   # %
GAIN       = BEST_SCORE - BASE_SCORE   # +20.0 pp

RED = "#E05A2B"

STAGE_COL = {
    "bootstrap": "#5B9BD5",
    "miprov2":   "#ED7D31",
    "gepa":      "#70AD47",
}

# ── Figure layout: left = token breakdown bars, right = gain summary ──────────
fig = plt.figure(figsize=(13, 6))
fig.patch.set_facecolor("#F7F9FC")
gs  = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.15)

ax_tok = fig.add_subplot(gs[0])   # token breakdown
ax_gn  = fig.add_subplot(gs[1])   # gain summary
for ax in [ax_tok, ax_gn]:
    ax.set_facecolor("#F7F9FC")

# ── LEFT: grouped bars — prompt vs completion, per stage ──────────────────────
stages = [
    ("bootstrap", "Bootstrap",
     BOOT_IN,  BOOT_OUT,  False,
     f"{N_BOOT} runs  |  n={BOOT_N} examples  |  {BOOT_RDS} rounds"),
    ("miprov2",   "MIPROv2",
     MIPR_IN,  MIPR_OUT,  False,
     f"{N_MIPR} runs  |  intensity: {MIPR_INT}"),
    ("gepa",      "GEPA*",
     GEPA_IN,  GEPA_OUT,  True,
     f"{N_GEPA} run  |  auto={GEPA_AUTO}  |  46 iters  |  895 rollouts  |  ~{GEPA_DUR_H:.1f}h"),
]

x    = np.arange(len(stages))
bw   = 0.32

for i, (key, lbl, in_v, out_v, is_est, run_info) in enumerate(stages):
    col   = STAGE_COL[key]
    hin   = "///" if is_est else None
    hout  = "///..." if is_est else "..."
    suf   = "*" if is_est else ""

    ax_tok.bar(x[i] - bw/2, in_v,  bw, color=col, alpha=0.95,
               edgecolor="white", linewidth=0.9, hatch=hin, label=f"{lbl} — prompt" if i == 0 else "_")
    ax_tok.bar(x[i] + bw/2, out_v, bw, color=col, alpha=0.50,
               edgecolor="white", linewidth=0.9, hatch=hout, label=f"{lbl} — completion" if i == 0 else "_")

    # value labels on bars
    ax_tok.text(x[i] - bw/2, in_v  + 0.15, f"{in_v:.1f}M{suf}",
                ha="center", va="bottom", fontsize=8.5, color="#222")
    if out_v > 0.05:
        ax_tok.text(x[i] + bw/2, out_v + 0.15, f"{out_v:.1f}M{suf}",
                    ha="center", va="bottom", fontsize=8.5, color="#222")

    # stage name label
    ax_tok.text(x[i], -0.8, lbl, ha="center", va="top", fontsize=10.5, fontweight="bold",
                color=col)
    # run count / config detail below stage name
    ax_tok.text(x[i], -1.8, run_info, ha="center", va="top", fontsize=7.8,
                color="#555", style="italic")

# Totals row at top
ax_tok.text(-0.5, 14.0,
            f"Total prompt:  {TOTAL_IN:.1f}M\nTotal completion:  {TOTAL_OUT:.1f}M\nTotal:  {TOTAL:.1f}M",
            fontsize=9.5, color="#333", va="top",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#ccc", lw=0.8))

# Legend
in_p  = mpatches.Patch(color="#888", alpha=0.95,             label="Prompt tokens (input)")
out_p = mpatches.Patch(color="#888", alpha=0.50, hatch="...", label="Completion tokens (output)")
b_p   = mpatches.Patch(color=STAGE_COL["bootstrap"],          label="Bootstrap")
m_p   = mpatches.Patch(color=STAGE_COL["miprov2"],            label="MIPROv2")
g_p   = mpatches.Patch(color=STAGE_COL["gepa"], hatch="///",  label="GEPA* (estimated)")
ax_tok.legend(handles=[in_p, out_p, b_p, m_p, g_p],
              fontsize=8, loc="upper right", framealpha=0.92)

ax_tok.set_xticks(x)
ax_tok.set_xticklabels(["", "", ""])   # labels drawn manually below
ax_tok.set_ylabel("Optimization Tokens (millions)", fontsize=11)
ax_tok.set_ylim(-3.5, 18)
ax_tok.set_xlim(-0.6, 2.6)
ax_tok.grid(axis="y", alpha=0.3, ls=":")
ax_tok.set_title(
    "Optimization Token Budget  |  gpt-4.1  |  v1 MIDAS Baseline\n"
    "Prompt vs Completion tokens per stage  (* GEPA log-derived: 895 rollouts × 8,633 tok/sample + teacher)",
    fontsize=10.5, fontweight="bold"
)

# ── RIGHT: gain summary ────────────────────────────────────────────────────────
ax_gn.set_xlim(0, 1)
ax_gn.set_ylim(0, 1)
ax_gn.axis("off")

# Background box
bg = mpatches.FancyBboxPatch((0.05, 0.05), 0.90, 0.90,
                              boxstyle="round,pad=0.02",
                              fc="#FFF4EF", ec=RED, lw=2.0,
                              transform=ax_gn.transAxes, zorder=1)
ax_gn.add_patch(bg)

ax_gn.text(0.5, 0.90, "Performance Gain", ha="center", va="top",
           fontsize=11, fontweight="bold", color="#444", transform=ax_gn.transAxes)

ax_gn.text(0.5, 0.76, f"{BASE_SCORE:.1f}%", ha="center", va="center",
           fontsize=18, color="#888", transform=ax_gn.transAxes)

ax_gn.annotate("", xy=(0.5, 0.58), xytext=(0.5, 0.68),
               xycoords="axes fraction", textcoords="axes fraction",
               arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.5,
                               mutation_scale=20))

ax_gn.text(0.5, 0.53, f"{BEST_SCORE:.1f}%", ha="center", va="center",
           fontsize=22, fontweight="bold", color=RED, transform=ax_gn.transAxes)

ax_gn.text(0.5, 0.38, f"+{GAIN:.1f} pp", ha="center", va="center",
           fontsize=26, fontweight="bold", color=RED, transform=ax_gn.transAxes)

ax_gn.plot([0.15, 0.85], [0.30, 0.30], color="#ddd", lw=1, transform=ax_gn.transAxes)

ax_gn.text(0.5, 0.23, f"Total budget:\n~{TOTAL:.1f}M tokens",
           ha="center", va="center", fontsize=10, color="#444", transform=ax_gn.transAxes,
           linespacing=1.6)
ax_gn.text(0.5, 0.10,
           f"{TOTAL_IN:.1f}M prompt\n+  {TOTAL_OUT:.1f}M completion",
           ha="center", va="center", fontsize=8.5, color="#666", transform=ax_gn.transAxes,
           linespacing=1.6)

plt.suptitle(
    "Is Prompt Optimization Worth It?  |  DCA (Document Change Analysis)  |  DSPy Full Tournament",
    fontsize=11, fontweight="bold", y=1.01
)

out = OUT_DIR / "chart_right_standalone.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

print(f"\nSummary:")
print(f"  gpt-4.1 | v1 MIDAS baseline")
print(f"  Bootstrap : {BOOT_IN:.1f}M prompt + {BOOT_OUT:.1f}M completion")
print(f"  MIPROv2   : {MIPR_IN:.1f}M prompt + {MIPR_OUT:.1f}M completion")
print(f"  GEPA*     : {GEPA_IN:.1f}M prompt + {GEPA_OUT:.1f}M completion  (estimated)")
print(f"  Total     : {TOTAL_IN:.1f}M prompt + {TOTAL_OUT:.1f}M completion = {TOTAL:.1f}M")
print(f"  Gain      : {BASE_SCORE:.1f}% -> {BEST_SCORE:.1f}% = +{GAIN:.1f} pp")
