"""
Focused analysis for slide:
  1. Two whole-tournament comparisons (gpt-4.1 t_4953  vs  gpt-4.1-mini t_20260504233822)
  2. Individual GEPA runs: GEPA opt tokens vs gain from tournament baseline
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE_DIR = Path(r"C:\Users\BHS1RNG\OneDrive - Bosch Group\AIQ-212\Code\meta-prompting\.cache\tournament")
OUT_DIR  = Path(r"C:\Users\BHS1RNG\OneDrive - Bosch Group\AIQ-212\Code\meta-prompting\analysis_output")
OUT_DIR.mkdir(exist_ok=True)

# Pricing ($/1M tokens)  gpt-4.1 input/output, gpt-4.1-mini input/output
PRICE = {
    "gpt-4.1":      {"in": 2.00, "out": 8.00},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
}

V1_MIDAS_TOURNAMENTS = {"t_4953"}


# ── helpers ──────────────────────────────────────────────────────────────────
def tok_cost(prompt_tok, comp_tok, model):
    p = PRICE.get(model, PRICE["gpt-4.1-mini"])
    return (prompt_tok * p["in"] + comp_tok * p["out"]) / 1e6


def sum_opt_tokens(t_dir: Path):
    """Sum all _optimization_tokens.json in a tournament dir, per stage."""
    stage_p, stage_c, stage_t = {}, {}, {}
    for f in t_dir.rglob("*optimization_tokens*.json"):
        tok = json.loads(f.read_text(encoding="utf-8"))
        sf = f.parent.name
        key = ("bootstrap" if "bootstrap" in sf
               else "miprov2" if "miprov2" in sf
               else "gepa")
        stage_t[key] = stage_t.get(key, 0) + tok.get("total_tokens", 0)
        stage_p[key] = stage_p.get(key, 0) + tok.get("prompt_tokens", 0)
        stage_c[key] = stage_c.get(key, 0) + tok.get("completion_tokens", 0)
    return stage_p, stage_c, stage_t


def get_baseline_and_best(t_dir: Path):
    """Return (baseline_score, best_non_baseline_score, model_short, stages_run)."""
    lb = json.loads((t_dir / "leaderboard.json").read_text(encoding="utf-8"))
    entries = lb.get("entries", [])
    bl = next((e for e in entries if e.get("config_name") == "seed_baseline_bare"), None)
    non_base = [e for e in entries if "baseline" not in e.get("config_name", "")]
    best = max(non_base, key=lambda e: e.get("val_total_score", 0)) if non_base else None
    model = ""
    sp = t_dir / "summary.json"
    if sp.exists():
        s = json.loads(sp.read_text(encoding="utf-8"))
        model = s.get("config", {}).get("student_model_name", "")
    model_short = "gpt-4.1-mini" if "mini" in model else "gpt-4.1"
    return (
        bl["val_total_score"] * 100 if bl else None,
        best["val_total_score"] * 100 if best else None,
        best["stage"] if best else "",
        model_short,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — Two whole tournaments
# ═══════════════════════════════════════════════════════════════════════════
T_GPT41      = BASE_DIR / "t_4953"
T_GPT41MINI  = BASE_DIR / "tournament_20260504233822"

tournaments_to_show = [
    ("t_4953\n(v1 MIDAS baseline)", T_GPT41,     "gpt-4.1",      "v1 MIDAS",     "#E05A2B"),
    ("20260504233822\n(CR Signature baseline)", T_GPT41MINI, "gpt-4.1-mini", "CR Signature", "#2B72C8"),
]

tour_data = []
for label, t_dir, model, btype, col in tournaments_to_show:
    sp, sc, st = sum_opt_tokens(t_dir)
    base, best, best_stage, _ = get_baseline_and_best(t_dir)
    total_p = sum(sp.values()); total_c = sum(sc.values()); total_t = sum(st.values())
    cost = tok_cost(total_p, total_c, model)
    gain = (best - base) if base and best else None
    tour_data.append({
        "label": label, "model": model, "btype": btype, "col": col,
        "stage_t": st, "stage_p": sp, "stage_c": sc,
        "total_tok": total_t, "cost_usd": cost,
        "base_score": base, "best_score": best, "best_stage": best_stage,
        "gain_pp": gain,
    })
    print(f"[{btype}] {model}")
    print(f"  Stages   : {dict(st)}")
    print(f"  Total tok: {total_t/1e6:.1f}M  Cost: ${cost:.1f}")
    print(f"  Baseline : {base:.2f}%   Best: {best:.2f}%   Gain: +{gain:.2f}pp")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — Individual GEPA runs across all available tournaments
# ═══════════════════════════════════════════════════════════════════════════
gepa_records = []

for gepa_tok_file in sorted(BASE_DIR.rglob("stage_3_gepa/*optimization_tokens*.json")):
    t_dir = gepa_tok_file.parent.parent    # tournament root
    t_name = t_dir.name

    tok = json.loads(gepa_tok_file.read_text(encoding="utf-8"))
    gepa_total = tok.get("total_tokens", 0)
    gepa_p     = tok.get("prompt_tokens", 0)
    gepa_c     = tok.get("completion_tokens", 0)
    if gepa_total == 0:
        continue

    config_name = gepa_tok_file.stem.replace("_optimization_tokens", "")

    lb_path = t_dir / "leaderboard.json"
    if not lb_path.exists():
        continue
    lb = json.loads(lb_path.read_text(encoding="utf-8"))
    entries = lb.get("entries", [])

    # Get this GEPA entry
    gepa_entry = next((e for e in entries if e.get("config_name") == config_name), None)
    if not gepa_entry:
        continue

    # Tournament baseline (seed_baseline_bare)
    bl = next((e for e in entries if e.get("config_name") == "seed_baseline_bare"), None)
    base_score = bl["val_total_score"] * 100 if bl else None

    # Parent entry score (what was fed into GEPA)
    parent_path = gepa_entry.get("parent_path", "")
    parent_config = Path(parent_path).stem if parent_path else None
    parent_entry = next((e for e in entries if e.get("config_name") == parent_config), None)
    parent_score = parent_entry["val_total_score"] * 100 if parent_entry else None

    gepa_score = gepa_entry["val_total_score"] * 100

    # Model
    model_str = ""
    sp = t_dir / "summary.json"
    if sp.exists():
        s = json.loads(sp.read_text(encoding="utf-8"))
        model_str = s.get("config", {}).get("student_model_name", "")
    model_short = "gpt-4.1-mini" if "mini" in model_str else "gpt-4.1"

    baseline_type = "v1 MIDAS" if t_name in V1_MIDAS_TOURNAMENTS else "CR Signature"

    gain_vs_base    = (gepa_score - base_score) if base_score else None
    gain_vs_parent  = (gepa_score - parent_score) if parent_score else None
    gepa_cost       = tok_cost(gepa_p, gepa_c, model_short)

    gepa_records.append({
        "tournament":      t_name,
        "config":          config_name,
        "model":           model_short,
        "baseline_type":   baseline_type,
        "gepa_tok":        gepa_total,
        "gepa_cost_usd":   gepa_cost,
        "base_score":      base_score,
        "parent_score":    parent_score,
        "gepa_score":      gepa_score,
        "gain_vs_base":    gain_vs_base,
        "gain_vs_parent":  gain_vs_parent,
    })

print(f"\nIndividual GEPA runs found: {len(gepa_records)}")
for r in sorted(gepa_records, key=lambda x: x["gepa_tok"]):
    print(f"  {r['tournament'][-8:]}  {r['model']:12s}  {r['baseline_type']:12s}  "
          f"gepa_tok={r['gepa_tok']/1e6:.1f}M  cost=${r['gepa_cost_usd']:.1f}  "
          f"base={r['base_score'] or 0:.1f}%  parent={r['parent_score'] or 0:.1f}%  "
          f"gepa={r['gepa_score']:.1f}%  gain_vs_base={r['gain_vs_base'] or 0:+.1f}pp  "
          f"gain_vs_parent={r['gain_vs_parent'] or 0:+.1f}pp")


# ═══════════════════════════════════════════════════════════════════════════
# CHART — Two-panel: whole tournament (left) + GEPA scatter (right)
# ═══════════════════════════════════════════════════════════════════════════
STAGE_COLOURS = {"bootstrap": "#5B9BD5", "miprov2": "#ED7D31", "gepa": "#70AD47"}
MTYPE_MARKER  = {"v1 MIDAS": "D", "CR Signature": "o"}
MTYPE_COL     = {"v1 MIDAS": "#E05A2B", "CR Signature": "#2B72C8"}

fig = plt.figure(figsize=(16, 7))
fig.patch.set_facecolor("#F7F9FC")

# Left panel — whole tournament stacked bar
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_facecolor("#F7F9FC")

x = np.arange(len(tour_data))
bar_w = 0.55

bottoms = [0, 0]
for stage in ["bootstrap", "miprov2", "gepa"]:
    vals = [d["stage_t"].get(stage, 0) / 1e6 for d in tour_data]
    bars = ax1.bar(x, vals, bar_w, bottom=bottoms, label=stage.capitalize(),
                   color=STAGE_COLOURS[stage], edgecolor="white", linewidth=0.8)
    bottoms = [bottoms[i] + vals[i] for i in range(len(vals))]

# Gain annotations on top
for i, d in enumerate(tour_data):
    total_m = d["total_tok"] / 1e6
    ax1.text(i, total_m + 0.4,
             f"+{d['gain_pp']:.1f} pp\n({d['base_score']:.0f}% → {d['best_score']:.0f}%)",
             ha="center", va="bottom", fontsize=9, fontweight="bold", color="#C00000")
    ax1.text(i, -1.8, f"${d['cost_usd']:.0f} one-time\n{d['total_tok']/1e6:.0f}M tokens",
             ha="center", va="top", fontsize=8.5, color="#444444")

ax1.set_xticks(x)
ax1.set_xticklabels([d["label"] for d in tour_data], fontsize=9.5)
ax1.set_ylabel("Optimization Tokens (millions, one-time)", fontsize=10)
ax1.set_title("Whole-Tournament Optimization Cost\n(Seed → Bootstrap → MIPROv2 → GEPA)",
              fontsize=11, fontweight="bold")
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(axis="y", alpha=0.3, ls=":")
ax1.set_ylim(-4, max(bottoms) * 1.25)
ax1.set_xlim(-0.55, 1.55)

# Right panel — GEPA individual scatter
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_facecolor("#F7F9FC")

valid_gepa = [r for r in gepa_records if r["gain_vs_base"] is not None]

for r in valid_gepa:
    col = MTYPE_COL[r["baseline_type"]]
    mk  = MTYPE_MARKER[r["baseline_type"]]
    sz  = 120 if r["model"] == "gpt-4.1" else 70
    ax2.scatter(r["gepa_tok"] / 1e6, r["gain_vs_base"],
                color=col, marker=mk, s=sz, edgecolors="white", linewidths=0.7, zorder=3, alpha=0.85)

# Averages per baseline type
for btype, col in MTYPE_COL.items():
    grp = [r for r in valid_gepa if r["baseline_type"] == btype]
    if not grp:
        continue
    avg_tok  = np.mean([r["gepa_tok"] / 1e6 for r in grp])
    avg_gain = np.mean([r["gain_vs_base"] for r in grp])
    max_gain = max(r["gain_vs_base"] for r in grp)
    ax2.scatter(avg_tok, avg_gain, color=col, marker="*", s=260, edgecolors="#222",
                linewidths=0.8, zorder=5, label=f"{btype} (n={len(grp)})\navg +{avg_gain:.1f}pp, max +{max_gain:.1f}pp")
    ax2.axhline(avg_gain, color=col, lw=1.2, ls="--", alpha=0.4)

# Model size legend items
m41_patch  = plt.Line2D([0],[0], marker="o", color="gray", markersize=9,  ls="none", label="student: gpt-4.1 (large dot)")
m41m_patch = plt.Line2D([0],[0], marker="o", color="gray", markersize=6,  ls="none", label="student: gpt-4.1-mini (small dot)")
ax2.axhline(0, color="#999", lw=0.8, ls=":")

ax2.set_xlabel("GEPA Optimization Tokens (millions, one-time)", fontsize=10)
ax2.set_ylabel("DCA Score Gain vs Tournament Baseline (pp)", fontsize=10)
ax2.set_title("GEPA Single-Stage Opt Tokens vs Gain\n(gain measured from tournament baseline to GEPA output)",
              fontsize=11, fontweight="bold")
ax2.legend(handles=ax2.get_legend_handles_labels()[0] + [m41_patch, m41m_patch],
           fontsize=8, loc="lower right", framealpha=0.85)
ax2.grid(True, alpha=0.3, ls=":")

plt.suptitle("Is Prompt Optimization Worth It?  — DCA (Document Change Analysis)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout(rect=[0, 0, 1, 1])
out = OUT_DIR / "chart_opt_cost_vs_gain_focused.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")


# ─── Summary numbers for the slide text box ──────────────────────────────────
print("\n" + "═"*70)
print("SLIDE KEY NUMBERS")
print("═"*70)
for d in tour_data:
    print(f"\n{d['model']}  [{d['btype']}]")
    print(f"  One-time optimization cost : {d['total_tok']/1e6:.0f}M tokens  ~${d['cost_usd']:.0f}")
    print(f"  Stages  : Bootstrap {d['stage_t'].get('bootstrap',0)/1e6:.1f}M  "
          f"| MIPROv2 {d['stage_t'].get('miprov2',0)/1e6:.1f}M  "
          f"| GEPA {d['stage_t'].get('gepa',0)/1e6:.1f}M")
    print(f"  Baseline: {d['base_score']:.1f}%  →  Best: {d['best_score']:.1f}%  (+{d['gain_pp']:.1f} pp)")

print("\nGEPA averages by baseline type:")
for btype in ["v1 MIDAS", "CR Signature"]:
    grp = [r for r in valid_gepa if r["baseline_type"] == btype]
    if not grp:
        continue
    print(f"  {btype} (n={len(grp)}):")
    print(f"    Avg GEPA opt tokens : {np.mean([r['gepa_tok'] for r in grp])/1e6:.1f}M")
    print(f"    Avg cost (GEPA only): ${np.mean([r['gepa_cost_usd'] for r in grp]):.1f}")
    print(f"    Avg gain vs baseline: +{np.mean([r['gain_vs_base'] for r in grp]):.1f}pp")
    print(f"    Max gain vs baseline: +{max(r['gain_vs_base'] for r in grp):.1f}pp")
    gvp = [r['gain_vs_parent'] for r in grp if r['gain_vs_parent'] is not None]
    if gvp:
        print(f"    Avg GEPA-only gain (vs its own input): +{np.mean(gvp):.1f}pp")
