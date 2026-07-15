"""
Final focused chart for the slide:
  Left  — Whole-tournament cost vs gain  (t_4953 gpt-4.1  vs  t_20260504233822 gpt-4.1-mini)
  Right — GEPA single-stage duration vs gain across all available runs
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

PRICE = {"gpt-4.1": {"in": 2.00, "out": 8.00},
         "gpt-4.1-mini": {"in": 0.40, "out": 1.60}}

STAGE_COL = {"bootstrap": "#5B9BD5", "miprov2": "#ED7D31", "gepa": "#70AD47"}
V1_COL, CR_COL = "#E05A2B", "#2B72C8"

# GEPA medium token estimate: 30 iter × (40 val × 4k tok + 5 reflect × 6k tok) × 1.5 merge overhead
# = 30 × (160k + 30k) × 1.5 = 8.55M ≈ 8M tokens
# Split: ~85% input (eval prompts + reflection input), ~15% output (generated instructions)
GEPA_MED_EST_IN  = 8.0 * 0.85   # M tokens input (estimated)
GEPA_MED_EST_OUT = 8.0 * 0.15   # M tokens output (estimated)

# ── helper ───────────────────────────────────────────────────────────────────
def tok_cost(prompt_tok, comp_tok, model):
    p = PRICE.get(model, PRICE["gpt-4.1-mini"])
    return (prompt_tok * p["in"] + comp_tok * p["out"]) / 1e6

def load_leaderboard(t_dir):
    p = t_dir / "leaderboard.json"
    return json.loads(p.read_text(encoding="utf-8")).get("entries", []) if p.exists() else []

def get_model(t_dir):
    sp = t_dir / "summary.json"
    if sp.exists():
        m = json.loads(sp.read_text(encoding="utf-8")).get("config", {}).get("student_model_name", "")
        return "gpt-4.1-mini" if "mini" in m else "gpt-4.1"
    return "gpt-4.1"

def sum_stage_tokens(t_dir):
    stg = {}
    for f in t_dir.rglob("*optimization_tokens*.json"):
        tok = json.loads(f.read_text(encoding="utf-8"))
        sf = f.parent.name
        key = "bootstrap" if "bootstrap" in sf else ("miprov2" if "miprov2" in sf else "gepa")
        prev = stg.get(key, {"p": 0, "c": 0, "t": 0})
        stg[key] = {
            "p": prev["p"] + tok.get("prompt_tokens", 0),
            "c": prev["c"] + tok.get("completion_tokens", 0),
            "t": prev["t"] + tok.get("total_tokens", 0),
        }
    return stg

# ══════════════════════════════════════════════════════════════════════════════
# DATA — Part 1: Two whole tournaments
# ══════════════════════════════════════════════════════════════════════════════
TOURNAMENTS = [
    ("t_4953",                "gpt-4.1",      "v1 MIDAS\n(weaker start)"),
    ("tournament_20260504233822", "gpt-4.1-mini", "CR Signature\n(strong start)"),
]

tour_rows = []
for t_name, model, label in TOURNAMENTS:
    t_dir   = BASE_DIR / t_name
    entries = load_leaderboard(t_dir)
    stg     = sum_stage_tokens(t_dir)

    bl   = next((e for e in entries if e.get("config_name") == "seed_baseline_bare"), None)
    base = bl["val_total_score"] * 100 if bl else None

    # GEPA duration (max across gepa entries in this tournament)
    gepa_dur_h = max(
        (e.get("metadata", {}).get("compile_duration_seconds") or 0) / 3600
        for e in entries if e.get("stage") == "gepa"
    ) if any(e.get("stage") == "gepa" for e in entries) else 0

    non_base = [e for e in entries if "baseline" not in e.get("config_name", "")]
    best_e   = max(non_base, key=lambda e: e.get("val_total_score", 0)) if non_base else None
    best     = best_e["val_total_score"] * 100 if best_e else None
    best_stg = best_e["stage"] if best_e else ""

    boot_p = stg.get("bootstrap", {}).get("p", 0)
    boot_c = stg.get("bootstrap", {}).get("c", 0)
    mipr_p = stg.get("miprov2",   {}).get("p", 0)
    mipr_c = stg.get("miprov2",   {}).get("c", 0)

    tour_rows.append({
        "label":       label,
        "model":       model,
        "col":         V1_COL if "v1" in label else CR_COL,
        "boot_in":     boot_p / 1e6,
        "boot_out":    boot_c / 1e6,
        "mipr_in":     mipr_p / 1e6,
        "mipr_out":    mipr_c / 1e6,
        "gepa_dur_h":  gepa_dur_h,
        "base":        base,
        "best":        best,
        "best_stg":    best_stg,
        "gain":        (best - base) if base and best else 0,
    })
    print(f"[{t_name}] model={model}  base={base:.1f}%  best={best:.1f}%  "
          f"gain={best-base:+.1f}pp  "
          f"boot_in={boot_p/1e6:.1f}M  boot_out={boot_c/1e6:.1f}M  "
          f"mipr_in={mipr_p/1e6:.1f}M  mipr_out={mipr_c/1e6:.1f}M  "
          f"gepa_dur={gepa_dur_h:.1f}h")

# ══════════════════════════════════════════════════════════════════════════════
# DATA — Part 2: All GEPA single-runs with compile_duration
# ══════════════════════════════════════════════════════════════════════════════
V1_TOURNAMENTS = {"t_4953"}
gepa_rows = []

for lb_path in sorted(BASE_DIR.glob("*/leaderboard.json")):
    t_dir   = lb_path.parent
    t_name  = t_dir.name
    entries = json.loads(lb_path.read_text(encoding="utf-8")).get("entries", [])
    if not entries:
        continue

    bl = next((e for e in entries if e.get("config_name") == "seed_baseline_bare"), None)
    base_score = bl["val_total_score"] * 100 if bl else None
    model_s = get_model(t_dir)
    btype   = "v1 MIDAS" if t_name in V1_TOURNAMENTS else "CR Signature"

    for e in entries:
        if e.get("stage") != "gepa":
            continue
        meta = e.get("metadata") or {}
        dur  = meta.get("compile_duration_seconds") or meta.get("optimization_duration_seconds")
        if not dur or dur <= 0:
            continue
        gepa_score = e["val_total_score"] * 100
        gain       = gepa_score - base_score if base_score else None
        # parent gain
        parent_cfg  = Path(e.get("parent_path", "")).stem
        parent_e    = next((x for x in entries if x.get("config_name") == parent_cfg), None)
        parent_score = parent_e["val_total_score"] * 100 if parent_e else None
        gain_par     = gepa_score - parent_score if parent_score else None

        gepa_rows.append({
            "tournament":  t_name,
            "model":       model_s,
            "btype":       btype,
            "dur_h":       dur / 3600,
            "base":        base_score,
            "gepa_score":  gepa_score,
            "gain":        gain,
            "gain_par":    gain_par,
        })

print(f"\nGEPA runs with duration: {len(gepa_rows)}")
for btype in ["v1 MIDAS", "CR Signature"]:
    g = [r for r in gepa_rows if r["btype"] == btype and r["gain"] is not None]
    if g:
        print(f"  [{btype}] n={len(g)}  avg_gain={np.mean([r['gain'] for r in g]):+.1f}pp  "
              f"max_gain={max(r['gain'] for r in g):+.1f}pp  "
              f"avg_dur={np.mean([r['dur_h'] for r in g]):.1f}h")

# ══════════════════════════════════════════════════════════════════════════════
# CHART
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 7.5))
fig.patch.set_facecolor("#F7F9FC")

ax1 = fig.add_subplot(1, 3, 1)   # input/output tokens per stage
ax2 = fig.add_subplot(1, 3, 2)   # GEPA scatter
ax3 = fig.add_subplot(1, 3, 3)   # total token budget vs gain
for ax in [ax1, ax2, ax3]:
    ax.set_facecolor("#F7F9FC")

# ── LEFT: Grouped bars per tournament — input vs output tokens, split by stage ──
# Layout: for each tournament, 2 groups (Bootstrap, MIPROv2), each group = [input, output]
# Plus a GEPA duration annotation

n_tours  = len(tour_rows)
n_stages = 3          # Bootstrap, MIPROv2, GEPA (estimated)
group_w  = 0.35
bar_w    = group_w / 2 - 0.02
gap      = 0.55        # gap between tournaments

positions = []
group_centers = {}    # (tour_idx, stage) -> center x
for ti in range(n_tours):
    base_x = ti * (n_stages * group_w + gap)
    for si, stage in enumerate(["bootstrap", "miprov2", "gepa"]):
        cx = base_x + si * group_w
        group_centers[(ti, stage)] = cx
        positions.append((ti, stage, cx))

IN_ALPHA  = 1.0
OUT_ALPHA = 0.55

for ti, r in enumerate(tour_rows):
    for stage, in_val, out_val, is_est in [
        ("bootstrap", r["boot_in"],      r["boot_out"],      False),
        ("miprov2",   r["mipr_in"],      r["mipr_out"],      False),
        ("gepa",      GEPA_MED_EST_IN,   GEPA_MED_EST_OUT,   True),
    ]:
        cx  = group_centers[(ti, stage)]
        col = STAGE_COL[stage]
        hin  = "///" if is_est else None
        hout = "///..." if is_est else "..."
        ax1.bar(cx - bar_w/2, in_val,  bar_w, color=col, alpha=IN_ALPHA,
                edgecolor="white", linewidth=0.8, hatch=hin)
        ax1.bar(cx + bar_w/2, out_val, bar_w, color=col, alpha=OUT_ALPHA,
                edgecolor="white", linewidth=0.8, hatch=hout)
        suf = "*" if is_est else ""
        if in_val > 0.3:
            ax1.text(cx - bar_w/2, in_val + 0.1, f"{in_val:.1f}M{suf}",
                     ha="center", va="bottom", fontsize=7, color="#333")
        if out_val > 0.3:
            ax1.text(cx + bar_w/2, out_val + 0.1, f"{out_val:.1f}M{suf}",
                     ha="center", va="bottom", fontsize=7, color="#333")

    boot_cx = group_centers[(ti, "bootstrap")]
    gepa_cx = group_centers[(ti, "gepa")]
    tour_cx = (boot_cx + gepa_cx) / 2
    max_val = max(r["boot_in"], r["boot_out"], r["mipr_in"], r["mipr_out"], GEPA_MED_EST_IN)

    if r["gepa_dur_h"] > 0:
        ax1.text(gepa_cx, GEPA_MED_EST_IN + 0.3,
                 f"~{r['gepa_dur_h']:.1f}h",
                 ha="center", va="bottom", fontsize=7, color=STAGE_COL["gepa"], style="italic")

    gain_str = f"+{r['gain']:.1f} pp" if r["gain"] >= 0 else f"{r['gain']:.1f} pp"
    gain_col = "#C00000" if r["gain"] >= 0 else "#888"
    ax1.text(tour_cx, max_val + 2.0,
             f"{gain_str}  ({r['base']:.0f}%→{r['best']:.0f}%)\nbest stage: {r['best_stg']}",
             ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=gain_col)

    ax1.text(tour_cx, -1.2, r["model"], ha="center", va="top", fontsize=9, color="#555")

# Stage group labels (Bootstrap / MIPROv2) on x-axis
xtick_pos, xtick_lbl = [], []
for ti, r in enumerate(tour_rows):
    for stage, lbl in [("bootstrap", "Bootstrap"), ("miprov2", "MIPROv2"), ("gepa", "GEPA*")]:
        cx = group_centers[(ti, stage)]
        xtick_pos.append(cx)
        xtick_lbl.append(lbl)
    cx0 = group_centers[(ti, "bootstrap")]
    cx1 = group_centers[(ti, "gepa")]
    ax1.text((cx0 + cx1) / 2, -2.8, r["label"],
             ha="center", va="top", fontsize=10, fontweight="bold", color=r["col"])

ax1.set_xticks(xtick_pos)
ax1.set_xticklabels(xtick_lbl, fontsize=9)
ax1.set_ylabel("Optimization Tokens (millions)", fontsize=10)
ax1.set_title("Optimization Input vs Output Tokens per Stage\n(Bootstrap & MIPROv2 tracked  |  GEPA* = ~8M estimated)",
              fontsize=10.5, fontweight="bold")
ax1.set_xlim(-0.35, group_centers[(1, "gepa")] + 0.4)
ax1.set_ylim(-4.5, max(max(r["boot_in"], r["mipr_in"]) for r in tour_rows) + 11)
ax1.grid(axis="y", alpha=0.3, ls=":")

in_patch   = mpatches.Patch(color="#888", alpha=IN_ALPHA,  label="Input tokens (prompt)")
out_patch  = mpatches.Patch(color="#888", alpha=OUT_ALPHA, hatch="...", label="Output tokens (completion)")
boot_patch = mpatches.Patch(color=STAGE_COL["bootstrap"], label="Bootstrap stage")
mipr_patch = mpatches.Patch(color=STAGE_COL["miprov2"],   label="MIPROv2 stage")
gepa_patch = mpatches.Patch(color=STAGE_COL["gepa"], hatch="///", label="GEPA* (estimated, not tracked)")
ax1.legend(handles=[in_patch, out_patch, boot_patch, mipr_patch, gepa_patch],
           fontsize=8, loc="upper right", framealpha=0.88)

# ── RIGHT: GEPA scatter — duration vs gain ──
valid = [r for r in gepa_rows if r["gain"] is not None]

for r in valid:
    col  = V1_COL if r["btype"] == "v1 MIDAS" else CR_COL
    sz   = 140 if r["model"] == "gpt-4.1" else 60
    mk   = "D" if r["btype"] == "v1 MIDAS" else "o"
    alpha = 0.85 if r["gain"] > 0 else 0.45
    ax2.scatter(r["dur_h"], r["gain"], color=col, marker=mk, s=sz,
                edgecolors="white", linewidths=0.6, zorder=3, alpha=alpha)

# Average lines + mean markers per group
for btype, col in [("v1 MIDAS", V1_COL), ("CR Signature", CR_COL)]:
    g = [r for r in valid if r["btype"] == btype and r["gain"] is not None]
    if not g:
        continue
    avg_gain = np.mean([r["gain"] for r in g])
    avg_dur  = np.mean([r["dur_h"] for r in g])
    ax2.axhline(avg_gain, color=col, lw=1.4, ls="--", alpha=0.45)
    ax2.scatter(avg_dur, avg_gain, color=col, marker="*", s=300, edgecolors="#111",
                linewidths=0.8, zorder=6,
                label=f"{btype} (n={len(g)})\navg: {avg_dur:.1f}h, +{avg_gain:.1f} pp gain")

ax2.axhline(0, color="#aaa", lw=1, ls=":")
ax2.fill_between([0, 15], 0, 35, alpha=0.04, color="green")
ax2.text(0.4, 32, "Gain zone", color="green", fontsize=9, alpha=0.6)
ax2.text(0.4, -6.5, "Regression zone", color="#888", fontsize=8.5)

ax2.set_xlabel("GEPA Compile Duration (hours, proxy for cost)", fontsize=10)
ax2.set_ylabel("DCA Score Gain vs Tournament Baseline (pp)", fontsize=10)
ax2.set_title("GEPA Alone: Runtime vs Gain\n(* = group average  |  diamond = v1 MIDAS  |  circle = CR Sig.)",
              fontsize=11, fontweight="bold")
ax2.legend(fontsize=8.5, loc="upper right", framealpha=0.88)
ax2.grid(True, alpha=0.3, ls=":")
ax2.set_xlim(0, 13)
ax2.set_ylim(-10, 36)

# Add note about gpt-4.1 vs mini
size_note = ("Large dot = gpt-4.1   Small dot = gpt-4.1-mini\n"
             "GEPA tokens not tracked (subprocess isolation)")
fig.text(0.50, 0.01, size_note, ha="center", fontsize=7.5, color="#777", style="italic")

# ── PANEL 3: Total opt token budget vs gain per tournament ──────────────────
GEPA_MED_EST_TOTAL = GEPA_MED_EST_IN + GEPA_MED_EST_OUT   # 8.0M

all_x_vals = []
for r in tour_rows:
    total_tok_m = r["boot_in"] + r["boot_out"] + r["mipr_in"] + r["mipr_out"] + GEPA_MED_EST_TOTAL
    all_x_vals.append(total_tok_m)
    gain = r["gain"]
    col  = r["col"]
    mk   = "D" if "v1" in r["label"] else "o"

    ax3.scatter(total_tok_m, gain, color=col, marker=mk, s=220,
                edgecolors="white", linewidths=1.2, zorder=4)

    va = "bottom" if gain >= 0 else "top"
    ax3.annotate(
        f"{r['model']}\n{r['label'].split(chr(10))[0]}\n"
        f"~{total_tok_m:.0f}M tok (incl. GEPA est.)\nGain: {gain:+.1f} pp",
        xy=(total_tok_m, gain),
        xytext=(0, 22 if gain >= 0 else -22),
        textcoords="offset points",
        ha="center", va=va,
        fontsize=9, color=col, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=col, lw=1.2),
    )

# Manual CR Signature standalone GEPA: 63.9% → 87%  (no full tournament, only GEPA)
cr_x    = GEPA_MED_EST_TOTAL   # only GEPA tokens (no Bootstrap/MIPROv2)
cr_gain = 87.0 - 63.9
all_x_vals.append(cr_x)
ax3.scatter(cr_x, cr_gain, color=CR_COL, marker="s", s=200,
            edgecolors="white", linewidths=1.2, zorder=4)
ax3.annotate(
    f"GEPA only\nCR Signature\n63.9%→87%\n~{cr_x:.0f}M tok (GEPA est.)\nGain: {cr_gain:+.1f} pp",
    xy=(cr_x, cr_gain),
    xytext=(-60, 15),
    textcoords="offset points",
    ha="center", va="bottom",
    fontsize=9, color=CR_COL, fontweight="bold",
    arrowprops=dict(arrowstyle="-", color=CR_COL, lw=1.2),
)

x_max = max(all_x_vals) * 1.35
ax3.axhline(0, color="#aaa", lw=1.2, ls="--")
ax3.fill_between([0, x_max], 0, 26, alpha=0.05, color="green")
ax3.text(0.5, 22, "Gain zone", color="green", fontsize=9, alpha=0.65)
ax3.text(0.5, -3, "No gain / regression", color="#888", fontsize=8.5)

v1_p = mpatches.Patch(color=V1_COL, label="v1 MIDAS baseline  (diamond)")
cr_p = mpatches.Patch(color=CR_COL, label="CR Signature baseline  (circle / square)")
ax3.legend(handles=[v1_p, cr_p], fontsize=9, loc="center right", framealpha=0.88)

ax3.set_xlabel("Total Optimization Token Budget (M)\nBootstrap + MIPROv2 + GEPA* (~8M est., all stages)", fontsize=10)
ax3.set_ylabel("Performance Gain: Baseline → Best (pp)", fontsize=10)
ax3.set_title("Tournament Token Budget vs Performance Gain\n(diamond/circle = full tournament  |  square = GEPA only)",
              fontsize=11, fontweight="bold")
ax3.grid(True, alpha=0.3, ls=":")
ax3.set_xlim(0, x_max)
ax3.set_ylim(-8, 30)

plt.suptitle("Is Prompt Optimization Worth It?  |  DCA (Document Change Analysis)  |  DSPy Tournament Pipeline",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout(rect=[0, 0.03, 1, 1])
out = OUT_DIR / "chart_final_opt_cost_vs_gain.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")

# ── Key numbers for slide ─────────────────────────────────────────────────────
print("\nKEY NUMBERS FOR SLIDE")
print("="*65)
for r in tour_rows:
    total_in  = r["boot_in"]  + r["mipr_in"]
    total_out = r["boot_out"] + r["mipr_out"]
    print(f"\n{r['model']}  [{r['label'].replace(chr(10),' ')}]")
    print(f"  Bootstrap  input : {r['boot_in']:.1f}M   output: {r['boot_out']:.1f}M")
    print(f"  MIPROv2    input : {r['mipr_in']:.1f}M   output: {r['mipr_out']:.1f}M")
    print(f"  Total      input : {total_in:.1f}M   output: {total_out:.1f}M   total: {total_in+total_out:.1f}M")
    print(f"  GEPA       ~{r['gepa_dur_h']:.1f}h compile  (tokens not tracked)")
    print(f"  Baseline   : {r['base']:.1f}%")
    print(f"  Best result: {r['best']:.1f}%  (stage: {r['best_stg']})")
    print(f"  Gain       : {r['gain']:+.1f} pp")

print("\nGEPA average gain by baseline type:")
for btype in ["v1 MIDAS", "CR Signature"]:
    g = [r for r in gepa_rows if r["btype"] == btype and r["gain"] is not None]
    if g:
        pos = [r for r in g if r["gain"] > 0]
        print(f"  {btype}: avg {np.mean([r['gain'] for r in g]):+.1f}pp "
              f"| positive runs: {len(pos)}/{len(g)} "
              f"| max: {max(r['gain'] for r in g):+.1f}pp "
              f"| avg dur: {np.mean([r['dur_h'] for r in g]):.1f}h")
