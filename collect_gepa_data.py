"""Collect GEPA compile durations + gains across all tournaments."""
import json
from pathlib import Path

BASE_DIR = Path(r"C:\Users\BHS1RNG\OneDrive - Bosch Group\AIQ-212\Code\meta-prompting\.cache\tournament")
V1_TOURNAMENTS = {"t_4953"}

rows = []

for lb_path in sorted(BASE_DIR.glob("*/leaderboard.json")):
    t_dir  = lb_path.parent
    t_name = t_dir.name
    lb     = json.loads(lb_path.read_text(encoding="utf-8"))
    entries = lb.get("entries", [])
    if not entries:
        continue

    bl = next((e for e in entries if e.get("config_name") == "seed_baseline_bare"), None)
    base_score = bl["val_total_score"] * 100 if bl else None

    model = ""
    sp = t_dir / "summary.json"
    if sp.exists():
        s = json.loads(sp.read_text(encoding="utf-8"))
        model = s.get("config", {}).get("student_model_name", "")
    model_short = "gpt-4.1-mini" if "mini" in model else "gpt-4.1"
    btype = "v1 MIDAS" if t_name in V1_TOURNAMENTS else "CR Signature"

    for e in entries:
        if e.get("stage") != "gepa":
            continue
        meta = e.get("metadata") or {}
        dur = meta.get("compile_duration_seconds") or meta.get("optimization_duration_seconds")
        if not dur or dur <= 0:
            continue
        gepa_score  = e["val_total_score"] * 100
        parent_path = e.get("parent_path", "")
        parent_cfg  = Path(parent_path).stem if parent_path else None
        parent_e    = next((x for x in entries if x.get("config_name") == parent_cfg), None)
        parent_score = parent_e["val_total_score"] * 100 if parent_e else None

        rows.append({
            "tournament":     t_name,
            "config":         e.get("config_name", ""),
            "model":          model_short,
            "btype":          btype,
            "dur_h":          round(dur / 3600, 2),
            "dur_min":        round(dur / 60, 1),
            "base_score":     round(base_score, 2) if base_score else None,
            "parent_score":   round(parent_score, 2) if parent_score else None,
            "gepa_score":     round(gepa_score, 2),
            "gain_vs_base":   round(gepa_score - base_score, 2) if base_score else None,
            "gain_vs_parent": round(gepa_score - parent_score, 2) if parent_score else None,
        })

rows.sort(key=lambda r: r["dur_h"])
print(f"GEPA runs with compile_duration: {len(rows)}")
print(f"{'Tournament':13} {'Model':12} {'BType':12} {'Dur(h)':>7} {'Dur(m)':>7} "
      f"{'Base%':>7} {'Parent%':>8} {'GEPA%':>7} {'GainBase':>9} {'GainPar':>8}")
print("-"*100)
for r in rows:
    print(f"{r['tournament'][-13:]:13} {r['model']:12} {r['btype']:12} {r['dur_h']:>7.2f} "
          f"{r['dur_min']:>7.1f} {r['base_score'] or 0:>7.2f} "
          f"{r['parent_score'] or 0:>8.2f} {r['gepa_score']:>7.2f} "
          f"{r['gain_vs_base'] or 0:>+9.2f} {r['gain_vs_parent'] or 0:>+8.2f}")

print("\nAverages by baseline type:")
for btype in ["v1 MIDAS", "CR Signature"]:
    g = [r for r in rows if r["btype"] == btype and r["gain_vs_base"] is not None]
    if not g:
        continue
    import statistics
    print(f"  [{btype}] n={len(g)}")
    print(f"    Avg dur (h)       : {statistics.mean(r['dur_h'] for r in g):.2f}")
    print(f"    Avg gain vs base  : {statistics.mean(r['gain_vs_base'] for r in g):+.2f} pp")
    print(f"    Max gain vs base  : {max(r['gain_vs_base'] for r in g):+.2f} pp")
    gvp = [r['gain_vs_parent'] for r in g if r['gain_vs_parent'] is not None]
    if gvp:
        print(f"    Avg GEPA-only gain: {statistics.mean(gvp):+.2f} pp  (vs its own input)")
