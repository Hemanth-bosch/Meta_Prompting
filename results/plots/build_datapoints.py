"""Parse the hand-maintained results/dca_experiments.xlsx (val sheet) into a
clean, auditable per-data-point CSV: results/plots/dca_datapoints.csv.

Each output row is ONE individual run's DCA total score, tagged with the
optimizer, model, seed source, and CoT flag, plus the source spreadsheet row
so every point is traceable. Repeatability rows (comma-separated score lists)
expand to multiple points.

Run:  python results/plots/build_datapoints.py
"""
import csv
import os
import re
import pandas as pd

XLSX = os.path.join("results", "dca_experiments.xlsx")
OUT = os.path.join("results", "plots", "dca_datapoints.csv")

_FLOAT = re.compile(r"(?<![\d.])(0?\.\d{2,4}|1\.0{1,3})(?![\d])")


def floats_in(cell) -> list[float]:
    """Extract plausible DCA scores (0..1) from a possibly comma-separated cell."""
    if cell is None:
        return []
    out = []
    for m in _FLOAT.findall(str(cell)):
        try:
            v = float(m)
        except ValueError:
            continue
        if 0.30 <= v <= 1.0:      # DCA totals live in this band; filters typos/junk
            out.append(round(v, 4))
    return out


def norm_model(*cells) -> str:
    s = " ".join(str(c) for c in cells if c and str(c) != "nan").lower().replace(" ", "")
    if "gpt-5.2" in s or "gpt5.2" in s:
        return "gpt-5.2"
    if "gpt-5-mini" in s or "gpt5-mini" in s or "gpt-5mini" in s:
        return "gpt-5-mini"
    if "4o-mini" in s:
        return "gpt-4o-mini"
    if "4.1-mini" in s or "4.1mini" in s:
        return "gpt-4.1-mini"
    if "4.1" in s:
        return "gpt-4.1"
    return "other"


def classify_optimizer(cat: str):
    c = (cat or "").lower()
    m = re.match(r"\s*v([1-4])\b", c)
    if m:
        return "human", f"v{m.group(1)}"
    if "miprov2" in c:
        return "miprov2", None
    if "bootstrap" in c:
        return "bootstrap", None
    if "gepa" in c:
        return "gepa", None
    return None, None


def seed_source(cell: str) -> str:
    s = (cell or "").lower()
    if "v1-prompt" in s or "v1 prompt" in s:
        return "v1"
    if "v2-prompt" in s or "v2 prompt" in s:
        return "v2"
    if "signature class doc" in s or "dca signature as seed" in s:
        return "cr-default"
    return ""


def cot_flag(*cells) -> str:
    s = " ".join(str(c) for c in cells if c).lower()
    if "no-cot" in s or "nocot" in s or "no cot" in s:
        return "nocot"
    if "cot" in s:
        return "cot"
    return ""


def main():
    df = pd.read_excel(XLSX, sheet_name="val", header=None, dtype=str)
    cat = hp = opt_model = seed = ""
    rows = []
    for r in range(2, df.shape[0]):
        def g(c):
            v = df.iat[r, c] if c < df.shape[1] else None
            return "" if v is None or str(v) == "nan" else str(v).strip()
        c0, c1, c2, c3, c4, c5, c6, c7 = g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7)
        if c0:
            cat = c0
        if c1:
            hp = c1
        if c2:
            opt_model = c2
        if c4:
            seed = c4
        opt, ver = classify_optimizer(cat)
        if opt is None:
            continue
        # Correctness guard: only trust col-7 as the TOTAL when this same row also
        # carries cls (col5) and chg (col6) — filters misaligned rows where col-7
        # holds a classification list instead of the total score.
        if not (c5 and c6):
            continue
        scores = floats_in(c7)
        if not scores:
            continue
        if opt == "human":
            # human rows embed the model in the category label: "v1 / GPT-4.1-mini"
            model = norm_model(cat.split("/", 1)[1] if "/" in cat else cat)
        else:
            model = norm_model(c3 or opt_model)     # infer model preferred, else opt model
        rows.append(dict(
            source_row=r + 1, optimizer=opt, prompt_version=(ver or ""),
            model=model, seed_source=(seed_source(seed) or "cr-default"),
            cot=cot_flag(cat, hp), n_points=len(scores), scores=scores,
        ))
    # ── Tournament best-score points (one per instance) from the tournament excel ──
    tour_pts = 0
    tour_xlsx = os.path.join("results", "excel", "tournament_top2.xlsx")
    if os.path.exists(tour_xlsx):
        td = pd.read_excel(tour_xlsx, sheet_name=0, dtype=str)
        cols = {str(c).strip().lower(): c for c in td.columns}
        sel = cols.get("selection"); tot = cols.get("total score"); inf = cols.get("inference model")
        for _, tr in td.iterrows():
            if sel and str(tr.get(sel, "")).strip().lower() != "best score":
                continue
            sc = floats_in(tr.get(tot))
            if not sc:
                continue
            rows.append(dict(source_row=f"tournament_top2:{tr.get(cols.get('tournament'))}",
                             optimizer="tournament", prompt_version="",
                             model=norm_model(tr.get(inf)), seed_source="cr-default",
                             cot="nocot", n_points=1, scores=sc[:1]))
            tour_pts += 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_row", "optimizer", "prompt_version", "model", "seed_source", "cot", "score"])
        for row in rows:
            for s in row["scores"]:
                w.writerow([row["source_row"], row["optimizer"], row["prompt_version"],
                            row["model"], row["seed_source"], row["cot"], s])
    print(f"Tournament best-score points added: {tour_pts}")
    total = sum(len(r["scores"]) for r in rows)
    print(f"Wrote {OUT}: {len(rows)} run-groups -> {total} data points")
    from collections import Counter
    by = Counter((r["optimizer"], r["model"], r["seed_source"], r["cot"]) for r in rows)
    for k, n in sorted(by.items()):
        print(f"  {k}: {n} groups")


if __name__ == "__main__":
    main()
