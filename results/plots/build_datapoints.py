"""Parse the hand-maintained results/dca_experiments.xlsx (val sheet) + the
tournament excel into a clean, auditable per-data-point CSV for plotting:
results/plots/dca_datapoints.csv.

Each output row is ONE run's DCA total score, tagged with optimizer, prompt
version + type (MiDAS-pipeline vs DSPy for humans), model, seed source, and CoT,
plus the source spreadsheet row so every point is traceable.

Run:  python results/plots/build_datapoints.py [path/to/dca_experiments.xlsx]
"""
import csv
import os
import re
import sys
import pandas as pd

XLSX = sys.argv[1] if len(sys.argv) > 1 else os.path.join("results", "dca_experiments.xlsx")
OUT = os.path.join("results", "plots", "dca_datapoints.csv")
TOUR = os.path.join("results", "excel", "tournament_top2.xlsx")

_FLOAT = re.compile(r"(?<![\d.])(0?\.\d{2,4}|1\.0{1,3})(?![\d])")


def floats_in(cell) -> list[float]:
    if cell is None:
        return []
    return [round(float(m), 4) for m in _FLOAT.findall(str(cell))
            if 0.30 <= float(m) <= 1.0]


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
    for k in ("miprov2", "bootstrap", "gepa"):
        if k in c:
            return k, None
    return None, None


def seed_from(col4: str, col15: str) -> str:
    c15 = (col15 or "").strip().lower()
    if c15 == "v1":            # GEPA rows carry an explicit seed in col15 — wins over
        return "v1"            # any carried-forward col4 from a previous section
    if c15 == "v2":
        return "v2"
    s = (col4 or "").lower()
    if "v1-prompt" in s or "v1 prompt" in s:
        return "v1"
    if "v2-prompt" in s or "v2 prompt" in s:
        return "v2"
    if "signature class doc" in s or "dca signature as seed" in s:
        return "cr-default"
    return ""


def prompt_type(cat: str) -> str:
    c = (cat or "").lower()
    if "midas like image passing" in c:
        return "midas"          # MiDAS original pipeline (non-DSPy)
    if "dspy style" in c:
        return "dspy"           # human prompt run through DSPy signature
    return "single"             # unlabeled single run (fallback DSPy point)


def cot_flag(*cells) -> str:
    s = " ".join(str(c) for c in cells if c).lower()
    if "no-cot" in s or "nocot" in s or "no cot" in s:
        return "nocot"
    return "cot" if "cot" in s else ""


def main():
    df = pd.read_excel(XLSX, sheet_name="val", header=None, dtype=str)
    cat = hp = opt_model = seed4 = ""
    rows = []
    for r in range(2, df.shape[0]):
        def g(c):
            v = df.iat[r, c] if c < df.shape[1] else None
            return "" if v is None or str(v) == "nan" else str(v).strip()
        c0, c1, c2, c3, c4, c5, c6, c7, c15 = (g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7), g(15))
        if c0:
            cat = c0
        if c1:
            hp = c1
        if c2:
            opt_model = c2
        if c4:
            seed4 = c4
        opt, ver = classify_optimizer(cat)
        if opt is None:
            continue
        scores = floats_in(c7)
        if not scores:
            continue
        if opt == "human":
            model = norm_model(cat.split("/", 1)[1] if "/" in cat else cat)
            rows.append(dict(src=r + 1, optimizer="human", prompt_version=ver,
                             prompt_type=prompt_type(cat), model=model,
                             seed_source="", cot="", scores=scores))
        else:
            # Guard: trust col-7 as the total only when cls+chg are also present.
            if not (c5 and c6):
                continue
            rows.append(dict(src=r + 1, optimizer=opt, prompt_version="",
                             prompt_type="", model=norm_model(c3 or opt_model),
                             seed_source=(seed_from(seed4, c15) or "cr-default"),
                             cot=cot_flag(cat, hp), scores=scores))

    # Tournament best-score per instance (from the tournament excel)
    tour = 0
    if os.path.exists(TOUR):
        td = pd.read_excel(TOUR, sheet_name=0, dtype=str)
        cm = {str(c).strip().lower(): c for c in td.columns}
        for _, tr in td.iterrows():
            if cm.get("selection") and str(tr[cm["selection"]]).strip().lower() != "best score":
                continue
            sc = floats_in(tr.get(cm.get("total score")))
            if not sc:
                continue
            rows.append(dict(src=f"tournament_top2", optimizer="tournament", prompt_version="",
                             prompt_type="", model=norm_model(tr.get(cm.get("inference model"))),
                             seed_source="cr-default", cot="nocot", scores=sc[:1]))
            tour += 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_row", "optimizer", "prompt_version", "prompt_type",
                    "model", "seed_source", "cot", "score"])
        for rr in rows:
            for s in rr["scores"]:
                w.writerow([rr["src"], rr["optimizer"], rr["prompt_version"], rr["prompt_type"],
                            rr["model"], rr["seed_source"], rr["cot"], s])
    n = sum(len(rr["scores"]) for rr in rows)
    print(f"Wrote {OUT}: {len(rows)} run-groups -> {n} points ({tour} tournament)")


if __name__ == "__main__":
    main()
