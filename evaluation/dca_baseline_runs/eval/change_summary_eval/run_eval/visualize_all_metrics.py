"""
visualize_all_metrics.py
------------------------
Produces visualizations from summary_eval_sample.json for all models and versions.
Generates:
 - heatmap_quality.png (precision, recall, f1, hallucination_rate)
 - bars_<metric>.png for each metric (precision, recall, f1, hallucination_rate, llm_score)

Usage:
    python visualize_all_metrics.py [path/to/summary_eval_sample.json]
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_DEFAULT_INPUT = Path(__file__).parents[1] / "eval_results" / "summary_eval_results.json"
INPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_INPUT
with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

model_averages = data.get("model_averages", {})
all_names = sorted(model_averages.keys())
if not all_names:
    raise SystemExit("No model_averages found in JSON")

# parse names like 'v2/gpt41' -> (version, base)
def parse(name):
    p = name.split('/', 1)
    return (p[0], p[1]) if len(p) == 2 else ("v?", name)

versions = sorted({parse(n)[0] for n in all_names})
bases    = sorted({parse(n)[1] for n in all_names})

MET_LABEL = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "hallucination_rate": "Hallucination Rate",
    "llm_score": "LLM Score (1-5)",
}
QUALITY = ["precision", "recall", "f1", "hallucination_rate"]
TARGET = QUALITY + ["llm_score"]

out_dir = Path(__file__).parents[1] / "eval_results"  # change_summary_eval/eval_results/
out_dir.mkdir(parents=True, exist_ok=True)

# helper
def val(name, metric):
    return model_averages.get(name, {}).get(metric)

# Heatmap for quality metrics (0-1)
rows = [f"{v}/{b}" for v in versions for b in bases]
mat = np.full((len(rows), len(QUALITY)), np.nan, dtype=float)
for i, r in enumerate(rows):
    for j, m in enumerate(QUALITY):
        v = val(r, m)
        if v is not None:
            mat[i, j] = v

# Build a display matrix for the colormap. For `hallucination_rate`, higher
# values are worse, so invert that column (1 - value) so the RdYlGn cmap maps
# high hallucination -> red. Keep `mat` unchanged for numeric overlays.
display_mat = mat.copy()
if "hallucination_rate" in QUALITY:
    hi = QUALITY.index("hallucination_rate")
    col = display_mat[:, hi]
    mask = ~np.isnan(col)
    display_mat[mask, hi] = 1.0 - col[mask]

fig, ax = plt.subplots(figsize=(6 + len(QUALITY)*1.2, max(6, len(rows)*0.18)))
im = ax.imshow(display_mat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='Score (0-1)')
ax.set_xticks(np.arange(len(QUALITY)))
ax.set_xticklabels([MET_LABEL[m] for m in QUALITY], rotation=45)
ax.set_yticks(np.arange(len(rows)))
ax.set_yticklabels(rows)
ax.set_title('Quality Metrics Heatmap (version/base rows)')
for i in range(len(rows)):
    for j in range(len(QUALITY)):
        v = mat[i, j]
        if not np.isnan(v):
            # Use display_mat to decide overlay text color (contrast), but
            # print the original value `v`.
            disp = display_mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                    fontsize=7, color='black' if 0.25 < disp < 0.75 else 'white')
plt.tight_layout()
fig.savefig(out_dir / 'heatmap_quality.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved heatmap_quality.png')

# Bar charts per metric: x = base models, grouped bars = versions
x = np.arange(len(bases))
width = min(0.14, 0.8 / max(1, len(versions)))
for metric in TARGET:
    fig, ax = plt.subplots(figsize=(max(8, len(bases)*1.2), 5))
    for i, v in enumerate(versions):
        vals = [val(f"{v}/{b}", metric) for b in bases]
        vals_plot = [0 if vv is None else ((vv - 1)/4 if metric == 'llm_score' else vv) for vv in vals]
        # for llm_score we normalise for visual height but label with raw value
        offset = (i - len(versions)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals_plot, width, label=v)
        for bar, raw in zip(bars, vals):
            if raw is not None:
                label = f"{raw:.2f}"
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, label, ha='center', va='bottom', fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(bases, rotation=45)
    if metric == 'llm_score':
        ax.set_ylabel('LLM Score (normalised on plot; labels show raw 1-5)')
        ax.set_ylim(0, 1.2)
    else:
        ax.set_ylabel('Score (0-1)')
        ax.set_ylim(0, 1.15)
    ax.set_title(MET_LABEL.get(metric, metric))
    ax.legend(title='Version')
    plt.tight_layout()
    fname = f"bars_{metric}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {fname}')

print('All visualizations saved to', out_dir)
