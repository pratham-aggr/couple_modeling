#!/usr/bin/env python
"""Grouped bar chart of temporal-split test R^2 across the five surface fluxes.

Reads r2_scores_test.json from each model's out_dir and produces a publication
figure (PDF + PNG). Models that haven't finished (no json yet) are skipped, so
the memory-free (lag0) bars appear automatically once that run completes.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FLUXES = ["TAUX", "TAUY", "SHFLX", "LHFLX", "QFLX"]

# (label, out_dir, color) in the cumulative order of the results table
MODELS = [
    ("Memory-free (lag 0)", "output_unet_lag0h_temporal",            "#9e9e9e"),
    ("MEMO (mem24h)",        "output_unet_mem24h_temporal",           "#4c72b0"),
    (r"+ $dT_s/dt$",         "output_unet_mem24h_dsst_temporal",      "#55a868"),
    ("+ dropout",            "output_unet_mem24h_dsst_temporal_drop", "#c44e52"),
    ("+ dropout + aug.",     "output_unet_mem24h_dsst_temporal_combo","#8172b3"),
]

present = []
for label, d, color in MODELS:
    f = ROOT / d / "r2_scores_test.json"
    if f.exists():
        r2 = json.loads(f.read_text())
        present.append((label, [r2[v] for v in FLUXES], color))
    else:
        print(f"skip (not finished): {d}")

n = len(present)
x = np.arange(len(FLUXES))
width = 0.8 / n

fig, ax = plt.subplots(figsize=(9, 4.5))
for i, (label, vals, color) in enumerate(present):
    off = (i - (n - 1) / 2) * width
    bars = ax.bar(x + off, vals, width, label=label, color=color,
                  edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(FLUXES)
ax.set_ylabel(r"Test $R^2$ (2013--2014, ocean/sea-ice)")
ax.set_ylim(0.0, 0.98)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.legend(ncol=3, frameon=False, fontsize=9, loc="upper center",
          bbox_to_anchor=(0.5, -0.12))
ax.set_title("Temporal-split skill by flux and model component")
fig.tight_layout()

for ext in ("pdf", "png"):
    out = ROOT / "papers" / f"fig_temporal_results.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
