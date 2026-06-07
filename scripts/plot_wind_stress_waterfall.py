#!/usr/bin/env python
"""Waterfall (step-up) chart showing how each model component cumulatively
improves wind-stress R² (TAUX and TAUY) on the held-out 2013–2014 test set.

Each bar represents the total R² at that stage; the coloured segment shows
the gain from adding the new component; the grey segment shows the
already-achieved R².

Output: papers/fig_wind_stress_waterfall.{pdf,png}
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent

# Ordered stages: (label for x-axis, out_dir or literal r2 dict)
STAGES = [
    ("Memory-\nfree",       "output_unet_lag0h_temporal"),
    ("+ Memory\n(mem24h)",  "output_unet_mem24h_temporal"),
    (r"+ $dT_s/dt$",        "output_unet_mem24h_dsst_temporal"),
    ("+ Dropout",           "output_unet_mem24h_dsst_temporal_drop"),
    ("+ Aug.",              "output_unet_mem24h_dsst_temporal_combo"),
]

FLUX_COLORS = {"TAUX": "#4c72b0", "TAUY": "#c44e52"}
GAIN_ALPHA  = 0.85
BASE_ALPHA  = 0.25


def load_r2(out_dir):
    p = ROOT / out_dir / "r2_scores_test.json"
    return json.loads(p.read_text())


r2_by_stage = []
for label, d in STAGES:
    r2 = load_r2(d)
    r2_by_stage.append((label, r2))

labels   = [s[0] for s in r2_by_stage]
taux_seq = [s[1]["TAUX"] for s in r2_by_stage]
tauy_seq = [s[1]["TAUY"] for s in r2_by_stage]

n = len(labels)
x = np.arange(n)
bar_w = 0.32
gap   = 0.04

fig, ax = plt.subplots(figsize=(9, 4.5))

for fi, (var, seq, c_gain) in enumerate(
        [("TAUX", taux_seq, FLUX_COLORS["TAUX"]),
         ("TAUY", tauy_seq, FLUX_COLORS["TAUY"])]):

    offset = (fi - 0.5) * (bar_w + gap)

    prev = 0.0
    for i, (r2, lbl) in enumerate(zip(seq, labels)):
        gain = max(r2 - prev, 0.0)   # never negative by design
        base = min(prev, r2)

        # grey base (already achieved)
        ax.bar(x[i] + offset, base, bar_w,
               color=c_gain, alpha=BASE_ALPHA, edgecolor="none")
        # coloured gain on top
        ax.bar(x[i] + offset, gain, bar_w, bottom=base,
               color=c_gain, alpha=GAIN_ALPHA, edgecolor="white", linewidth=0.5)

        # annotate final value at top of bar
        ax.text(x[i] + offset, r2 + 0.008, f"{r2:.3f}",
                ha="center", va="bottom", fontsize=7.5,
                color=c_gain if gain > 0.01 else "grey")

        prev = r2

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel(r"Test $R^2$ (2013–2014, ocean/sea-ice)", fontsize=9)
ax.set_ylim(0.0, 0.82)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.set_title("Cumulative wind-stress improvement by model component", fontsize=10)

patches = [mpatches.Patch(color=FLUX_COLORS[v], alpha=GAIN_ALPHA, label=v)
           for v in ("TAUX", "TAUY")]
ax.legend(handles=patches, fontsize=9, frameon=False, loc="upper left")

fig.tight_layout()

for ext in ("pdf", "png"):
    out = ROOT / "papers" / f"fig_wind_stress_waterfall.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
