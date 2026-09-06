#!/usr/bin/env python
"""Debug: global-mean SST by year, WITH vs WITHOUT the equatorial-Pacific
ENSO-tongue region masked out, to see whether the finetuned run's cooling
drift is being driven by cold-tongue/equatorial-Pacific behavior or is a
broader (e.g. extratropical/polar) signal.

Tongue mask: 5S-5N, 150E-280E (covers Nino3+3.4+4 -- the whole equatorial
Pacific cold-tongue band, not just the Nino3.4 box) is EXCLUDED from the
"no-tongue" series.

Usage:  python plot_sst_annual_no_tongue.py [--run DIR] [--out PNG]
"""
import argparse, glob, re
import numpy as np, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

RUN_DEFAULT = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_sensreg50yr/run"

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=RUN_DEFAULT)
ap.add_argument("--out", default="/glade/u/home/praggarwal/couple/output/sst_annual_no_tongue.png")
args = ap.parse_args()

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 13.5,
    "axes.labelsize": 12, "legend.fontsize": 11.5, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "axes.linewidth": 0, "axes.edgecolor": "#4d4d4d",
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#ececec",
    "grid.linewidth": 0.7, "xtick.color": "#6b7177", "ytick.color": "#6b7177",
    "text.color": "#1a1a1a", "axes.labelcolor": "#3d3d3d",
    "savefig.dpi": 300, "figure.dpi": 140,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "xtick.major.size": 0, "ytick.major.size": 0,
})
C_ALL, C_NOTONGUE, C_TONGUE, MUTE = "#D55E00", "#0072B2", "#CC79A7", "#6b7177"

mfiles = sorted(glob.glob(f"{args.run}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
if not mfiles: raise SystemExit(f"no monthly pop.h files in {args.run}")
def ym(f): m=re.search(r"\.(\d{4})-(\d{2})\.nc$", f); return int(m.group(1)), int(m.group(2))
mkeys=[ym(f) for f in mfiles]

g = xr.open_dataset(mfiles[0])
TAREA = np.nan_to_num(g["TAREA"].values.astype(np.float64))
TLAT, TLON = g["TLAT"].values, g["TLONG"].values
tongue = (TLAT >= -5) & (TLAT <= 5) & (TLON >= 150) & (TLON <= 280)
print(f"tongue mask covers {tongue.sum()} cells, "
      f"{100*TAREA[tongue].sum()/TAREA.sum():.2f}% of global ocean area")

def wmean(sst2d, weight_mask):
    s = np.asarray(sst2d, np.float64)
    m = np.isfinite(s) & weight_mask
    w = TAREA * m
    return float((s[m]*w[m]).sum() / w[m].sum())

all_mask = np.ones_like(TLAT, dtype=bool)
notongue_mask = ~tongue

by_ym_all, by_ym_notongue, by_ym_tongue = {}, {}, {}
for f, k in zip(mfiles, mkeys):
    sst = xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values
    by_ym_all[k] = wmean(sst, all_mask)
    by_ym_notongue[k] = wmean(sst, notongue_mask)
    by_ym_tongue[k] = wmean(sst, tongue)

years_avail = sorted({y for (y, m) in mkeys})
full_years = [y for y in years_avail if all((y, m) in by_ym_all for m in range(1, 13))]
if not full_years: raise SystemExit("no complete calendar year yet")

def annual(byym):
    return np.array([np.mean([byym[(y, m)] for m in range(1, 13)]) for y in full_years])

a_all = annual(by_ym_all)
a_notongue = annual(by_ym_notongue)
a_tongue = annual(by_ym_tongue)
y0, y1 = full_years[0], full_years[-1]
print(f"finetuned: {len(full_years)} complete years [{y0}-{y1}]")

print()
print(f"{'year':6s}{'global':>10s}{'no-tongue':>12s}{'tongue-only':>14s}")
for y, va, vn, vt in zip(full_years, a_all, a_notongue, a_tongue):
    print(f"{y:6d}{va:10.3f}{vn:12.3f}{vt:14.3f}")

print()
print(f"Global drift  {y0}->{y1}: {a_all[-1]-a_all[0]:+.3f} K")
print(f"No-tongue drift {y0}->{y1}: {a_notongue[-1]-a_notongue[0]:+.3f} K")
print(f"Tongue-only drift {y0}->{y1}: {a_tongue[-1]-a_tongue[0]:+.3f} K")

# ---------- figure ----------
fig, ax = plt.subplots(figsize=(9.0, 4.6))
fig.subplots_adjust(top=0.86, right=0.97, left=0.10, bottom=0.11)
ax.plot(full_years, a_all, color=C_ALL, lw=2.4, solid_capstyle="round",
        marker="o", ms=5, mfc=C_ALL, mec="none", zorder=3,
        label=f"Global (full grid)  (Δ {a_all[-1]-a_all[0]:+.2f} K)")
ax.plot(full_years, a_notongue, color=C_NOTONGUE, lw=2.2, ls=(0, (3, 1, 1, 1)), dash_capstyle="round",
        marker="o", ms=5, mfc=C_NOTONGUE, mec="none", zorder=2,
        label=f"Global, ENSO-tongue excluded  (Δ {a_notongue[-1]-a_notongue[0]:+.2f} K)")
ax.plot(full_years, a_tongue, color=C_TONGUE, lw=2.0, ls=(0, (4, 1)), dash_capstyle="round",
        marker="o", ms=5, mfc=C_TONGUE, mec="none", zorder=1,
        label=f"ENSO tongue only (5S-5N,150-280E)  (Δ {a_tongue[-1]-a_tongue[0]:+.2f} K)")
ax.set_ylabel("Area-weighted mean SST (°C)")
fig.text(0.10, 0.97, "Global-mean SST: with vs without the ENSO tongue", ha="left", va="top",
         fontsize=14.5, fontweight="bold")
fig.text(0.10, 0.905, f"{y0}–{y1} annual mean, finetuned coupled sim", ha="left", va="top",
         color=MUTE, fontsize=10.5)
ax.legend(loc="best", frameon=False, handlelength=1.8, borderaxespad=0.3, labelcolor="linecolor")
ax.margins(x=0.03)
ax.xaxis.set_major_locator(mtick.MultipleLocator(5))
fig.savefig(args.out); fig.savefig(args.out.replace(".png", ".pdf"))
print(f"\nwrote {args.out} (+ .pdf)")
