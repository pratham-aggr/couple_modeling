#!/usr/bin/env python
"""FASST vs CESM2 piControl: Global-mean SST annual, publication style.
Two-panel: (a) 1980-1999 [transient, larger FASST-piControl separation],
(b) 2000-latest [quasi-equilibrium, separation maxes out around ~0.5 K].
Independent y-axis scaling per panel (the two eras have very different
spread) -- a single shared axis would flatten panel (b) into an
indistinguishable line. Kelvin throughout. Mean+/-std reported only in the
later (quasi-equilibrium) panel, as a minimal colored annotation, not a
boxed legend -- the transient panel's mean/std would not be a meaningful
summary of a non-stationary period.

Usage: python make_fasst_sst_annual.py --out FILE.png [--split-year 2000]
"""
import argparse, glob, re
import numpy as np, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

FASST_RUN = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw_iceloop/run"
PICTL_CACHE = "output/picontrol_gmsst_annual.npz"
C2K = 273.15

ap = argparse.ArgumentParser()
ap.add_argument("--split-year", type=int, default=2000,
                help="last year of panel (a) + 1 == first year of panel (b)")
ap.add_argument("--out", required=True)
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
C_SIM, C_TRUTH, MUTE = "#D55E00", "#5b6167", "#6b7177"

def ym(f):
    m = re.search(r"\.(\d{4})-(\d{2})\.nc$", f)
    return int(m.group(1)), int(m.group(2))

def gmean(sst2d, w):
    s = np.asarray(sst2d, np.float64); m = np.isfinite(s)
    return float(np.nansum(np.where(m, s, 0) * w) / (w * m).sum())

# ---------- FASST annual global-mean SST (Kelvin) ----------
mfiles = sorted(glob.glob(f"{FASST_RUN}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
mkeys = [ym(f) for f in mfiles]
g = xr.open_dataset(mfiles[0])
TAREA = np.nan_to_num(g["TAREA"].values.astype(np.float64))
msst_by_ym = {k: gmean(xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values, TAREA)
              for f, k in zip(mfiles, mkeys)}
years_avail = sorted({y for (y, m) in mkeys})
full_years = [y for y in years_avail if all((y, m) in msst_by_ym for m in range(1, 13))]
m_annual = np.array([np.mean([msst_by_ym[(y, m)] for m in range(1, 13)]) for y in full_years]) + C2K
y0, y1 = full_years[0], full_years[-1]
print(f"FASST: {len(full_years)} complete years [{y0}-{y1}]")

# ---------- piControl (elapsed-year overlay, cached annual series, Kelvin) ----------
z = np.load(PICTL_CACHE)
pic_years, pic_annual_C = z["years"], z["annual"]
n_pic = min(len(pic_years), len(full_years))
pic_window = pic_annual_C[:n_pic] + C2K
pic_x = np.array(full_years[:n_pic])

# ---------- split into two panels ----------
split = args.split_year
maskA = pic_x < split
maskB = ~maskA
yA0, yA1 = pic_x[maskA][0], pic_x[maskA][-1]
yB0, yB1 = pic_x[maskB][0], pic_x[maskB][-1]

fasst_A = m_annual[:len(pic_x)][maskA]; fasst_B = m_annual[:len(pic_x)][maskB]
pic_A = pic_window[maskA]; pic_B = pic_window[maskB]

# ---------- figure ----------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.0))
fig.subplots_adjust(top=0.84, right=0.97, left=0.075, bottom=0.20, wspace=0.22)

for ax, x, pic_y, fasst_y, xlim in [
    (axL, pic_x[maskA], pic_A, fasst_A, (yA0, yA1)),
    (axR, pic_x[maskB], pic_B, fasst_B, (yB0, yB1)),
]:
    ax.plot(x, pic_y, color=C_TRUTH, lw=1.6, ls=(0, (5, 2)), dash_capstyle="round",
            marker="o", ms=4.5, mfc=C_TRUTH, mec="none", zorder=1, label="CESM2 piControl")
    ax.plot(x, fasst_y, color=C_SIM, lw=2.4, solid_capstyle="round",
            marker="o", ms=5.5, mfc=C_SIM, mec="none", zorder=3, label="FASST")
    ax.set_ylabel("Global-mean SST (K)")
    ax.margins(x=0.03, y=0.15)
    ax.xaxis.set_major_locator(mtick.MultipleLocator(5 if (xlim[1] - xlim[0]) < 30 else 10))

axL.set_title(f"{yA0}–{yA1}", fontsize=12, color=MUTE, loc="left")
axR.set_title(f"{yB0}–{yB1}", fontsize=12, color=MUTE, loc="left")

# minimal descriptive stats -- ONLY the later (quasi-equilibrium) panel
axR.text(0.03, 0.05, f"FASST  {fasst_B.mean():.2f} ± {fasst_B.std():.2f} K",
          transform=axR.transAxes, color=C_SIM, fontsize=10.5, ha="left", va="bottom")
axR.text(0.03, 0.12, f"CESM2 piControl  {pic_B.mean():.2f} ± {pic_B.std():.2f} K",
          transform=axR.transAxes, color=C_TRUTH, fontsize=10.5, ha="left", va="bottom")

fig.text(0.075, 0.97, f"Global-mean SST by year ({y0}–{y1})", ha="left", va="top",
         fontsize=15, fontweight="bold")

handles, labels = axL.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 0.0), handlelength=1.6, labelcolor="linecolor")

fig.savefig(args.out); fig.savefig(args.out.replace(".png", ".pdf"))
print(f"wrote {args.out} (+ .pdf)")
