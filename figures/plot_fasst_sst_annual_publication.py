#!/usr/bin/env python
"""FASST vs CESM2 piControl: Global-mean SST annual, publication style.
Matches the exact rcParams/color/layout convention of plot_sst_annual_publication.py.

Adds: vertical marker at the first year the FASST and piControl curves cross,
and a leaner, more minimal chrome (horizontal-only grid, frameless legend).

Usage: python make_fasst_sst_annual.py --year-min 1980 --out FILE.png
       python make_fasst_sst_annual.py --year-min 2000 --out FILE.png
"""
import argparse, glob, re, os
import numpy as np, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

FASST_RUN = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw_iceloop/run"
PICTL_CACHE = "output/picontrol_gmsst_annual.npz"

ap = argparse.ArgumentParser()
ap.add_argument("--year-min", type=int, default=1980)
ap.add_argument("--year-max", type=int, default=9999)
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

# ---------- FASST annual global-mean SST ----------
mfiles = sorted(glob.glob(f"{FASST_RUN}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
mfiles = [f for f in mfiles if args.year_min <= ym(f)[0] <= args.year_max]
if not mfiles:
    raise SystemExit(f"no monthly pop.h files in [{args.year_min},{args.year_max}]")
mkeys = [ym(f) for f in mfiles]
g = xr.open_dataset(mfiles[0])
TAREA = np.nan_to_num(g["TAREA"].values.astype(np.float64))
msst_by_ym = {k: gmean(xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values, TAREA)
              for f, k in zip(mfiles, mkeys)}
years_avail = sorted({y for (y, m) in mkeys})
full_years = [y for y in years_avail if all((y, m) in msst_by_ym for m in range(1, 13))]
if not full_years:
    raise SystemExit("no complete calendar year in range")
m_annual = np.array([np.mean([msst_by_ym[(y, m)] for m in range(1, 13)]) for y in full_years])
y0, y1 = full_years[0], full_years[-1]
print(f"FASST: {len(full_years)} complete years [{y0}-{y1}]  mean={m_annual.mean():.3f} std={m_annual.std():.3f}")

# ---------- piControl (elapsed-year overlay, cached annual series) ----------
z = np.load(PICTL_CACHE)
pic_years, pic_annual = z["years"], z["annual"]
n_pic = min(len(pic_years), len(full_years))
pic_window = pic_annual[:n_pic]
pic_mean, pic_sd = float(pic_window.mean()), float(pic_window.std())
print(f"piControl (matched {n_pic} yr)  mean={pic_mean:.3f} std={pic_sd:.3f}")

# ---------- first intersection of the two curves (elapsed-year overlay) ----------
# Both series share the same x-axis (full_years[:n_pic]) since piControl is plotted
# as an elapsed-year overlay against the FASST calendar years, so we can compare
# them index-for-index and linearly interpolate between the bracketing years.
x_common = np.array(full_years[:n_pic], dtype=float)
diff = m_annual[:n_pic] - pic_window
intersect_x = None
for i in range(1, n_pic):
    d0, d1 = diff[i - 1], diff[i]
    if d0 == 0:
        intersect_x = x_common[i - 1]
        break
    if np.sign(d0) != np.sign(d1):
        frac = d0 / (d0 - d1)
        intersect_x = x_common[i - 1] + frac * (x_common[i] - x_common[i - 1])
        break

if intersect_x is not None:
    print(f"First FASST/piControl intersection at year ≈ {intersect_x:.2f}")
else:
    print("No intersection found in the overlapping range "
          "(one curve stays above the other throughout).")

# ---------- figure ----------
fig, ax = plt.subplots(figsize=(9.0, 4.6))
fig.subplots_adjust(top=0.86, right=0.97, left=0.10, bottom=0.11)

# minimal chrome: horizontal gridlines only
ax.xaxis.grid(False)
ax.yaxis.grid(True)

pic_x = np.array(full_years[:n_pic])
ax.plot(pic_x, pic_window, color=C_TRUTH, lw=1.8, ls=(0, (5, 2)), dash_capstyle="round",
        marker="o", ms=4, mfc=C_TRUTH, mec="none", zorder=1,
        label=f"CESM2 piControl   mean {pic_mean:.2f}, std {pic_sd:.2f} °C")
ax.plot(full_years, m_annual, color=C_SIM, lw=2.4, solid_capstyle="round",
        marker="o", ms=5, mfc=C_SIM, mec="none", zorder=3,
        label=f"FASST   mean {m_annual.mean():.2f}, std {m_annual.std():.2f} °C")

if intersect_x is not None:
    ax.axvline(intersect_x, color=MUTE, lw=1.0, ls=(0, (1, 1.6)), zorder=0, alpha=0.75)
    ax.text(intersect_x, 1.015, f"{intersect_x:.1f}", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=9.5, color=MUTE)

ax.set_ylabel("Global-mean SST (°C)")
fig.text(0.10, 0.97, "Global-mean SST by year", ha="left", va="top", fontsize=15, fontweight="bold")
fig.text(0.10, 0.905, f"{y0}–{y1} annual mean, area-weighted", ha="left", va="top", color=MUTE, fontsize=10.5)
ax.legend(loc="lower left", frameon=False, handlelength=1.6, borderaxespad=0.6,
          labelcolor="linecolor")
ax.margins(x=0.03, y=0.15)
ax.xaxis.set_major_locator(mtick.MultipleLocator(5 if (y1 - y0) < 60 else 10))
fig.savefig(args.out); fig.savefig(args.out.replace(".png", ".pdf"))
print(f"wrote {args.out} (+ .pdf)")