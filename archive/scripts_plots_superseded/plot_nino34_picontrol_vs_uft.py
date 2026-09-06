#!/usr/bin/env python
"""Nino-3.4 mean SST: CESM2 piControl (1200-yr unforced control) vs the
unfinetuned coupled ML-POP2 run.

piControl `tos` is on the native gn grid = gx1v7, the same 384x320 the sims run
on, so the identical box mask and TAREA weights apply with no regridding.

piControl runs on model years 1-1200 and cannot share a calendar axis with the
sim, so it is drawn two ways: a mean +/- 1 sigma band across the full axis (the
reference level), and its first N years overlaid by ELAPSED model year for a
like-for-like comparison of the drift.

Cached to --cache / the piControl npz; delete either to rebuild.
Usage: python plot_nino34_picontrol_vs_uft.py [--out PNG]
"""
import argparse, glob, re, os
import numpy as np, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

RUN_UFT = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw_iceloop/run"
PICTL_DIR = ("/glade/collections/cmip/CMIP6/CMIP/NCAR/CESM2/piControl/"
             "r1i1p1f1/Omon/tos/gn/latest")

ap = argparse.ArgumentParser()
ap.add_argument("--run-uft", default=RUN_UFT)
ap.add_argument("--out", default="output/nino34_picontrol_vs_uft.png")
ap.add_argument("--cache", default="output/nino34_picontrol_vs_uft_cache.npz")
args = ap.parse_args()

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11.5, "axes.labelsize": 12,
    "legend.fontsize": 10.5, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.linewidth": 0, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#ececec", "grid.linewidth": 0.7,
    "xtick.color": "#6b7177", "ytick.color": "#6b7177",
    "text.color": "#1a1a1a", "axes.labelcolor": "#3d3d3d",
    "savefig.dpi": 300, "figure.dpi": 140,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "xtick.major.size": 0, "ytick.major.size": 0,
})
C_PIC, C_UFT, MUTE = "#9467bd", "#009E73", "#6b7177"

ym = lambda f: (lambda m: (int(m.group(1)), int(m.group(2))))(
    re.search(r"\.(\d{4})-(\d{2})\.nc$", f))

def box_weights(sample):
    """Nino-3.4 box (5S-5N, 190-240E) + TAREA weights on gx1v7."""
    g = xr.open_dataset(sample)
    TLAT, TLONG = g.TLAT.values, g.TLONG.values
    TAREA = np.nan_to_num(g.TAREA.values.astype(np.float64))
    box = (TLAT >= -5) & (TLAT <= 5) & (TLONG >= 190) & (TLONG <= 240)
    jj, ii = np.where(box)
    J0, J1, I0, I1 = jj.min(), jj.max() + 1, ii.min(), ii.max() + 1
    return J0, J1, I0, I1, box[J0:J1, I0:I1], (TAREA * box)[J0:J1, I0:I1]

def wmean(sub, m, w):
    s = np.asarray(sub, np.float64)
    ok = np.isfinite(s) & m
    return float(np.sum(np.where(ok, s, 0) * w) / w[m].sum())

uft_files = sorted(glob.glob(f"{args.run_uft}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
J0, J1, I0, I1, BM, BW = box_weights(uft_files[0])

def annual(d):
    ys = sorted({y for y, _ in d if all((y, mo) in d for mo in range(1, 13))})
    return np.array(ys), np.array([np.mean([d[(y, mo)] for mo in range(1, 13)]) for y in ys])

if os.path.exists(args.cache):
    z = np.load(args.cache)
    uy, ua, py, pa = z["uy"], z["ua"], z["py"], z["pa"]
    print(f"loaded cache {args.cache}")
else:
    # --- unfinetuned sim ---
    du = {ym(f): wmean(xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values[J0:J1, I0:I1], BM, BW)
          for f in uft_files}
    uy, ua = annual(du)
    # --- piControl: contiguous monthly, 12 per year, NOLEAP ---
    pyy, paa = [], []
    for f in sorted(glob.glob(f"{PICTL_DIR}/*.nc")):
        ds = xr.open_dataset(f, decode_times=False)
        tos = ds["tos"].isel(nlat=slice(J0, J1), nlon=slice(I0, I1))
        t0 = int(round(float(ds["time"].values[0]) / 365.0))
        n = tos.sizes["time"]
        for i0 in range(0, n - n % 12, 12):
            blk = tos.isel(time=slice(i0, i0 + 12)).values.astype(np.float64)
            mv = [wmean(blk[k], BM, BW) for k in range(12)]
            pyy.append(t0 + i0 // 12 + 1); paa.append(float(np.mean(mv)))
        ds.close()
    py, pa = np.array(pyy), np.array(paa)
    np.savez(args.cache, uy=uy, ua=ua, py=py, pa=pa)
    print(f"wrote cache {args.cache}")

p_mean, p_sd = float(pa.mean()), float(pa.std())
trend = lambda y, a: np.polyfit(y, a, 1)[0] * 100
W = len(uy)
roll = np.array([trend(py[i:i+W], pa[i:i+W]) for i in range(len(py) - W)])
print(f"piControl : yr {py[0]}-{py[-1]} ({len(py)} yr)  mean {p_mean:.3f} +/- {p_sd:.3f} degC  "
      f"trend {trend(py, pa):+.3f} K/century")
print(f"  rolling {W}-yr trends: mean {roll.mean():+.3f} sd {roll.std():.3f} "
      f"range {roll.min():+.3f}..{roll.max():+.3f}")
print(f"unfinetuned: {uy[0]}-{uy[-1]} ({len(uy)} yr)  mean {ua.mean():.3f} degC  "
      f"trend {trend(uy, ua):+.3f} K/century  total {ua[-1]-ua[0]:+.3f} K")
m20 = uy >= uy[0] + 20
print(f"  from {uy[0]+20}: trend {trend(uy[m20], ua[m20]):+.3f} K/century ({m20.sum()} yr)")

# ---------------- figure ----------------
fig, ax = plt.subplots(figsize=(9.8, 5.8))
fig.subplots_adjust(top=0.705, right=0.975, left=0.085, bottom=0.105)

ax.axhspan(p_mean - p_sd, p_mean + p_sd, color=C_PIC, alpha=0.13, zorder=0)
ax.axhline(p_mean, color=C_PIC, lw=1.4, ls=":", zorder=1,
           label=f"piControl mean $\\pm$1$\\sigma$   {p_mean:.2f} $\\pm$ {p_sd:.2f} °C  ({len(py)} yr)")
npic = min(len(py), len(uy))
ax.plot(uy[0] + np.arange(npic), pa[:npic], color=C_PIC, lw=1.5, alpha=0.85, zorder=2,
        label=f"CESM2 piControl (elapsed model yr)   {trend(py[:npic],pa[:npic]):+.2f} K/century")
ax.plot(uy, ua, color=C_UFT, lw=2.2, solid_capstyle="round", zorder=3,
        label=f"Unfinetuned coupled sim   {trend(uy,ua):+.2f} K/century")

ax.set_ylabel("Niño-3.4 SST (°C)")
ax.set_xlabel("Year  (piControl: elapsed model year)")
ax.margins(x=0.01); ax.grid(axis="x", visible=False)
ax.xaxis.set_major_locator(mtick.MultipleLocator(10))
fig.text(0.085, 0.985, "Niño-3.4 mean SST: unfinetuned sim vs CESM2 piControl",
         ha="left", va="top", fontsize=14.5, fontweight="bold")
fig.text(0.085, 0.935,
         "5°S–5°N, 190–240°E, area-weighted · annual means · piControl is unforced "
         "and on model years 1–1200,\nso it is shown as a level (band) and by elapsed year, not on the sim's calendar",
         ha="left", va="top", color=MUTE, fontsize=9.5)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), frameon=False,
          handlelength=2.2, borderaxespad=0.0, labelcolor="linecolor",
          ncol=1, handletextpad=0.7)
fig.savefig(args.out); fig.savefig(args.out.replace(".png", ".pdf"))
print(f"wrote {args.out} (+ .pdf)")
