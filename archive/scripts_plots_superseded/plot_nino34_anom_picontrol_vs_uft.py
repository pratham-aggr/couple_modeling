#!/usr/bin/env python
"""Nino-3.4 ANOMALY: CESM2 piControl (1200-yr unforced control) vs the
unfinetuned coupled ML-POP2 run.

Both are deseasonalized with their OWN full-record monthly climatology, on the
same gx1v7 box (5S-5N, 190-240E, TAREA-weighted) -- piControl `tos` is on the
native gn grid, which IS gx1v7, so no regridding is involved.

The sim drifts ~7 K, and a drift inflates the anomaly sigma without adding any
ENSO variability, so the legend reports sigma BOTH raw and after removing a
linear trend; the detrended number is the like-for-like ENSO amplitude.

Monthly box means cached to --cache; delete it to rebuild.
Usage: python plot_nino34_anom_picontrol_vs_uft.py [--out PNG]
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
ap.add_argument("--out", default="output/nino34_anomaly_picontrol_vs_uft.png")
ap.add_argument("--cache", default="output/nino34_monthly_picontrol_vs_uft_cache.npz")
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

if os.path.exists(args.cache):
    z = np.load(args.cache)
    ut, uv, umo, pt, pv, pmo = (z[k] for k in ("ut", "uv", "umo", "pt", "pv", "pmo"))
    print(f"loaded cache {args.cache}")
else:
    ut, uv, umo = [], [], []
    for f in uft_files:
        y, m = ym(f)
        uv.append(wmean(xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values[J0:J1, I0:I1], BM, BW))
        ut.append(y + (m - 0.5) / 12.0); umo.append(m)
    pt, pv, pmo = [], [], []
    for f in sorted(glob.glob(f"{PICTL_DIR}/*.nc")):
        ds = xr.open_dataset(f, decode_times=False)
        tos = ds["tos"].isel(nlat=slice(J0, J1), nlon=slice(I0, I1))
        t0 = int(round(float(ds["time"].values[0]) / 365.0))
        n = tos.sizes["time"]
        blk = tos.values.astype(np.float64)
        for i in range(n):
            yr, mo = t0 + i // 12 + 1, i % 12 + 1
            pv.append(wmean(blk[i], BM, BW)); pt.append(yr + (mo - 0.5) / 12.0); pmo.append(mo)
        ds.close()
    ut, uv, umo = map(np.array, (ut, uv, umo))
    pt, pv, pmo = map(np.array, (pt, pv, pmo))
    np.savez(args.cache, ut=ut, uv=uv, umo=umo, pt=pt, pv=pv, pmo=pmo)
    print(f"wrote cache {args.cache}")

def deseason(v, mo):
    a = v.astype(np.float64).copy()
    for k in range(1, 13):
        s = mo == k
        if s.any(): a[s] = v[s] - v[s].mean()
    return a

ua, pa = deseason(uv, umo), deseason(pv, pmo)
detrend = lambda t, a: a - np.polyval(np.polyfit(t, a, 1), t)
uad, pad = detrend(ut, ua), detrend(pt, pa)

print(f"piControl  : {len(pv)} months ({len(pv)//12} yr)  sigma raw {pa.std():.3f} K  "
      f"detrended {pad.std():.3f} K")
print(f"unfinetuned: {len(uv)} months ({len(uv)//12} yr)  sigma raw {ua.std():.3f} K  "
      f"detrended {uad.std():.3f} K")
# piControl sigma over rolling windows the length of the sim -> is the sim's
# amplitude inside what an unforced 67-yr chunk of the control can produce?
W = len(uv)
roll = np.array([pa[i:i+W].std() for i in range(0, len(pa) - W, 12)])
print(f"  piControl rolling {W//12}-yr sigma: mean {roll.mean():.3f} sd {roll.std():.3f} "
      f"range {roll.min():.3f}..{roll.max():.3f}")

# ---------------- figure ----------------
y0 = ut[0]
npic = min(len(pt), len(ut))
fig, ax = plt.subplots(figsize=(10.2, 5.6))
fig.subplots_adjust(top=0.715, right=0.975, left=0.08, bottom=0.105)
ax.axhline(0, color="#c7c7c7", lw=0.9, zorder=0)
ax.plot(y0 + (pt[:npic] - pt[0]), pa[:npic], color=C_PIC, lw=1.2, alpha=0.85, zorder=2,
        label=f"CESM2 piControl (elapsed model yr)   σ = {pa.std():.2f} K "
              f"(detrended {pad.std():.2f})")
ax.plot(ut, ua, color=C_UFT, lw=1.7, solid_capstyle="round", zorder=3,
        label=f"Unfinetuned coupled sim   σ = {ua.std():.2f} K "
              f"(detrended {uad.std():.2f})")
ax.set_ylabel("Niño-3.4 anomaly (K)")
ax.set_xlabel("Year  (piControl: elapsed model year)")
ax.margins(x=0.008); ax.grid(axis="x", visible=False)
ax.xaxis.set_major_locator(mtick.MultipleLocator(10))
fig.text(0.08, 0.985, "Niño-3.4 anomaly: unfinetuned sim vs CESM2 piControl",
         ha="left", va="top", fontsize=14.5, fontweight="bold")
fig.text(0.08, 0.935,
         "5°S–5°N, 190–240°E, area-weighted · deseasonalized by each series' own full-record\n"
         "monthly climatology · the sim's ~7 K drift inflates its raw σ, so detrended σ is the "
         "like-for-like ENSO amplitude",
         ha="left", va="top", color=MUTE, fontsize=9.5)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), frameon=False,
          handlelength=2.2, borderaxespad=0.0, labelcolor="linecolor", handletextpad=0.7)
fig.savefig(args.out); fig.savefig(args.out.replace(".png", ".pdf"))
print(f"wrote {args.out} (+ .pdf)")
