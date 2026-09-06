#!/usr/bin/env python
"""Nino-3.4: finetuned (sensreg, ke9j7j75) vs unfinetuned (pz220f0b base)
coupled ML-POP2 sims, against CAM6/CREDIT (the atmosphere emulator's training
truth) and the CESM2-LE member LE2-1231.002.

Two panels:
  (a) monthly Nino-3.4 anomaly, deseasonalized with each series' OWN monthly
      climatology over a COMMON baseline period (default 1980-2014, the span
      all four sources cover).
  (b) annual-mean Nino-3.4 SST (absolute).

Box: 5S-5N, 190-240E, area-weighted (TAREA on gx1v7, cos(lat) on the CAM6 grid).

Monthly box means are cached to --cache so re-runs are fast; delete it to rebuild.

Usage: python plot_nino34_four.py [--out PNG] [--base-y0 1980 --base-y1 2014]
"""
import argparse, glob, re, os
from datetime import timedelta
import numpy as np, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

RUN_FT  = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_sensreg50yr/run"
RUN_UFT = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw_iceloop/run"
LE2_SST_DIR = "/glade/campaign/cgd/cesm/CESM2-LE/ocn/proc/tseries/month_1/SST"
CAM6_DIR    = "/glade/derecho/scratch/wchapman/b_credit_runs"

ap = argparse.ArgumentParser()
ap.add_argument("--run-ft", default=RUN_FT)
ap.add_argument("--run-uft", default=RUN_UFT)
ap.add_argument("--ref-member", default="LE2-1231.002")
ap.add_argument("--base-y0", type=int, default=1980)
ap.add_argument("--base-y1", type=int, default=2014)
ap.add_argument("--out", default="/glade/u/home/praggarwal/couple/output/nino34_four.png")
ap.add_argument("--cache", default="/glade/u/home/praggarwal/couple/output/nino34_four_cache.npz")
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
# Okabe-Ito CVD-safe; each series also gets its own dash pattern.
C = {"ft": "#D55E00", "uft": "#009E73", "cam6": "#0072B2", "le2": "#5b6167"}
MUTE = "#6b7177"

ym = lambda f: (lambda m: (int(m.group(1)), int(m.group(2))))(
    re.search(r"\.(\d{4})-(\d{2})\.nc$", f))
fyr = lambda k: k[0] + (k[1] - 0.5) / 12.0

def gx_box(sample):
    g = xr.open_dataset(sample)
    TLAT, TLONG = g.TLAT.values, g.TLONG.values
    TAREA = np.nan_to_num(g.TAREA.values.astype(np.float64))
    box = (TLAT >= -5) & (TLAT <= 5) & (TLONG >= 190) & (TLONG <= 240)
    jj, ii = np.where(box)
    J0, J1, I0, I1 = jj.min(), jj.max() + 1, ii.min(), ii.max() + 1
    return J0, J1, I0, I1, box[J0:J1, I0:I1], (TAREA * box)[J0:J1, I0:I1]

def wmean(sub, m, w):
    s = np.asarray(sub, np.float64)
    return float(np.nansum(np.where(np.isfinite(s) & m, s, 0) * w) / w.sum())

# ---------------- monthly box means, per source ----------------
def from_rundir(rundir):
    fs = sorted(glob.glob(f"{rundir}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
    if not fs: return {}
    J0, J1, I0, I1, m, w = gx_box(fs[0])
    return {ym(f): wmean(xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values[J0:J1, I0:I1], m, w)
            for f in fs}

def from_le2(y0, y1):
    J0, J1, I0, I1, m, w = gx_box(sorted(glob.glob(f"{args.run_ft}/*.pop.h.*-01.nc"))[0])
    out = {}
    for f in sorted(glob.glob(f"{LE2_SST_DIR}/*{args.ref_member}*.pop.h.SST.*.nc")):
        r = re.search(r"\.(\d{4})\d\d-(\d{4})\d\d\.nc$", f)
        if not r or int(r.group(1)) > y1 or int(r.group(2)) < y0: continue
        ds = xr.open_dataset(f)["SST"]
        if "z_t" in ds.dims: ds = ds.isel(z_t=0)
        sub = ds.isel(nlat=slice(J0, J1), nlon=slice(I0, I1)).values
        for i in range(ds.sizes["time"]):
            t = ds["time"].values[i] - timedelta(days=15)   # POP stamps end-of-month
            if y0 <= t.year <= y1: out[(t.year, t.month)] = wmean(sub[i], m, w)
    return out

def from_cam6(y0, y1):
    files = {y: os.path.join(CAM6_DIR,
             f"b.e21.CREDIT_climate_branch_1980_{y}_zmdata_ERA5scaled_zmdata_Qtot.zarr")
             for y in range(y0, y1 + 1)}
    files = {y: f for y, f in files.items() if os.path.isdir(f)}
    if not files: return {}
    g0 = xr.open_zarr(next(iter(files.values())), consolidated=False)
    latm, lonm = np.meshgrid(g0["latitude"].values, g0["longitude"].values, indexing="ij")
    box = (latm >= -5) & (latm <= 5) & (lonm >= 190) & (lonm <= 240)
    jj, ii = np.where(box)
    J0, J1, I0, I1 = jj.min(), jj.max() + 1, ii.min(), ii.max() + 1
    m = box[J0:J1, I0:I1]
    w = (np.cos(np.deg2rad(latm)) * box)[J0:J1, I0:I1]
    out = {}
    for y, f in files.items():
        ds = xr.open_zarr(f, consolidated=False)["SST"]
        arr = ds.isel(latitude=slice(J0, J1), longitude=slice(I0, I1)).values
        mon = np.array([int(str(t)[5:7]) for t in ds["time"].values])
        for k in range(1, 13):
            sel = mon == k
            if sel.any(): out[(y, k)] = wmean(arr[sel].mean(axis=0) - 273.15, m, w)
    return out

def pack(d):   return np.array([[k[0], k[1], v] for k, v in sorted(d.items())], np.float64)
def unpack(a): return {(int(r[0]), int(r[1])): r[2] for r in a}

if os.path.exists(args.cache):
    z = np.load(args.cache)
    S = {k: unpack(z[k]) for k in z.files}
    print(f"loaded cache {args.cache}")
else:
    S = {"ft": from_rundir(args.run_ft), "uft": from_rundir(args.run_uft)}
    yrs = sorted({k[0] for k in S["ft"]} | {k[0] for k in S["uft"]})
    S["le2"]  = from_le2(yrs[0], yrs[-1])
    S["cam6"] = from_cam6(yrs[0], yrs[-1])
    np.savez(args.cache, **{k: pack(v) for k, v in S.items()})
    print(f"wrote cache {args.cache}")

for k, v in S.items():
    ks = sorted(v)
    print(f"  {k:5s} {len(v):4d} months  [{ks[0]}..{ks[-1]}]" if v else f"  {k:5s} EMPTY")

# ---------------- anomalies on a COMMON baseline ----------------
def anomaly(d):
    """Deseasonalize with this series' own monthly climatology over the common
    baseline window; series lacking baseline coverage fall back to their own span."""
    ks = sorted(d)
    base = [k for k in ks if args.base_y0 <= k[0] <= args.base_y1] or ks
    clim = {}
    for mo in range(1, 13):
        v = [d[k] for k in base if k[1] == mo]
        if v: clim[mo] = float(np.mean(v))
    t = np.array([fyr(k) for k in ks])
    a = np.array([d[k] - clim.get(k[1], np.nan) for k in ks])
    return t, a, clim

def annual(d):
    yrs = sorted({k[0] for k in d if all((k[0], mo) in d for mo in range(1, 13))})
    return np.array(yrs), np.array([np.mean([d[(y, mo)] for mo in range(1, 13)]) for y in yrs])

LBL = {"ft": "Finetuned (sensreg) coupled sim", "uft": "Unfinetuned coupled sim",
       "cam6": "CAM6 / CREDIT (training truth)", "le2": "CESM2-LE (LE2-1231.002)"}
LS  = {"ft": "-", "uft": (0, (3, 1, 1, 1)), "cam6": (0, (1, 1.4)), "le2": (0, (5, 2))}
LW  = {"ft": 2.0, "uft": 1.6, "cam6": 1.4, "le2": 1.4}
ORD = ["le2", "cam6", "uft", "ft"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.6, 8.0), sharex=True)
fig.subplots_adjust(top=0.805, right=0.975, left=0.085, bottom=0.075, hspace=0.34)

ax1.axhline(0, color="#c7c7c7", lw=0.9, zorder=0)
for i, k in enumerate(ORD):
    if not S[k]: continue
    t, a, _ = anomaly(S[k])
    ax1.plot(t, a, color=C[k], lw=LW[k], ls=LS[k], dash_capstyle="round",
             solid_capstyle="round", zorder=1 + i,
             label=f"{LBL[k]}   σ = {np.nanstd(a):.2f} K")
    ys, ya = annual(S[k])
    ax2.plot(ys + 0.5, ya, color=C[k], lw=LW[k] + 0.4, ls=LS[k], dash_capstyle="round",
             solid_capstyle="round", zorder=1 + i,
             label=f"{LBL[k]}   mean {ya.mean():.2f} °C")

ax1.set_ylabel("Niño-3.4 monthly anomaly (K)")
ax2.set_ylabel("Niño-3.4 annual-mean SST (°C)")
ax2.set_xlabel("Year")
for ax in (ax1, ax2):
    ax.margins(x=0.008)
    ax.grid(axis="x", visible=False)
    ax.xaxis.set_major_locator(mtick.MultipleLocator(5))

fig.text(0.085, 0.985, "Niño-3.4 SST in the coupled ML–POP2 system",
         ha="left", va="top", fontsize=15, fontweight="bold")
fig.text(0.085, 0.945,
         f"5°S–5°N, 190–240°E, area-weighted · anomalies vs each series' own "
         f"{args.base_y0}–{args.base_y1} monthly climatology",
         ha="left", va="top", color=MUTE, fontsize=10)
# panel labels and legends sit ABOVE each axes so they never overlap the data
for ax, tag in ((ax1, "a"), (ax2, "b")):
    ax.text(-0.078, 1.185, tag, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11.5, fontweight="bold", color="#3d3d3d")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), frameon=False,
              handlelength=2.2, borderaxespad=0.0, labelcolor="linecolor",
              ncol=2, columnspacing=1.8, handletextpad=0.7)

fig.savefig(args.out); fig.savefig(args.out.replace(".png", ".pdf"))
print(f"wrote {args.out} (+ .pdf)")
for k in ORD:
    if not S[k]: continue
    t, a, _ = anomaly(S[k]); ys, ya = annual(S[k])
    print(f"  {k:5s} anom sigma {np.nanstd(a):.3f} K | annual mean {ya.mean():.3f} C "
          f"| {ys[0]}-{ys[-1]}")
