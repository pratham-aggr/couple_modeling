#!/usr/bin/env python
"""Publication figure: GLOBAL-mean SST by year -- finetuned coupled ML-POP2 sim
vs two ground truths (CESM2-LE member LE2-1231.002, CAM6/CREDIT training
truth). Three series only, for a clean paper figure. One point per completed
calendar year (area-weighted, TAREA-weighted over the full gx1v7 ocean grid).

Re-runnable: extends automatically as the finetuned run advances (currently
self-chaining past 50 years, job stream via run_pop_sensreg_50yr.pbs).

Usage:  python plot_sst_annual_publication.py [--run DIR] [--out PNG] [--ref-member ID]
"""
import argparse, glob, re, os
from datetime import timedelta
import numpy as np, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

RUN_DEFAULT = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_sensreg50yr/run"
LE2_SST_DIR = "/glade/campaign/cgd/cesm/CESM2-LE/ocn/proc/tseries/month_1/SST"
CAM6_DIR    = "/glade/derecho/scratch/wchapman/b_credit_runs"

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=RUN_DEFAULT, help="finetuned (sensreg) rundir")
ap.add_argument("--out", default="/glade/u/home/praggarwal/couple/output/sst_annual_publication.png")
ap.add_argument("--ref-member", default="LE2-1231.002")
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
C_SIM, C_TRUTH, C_CAM6, MUTE = "#D55E00", "#5b6167", "#0072B2", "#6b7177"

# ---------- model months (auto-detect) ----------
mfiles = sorted(glob.glob(f"{args.run}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
if not mfiles: raise SystemExit(f"no monthly pop.h files in {args.run}")
def ym(f): m=re.search(r"\.(\d{4})-(\d{2})\.nc$", f); return int(m.group(1)), int(m.group(2))
mkeys=[ym(f) for f in mfiles]

# ---------- global area weights (TAREA, full grid) ----------
g=xr.open_dataset(mfiles[0])
TAREA=np.nan_to_num(g["TAREA"].values.astype(np.float64))
def gmean(sst2d):
    s=np.asarray(sst2d,np.float64); m=np.isfinite(s)
    return float(np.nansum(np.where(m,s,0)*TAREA)/ (TAREA*m).sum())

# ---------- model monthly global-mean SST, keep only FULL calendar years ----------
msst_by_ym = {k: gmean(xr.open_dataset(f)["TEMP"].isel(time=0,z_t=0).values)
              for f, k in zip(mfiles, mkeys)}
years_avail = sorted({y for (y, m) in mkeys})
full_years = [y for y in years_avail if all((y, m) in msst_by_ym for m in range(1, 13))]
if not full_years: raise SystemExit("no complete calendar year yet")
m_annual = np.array([np.mean([msst_by_ym[(y, m)] for m in range(1, 13)]) for y in full_years])
y0, y1 = full_years[0], full_years[-1]
print(f"finetuned: {len(full_years)} complete years  [{y0}-{y1}]")

# ---------- LE2 truth global-mean SST, same years ----------
want_ym = {(y, m) for y in full_years for m in range(1, 13)}
mfs=[f for f in sorted(glob.glob(f"{LE2_SST_DIR}/*{args.ref_member}*.pop.h.SST.*.nc"))
     if (lambda r: r and int(r.group(1))<=y1 and int(r.group(2))>=y0)(re.search(r"\.(\d{4})\d\d-(\d{4})\d\d\.nc$",f))]
tvals={}
for f in mfs:
    ds=xr.open_dataset(f)["SST"]
    if "z_t" in ds.dims: ds=ds.isel(z_t=0)
    for i in range(ds.sizes["time"]):
        t=ds["time"].values[i]-timedelta(days=15)   # CESM stamps END of month -> mid
        key=(t.year,t.month)
        if key in want_ym: tvals[key]=gmean(ds.isel(time=i).values)
t_annual = np.array([np.mean([tvals[(y, m)] for m in range(1, 13)]) for y in full_years])

# ---------- CAM6 (CREDIT training truth) global-mean SST -- only covers 1980-2014 ----------
cam_files = {y: os.path.join(CAM6_DIR, f"b.e21.CREDIT_climate_branch_1980_{y}_zmdata_ERA5scaled_zmdata_Qtot.zarr")
             for y in full_years}
cam_files = {y: f for y, f in cam_files.items() if os.path.isdir(f)}
cam_years, cam_annual = [], None
if cam_files:
    g0 = xr.open_zarr(next(iter(cam_files.values())), consolidated=False)
    lat = g0["latitude"].values
    coslat = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, g0.sizes["longitude"]))
    cam_by_year = {}
    for y, f in cam_files.items():
        ds = xr.open_zarr(f, consolidated=False)
        sst = ds["SST"].values - 273.15                     # K -> degC
        ocnw = coslat * (1.0 - ds["LANDFRAC"].values)        # (time, nlat, nlon)
        cam_by_year[y] = float(np.nansum(sst * ocnw) / np.nansum(ocnw))
    cam_years = sorted(cam_by_year)
    cam_annual = np.array([cam_by_year[y] for y in cam_years])

# ---------- figure ----------
fig,ax=plt.subplots(figsize=(9.0,4.6))
fig.subplots_adjust(top=0.86, right=0.97, left=0.10, bottom=0.11)
ax.plot(full_years, t_annual, color=C_TRUTH, lw=1.8, ls=(0,(5,2)), dash_capstyle="round",
        marker="o", ms=4.5, mfc=C_TRUTH, mec="none", zorder=1,
        label=f"CESM2-LE truth  (mean {t_annual.mean():.2f} °C)")
if cam_annual is not None:
    ax.plot(cam_years, cam_annual, color=C_CAM6, lw=1.8, ls=(0,(1,1.4)), dash_capstyle="round",
            marker="o", ms=4.5, mfc=C_CAM6, mec="none", zorder=2,
            label=f"CAM6 (training truth)  (mean {cam_annual.mean():.2f} °C)")
ax.plot(full_years, m_annual, color=C_SIM, lw=2.6, solid_capstyle="round",
        marker="o", ms=5.5, mfc=C_SIM, mec="none", zorder=3,
        label=f"Finetuned coupled sim  (mean {m_annual.mean():.2f} °C)")
ax.set_ylabel("Global-mean SST (°C)")
fig.text(0.10, 0.97, "Global-mean SST by year", ha="left", va="top", fontsize=15, fontweight="bold")
fig.text(0.10, 0.905, f"{y0}–{y1} annual mean, area-weighted", ha="left", va="top", color=MUTE, fontsize=10.5)
ax.legend(loc="best", frameon=False, handlelength=1.6, borderaxespad=0.3, labelcolor="linecolor")
ax.margins(x=0.03)
ax.xaxis.set_major_locator(mtick.MultipleLocator(5))
fig.savefig(args.out); fig.savefig(args.out.replace(".png",".pdf"))
print(f"wrote {args.out} (+ .pdf)")
for i, y in enumerate(full_years):
    line = f"  {y}  sim {m_annual[i]:6.3f}  truth {t_annual[i]:6.3f}  diff {m_annual[i]-t_annual[i]:+.3f}"
    if cam_annual is not None and y in cam_by_year:
        line += f"  cam6 {cam_by_year[y]:6.3f}"
    print(line)
