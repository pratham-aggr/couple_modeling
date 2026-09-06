#!/usr/bin/env python
"""Niño-3.4 SST by YEAR: coupled ML-POP2 sim vs CESM2-LE truth (LE2-1231.002).
One point per completed calendar year (annual mean of the monthly box-mean SST),
clean line comparison — the low-frequency companion to the monthly anomaly figure.

Re-runnable: only full calendar years present in the run directory are plotted, so
re-running as the simulation advances extends the figure automatically.

Usage:  python plot_nino34_annual.py [--run DIR] [--out PNG] [--ref-member ID]
"""
import argparse, glob, re, os
from datetime import timedelta
import numpy as np, xarray as xr
import matplotlib
from scipy import signal
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

RUN_DEFAULT = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_sensreg50yr/run"
LE2_SST_DIR = "/glade/campaign/cgd/cesm/CESM2-LE/ocn/proc/tseries/month_1/SST"
CAM6_DIR    = "/glade/derecho/scratch/wchapman/b_credit_runs"

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=RUN_DEFAULT)
ap.add_argument("--out", default="/glade/u/home/praggarwal/couple/output/nino34_annual.png")
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

# ---------- Niño-3.4 box + weights ----------
g=xr.open_dataset(mfiles[0]); TLAT,TLONG=g.TLAT.values,g.TLONG.values
TAREA=np.nan_to_num(g.TAREA.values.astype(np.float64))
box=(TLAT>=-5)&(TLAT<=5)&(TLONG>=190)&(TLONG<=240)
jj,ii=np.where(box); J0,J1,I0,I1=jj.min(),jj.max()+1,ii.min(),ii.max()+1
sub_box=box[J0:J1,I0:I1]; sub_w=(TAREA*box)[J0:J1,I0:I1]; sub_wsum=sub_w.sum()
def bmean(sub):
    s=np.asarray(sub,np.float64)
    return float(np.nansum(np.where(np.isfinite(s)&sub_box,s,0)*sub_w)/sub_wsum)

# ---------- model monthly box-mean SST, then keep only FULL calendar years ----------
msst_by_ym = {k: bmean(xr.open_dataset(f)["TEMP"].isel(time=0,z_t=0).values[J0:J1,I0:I1])
              for f, k in zip(mfiles, mkeys)}
years_avail = sorted({y for (y, m) in mkeys})
full_years = [y for y in years_avail if all((y, m) in msst_by_ym for m in range(1, 13))]
if not full_years: raise SystemExit("no complete calendar year yet — need 12 months of a year")
m_annual = np.array([np.mean([msst_by_ym[(y, m)] for m in range(1, 13)]) for y in full_years])
y0, y1 = full_years[0], full_years[-1]
print(f"model: {len(full_years)} complete years  [{y0}-{y1}]")

# ---------- LE2 truth: same years, annual mean ----------
want_ym = {(y, m) for y in full_years for m in range(1, 13)}
mfs=[f for f in sorted(glob.glob(f"{LE2_SST_DIR}/*{args.ref_member}*.pop.h.SST.*.nc"))
     if (lambda r: r and int(r.group(1))<=y1 and int(r.group(2))>=y0)(re.search(r"\.(\d{4})\d\d-(\d{4})\d\d\.nc$",f))]
tvals={}
for f in mfs:
    ds=xr.open_dataset(f)["SST"]
    if "z_t" in ds.dims: ds=ds.isel(z_t=0)
    sub=ds.isel(nlat=slice(J0,J1),nlon=slice(I0,I1)).values
    for i in range(ds.sizes["time"]):
        t=ds["time"].values[i]-timedelta(days=15)   # CESM stamps END of month -> mid
        key=(t.year,t.month)
        if key in want_ym: tvals[key]=bmean(sub[i])
t_annual = np.array([np.mean([tvals[(y, m)] for m in range(1, 13)]) for y in full_years])

# ---------- CAM6 (CREDIT training truth): same years, annual mean ----------
# Native CAM f09 lat-lon grid (192x288); cos(lat)-weighted box mean, same box.
# This is the SST boundary condition our atmosphere emulator trains against --
# the CAM6-analog reference, mirroring how the CAMulator paper always shows
# CAM6 alongside its own model's output.
cam_files = {y: os.path.join(CAM6_DIR, f"b.e21.CREDIT_climate_branch_1980_{y}_zmdata_ERA5scaled_zmdata_Qtot.zarr")
             for y in full_years}
cam_files = {y: f for y, f in cam_files.items() if os.path.isdir(f)}
cam_annual = None
if cam_files:
    g0 = xr.open_zarr(next(iter(cam_files.values())), consolidated=False)
    lat = g0["latitude"].values; lon = g0["longitude"].values
    latm, lonm = np.meshgrid(lat, lon, indexing="ij")
    cbox = (latm >= -5) & (latm <= 5) & (lonm >= 190) & (lonm <= 240)
    cw = np.cos(np.deg2rad(latm)) * cbox
    jj, ii = np.where(cbox); J0c, J1c, I0c, I1c = jj.min(), jj.max()+1, ii.min(), ii.max()+1
    sub_cbox = cbox[J0c:J1c, I0c:I1c]; sub_cw = cw[J0c:J1c, I0c:I1c]; cwsum = sub_cw.sum()
    cam_by_year = {}
    for y, f in cam_files.items():
        arr = xr.open_zarr(f, consolidated=False)["SST"].isel(
            latitude=slice(J0c, J1c), longitude=slice(I0c, I1c)).values
        fld = arr.mean(axis=0)
        cam_by_year[y] = float(np.nansum(np.where(sub_cbox, fld, 0) * sub_cw) / cwsum) - 273.15  # K -> degC
    if len(cam_by_year) == len(full_years):
        cam_annual = np.array([cam_by_year[y] for y in full_years])

# ---------- figure ----------
fig,ax=plt.subplots(figsize=(8.6,4.3))
fig.subplots_adjust(top=0.86, right=0.97, left=0.10, bottom=0.11)
ax.plot(full_years, t_annual, color=C_TRUTH, lw=1.6, ls=(0,(5,2)), dash_capstyle="round",
        marker="o", ms=4.5, mfc=C_TRUTH, mec="none", zorder=2,
        label=f"CESM2-LE truth  (mean {t_annual.mean():.2f} °C)")
if cam_annual is not None:
    ax.plot(full_years, cam_annual, color=C_CAM6, lw=1.6, ls=(0,(1,1.4)), dash_capstyle="round",
            marker="o", ms=4.5, mfc=C_CAM6, mec="none", zorder=1,
            label=f"CAM6 (training truth)  (mean {cam_annual.mean():.2f} °C)")
ax.plot(full_years, m_annual, color=C_SIM, lw=2.4, solid_capstyle="round",
        marker="o", ms=5.5, mfc=C_SIM, mec="none", zorder=3,
        label=f"Coupled ML–POP2  (mean {m_annual.mean():.2f} °C)")
ax.set_ylabel("Niño-3.4 SST (°C)")
fig.text(0.10, 0.97, "Niño-3.4 SST by year", ha="left", va="top", fontsize=15, fontweight="bold")
fig.text(0.10, 0.905, f"{y0}–{y1} annual mean", ha="left", va="top", color=MUTE, fontsize=10.5)
ax.legend(loc="lower left", frameon=False, handlelength=1.6, borderaxespad=0.3, labelcolor="linecolor")
ax.margins(x=0.03)
ax.xaxis.set_major_locator(mtick.MultipleLocator(1))
fig.savefig(args.out); fig.savefig(args.out.replace(".png",".pdf"))
print(f"wrote {args.out} (+ .pdf)")
for i, y in enumerate(full_years):
    line = f"  {y}  sim {m_annual[i]:6.2f}  truth {t_annual[i]:6.2f}  diff {m_annual[i]-t_annual[i]:+.2f}"
    if cam_annual is not None:
        line += f"  cam6 {cam_annual[i]:6.2f}"
    print(line)
