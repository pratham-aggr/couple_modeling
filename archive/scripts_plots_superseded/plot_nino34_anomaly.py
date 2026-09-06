#!/usr/bin/env python
"""Simple, single-panel Niño-3.4 ANOMALY figure: sensreg coupled sim vs a single
CESM2-LE reference member (LE2-1231.002) vs CAM6 (the ERA5-scaled CREDIT run our
atmosphere emulator is trained on -- its prescribed SST boundary condition,
analogous to how the CAMulator paper compares against its own CAM6 reference).
Re-runnable — auto-detects whatever monthly pop.h files exist and plots exactly
that span.

Usage:  python plot_nino34_anomaly.py [--run DIR] [--out PNG] [--ref-member ID]
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
ap.add_argument("--run", default=RUN_DEFAULT)
ap.add_argument("--out", default="/glade/u/home/praggarwal/couple/output/nino34_anomaly.png")
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
# Colorblind-safe (Okabe-Ito) AND black/white-print-safe: three series, three
# hues from the CVD-safe palette, EACH also carrying a distinct linestyle so
# identity never depends on hue discrimination alone.
C_SIM, C_TRUTH, C_CAM6, MUTE = "#D55E00", "#5b6167", "#0072B2", "#6b7177"

# ---------- model months (auto-detect) ----------
mfiles = sorted(glob.glob(f"{args.run}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
if not mfiles: raise SystemExit(f"no monthly pop.h files in {args.run}")
def ym(f): m=re.search(r"\.(\d{4})-(\d{2})\.nc$", f); return int(m.group(1)), int(m.group(2))
mkeys=[ym(f) for f in mfiles]; y0,y1=mkeys[0][0],mkeys[-1][0]

# ---------- Niño-3.4 box + weights (gx1v7 curvilinear grid) ----------
g=xr.open_dataset(mfiles[0]); TLAT,TLONG=g.TLAT.values,g.TLONG.values
TAREA=np.nan_to_num(g.TAREA.values.astype(np.float64))
box=(TLAT>=-5)&(TLAT<=5)&(TLONG>=190)&(TLONG<=240)
jj,ii=np.where(box); J0,J1,I0,I1=jj.min(),jj.max()+1,ii.min(),ii.max()+1
sub_box=box[J0:J1,I0:I1]; sub_w=(TAREA*box)[J0:J1,I0:I1]; sub_wsum=sub_w.sum()
def bmean(sub):
    s=np.asarray(sub,np.float64)
    return float(np.nansum(np.where(np.isfinite(s)&sub_box,s,0)*sub_w)/sub_wsum)

def deseason(series, months):
    a=series.astype(np.float64).copy()
    for k in range(1,13):
        sel=months==k
        if sel.any(): a[sel]=series[sel]-series[sel].mean()
    return a
fyr=lambda yy,mm: yy+(mm-0.5)/12.0

# ---------- model anomaly ----------
mt=np.array([fyr(*k) for k in mkeys]); mmon=np.array([k[1] for k in mkeys])
msst=np.array([bmean(xr.open_dataset(f)["TEMP"].isel(time=0,z_t=0).values[J0:J1,I0:I1]) for f in mfiles])
ma=deseason(msst,mmon)

# ---------- LE2 reference member anomaly ----------
want=set(mkeys)
mfs=[f for f in sorted(glob.glob(f"{LE2_SST_DIR}/*{args.ref_member}*.pop.h.SST.*.nc"))
     if (lambda r: r and int(r.group(1))<=y1 and int(r.group(2))>=y0)(re.search(r"\.(\d{4})\d\d-(\d{4})\d\d\.nc$",f))]
vals={}
for f in mfs:
    ds=xr.open_dataset(f)["SST"]
    if "z_t" in ds.dims: ds=ds.isel(z_t=0)
    sub=ds.isel(nlat=slice(J0,J1),nlon=slice(I0,I1)).values
    for i in range(ds.sizes["time"]):
        t=ds["time"].values[i]-timedelta(days=15)   # CESM stamps END of month -> shift to mid
        key=(t.year,t.month)
        if key in want: vals[key]=bmean(sub[i])
ref_series=np.array([vals.get(k,np.nan) for k in mkeys])
ref_a=deseason(ref_series,mmon)

# ---------- CAM6 (CREDIT training truth) Niño-3.4 anomaly ----------
# Native CAM f09 lat-lon grid (192x288); cos(lat)-weighted box mean, same
# 5S-5N/170W-120W box, same deseasonalizing convention as the other two series.
# This is the SST boundary condition our atmosphere emulator trains against
# (b.e21.CREDIT_climate_branch_*_ERA5scaled_..._Qtot.zarr) -- the CAM6-analog
# reference, mirroring how the CAMulator paper always shows CAM6 alongside its
# emulator's own output.
cam_years = sorted({y for (y, m) in mkeys})
cam_files = {y: os.path.join(CAM6_DIR, f"b.e21.CREDIT_climate_branch_1980_{y}_zmdata_ERA5scaled_zmdata_Qtot.zarr")
             for y in cam_years}
cam_files = {y: f for y, f in cam_files.items() if os.path.isdir(f)}
cam_vals = {}
if cam_files:
    g0 = xr.open_zarr(next(iter(cam_files.values())), consolidated=False)
    lat = g0["latitude"].values; lon = g0["longitude"].values
    latm, lonm = np.meshgrid(lat, lon, indexing="ij")
    cbox = (latm >= -5) & (latm <= 5) & (lonm >= 190) & (lonm <= 240)
    cw = np.cos(np.deg2rad(latm)) * cbox
    cwsum = cw.sum()
    for y, f in cam_files.items():
        ds = xr.open_zarr(f, consolidated=False)["SST"]
        jj, ii = np.where(cbox)
        J0c, J1c, I0c, I1c = jj.min(), jj.max()+1, ii.min(), ii.max()+1
        sub_cbox = cbox[J0c:J1c, I0c:I1c]; sub_cw = cw[J0c:J1c, I0c:I1c]
        arr = ds.isel(latitude=slice(J0c, J1c), longitude=slice(I0c, I1c)).values
        tvals = ds["time"].values
        months = np.array([int(str(t)[5:7]) for t in tvals])
        for m in range(1, 13):
            sel = months == m
            if not sel.any(): continue
            fld = arr[sel].mean(axis=0)
            cam_vals[(y, m)] = float(np.nansum(np.where(sub_cbox, fld, 0) * sub_cw) / cwsum)
cam_series = np.array([cam_vals.get(k, np.nan) for k in mkeys])
cam_a = deseason(cam_series, mmon) if np.isfinite(cam_series).any() else None

# ---------- figure: single panel, minimal ----------
fig,ax=plt.subplots(figsize=(9.2,4.3))
fig.subplots_adjust(top=0.86, right=0.97, left=0.09, bottom=0.11)
ax.axhline(0,color="#c7c7c7",lw=0.9,zorder=0)
ax.plot(mt,ref_a,color=C_TRUTH,lw=1.6,ls=(0,(5,2)),dash_capstyle="round",zorder=2,
        label=f"CESM2-LE truth   σ = {np.nanstd(ref_a):.2f} K")
if cam_a is not None:
    ax.plot(mt,cam_a,color=C_CAM6,lw=1.6,ls=(0,(1,1.4)),dash_capstyle="round",zorder=1,
            label=f"CAM6 (training truth)   σ = {np.nanstd(cam_a):.2f} K")
ax.plot(mt,ma,color=C_SIM,lw=2.4,solid_capstyle="round",zorder=3,
        label=f"Coupled ML–POP2   σ = {ma.std():.2f} K")

ax.set_ylabel("Niño-3.4 anomaly (K)")
fig.text(0.09, 0.97, "Niño-3.4 anomaly", ha="left", va="top",
         fontsize=15, fontweight="bold")
fig.text(0.09, 0.905, f"{y0}–{y1}, deseasonalized", ha="left", va="top",
         color=MUTE, fontsize=10.5)
ax.legend(loc="lower left", frameon=False, handlelength=1.6,
          borderaxespad=0.3, labelcolor="linecolor")
ax.margins(x=0.008); ax.grid(axis="x", visible=False)
ax.xaxis.set_major_locator(mtick.MultipleLocator(2))
ax.yaxis.set_major_locator(mtick.MultipleLocator(2))
fig.savefig(args.out); fig.savefig(args.out.replace(".png",".pdf"))
print(f"wrote {args.out} (+ .pdf)")
print(f"  sim   anomaly σ {ma.std():.3f} K")
print(f"  truth anomaly σ {np.nanstd(ref_a):.3f} K")
if cam_a is not None:
    print(f"  CAM6  anomaly σ {np.nanstd(cam_a):.3f} K  ({len(cam_files)} yr matched)")
else:
    print("  CAM6  series: no matching zarr years found")
