"""Global-mean SST timeseries: bias-corrected sim + live baseline sim, plotted
alongside CESM2-LE truth and CAM6 (CREDIT training truth) references.
"""
import glob, re, os
from datetime import timedelta
import numpy as np, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_BC = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_debiascheck/run"
RUN_LIVE = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_sensreg50yr/run"
RUN_V15 = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_v15unet5yr/run"
# unfinetuned pz220f0b base (no dQ/dSST sensreg fine-tune), interactive ice
RUN_UFT = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw_iceloop/run"
LE2_SST_DIR = "/glade/campaign/cgd/cesm/CESM2-LE/ocn/proc/tseries/month_1/SST"
CAM6_DIR = "/glade/derecho/scratch/wchapman/b_credit_runs"
REF_MEMBER = "LE2-1231.002"
# CESM2 piControl (unforced control, native gx1v7 = same grid as the sims).
# 1200 model years, so it cannot share a calendar axis with the sims -- it is
# used as a drift/variability REFERENCE: mean +/- 1 sigma band across the axis,
# plus its first N years overlaid by ELAPSED year for a like-for-like drift rate.
PICTL_DIR = ("/glade/collections/cmip/CMIP6/CMIP/NCAR/CESM2/piControl/"
             "r1i1p1f1/Omon/tos/gn/latest")
PICTL_CACHE = "output/picontrol_gmsst_annual.npz"

def gmean(sst2d, TAREA):
    s = np.asarray(sst2d, np.float64)
    m = np.isfinite(s)
    return float(np.nansum(np.where(m, s, 0) * TAREA) / (TAREA * m).sum())

def annual_from_rundir(rundir):
    mfiles = sorted(glob.glob(f"{rundir}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
    def ym(f):
        m = re.search(r"\.(\d{4})-(\d{2})\.nc$", f)
        return int(m.group(1)), int(m.group(2))
    mkeys = [ym(f) for f in mfiles]
    g = xr.open_dataset(mfiles[0])
    TAREA = np.nan_to_num(g["TAREA"].values.astype(np.float64))
    msst = {k: gmean(xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values, TAREA)
            for f, k in zip(mfiles, mkeys)}
    years_avail = sorted({y for (y, m) in mkeys})
    full_years = [y for y in years_avail if all((y, m) in msst for m in range(1, 13))]
    annual = np.array([np.mean([msst[(y, m)] for m in range(1, 13)]) for y in full_years])
    return full_years, annual, TAREA

years_bc, ann_bc, TAREA = annual_from_rundir(RUN_BC)
years_live, ann_live, _ = annual_from_rundir(RUN_LIVE)
years_v15, ann_v15, _ = annual_from_rundir(RUN_V15)
years_uft, ann_uft, _ = annual_from_rundir(RUN_UFT)
print(f"bias-corrected: {years_bc[0]}-{years_bc[-1]} ({len(years_bc)} yr)")
print(f"live baseline : {years_live[0]}-{years_live[-1]} ({len(years_live)} yr)")
print(f"v15 (real winds): {years_v15[0]}-{years_v15[-1]} ({len(years_v15)} yr)")
print(f"unfinetuned    : {years_uft[0]}-{years_uft[-1]} ({len(years_uft)} yr)")

y0 = min(years_bc[0], years_live[0], years_v15[0], years_uft[0])
y1 = max(years_bc[-1], years_live[-1], years_v15[-1], years_uft[-1])

# ---------- LE2 truth ----------
want_ym = {(y, m) for y in range(y0, y1 + 1) for m in range(1, 13)}
mfs = [f for f in sorted(glob.glob(f"{LE2_SST_DIR}/*{REF_MEMBER}*.pop.h.SST.*.nc"))
       if (lambda r: r and int(r.group(1)) <= y1 and int(r.group(2)) >= y0)(
           re.search(r"\.(\d{4})\d\d-(\d{4})\d\d\.nc$", f))]
tvals = {}
for f in mfs:
    ds = xr.open_dataset(f)["SST"]
    if "z_t" in ds.dims:
        ds = ds.isel(z_t=0)
    for i in range(ds.sizes["time"]):
        t = ds["time"].values[i] - timedelta(days=15)
        key = (t.year, t.month)
        if key in want_ym:
            tvals[key] = gmean(ds.isel(time=i).values, TAREA)
truth_years = [y for y in range(y0, y1 + 1) if all((y, m) in tvals for m in range(1, 13))]
truth_annual = np.array([np.mean([tvals[(y, m)] for m in range(1, 13)]) for y in truth_years])

# ---------- CAM6 (CREDIT training truth), only covers 1980-2014 ----------
cam_files = {y: os.path.join(CAM6_DIR, f"b.e21.CREDIT_climate_branch_1980_{y}_zmdata_ERA5scaled_zmdata_Qtot.zarr")
             for y in range(y0, y1 + 1)}
cam_files = {y: f for y, f in cam_files.items() if os.path.isdir(f)}
cam_years, cam_annual = [], None
if cam_files:
    g0 = xr.open_zarr(next(iter(cam_files.values())), consolidated=False)
    lat = g0["latitude"].values
    coslat = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, g0.sizes["longitude"]))
    cam_by_year = {}
    for y, f in cam_files.items():
        ds = xr.open_zarr(f, consolidated=False)
        sst = ds["SST"].values - 273.15
        ocnw = coslat * (1.0 - ds["LANDFRAC"].values)
        cam_by_year[y] = float(np.nansum(sst * ocnw) / np.nansum(ocnw))
    cam_years = sorted(cam_by_year)
    cam_annual = np.array([cam_by_year[y] for y in cam_years])

# ---------- CESM2 piControl (unforced control) ----------
def picontrol_annual(TAREA):
    if os.path.exists(PICTL_CACHE):
        z = np.load(PICTL_CACHE)
        return z["years"], z["annual"]
    files = sorted(glob.glob(f"{PICTL_DIR}/*.nc"))
    yrs, ann = [], []
    w0 = None
    for f in files:
        ds = xr.open_dataset(f, decode_times=False)
        tos = ds["tos"]
        # months since 0001-01; the files are contiguous monthly, 12 per year
        t0 = int(round(float(ds["time"].values[0]) / 365.0))   # NOLEAP -> years
        n = tos.sizes["time"]
        for i0 in range(0, n - n % 12, 12):
            blk = tos.isel(time=slice(i0, i0 + 12)).values.astype(np.float64)
            if w0 is None:
                w0 = TAREA * np.isfinite(blk[0])
                wsum = w0.sum()
            m = np.nansum(np.where(np.isfinite(blk), blk, 0) * w0, axis=(1, 2)) / wsum
            yrs.append(t0 + i0 // 12 + 1)
            ann.append(float(m.mean()))
        ds.close()
    years = np.array(yrs); annual = np.array(ann)
    np.savez(PICTL_CACHE, years=years, annual=annual)
    return years, annual

pic_years, pic_annual = picontrol_annual(TAREA)
pic_mean, pic_sd = float(pic_annual.mean()), float(pic_annual.std())
print(f"piControl      : model yr {pic_years[0]}-{pic_years[-1]} ({len(pic_years)} yr), "
      f"mean {pic_mean:.3f} +/- {pic_sd:.3f} degC, "
      f"end-start {pic_annual[-1]-pic_annual[0]:+.3f} K")

# ---------- figure ----------
fig, ax = plt.subplots(figsize=(10, 6))
# piControl reference level: band first so every series draws on top of it
ax.axhspan(pic_mean - pic_sd, pic_mean + pic_sd, color="#9467bd", alpha=0.12, zorder=0)
ax.axhline(pic_mean, color="#9467bd", lw=1.4, ls=":", zorder=0,
           label=f"CESM2 piControl mean $\\pm$1$\\sigma$ ({pic_mean:.2f} °C, {len(pic_years)} yr)")
# piControl overlaid by ELAPSED year (model yr 1 -> {y0}) for a like-for-like drift rate
_npic = min(len(pic_years), y1 - y0 + 1)
ax.plot(np.arange(y0, y0 + _npic), pic_annual[:_npic], color="#9467bd", lw=1.3,
        alpha=0.75, zorder=1, label="CESM2 piControl (elapsed model yr, unforced)")
ax.plot(truth_years, truth_annual, color="#5b6167", lw=1.8, ls=(0, (5, 2)),
        marker="o", ms=4, label=f"CESM2-LE truth ({REF_MEMBER})")
if cam_annual is not None:
    ax.plot(cam_years, cam_annual, color="#0072B2", lw=1.8, ls=(0, (1, 1.4)),
            marker="o", ms=4, label="CAM6 (CREDIT training truth)")
ax.plot(years_live, ann_live, 's-', color='tab:blue', ms=3, lw=1.8,
        label='live baseline sensreg50yr (7232171)')
ax.plot(years_bc, ann_bc, 'o-', color='tab:red', ms=4.5, lw=2.2,
        label='bias-corrected (7237474)')
ax.plot(years_v15, ann_v15, '^-', color='tab:green', ms=5, lw=1.8,
        label='v15 sensreg + prescribed real winds (6907766)')
ax.plot(years_uft, ann_uft, 'd-', color='tab:orange', ms=3, lw=1.6,
        label='unfinetuned pz220f0b, interactive ice (7239950)')
ax.set_xlabel("year")
ax.set_ylabel("Global-mean SST (degC)")
ax.set_title("Global-mean SST timeseries: sims vs truth references")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
out = "output/sst_global_timeseries_with_truth.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")
