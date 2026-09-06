#!/usr/bin/env python
"""Publication figure: Nino-3.4 SST by year -- finetuned coupled ML-POP2 sim
vs two ground truths (CESM2-LE member LE2-1231.002, and CAM6/CREDIT training
truth). Three series only (no unfinetuned variants) for a clean paper figure.

Re-runnable: extends automatically as the finetuned run advances (currently
self-chaining past 50 years, job stream via run_pop_sensreg_50yr.pbs).

Usage:  python plot_nino34_publication.py [--run-ft DIR] [--out PNG]
"""
import argparse, glob, re, os
from datetime import timedelta
import numpy as np, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

RUN_FT_DEFAULT = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_sensreg50yr/run"
LE2_SST_DIR = "/glade/campaign/cgd/cesm/CESM2-LE/ocn/proc/tseries/month_1/SST"
CAM6_DIR    = "/glade/derecho/scratch/wchapman/b_credit_runs"

ap = argparse.ArgumentParser()
ap.add_argument("--run-ft", default=RUN_FT_DEFAULT, help="finetuned (sensreg) rundir")
ap.add_argument("--out", default="/glade/u/home/praggarwal/couple/output/nino34_publication.png")
ap.add_argument("--ref-member", default="LE2-1231.002")
args = ap.parse_args()

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 13.5,
    "axes.labelsize": 12, "legend.fontsize": 11, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "axes.linewidth": 0, "axes.edgecolor": "#4d4d4d",
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#ececec",
    "grid.linewidth": 0.7, "xtick.color": "#6b7177", "ytick.color": "#6b7177",
    "text.color": "#1a1a1a", "axes.labelcolor": "#3d3d3d",
    "savefig.dpi": 300, "figure.dpi": 140,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "xtick.major.size": 0, "ytick.major.size": 0,
})
# Okabe-Ito CVD-safe palette; each series also has its own linestyle so
# identity survives black/white printing.
C_FT, C_TRUTH, C_CAM6, MUTE = "#D55E00", "#5b6167", "#0072B2", "#6b7177"

def ym(f):
    m = re.search(r"\.(\d{4})-(\d{2})\.nc$", f)
    return int(m.group(1)), int(m.group(2))

def load_box_weights(sample_file):
    g = xr.open_dataset(sample_file)
    TLAT, TLONG = g.TLAT.values, g.TLONG.values
    TAREA = np.nan_to_num(g.TAREA.values.astype(np.float64))
    box = (TLAT >= -5) & (TLAT <= 5) & (TLONG >= 190) & (TLONG <= 240)
    jj, ii = np.where(box)
    J0, J1, I0, I1 = jj.min(), jj.max()+1, ii.min(), ii.max()+1
    sub_box = box[J0:J1, I0:I1]; sub_w = (TAREA*box)[J0:J1, I0:I1]
    return J0, J1, I0, I1, sub_box, sub_w, sub_w.sum()

def bmean(sub, sub_box, sub_w, sub_wsum):
    s = np.asarray(sub, np.float64)
    return float(np.nansum(np.where(np.isfinite(s) & sub_box, s, 0) * sub_w) / sub_wsum)

def annual_series_from_rundir(rundir):
    mfiles = sorted(glob.glob(f"{rundir}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
    if not mfiles:
        return None, None, None
    mkeys = [ym(f) for f in mfiles]
    box = load_box_weights(mfiles[0])
    J0, J1, I0, I1, sub_box, sub_w, sub_wsum = box
    msst_by_ym = {k: bmean(xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values[J0:J1, I0:I1],
                            sub_box, sub_w, sub_wsum)
                  for f, k in zip(mfiles, mkeys)}
    years_avail = sorted({y for (y, m) in mkeys})
    full_years = [y for y in years_avail if all((y, m) in msst_by_ym for m in range(1, 13))]
    if not full_years:
        return None, None, None
    annual = np.array([np.mean([msst_by_ym[(y, m)] for m in range(1, 13)]) for y in full_years])
    return full_years, annual, box

ft_years, ft_annual, ft_box = annual_series_from_rundir(args.run_ft)
if ft_years is None:
    raise SystemExit("no complete calendar year in finetuned rundir")
print(f"finetuned: {len(ft_years)} complete years  [{ft_years[0]}-{ft_years[-1]}]")

y0, y1 = ft_years[0], ft_years[-1]
J0, J1, I0, I1, sub_box, sub_w, sub_wsum = ft_box

# ---------- CESM2-LE truth ----------
mfs = [f for f in sorted(glob.glob(f"{LE2_SST_DIR}/*{args.ref_member}*.pop.h.SST.*.nc"))
       if (lambda r: r and int(r.group(1)) <= y1 and int(r.group(2)) >= y0)(
           re.search(r"\.(\d{4})\d\d-(\d{4})\d\d\.nc$", f))]
tvals = {}
for f in mfs:
    ds = xr.open_dataset(f)["SST"]
    if "z_t" in ds.dims: ds = ds.isel(z_t=0)
    sub = ds.isel(nlat=slice(J0, J1), nlon=slice(I0, I1)).values
    for i in range(ds.sizes["time"]):
        t = ds["time"].values[i] - timedelta(days=15)
        key = (t.year, t.month)
        if y0 <= key[0] <= y1: tvals[key] = bmean(sub[i], sub_box, sub_w, sub_wsum)
truth_years = [y for y in ft_years if all((y, m) in tvals for m in range(1, 13))]
t_annual = np.array([np.mean([tvals[(y, m)] for m in range(1, 13)]) for y in truth_years]) if truth_years else None

# ---------- CAM6 (training truth) -- only covers 1980-2014 ----------
cam_files = {y: os.path.join(CAM6_DIR, f"b.e21.CREDIT_climate_branch_1980_{y}_zmdata_ERA5scaled_zmdata_Qtot.zarr")
             for y in ft_years}
cam_files = {y: f for y, f in cam_files.items() if os.path.isdir(f)}
cam_by_year = {}
if cam_files:
    g0 = xr.open_zarr(next(iter(cam_files.values())), consolidated=False)
    lat = g0["latitude"].values; lon = g0["longitude"].values
    latm, lonm = np.meshgrid(lat, lon, indexing="ij")
    cbox = (latm >= -5) & (latm <= 5) & (lonm >= 190) & (lonm <= 240)
    cw = np.cos(np.deg2rad(latm)) * cbox
    jj, ii = np.where(cbox); J0c, J1c, I0c, I1c = jj.min(), jj.max()+1, ii.min(), ii.max()+1
    sub_cbox = cbox[J0c:J1c, I0c:I1c]; sub_cw = cw[J0c:J1c, I0c:I1c]; cwsum = sub_cw.sum()
    for y, f in cam_files.items():
        arr = xr.open_zarr(f, consolidated=False)["SST"].isel(
            latitude=slice(J0c, J1c), longitude=slice(I0c, I1c)).values
        fld = arr.mean(axis=0) - 273.15
        cam_by_year[y] = float(np.nansum(np.where(sub_cbox, fld, 0) * sub_cw) / cwsum)
cam_years = sorted(cam_by_year)
cam_annual = np.array([cam_by_year[y] for y in cam_years]) if cam_years else None

# ---------- figure ----------
fig, ax = plt.subplots(figsize=(9.0, 4.6))
fig.subplots_adjust(top=0.86, right=0.97, left=0.09, bottom=0.11)

if t_annual is not None:
    ax.plot(truth_years, t_annual, color=C_TRUTH, lw=1.8, ls=(0, (5, 2)), dash_capstyle="round",
            marker="o", ms=4.5, mfc=C_TRUTH, mec="none", zorder=1,
            label=f"CESM2-LE truth  (mean {t_annual.mean():.2f} °C)")
if cam_annual is not None:
    ax.plot(cam_years, cam_annual, color=C_CAM6, lw=1.8, ls=(0, (1, 1.4)), dash_capstyle="round",
            marker="o", ms=4.5, mfc=C_CAM6, mec="none", zorder=2,
            label=f"CAM6 (training truth)  (mean {cam_annual.mean():.2f} °C)")
ax.plot(ft_years, ft_annual, color=C_FT, lw=2.6, solid_capstyle="round",
        marker="o", ms=5.5, mfc=C_FT, mec="none", zorder=3,
        label=f"Finetuned coupled sim  (mean {ft_annual.mean():.2f} °C)")

ax.set_ylabel("Niño-3.4 SST (°C)")
fig.text(0.09, 0.97, "Niño-3.4 SST by year", ha="left", va="top",
         fontsize=14.5, fontweight="bold")
fig.text(0.09, 0.905, f"{y0}–{y1} annual mean", ha="left", va="top", color=MUTE, fontsize=10.5)
ax.legend(loc="lower left", frameon=False, handlelength=1.8, borderaxespad=0.3, labelcolor="linecolor")
ax.margins(x=0.03)
ax.xaxis.set_major_locator(mtick.MultipleLocator(5))
fig.savefig(args.out); fig.savefig(args.out.replace(".png", ".pdf"))
print(f"wrote {args.out} (+ .pdf)")

if t_annual is not None:
    print("finetuned vs CESM2-LE truth:")
    for y, m, t in zip(truth_years, ft_annual[:len(truth_years)], t_annual):
        print(f"  {y}  ft {m:6.2f}  truth {t:6.2f}  diff {m-t:+.2f}")
