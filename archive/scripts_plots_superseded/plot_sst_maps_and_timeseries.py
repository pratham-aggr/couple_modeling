#!/usr/bin/env python
"""Raw SST: (1) global-mean annual time series and (2) annual-mean spatial maps,
side by side for CAM6 (training truth), CESM2-LE (LE2-1231.002, truth),
finetuned coupled sim, and unfinetuned (interactive-ice) coupled sim.

Two output files:
  sst_maps_timeseries.png/.pdf     -- 4-panel time series (identical to plot_sst_annual.py numbers)
  sst_maps_snapshot_<YEAR>.png/.pdf -- 2x2 raw annual-mean SST maps for one common year

Usage: python plot_sst_maps_and_timeseries.py [--map-year YYYY]
"""
import argparse, glob, re, os
from datetime import timedelta
import numpy as np, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

RUN_FT      = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_sensreg50yr/run"
RUN_UFT     = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw_iceloop/run"
LE2_SST_DIR = "/glade/campaign/cgd/cesm/CESM2-LE/ocn/proc/tseries/month_1/SST"
CAM6_DIR    = "/glade/derecho/scratch/wchapman/b_credit_runs"
REF_MEMBER  = "LE2-1231.002"

ap = argparse.ArgumentParser()
ap.add_argument("--map-year", type=int, default=2014, help="year for the snapshot maps (must be <=2014 for CAM6)")
ap.add_argument("--outdir", default="/glade/u/home/praggarwal/couple/output")
args = ap.parse_args()

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 13,
    "axes.labelsize": 11.5, "legend.fontsize": 11, "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5, "axes.linewidth": 0, "axes.edgecolor": "#4d4d4d",
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#ececec",
    "grid.linewidth": 0.7, "xtick.color": "#6b7177", "ytick.color": "#6b7177",
    "text.color": "#1a1a1a", "axes.labelcolor": "#3d3d3d",
    "savefig.dpi": 300, "figure.dpi": 140,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "xtick.major.size": 0, "ytick.major.size": 0,
})
C_FT, C_UFT, C_TRUTH, C_CAM6, MUTE = "#D55E00", "#009E73", "#5b6167", "#0072B2", "#6b7177"

def ym(f): m = re.search(r"\.(\d{4})-(\d{2})\.nc$", f); return int(m.group(1)), int(m.group(2))

# ================= TIME SERIES =================
g = xr.open_dataset(sorted(glob.glob(f"{RUN_FT}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))[0])
TAREA = np.nan_to_num(g["TAREA"].values.astype(np.float64))
def gmean(sst2d):
    s = np.asarray(sst2d, np.float64); m = np.isfinite(s)
    return float(np.nansum(np.where(m, s, 0)*TAREA) / (TAREA*m).sum())

def annual_gmean_series(rundir):
    files = sorted(glob.glob(f"{rundir}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
    if not files: return None, None
    keys = [ym(f) for f in files]
    byym = {k: gmean(xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values) for f, k in zip(files, keys)}
    years = sorted({y for y, m in keys})
    full = [y for y in years if all((y, m) in byym for m in range(1, 13))]
    if not full: return None, None
    return full, np.array([np.mean([byym[(y, m)] for m in range(1, 13)]) for y in full])

ft_years, ft_annual = annual_gmean_series(RUN_FT)
uft_years, uft_annual = annual_gmean_series(RUN_UFT)
print(f"finetuned:   {len(ft_years)} yr [{ft_years[0]}-{ft_years[-1]}]")
print(f"unfinetuned: {len(uft_years)} yr [{uft_years[0]}-{uft_years[-1]}]")

all_years = sorted(set(ft_years) | set(uft_years))
y0, y1 = all_years[0], all_years[-1]

mfs = [f for f in sorted(glob.glob(f"{LE2_SST_DIR}/*{REF_MEMBER}*.pop.h.SST.*.nc"))
       if (lambda r: r and int(r.group(1)) <= y1 and int(r.group(2)) >= y0)(
           re.search(r"\.(\d{4})\d\d-(\d{4})\d\d\.nc$", f))]
tvals = {}
for f in mfs:
    ds = xr.open_dataset(f)["SST"]
    if "z_t" in ds.dims: ds = ds.isel(z_t=0)
    for i in range(ds.sizes["time"]):
        t = ds["time"].values[i] - timedelta(days=15)
        key = (t.year, t.month)
        if y0 <= key[0] <= y1: tvals[key] = gmean(ds.isel(time=i).values)
truth_years = [y for y in all_years if all((y, m) in tvals for m in range(1, 13))]
truth_annual = np.array([np.mean([tvals[(y, m)] for m in range(1, 13)]) for y in truth_years])

cam_files = {y: os.path.join(CAM6_DIR, f"b.e21.CREDIT_climate_branch_1980_{y}_zmdata_ERA5scaled_zmdata_Qtot.zarr") for y in all_years}
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
        cam_by_year[y] = float(np.nansum(sst*ocnw) / np.nansum(ocnw))
    cam_years = sorted(cam_by_year)
    cam_annual = np.array([cam_by_year[y] for y in cam_years])

fig, ax = plt.subplots(figsize=(9.2, 4.6))
fig.subplots_adjust(top=0.86, right=0.97, left=0.10, bottom=0.11)
ax.plot(truth_years, truth_annual, color=C_TRUTH, lw=1.8, ls=(0,(5,2)), dash_capstyle="round",
        marker="o", ms=4.5, mfc=C_TRUTH, mec="none", zorder=1,
        label=f"CESM2-LE truth (mean {truth_annual.mean():.2f} °C)")
if cam_annual is not None:
    ax.plot(cam_years, cam_annual, color=C_CAM6, lw=1.8, ls=(0,(1,1.4)), dash_capstyle="round",
            marker="o", ms=4.5, mfc=C_CAM6, mec="none", zorder=2,
            label=f"CAM6 training truth (mean {cam_annual.mean():.2f} °C)")
ax.plot(uft_years, uft_annual, color=C_UFT, lw=2.2, ls=(0,(3,1,1,1)), dash_capstyle="round",
        marker="o", ms=5, mfc=C_UFT, mec="none", zorder=2,
        label=f"Unfinetuned sim (mean {uft_annual.mean():.2f} °C)")
ax.plot(ft_years, ft_annual, color=C_FT, lw=2.6, solid_capstyle="round",
        marker="o", ms=5.5, mfc=C_FT, mec="none", zorder=3,
        label=f"Finetuned sim (mean {ft_annual.mean():.2f} °C)")
ax.set_ylabel("Global-mean SST (°C)")
fig.text(0.10, 0.97, "Raw global-mean SST by year", ha="left", va="top", fontsize=15, fontweight="bold")
fig.text(0.10, 0.905, f"{y0}–{y1} annual mean, area-weighted", ha="left", va="top", color=MUTE, fontsize=10.5)
ax.legend(loc="best", frameon=False, handlelength=1.6, borderaxespad=0.3, labelcolor="linecolor")
ax.margins(x=0.03)
ax.xaxis.set_major_locator(mtick.MultipleLocator(5))
ts_out = os.path.join(args.outdir, "sst_maps_timeseries.png")
fig.savefig(ts_out); fig.savefig(ts_out.replace(".png", ".pdf"))
print(f"wrote {ts_out} (+ .pdf)")

# ================= SNAPSHOT MAPS =================
Y = args.map_year
fig2, axes = plt.subplots(2, 2, figsize=(13, 8.4), subplot_kw={})
fig2.subplots_adjust(top=0.90, bottom=0.06, left=0.04, right=0.98, hspace=0.28, wspace=0.12)

vmin, vmax = -2, 32
cmap = "RdYlBu_r"

def annual_map_gx1v7(rundir, year):
    files = sorted(glob.glob(f"{rundir}/*.pop.h.{year}-[0-9][0-9].nc"))
    if len(files) != 12: return None
    arrs = [xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values for f in files]
    return np.nanmean(np.stack(arrs), axis=0)

def annual_map_le2(year):
    mfs = [f for f in sorted(glob.glob(f"{LE2_SST_DIR}/*{REF_MEMBER}*.pop.h.SST.*.nc"))
           if (lambda r: r and int(r.group(1)) <= year <= int(r.group(2)))(
               re.search(r"\.(\d{4})\d\d-(\d{4})\d\d\.nc$", f))]
    vals = []
    for f in mfs:
        ds = xr.open_dataset(f)["SST"]
        if "z_t" in ds.dims: ds = ds.isel(z_t=0)
        for i in range(ds.sizes["time"]):
            t = ds["time"].values[i] - timedelta(days=15)
            if t.year == year: vals.append(ds.isel(time=i).values)
    return np.nanmean(np.stack(vals), axis=0) if len(vals) == 12 else None

def annual_map_cam6(year):
    f = os.path.join(CAM6_DIR, f"b.e21.CREDIT_climate_branch_1980_{year}_zmdata_ERA5scaled_zmdata_Qtot.zarr")
    if not os.path.isdir(f): return None
    ds = xr.open_zarr(f, consolidated=False)
    sst = (ds["SST"].values - 273.15).mean(axis=0)
    lf = ds["LANDFRAC"].isel(time=0).values
    sst = np.where(lf > 0.5, np.nan, sst)
    return sst, ds.latitude.values, ds.longitude.values

TLON = g["TLONG"].values
TLAT = g["TLAT"].values

def pop_seam_mask(tlon, tlat):
    """Every gx1v7 row crosses the prime meridian somewhere (TLONG wraps
    ~359.9 -> ~0.1). pcolormesh's shading='auto' infers cell EDGES by
    linearly interpolating between consecutive CENTER values with no idea
    the coordinate is periodic, so at the wrap it computes an edge at
    (359.9+0.1)/2 ~= 180 -- one phantom cell stretched across nearly the
    whole plot width. This is invisible wherever that phantom cell lands on
    a NaN (land) point, and only shows up as a thin streak at the handful of
    latitude rows where it happens to land on open ocean (exactly what was
    flagged as a "gridding problem" at ~45N here). Fix: mask by the RAW
    (non-wrap-aware) jump in TLONG -- the wrap-aware geographic distance is
    irrelevant, since it's pcolormesh's naive interpolation that breaks, not
    the actual geography. Also catch the tripole j-fold via TLAT jumps."""
    bad = np.zeros(tlon.shape, dtype=bool)
    dlon_i = np.abs(np.diff(tlon, axis=1))             # NOT wrap-aware, on purpose
    jump_i = dlon_i > 180
    bad[:, :-1] |= jump_i; bad[:, 1:] |= jump_i
    dlat_j = np.abs(np.diff(tlat, axis=0))
    jump_j = dlat_j > 10
    bad[:-1, :] |= jump_j; bad[1:, :] |= jump_j
    return bad

SEAM = pop_seam_mask(TLON, TLAT)
print(f"gx1v7 seam mask: {SEAM.sum()} cells ({100*SEAM.mean():.2f}%) excluded from maps only "
      f"(area-weighted means above are unaffected)")

panels = []
m_ft = annual_map_gx1v7(RUN_FT, Y)
panels.append(("Finetuned sim", m_ft, "gx1v7"))
m_uft = annual_map_gx1v7(RUN_UFT, Y)
panels.append(("Unfinetuned sim", m_uft, "gx1v7"))
m_truth = annual_map_le2(Y)
panels.append(("CESM2-LE truth", m_truth, "gx1v7"))
cam_res = annual_map_cam6(Y)
if cam_res is not None:
    m_cam, cam_lat, cam_lon = cam_res
else:
    m_cam, cam_lat, cam_lon = None, None, None
panels.append(("CAM6 training truth", m_cam, "latlon"))

for ax, (title, field, kind) in zip(axes.flat, panels):
    if field is None:
        ax.text(0.5, 0.5, f"{title}\n(no data for {Y})", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        continue
    if kind == "gx1v7":
        field_plot = np.where((TAREA > 0) & ~SEAM, field, np.nan)
        pcm = ax.pcolormesh(TLON, TLAT, field_plot, vmin=vmin, vmax=vmax, cmap=cmap, shading="auto")
    else:
        pcm = ax.pcolormesh(cam_lon, cam_lat, field, vmin=vmin, vmax=vmax, cmap=cmap, shading="auto")
    ax.set_title(title, fontsize=12.5)
    ax.set_xticks([0, 90, 180, 270, 360]); ax.set_yticks([-90, -45, 0, 45, 90])
    ax.set_facecolor("#dddddd")

cbar = fig2.colorbar(pcm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.06, aspect=40)
cbar.set_label("SST (°C)")
fig2.text(0.04, 0.965, f"Raw annual-mean SST — {Y}", ha="left", va="top", fontsize=15.5, fontweight="bold")
map_out = os.path.join(args.outdir, f"sst_maps_snapshot_{Y}.png")
fig2.savefig(map_out); fig2.savefig(map_out.replace(".png", ".pdf"))
print(f"wrote {map_out} (+ .pdf)")
