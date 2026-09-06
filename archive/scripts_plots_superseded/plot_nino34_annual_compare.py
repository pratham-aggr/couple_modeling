#!/usr/bin/env python
"""Niño-3.4 SST by YEAR: finetuned (sensreg) vs UNFINETUNED (base MSE-only,
no dQ/dSST fix) coupled ML-POP2 sims, vs two ground truths (CESM2-LE member
LE2-1231.002 and CAM6, the ERA5-scaled CREDIT run our atmosphere emulator
trains on). One point per completed calendar year (annual mean of the monthly
box-mean SST). Each sim plots over its own available years -- they are not
required to span the same range (the unfinetuned run is a shorter, separate
experiment).

Re-runnable: only full calendar years present in each run directory are used,
so re-running as either simulation advances extends its own line automatically.

Usage:  python plot_nino34_annual_compare.py [--run-ft DIR] [--run-uft DIR]
                                              [--out PNG] [--ref-member ID]
"""
import argparse, glob, re, os
from datetime import timedelta
import numpy as np, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

RUN_FT_DEFAULT      = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_sensreg50yr/run"
RUN_UFT_DEFAULT     = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw1yr/run"
RUN_UFT_LOOP_DEFAULT = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw_iceloop/run"
LE2_SST_DIR = "/glade/campaign/cgd/cesm/CESM2-LE/ocn/proc/tseries/month_1/SST"
CAM6_DIR    = "/glade/derecho/scratch/wchapman/b_credit_runs"

ap = argparse.ArgumentParser()
ap.add_argument("--run-ft", default=RUN_FT_DEFAULT, help="finetuned (sensreg) rundir")
ap.add_argument("--run-uft", default=RUN_UFT_DEFAULT, help="unfinetuned, climatology-ice rundir")
ap.add_argument("--run-uft-loop", default=RUN_UFT_LOOP_DEFAULT, help="unfinetuned, interactive-ice rundir")
ap.add_argument("--out", default="/glade/u/home/praggarwal/couple/output/nino34_annual_compare.png")
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
# Okabe-Ito CVD-safe palette, four hues, each with its OWN linestyle so
# identity survives black/white printing regardless of color perception.
C_FT, C_UFT, C_UFTLOOP, C_TRUTH, C_CAM6, MUTE = "#D55E00", "#009E73", "#CC79A7", "#5b6167", "#0072B2", "#6b7177"

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
        return None, None
    mkeys = [ym(f) for f in mfiles]
    J0, J1, I0, I1, sub_box, sub_w, sub_wsum = load_box_weights(mfiles[0])
    msst_by_ym = {k: bmean(xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values[J0:J1, I0:I1],
                            sub_box, sub_w, sub_wsum)
                  for f, k in zip(mfiles, mkeys)}
    years_avail = sorted({y for (y, m) in mkeys})
    full_years = [y for y in years_avail if all((y, m) in msst_by_ym for m in range(1, 13))]
    if not full_years:
        return None, None
    annual = np.array([np.mean([msst_by_ym[(y, m)] for m in range(1, 13)]) for y in full_years])
    return full_years, annual, (J0, J1, I0, I1, sub_box, sub_w, sub_wsum)

ft_years, ft_annual, ft_box = annual_series_from_rundir(args.run_ft)
uft_years, uft_annual, uft_box = annual_series_from_rundir(args.run_uft)
uftloop_years, uftloop_annual, uftloop_box = annual_series_from_rundir(args.run_uft_loop)
if ft_years is None and uft_years is None and uftloop_years is None:
    raise SystemExit("no complete calendar year in any rundir")

print(f"finetuned:            {len(ft_years) if ft_years else 0} complete years"
      + (f"  [{ft_years[0]}-{ft_years[-1]}]" if ft_years else ""))
print(f"unfinetuned:          {len(uft_years) if uft_years else 0} complete years"
      + (f"  [{uft_years[0]}-{uft_years[-1]}]" if uft_years else ""))
print(f"unfinetuned+ice-loop: {len(uftloop_years) if uftloop_years else 0} complete years"
      + (f"  [{uftloop_years[0]}-{uftloop_years[-1]}]" if uftloop_years else ""))

# ---------- ground truths: cover the UNION of years needed ----------
all_years = sorted(set(ft_years or []) | set(uft_years or []) | set(uftloop_years or []))
y0, y1 = all_years[0], all_years[-1]
want_ym = {(y, m) for y in all_years for m in range(1, 13)}

# use whichever box weights are available (same gx1v7 grid either way)
J0, J1, I0, I1, sub_box, sub_w, sub_wsum = ft_box or uft_box or uftloop_box

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
        if key in want_ym: tvals[key] = bmean(sub[i], sub_box, sub_w, sub_wsum)

def truth_annual_for(years):
    return np.array([np.mean([tvals[(y, m)] for m in range(1, 13)]) for y in years]) if years else None

t_annual_ft  = truth_annual_for(ft_years)
t_annual_uft = truth_annual_for(uft_years)

# ---------- CAM6 (training truth), same union of years ----------
cam_files = {y: os.path.join(CAM6_DIR, f"b.e21.CREDIT_climate_branch_1980_{y}_zmdata_ERA5scaled_zmdata_Qtot.zarr")
             for y in all_years}
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

def cam_annual_for(years):
    if not years or not all(y in cam_by_year for y in years):
        return None
    return np.array([cam_by_year[y] for y in years])

cam_annual_ft  = cam_annual_for(ft_years)
cam_annual_uft = cam_annual_for(uft_years)

# ---------- figure ----------
fig, ax = plt.subplots(figsize=(9.0, 4.6))
fig.subplots_adjust(top=0.86, right=0.97, left=0.09, bottom=0.11)

# ground truths (plotted once each, over the full union of years so both sims
# can be read against the same reference lines)
have_all_truth = all((y, m) in tvals for y in all_years for m in range(1, 13))
t_annual_all = truth_annual_for(all_years) if have_all_truth else None
if t_annual_all is not None:
    ax.plot(all_years, t_annual_all, color=C_TRUTH, lw=1.6, ls=(0, (5, 2)), dash_capstyle="round",
            marker="o", ms=4, mfc=C_TRUTH, mec="none", zorder=1,
            label=f"CESM2-LE truth  (mean {t_annual_all.mean():.2f} °C)")
cam_annual_all = cam_annual_for(all_years)
if cam_annual_all is not None:
    ax.plot(all_years, cam_annual_all, color=C_CAM6, lw=1.6, ls=(0, (1, 1.4)), dash_capstyle="round",
            marker="o", ms=4, mfc=C_CAM6, mec="none", zorder=1,
            label=f"CAM6 (training truth)  (mean {cam_annual_all.mean():.2f} °C)")

if uft_years:
    ax.plot(uft_years, uft_annual, color=C_UFT, lw=2.0, ls=(0, (3, 1, 1, 1)), dash_capstyle="round",
            marker="o", ms=5, mfc=C_UFT, mec="none", zorder=2,
            label=f"Unfinetuned sim  (mean {uft_annual.mean():.2f} °C)")
if uftloop_years:
    ax.plot(uftloop_years, uftloop_annual, color=C_UFTLOOP, lw=2.0, ls=(0, (4, 1)), dash_capstyle="round",
            marker="o", ms=5, mfc=C_UFTLOOP, mec="none", zorder=2,
            label=f"Unfinetuned, interactive-ice sim  (mean {uftloop_annual.mean():.2f} °C)")
if ft_years:
    ax.plot(ft_years, ft_annual, color=C_FT, lw=2.4, solid_capstyle="round",
            marker="o", ms=5.5, mfc=C_FT, mec="none", zorder=3,
            label=f"Finetuned (sensreg) sim  (mean {ft_annual.mean():.2f} °C)")

ax.set_ylabel("Niño-3.4 SST (°C)")
fig.text(0.09, 0.97, "Niño-3.4 SST by year: finetuned vs unfinetuned", ha="left", va="top",
         fontsize=14.5, fontweight="bold")
fig.text(0.09, 0.905, f"{y0}–{y1} annual mean", ha="left", va="top", color=MUTE, fontsize=10.5)
ax.legend(loc="lower left", frameon=False, handlelength=1.8, borderaxespad=0.3, labelcolor="linecolor")
ax.margins(x=0.03)
ax.xaxis.set_major_locator(mtick.MultipleLocator(1))
fig.savefig(args.out); fig.savefig(args.out.replace(".png", ".pdf"))
print(f"wrote {args.out} (+ .pdf)")

if ft_years and t_annual_ft is not None:
    print("finetuned vs truth:")
    for y, m, t in zip(ft_years, ft_annual, t_annual_ft):
        print(f"  {y}  ft {m:6.2f}  truth {t:6.2f}  diff {m-t:+.2f}")
if uft_years and t_annual_uft is not None:
    print("unfinetuned vs truth:")
    for y, m, t in zip(uft_years, uft_annual, t_annual_uft):
        print(f"  {y}  uft {m:6.2f}  truth {t:6.2f}  diff {m-t:+.2f}")
t_annual_uftloop = truth_annual_for(uftloop_years)
if uftloop_years and t_annual_uftloop is not None:
    print("unfinetuned+ice-loop vs truth:")
    for y, m, t in zip(uftloop_years, uftloop_annual, t_annual_uftloop):
        print(f"  {y}  uft-loop {m:6.2f}  truth {t:6.2f}  diff {m-t:+.2f}")
