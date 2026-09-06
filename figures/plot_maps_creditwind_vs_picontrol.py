"""Spatial SST quality maps: pz220f0b PRESCRIBED-REAL-WIND run (--credit_wind,
memo_pop_standalone_gx1v7_diagfw_creditwind1yr, perpetual GPU chain) vs CESM2
piControl "truth" vs error, for global + several regions.

Straight adaptation of plot_maps_pz220f0b_vs_picontrol.py (own-emulated-wind
production run vs piControl) for the credit_wind experiment -- same regridding
(shared kNN ScatterToRegular onto a common 0.5-deg lat/lon grid), same
elapsed-model-year-matched piControl window (this run's own calendar is
Feb-anchored, not Jan, because it cold-started 1980-02-01 from a POP-native
restart -- see run_pop_diagfw_creditwind_1yr.pbs's comments), same
auto-detected post-transient stabilization window, same 5-region layout.
RE-RUNNABLE BY DESIGN: uses whatever months are on disk right now.

Outputs (overwritten in place on every re-run):
  output/creditwind_experiment/maps_creditwind_vs_picontrol.pdf
  output/creditwind_experiment/maps_creditwind_vs_picontrol_stats.txt

Usage: python plot_maps_creditwind_vs_picontrol.py
       [--window 15] [--slope-thresh 0.02] [--persist 5]
"""
import argparse, glob, re
import numpy as np, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # editable text in the vector PDF
    "axes.titlesize": 14, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
})
import sys
sys.path.insert(0, "/glade/u/home/praggarwal/couple/camulator_ud/climate")
from model_server import ScatterToRegular
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--window", type=int, default=15, help="trailing-slope window, years")
ap.add_argument("--slope-thresh", type=float, default=0.02, help="K/yr; below this = 'flat'")
ap.add_argument("--persist", type=int, default=5, help="consecutive flat years required")
ap.add_argument("--min-model-year", type=int, default=None,
                 help="override the auto-detected stabilization year (elapsed model year, 1-based)")
ap.add_argument("--max-model-year", type=int, default=None,
                 help="cap the window's end (elapsed model year, 1-based); default: use all available data")
args = ap.parse_args()

PZ_DIR = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw_creditwind1yr/run"
PI_DIR = ("/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/"
          "b.e21.B1850.f09_g17.CMIP6-piControl.001/ocn/proc/tseries/month_1")
PI_GLOB = f"{PI_DIR}/*.pop.h.SST.*.nc"

DST_NLAT, DST_NLON = 180, 360
DST_LATS = np.linspace(-89.5, 89.5, DST_NLAT)
DST_LONS = np.linspace(0.5, 359.5, DST_NLON)

REGIONS = [
    ("Global",            (-90, 90),  (0, 360)),
    ("Tropical Pacific",  (-20, 20),  (120, 290)),
    ("North Atlantic",    (20, 70),   (260, 360)),
    ("Southern Ocean",    (-70, -40), (0, 360)),
    ("Arctic",            (60, 90),   (0, 360)),
]

def uxyz(lon, lat):
    lon = np.radians(lon); lat = np.radians(lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)

def ym(f):
    m = re.search(r"\.(\d{4})-(\d{2})\.nc$", f)
    return int(m.group(1)), int(m.group(2))

# =====================================================================
# 1) discover ALL credit_wind months currently on disk
# =====================================================================
mfiles = sorted(glob.glob(f"{PZ_DIR}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
if not mfiles:
    raise SystemExit(f"no credit_wind monthly files found in {PZ_DIR}")
mkeys = [ym(f) for f in mfiles]
pz_year0 = mkeys[0][0]
years_avail = sorted({y for (y, m) in mkeys})
full_years = np.array([y for y in years_avail if sum(1 for (yy, mm) in mkeys if yy == y) == 12])
print(f"credit_wind: {len(mfiles)} months on disk [{mkeys[0][0]}-{mkeys[0][1]:02d} to "
      f"{mkeys[-1][0]}-{mkeys[-1][1]:02d}], {len(full_years)} complete calendar years "
      f"[{full_years[0]}-{full_years[-1]}]  (run itself is Feb-anchored: elapsed model yr 1 = "
      f"1980-02 to 1981-01, so calendar year {pz_year0} is a partial first year -- expected)")

# =====================================================================
# 2) grid + shared regridder (checked fresh every run)
# =====================================================================
g0 = xr.open_dataset(mfiles[0])
tlon = g0["TLONG"].values.astype("f8"); tlat = g0["TLAT"].values.astype("f8")
kmt = g0["KMT"].values

pi_first = sorted(glob.glob(PI_GLOB))
if not pi_first:
    raise SystemExit(f"no piControl SST files found under {PI_DIR}")
ds_pi_check = xr.open_dataset(pi_first[0])
assert np.allclose(tlon, ds_pi_check["TLONG"].values.astype("f8"))
assert np.allclose(tlat, ds_pi_check["TLAT"].values.astype("f8"))
assert np.array_equal(kmt, ds_pi_check["KMT"].values)
print("GRID CHECK: credit_wind and piControl TLAT/TLONG/KMT are identical -- safe to regrid once, reuse.")

ocean = (kmt > 0).astype(np.int32)
s2r = ScatterToRegular(tlon, tlat, ocean, DST_LATS, DST_LONS, k=4)
tree = cKDTree(uxyz(tlon.ravel(), tlat.ravel()))
dl, dn = np.meshgrid(DST_LATS, DST_LONS, indexing="ij")
_, ni = tree.query(uxyz(dn.ravel(), dl.ravel()), k=1)
dst_ocean = (ocean.ravel()[ni] > 0).reshape(DST_NLAT, DST_NLON)

# =====================================================================
# 3) auto-detect the post-transient (stabilization) year from credit_wind's
#    own annual global-mean SST trajectory -- same method as the production
#    2-panel/maps figures, so all figures agree about the spin-up window.
# =====================================================================
msst_by_ym = {}
for f, (y, m) in zip(mfiles, mkeys):
    if y in full_years:
        msst_by_ym[(y, m)] = xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values.astype("f8")

def gmean_native(field2d, w):
    s = np.asarray(field2d, np.float64); m = np.isfinite(s)
    return float(np.nansum(np.where(m, s, 0) * w) / (w * m).sum())

TAREA = np.nan_to_num(g0["TAREA"].values.astype(np.float64))
pz_ann_gmsst = np.array([np.mean([gmean_native(msst_by_ym[(y, m)], TAREA) for m in range(1, 13)])
                          for y in full_years])

def detect_stabilization_year(yrs, series, window, thresh, persist):
    slopes = np.full(len(yrs), np.nan)
    for i in range(window, len(yrs)):
        x = yrs[i - window:i + 1].astype(np.float64)
        yv = series[i - window:i + 1]
        A = np.vstack([x - x.mean(), np.ones_like(x)]).T
        slope, _ = np.linalg.lstsq(A, yv, rcond=None)[0]
        slopes[i] = slope
    flat = np.abs(slopes) < thresh
    for i in range(window, len(yrs) - persist + 1):
        if flat[i:i + persist].all():
            return int(yrs[i])
    return int(yrs[min(window, len(yrs) - 1)])  # fallback: not enough data to confirm flatness yet

if args.min_model_year is not None:
    YEAR_MIN = pz_year0 + args.min_model_year - 1
    print(f"--min-model-year {args.min_model_year}: overriding auto-detected stabilization year")
elif len(full_years) > args.window + args.persist:
    YEAR_MIN = detect_stabilization_year(full_years, pz_ann_gmsst, args.window, args.slope_thresh, args.persist)
else:
    YEAR_MIN = full_years[0]
    print(f"WARNING: only {len(full_years)} complete years available -- too few to confirm "
          f"stabilization; using ALL of them (spin-up may still be included).")

if args.max_model_year is not None:
    YEAR_MAX = pz_year0 + args.max_model_year - 1
    if YEAR_MAX > int(full_years[-1]):
        raise SystemExit(f"--max-model-year {args.max_model_year} (calendar year {YEAR_MAX}) "
                          f"exceeds available data (latest complete year {full_years[-1]})")
    print(f"--max-model-year {args.max_model_year}: capping window end")
else:
    YEAR_MAX = int(full_years[-1])
print(f"post-transient window: credit_wind years {YEAR_MIN}-{YEAR_MAX} "
      f"(elapsed model yr {YEAR_MIN - pz_year0 + 1}-{YEAR_MAX - pz_year0 + 1})")

mfiles_win = [f for f, (y, m) in zip(mfiles, mkeys) if YEAR_MIN <= y <= YEAR_MAX]
n_needed = len(mfiles_win)

# =====================================================================
# 4) piControl: pull an ELAPSED-MODEL-YEAR-matched window (not the same
#    absolute months starting from piControl's own year 1)
# =====================================================================
elapsed_lo = YEAR_MIN - pz_year0            # 0-based months to skip
elapsed_hi_months = YEAR_MAX - pz_year0 + 1  # 0-based months to end at (exclusive)
pi_month_lo = elapsed_lo * 12
pi_month_hi = elapsed_hi_months * 12

pi_blocks = sorted(glob.glob(PI_GLOB))
sst_chunks, months_have = [], 0
PI_TAREA = PI_TLAT = PI_TLONG = None
for f in pi_blocks:
    if months_have >= pi_month_hi:
        break
    dsb = xr.open_dataset(f)
    if PI_TAREA is None:
        PI_TAREA = np.nan_to_num(dsb["TAREA"].values.astype(np.float64))
    chunk = dsb["SST"].isel(z_t=0).values.astype("f8")
    sst_chunks.append(chunk)
    months_have += chunk.shape[0]
sst_all = np.concatenate(sst_chunks, axis=0)
if months_have < pi_month_hi:
    raise SystemExit(f"piControl only has {months_have} months available; need "
                      f"{pi_month_hi} to match credit_wind's elapsed window "
                      f"(years {YEAR_MIN}-{YEAR_MAX}, elapsed {elapsed_lo+1}-{elapsed_hi_months}). "
                      f"Add more piControl blocks or shorten the window.")
pi_window = sst_all[pi_month_lo:pi_month_hi]
print(f"piControl: elapsed-matched window, months {pi_month_lo}-{pi_month_hi} "
      f"({pi_window.shape[0]} months = piControl's own elapsed yr "
      f"{elapsed_lo+1}-{elapsed_hi_months})")
assert pi_window.shape[0] == n_needed, (pi_window.shape[0], n_needed)

# ---------- credit_wind time-mean SST (native grid, then regrid once) ----------
sst_sum = np.zeros_like(tlon)
for f in mfiles_win:
    sst_sum += xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values.astype("f8")
pz_mean_native = sst_sum / n_needed

# ---------- piControl time-mean SST (elapsed-matched window) ----------
pi_mean_native = np.nanmean(pi_window, axis=0)

pz_reg = np.ma.masked_where(~dst_ocean, s2r(np.nan_to_num(np.where(kmt > 0, pz_mean_native, np.nan), nan=0.0)))
pi_reg = np.ma.masked_where(~dst_ocean, s2r(np.nan_to_num(np.where(kmt > 0, pi_mean_native, np.nan), nan=0.0)))
err_reg = pz_reg - pi_reg

# =====================================================================
# 5) figure: rows = regions, cols = [truth (piControl), predicted (credit_wind), error]
# =====================================================================
fig, axes = plt.subplots(len(REGIONS), 3, figsize=(15, 3.3 * len(REGIONS)))
cmap_sst = plt.get_cmap("RdYlBu_r").copy(); cmap_sst.set_bad("dimgray")
cmap_err = plt.get_cmap("RdBu_r").copy(); cmap_err.set_bad("dimgray")

stats_lines = []
for r, (name, (lat0, lat1), (lon0, lon1)) in enumerate(REGIONS):
    j0 = np.searchsorted(DST_LATS, lat0); j1 = np.searchsorted(DST_LATS, lat1)
    if lon0 < lon1:
        i0 = np.searchsorted(DST_LONS, lon0); i1 = np.searchsorted(DST_LONS, lon1)
        sl = (slice(j0, j1), slice(i0, i1))
        extent = [lon0, lon1, lat0, lat1]
    else:
        sl = (slice(j0, j1), slice(None))
        extent = [0, 360, lat0, lat1]

    err_field = err_reg[sl]
    err_valid = err_field.compressed() if np.ma.isMaskedArray(err_field) else err_field[np.isfinite(err_field)]
    err_p99 = float(np.nanpercentile(np.abs(err_valid), 99)) if err_valid.size else 3.0
    err_vmax = max(err_p99, 3.0)
    bias = float(np.nanmean(err_valid)); rmse = float(np.sqrt(np.nanmean(err_valid ** 2)))
    line = (f"{name:18s} bias={bias:+6.3f}  RMSE={rmse:5.3f}  "
            f"min={np.nanmin(err_valid):+7.2f}  max={np.nanmax(err_valid):+7.2f}  "
            f"p99|err|={err_p99:5.2f}")
    print(line); stats_lines.append(line)

    COL_TITLES = ["CESM2 piControl (truth)", "credit_wind (predicted)", "Error (credit_wind − truth)"]
    C2K = 273.15
    for c, (field, cmap, vmin, vmax, cbl) in enumerate([
        (pi_reg[sl] + C2K, cmap_sst, -2 + C2K, 32 + C2K, "K"),
        (pz_reg[sl] + C2K, cmap_sst, -2 + C2K, 32 + C2K, "K"),
        (err_reg[sl], cmap_err, -err_vmax, err_vmax, "K"),
    ]):
        ax = axes[r, c]
        im = ax.imshow(field, origin="lower", extent=extent, vmin=vmin, vmax=vmax,
                        cmap=cmap, aspect="auto")
        if r == 0:
            ax.set_title(COL_TITLES[c], fontsize=15, fontweight="bold")
        if c == 0:
            ax.set_ylabel(name, fontsize=14, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=(cbl if c == 2 else None))

fig.suptitle("pz220f0b (--credit_wind, prescribed real wind) vs. CESM2 piControl Sea Surface Temperature "
             f"{YEAR_MIN-pz_year0+1}-{YEAR_MAX-pz_year0+1}yr",
             fontsize=18, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out_pdf = "output/creditwind_experiment/maps_creditwind_vs_picontrol.pdf"
fig.savefig(out_pdf)
print(f"wrote {out_pdf}")

rmse_global = float(np.sqrt(np.nanmean(np.where(dst_ocean, err_reg, np.nan) ** 2)))
bias_global = float(np.nanmean(np.where(dst_ocean, err_reg, np.nan)))
print(f"global (grid-cell, unweighted-pixel) bias={bias_global:+.3f} K  RMSE={rmse_global:.3f} K")

out_txt = "output/creditwind_experiment/maps_creditwind_vs_picontrol_stats.txt"
with open(out_txt, "w") as fh:
    fh.write(f"credit_wind vs CESM2 piControl SST -- regional bias/RMSE\n")
    fh.write(f"credit_wind months on disk : {len(mfiles)}\n")
    fh.write(f"post-transient window      : years {YEAR_MIN}-{YEAR_MAX} "
             f"(elapsed model yr {YEAR_MIN-pz_year0+1}-{YEAR_MAX-pz_year0+1})\n")
    fh.write(f"piControl window (elapsed-matched): piControl elapsed yr "
             f"{elapsed_lo+1}-{elapsed_hi_months}\n\n")
    for line in stats_lines:
        fh.write(line + "\n")
    fh.write(f"\n{'Global (grid-cell)':18s} bias={bias_global:+6.3f}  RMSE={rmse_global:.3f}\n")
print(f"wrote {out_txt}")
