#!/usr/bin/env python
"""FASST vs CESM2 piControl -- compact 2-panel NeurIPS/CCAI workshop figure.

Panel 1: global-mean SST (K), continuous 1980-latest, no panel break.
Panel 2: Nino-3.4 anomaly (K), same continuous x-axis.

Spin-up transient is marked (not hard-cut): a trailing-window rolling-slope
detector finds the first year after which FASST's global-mean SST stays flat
(|slope| below --slope-thresh for --persist consecutive years), draws a
light shaded span before it and a dashed vline at it, and restricts the
reported mean/std statistics to the post-transient window only.

Vector PDF output (matplotlib, not plotly) for direct \\includegraphics use.

Usage: python make_fasst_neurips_2panel.py --out FILE.pdf
       [--window 15] [--slope-thresh 0.02] [--persist 5]
"""
import argparse, glob, re
import numpy as np, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

FASST_RUN = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw_iceloop/run"
PI_FILE = ("/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/"
           "b.e21.B1850.f09_g17.CMIP6-piControl.001/ocn/proc/tseries/month_1/"
           "b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.SST.000101-009912.nc")
PICTL_CACHE = "output/picontrol_gmsst_annual.npz"
C2K = 273.15

ap = argparse.ArgumentParser()
ap.add_argument("--window", type=int, default=15, help="trailing-slope window, years")
ap.add_argument("--slope-thresh", type=float, default=0.02, help="K/yr; below this = 'flat'")
ap.add_argument("--persist", type=int, default=5, help="consecutive flat years required")
ap.add_argument("--out", required=True)
ap.add_argument("--max-model-year", type=int, default=None,
                 help="truncate to this many elapsed model years from the run start "
                      "(e.g. 140); default: use all available data")
args = ap.parse_args()

# ------------------------------------------------------------- styling ----
# reverted to the known-good working version (pre print-scale pass)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.titlesize": 9.5, "axes.labelsize": 9.5,
    "legend.fontsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#ececec", "grid.linewidth": 0.6,
    "xtick.color": "#6b7177", "ytick.color": "#6b7177",
    "text.color": "#1a1a1a", "axes.labelcolor": "#3d3d3d",
    "pdf.fonttype": 42, "ps.fonttype": 42,   # editable text in the vector PDF
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "xtick.major.size": 0, "ytick.major.size": 0,
})
C_SIM, C_TRUTH, MUTE = "#D2691E", "#6b6b6b", "#6b7177"
LW_TRUTH, LW_SIM = 1.3, 2.0
ANNOT_FS = 8.0
PANEL_LABEL_FS = 11.0

def ym(f):
    m = re.search(r"\.(\d{4})-(\d{2})\.nc$", f)
    return int(m.group(1)), int(m.group(2))
fyr = lambda yy, mm: yy + (mm - 0.5) / 12.0

def gmean(sst2d, w):
    s = np.asarray(sst2d, np.float64); m = np.isfinite(s)
    return float(np.nansum(np.where(m, s, 0) * w) / (w * m).sum())

def deseason(series, months):
    a = series.astype(np.float64).copy()
    for k in range(1, 13):
        sel = months == k
        if sel.any():
            a[sel] = series[sel] - series[sel].mean()
    return a

# =====================================================================
# LOAD: global-mean SST, annual, Kelvin
# =====================================================================
mfiles = sorted(glob.glob(f"{FASST_RUN}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
mkeys = [ym(f) for f in mfiles]
if args.max_model_year is not None:
    _y0 = mkeys[0][0]
    _cutoff = _y0 + args.max_model_year - 1   # last calendar year kept (model yr 1 = _y0)
    _keep = [(f, k) for f, k in zip(mfiles, mkeys) if k[0] <= _cutoff]
    mfiles = [f for f, k in _keep]; mkeys = [k for f, k in _keep]
    print(f"--max-model-year {args.max_model_year}: truncated to calendar years "
          f"{_y0}-{_cutoff} ({len(mfiles)} months)")
y0, y1 = mkeys[0][0], mkeys[-1][0]
g = xr.open_dataset(mfiles[0])
TAREA = np.nan_to_num(g["TAREA"].values.astype(np.float64))
TLAT, TLONG = g.TLAT.values, g.TLONG.values

msst_by_ym = {k: gmean(xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values, TAREA)
              for f, k in zip(mfiles, mkeys)}
years_avail = sorted({y for (y, m) in mkeys})
full_years = [y for y in years_avail if all((y, m) in msst_by_ym for m in range(1, 13))]
fasst_sst_annual = np.array([np.mean([msst_by_ym[(y, m)] for m in range(1, 13)])
                              for y in full_years]) + C2K
full_years = np.array(full_years)

z = np.load(PICTL_CACHE)
pic_years_raw, pic_annual_C = z["years"], z["annual"]
n_pic = min(len(pic_years_raw), len(full_years))
pic_sst_annual = pic_annual_C[:n_pic] + C2K
years = full_years[:n_pic]
fasst_sst_annual = fasst_sst_annual[:n_pic]

print(f"FASST SST: {len(years)} complete years [{years[0]}-{years[-1]}]")

# =====================================================================
# LOAD: Nino-3.4 anomaly, monthly, K
# =====================================================================
box = (TLAT >= -5) & (TLAT <= 5) & (TLONG >= 190) & (TLONG <= 240)
jj, ii = np.where(box); J0, J1, I0, I1 = jj.min(), jj.max()+1, ii.min(), ii.max()+1
sub_box = box[J0:J1, I0:I1]; sub_w = (TAREA * box)[J0:J1, I0:I1]; sub_wsum = sub_w.sum()
def bmean(sub):
    s = np.asarray(sub, np.float64)
    return float(np.nansum(np.where(np.isfinite(s) & sub_box, s, 0) * sub_w) / sub_wsum)

mt = np.array([fyr(*k) for k in mkeys]); mmon = np.array([k[1] for k in mkeys])
msst34 = np.array([bmean(xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values[J0:J1, I0:I1])
                    for f in mfiles])
fasst_n34 = deseason(msst34, mmon)

# The FASST run keeps growing (background chain); load as many consecutive
# ~100-yr piControl SST blocks as needed to cover it, rather than hardcoding
# a single block -- that mismatch (piControl monthly file capped at 99 yr vs
# the annual piControl cache going further) was exactly what made panel (a)
# and panel (b) show different x-extents ("misaligned") in the previous version.
PI_DIR = ("/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/"
          "b.e21.B1850.f09_g17.CMIP6-piControl.001/ocn/proc/tseries/month_1")
pi_blocks = sorted(glob.glob(f"{PI_DIR}/*.pop.h.SST.*.nc"))
n_target = len(mkeys)
sst_chunks, months_have = [], 0
for f in pi_blocks:
    if months_have >= n_target:
        break
    dsb = xr.open_dataset(f)
    if "PI_TAREA" not in dir():
        PI_TAREA = np.nan_to_num(dsb["TAREA"].values.astype(np.float64))
        PI_TLAT, PI_TLONG = dsb["TLAT"].values, dsb["TLONG"].values
        pi_box = (PI_TLAT >= -5) & (PI_TLAT <= 5) & (PI_TLONG >= 190) & (PI_TLONG <= 240)
        pjj, pii = np.where(pi_box); PJ0, PJ1, PI0, PI1 = pjj.min(), pjj.max()+1, pii.min(), pii.max()+1
        pi_sub_box = pi_box[PJ0:PJ1, PI0:PI1]; pi_sub_w = (PI_TAREA * pi_box)[PJ0:PJ1, PI0:PI1]
        pi_sub_wsum = pi_sub_w.sum()
    chunk = dsb["SST"].isel(z_t=0).values[:, PJ0:PJ1, PI0:PI1]
    sst_chunks.append(chunk)
    months_have += chunk.shape[0]
n_pic_avail = months_have
n_needed = min(len(mkeys), n_pic_avail)
if n_needed < len(mkeys):
    print(f"NOTE: FASST run ({len(mkeys)} months) now exceeds ALL available piControl "
          f"monthly SST blocks combined ({n_pic_avail} months) -- truncating the "
          f"Nino3.4 panel (and its FASST series) to the {n_needed}-month overlap.")
def pi_bmean(sub):
    s = np.asarray(sub, np.float64)
    return float(np.nansum(np.where(np.isfinite(s) & pi_sub_box, s, 0) * pi_sub_w) / pi_sub_wsum)

sst_all = np.concatenate(sst_chunks, axis=0)[:n_needed]
pi_n34_raw = np.array([pi_bmean(sst_all[i]) for i in range(n_needed)])
pi_months = np.array([(1 + i % 12) for i in range(n_needed)])
pic_n34 = deseason(pi_n34_raw, pi_months)
pic_mt = y0 + np.arange(n_needed) / 12.0 + (0.5 / 12.0)

# truncate FASST's own Nino3.4 series to the same overlap so both panel-2 series
# end together (matches how panel 1's SST-annual series are already truncated)
mt = mt[:n_needed]; mmon = mmon[:n_needed]; fasst_n34 = fasst_n34[:n_needed]

print(f"FASST Nino3.4: {y0}-{y1} ({len(mkeys)} months)")

# =====================================================================
# STABILIZATION DETECTION: trailing-window rolling slope of FASST annual SST
# =====================================================================
def detect_stabilization_year(yrs, series, window, thresh, persist):
    """First year y such that the trailing `window`-yr slope at y, and at the
    next `persist`-1 years, all stay below `thresh` in magnitude (K/yr)."""
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
            return int(yrs[i]), slopes
    return int(yrs[window]), slopes  # fallback: earliest slope estimate available

stab_year, slopes = detect_stabilization_year(years, fasst_sst_annual,
                                               args.window, args.slope_thresh, args.persist)
print(f"detected stabilization year: {stab_year}  "
      f"(trailing {args.window}-yr slope < {args.slope_thresh} K/yr, "
      f"sustained {args.persist} yr)")

print(f"\n=== trailing {args.window}-yr rolling slope of FASST global-mean SST, 1995-2015 ===")
print(f"{'year':>6s}{'slope (K/yr)':>16s}   flat(<thresh)?")
for yr, sl in zip(years, slopes):
    if 1995 <= yr <= 2015:
        flag = "yes" if (np.isfinite(sl) and abs(sl) < args.slope_thresh) else ("n/a" if not np.isfinite(sl) else "no")
        print(f"{yr:6d}{sl:16.4f}   {flag}")

# ---------- post-transient stats (both panels, both series) ----------
post_y = years >= stab_year
post_m = mt >= stab_year
post_m_pic = pic_mt >= stab_year

fasst_sst_mean, fasst_sst_std = fasst_sst_annual[post_y].mean(), fasst_sst_annual[post_y].std()
pic_sst_mean, pic_sst_std = pic_sst_annual[post_y].mean(), pic_sst_annual[post_y].std()
fasst_n34_mean, fasst_n34_std = fasst_n34[post_m].mean(), fasst_n34[post_m].std()
pic_n34_mean, pic_n34_std = pic_n34[post_m_pic].mean(), pic_n34[post_m_pic].std()

print(f"\n=== post-transient ({stab_year}-{years[-1]}) stats, LaTeX-table order ===")
print(f"{'':20s}{'FASST':>14s}{'piControl':>14s}")
print(f"{'Mean SST (K)':20s}{fasst_sst_mean:14.2f}{pic_sst_mean:14.2f}")
print(f"{'Std SST (K)':20s}{fasst_sst_std:14.2f}{pic_sst_std:14.2f}")
print(f"{'Mean Nino-3.4 (K)':20s}{fasst_n34_mean:14.2f}{pic_n34_mean:14.2f}")
print(f"{'Std Nino-3.4 (K)':20s}{fasst_n34_std:14.2f}{pic_n34_std:14.2f}")

# ---------- ENSO variability flag (direction-safe) ----------
pct_diff = (fasst_n34_std - pic_n34_std) / pic_n34_std * 100.0
if abs(pct_diff) > 30.0:
    direction = "OVER-producing" if pct_diff > 0 else "UNDER-producing"
    print(f"\n*** ENSO VARIABILITY FLAG: FASST Nino-3.4 std ({fasst_n34_std:.2f} K) is "
          f"{pct_diff:+.1f}% vs piControl ({pic_n34_std:.2f} K) -- "
          f"FASST is {direction} ENSO variability relative to piControl. ***")
else:
    print(f"\nENSO variability check: FASST Nino-3.4 std is {pct_diff:+.1f}% vs piControl "
          f"-- within the +/-30% band, not flagged.")

# ---------- distribution shape check (post-transient window) ----------
from scipy import stats as _stats
fasst_n34_post = fasst_n34[post_m]
pic_n34_post = pic_n34[post_m_pic]
fasst_skew = float(_stats.skew(fasst_n34_post))
pic_skew = float(_stats.skew(pic_n34_post))
print(f"\n=== Nino-3.4 distribution shape, post-transient ({stab_year}-{years[-1]}) ===")
print(f"{'':20s}{'FASST':>14s}{'piControl':>14s}")
print(f"{'Skewness':20s}{fasst_skew:14.3f}{pic_skew:14.3f}")
skew_diff = fasst_skew - pic_skew
if abs(skew_diff) > 0.3:
    if pic_skew > 0 and fasst_skew < pic_skew:
        note = ("FASST's distribution is LESS positively skewed than piControl -- "
                 "consistent with FASST underrepresenting the sharp warm-phase (El "
                 "Nino) excursions relative to the cooler/flatter La Nina side.")
    elif fasst_skew > pic_skew:
        note = "FASST's distribution is MORE positively skewed (warm-side-heavy) than piControl."
    else:
        note = "FASST and piControl skew differ by more than 0.3 but not in a simple warm/cold sense -- inspect the histogram directly."
    print(f"*** SKEW FLAG: |skew difference| = {abs(skew_diff):.3f} > 0.3 -- {note} ***")
else:
    print(f"Skew difference ({skew_diff:+.3f}) is small -- no strong asymmetry claim supported by this alone.")

# =====================================================================
# FIGURE: 2 stacked panels, continuous x-axis, no marginal histogram (removed
# per request) -- shared legend removed earlier; series named via in-panel text.
# =====================================================================
# x-axis in MODEL YEAR (elapsed year since sim start, model yr 1 = years[0])
# rather than calendar year -- piControl has no real calendar date anyway, so
# calendar-year labeling was always somewhat fictitious for that series.
MODEL_YEAR0 = years[0]
years_x = years - MODEL_YEAR0 + 1
mt_x = mt - MODEL_YEAR0 + 1
pic_mt_x = pic_mt - MODEL_YEAR0 + 1
stab_year_x = stab_year - MODEL_YEAR0 + 1

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 5.0), sharex=True)
fig.subplots_adjust(top=0.88, right=0.97, left=0.13, bottom=0.14, hspace=0.32)
fig.suptitle("FASST vs. CESM2 piControl", fontsize=13, fontweight="bold", y=0.985)

for ax in (ax1, ax2):
    ax.axvspan(years_x[0], stab_year_x, color=MUTE, alpha=0.08, zorder=0)
    ax.axvline(stab_year_x, color=MUTE, lw=1.0, ls=(0, (2, 1.5)), zorder=1)

# --- panel 1: global-mean SST ---
ax1.plot(years_x, pic_sst_annual, color=C_TRUTH, lw=LW_TRUTH, ls=(0, (5, 2)),
         dash_capstyle="round", marker="o", ms=2.6, mfc=C_TRUTH, mec="none",
         zorder=2, label="CESM2 piControl")
ax1.plot(years_x, fasst_sst_annual, color=C_SIM, lw=LW_SIM, solid_capstyle="round",
         marker="o", ms=3.0, mfc=C_SIM, mec="none", zorder=3, label="FASST")
ax1.set_ylabel("Global-mean SST (K)")
ax1.set_title("(a)", loc="left", fontweight="bold", fontsize=PANEL_LABEL_FS)

# --- panel 2: Nino-3.4 anomaly ---
ax2.axhline(0, color="#c7c7c7", lw=0.8, zorder=0)
ax2.plot(pic_mt_x, pic_n34, color=C_TRUTH, lw=LW_TRUTH, ls=(0, (5, 2)),
         dash_capstyle="round", zorder=2, label="CESM2 piControl")
ax2.plot(mt_x, fasst_n34, color=C_SIM, lw=LW_SIM, solid_capstyle="round",
         zorder=3, label="FASST")
ax2.set_ylabel("Niño-3.4 anomaly (K)")
ax2.set_title("(b)", loc="left", fontweight="bold", fontsize=PANEL_LABEL_FS)
ax2.set_xlabel("Model year")
ax2.xaxis.set_major_locator(mtick.MultipleLocator(10))

# ---------- annotation placement: check corners for data overlap first ----------
def pick_corner(ax, x_all, y_series_list, xlim, ylim):
    """Return (ha, va, x_frac, y_frac) for the corner with the most whitespace,
    checked against the actual plotted data in that corner's x/y strip."""
    x0, x1 = xlim; y0_, y1_ = ylim
    candidates = {
        "upper right": (0.97, 0.95, "right", "top"),
        "upper left":  (0.03, 0.95, "left",  "top"),
        "lower right": (0.97, 0.05, "right", "bottom"),
        "lower left":  (0.03, 0.05, "left",  "bottom"),
    }
    best, best_score = None, -1
    for name, (xf, yf, ha, va) in candidates.items():
        # define an x-window: last/first ~20% of the axis depending on corner side
        if "right" in name:
            xmask = x_all >= (x1 - 0.22 * (x1 - x0))
        else:
            xmask = x_all <= (x0 + 0.22 * (x1 - x0))
        if not xmask.any():
            continue
        local_vals = np.concatenate([np.asarray(s)[xmask[:len(s)]] if len(s) == len(xmask)
                                      else np.asarray(s) for s in y_series_list])
        local_vals = local_vals[np.isfinite(local_vals)]
        if local_vals.size == 0:
            continue
        # score = distance from this corner's y-band to the nearest data value
        # (bigger distance = more whitespace = better)
        if "upper" in name:
            score = (y1_ - local_vals.max()) / (y1_ - y0_)
        else:
            score = (local_vals.min() - y0_) / (y1_ - y0_)
        if score > best_score:
            best_score, best = score, (ha, va, xf, yf)
    return best or ("right", "top", 0.97, 0.95)

ha1, va1, xf1, yf1 = pick_corner(ax1, years, [pic_sst_annual, fasst_sst_annual],
                                   (years[0], years[-1]), ax1.get_ylim())
ax1.text(xf1, yf1, f"FASST  {fasst_sst_mean:.2f} ± {fasst_sst_std:.2f} K",
         transform=ax1.transAxes, color=C_SIM, fontsize=ANNOT_FS, ha=ha1, va=va1)
dy = -0.07 if va1 == "top" else 0.07
ax1.text(xf1, yf1 + dy, f"CESM2 piControl  {pic_sst_mean:.2f} ± {pic_sst_std:.2f} K",
         transform=ax1.transAxes, color=C_TRUTH, fontsize=ANNOT_FS, ha=ha1, va=va1)

ha2, va2, xf2, yf2 = pick_corner(ax2, mt, [pic_n34, fasst_n34],
                                   (mt[0], mt[-1]), ax2.get_ylim())
ax2.text(xf2, yf2, f"FASST  {fasst_n34_mean:+.2f} ± {fasst_n34_std:.2f} K",
         transform=ax2.transAxes, color=C_SIM, fontsize=ANNOT_FS, ha=ha2, va=va2)
dy2 = -0.09 if va2 == "top" else 0.09
ax2.text(xf2, yf2 + dy2, f"CESM2 piControl  {pic_n34_mean:+.2f} ± {pic_n34_std:.2f} K",
         transform=ax2.transAxes, color=C_TRUTH, fontsize=ANNOT_FS, ha=ha2, va=va2)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 0.0), handlelength=1.6, labelcolor="linecolor")

fig.savefig(args.out)          # vector PDF
if args.out.endswith(".pdf"):
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=200)  # quick raster preview only
print(f"\nwrote {args.out}")
