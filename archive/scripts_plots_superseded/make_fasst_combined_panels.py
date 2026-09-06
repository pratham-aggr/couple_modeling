#!/usr/bin/env python
"""FASST vs CESM2 piControl: combined 2x2 publication figure.
Row 1: global-mean SST (K).  Row 2: Nino-3.4 anomaly (K).
Columns: (a) 1980-1999 [transient], (b) 2000-latest [quasi-equilibrium].
Independent y-axis scaling per panel (each era has very different spread).
Mean+/-std reported only in the later (quasi-equilibrium) column, as a
minimal colored annotation -- not a meaningful summary for the transient
column. ONE shared legend for the whole figure (bottom), consistent
color/line-style across all 4 panels.

Usage: python make_fasst_combined_panels.py --out FILE.png [--split-year 2000]
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
ap.add_argument("--split-year", type=int, default=2000)
ap.add_argument("--out", required=True)
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
C_SIM, C_TRUTH, MUTE = "#D55E00", "#5b6167", "#6b7177"
LW_TRUTH, LW_SIM = 1.6, 2.4

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

mfiles = sorted(glob.glob(f"{FASST_RUN}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
mkeys = [ym(f) for f in mfiles]
y0, y1 = mkeys[0][0], mkeys[-1][0]
g = xr.open_dataset(mfiles[0])
TAREA = np.nan_to_num(g["TAREA"].values.astype(np.float64))
TLAT, TLONG = g.TLAT.values, g.TLONG.values

# =====================================================================
# ROW 1: global-mean SST, annual, Kelvin
# =====================================================================
msst_by_ym = {k: gmean(xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values, TAREA)
              for f, k in zip(mfiles, mkeys)}
years_avail = sorted({y for (y, m) in mkeys})
full_years = [y for y in years_avail if all((y, m) in msst_by_ym for m in range(1, 13))]
sst_annual = np.array([np.mean([msst_by_ym[(y, m)] for m in range(1, 13)]) for y in full_years]) + C2K
print(f"FASST SST: {len(full_years)} complete years [{full_years[0]}-{full_years[-1]}]")

z = np.load(PICTL_CACHE)
pic_years, pic_annual_C = z["years"], z["annual"]
n_pic = min(len(pic_years), len(full_years))
pic_sst = pic_annual_C[:n_pic] + C2K
pic_yrs = np.array(full_years[:n_pic])
fasst_sst = sst_annual[:n_pic]

# =====================================================================
# ROW 2: Nino-3.4 anomaly, monthly, K (delta-T == same value in C or K)
# =====================================================================
box = (TLAT >= -5) & (TLAT <= 5) & (TLONG >= 190) & (TLONG <= 240)
jj, ii = np.where(box); J0, J1, I0, I1 = jj.min(), jj.max()+1, ii.min(), ii.max()+1
sub_box = box[J0:J1, I0:I1]; sub_w = (TAREA * box)[J0:J1, I0:I1]; sub_wsum = sub_w.sum()
def bmean(sub):
    s = np.asarray(sub, np.float64)
    return float(np.nansum(np.where(np.isfinite(s) & sub_box, s, 0) * sub_w) / sub_wsum)

mt = np.array([fyr(*k) for k in mkeys]); mmon = np.array([k[1] for k in mkeys])
msst34 = np.array([bmean(xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values[J0:J1, I0:I1]) for f in mfiles])
ma = deseason(msst34, mmon)
print(f"FASST Nino3.4: {y0}-{y1} ({len(mkeys)} months)")

n_needed = len(mkeys)
ds = xr.open_dataset(PI_FILE)
PI_TAREA = np.nan_to_num(ds["TAREA"].values.astype(np.float64))
PI_TLAT, PI_TLONG = ds["TLAT"].values, ds["TLONG"].values
pi_box = (PI_TLAT >= -5) & (PI_TLAT <= 5) & (PI_TLONG >= 190) & (PI_TLONG <= 240)
pjj, pii = np.where(pi_box); PJ0, PJ1, PI0, PI1 = pjj.min(), pjj.max()+1, pii.min(), pii.max()+1
pi_sub_box = pi_box[PJ0:PJ1, PI0:PI1]; pi_sub_w = (PI_TAREA * pi_box)[PJ0:PJ1, PI0:PI1]; pi_sub_wsum = pi_sub_w.sum()
def pi_bmean(sub):
    s = np.asarray(sub, np.float64)
    return float(np.nansum(np.where(np.isfinite(s) & pi_sub_box, s, 0) * pi_sub_w) / pi_sub_wsum)

sst_all = ds["SST"].isel(z_t=0).values[:n_needed, PJ0:PJ1, PI0:PI1]
pi_series = np.array([pi_bmean(sst_all[i]) for i in range(n_needed)])
pi_months = np.array([(1 + i % 12) for i in range(n_needed)])
pi_a = deseason(pi_series, pi_months)
pi_mt = y0 + np.arange(n_needed) / 12.0 + (0.5 / 12.0)

# =====================================================================
# figure: 2 rows x 2 cols, ONE shared legend
# =====================================================================
split = args.split_year
fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
fig.subplots_adjust(top=0.90, right=0.97, left=0.075, bottom=0.09, wspace=0.22, hspace=0.32)

# --- row 1: SST ---
maskA_y = pic_yrs < split; maskB_y = ~maskA_y
for ax, m in [(axes[0, 0], maskA_y), (axes[0, 1], maskB_y)]:
    ax.plot(pic_yrs[m], pic_sst[m], color=C_TRUTH, lw=LW_TRUTH, ls=(0, (5, 2)),
            dash_capstyle="round", marker="o", ms=4.5, mfc=C_TRUTH, mec="none",
            zorder=1, label="CESM2 piControl")
    ax.plot(pic_yrs[m], fasst_sst[m], color=C_SIM, lw=LW_SIM, solid_capstyle="round",
            marker="o", ms=5.5, mfc=C_SIM, mec="none", zorder=3, label="FASST")
    ax.margins(x=0.03, y=0.15)
    span = pic_yrs[m][-1] - pic_yrs[m][0]
    ax.xaxis.set_major_locator(mtick.MultipleLocator(5 if span < 30 else 10))
axes[0, 0].set_ylabel("Global-mean SST (K)")
axes[0, 0].set_title(f"{pic_yrs[maskA_y][0]}–{pic_yrs[maskA_y][-1]}", fontsize=12, color=MUTE, loc="left")
axes[0, 1].set_title(f"{pic_yrs[maskB_y][0]}–{pic_yrs[maskB_y][-1]}", fontsize=12, color=MUTE, loc="left")
axes[0, 1].text(0.03, 0.93, f"FASST  {fasst_sst[maskB_y].mean():.2f} ± {fasst_sst[maskB_y].std():.2f} K",
                transform=axes[0, 1].transAxes, color=C_SIM, fontsize=10.5, ha="left", va="top")
axes[0, 1].text(0.03, 0.86, f"CESM2 piControl  {pic_sst[maskB_y].mean():.2f} ± {pic_sst[maskB_y].std():.2f} K",
                transform=axes[0, 1].transAxes, color=C_TRUTH, fontsize=10.5, ha="left", va="top")

# --- row 2: Nino3.4 anomaly ---
maskA_m = mt < split; maskB_m = ~maskA_m
for ax, m in [(axes[1, 0], maskA_m), (axes[1, 1], maskB_m)]:
    ax.axhline(0, color="#c7c7c7", lw=0.9, zorder=0)
    ax.plot(pi_mt[m], pi_a[m], color=C_TRUTH, lw=LW_TRUTH, ls=(0, (5, 2)),
            dash_capstyle="round", zorder=1, label="CESM2 piControl")
    ax.plot(mt[m], ma[m], color=C_SIM, lw=LW_SIM, solid_capstyle="round", zorder=2, label="FASST")
    ax.margins(x=0.008, y=0.15); ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_locator(mtick.MultipleLocator(2))
    span = mt[m][-1] - mt[m][0]
    ax.xaxis.set_major_locator(mtick.MultipleLocator(5 if span < 30 else 10))
axes[1, 0].set_ylabel("Niño-3.4 anomaly (K)")
axes[1, 1].text(0.03, 0.05, f"FASST  {ma[maskB_m].mean():+.2f} ± {ma[maskB_m].std():.2f} K",
                transform=axes[1, 1].transAxes, color=C_SIM, fontsize=10.5, ha="left", va="bottom")
axes[1, 1].text(0.03, 0.12, f"CESM2 piControl  {pi_a[maskB_m].mean():+.2f} ± {pi_a[maskB_m].std():.2f} K",
                transform=axes[1, 1].transAxes, color=C_TRUTH, fontsize=10.5, ha="left", va="bottom")

fig.text(0.075, 0.97, f"FASST vs CESM2 piControl ({y0}–{y1})", ha="left", va="top",
         fontsize=15, fontweight="bold")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 0.0), handlelength=1.6, labelcolor="linecolor")

fig.savefig(args.out); fig.savefig(args.out.replace(".png", ".pdf"))
print(f"wrote {args.out} (+ .pdf)")
