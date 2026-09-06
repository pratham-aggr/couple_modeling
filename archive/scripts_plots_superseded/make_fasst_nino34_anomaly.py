#!/usr/bin/env python
"""FASST vs CESM2 piControl: Nino-3.4 anomaly, publication style.
Two-panel: (a) 1980-1999 [transient], (b) 2000-latest [quasi-equilibrium].
Independent y-axis scaling per panel. Anomaly (delta-T) is numerically
identical in K and degC, but reported/labeled in K throughout for
consistency with the companion SST-annual figure. Mean+/-std reported only
in the later panel, as a minimal colored annotation (no boxed legend).

Usage: python make_fasst_nino34_anomaly.py --out FILE.png [--split-year 2000]
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

def ym(f):
    m = re.search(r"\.(\d{4})-(\d{2})\.nc$", f)
    return int(m.group(1)), int(m.group(2))
fyr = lambda yy, mm: yy + (mm - 0.5) / 12.0

def deseason(series, months):
    a = series.astype(np.float64).copy()
    for k in range(1, 13):
        sel = months == k
        if sel.any():
            a[sel] = series[sel] - series[sel].mean()
    return a

# ---------- FASST monthly Nino3.4 (K anomaly == degC anomaly) ----------
mfiles = sorted(glob.glob(f"{FASST_RUN}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
mkeys = [ym(f) for f in mfiles]
y0, y1 = mkeys[0][0], mkeys[-1][0]

g = xr.open_dataset(mfiles[0]); TLAT, TLONG = g.TLAT.values, g.TLONG.values
TAREA = np.nan_to_num(g.TAREA.values.astype(np.float64))
box = (TLAT >= -5) & (TLAT <= 5) & (TLONG >= 190) & (TLONG <= 240)
jj, ii = np.where(box); J0, J1, I0, I1 = jj.min(), jj.max()+1, ii.min(), ii.max()+1
sub_box = box[J0:J1, I0:I1]; sub_w = (TAREA * box)[J0:J1, I0:I1]; sub_wsum = sub_w.sum()
def bmean(sub):
    s = np.asarray(sub, np.float64)
    return float(np.nansum(np.where(np.isfinite(s) & sub_box, s, 0) * sub_w) / sub_wsum)

mt = np.array([fyr(*k) for k in mkeys]); mmon = np.array([k[1] for k in mkeys])
msst = np.array([bmean(xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values[J0:J1, I0:I1]) for f in mfiles])
ma = deseason(msst, mmon)
print(f"FASST: {y0}-{y1} ({len(mkeys)} months)")

# ---------- piControl, monthly, matching-length window ----------
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

# ---------- split into two panels (monthly resolution) ----------
split = args.split_year
maskA = mt < split
maskB = ~maskA
tA0, tA1 = mt[maskA][0], mt[maskA][-1]
tB0, tB1 = mt[maskB][0], mt[maskB][-1]

# ---------- figure ----------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 4.6))
fig.subplots_adjust(top=0.83, right=0.97, left=0.075, bottom=0.20, wspace=0.20)

for ax, m_fasst, m_pic in [(axL, maskA, maskA), (axR, maskB, maskB)]:
    ax.axhline(0, color="#c7c7c7", lw=0.9, zorder=0)
    ax.plot(pi_mt[m_pic], pi_a[m_pic], color=C_TRUTH, lw=1.6, ls=(0, (5, 2)),
            dash_capstyle="round", zorder=1, label="CESM2 piControl")
    ax.plot(mt[m_fasst], ma[m_fasst], color=C_SIM, lw=2.4, solid_capstyle="round",
            zorder=2, label="FASST")
    ax.margins(x=0.008, y=0.15); ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_locator(mtick.MultipleLocator(2))

axL.set_ylabel("Niño-3.4 anomaly (K)")
axL.set_title(f"{int(tA0)}–{int(tA1)}", fontsize=12, color=MUTE, loc="left")
axR.set_title(f"{int(tB0)}–{int(tB1)}", fontsize=12, color=MUTE, loc="left")
axL.xaxis.set_major_locator(mtick.MultipleLocator(5))
axR.xaxis.set_major_locator(mtick.MultipleLocator(5 if (tB1 - tB0) < 30 else 10))

# minimal descriptive stats -- ONLY the later (quasi-equilibrium) panel
axR.text(0.03, 0.05, f"FASST  {ma[maskB].mean():+.2f} ± {ma[maskB].std():.2f} K",
          transform=axR.transAxes, color=C_SIM, fontsize=10.5, ha="left", va="bottom")
axR.text(0.03, 0.12, f"CESM2 piControl  {pi_a[maskB].mean():+.2f} ± {pi_a[maskB].std():.2f} K",
          transform=axR.transAxes, color=C_TRUTH, fontsize=10.5, ha="left", va="bottom")

fig.text(0.075, 0.97, f"Niño-3.4 anomaly ({y0}–{y1})", ha="left", va="top",
         fontsize=15, fontweight="bold")

handles, labels = axL.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 0.0), handlelength=1.6, labelcolor="linecolor")

fig.savefig(args.out); fig.savefig(args.out.replace(".png", ".pdf"))
print(f"wrote {args.out} (+ .pdf)")
