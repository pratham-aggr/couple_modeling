"""Plot the full 1-year monthly response to the -2 W/m^2 downwelling-LW forcing
perturbation (job 7145482). Theme strictly matches
output/fasst_pub/fasst_neurips_2panel_140yr.pdf (make_fasst_neurips_2panel.py).

pert  = memo_pop_standalone_gx1v7_diagfw_iceemu_m2_1yr  (--flds_forcing_wm2 -2.0)
ctrl  = memo_pop_standalone_gx1v7_diagfw_iceemu1yr       (emu-aice control)
Anomaly = area-weighted global-mean SST(pert) - SST(ctrl), monthly, 1980.
"""
import glob, re
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------- styling ----
# identical rcParams block to make_fasst_neurips_2panel.py
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.titlesize": 9.5, "axes.labelsize": 9.5,
    "legend.fontsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#ececec", "grid.linewidth": 0.6,
    "xtick.color": "#6b7177", "ytick.color": "#6b7177",
    "text.color": "#1a1a1a", "axes.labelcolor": "#3d3d3d",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "xtick.major.size": 0, "ytick.major.size": 0,
})
C_SIM, MUTE = "#D2691E", "#6b7177"
LW_SIM = 2.0
ANNOT_FS = 8.0

S = "/glade/derecho/scratch/praggarwal"
PERT = f"{S}/memo_pop_standalone_gx1v7_diagfw_iceemu_m2_1yr/run"
CTRL = f"{S}/memo_pop_standalone_gx1v7_diagfw_iceemu1yr/run"
YEAR = 1980
OUT = "output/fasst_pub/flds_pert.pdf"


def ym(f):
    m = re.search(r"\.(\d{4})-(\d{2})\.nc$", f)
    return int(m.group(1)), int(m.group(2))


def gmsst_series(rundir, year):
    files = sorted(glob.glob(f"{rundir}/*.pop.h.{year}-*.nc"))
    out = {}
    for f in files:
        _, mo = ym(f)
        ds = xr.open_dataset(f)
        sst = ds["TEMP"].isel(time=0, z_t=0) if "TEMP" in ds else ds["SST"].isel(time=0)
        w = np.nan_to_num(ds["TAREA"].values.astype(np.float64))
        s = sst.values.astype(np.float64)
        ok = np.isfinite(s)
        out[mo] = float(np.nansum(np.where(ok, s, 0) * w) / (w * ok).sum())
        ds.close()
    return out


ctrl = gmsst_series(CTRL, YEAR)
pert = gmsst_series(PERT, YEAR)

months = [m for m in range(1, 13) if m in ctrl and m in pert]
if len(months) != 12:
    missing = sorted(set(range(1, 13)) - set(months))
    raise SystemExit(f"missing months: {missing}")

anom = np.array([pert[m] - ctrl[m] for m in months])
peak_i = int(np.argmin(anom))
mon_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

print(f"{'mo':>4} {'ctrl':>10} {'pert':>10} {'anom(K)':>10}")
for m, a in zip(months, anom):
    print(f"{mon_labels[m-1]:>4} {ctrl[m]:10.4f} {pert[m]:10.4f} {a:10.4f}")
print(f"peak: {mon_labels[peak_i]} {anom[peak_i]:+.3f} K; year-end (Dec): {anom[-1]:+.3f} K")

# elapsed model-year fraction within year 1, matching the 2-panel figure's
# "Model year" convention (model yr 1 = the run's first year)
x = np.array([1.0 + (m - 0.5) / 12.0 for m in months])

fig, ax = plt.subplots(figsize=(6.4, 3.6))
fig.subplots_adjust(top=0.86, right=0.97, left=0.13, bottom=0.16)
fig.suptitle(r"FASST Response to a $\mathbf{-2}$ W m$^{\mathbf{-2}}$ Downwelling-LW Forcing",
             fontsize=11, fontweight="bold", y=0.985)

ax.axhline(0, color="#c7c7c7", lw=0.8, zorder=0)
ax.plot(x, anom, color=C_SIM, lw=LW_SIM, solid_capstyle="round",
        marker="o", ms=3.0, mfc=C_SIM, mec="none", zorder=3)

ax.set_ylabel(r"$\Delta$SST (K)")
ax.set_xlabel("Model year")
ax.set_xlim(1.0, 2.0)
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1 / 6))
ax.set_xticks(1.0 + np.arange(12) / 12.0 + (0.5 / 12.0))
ax.set_xticklabels(mon_labels)

pct_dec_dec = (ctrl[months[-1]] - pert[months[-1]]) / ctrl[months[-1]] * 100.0
ax.text(0.97, 0.94,
        f"{anom[0]:+.2f} K ({mon_labels[0]}) to {anom[-1]:+.2f} K ({mon_labels[-1]})\n"
        f"peak {anom[peak_i]:+.2f} K ({mon_labels[peak_i]})\n"
        f"{pct_dec_dec:.2f}% SST decrease by year-end",
        transform=ax.transAxes, fontsize=ANNOT_FS, color=MUTE, ha="right", va="top")

fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
print(f"\nwrote {OUT}")
