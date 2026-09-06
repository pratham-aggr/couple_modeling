"""Remake output/fasst_pub/pert.pdf from real full monthly data (not annual means),
strictly matching the theme of make_fasst_neurips_2panel.py /
output/fasst_pub/fasst_neurips_2panel_140yr.pdf.

Interactive-ice (FASST config) pair, per verify_pert2K_recovery.py:
  pert = memo_pop_standalone_gx1v7_diagfw_iceemu_pert2K   (+2 K top-100m IC kick)
  ctrl = memo_pop_standalone_gx1v7_diagfw_iceloop         (FASST production run;
         read-only here, only its first 5 already-written years are touched)
Anomaly = area-weighted global-mean SST(pert) - SST(ctrl), monthly, 1980-1984.
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
PERT = f"{S}/memo_pop_standalone_gx1v7_diagfw_iceemu_pert2K/run"
CTRL = f"{S}/memo_pop_standalone_gx1v7_diagfw_iceloop/run"
YEARS = range(1980, 1985)
OUT = "output/fasst_pub/temp_pert.pdf"


def ym(f):
    m = re.search(r"\.(\d{4})-(\d{2})\.nc$", f)
    return int(m.group(1)), int(m.group(2))


def gmsst_series(rundir, years):
    files = sorted(glob.glob(f"{rundir}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
    out = {}
    for f in files:
        y, mo = ym(f)
        if y not in years:
            continue
        ds = xr.open_dataset(f)
        sst = ds["TEMP"].isel(time=0, z_t=0) if "TEMP" in ds else ds["SST"].isel(time=0)
        w = np.nan_to_num(ds["TAREA"].values.astype(np.float64))
        s = sst.values.astype(np.float64)
        ok = np.isfinite(s)
        out[(y, mo)] = float(np.nansum(np.where(ok, s, 0) * w) / (w * ok).sum())
        ds.close()
    return out


ctrl = gmsst_series(CTRL, YEARS)
pert = gmsst_series(PERT, YEARS)

keys = sorted(k for k in pert if k in ctrl)
missing_expected = [(y, m) for y in YEARS for m in range(1, 13)]
missing = [k for k in missing_expected if k not in dict.fromkeys(keys)]
if missing:
    print(f"NOTE: {len(missing)} months missing from the overlap: {missing[:5]}...")

anom = np.array([pert[k] - ctrl[k] for k in keys])
y0 = keys[0][0]
x = np.array([(y - y0) + (m - 0.5) / 12.0 for (y, m) in keys])

# annual means for the headline numbers (matches the original pert.pdf annotation)
ann = {}
for y in YEARS:
    vals = [pert[(y, m)] - ctrl[(y, m)] for m in range(1, 13) if (y, m) in pert and (y, m) in ctrl]
    if len(vals) == 12:
        ann[y] = np.mean(vals)
ann_years = sorted(ann)
print(f"{'year':>6} {'model_yr':>9} {'anom(K)':>9}")
for i, y in enumerate(ann_years):
    print(f"{y:>6} {i:>9} {ann[y]:>+9.3f}")
if len(ann_years) >= 2:
    a0, a1 = ann[ann_years[0]], ann[ann_years[-1]]
    print(f"decay: {a0:+.2f} -> {a1:+.2f} K "
          f"({100*(1 - round(a1, 2)/round(a0, 2)):.0f}% reduction, years 0-{len(ann_years)-1})")

fig, ax = plt.subplots(figsize=(6.4, 3.6))
fig.subplots_adjust(top=0.86, right=0.97, left=0.13, bottom=0.16)
fig.suptitle("FASST Response to a +2 K Ocean Temperature Perturbation",
             fontsize=11, fontweight="bold", y=0.985)

ax.axhline(0, color="#c7c7c7", lw=0.8, zorder=0)
ax.plot(x, anom, color=C_SIM, lw=LW_SIM, solid_capstyle="round",
        marker="o", ms=2.4, mfc=C_SIM, mec="none", zorder=3)

ax.set_ylabel(r"$\Delta$SST (K)")
ax.set_xlabel("Model year")
ax.set_xlim(x.min(), x.max())
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
data_lo, data_hi = anom.min(), anom.max()
pad = 0.08 * (data_hi - data_lo)
ax.set_ylim(data_lo - pad, data_hi + 2.2 * pad)

if len(ann_years) >= 2:
    a0, a1 = ann[ann_years[0]], ann[ann_years[-1]]
    ax.text(0.97, 0.94,
            f"{a0:+.2f} K to {a1:+.2f} K in {len(ann_years)} years\n"
            f"{100*(1 - round(a1, 2)/round(a0, 2)):.0f}% reduction",
            transform=ax.transAxes, fontsize=ANNOT_FS, color=MUTE, ha="right", va="top")

fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
print(f"\nwrote {OUT}")
