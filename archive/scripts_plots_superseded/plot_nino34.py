#!/usr/bin/env python
"""Publication-ready Niño-3.4 ENSO comparison: finetuned (sensreg) + unfinetuned
(base MSE-only) coupled ML-POP2 sims vs CESM2-LE truth (LE2-1231.002) vs CAM6
(the ERA5-scaled CREDIT run our atmosphere emulator trains on) -- two panels,
raw SST and deseasonalized anomaly.

Re-runnable: scans each run directory for whatever monthly pop.h files exist and
plots exactly that span, so re-running as either simulation advances extends its
own line automatically. The unfinetuned sim need not span the same months as the
finetuned one (it's a separate, shorter experiment).

Niño-3.4 box: 5S-5N, 170W-120W (TLONG 190-240E). Anomalies deseasonalized by each
series' own monthly climatology over its own available period.

Usage:  python plot_nino34.py [--run DIR] [--run-uft DIR] [--out PNG] [--ref-member ID]
"""
import argparse, glob, re, os
from datetime import timedelta
import numpy as np, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

RUN_DEFAULT     = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_sensreg50yr/run"
RUN_UFT_DEFAULT = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw1yr/run"
LE2_SST_DIR = "/glade/campaign/cgd/cesm/CESM2-LE/ocn/proc/tseries/month_1/SST"
CAM6_DIR    = "/glade/derecho/scratch/wchapman/b_credit_runs"

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=RUN_DEFAULT, help="finetuned (sensreg) rundir")
ap.add_argument("--run-uft", default=RUN_UFT_DEFAULT, help="unfinetuned (base MSE-only) rundir")
ap.add_argument("--out", default="/glade/u/home/praggarwal/couple/output/nino34_sensreg_vs_le2.png")
ap.add_argument("--ref-member", default="LE2-1231.002", help="LE2 member drawn as the truth line")
args = ap.parse_args()

# ---------- style ----------
# Okabe-Ito CVD-safe palette, four hues, each with its OWN linestyle so identity
# survives black/white printing regardless of color perception.
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11.5, "axes.titlesize": 12.5,
    "axes.labelsize": 11.5, "legend.fontsize": 10, "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5, "axes.linewidth": 0.8, "axes.edgecolor": "#4d4d4d",
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e6e6",
    "grid.linewidth": 0.7, "xtick.color": "#4d4d4d", "ytick.color": "#4d4d4d",
    "text.color": "#1a1a1a", "axes.labelcolor": "#1a1a1a",
    "savefig.dpi": 300, "figure.dpi": 140,
    "axes.spines.top": False, "axes.spines.right": False,
})
C_SIM, C_UFT, C_TRUTH, C_CAM6, MUTE = "#D55E00", "#009E73", "#5b6167", "#0072B2", "#6b7177"
INK = "#1a1a1a"

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

def deseason(series, months):
    a = series.astype(np.float64).copy()
    for k in range(1, 13):
        sel = months == k
        if sel.any(): a[sel] = series[sel] - series[sel].mean()
    return a

fyr = lambda yy, mm: yy + (mm - 0.5) / 12.0

def monthly_series(rundir):
    mfiles = sorted(glob.glob(f"{rundir}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
    if not mfiles: return None
    mkeys = [ym(f) for f in mfiles]
    box = load_box_weights(mfiles[0])
    J0, J1, I0, I1, sub_box, sub_w, sub_wsum = box
    sst = np.array([bmean(xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values[J0:J1, I0:I1],
                           sub_box, sub_w, sub_wsum) for f in mfiles])
    mmon = np.array([k[1] for k in mkeys])
    mt = np.array([fyr(*k) for k in mkeys])
    return dict(mkeys=mkeys, mt=mt, sst=sst, a=deseason(sst, mmon), box=box)

ft  = monthly_series(args.run)
uft = monthly_series(args.run_uft)
if ft is None: raise SystemExit(f"no monthly pop.h files in {args.run}")
print(f"finetuned span {ft['mkeys'][0][0]}-{ft['mkeys'][0][1]:02d} .. "
      f"{ft['mkeys'][-1][0]}-{ft['mkeys'][-1][1]:02d} ({len(ft['mkeys'])} mo)")
if uft is not None:
    print(f"unfinetuned span {uft['mkeys'][0][0]}-{uft['mkeys'][0][1]:02d} .. "
          f"{uft['mkeys'][-1][0]}-{uft['mkeys'][-1][1]:02d} ({len(uft['mkeys'])} mo)")

all_mkeys = sorted(set(ft['mkeys']) | set(uft['mkeys'] if uft else []))
y0, y1 = all_mkeys[0][0], all_mkeys[-1][0]
J0, J1, I0, I1, sub_box, sub_w, sub_wsum = ft['box']

# ---------- LE2 truth, over the union of months ----------
want = set(all_mkeys)
mfs = [f for f in sorted(glob.glob(f"{LE2_SST_DIR}/*{args.ref_member}*.pop.h.SST.*.nc"))
       if (lambda r: r and int(r.group(1)) <= y1 and int(r.group(2)) >= y0)(
           re.search(r"\.(\d{4})\d\d-(\d{4})\d\d\.nc$", f))]
tvals = {}
for f in mfs:
    ds = xr.open_dataset(f)["SST"]
    if "z_t" in ds.dims: ds = ds.isel(z_t=0)
    sub = ds.isel(nlat=slice(J0, J1), nlon=slice(I0, I1)).values
    for i in range(ds.sizes["time"]):
        t = ds["time"].values[i] - timedelta(days=15)   # CESM stamps END of month -> shift to mid
        key = (t.year, t.month)
        if key in want: tvals[key] = bmean(sub[i], sub_box, sub_w, sub_wsum)

def truth_for(mkeys):
    series = np.array([tvals.get(k, np.nan) for k in mkeys])
    mmon = np.array([k[1] for k in mkeys])
    return series, deseason(series, mmon)

ref_series_all, ref_a_all = truth_for(all_mkeys)
t_mt_all = np.array([fyr(*k) for k in all_mkeys])
print(f"reference member {args.ref_member}: {np.isfinite(ref_series_all).sum()} of {len(all_mkeys)} months matched")

# ---------- CAM6 (training truth), over the union of years ----------
years_needed = sorted({y for (y, m) in all_mkeys})
cam_files = {y: os.path.join(CAM6_DIR, f"b.e21.CREDIT_climate_branch_1980_{y}_zmdata_ERA5scaled_zmdata_Qtot.zarr")
             for y in years_needed}
cam_files = {y: f for y, f in cam_files.items() if os.path.isdir(f)}
cam_vals = {}
if cam_files:
    g0 = xr.open_zarr(next(iter(cam_files.values())), consolidated=False)
    lat = g0["latitude"].values; lon = g0["longitude"].values
    latm, lonm = np.meshgrid(lat, lon, indexing="ij")
    cbox = (latm >= -5) & (latm <= 5) & (lonm >= 190) & (lonm <= 240)
    cw = np.cos(np.deg2rad(latm)) * cbox
    jj, ii = np.where(cbox); J0c, J1c, I0c, I1c = jj.min(), jj.max()+1, ii.min(), ii.max()+1
    sub_cbox = cbox[J0c:J1c, I0c:I1c]; sub_cw = cw[J0c:J1c, I0c:I1c]; cwsum = sub_cw.sum()
    for y, f in cam_files.items():
        ds = xr.open_zarr(f, consolidated=False)["SST"]
        arr = ds.isel(latitude=slice(J0c, J1c), longitude=slice(I0c, I1c)).values
        months = np.array([int(str(t)[5:7]) for t in ds["time"].values])
        for m in range(1, 13):
            sel = months == m
            if not sel.any(): continue
            fld = arr[sel].mean(axis=0) - 273.15
            cam_vals[(y, m)] = float(np.nansum(np.where(sub_cbox, fld, 0) * sub_cw) / cwsum)

cam_series_all = None
if cam_vals and all(k in cam_vals for k in all_mkeys):
    cam_series_all = np.array([cam_vals[k] for k in all_mkeys])
    mmon_all = np.array([k[1] for k in all_mkeys])
    cam_a_all = deseason(cam_series_all, mmon_all)

# ---------- figure ----------
fig, ax = plt.subplots(2, 1, figsize=(9.4, 6.8), sharex=True, constrained_layout=True)

# (a) raw SST
ax[0].plot(t_mt_all, ref_series_all, color=C_TRUTH, lw=1.5, ls=(0,(5,2)), dash_capstyle="round",
           label=f"CESM2-LE truth ({args.ref_member})")
if cam_series_all is not None:
    ax[0].plot(t_mt_all, cam_series_all, color=C_CAM6, lw=1.5, ls=(0,(1,1.4)), dash_capstyle="round",
               label="CAM6 (training truth)")
if uft is not None:
    ax[0].plot(uft['mt'], uft['sst'], color=C_UFT, lw=1.8, ls=(0,(3,1,1,1)), dash_capstyle="round",
               label="Unfinetuned sim")
ax[0].plot(ft['mt'], ft['sst'], color=C_SIM, lw=2.2, solid_capstyle="round", label="Finetuned (sensreg) sim")
ax[0].set_ylabel("Niño-3.4 SST  (°C)")
fig.suptitle("Niño-3.4 SST (5°S–5°N, 170°W–120°W): finetuned + unfinetuned vs CESM2-LE / CAM6",
             x=0.045, ha="left", fontsize=12.5, color=INK)
ax[0].annotate("a", xy=(0,1), xycoords="axes fraction", xytext=(-38,4),
               textcoords="offset points", fontweight="bold", fontsize=12, va="bottom")

# (b) anomaly
ax[1].axhline(0, color="#b0b0b0", lw=0.8, zorder=0)
ax[1].plot(t_mt_all, ref_a_all, color=C_TRUTH, lw=1.5, ls=(0,(5,2)), dash_capstyle="round",
           label=f"truth  σ = {np.nanstd(ref_a_all):.2f} K")
if cam_series_all is not None:
    ax[1].plot(t_mt_all, cam_a_all, color=C_CAM6, lw=1.5, ls=(0,(1,1.4)), dash_capstyle="round",
               label=f"CAM6  σ = {np.nanstd(cam_a_all):.2f} K")
if uft is not None:
    ax[1].plot(uft['mt'], uft['a'], color=C_UFT, lw=1.8, ls=(0,(3,1,1,1)), dash_capstyle="round",
               label=f"unfinetuned  σ = {uft['a'].std():.2f} K")
ax[1].plot(ft['mt'], ft['a'], color=C_SIM, lw=2.2, solid_capstyle="round",
           label=f"finetuned  σ = {ft['a'].std():.2f} K")
ax[1].set_ylabel("Niño-3.4 anomaly  (K)"); ax[1].set_xlabel("Year")
ax[1].annotate("b", xy=(0,1), xycoords="axes fraction", xytext=(-38,4),
               textcoords="offset points", fontweight="bold", fontsize=12, va="bottom")

ax[0].legend(loc="lower left", frameon=False, handlelength=1.6, borderaxespad=0.2)
ax[1].legend(loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.2)
for a in ax:
    a.margins(x=0.008); a.grid(axis="x", visible=False)
    a.xaxis.set_major_locator(mtick.MultipleLocator(1))
    a.tick_params(length=3, width=0.8)
fig.savefig(args.out); fig.savefig(args.out.replace(".png", ".pdf"))
print(f"wrote {args.out} (+ .pdf)")
print(f"  finetuned   mean {ft['sst'].mean():.2f}°C  anomaly σ {ft['a'].std():.3f} K")
if uft is not None:
    print(f"  unfinetuned mean {uft['sst'].mean():.2f}°C  anomaly σ {uft['a'].std():.3f} K")
print(f"  truth       mean {np.nanmean(ref_series_all):.2f}°C  anomaly σ {np.nanstd(ref_a_all):.3f} K")
if cam_series_all is not None:
    print(f"  CAM6        mean {np.nanmean(cam_series_all):.2f}°C  anomaly σ {np.nanstd(cam_a_all):.3f} K")
np.savez(os.path.splitext(args.out)[0]+".npz", mt=ft['mt'], msst=ft['sst'], ma=ft['a'],
         ref_series=ref_series_all, ref_a=ref_a_all)
