#!/usr/bin/env python
"""Annual-mean SST OUTSIDE the ENSO cold tongue, to test whether the coupled
sims' Nino-3.4 cold drift is a local cold-tongue defect or a basin-wide one.

Panels, all four series (finetuned sim, unfinetuned sim, CAM6/CREDIT, CESM2-LE):
  (a) 60S-60N ocean mean SST
  (b) the same mean with the tropical Pacific cold tongue REMOVED
      (10S-10N, 160-280E)
  (c) the two differenced -- how much of each series' mean the tongue carries

Ocean-only and 60S-60N throughout: the CAM6 SST field is filled with 273.0 K
over land and ice, so land is masked with LANDFRAC<0.5 and the ice-filled polar
latitudes are excluded rather than trusted.

Cached to --cache; delete it to rebuild.
Usage: python plot_sst_mean_excl_tongue.py [--out PNG]
"""
import argparse, glob, re, os
from datetime import timedelta
import numpy as np, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

RUN_FT  = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_sensreg50yr/run"
RUN_UFT = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw_iceloop/run"
LE2_SST_DIR = "/glade/campaign/cgd/cesm/CESM2-LE/ocn/proc/tseries/month_1/SST"
CAM6_DIR    = "/glade/derecho/scratch/wchapman/b_credit_runs"

# cold tongue / ENSO region excluded in panel (b)
TG_LAT, TG_LON = (-10.0, 10.0), (160.0, 280.0)
LAT_LIM = 60.0

ap = argparse.ArgumentParser()
ap.add_argument("--run-ft", default=RUN_FT)
ap.add_argument("--run-uft", default=RUN_UFT)
ap.add_argument("--ref-member", default="LE2-1231.002")
ap.add_argument("--out", default="/glade/u/home/praggarwal/couple/output/sst_mean_excl_tongue.png")
ap.add_argument("--cache", default="/glade/u/home/praggarwal/couple/output/sst_mean_excl_tongue_cache.npz")
args = ap.parse_args()

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11.5, "axes.labelsize": 12,
    "legend.fontsize": 10.5, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.linewidth": 0, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#ececec", "grid.linewidth": 0.7,
    "xtick.color": "#6b7177", "ytick.color": "#6b7177",
    "text.color": "#1a1a1a", "axes.labelcolor": "#3d3d3d",
    "savefig.dpi": 300, "figure.dpi": 140,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "xtick.major.size": 0, "ytick.major.size": 0,
})
C = {"ft": "#D55E00", "uft": "#009E73", "cam6": "#0072B2", "le2": "#5b6167"}
MUTE = "#6b7177"

ym = lambda f: (lambda m: (int(m.group(1)), int(m.group(2))))(
    re.search(r"\.(\d{4})-(\d{2})\.nc$", f))

def wmean(fld, w):
    f = np.asarray(fld, np.float64)
    ok = np.isfinite(f) & (w > 0)
    return float(np.sum(f[ok] * w[ok]) / np.sum(w[ok]))

# ---------------- gx1v7 weights (sims + CESM2-LE) ----------------
def gx_weights(sample):
    g = xr.open_dataset(sample)
    TLAT = g.TLAT.values
    TLONG = np.mod(g.TLONG.values, 360.0)
    TAREA = np.nan_to_num(g.TAREA.values.astype(np.float64))
    base = TAREA * (np.abs(TLAT) <= LAT_LIM)
    tongue = ((TLAT >= TG_LAT[0]) & (TLAT <= TG_LAT[1]) &
              (TLONG >= TG_LON[0]) & (TLONG <= TG_LON[1]))
    return base, base * (~tongue)

# ---------------- CAM6 weights (regular lat/lon, LANDFRAC-masked) ----------------
def cam_weights(sample_zarr):
    d = xr.open_zarr(sample_zarr, consolidated=False)
    lat, lon = d["latitude"].values, d["longitude"].values
    latm, lonm = np.meshgrid(lat, lon, indexing="ij")
    lf = d["LANDFRAC"]
    lf = lf.isel(time=0).values if "time" in lf.dims else lf.values
    ocean = (lf < 0.5) & (np.abs(latm) <= LAT_LIM)
    base = np.cos(np.deg2rad(latm)) * ocean
    tongue = ((latm >= TG_LAT[0]) & (latm <= TG_LAT[1]) &
              (np.mod(lonm, 360.0) >= TG_LON[0]) & (np.mod(lonm, 360.0) <= TG_LON[1]))
    return base, base * (~tongue)

def from_rundir(rundir):
    fs = sorted(glob.glob(f"{rundir}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
    if not fs: return {}
    wa, wb = gx_weights(fs[0])
    out = {}
    for f in fs:
        T = xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values
        out[ym(f)] = (wmean(T, wa), wmean(T, wb))
    return out

def from_le2(y0, y1):
    wa, wb = gx_weights(sorted(glob.glob(f"{args.run_ft}/*.pop.h.*-01.nc"))[0])
    out = {}
    for f in sorted(glob.glob(f"{LE2_SST_DIR}/*{args.ref_member}*.pop.h.SST.*.nc")):
        r = re.search(r"\.(\d{4})\d\d-(\d{4})\d\d\.nc$", f)
        if not r or int(r.group(1)) > y1 or int(r.group(2)) < y0: continue
        ds = xr.open_dataset(f)["SST"]
        if "z_t" in ds.dims: ds = ds.isel(z_t=0)
        for i in range(ds.sizes["time"]):
            t = ds["time"].values[i] - timedelta(days=15)   # POP stamps end-of-month
            if not (y0 <= t.year <= y1): continue
            fld = ds.isel(time=i).values
            out[(t.year, t.month)] = (wmean(fld, wa), wmean(fld, wb))
    return out

def from_cam6(y0, y1):
    files = {y: os.path.join(CAM6_DIR,
             f"b.e21.CREDIT_climate_branch_1980_{y}_zmdata_ERA5scaled_zmdata_Qtot.zarr")
             for y in range(y0, y1 + 1)}
    files = {y: f for y, f in files.items() if os.path.isdir(f)}
    if not files: return {}
    wa, wb = cam_weights(next(iter(files.values())))
    out = {}
    for y, f in files.items():
        ds = xr.open_zarr(f, consolidated=False)["SST"]
        mon = np.array([int(str(t)[5:7]) for t in ds["time"].values])
        for k in range(1, 13):
            sel = np.where(mon == k)[0]
            if not sel.size: continue
            fld = ds.isel(time=slice(sel[0], sel[-1] + 1)).mean("time").values - 273.15
            out[(y, k)] = (wmean(fld, wa), wmean(fld, wb))
    return out

def pack(d):   return np.array([[k[0], k[1], v[0], v[1]] for k, v in sorted(d.items())], np.float64)
def unpack(a): return {(int(r[0]), int(r[1])): (r[2], r[3]) for r in a}

if os.path.exists(args.cache):
    z = np.load(args.cache); S = {k: unpack(z[k]) for k in z.files}
    print(f"loaded cache {args.cache}")
else:
    S = {"ft": from_rundir(args.run_ft), "uft": from_rundir(args.run_uft)}
    yrs = sorted({k[0] for k in S["ft"]} | {k[0] for k in S["uft"]})
    S["le2"]  = from_le2(yrs[0], yrs[-1])
    S["cam6"] = from_cam6(yrs[0], yrs[-1])
    np.savez(args.cache, **{k: pack(v) for k, v in S.items()})
    print(f"wrote cache {args.cache}")

for k, v in S.items():
    ks = sorted(v)
    print(f"  {k:5s} {len(v):4d} months  [{ks[0]}..{ks[-1]}]" if v else f"  {k:5s} EMPTY")

def annual(d, j):
    yrs = sorted({k[0] for k in d if all((k[0], mo) in d for mo in range(1, 13))})
    return (np.array(yrs),
            np.array([np.mean([d[(y, mo)][j] for mo in range(1, 13)]) for y in yrs]))

LBL = {"ft": "Finetuned (sensreg) coupled sim", "uft": "Unfinetuned coupled sim",
       "cam6": "CAM6 / CREDIT (training truth)", "le2": "CESM2-LE (LE2-1231.002)"}
LS  = {"ft": "-", "uft": (0, (3, 1, 1, 1)), "cam6": (0, (1, 1.4)), "le2": (0, (5, 2))}
LW  = {"ft": 2.2, "uft": 1.8, "cam6": 1.6, "le2": 1.6}
ORD = ["le2", "cam6", "uft", "ft"]

fig, axes = plt.subplots(3, 1, figsize=(9.6, 10.4), sharex=True)
fig.subplots_adjust(top=0.855, right=0.975, left=0.095, bottom=0.06, hspace=0.30)
ax1, ax2, ax3 = axes

for i, k in enumerate(ORD):
    if not S[k]: continue
    ys, va = annual(S[k], 0)
    _,  vb = annual(S[k], 1)
    kw = dict(color=C[k], lw=LW[k], ls=LS[k], dash_capstyle="round",
              solid_capstyle="round", zorder=1 + i)
    ax1.plot(ys + 0.5, va, label=f"{LBL[k]}   mean {va.mean():.2f} °C", **kw)
    ax2.plot(ys + 0.5, vb, label=f"{LBL[k]}   mean {vb.mean():.2f} °C", **kw)
    ax3.plot(ys + 0.5, va - vb, label=f"{LBL[k]}   mean {(va-vb).mean():+.3f} K", **kw)

ax1.set_ylabel("SST, 60°S–60°N (°C)")
ax2.set_ylabel("SST, tongue excluded (°C)")
ax3.set_ylabel("with − without tongue (K)")
ax3.set_xlabel("Year")
ax3.axhline(0, color="#c7c7c7", lw=0.9, zorder=0)
for ax in axes:
    ax.margins(x=0.008)
    ax.grid(axis="x", visible=False)
    ax.xaxis.set_major_locator(mtick.MultipleLocator(5))

fig.text(0.095, 0.985, "Annual-mean SST with and without the ENSO cold tongue",
         ha="left", va="top", fontsize=15, fontweight="bold")
fig.text(0.095, 0.948,
         f"ocean only, 60°S–60°N · tongue = {abs(TG_LAT[0]):.0f}°S–{TG_LAT[1]:.0f}°N, "
         f"{TG_LON[0]:.0f}–{TG_LON[1]:.0f}°E · CAM6 land masked with LANDFRAC<0.5",
         ha="left", va="top", color=MUTE, fontsize=10)
for ax, tag in zip(axes, "abc"):
    ax.text(-0.088, 1.15, tag, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11.5, fontweight="bold", color="#3d3d3d")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), frameon=False,
              handlelength=2.2, borderaxespad=0.0, labelcolor="linecolor",
              ncol=2, columnspacing=1.8, handletextpad=0.7)

fig.savefig(args.out); fig.savefig(args.out.replace(".png", ".pdf"))
print(f"wrote {args.out} (+ .pdf)")
for k in ORD:
    if not S[k]: continue
    ys, va = annual(S[k], 0); _, vb = annual(S[k], 1)
    print(f"  {k:5s} {ys[0]}-{ys[-1]}  all {va.mean():6.3f} C  no-tongue {vb.mean():6.3f} C "
          f"| drift(all) {va[-1]-va[0]:+.3f} K  drift(no-tongue) {vb[-1]-vb[0]:+.3f} K")
