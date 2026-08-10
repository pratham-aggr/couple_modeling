"""Niño3.4 / ONI for the v15 UNet standalone run, scored against LE2-1231.002 truth.

Box-averages SST (5S-5N, 170W-120W) each month from the v15 34yr run and from LE2,
removes the LE2 monthly climatology -> anomaly, 3-month running mean -> ONI, and
plots model vs truth.  Output: output/v15unet_nino34_oni_LE2.png
"""
import glob, re
import numpy as np
import xarray as xr
import numpy.ma as ma
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNDIR = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_v15unet34yr/run"
LE2DIR = "/glade/campaign/cgd/cesm/CESM2-LE/ocn/proc/tseries/month_1/SST"
LE2 = [f"{LE2DIR}/b.e21.BHISTcmip6.f09_g17.LE2-1231.002.pop.h.SST.{d}.nc"
       for d in ("198001-198912", "199001-199912")]
OUT = "/glade/u/home/praggarwal/couple/output/v15unet_nino34_oni_LE2.png"
NYEARS = 18   # 1980-1997


def box_mean_series(sst3d, box, w):
    return np.array([np.average(np.nan_to_num(sst3d[t])[box], weights=w)
                     for t in range(sst3d.shape[0])])


def run3(a):
    o = np.full_like(a, np.nan)
    for i in range(1, len(a) - 1):
        o[i] = a[i - 1:i + 2].mean()
    return o


def main():
    # --- grid + Niño3.4 box (identical grid for model and LE2) ---
    files = sorted(glob.glob(f"{RUNDIR}/g.e21.MEMO_GIAF_v01.pop.h.????-??.nc"))[:NYEARS * 12]
    with xr.open_dataset(files[0], decode_times=False) as d0:
        tlat, tlon = d0["TLAT"].values, d0["TLONG"].values
        tarea, kmt = d0["TAREA"].values.astype(float), d0["KMT"].values
    box = (tlat >= -5) & (tlat <= 5) & (tlon >= 190) & (tlon <= 240) & (kmt > 0)
    w = tarea[box]
    mon = np.tile(np.arange(1, 13), NYEARS)
    year = np.repeat(np.arange(1980, 1980 + NYEARS), 12)

    # --- model Niño3.4 SST ---
    mod = np.array([xr.open_dataset(f, decode_times=False)["TEMP"].isel(time=0, z_t=0).values.astype(float)
                    for f in files])
    mod = box_mean_series(mod, box, w)

    # --- LE2 truth Niño3.4 SST + climatology ---
    le2 = np.concatenate([xr.open_dataset(f, decode_times=False)["SST"].isel(z_t=0).values
                          for f in LE2], axis=0)[:NYEARS * 12]
    le2 = box_mean_series(le2, box, w)
    clim = np.array([le2[mon == m].mean() for m in range(1, 13)])   # LE2 monthly climatology

    oni_mod = run3(mod - clim[mon - 1])
    oni_le2 = run3(le2 - clim[mon - 1])
    r = ma.corrcoef(ma.masked_invalid(oni_mod), ma.masked_invalid(oni_le2))[0, 1]
    amp = np.nanstd(oni_mod) / np.nanstd(oni_le2)
    print(f"mean Niño3.4: model {mod.mean():.2f}C  LE2 {clim.mean():.2f}C  (bias {mod.mean()-clim.mean():+.2f} K)")
    print(f"ONI std: model {np.nanstd(oni_mod):.2f}  LE2 {np.nanstd(oni_le2):.2f}  ({amp:.2f}x) ; r={r:.2f}")

    x = year + (mon - 1) / 12.0
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(0.5, color="r", ls=":", lw=0.7); ax.axhline(-0.5, color="b", ls=":", lw=0.7)
    ax.plot(x, oni_le2, color="0.35", lw=1.7, label="LE2 truth ONI")
    ax.plot(x, oni_mod, color="tab:red", lw=1.3, label="v15 model ONI (anom vs LE2 clim)")
    ax.set_xlim(x[0], x[-1]); ax.set_ylabel("ONI  (Niño3.4 anomaly, K)")
    ax.set_title(f"Niño3.4 / ONI — v15 UNet vs LE2 truth, 1980–1997   (r={r:.2f},  amp {amp:.1f}×)")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT, dpi=130)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
