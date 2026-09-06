"""gen_emulated_icefrac.py — generate the CICE emulator's aice over 1980-2014,
aligned to the flux cache, for the "train flux net on emulated ICEFRAC" experiment.

Drives the autoregressive CICE emulator (output_cice_solin) offline, forced by the
gx1v7 CREDIT SST (the same SST the flux net trained on) + analytic SOLIN, rolling
CONTINUOUSLY across all 35 years (faithful to the coupled runtime, which never
resets ice per year). Records the APPLIED aice (incl. the --ice_virtual skin floor,
matching the deployed run) at every 6-h step -> emu_aice_global (51100,384,320).

The flux cache is 35 per-year blocks of 1456 samples (each year drops its first 4
steps as the memory ramp), so sample i maps to global timestep:
    g_now(i)  = (i // 1456) * 1460 + (i % 1456) + 4
    g_prev(i) = g_now(i) - 4                      # == the ICEFRAC_prev channel
train_unet.py (with --icefrac_from_file) will use this map to overwrite the
ICEFRAC (ch1) and ICEFRAC_prev (ch4) input channels with emu aice.

Output: emu_aice_global.npy (float16, [0,1], gx1v7 384x320).
"""
import sys, numpy as np, zarr, cftime
from datetime import timedelta, datetime
sys.path.insert(0, "/glade/u/home/praggarwal/couple/camulator_ud/climate")
from model_server import gx1v7_grid, GX1V7_DOMAIN
from cice_coupler import CiceCoupler

CICE_DIR = "/glade/u/home/praggarwal/couple/output/output_cice_solin"
CICE_IC  = "/glade/derecho/scratch/praggarwal/couple_cache_cice_nowind/ice_ic_1980-01.npz"
ZARR     = "/glade/derecho/scratch/praggarwal/zarr_gx1v7/b.e21.CREDIT_gx1v7_{y}.zarr"
Y0, Y1   = 1980, 2014
STEPS_PER_YEAR = 1460
OUT      = "/glade/derecho/scratch/praggarwal/emu_aice_global.npy"


def main():
    gx_xc, gx_yc, gx_mask = gx1v7_grid(GX1V7_DOMAIN)
    ocean = (gx_mask > 0)
    cice = CiceCoupler(CICE_DIR, "best_model.pt", CICE_IC, ocean_mask=ocean,
                       device="cpu", melth_cap=200.0, lat=gx_yc, ramp_days=20.0,
                       virtual_ice=True)   # match the deployed --ice_virtual run
    nj, ni = gx_mask.shape
    years = list(range(Y0, Y1 + 1))
    T = len(years) * STEPS_PER_YEAR
    out = np.empty((T, nj, ni), dtype=np.float16)

    g = 0
    for y in years:
        zg = zarr.open(ZARR.format(y=y), mode="r")
        sst_yr = np.asarray(zg["SST"][:])            # (1460, nj, ni), K or degC
        if np.nanmean(sst_yr[np.isfinite(sst_yr)]) > 100.0:
            sst_yr = sst_yr - 273.15                 # K -> degC for the emulator
        sst_yr = np.nan_to_num(sst_yr, nan=-1.8)     # freezing over land/ice fill
        for t in range(STEPS_PER_YEAR):
            # 6-hourly noleap clock starting 1980-01-01 00:00
            # noleap clock (no Feb 29) -> python datetime, matching the server's
            # cesm_ymd_tod_to_dt(ymd,tod) (which the coupler compares as datetime).
            c = cftime.DatetimeNoLeap(y, 1, 1) + timedelta(hours=6 * t)
            now_dt = datetime(c.year, c.month, c.day, c.hour)
            cice.maybe_step(sst_yr[t].astype(np.float64), now_dt)
            out[g] = cice.aice.astype(np.float16)
            g += 1
        am = float(cice.aice[ocean].mean()); ap = float(cice.aice[cice.aice > 0.01].mean() if (cice.aice > 0.01).any() else 0)
        print(f"  {y}: rolled 1460 steps  aice(ocean mean)={am:.4f}  aice(>0.01 mean)={ap:.3f}", flush=True)

    np.save(OUT, out)
    print(f"wrote {OUT}  shape={out.shape} dtype={out.dtype}  "
          f"range[{float(out.min()):.3f},{float(out.max()):.3f}]")


if __name__ == "__main__":
    main()
