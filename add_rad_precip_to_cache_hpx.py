"""
add_rad_precip_to_cache_hpx.py
==============================
Append radiation and precipitation targets to a HEALPix cache, regridded to
HEALPix NESTED (npix,) with the same bilinear operator and the same memory-lag
alignment (sample i -> t_now = i + memory_lag_steps) as preprocess_data_hpx.py.

Writes:
    Y_rad.npy     (N, 2, npix)  float32  [FSDS_J, FLDS_J]   (J m-2 per 6h, raw)
    Y_precip.npy  (N, 1, npix)  float32  [PRECT]            (m per 6h, raw)

Usage:
    python add_rad_precip_to_cache_hpx.py --cache_dir /glade/work/praggarwal/couple_cache_hpx64
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import xarray as xr

from healpix_grid import build_regrid_weights
from preprocess_data_hpx import regrid_series, LAT, LON

ZARR_GLOB = (
    "/glade/derecho/scratch/wchapman/b_credit_runs/"
    "b.e21.CREDIT_climate_branch_1980_????_zmdata_ERA5scaled_zmdata_Qtot.zarr"
)
RAD_VARS    = ["FSDS_J", "FLDS_J"]
PRECIP_VARS = ["PRECT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="/glade/work/praggarwal/couple_cache_hpx64")
    ap.add_argument("--zarr_glob", default=ZARR_GLOB)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    meta      = json.load(open(cache_dir / "meta.json"))
    N         = meta["n_samples"]
    nside     = meta["nside"]
    npix      = meta["npix"]
    mem_lag   = meta["memory_lag_steps"]
    assert meta.get("memory", False), "expects a memory cache"

    idx, w = build_regrid_weights(nside, LAT, LON)

    def fill(a):
        return np.where(np.isfinite(a), a, 0.0).astype(np.float32)

    zarr_paths = sorted(glob.glob(args.zarr_glob))
    counts = []
    for zp in zarr_paths:
        ds = xr.open_zarr(zp, consolidated=False)
        counts.append(len(ds["time"]) - mem_lag)
        ds.close()
    assert sum(counts) == N, f"count mismatch {sum(counts)} vs {N}"
    print(f"HEALPix nside={nside} npix={npix}  {N} samples")

    Yr = np.lib.format.open_memmap(cache_dir / "Y_rad.npy", mode="w+",
                                   dtype=np.float32, shape=(N, len(RAD_VARS), npix))
    Yp = np.lib.format.open_memmap(cache_dir / "Y_precip.npy", mode="w+",
                                   dtype=np.float32, shape=(N, len(PRECIP_VARS), npix))

    t0 = time.time()
    offset = 0
    for i, (zp, n) in enumerate(zip(zarr_paths, counts)):
        ds = xr.open_zarr(zp, consolidated=False)
        T  = len(ds["time"])
        now = slice(mem_lag, T)
        rad = [regrid_series(fill(ds[v].values), idx, w)[now] for v in RAD_VARS]
        pre = [regrid_series(fill(ds[v].values), idx, w)[now] for v in PRECIP_VARS]
        ds.close()
        Yr[offset:offset+n] = np.stack(rad, axis=1)
        Yp[offset:offset+n] = np.stack(pre, axis=1)
        offset += n
        if (i + 1) % 5 == 0 or i == len(zarr_paths) - 1:
            print(f"  {i+1:3d}/{len(zarr_paths)} years  {offset:6d}  {time.time()-t0:.0f}s",
                  flush=True)

    del Yr, Yp
    assert offset == N
    print(f"Saved Y_rad.npy + Y_precip.npy in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
