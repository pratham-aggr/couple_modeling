"""
add_co2_to_cache.py
===================
Append co2vmr to an existing data cache as co2.npy.

Works for both daily-mean and 6-hourly caches.
Reads lag/daily mode from the cache's meta.json automatically.

Usage:
    python add_co2_to_cache.py --cache_dir /glade/work/praggarwal/couple_cache
    python add_co2_to_cache.py --cache_dir /glade/work/praggarwal/couple_cache_lag0h
    python add_co2_to_cache.py --cache_dir /glade/work/praggarwal/couple_cache_lag12h
    python add_co2_to_cache.py --cache_dir /glade/work/praggarwal/couple_cache_lag24h
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import xarray as xr

ZARR_GLOB = (
    "/glade/derecho/scratch/wchapman/b_credit_runs/"
    "b.e21.CREDIT_climate_branch_1980_????_zmdata_ERA5scaled_zmdata_Qtot.zarr"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="/glade/work/praggarwal/couple_cache")
    args = parser.parse_args()

    cache_dir        = Path(args.cache_dir)
    meta             = json.load(open(cache_dir / "meta.json"))
    N                = meta["n_samples"]
    daily            = meta["daily"]
    lag_steps        = meta["lag_steps"]
    memory           = meta.get("memory", False)
    memory_lag_steps = meta.get("memory_lag_steps", 0)

    zarr_paths = sorted(glob.glob(ZARR_GLOB))
    print(f"Found {len(zarr_paths)} zarr stores")
    print(f"Cache: {N} samples, daily={daily}, lag_steps={lag_steps}, memory={memory}")

    co2_all = []
    t0 = time.time()
    for i, zp in enumerate(zarr_paths):
        ds  = xr.open_zarr(zp, consolidated=False)
        T   = len(ds["time"])
        co2 = ds["co2vmr"].values.astype(np.float32)  # (T,) — 6h steps
        ds.close()

        if daily:
            n_days  = T // 4
            co2_t   = co2[:n_days*4].reshape(n_days, 4).mean(axis=1)  # (n_days,)
            n_steps = n_days
        else:
            co2_t   = co2   # keep 6h resolution
            n_steps = T

        if memory:
            # Sample i uses t_now = i + memory_lag_steps; CO2 is at t_now
            n_pairs = n_steps - memory_lag_steps
            co2_all.append(co2_t[memory_lag_steps:memory_lag_steps + n_pairs])
        else:
            n_pairs = n_steps - lag_steps if lag_steps > 0 else n_steps
            co2_all.append(co2_t[:n_pairs])

        if (i + 1) % 5 == 0 or i == len(zarr_paths) - 1:
            print(f"  {i+1:3d}/{len(zarr_paths)} years  {time.time()-t0:.0f}s")

    co2_arr = np.concatenate(co2_all).astype(np.float32)
    assert len(co2_arr) == N, f"Size mismatch: {len(co2_arr)} vs {N}"

    out = cache_dir / "co2.npy"
    np.save(out, co2_arr)
    print(f"\nCO2 range: {co2_arr.min()*1e6:.1f} – {co2_arr.max()*1e6:.1f} ppm")
    print(f"Saved: {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
