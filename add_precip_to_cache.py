"""
add_precip_to_cache.py
======================
Append total precipitation to an existing data cache as Y_precip.npy of
shape (N, 1, H, W) — channel [PRECT], liquid-water-equivalent depth in
metres per 6h step (the same per-6h-step accumulation convention as the
energy fluxes; PRECT*1000/21600 -> kg m-2 s-1 for the coupler).

Lines up exactly with the cache's Y.npy / Y_rad.npy sample ordering, so it
can be concatenated as one more output channel at training time
(train_unet.py --with_precip). Mirrors add_rad_to_cache.py.

Usage:
    python add_precip_to_cache.py --cache_dir /glade/work/praggarwal/couple_cache_mem24h
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
PRECIP_VARS = ["PRECT"]
H, W = 192, 288


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="/glade/work/praggarwal/couple_cache_mem24h")
    parser.add_argument("--zarr_glob", default=ZARR_GLOB)
    args = parser.parse_args()

    cache_dir        = Path(args.cache_dir)
    meta             = json.load(open(cache_dir / "meta.json"))
    N                = meta["n_samples"]
    daily            = meta["daily"]
    memory           = meta.get("memory", False)
    memory_lag_steps = meta.get("memory_lag_steps", 0)

    if not memory:
        raise SystemExit("add_precip_to_cache only supports the memory cache (memory=true).")
    if daily:
        raise SystemExit("Memory cache is expected to be 6-hourly (daily=false).")

    zarr_paths = sorted(glob.glob(args.zarr_glob))
    if not zarr_paths:
        raise SystemExit(f"No zarr stores matching: {args.zarr_glob}")
    print(f"Found {len(zarr_paths)} zarr stores")
    print(f"Cache: {N} samples, memory_lag_steps={memory_lag_steps}, vars={PRECIP_VARS}")

    def fill(a):
        return np.where(np.isfinite(a), a, 0.0).astype(np.float32)

    # Pass 1: confirm alignment with the existing cache before writing.
    counts = []
    for zp in zarr_paths:
        ds = xr.open_zarr(zp, consolidated=False)
        counts.append(len(ds["time"]) - memory_lag_steps)
        ds.close()
    total = sum(counts)
    assert total == N, f"Sample-count mismatch: {total} (precip) vs {N} (cache). Aborting."

    out_path = cache_dir / "Y_precip.npy"
    Yp = np.lib.format.open_memmap(out_path, mode="w+",
                                   dtype=np.float32, shape=(N, len(PRECIP_VARS), H, W))

    t0 = time.time()
    offset = 0
    for i, (zp, n) in enumerate(zip(zarr_paths, counts)):
        ds = xr.open_zarr(zp, consolidated=False)
        T  = len(ds["time"])
        now = slice(memory_lag_steps, T)          # t_now, matches Y.npy
        chans = [fill(ds[v].values)[now] for v in PRECIP_VARS]
        ds.close()
        Yp[offset:offset+n] = np.stack(chans, axis=1)   # (n, 1, H, W)
        offset += n
        if (i + 1) % 5 == 0 or i == len(zarr_paths) - 1:
            print(f"  {i+1:3d}/{len(zarr_paths)} years  {offset:6d} samples  {time.time()-t0:.0f}s")

    del Yp  # flush
    assert offset == N, f"Wrote {offset} != {N}"
    print(f"\nSaved: {out_path}  ({out_path.stat().st_size/1e9:.1f} GB)")
    print(f"Done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
