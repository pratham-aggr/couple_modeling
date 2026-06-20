"""
add_rad_to_cache.py
===================
Append downward surface radiation targets to an existing data cache as
Y_rad.npy of shape (N, 2, H, W) — channels [FSDS_J, FLDS_J] in J m-2 per
6h step. These line up exactly with the cache's Y.npy sample ordering, so
they can be concatenated as 2 extra output channels at training time
(train_unet.py --with_rad).

Currently supports the memory cache (couple_cache_mem24h), which is what the
best temporal-split config uses. Alignment mirrors preprocess_data.py's
load_year_full_memory: sample i → t_now = i + memory_lag_steps, target at t_now.

Usage:
    python add_rad_to_cache.py --cache_dir /glade/work/praggarwal/couple_cache_mem24h
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
RAD_VARS = ["FSDS_J", "FLDS_J"]
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
    lag_steps        = meta["lag_steps"]
    memory           = meta.get("memory", False)
    memory_lag_steps = meta.get("memory_lag_steps", 0)

    if not memory:
        raise SystemExit("add_rad_to_cache only supports the memory cache (memory=true).")
    if daily:
        raise SystemExit("Memory cache is expected to be 6-hourly (daily=false).")

    zarr_paths = sorted(glob.glob(args.zarr_glob))
    if not zarr_paths:
        raise SystemExit(f"No zarr stores matching: {args.zarr_glob}")
    print(f"Found {len(zarr_paths)} zarr stores")
    print(f"Cache: {N} samples, memory_lag_steps={memory_lag_steps}, vars={RAD_VARS}")

    def fill(a):
        return np.where(np.isfinite(a), a, 0.0).astype(np.float32)

    # Pass 1: count, to confirm alignment with the existing cache before writing.
    counts = []
    for zp in zarr_paths:
        ds = xr.open_zarr(zp, consolidated=False)
        counts.append(len(ds["time"]) - memory_lag_steps)
        ds.close()
    total = sum(counts)
    assert total == N, f"Sample-count mismatch: {total} (rad) vs {N} (cache). Aborting."

    out_path = cache_dir / "Y_rad.npy"
    Yr = np.lib.format.open_memmap(out_path, mode="w+",
                                   dtype=np.float32, shape=(N, len(RAD_VARS), H, W))

    t0 = time.time()
    offset = 0
    for i, (zp, n) in enumerate(zip(zarr_paths, counts)):
        ds = xr.open_zarr(zp, consolidated=False)
        T  = len(ds["time"])
        now = slice(memory_lag_steps, T)          # t_now, matches Y.npy
        chans = [fill(ds[v].values)[now] for v in RAD_VARS]
        ds.close()
        Yr[offset:offset+n] = np.stack(chans, axis=1)   # (n, 2, H, W)
        offset += n
        if (i + 1) % 5 == 0 or i == len(zarr_paths) - 1:
            print(f"  {i+1:3d}/{len(zarr_paths)} years  {offset:6d} samples  {time.time()-t0:.0f}s")

    del Yr  # flush
    assert offset == N, f"Wrote {offset} != {N}"
    print(f"\nSaved: {out_path}  ({out_path.stat().st_size/1e9:.1f} GB)")
    print(f"Done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
