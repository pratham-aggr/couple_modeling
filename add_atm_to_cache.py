"""
add_atm_to_cache.py
===================
Append near-surface ("bottom" model level) atmospheric state at t-1 to an
existing memory cache as Y_atm.npy of shape (N, 5, H, W) — channels
[Ubot, Vbot, Tbot, Qbot, PS]. These are used as AUXILIARY prediction targets
(train_unet.py --with_atm), concatenated after the flux/rad/precip targets.

Timing: the flux target at sample i is at t_now = i + memory_lag_steps
(slice(memory_lag_steps, T), mirroring preprocess_data.py / add_rad_to_cache.py).
The atmospheric state is taken ONE 6-hourly step earlier, t_now - 1, i.e.
slice(memory_lag_steps - 1, T - 1). Same row count / alignment as Y.npy, so the
rows line up 1:1 with X.npy / Y.npy / Y_rad.npy / Y_precip.npy.

Bottom level: U/V/T/Q are 4D (time, level, lat, lon) with `level` ordered
top->surface (3.6 hPa ... 992 hPa), so the near-surface level is the LAST index
(isel(level=-1)). PS is already 2D (surface pressure, Pa).

Usage:
    python add_atm_to_cache.py --cache_dir /glade/work/praggarwal/couple_cache_mem24h
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
# Output channel names (cache build order). Ubot/Vbot/Tbot/Qbot are the lowest
# model level of U/V/T/Q; PS is surface pressure (already 2D).
ATM_VARS   = ["Ubot", "Vbot", "Tbot", "Qbot", "PS"]
LEVEL_VARS = {"Ubot": "U", "Vbot": "V", "Tbot": "T", "Qbot": "Q"}  # take level=-1
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
        raise SystemExit("add_atm_to_cache only supports the memory cache (memory=true).")
    if daily:
        raise SystemExit("Memory cache is expected to be 6-hourly (daily=false).")
    if memory_lag_steps < 1:
        raise SystemExit("Need memory_lag_steps >= 1 for a t-1 atm slice.")

    zarr_paths = sorted(glob.glob(args.zarr_glob))
    if not zarr_paths:
        raise SystemExit(f"No zarr stores matching: {args.zarr_glob}")
    print(f"Found {len(zarr_paths)} zarr stores")
    print(f"Cache: {N} samples, memory_lag_steps={memory_lag_steps}, vars={ATM_VARS} (atm at t-1)")

    def fill(a):
        return np.where(np.isfinite(a), a, 0.0).astype(np.float32)

    # Pass 1: count, to confirm alignment with the existing cache before writing.
    counts = []
    for zp in zarr_paths:
        ds = xr.open_zarr(zp, consolidated=False)
        counts.append(len(ds["time"]) - memory_lag_steps)
        ds.close()
    total = sum(counts)
    assert total == N, f"Sample-count mismatch: {total} (atm) vs {N} (cache). Aborting."

    out_path = cache_dir / "Y_atm.npy"
    Ya = np.lib.format.open_memmap(out_path, mode="w+",
                                   dtype=np.float32, shape=(N, len(ATM_VARS), H, W))

    t0 = time.time()
    offset = 0
    for i, (zp, n) in enumerate(zip(zarr_paths, counts)):
        ds = xr.open_zarr(zp, consolidated=False)
        T  = len(ds["time"])
        # atm at t-1 = one 6h step before the flux target's t_now slice(mem_lag, T)
        prev1 = slice(memory_lag_steps - 1, T - 1)
        chans = []
        for v in ATM_VARS:
            if v in LEVEL_VARS:                       # bottom model level (last index)
                arr = ds[LEVEL_VARS[v]].isel(level=-1).values
            else:                                     # PS — already 2D
                arr = ds[v].values
            chans.append(fill(arr)[prev1])
        ds.close()
        Ya[offset:offset+n] = np.stack(chans, axis=1)   # (n, 5, H, W)
        offset += n
        if (i + 1) % 5 == 0 or i == len(zarr_paths) - 1:
            print(f"  {i+1:3d}/{len(zarr_paths)} years  {offset:6d} samples  {time.time()-t0:.0f}s")

    del Ya  # flush
    assert offset == N, f"Wrote {offset} != {N}"
    print(f"\nSaved: {out_path}  ({out_path.stat().st_size/1e9:.1f} GB)")
    print(f"Done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
