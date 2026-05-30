"""
preprocess_data.py
==================
One-time preprocessing: load all zarr stores, stack into numpy arrays,
and save to a cache directory. Subsequent training runs load from cache
in seconds instead of ~35 minutes.

Cache layout:
    <cache_dir>/X.npy          (N, C, H, W)  float32  — inputs
    <cache_dir>/Y.npy          (N, 5, H, W)  float32  — targets
    <cache_dir>/mask.npy       (N, H, W)     float32  — ocean mask
    <cache_dir>/normalizer.npz               — mean/std computed on full dataset
    <cache_dir>/meta.json                    — variable names, shapes, lag info

Usage:
    # 24-hour daily-mean pairs (original)
    python preprocess_data.py --cache_dir /glade/work/praggarwal/couple_cache

    # 0-hour simultaneous (6-hourly)
    python preprocess_data.py --lag 0 --cache_dir /glade/work/praggarwal/couple_cache_lag0h

    # 12-hour ahead (6-hourly)
    python preprocess_data.py --lag 12 --cache_dir /glade/work/praggarwal/couple_cache_lag12h

    # Memory experiment: current state + 24h-ago state → current fluxes
    python preprocess_data.py --memory
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
TARGET_VARS       = ["TAUX", "TAUY", "SHFLX", "LHFLX", "QFLX"]
INPUT_VARS        = ["SST", "ICEFRAC", "SOLIN"]
INPUT_VARS_MEMORY = ["SST", "ICEFRAC", "SOLIN", "SST_prev", "ICEFRAC_prev", "SOLIN_prev"]
MEMORY_LAG_STEPS  = 4   # 4 × 6h = 24h
H, W = 192, 288


def load_year_full(zarr_path, tgt_vars, daily: bool, lag_steps: int):
    """Load all pairs from one year (no subsampling).

    daily=True:  average 4×6h → 1 daily mean, then shift by lag_steps days
    daily=False: keep raw 6h timesteps, shift by lag_steps 6h-steps
    lag_steps=0: X[t] and Y[t] are the same timestep (simultaneous)
    """
    ds = xr.open_zarr(zarr_path, consolidated=False)
    T  = len(ds["time"])

    def fill(a): return np.where(np.isfinite(a), a, 0.0)

    sst_raw = ds["SST"].values.astype(np.float32)
    ifrac   = np.clip(ds["ICEFRAC"].values.astype(np.float32), 0, 1)
    solin   = ds["SOLIN"].values.astype(np.float32)
    tgts    = {v: ds[v].values.astype(np.float32) for v in tgt_vars}
    ds.close()

    if daily:
        n_days = T // 4
        def dmean(a):
            return a[:n_days*4].reshape(n_days, 4, H, W).mean(axis=1)
        sst_t   = dmean(fill(sst_raw))
        ifrac_t = dmean(fill(ifrac))
        solin_t = dmean(fill(solin))
        tgts_t  = {v: dmean(fill(tgts[v])) for v in tgt_vars}
        ocean_t = dmean(np.isfinite(sst_raw).astype(np.float32)) > 0.5
        n_steps = n_days
    else:
        sst_t   = fill(sst_raw)
        ifrac_t = fill(ifrac)
        solin_t = fill(solin)
        tgts_t  = {v: fill(tgts[v]) for v in tgt_vars}
        ocean_t = np.isfinite(sst_raw).astype(np.float32)
        n_steps = T

    n_pairs = n_steps - lag_steps if lag_steps > 0 else n_steps
    X    = np.stack([sst_t[:n_pairs], ifrac_t[:n_pairs], solin_t[:n_pairs]], axis=1)
    Y    = np.stack([tgts_t[v][lag_steps:n_pairs + lag_steps] for v in tgt_vars], axis=1)
    mask = ocean_t[:n_pairs].astype(np.float32)
    return X, Y, mask


def load_year_full_memory(zarr_path, tgt_vars, memory_lag_steps=MEMORY_LAG_STEPS):
    """Build pairs for the memory experiment: X = [state_t, state_{t-lag}] → Y = fluxes_t.

    X stores all 6 channels: SST[t], ICEFRAC[t], SOLIN[t], SST[t-lag], ICEFRAC[t-lag], SOLIN[t-lag].
    Training selects a subset of channels via --prev_solin.
    """
    ds = xr.open_zarr(zarr_path, consolidated=False)
    T  = len(ds["time"])

    def fill(a): return np.where(np.isfinite(a), a, 0.0)

    sst_raw = ds["SST"].values.astype(np.float32)
    ifrac   = np.clip(ds["ICEFRAC"].values.astype(np.float32), 0, 1)
    solin   = ds["SOLIN"].values.astype(np.float32)
    tgts    = {v: ds[v].values.astype(np.float32) for v in tgt_vars}
    ds.close()

    sst_t   = fill(sst_raw)
    ifrac_t = fill(ifrac)
    solin_t = fill(solin)
    tgts_t  = {v: fill(tgts[v]) for v in tgt_vars}
    ocean_t = np.isfinite(sst_raw).astype(np.float32)

    # Sample i: t_now = i + memory_lag_steps,  t_prev = i
    n_pairs = T - memory_lag_steps
    now  = slice(memory_lag_steps, T)
    prev = slice(0, T - memory_lag_steps)

    X = np.stack([
        sst_t[now],   ifrac_t[now],   solin_t[now],
        sst_t[prev],  ifrac_t[prev],  solin_t[prev],
    ], axis=1)  # (n_pairs, 6, H, W)
    Y    = np.stack([tgts_t[v][now] for v in tgt_vars], axis=1)  # (n_pairs, n_out, H, W)
    mask = ocean_t[now].astype(np.float32)                        # (n_pairs, H, W)
    return X, Y, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default=None,
                        help="Output directory for cache files. "
                             "Defaults to couple_cache (lag=24), couple_cache_lag<N>h, "
                             "or couple_cache_mem24h (--memory).")
    parser.add_argument("--lag", type=int, default=24,
                        help="Prediction lag in hours (default 24). "
                             "0 = simultaneous, 12 = 12h ahead. "
                             "Values other than 24 use 6-hourly resolution.")
    parser.add_argument("--sixhour", action="store_true",
                        help="Force 6-hourly resolution even for --lag 24.")
    parser.add_argument("--memory", action="store_true",
                        help="Memory experiment: X = [state_t, state_{t-24h}] → Y = fluxes_t. "
                             "Stores all 6 channels; training selects subset via --prev_solin.")
    args = parser.parse_args()

    memory    = args.memory
    daily     = (not args.sixhour and args.lag == 24) if not memory else False
    lag_steps = (args.lag // 6 if not daily else 1)   if not memory else 0

    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    elif memory:
        cache_dir = Path("/glade/work/praggarwal/couple_cache_mem24h")
    elif daily:
        cache_dir = Path("/glade/work/praggarwal/couple_cache")
    else:
        cache_dir = Path(f"/glade/work/praggarwal/couple_cache_lag{args.lag}h")

    cache_dir.mkdir(parents=True, exist_ok=True)

    zarr_paths = sorted(glob.glob(ZARR_GLOB))
    if not zarr_paths:
        raise RuntimeError(f"No zarr stores found: {ZARR_GLOB}")

    ds0 = xr.open_zarr(zarr_paths[0], consolidated=False)
    tgt_vars = [v for v in TARGET_VARS if v in ds0.data_vars]
    ds0.close()

    input_vars_list = INPUT_VARS_MEMORY if memory else INPUT_VARS
    n_x_channels    = len(input_vars_list)

    print(f"Found {len(zarr_paths)} zarr stores")
    if memory:
        print(f"Mode: memory experiment  (current + 24h-ago state → current fluxes)")
    else:
        print(f"Mode: {'daily means' if daily else '6-hourly'}, lag={args.lag}h ({lag_steps} steps)")
    print(f"Input vars:  {input_vars_list}")
    print(f"Target vars: {tgt_vars}")
    print(f"Output:      {cache_dir}")

    t0 = time.time()

    # --- Pass 1: count total samples without loading data into RAM ---
    print("Pass 1: counting samples ...")
    counts = []
    for zp in zarr_paths:
        ds = xr.open_zarr(zp, consolidated=False)
        T  = len(ds["time"])
        ds.close()
        if memory:
            n_pairs = T - MEMORY_LAG_STEPS
        else:
            n_steps = T // 4 if daily else T
            n_pairs = n_steps - lag_steps if lag_steps > 0 else n_steps
        counts.append(n_pairs)
    N = sum(counts)
    n_out = len(tgt_vars)
    print(f"Total: {N} samples  ({N * (n_x_channels + n_out + 1) * H * W * 4 / 1e9:.1f} GB on disk)")

    # --- Pre-allocate memmap files ---
    print(f"Allocating memmap files in {cache_dir} ...")
    X_mm    = np.lib.format.open_memmap(cache_dir / "X.npy",    mode="w+",
                                         dtype=np.float32, shape=(N, n_x_channels, H, W))
    Y_mm    = np.lib.format.open_memmap(cache_dir / "Y.npy",    mode="w+",
                                         dtype=np.float32, shape=(N, n_out,         H, W))
    mask_mm = np.lib.format.open_memmap(cache_dir / "mask.npy", mode="w+",
                                         dtype=np.float32, shape=(N,                H, W))

    # --- Pass 2: fill memmaps year by year (O(year) RAM) ---
    print("Pass 2: filling cache ...")
    offset = 0
    for i, (zp, n) in enumerate(zip(zarr_paths, counts)):
        if memory:
            X, Y, mask = load_year_full_memory(zp, tgt_vars, MEMORY_LAG_STEPS)
        else:
            X, Y, mask = load_year_full(zp, tgt_vars, daily, lag_steps)
        X_mm   [offset:offset+n] = X
        Y_mm   [offset:offset+n] = Y
        mask_mm[offset:offset+n] = mask
        offset += n
        if (i + 1) % 5 == 0 or i == len(zarr_paths) - 1:
            print(f"  {i+1:3d}/{len(zarr_paths)} years  {offset:6d} samples  {time.time()-t0:.0f}s")

    # Flush to disk before computing stats
    del X_mm, Y_mm, mask_mm

    # --- Compute normalisation stats via chunked reads (low RAM) ---
    print("Computing normalisation stats ...")
    X_r = np.load(cache_dir / "X.npy",    mmap_mode="r")
    Y_r = np.load(cache_dir / "Y.npy",    mmap_mode="r")
    chunk = 1000

    def chan_stats(arr):
        C = arr.shape[1]
        s1 = np.zeros(C, dtype=np.float64)
        s2 = np.zeros(C, dtype=np.float64)
        n_pts = 0
        for start in range(0, len(arr), chunk):
            a = arr[start:start+chunk].astype(np.float64)
            s1 += a.sum(axis=(0, 2, 3))
            s2 += (a ** 2).sum(axis=(0, 2, 3))
            n_pts += a.shape[0] * H * W
        mean = (s1 / n_pts).astype(np.float32)
        std  = np.sqrt(np.maximum(s2 / n_pts - (s1 / n_pts) ** 2, 0)).astype(np.float32)
        return mean, std

    x_mean, x_std = chan_stats(X_r)
    y_mean, y_std = chan_stats(Y_r)

    for i, v in enumerate(input_vars_list):
        print(f"  {v:15s}: mean={x_mean[i]:.3f}  std={x_std[i]:.3f}")
    for i, v in enumerate(tgt_vars):
        print(f"  {v:15s}: mean={y_mean[i]:.4e}  std={y_std[i]:.4e}")

    np.savez(cache_dir / "normalizer.npz",
             x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    json.dump({
        "n_samples":        N,
        "input_vars":       input_vars_list,
        "target_vars":      tgt_vars,
        "lag_hours":        args.lag if not memory else 0,
        "lag_steps":        lag_steps,
        "daily":            daily,
        "memory":           memory,
        "memory_lag_steps": MEMORY_LAG_STEPS if memory else 0,
        "H": H, "W": W,
        "zarr_glob":        ZARR_GLOB,
    }, open(cache_dir / "meta.json", "w"), indent=2)

    print(f"Done in {(time.time()-t0)/60:.1f} min")
    print(f"Cache: {cache_dir}/  ({sum(f.stat().st_size for f in cache_dir.iterdir())/1e9:.1f} GB total)")


if __name__ == "__main__":
    main()
