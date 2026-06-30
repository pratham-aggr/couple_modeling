"""
build_cache_gx1v7.py
====================
One-shot cache builder for gx1v7 training data (384×320 native ocean grid).

Reads the remapped zarr stores from zarr_gx1v7/ and writes flat numpy memmaps
to couple_cache_gx1v7_mem24h/ so that train_unet.py loads in seconds instead
of ~7 minutes.

Cache layout (mirrors couple_cache_mem24h exactly, just H=384 W=320):
    X.npy          (N, 6, 384, 320)  float32  [SST, ICEFRAC, SOLIN, *_prev]
    Y.npy          (N, 5, 384, 320)  float32  [TAUX, TAUY, SHFLX, LHFLX, QFLX]
    mask.npy       (N,    384, 320)  float32  ocean mask
    Y_rad.npy      (N, 2, 384, 320)  float32  [FSDS_J, FLDS_J]
    Y_precip.npy   (N, 1, 384, 320)  float32  [PRECT]
    normalizer.npz                   mean/std computed on training years (2010-)
    meta.json                        provenance

All arrays share the same sample index: sample i corresponds to
  t_now  = i + 4   (current 6h step)
  t_prev = i       (24h ago)
within each year's zarr (memory_lag_steps=4).

Usage:
    python build_cache_gx1v7.py
    python build_cache_gx1v7.py --cache_dir /path/to/output
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import xarray as xr

ZARR_GLOB = (
    "/glade/derecho/scratch/praggarwal/zarr_gx1v7/"
    "b.e21.CREDIT_gx1v7_????.zarr"
)
CACHE_DIR       = "/glade/work/praggarwal/couple_cache_gx1v7_mem24h"
MEMORY_LAG      = 4                            # steps (4 × 6h = 24h)
TARGET_VARS     = ["TAUX", "TAUY", "SHFLX", "LHFLX", "QFLX"]
INPUT_VARS      = ["SST", "ICEFRAC", "SOLIN", "SST_prev", "ICEFRAC_prev", "SOLIN_prev"]
RAD_VARS        = ["FSDS_J", "FLDS_J"]
PRECIP_VARS     = ["PRECT"]
H, W            = 384, 320


def fill(a):
    return np.where(np.isfinite(a), a, 0.0).astype(np.float32)


def load_year(zarr_path, tgt_vars, rad_vars, precip_vars):
    """Return (X, Y, mask, Y_rad, Y_precip) for one year."""
    ds   = xr.open_zarr(zarr_path, consolidated=False)
    T    = len(ds["time"])
    now  = slice(MEMORY_LAG, T)
    prev = slice(0, T - MEMORY_LAG)

    sst_raw = ds["SST"].values.astype(np.float32)
    sst_f   = fill(sst_raw)
    ifrac_f = np.clip(fill(ds["ICEFRAC"].values.astype(np.float32)), 0.0, 1.0)
    solin_f = fill(ds["SOLIN"].values.astype(np.float32))

    tgts = {v: fill(ds[v].values.astype(np.float32)) for v in tgt_vars}
    rads = {v: fill(ds[v].values.astype(np.float32)) for v in rad_vars}
    prec = {v: fill(ds[v].values.astype(np.float32)) for v in precip_vars}
    ds.close()

    X = np.stack([
        sst_f[now], ifrac_f[now], solin_f[now],
        sst_f[prev], ifrac_f[prev], solin_f[prev],
    ], axis=1).astype(np.float32)                          # (n, 6, H, W)

    Y     = np.stack([tgts[v][now] for v in tgt_vars], axis=1).astype(np.float32)
    mask  = np.isfinite(sst_raw[now]).astype(np.float32)  # (n, H, W)
    Y_rad = np.stack([rads[v][now] for v in rad_vars],    axis=1).astype(np.float32)
    Y_pre = np.stack([prec[v][now] for v in precip_vars], axis=1).astype(np.float32)

    return X, Y, mask, Y_rad, Y_pre


def chan_stats_chunked(arr, chunk=500):
    """Compute per-channel mean/std over (N, C, H, W) memmap without full RAM load."""
    C    = arr.shape[1]
    s1   = np.zeros(C, dtype=np.float64)
    s2   = np.zeros(C, dtype=np.float64)
    npts = 0
    for start in range(0, len(arr), chunk):
        a    = arr[start:start+chunk].astype(np.float64)
        s1  += a.sum(axis=(0, 2, 3))
        s2  += (a ** 2).sum(axis=(0, 2, 3))
        npts += a.shape[0] * H * W
    mean = (s1 / npts).astype(np.float32)
    std  = np.sqrt(np.maximum(s2 / npts - (s1 / npts) ** 2, 0)).astype(np.float32)
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default=CACHE_DIR)
    parser.add_argument("--zarr_glob", default=ZARR_GLOB)
    args = parser.parse_args()

    cache_dir  = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    zarr_paths = sorted(glob.glob(args.zarr_glob))
    if not zarr_paths:
        raise RuntimeError(f"No zarr stores found: {args.zarr_glob}")
    print(f"Found {len(zarr_paths)} zarr stores")
    print(f"Grid: H={H} W={W}   memory_lag={MEMORY_LAG} steps (24h)")
    print(f"Output: {cache_dir}")

    # --- Pass 1: count samples ---
    print("\nPass 1: counting samples ...")
    counts = []
    for zp in zarr_paths:
        ds = xr.open_zarr(zp, consolidated=False)
        counts.append(len(ds["time"]) - MEMORY_LAG)
        ds.close()
    N     = sum(counts)
    n_out = len(TARGET_VARS)
    disk_gb = N * (6 + n_out + 1 + len(RAD_VARS) + len(PRECIP_VARS)) * H * W * 4 / 1e9
    print(f"Total: {N} samples  (approx {disk_gb:.1f} GB on disk)")

    # --- Pre-allocate memmaps ---
    print("\nAllocating memmaps ...")
    X_mm    = np.lib.format.open_memmap(cache_dir/"X.npy",       mode="w+",
                                         dtype=np.float32, shape=(N, 6,         H, W))
    Y_mm    = np.lib.format.open_memmap(cache_dir/"Y.npy",       mode="w+",
                                         dtype=np.float32, shape=(N, n_out,     H, W))
    mask_mm = np.lib.format.open_memmap(cache_dir/"mask.npy",    mode="w+",
                                         dtype=np.float32, shape=(N,            H, W))
    Yr_mm   = np.lib.format.open_memmap(cache_dir/"Y_rad.npy",   mode="w+",
                                         dtype=np.float32, shape=(N, len(RAD_VARS),    H, W))
    Yp_mm   = np.lib.format.open_memmap(cache_dir/"Y_precip.npy",mode="w+",
                                         dtype=np.float32, shape=(N, len(PRECIP_VARS), H, W))

    # --- Pass 2: fill year by year ---
    print("\nPass 2: filling cache (single read per year) ...")
    t0     = time.time()
    offset = 0
    for i, (zp, n) in enumerate(zip(zarr_paths, counts)):
        X, Y, mask, Y_rad, Y_pre = load_year(zp, TARGET_VARS, RAD_VARS, PRECIP_VARS)
        X_mm   [offset:offset+n] = X
        Y_mm   [offset:offset+n] = Y
        mask_mm[offset:offset+n] = mask
        Yr_mm  [offset:offset+n] = Y_rad
        Yp_mm  [offset:offset+n] = Y_pre
        offset += n
        elapsed = time.time() - t0
        print(f"  {i+1:3d}/{len(zarr_paths)} ({Path(zp).name})  "
              f"{offset:6d} samples  {elapsed:.0f}s")

    del X_mm, Y_mm, mask_mm, Yr_mm, Yp_mm   # flush to disk
    print(f"\nAll years written in {(time.time()-t0)/60:.1f} min")

    # --- Pass 3: normalisation stats (chunked, low RAM) ---
    print("\nPass 3: computing normalisation stats ...")
    X_r = np.load(cache_dir / "X.npy", mmap_mode="r")
    Y_r = np.load(cache_dir / "Y.npy", mmap_mode="r")
    x_mean, x_std = chan_stats_chunked(X_r)
    y_mean, y_std = chan_stats_chunked(Y_r)

    print("  Input stats:")
    for j, v in enumerate(INPUT_VARS):
        print(f"    {v:18s}: mean={x_mean[j]:+.4f}  std={x_std[j]:.4f}")
    print("  Target stats:")
    for j, v in enumerate(TARGET_VARS):
        print(f"    {v:18s}: mean={y_mean[j]:+.4e}  std={y_std[j]:.4e}")

    np.savez(cache_dir / "normalizer.npz",
             x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    # --- meta.json ---
    json.dump({
        "n_samples":        N,
        "input_vars":       INPUT_VARS,
        "target_vars":      TARGET_VARS,
        "rad_vars":         RAD_VARS,
        "precip_vars":      PRECIP_VARS,
        "lag_hours":        0,
        "lag_steps":        0,
        "daily":            False,
        "memory":           True,
        "memory_lag_steps": MEMORY_LAG,
        "H": H, "W": W,
        "zarr_glob":        args.zarr_glob,
    }, open(cache_dir / "meta.json", "w"), indent=2)

    total_gb = sum(f.stat().st_size for f in cache_dir.iterdir()) / 1e9
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print(f"Cache: {cache_dir}/  ({total_gb:.1f} GB total)")
    print("Files:", [f.name for f in sorted(cache_dir.iterdir())])


if __name__ == "__main__":
    main()
