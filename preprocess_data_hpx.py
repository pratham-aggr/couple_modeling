"""
preprocess_data_hpx.py
======================
HEALPix-native counterpart of preprocess_data.py (memory mode only).

Identical zarr reading, memory-lag alignment, fill()=NaN->0, ICEFRAC clip, and
RAW units (heat fluxes stay J/m^2 per 6h) — the ONLY difference is that every 2D
field (192,288) is bilinearly regridded to HEALPix NESTED (npix,) via the
precomputed sparse operator in healpix_grid.py.  This is the single, one-time
lat-lon -> HEALPix regrid; everything downstream is HEALPix-native.

Cache layout (<cache_dir>, default couple_cache_hpx64/):
    X.npy          (N, 6, npix)  float32  [SST,ICEFRAC,SOLIN,SST_prev,ICEFRAC_prev,SOLIN_prev]
    Y.npy          (N, 5, npix)  float32  [TAUX,TAUY,SHFLX,LHFLX,QFLX]
    mask.npy       (N, npix)      float32  ocean mask (NN-regridded isfinite(SST))
    normalizer.npz                          base 6->5 stats (train-only via precompute_norm_hpx)
    meta.json                               + nside, npix

Usage:
    python preprocess_data_hpx.py --nside 64
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import xarray as xr
import healpy as hp

from healpix_grid import build_regrid_weights, build_nn_index

ZARR_GLOB = (
    "/glade/derecho/scratch/wchapman/b_credit_runs/"
    "b.e21.CREDIT_climate_branch_1980_????_zmdata_ERA5scaled_zmdata_Qtot.zarr"
)
TARGET_VARS       = ["TAUX", "TAUY", "SHFLX", "LHFLX", "QFLX"]
INPUT_VARS_MEMORY = ["SST", "ICEFRAC", "SOLIN", "SST_prev", "ICEFRAC_prev", "SOLIN_prev"]
MEMORY_LAG_STEPS  = 4   # 4 x 6h = 24h
H, W = 192, 288
LAT = np.linspace(-90.0, 90.0, H)
LON = np.linspace(0.0, 358.75, W)


def regrid_series(arr_thw, idx, w, chunk=365):
    """(T,H,W) lat-lon -> (T,npix) HEALPix via bilinear weights, chunked over T."""
    T = arr_thw.shape[0]
    npix = idx.shape[0]
    out = np.empty((T, npix), dtype=np.float32)
    flat = arr_thw.reshape(T, H * W)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        g = flat[s:e][:, idx]                       # (t, npix, 4)
        out[s:e] = (g * w[None]).sum(axis=2).astype(np.float32)
    return out


def regrid_nn_series(arr_thw, nn_idx):
    """(T,H,W) -> (T,npix) nearest-neighbour (for the 0/1 mask)."""
    T = arr_thw.shape[0]
    return arr_thw.reshape(T, H * W)[:, nn_idx].astype(np.float32)


def load_year_memory_hpx(zarr_path, tgt_vars, idx, w, nn_idx, mem_lag_steps):
    """Memory pairs on HEALPix: X=[state_t, state_{t-lag}] (npix) -> Y=fluxes_t (npix)."""
    ds = xr.open_zarr(zarr_path, consolidated=False)
    T  = len(ds["time"])

    def fill(a): return np.where(np.isfinite(a), a, 0.0)

    sst_raw = ds["SST"].values.astype(np.float32)
    ifrac   = np.clip(ds["ICEFRAC"].values.astype(np.float32), 0, 1)
    solin   = ds["SOLIN"].values.astype(np.float32)
    tgts    = {v: ds[v].values.astype(np.float32) for v in tgt_vars}
    ds.close()

    # regrid every field to HEALPix (T, npix)
    sst_h   = regrid_series(fill(sst_raw), idx, w)
    ifrac_h = regrid_series(fill(ifrac),   idx, w)
    solin_h = regrid_series(fill(solin),   idx, w)
    tgts_h  = {v: regrid_series(fill(tgts[v]), idx, w) for v in tgt_vars}
    ocean_h = regrid_nn_series(np.isfinite(sst_raw).astype(np.float32), nn_idx)

    now  = slice(mem_lag_steps, T)
    prev = slice(0, T - mem_lag_steps)

    X = np.stack([
        sst_h[now],   ifrac_h[now],   solin_h[now],
        sst_h[prev],  ifrac_h[prev],  solin_h[prev],
    ], axis=1)                                         # (n_pairs, 6, npix)
    Y    = np.stack([tgts_h[v][now] for v in tgt_vars], axis=1)  # (n_pairs, 5, npix)
    mask = ocean_h[now]                                          # (n_pairs, npix)
    return X, Y, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--memory_lag", type=int, default=24)
    args = ap.parse_args()

    nside = args.nside
    npix  = hp.nside2npix(nside)
    mem_lag_steps = args.memory_lag // 6

    cache_dir = Path(args.cache_dir) if args.cache_dir else \
        Path(f"/glade/work/praggarwal/couple_cache_hpx{nside}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    zarr_paths = sorted(glob.glob(ZARR_GLOB))
    if not zarr_paths:
        raise RuntimeError(f"No zarr stores found: {ZARR_GLOB}")

    ds0 = xr.open_zarr(zarr_paths[0], consolidated=False)
    tgt_vars = [v for v in TARGET_VARS if v in ds0.data_vars]
    ds0.close()
    n_out = len(tgt_vars)

    print(f"HEALPix nside={nside}  npix={npix}")
    print(f"Found {len(zarr_paths)} zarr stores")
    print(f"Input vars:  {INPUT_VARS_MEMORY}")
    print(f"Target vars: {tgt_vars}")
    print(f"Output:      {cache_dir}")

    # build regrid operator once
    idx, w = build_regrid_weights(nside, LAT, LON)
    nn_idx = build_nn_index(nside, LAT, LON)

    t0 = time.time()

    # Pass 1: count
    print("Pass 1: counting samples ...")
    counts = []
    for zp in zarr_paths:
        ds = xr.open_zarr(zp, consolidated=False)
        counts.append(len(ds["time"]) - mem_lag_steps)
        ds.close()
    N = sum(counts)
    print(f"Total: {N} samples  "
          f"({N * (6 + n_out + 1) * npix * 4 / 1e9:.1f} GB on disk)")

    # Pre-allocate memmaps
    print(f"Allocating memmap files in {cache_dir} ...")
    X_mm    = np.lib.format.open_memmap(cache_dir / "X.npy",    mode="w+",
                                        dtype=np.float32, shape=(N, 6,     npix))
    Y_mm    = np.lib.format.open_memmap(cache_dir / "Y.npy",    mode="w+",
                                        dtype=np.float32, shape=(N, n_out, npix))
    mask_mm = np.lib.format.open_memmap(cache_dir / "mask.npy", mode="w+",
                                        dtype=np.float32, shape=(N,        npix))

    # Pass 2: fill
    print("Pass 2: filling cache ...")
    offset = 0
    for i, (zp, n) in enumerate(zip(zarr_paths, counts)):
        X, Y, mask = load_year_memory_hpx(zp, tgt_vars, idx, w, nn_idx, mem_lag_steps)
        X_mm   [offset:offset+n] = X
        Y_mm   [offset:offset+n] = Y
        mask_mm[offset:offset+n] = mask
        offset += n
        if (i + 1) % 5 == 0 or i == len(zarr_paths) - 1:
            print(f"  {i+1:3d}/{len(zarr_paths)} years  {offset:6d} samples  {time.time()-t0:.0f}s",
                  flush=True)

    del X_mm, Y_mm, mask_mm

    # Provisional normaliser over ALL samples (precompute_norm_hpx.py recomputes
    # train-only stats with dsst/rad/precip before training).
    print("Computing provisional normalisation stats ...")
    X_r = np.load(cache_dir / "X.npy", mmap_mode="r")
    Y_r = np.load(cache_dir / "Y.npy", mmap_mode="r")
    chunk = 1000

    def chan_stats(arr):
        C = arr.shape[1]
        s1 = np.zeros(C); s2 = np.zeros(C); n_pts = 0
        for s in range(0, len(arr), chunk):
            a = arr[s:s+chunk].astype(np.float64)
            s1 += a.sum(axis=(0, 2)); s2 += (a**2).sum(axis=(0, 2))
            n_pts += a.shape[0] * arr.shape[2]
        mean = (s1 / n_pts).astype(np.float32)
        std  = np.sqrt(np.maximum(s2/n_pts - (s1/n_pts)**2, 0)).astype(np.float32)
        return mean, std

    x_mean, x_std = chan_stats(X_r)
    y_mean, y_std = chan_stats(Y_r)
    for i, v in enumerate(INPUT_VARS_MEMORY):
        print(f"  {v:15s}: mean={x_mean[i]:.3f}  std={x_std[i]:.3f}")
    for i, v in enumerate(tgt_vars):
        print(f"  {v:15s}: mean={y_mean[i]:.4e}  std={y_std[i]:.4e}")

    np.savez(cache_dir / "normalizer.npz",
             x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    json.dump({
        "n_samples":        N,
        "input_vars":       INPUT_VARS_MEMORY,
        "target_vars":      tgt_vars,
        "lag_hours":        0,
        "lag_steps":        0,
        "daily":            False,
        "memory":           True,
        "memory_lag_steps": mem_lag_steps,
        "nside":            nside,
        "npix":             npix,
        "H": H, "W": W,
        "zarr_glob":        ZARR_GLOB,
    }, open(cache_dir / "meta.json", "w"), indent=2)

    print(f"Done in {(time.time()-t0)/60:.1f} min")
    print(f"Cache: {cache_dir}/  "
          f"({sum(f.stat().st_size for f in cache_dir.iterdir())/1e9:.1f} GB total)")


if __name__ == "__main__":
    main()
