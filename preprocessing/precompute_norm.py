"""
precompute_norm.py
==================
Compute channel-wise normalisation stats from training samples only
(years <= train_end_year) and save normalizer.npz to --out_dir.

Run once before training with --split_mode temporal so training
startup doesn't have to recompute stats from 45k maps.

Usage:
    python precompute_norm.py \
        --cache_dir /glade/work/praggarwal/couple_cache_mem24h \
        --out_dir   ./output/output_unet_mem24h_temporal \
        --zarr_glob "/glade/.../b.e21...????.zarr" \
        --train_end_year 2010 \
        --mem_channels 0 1 2 3 4
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import xarray as xr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir",      required=True)
    parser.add_argument("--out_dir",        required=True)
    parser.add_argument("--zarr_glob",      required=True)
    parser.add_argument("--train_end_year", type=int, default=2010)
    parser.add_argument("--mem_channels",   type=int, nargs="+", default=[0,1,2,3,4],
                        help="Channel indices to select from X cache (e.g. 0 1 2 3 4 for "
                             "SST ICEFRAC SOLIN SST_prev ICEFRAC_prev)")
    parser.add_argument("--dsst_dt", action="store_true",
                        help="Append (SST[t]-SST_prev)/86400 as an extra channel")
    parser.add_argument("--with_rad", action="store_true",
                        help="Include Y_rad.npy (FSDS_J, FLDS_J) as 2 extra output channels.")
    parser.add_argument("--with_precip", action="store_true",
                        help="Include Y_precip.npy (PRECT) as 1 extra output channel (after rad).")
    parser.add_argument("--with_atm", action="store_true",
                        help="Include Y_atm.npy (Ubot,Vbot,Tbot,Qbot,PS at t-1) as 5 extra "
                             "output channels, appended last (after precip).")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load cache (memory-mapped — doesn't pull all data into RAM)
    print(f"Loading cache from {cache_dir} ...")
    X_np    = np.load(cache_dir / "X.npy",    mmap_mode="r")
    Y_np    = np.load(cache_dir / "Y.npy",    mmap_mode="r")
    Yrad_np = None
    if args.with_rad:
        yrad_path = cache_dir / "Y_rad.npy"
        if not yrad_path.exists():
            raise RuntimeError(f"{yrad_path} not found — run add_rad_to_cache.py first")
        Yrad_np = np.load(yrad_path, mmap_mode="r")
        print(f"  Radiation targets enabled (Y_rad.npy, {Yrad_np.shape[1]} channels)")
    Yprecip_np = None
    if args.with_precip:
        yprecip_path = cache_dir / "Y_precip.npy"
        if not yprecip_path.exists():
            raise RuntimeError(f"{yprecip_path} not found — run add_precip_to_cache.py first")
        Yprecip_np = np.load(yprecip_path, mmap_mode="r")
        print(f"  Precipitation target enabled (Y_precip.npy, {Yprecip_np.shape[1]} channels)")
    Yatm_np = None
    if args.with_atm:
        yatm_path = cache_dir / "Y_atm.npy"
        if not yatm_path.exists():
            raise RuntimeError(f"{yatm_path} not found — run add_atm_to_cache.py first")
        Yatm_np = np.load(yatm_path, mmap_mode="r")
        print(f"  Auxiliary atm targets enabled (Y_atm.npy, {Yatm_np.shape[1]} channels)")
    N_full  = len(X_np)
    print(f"  Cache size: {N_full} samples")

    # Load meta to get memory_lag_steps
    meta = json.load(open(cache_dir / "meta.json"))
    mem_lag_steps = meta.get("memory_lag_steps", 4)

    # Build year label for every cache sample from zarr time coordinates
    zarr_paths = sorted(glob.glob(args.zarr_glob))
    if not zarr_paths:
        raise RuntimeError(f"No zarr stores found: {args.zarr_glob}")
    print(f"Building year labels from {len(zarr_paths)} zarr files ...")
    years_list = []
    for zp in zarr_paths:
        ds = xr.open_zarr(zp, consolidated=False)
        times = ds["time"].values
        ds.close()
        valid_times = times[mem_lag_steps:]
        yr = int(valid_times[0].year)
        years_list.extend([yr] * len(valid_times))
    years_cache = np.array(years_list, dtype=np.int32)
    assert len(years_cache) == N_full, \
        f"Year array length {len(years_cache)} != cache size {N_full}"

    # Training indices: years <= train_end_year
    train_mask = years_cache <= args.train_end_year
    train_idx  = np.where(train_mask)[0]
    print(f"  Training samples (years ≤ {args.train_end_year}): {len(train_idx)}")

    # Compute normaliser from training samples only
    t0 = time.time()
    print("Computing normalisation stats ...")
    chs = args.mem_channels

    # Stream through training samples to accumulate sums (avoids loading all at once)
    N_trn = len(train_idx)
    n_in  = len(chs) + (1 if args.dsst_dt else 0)
    n_out = (Y_np.shape[1]
             + (Yrad_np.shape[1]    if Yrad_np    is not None else 0)
             + (Yprecip_np.shape[1] if Yprecip_np is not None else 0)
             + (Yatm_np.shape[1]    if Yatm_np    is not None else 0))
    H, W  = X_np.shape[2], X_np.shape[3]

    x_sum  = np.zeros(n_in,  dtype=np.float64)
    x_sum2 = np.zeros(n_in,  dtype=np.float64)
    y_sum  = np.zeros(n_out, dtype=np.float64)
    y_sum2 = np.zeros(n_out, dtype=np.float64)
    count  = 0

    for k, idx in enumerate(train_idx):
        x = X_np[idx][chs].astype(np.float64)   # (n_ch, H, W)
        if args.dsst_dt:
            dsst = ((X_np[idx][0] - X_np[idx][3]) / 86400.0)[None]
            x = np.concatenate([x, dsst.astype(np.float64)], axis=0)
        y = Y_np[idx].astype(np.float64)         # (5, H, W)
        if Yrad_np is not None:
            y = np.concatenate([y, Yrad_np[idx].astype(np.float64)], axis=0)
        if Yprecip_np is not None:
            y = np.concatenate([y, Yprecip_np[idx].astype(np.float64)], axis=0)
        if Yatm_np is not None:
            y = np.concatenate([y, Yatm_np[idx].astype(np.float64)], axis=0)
        pixels = H * W
        x_sum  += x.sum(axis=(1, 2))
        x_sum2 += (x ** 2).sum(axis=(1, 2))
        y_sum  += y.sum(axis=(1, 2))
        y_sum2 += (y ** 2).sum(axis=(1, 2))
        count  += pixels
        if (k + 1) % 5000 == 0 or k == N_trn - 1:
            print(f"  {k+1}/{N_trn} samples  ({time.time()-t0:.0f}s)")

    x_mean = (x_sum  / count).astype(np.float32)
    x_std  = np.sqrt(np.maximum(x_sum2 / count - x_mean.astype(np.float64) ** 2, 0)).astype(np.float32)
    y_mean = (y_sum  / count).astype(np.float32)
    y_std  = np.sqrt(np.maximum(y_sum2 / count - y_mean.astype(np.float64) ** 2, 0)).astype(np.float32)

    out_path = out_dir / "normalizer.npz"
    np.savez(out_path, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    print(f"\nSaved to {out_path}  ({time.time()-t0:.1f}s total)")

    input_vars = ["SST", "ICEFRAC", "SOLIN", "SST_prev", "ICEFRAC_prev",
                  "SOLIN_prev", "dSST_dt"]
    ch_names = [input_vars[c] for c in chs] + (["dSST_dt"] if args.dsst_dt else [])
    target_vars = ["TAUX", "TAUY", "SHFLX", "LHFLX", "QFLX"]
    if args.with_rad:
        target_vars = target_vars + ["FSDS_J", "FLDS_J"]
    if args.with_precip:
        target_vars = target_vars + ["PRECT"]
    if args.with_atm:
        target_vars = target_vars + ["Ubot", "Vbot", "Tbot", "Qbot", "PS"]
    print("\nInput stats:")
    for v, m, s in zip(ch_names, x_mean, x_std):
        print(f"  {v:12s}: mean={m:.4e}  std={s:.4e}")
    print("Target stats:")
    for v, m, s in zip(target_vars, y_mean, y_std):
        print(f"  {v:12s}: mean={m:.4e}  std={s:.4e}")


if __name__ == "__main__":
    main()
