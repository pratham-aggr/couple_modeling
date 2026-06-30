"""
precompute_norm_hpx.py
======================
HEALPix counterpart of precompute_norm.py: channel-wise normaliser from
TRAIN-only samples (years <= train_end_year), for a (N, C, npix) cache.
Identical logic; the only difference is the spatial axis is 1D (npix) so we
sum over axis 1 and count += npix per sample.

Usage:
    python precompute_norm_hpx.py \
        --cache_dir /glade/work/praggarwal/couple_cache_hpx64 \
        --out_dir   ./output/output_hpx64_mem24h \
        --zarr_glob "/glade/.../b.e21...????.zarr" \
        --train_end_year 2010 --mem_channels 0 1 2 3 4 \
        --dsst_dt --with_rad --with_precip
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import xarray as xr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--zarr_glob", required=True)
    ap.add_argument("--train_end_year", type=int, default=2010)
    ap.add_argument("--mem_channels", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--dsst_dt", action="store_true")
    ap.add_argument("--with_rad", action="store_true")
    ap.add_argument("--with_precip", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, use only the first N train samples (quick wiring test).")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_np = np.load(cache_dir / "X.npy", mmap_mode="r")   # (N, 6, npix)
    Y_np = np.load(cache_dir / "Y.npy", mmap_mode="r")   # (N, 5, npix)
    Yrad_np = np.load(cache_dir / "Y_rad.npy", mmap_mode="r") if args.with_rad else None
    Yprecip_np = np.load(cache_dir / "Y_precip.npy", mmap_mode="r") if args.with_precip else None
    N_full = len(X_np)
    npix   = X_np.shape[2]

    meta = json.load(open(cache_dir / "meta.json"))
    mem_lag_steps = meta.get("memory_lag_steps", 4)

    # year label per cache sample (mirrors precompute_norm.py)
    zarr_paths = sorted(glob.glob(args.zarr_glob))
    years_list = []
    for zp in zarr_paths:
        ds = xr.open_zarr(zp, consolidated=False)
        times = ds["time"].values
        ds.close()
        valid_times = times[mem_lag_steps:]
        years_list.extend([int(valid_times[0].year)] * len(valid_times))
    years_cache = np.array(years_list, dtype=np.int32)
    assert len(years_cache) == N_full, f"{len(years_cache)} != {N_full}"

    train_idx = np.where(years_cache <= args.train_end_year)[0]
    if args.limit:
        train_idx = train_idx[:args.limit]
    print(f"npix={npix}  train samples (<= {args.train_end_year}): {len(train_idx)}"
          + (f"  [LIMITED to {args.limit}]" if args.limit else ""))

    chs   = args.mem_channels
    n_in  = len(chs) + (1 if args.dsst_dt else 0)
    n_out = (Y_np.shape[1]
             + (Yrad_np.shape[1]    if Yrad_np    is not None else 0)
             + (Yprecip_np.shape[1] if Yprecip_np is not None else 0))

    x_sum  = np.zeros(n_in);  x_sum2 = np.zeros(n_in)
    y_sum  = np.zeros(n_out); y_sum2 = np.zeros(n_out)
    count  = 0
    t0 = time.time()

    for k, idx in enumerate(train_idx):
        x = X_np[idx][chs].astype(np.float64)            # (n_ch, npix)
        if args.dsst_dt:
            dsst = ((X_np[idx][0] - X_np[idx][3]) / 86400.0)[None]
            x = np.concatenate([x, dsst.astype(np.float64)], axis=0)
        y = Y_np[idx].astype(np.float64)                 # (5, npix)
        if Yrad_np is not None:
            y = np.concatenate([y, Yrad_np[idx].astype(np.float64)], axis=0)
        if Yprecip_np is not None:
            y = np.concatenate([y, Yprecip_np[idx].astype(np.float64)], axis=0)
        x_sum  += x.sum(axis=1);  x_sum2 += (x ** 2).sum(axis=1)
        y_sum  += y.sum(axis=1);  y_sum2 += (y ** 2).sum(axis=1)
        count  += npix
        if (k + 1) % 5000 == 0 or k == len(train_idx) - 1:
            print(f"  {k+1}/{len(train_idx)}  ({time.time()-t0:.0f}s)", flush=True)

    x_mean = (x_sum / count).astype(np.float32)
    x_std  = np.sqrt(np.maximum(x_sum2/count - (x_sum/count)**2, 0)).astype(np.float32)
    y_mean = (y_sum / count).astype(np.float32)
    y_std  = np.sqrt(np.maximum(y_sum2/count - (y_sum/count)**2, 0)).astype(np.float32)

    np.savez(out_dir / "normalizer.npz",
             x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    in_names = ["SST", "ICEFRAC", "SOLIN", "SST_prev", "ICEFRAC_prev", "SOLIN_prev", "dSST_dt"]
    ch_names = [in_names[c] for c in chs] + (["dSST_dt"] if args.dsst_dt else [])
    tgt = ["TAUX", "TAUY", "SHFLX", "LHFLX", "QFLX"] \
        + (["FSDS_J", "FLDS_J"] if args.with_rad else []) \
        + (["PRECT"] if args.with_precip else [])
    print("\nInput stats:")
    for v, m, s in zip(ch_names, x_mean, x_std):
        print(f"  {v:12s}: mean={m:.4e}  std={s:.4e}")
    print("Target stats:")
    for v, m, s in zip(tgt, y_mean, y_std):
        print(f"  {v:12s}: mean={m:.4e}  std={s:.4e}")
    print(f"\nSaved {out_dir/'normalizer.npz'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
