"""
precompute_norm_gx1v7.py
========================
Precompute training-only normalisation stats for the gx1v7 run and save to
output_unet_gx1v7/normalizer.npz so that train_unet.py skips the chunked
norm computation entirely on every subsequent run.

Training years: all years in the cache EXCEPT val (2011-2012) and test (2013-2014).
Matches the --split_mode temporal --val_years 2011 2012 --test_years 2013 2014 flags.

Usage:
    python precompute_norm_gx1v7.py
"""

import glob
import json
import time
from pathlib import Path

import numpy as np
import xarray as xr

CACHE_DIR  = "/glade/work/praggarwal/couple_cache_gx1v7_mem24h"
ZARR_GLOB  = "/glade/derecho/scratch/praggarwal/zarr_gx1v7/b.e21.CREDIT_gx1v7_????.zarr"
OUT_DIR    = "/glade/u/home/praggarwal/couple/output/output_unet_gx1v7"
VAL_YEARS  = (2011, 2012)
TEST_YEARS = (2013, 2014)
MEM_CHANNELS = [0, 1, 2, 3, 4]   # SST, ICEFRAC, SOLIN, SST_prev, ICEFRAC_prev (no prev_solin)
DSST_DT    = True
MEMORY_LAG = 4                    # steps (matches build_cache_gx1v7.py)
CHUNK      = 256

cache_dir = Path(CACHE_DIR)
out_dir   = Path(OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "normalizer.npz"
if out_path.exists():
    print(f"Already exists: {out_path}  — delete it to recompute.")
    raise SystemExit(0)

# Load memmaps (read-only, no RAM copy)
print("Opening memmaps ...")
X_mm    = np.load(cache_dir / "X.npy",       mmap_mode="r")  # (N, 6, H, W)
Y_mm    = np.load(cache_dir / "Y.npy",       mmap_mode="r")  # (N, 5, H, W)
Yr_mm   = np.load(cache_dir / "Y_rad.npy",   mmap_mode="r")  # (N, 2, H, W)
Yp_mm   = np.load(cache_dir / "Y_precip.npy",mmap_mode="r")  # (N, 1, H, W)
N, _, H, W = X_mm.shape
print(f"  N={N}  H={H}  W={W}")

# Build year array from zarr time coords (same logic as train_unet.py _load_years)
zarr_paths = sorted(glob.glob(ZARR_GLOB))
print(f"Found {len(zarr_paths)} zarr stores for year labels")
years = []
for zp in zarr_paths:
    ds = xr.open_zarr(zp, consolidated=False)
    T  = len(ds["time"])
    tv = ds["time"].values
    ds.close()
    # memory mode: sample i → t_now = i + MEMORY_LAG
    for t_now in range(MEMORY_LAG, T):
        years.append(int(tv[t_now].year))
years = np.array(years, dtype=np.int32)
assert len(years) == N, f"year array len {len(years)} != N {N}"

# Training mask: exclude val and test years
val_mask  = (years >= VAL_YEARS[0])  & (years <= VAL_YEARS[1])
test_mask = (years >= TEST_YEARS[0]) & (years <= TEST_YEARS[1])
train_idx = np.where(~val_mask & ~test_mask)[0]
print(f"Train samples: {len(train_idx)} / {N}  "
      f"(excl val {VAL_YEARS}, test {TEST_YEARS})")

# Chunked stats
C_x = len(MEM_CHANNELS) + (1 if DSST_DT else 0)
C_y = Y_mm.shape[1] + Yr_mm.shape[1] + Yp_mm.shape[1]
s1x = np.zeros(C_x, np.float64); s2x = np.zeros(C_x, np.float64)
s1y = np.zeros(C_y, np.float64); s2y = np.zeros(C_y, np.float64)
npts = 0

print(f"Computing stats over {len(train_idx)} training samples  (chunk={CHUNK}) ...")
t0 = time.time()
for start in range(0, len(train_idx), CHUNK):
    bi  = train_idx[start:start+CHUNK]
    raw = X_mm[bi].astype(np.float64)          # (b, 6, H, W)
    x   = raw[:, MEM_CHANNELS]
    if DSST_DT:
        dsst = ((raw[:, 0] - raw[:, 3]) / 86400.0)[:, None]
        x = np.concatenate([x, dsst], axis=1)
    y = np.concatenate([Y_mm[bi].astype(np.float64),
                        Yr_mm[bi].astype(np.float64),
                        Yp_mm[bi].astype(np.float64)], axis=1)
    s1x += x.sum(axis=(0, 2, 3)); s2x += (x**2).sum(axis=(0, 2, 3))
    s1y += y.sum(axis=(0, 2, 3)); s2y += (y**2).sum(axis=(0, 2, 3))
    npts += len(bi) * H * W
    if (start // CHUNK) % 100 == 0:
        print(f"  {start+len(bi):6d}/{len(train_idx)}  {time.time()-t0:.0f}s")

xm = (s1x/npts).astype(np.float32)
xs = np.sqrt(np.maximum(s2x/npts - (s1x/npts)**2, 0)).astype(np.float32)
ym = (s1y/npts).astype(np.float32)
ys = np.sqrt(np.maximum(s2y/npts - (s1y/npts)**2, 0)).astype(np.float32)

INPUT_VARS  = ["SST", "ICEFRAC", "SOLIN", "SST_prev", "ICEFRAC_prev", "dSST_dt"]
TARGET_VARS = ["TAUX", "TAUY", "SHFLX", "LHFLX", "QFLX", "FSDS_J", "FLDS_J", "PRECT"]
print("\nInput stats:")
for j, v in enumerate(INPUT_VARS):
    print(f"  {v:18s}: mean={xm[j]:+.4f}  std={xs[j]:.4f}")
print("Target stats:")
for j, v in enumerate(TARGET_VARS):
    print(f"  {v:18s}: mean={ym[j]:+.4e}  std={ys[j]:.4e}")

np.savez(out_path, x_mean=xm, x_std=xs, y_mean=ym, y_std=ys)
print(f"\nSaved: {out_path}  ({out_path.stat().st_size/1e3:.0f} KB)")
print(f"Done in {(time.time()-t0)/60:.1f} min")
