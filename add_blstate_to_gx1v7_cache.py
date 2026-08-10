"""
add_blstate_to_gx1v7_cache.py
=============================
Append near-surface ("bottom" model level) atmospheric state at t_now to the
NATIVE gx1v7 memory cache as X_atm.npy of shape (N, 5, 384, 320) — channels
[Ubot, Vbot, Tbot, Qbot, PS]. These are used as extra INPUT channels
(train_unet.py --with_atm_in): the coupled server prescribes the same CREDIT
boundary-layer state at run time, so the network finally sees the synoptic
wind field instead of inferring storms from the (standalone-empty) SST-tendency
channel.

Timing: INPUT timing, not the aux-target timing of add_atm_to_cache.py.
The cache's "now" input channels (SST, ICEFRAC, SOLIN) at sample i are at
t_now = i + memory_lag_steps, i.e. slice(memory_lag_steps, T). The BL state is
concurrent exogenous forcing — same slice. Same row count / alignment as X.npy.

Remap: identical transform to the training zarrs (remap_zarr_gx1v7.py, recovered
from git 5ecfe65^): bilinear RegularToScatter from the 192x288 regular S->N grid
onto gx1v7 xc/yc from domain.ocn.gx1v7.210716.nc.

Bottom level: U/V/T/Q are (time, level, lat, lon) with level ordered
top->surface, so near-surface = isel(level=-1). PS is already 2D (Pa).
Units are raw zarr units (m/s, K, kg/kg, Pa) — matching the runtime
blstate_YYYY.nc files built by build_blstate_forcing.py.

Usage:
    python add_blstate_to_gx1v7_cache.py \
        [--cache_dir /glade/work/praggarwal/couple_cache_gx1v7_mem24h] \
        [--workers 6]
"""

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import xarray as xr
import netCDF4 as nc

ZARR_GLOB = (
    "/glade/derecho/scratch/wchapman/b_credit_runs/"
    "b.e21.CREDIT_climate_branch_1980_????_zmdata_ERA5scaled_zmdata_Qtot.zarr"
)
GX1V7_DOMAIN = (
    "/glade/campaign/cesm/cesmdata/inputdata/share/domains/"
    "domain.ocn.gx1v7.210716.nc"
)
ATM_VARS   = ["Ubot", "Vbot", "Tbot", "Qbot", "PS"]
LEVEL_VARS = {"Ubot": "U", "Vbot": "V", "Tbot": "T", "Qbot": "Q"}  # take level=-1
NJ, NI = 384, 320          # gx1v7 (H, W) as stored in the cache
TCHUNK = 128               # frames remapped per block (bounds peak RAM)


class RegularToScatter:
    """Bilinear remap from regular S->N lat/lon grid onto arbitrary target points.
    Verbatim from remap_zarr_gx1v7.py so X_atm matches the training zarrs exactly."""
    def __init__(self, src_lats, src_lons, dst_lon2d, dst_lat2d):
        nlat = len(src_lats); nlon = len(src_lons)
        flat_lat = np.clip(dst_lat2d.ravel(), src_lats[0], src_lats[-1])
        flat_lon = dst_lon2d.ravel() % 360.0
        i0 = np.clip(np.searchsorted(src_lats, flat_lat, side="right") - 1, 0, nlat - 2)
        i1 = i0 + 1
        j0 = np.clip(np.searchsorted(src_lons, flat_lon, side="right") - 1, 0, nlon - 1)
        j1 = (j0 + 1) % nlon
        lon_right = np.where(j0 < nlon - 1, src_lons[j1], src_lons[0] + 360.0)
        dlat = src_lats[i1] - src_lats[i0]
        dlon = lon_right - src_lons[j0]
        a = np.clip((flat_lat - src_lats[i0]) / np.where(dlat == 0, 1.0, dlat), 0.0, 1.0)
        b = np.clip((flat_lon - src_lons[j0]) / np.where(dlon == 0, 1.0, dlon), 0.0, 1.0)
        self.i0 = i0; self.i1 = i1; self.j0 = j0; self.j1 = j1
        self.w00 = (1 - a) * (1 - b); self.w01 = (1 - a) * b
        self.w10 = a * (1 - b);       self.w11 = a * b
        self.shape_out = dst_lon2d.shape

    def batch(self, fields):
        """fields: (T, H, W) → (T, NJ, NI)"""
        out = (self.w00 * fields[:, self.i0, self.j0] + self.w01 * fields[:, self.i0, self.j1]
             + self.w10 * fields[:, self.i1, self.j0] + self.w11 * fields[:, self.i1, self.j1])
        return out.reshape((fields.shape[0],) + self.shape_out)


# Globals inherited by fork()ed workers (weights built once in the parent).
_REMAP = None
_OUT_PATH = None
_MEM_LAG = None


def _build_year(job):
    """Read one year's BL state, remap to gx1v7, write rows into X_atm.npy."""
    zp, offset, n = job
    t0 = time.time()
    ds = xr.open_zarr(zp, consolidated=False)
    T = len(ds["time"])
    now = slice(_MEM_LAG, T)
    Xa = np.lib.format.open_memmap(_OUT_PATH, mode="r+")
    for ci, v in enumerate(ATM_VARS):
        if v in LEVEL_VARS:
            arr = ds[LEVEL_VARS[v]].isel(level=-1).values.astype(np.float32)
        else:
            arr = ds[v].values.astype(np.float32)
        arr = np.where(np.isfinite(arr), arr, 0.0)[now]      # (n, 192, 288)
        for s in range(0, n, TCHUNK):
            e = min(s + TCHUNK, n)
            Xa[offset + s:offset + e, ci] = _REMAP.batch(arr[s:e]).astype(np.float32)
    ds.close()
    del Xa
    return Path(zp).name, n, time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="/glade/work/praggarwal/couple_cache_gx1v7_mem24h")
    parser.add_argument("--zarr_glob", default=ZARR_GLOB)
    parser.add_argument("--domain",    default=GX1V7_DOMAIN)
    parser.add_argument("--workers",   type=int, default=6)
    args = parser.parse_args()

    global _REMAP, _OUT_PATH, _MEM_LAG

    cache_dir = Path(args.cache_dir)
    meta = json.load(open(cache_dir / "meta.json"))
    N = meta["n_samples"]
    _MEM_LAG = meta.get("memory_lag_steps", 0)
    if not meta.get("memory", False) or meta.get("daily", True) or _MEM_LAG < 1:
        raise SystemExit("Expected a 6-hourly memory cache with memory_lag_steps >= 1.")
    if (meta["H"], meta["W"]) != (NJ, NI):
        raise SystemExit(f"Cache grid {meta['H']}x{meta['W']} != gx1v7 {NJ}x{NI}.")

    # Single-writer lock: the build may be submitted on both Derecho and Casper
    # as a queue-wait hedge; whichever starts first takes the lock, the other
    # exits. The lock is deliberately left behind on success (a finished build
    # must not be re-run over); delete X_atm.npy.lock manually to force a rebuild.
    lock = cache_dir / "X_atm.npy.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.uname().nodename} pid={os.getpid()} {time.ctime()}\n".encode())
        os.close(fd)
    except FileExistsError:
        print(f"Lock {lock} exists ({lock.read_text().strip()}) — "
              f"another build ran/is running. Exiting.")
        sys.exit(0)

    zarr_paths = sorted(glob.glob(args.zarr_glob))
    if not zarr_paths:
        raise SystemExit(f"No zarr stores matching: {args.zarr_glob}")
    print(f"Found {len(zarr_paths)} zarr stores")
    print(f"Cache: {N} samples, memory_lag_steps={_MEM_LAG}, vars={ATM_VARS} (BL state at t_now)")

    # Pass 1: count, to confirm alignment with the existing cache before writing.
    counts = []
    for zp in zarr_paths:
        ds = xr.open_zarr(zp, consolidated=False)
        counts.append(len(ds["time"]) - _MEM_LAG)
        ds.close()
    total = sum(counts)
    assert total == N, f"Sample-count mismatch: {total} (blstate) vs {N} (cache). Aborting."

    print("Building remap weights (identical to remap_zarr_gx1v7.py) ...")
    src_lats = np.linspace(-90.0,  90.0, 192)
    src_lons = np.linspace(  0.0, 360.0, 288, endpoint=False)
    with nc.Dataset(args.domain, "r") as dom:
        xc = dom["xc"][:].data.astype(np.float64)   # (384, 320)
        yc = dom["yc"][:].data.astype(np.float64)
    _REMAP = RegularToScatter(src_lats, src_lons, xc, yc)
    print(f"  Remap ready: 192x288 -> {_REMAP.shape_out}")

    out_path = cache_dir / "X_atm.npy"
    _OUT_PATH = str(out_path)
    Xa = np.lib.format.open_memmap(out_path, mode="w+",
                                   dtype=np.float32, shape=(N, len(ATM_VARS), NJ, NI))
    del Xa  # header written; workers reopen r+

    jobs, offset = [], 0
    for zp, n in zip(zarr_paths, counts):
        jobs.append((zp, offset, n))
        offset += n

    t0 = time.time()
    done = 0
    with Pool(args.workers) as pool:
        for name, n, dt in pool.imap_unordered(_build_year, jobs):
            done += 1
            print(f"  {done:3d}/{len(jobs)}  {name}  {n} rows  {dt:.0f}s  "
                  f"(elapsed {(time.time()-t0)/60:.1f} min)", flush=True)

    # Record the new channels in the cache metadata (additive key).
    meta["atm_in_vars"] = ATM_VARS
    meta["atm_in_timing"] = "t_now (slice(memory_lag_steps, T)) — input/forcing timing"
    json.dump(meta, open(cache_dir / "meta.json", "w"), indent=2)

    print(f"\nSaved: {out_path}  ({out_path.stat().st_size/1e9:.1f} GB)")
    print(f"Done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
