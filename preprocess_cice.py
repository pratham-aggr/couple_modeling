"""
preprocess_cice.py
==================
One-time preprocessing for the CICE sea-ice emulator (companion to
preprocess_data.py for MEMO).  Unlike MEMO — which trains on the 6-hourly
CREDIT lat-lon zarr — the sea-ice model trains on the CESM2-LE **monthly**
ice+ocean tseries, NATIVELY on POP/CICE's gx1v7 grid (384 x 320).  That is the
same grid the standalone POP run uses, so the eventual coupling path needs NO
regrid.

Why gx1v7/monthly and not the CREDIT zarr: the CREDIT data is atmospheric and
carries only ICEFRAC (concentration).  A sea-ice model that can supply the
missing air-sea momentum damping (the ice->ocean stress strocnx/strocny that
the standalone blow-up analysis flagged) needs the full ice state, which lives
only in the CICE history — available on gx1v7 for the whole CESM2-LE.

Inputs  (7ch): SST, uatm, vatm, aice_prev, hi_prev, uvel_prev, vvel_prev
Targets (6ch): aice, hi, uvel, vvel, strocnx, strocny
    - SST (ocn tseries, degC) is the ocean state the ice responds to.
    - uatm/vatm (ice tseries, m/s) are the 10 m winds driving ice drift/stress.
    - *_prev are last month's ice state (1-step memory), the analogue of MEMO's
      SST_prev/ICEFRAC_prev lag channels.
    - strocnx/strocny (ice tseries, N/m2) are the ice->ocean stress = the
      momentum sink POP is missing under ice.

Everything is kept in RAW units (like MEMO); the Normalizer handles scaling.

Cache layout (mirrors preprocess_data.py):
    <cache_dir>/X.npy     (N, 7, 384, 320)  float32
    <cache_dir>/Y.npy     (N, 6, 384, 320)  float32
    <cache_dir>/mask.npy  (N, 384, 320)     float32   (1 = ocean)
    <cache_dir>/meta.json                    variable names / shapes / lag

Usage:
    python preprocess_cice.py --cache_dir /glade/derecho/scratch/praggarwal/couple_cache_cice
    python preprocess_cice.py --members LE2-1231.002 LE2-1231.003 --year_end 2014
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import xarray as xr

ICE_ROOT = "/glade/campaign/cesm/collections/CESM2-LE/ice/proc/tseries/month_1"
OCN_ROOT = "/glade/campaign/cesm/collections/CESM2-LE/ocn/proc/tseries/month_1"

# ice-side fields (all on gx1v7, nj=384 x ni=320)
ICE_TARGETS = ["aice", "hi", "uvel", "vvel", "strocnx", "strocny"]
ICE_FORCING = ["uatm", "vatm"]           # atmospheric drivers carried in ice tseries
ICE_STATE   = ["aice", "hi", "uvel", "vvel"]  # channels carried back as *_prev memory
# ocn-side field (input the ice responds to)
OCN_INPUT   = ["SST"]

NJ, NI = 384, 320
MEMORY_LAG_STEPS = 1  # 1 month back (monthly cadence)


def _mf(root, member, var, year_start, year_end):
    """Read all tseries chunks for one (member,var), concatenated along time and
    trimmed to [year_start, year_end].  Returns (values (T,nj,ni) float32, times).

    Loops open_dataset + numpy concat instead of open_mfdataset so we do not
    depend on dask (not installed in the atm env) and keep peak RAM to one
    chunk at a time."""
    paths = sorted(glob.glob(f"{root}/{var}/*{member}*.nc"))
    if not paths:
        raise FileNotFoundError(f"no tseries for {var} member {member} under {root}")
    vals, times = [], []
    for p in paths:
        ds = xr.open_dataset(p, decode_timedelta=False)
        da = ds[var]
        # SST carries a singleton surface z_t; drop any vertical singleton.
        for extra in ("z_t", "z_w", "z_t_150m"):
            if extra in da.dims:
                da = da.isel({extra: 0}, drop=True)
        yrs = da["time"].dt.year
        da = da.sel(time=(yrs >= year_start) & (yrs <= year_end))
        if da["time"].size == 0:
            ds.close(); continue
        vals.append(da.values.astype(np.float32))
        times.append(da["time"].values)
        ds.close()
    if not vals:
        raise ValueError(f"{var} member {member}: no timesteps in [{year_start},{year_end}]")
    return np.concatenate(vals, 0), np.concatenate(times, 0)


def load_member(member, year_start, year_end):
    """Build (X, Y, mask) for one ensemble member.

    Time alignment: ice and ocn tseries share an identical monthly time axis
    (verified: same cftime stamps).  We load the union var set, then apply the
    1-month memory lag by slicing now=slice(1,T) / prev=slice(0,T-1)."""
    def fill(a):
        return np.where(np.isfinite(a), a, 0.0).astype(np.float32)

    t0 = time.time()
    raw   = {v: _mf(ICE_ROOT, member, v, year_start, year_end)
             for v in set(ICE_TARGETS) | set(ICE_FORCING)}
    sst_v, sst_t = _mf(OCN_ROOT, member, "SST", year_start, year_end)

    # align on the shared time axis (guards against a stray missing month)
    tcommon = sst_t
    for v in raw:
        tcommon = np.intersect1d(tcommon, raw[v][1])

    def _pick(vals, times):
        # np.intersect1d returns the sorted common stamps; keep matching rows
        return vals[np.isin(times, tcommon)]

    sst = _pick(sst_v, sst_t)
    ice = {v: _pick(raw[v][0], raw[v][1]) for v in raw}
    T = sst.shape[0]
    print(f"  {member}: {T} months  ({time.time()-t0:.0f}s load)")

    # ocean mask = where SST is defined (land is _FillValue -> NaN)
    ocean = np.isfinite(sst)              # (T, nj, ni) bool
    sstf  = fill(sst)
    icef  = {v: fill(ice[v]) for v in ice}

    now  = slice(MEMORY_LAG_STEPS, T)     # t
    prev = slice(0, T - MEMORY_LAG_STEPS) # t-1

    X = np.stack(
        [sstf[now], icef["uatm"][now], icef["vatm"][now]] +
        [icef[v][prev] for v in ICE_STATE],           # aice_prev,hi_prev,uvel_prev,vvel_prev
        axis=1).astype(np.float32)                    # (N, 7, nj, ni)
    Y = np.stack([icef[v][now] for v in ICE_TARGETS], axis=1).astype(np.float32)  # (N,6,nj,ni)
    mask = ocean[now].astype(np.float32)              # (N, nj, ni)
    return X, Y, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--members", nargs="+", default=["LE2-1231.002"],
                    help="CESM2-LE member id(s), e.g. LE2-1231.002")
    ap.add_argument("--year_start", type=int, default=1850)
    ap.add_argument("--year_end",   type=int, default=2100)
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    Xs, Ys, Ms = [], [], []
    for m in args.members:
        X, Y, mask = load_member(m, args.year_start, args.year_end)
        Xs.append(X); Ys.append(Y); Ms.append(mask)
    X = np.concatenate(Xs, 0); Y = np.concatenate(Ys, 0); mask = np.concatenate(Ms, 0)
    del Xs, Ys, Ms

    np.save(cache / "X.npy", X)
    np.save(cache / "Y.npy", Y)
    np.save(cache / "mask.npy", mask)

    input_vars  = OCN_INPUT + ICE_FORCING + [f"{v}_prev" for v in ICE_STATE]
    meta = {
        "grid": "gx1v7", "nj": NJ, "ni": NI,
        "members": args.members, "year_start": args.year_start, "year_end": args.year_end,
        "memory_lag_steps": MEMORY_LAG_STEPS, "cadence": "monthly",
        "input_vars": input_vars, "output_vars": ICE_TARGETS,
        "n_in": X.shape[1], "n_out": Y.shape[1], "N": int(X.shape[0]),
        "X_shape": list(X.shape), "Y_shape": list(Y.shape),
        "units": "raw (aice[0-1], hi[m], vel[m/s], strocn[N/m2], SST[degC], uatm[m/s])",
    }
    (cache / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Wrote cache:", cache)
    print("  X", X.shape, "Y", Y.shape, "mask", mask.shape)
    print("  input_vars ", input_vars)
    print("  output_vars", ICE_TARGETS)


if __name__ == "__main__":
    main()
