"""
extract_ice_ic.py
=================
Extract the gx1v7 sea-ice initial condition (aice, hi, uvel, vvel) for the
standalone MEMO->POP run's start month from the SAME CESM2-LE member the ocean
restart comes from (LE2-1231.002, 1980-01).  The online CICE emulator is seeded
with this so its first autoregressive step starts from a physically consistent
ice state (not zeros).  Saved as a small npz the server loads at startup.
"""
import argparse, glob
import numpy as np
import xarray as xr

ICE_ROOT = "/glade/campaign/cesm/collections/CESM2-LE/ice/proc/tseries/month_1"
FIELDS = ["aice", "hi", "uvel", "vvel"]


def _find(var, member, year):
    # historical (BHIST) files cover 1980; pick the file whose range contains `year`
    for p in sorted(glob.glob(f"{ICE_ROOT}/{var}/*{member}*.nc")):
        yr = p.split(".")[-2]                     # e.g. 197001-197912
        y0, y1 = int(yr[:4]), int(yr[7:11])
        if y0 <= year <= y1:
            return p
    raise FileNotFoundError(f"{var} {member} year {year}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default="LE2-1231.002")
    ap.add_argument("--year", type=int, default=1980)
    ap.add_argument("--month", type=int, default=1)
    ap.add_argument("--out", default="/glade/derecho/scratch/praggarwal/couple_cache_cice_nowind/ice_ic_1980-01.npz")
    args = ap.parse_args()

    ic = {}
    for v in FIELDS:
        ds = xr.open_dataset(_find(v, args.member, args.year), decode_timedelta=False)
        # CICE monthly data is stamped at period-END (Jan-1980 mean -> 1980-02-01),
        # so the first record of the file covering `year` is that year's January mean
        # -- the physically consistent early-1980 ice state to seed the run.
        arr = np.asarray(ds[v].isel(time=0).values, dtype=np.float32)
        print(f"  {v}: seeded from stamp {str(ds['time'].values[0])} (= Jan {args.year} mean)")
        arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)   # land/fill -> 0
        ic[v] = arr
        print(f"  {v:6s} shape={arr.shape} min={arr.min():.4f} max={arr.max():.4f} "
              f"mean(ice>0)={(arr[arr>0].mean() if (arr>0).any() else 0):.4f}")
    np.savez(args.out, **ic)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
