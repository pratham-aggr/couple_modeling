"""build_solin_clim.py — one-year CYCLIC SOLIN forcing for the self-contained
500-yr run.

The deployed forcing file (b.e21.CREDIT_climate_branch_1980_2014.nc) spans 35 years
(51100 = 35*1460 six-hourly steps). model_server.py treats a multi-year forcing as
NON-cyclic and walks the index off the end at 2014 -> crash. If the forcing file
holds a SINGLE year, the server sets _cyclic_year and wraps the index forever
(and load_blstate then stays pinned to year 1980 -> only blstate_1980.nc needed).

We build that single year as the 35-year climatological mean of SOLIN for each of
the 1460 within-year 6-hourly slots (noleap), written on the 1980 time axis so the
server's start_datetime '1980-01-01 00:00:00' still resolves. SOLIN is TOA
insolation (no meaningful interannual variability), so the climatology is the
correct self-contained choice. ICEFRAC is carried too (harmless; only read if a
config sets icefrac_var).
"""
import numpy as np
import xarray as xr

SRC = "/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/b.e21.CREDIT_climate_branch_1980_2014.nc"
OUT = "/glade/derecho/scratch/praggarwal/forcing_solin_clim_1yr.nc"
NSTEP = 1460  # 365 days * 4 (noleap, 6-hourly)


def main():
    ds = xr.open_dataset(SRC, decode_times=True)
    nt = ds.sizes["time"]
    assert nt % NSTEP == 0, f"time {nt} not a multiple of {NSTEP}"
    nyear = nt // NSTEP
    print(f"src: {nt} steps = {nyear} years x {NSTEP}")

    keep = [v for v in ("SOLIN", "ICEFRAC") if v in ds.data_vars]
    out = xr.Dataset(coords={"time": ds["time"].isel(time=slice(0, NSTEP)),
                             "latitude": ds["latitude"], "longitude": ds["longitude"]})
    for v in keep:
        # memory-safe: accumulate one year at a time (loading the full 51100-step
        # array + float64 cast OOMs the login node) into a (1460,lat,lon) sum.
        acc = np.zeros((NSTEP, *ds[v].shape[1:]), dtype=np.float64)
        for y in range(nyear):
            acc += ds[v].isel(time=slice(y * NSTEP, (y + 1) * NSTEP)).values.astype(np.float64)
        clim = (acc / nyear).astype(np.float32)             # (1460, lat, lon)
        out[v] = (("time", "latitude", "longitude"), clim)
        out[v].attrs = dict(ds[v].attrs)
        print(f"  {v}: clim mean {float(clim.mean()):.2f}, "
              f"range [{float(clim.min()):.1f}, {float(clim.max()):.1f}]")

    yrs = sorted({t.year for t in out.indexes["time"]})
    assert yrs == [1980], f"time axis must be a single year, got {yrs}"
    out.attrs["note"] = ("cyclic 1-year SOLIN climatology (35-yr mean 1980-2014 per "
                         "6-hourly slot) for the self-contained MEMO 500-yr run")
    out.to_netcdf(OUT)
    print(f"wrote {OUT}  ({out.sizes['time']} steps, single year {yrs[0]} -> cyclic)")


if __name__ == "__main__":
    main()
