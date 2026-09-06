"""
build_blstate_forcing.py
========================
Build per-year boundary-layer-state forcing files for the hybrid (OMIP-style)
standalone POP run — the v11 `--bulk_fluxes` path in
camulator_ud/climate/model_server.py.

Extracts from the wchapman CREDIT zarrs (6-hourly, CAM 192x288 grid, noleap):
  Ubot,Vbot,Tbot,Qbot : bottom model level, isel(level=-1) (992.6 hPa ~ 60 m;
                        level is ordered top->surface, same as add_atm_to_cache.py)
  PS                  : surface pressure (Pa)
  pbot                : bottom-level pressure = hyam[-1]*P0 + hybm[-1]*PS (Pa)
  FSDS, FLDS          : downwelling SW/LW at surface (W/m2, stored as-is)
  PRECT               : total precip, converted m/6h -> kg/m2/s (x1000/21600)

All fields stay on the CAM grid; the server remaps to gx1v7 at runtime with the
same RegularToScatter transform it already uses for SOLIN. Output is one
NetCDF4 per year, float32, chunked (1,192,288) for fast single-step reads.

Usage:
    python build_blstate_forcing.py --y0 1980 --y1 1984 \
        --out_dir /glade/derecho/scratch/praggarwal/blstate_cam
"""

import argparse
import glob
import time as _time

import numpy as np
import xarray as xr
import netCDF4 as nc

ZARR_PAT = ("/glade/derecho/scratch/wchapman/b_credit_runs/"
            "b.e21.CREDIT_climate_branch_1980_{year}_zmdata_ERA5scaled_zmdata_Qtot.zarr")
LEVEL_VARS = {"Ubot": "U", "Vbot": "V", "Tbot": "T", "Qbot": "Q"}
SURF_VARS = {"PS": "PS", "FSDS": "FSDS", "FLDS": "FLDS", "PRECT": "PRECT"}
UNITS = {"Ubot": "m/s", "Vbot": "m/s", "Tbot": "K", "Qbot": "kg/kg",
         "PS": "Pa", "pbot": "Pa", "FSDS": "W/m2 downwelling",
         "FLDS": "W/m2 downwelling", "PRECT": "kg/m2/s"}
BATCH = 80  # time steps per read (zarr time chunk = 4)


def build_year(year: int, out_dir: str):
    zpath = ZARR_PAT.format(year=year)
    ds = xr.open_zarr(zpath, consolidated=False)
    nt, nlat, nlon = ds.sizes["time"], ds.sizes["latitude"], ds.sizes["longitude"]
    out_path = f"{out_dir}/blstate_{year}.nc"
    print(f"[{year}] {zpath} -> {out_path}  ({nt} steps)", flush=True)

    with nc.Dataset(out_path, "w") as o:
        o.createDimension("time", nt)
        o.createDimension("latitude", nlat)
        o.createDimension("longitude", nlon)
        o.setncattr("source", zpath)
        o.setncattr("description",
                    "CREDIT boundary-layer state for MEMO --bulk_fluxes "
                    "(bottom level = zarr isel(level=-1), 992.6 hPa)")
        vlat = o.createVariable("latitude", "f8", ("latitude",))
        vlon = o.createVariable("longitude", "f8", ("longitude",))
        vlat[:] = ds["latitude"].values
        vlon[:] = ds["longitude"].values
        vtime = o.createVariable("time", "f8", ("time",))
        vtime.units = f"days since {year}-01-01 00:00:00"
        vtime.calendar = "noleap"
        t0 = ds["time"].values[0]
        vtime[:] = np.array(
            [(t - t0).total_seconds() / 86400.0 for t in ds["time"].values])

        ov = {}
        for name in list(LEVEL_VARS) + ["PS", "pbot", "FSDS", "FLDS", "PRECT"]:
            v = o.createVariable(name, "f4", ("time", "latitude", "longitude"),
                                 chunksizes=(1, nlat, nlon))
            v.units = UNITS[name]
            ov[name] = v

        hyam_b = ds["hyam"].isel(level=-1).values  # (time,)
        hybm_b = ds["hybm"].isel(level=-1).values
        p0 = ds["P0"].values
        tic = _time.time()
        for i0 in range(0, nt, BATCH):
            i1 = min(i0 + BATCH, nt)
            sl = slice(i0, i1)
            for name, src in LEVEL_VARS.items():
                ov[name][sl] = (ds[src].isel(time=sl, level=-1)
                                .values.astype(np.float32))
            ps = ds["PS"].isel(time=sl).values.astype(np.float64)
            ov["PS"][sl] = ps.astype(np.float32)
            ov["pbot"][sl] = (hyam_b[sl, None, None] * p0[sl, None, None]
                              + hybm_b[sl, None, None] * ps).astype(np.float32)
            ov["FSDS"][sl] = ds["FSDS"].isel(time=sl).values.astype(np.float32)
            ov["FLDS"][sl] = ds["FLDS"].isel(time=sl).values.astype(np.float32)
            ov["PRECT"][sl] = (ds["PRECT"].isel(time=sl).values
                               * 1000.0 / 21600.0).astype(np.float32)
            print(f"  [{year}] {i1}/{nt}  ({_time.time()-tic:.0f}s)", flush=True)

    # quick self-check
    with nc.Dataset(out_path) as o:
        for name in ("Tbot", "PS", "pbot", "FSDS", "PRECT"):
            a = o[name][0]
            print(f"  check {name}: mean={float(a.mean()):.4g} "
                  f"min={float(a.min()):.4g} max={float(a.max()):.4g}", flush=True)
    print(f"[{year}] done", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--y0", type=int, required=True)
    p.add_argument("--y1", type=int, required=True)
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()
    import os
    os.makedirs(a.out_dir, exist_ok=True)
    for y in range(a.y0, a.y1 + 1):
        build_year(y, a.out_dir)
