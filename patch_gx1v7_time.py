"""
Patch the time coordinate in all gx1v7 zarrs from integer indices back to
the original cftime.DatetimeNoLeap values copied from the source zarrs.

Only the time/ directory inside each zarr is replaced — the large field data
arrays are untouched.  Runs in seconds.
"""
import glob
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

SRC_GLOB = (
    "/glade/derecho/scratch/wchapman/b_credit_runs/"
    "b.e21.CREDIT_climate_branch_1980_????_zmdata_ERA5scaled_zmdata_Qtot.zarr"
)
DST_DIR = Path("/glade/derecho/scratch/praggarwal/zarr_gx1v7")

# Build year → source-path mapping
src_by_year = {}
for sp in sorted(glob.glob(SRC_GLOB)):
    parts = Path(sp).name.split("_")
    year = parts[4]           # e.g. '2007'
    src_by_year[year] = sp

print(f"Found {len(src_by_year)} source zarrs")

for year, src_path in sorted(src_by_year.items()):
    dst_path = DST_DIR / f"b.e21.CREDIT_gx1v7_{year}.zarr"
    if not dst_path.exists():
        print(f"  SKIP (missing dst): {year}")
        continue

    # Pull time values + CF encoding from source
    ds_src = xr.open_zarr(src_path, consolidated=False)
    time_vals = ds_src["time"].values                          # cftime array
    time_enc  = {k: v for k, v in ds_src["time"].encoding.items()
                 if k in ("units", "calendar", "dtype", "_FillValue")}
    ds_src.close()

    # Write time to a tiny temp zarr to get the properly encoded directory
    tmp_path = DST_DIR / f"_tmp_time_{year}.zarr"
    tiny_ds = xr.Dataset({"time": ("time", time_vals)})
    tiny_ds.to_zarr(str(tmp_path), mode="w",
                    encoding={"time": time_enc} if time_enc else None,
                    consolidated=False)

    # Splice the time/ dir into the destination zarr
    dst_time = dst_path / "time"
    if dst_time.exists():
        shutil.rmtree(str(dst_time))
    shutil.copytree(str(tmp_path / "time"), str(dst_time))
    shutil.rmtree(str(tmp_path))

    print(f"  Patched {year}: sample = {time_vals[0]}")

print("Done — verifying one zarr:")
ds = xr.open_zarr(str(DST_DIR / "b.e21.CREDIT_gx1v7_2007.zarr"), consolidated=False)
print("  time[0]:", ds["time"].values[0], " → .year:", ds["time"].values[0].year)
ds.close()
