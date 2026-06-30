"""
remap_zarr_gx1v7.py
====================
Remap CREDIT training zarr files from 192×288 regular lat/lon to 320×384 gx1v7
curvilinear grid and save as new zarr stores.

The gx1v7 output is stored as (time, latitude=384, longitude=320) — same dimension
names as the source zarr — using j/i indices as the fake 1-D coordinate values so
train_unet.py can open them with xr.open_zarr without modification.

Usage:
    python remap_zarr_gx1v7.py \
        --zarr_glob "/glade/derecho/scratch/wchapman/b_credit_runs/*.zarr" \
        --out_dir   /glade/derecho/scratch/praggarwal/zarr_gx1v7 \
        [--domain   /path/to/domain.ocn.gx1v7.210716.nc]
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import netCDF4 as nc


GX1V7_DOMAIN = (
    "/glade/campaign/cesm/cesmdata/inputdata/share/domains/"
    "domain.ocn.gx1v7.210716.nc"
)

# Variables to remap: all inputs + all possible output targets
REMAP_VARS = [
    "SST", "ICEFRAC", "SOLIN",
    "TAUX", "TAUY", "SHFLX", "LHFLX", "QFLX",
    "FSDS_J", "FLDS_J", "PRECT",
]


class RegularToScatter:
    """Bilinear remap from regular S->N lat/lon grid onto arbitrary target points."""
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


def load_gx1v7_domain(path):
    with nc.Dataset(str(path), "r") as ds:
        xc   = ds["xc"][:].data.astype(np.float64)   # (384, 320)
        yc   = ds["yc"][:].data.astype(np.float64)   # (384, 320)
    print(f"  gx1v7 domain: xc {xc.shape}, yc {yc.shape}")
    return xc, yc


def remap_year(zarr_path, out_path, remap, vars_present):
    """Remap one year's zarr from 192×288 to gx1v7 384×320 and write to out_path."""
    ds = xr.open_zarr(zarr_path, consolidated=False)
    T  = len(ds["time"])
    NJ, NI = remap.shape_out  # 384, 320
    time_vals = ds["time"].values   # preserve cftime so .year works in train_unet.py
    time_enc  = {k: v for k, v in ds["time"].encoding.items()
                 if k in ("units", "calendar", "dtype", "_FillValue")}

    data_vars = {}
    for var in vars_present:
        raw = ds[var].values.astype(np.float32)   # (T, 192, 288)
        remapped = remap.batch(raw).astype(np.float32)   # (T, 384, 320)
        data_vars[var] = xr.DataArray(
            remapped,
            dims=["time", "latitude", "longitude"],
        )
        print(f"    {var}: {raw.shape} → {remapped.shape}")

    ds.close()

    out_ds = xr.Dataset(
        data_vars,
        coords={
            "time":      time_vals,
            "latitude":  np.arange(NJ, dtype=np.float32),
            "longitude": np.arange(NI, dtype=np.float32),
        },
    )
    out_ds.attrs["source"] = str(zarr_path)
    out_ds.attrs["grid"]   = "gx1v7 NJ=384 NI=320 (remapped from 192x288 regular)"

    encoding = {v: {"chunks": (4, NJ, NI)} for v in vars_present}
    if time_enc:
        encoding["time"] = time_enc
    out_ds.to_zarr(str(out_path), mode="w", encoding=encoding, consolidated=True)
    print(f"  Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_glob", required=True)
    parser.add_argument("--out_dir",   required=True)
    parser.add_argument("--domain",    default=GX1V7_DOMAIN)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zarr_paths = sorted(glob.glob(args.zarr_glob))
    if not zarr_paths:
        sys.exit(f"No zarr files found: {args.zarr_glob}")
    print(f"Found {len(zarr_paths)} zarr files")

    # Build remap weights once
    print("Building remap weights ...")
    src_lats = np.linspace(-90.0,  90.0, 192)
    src_lons = np.linspace(  0.0, 360.0, 288, endpoint=False)
    xc, yc = load_gx1v7_domain(args.domain)
    remap = RegularToScatter(src_lats, src_lons, xc, yc)
    print(f"  Remap ready: 192×288 → {remap.shape_out}")

    # Check which REMAP_VARS actually exist in the zarr
    ds0 = xr.open_zarr(zarr_paths[0], consolidated=False)
    vars_present = [v for v in REMAP_VARS if v in ds0]
    missing = [v for v in REMAP_VARS if v not in ds0]
    ds0.close()
    if missing:
        print(f"  WARNING: vars not in zarr (skipped): {missing}")
    print(f"  Remapping {len(vars_present)} vars: {vars_present}")

    for zp in zarr_paths:
        # Extract year from filename: ...branch_1980_2007_zmdata...zarr -> 2007
        # Split: ['b.e21.CREDIT', 'climate', 'branch', '1980', '2007', 'zmdata', ...]
        name = Path(zp).name
        parts = name.split("_")
        year_tag = parts[4] if len(parts) > 4 else name[:40]
        out_path = out_dir / f"b.e21.CREDIT_gx1v7_{year_tag}.zarr"
        if out_path.exists():
            print(f"Skip (exists): {out_path.name}")
            continue
        print(f"Processing {name}  →  {out_path.name}")
        remap_year(zp, out_path, remap, vars_present)

    print("Done.")


if __name__ == "__main__":
    main()
