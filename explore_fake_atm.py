"""
Fake Atmosphere Exploration
============================
Goal: understand what the ocean produces and how well it correlates
with the atmospheric fields we need to predict.

Usage:
    python explore_fake_atm.py --zarr_path /path/to/your.zarr
"""

import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# TOA Insolation (analytical, no file needed)
# ---------------------------------------------------------------------------

def compute_toa_insolation(lat_deg: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """
    Compute daily-mean TOA insolation [W/m2].

    Parameters
    ----------
    lat_deg : (nlat,) degrees
    doy     : (ntime,) day-of-year  1..365

    Returns
    -------
    S : (ntime, nlat) W/m2
    """
    S0 = 1361.0  # solar constant W/m2

    lat = np.deg2rad(lat_deg)                          # (nlat,)
    dec = np.deg2rad(23.45 * np.sin(np.deg2rad(360 / 365 * (doy - 81))))  # (ntime,)

    # daily mean insolation: S0 * cos(zenith) integrated over daylight hours
    # H = half-day length in radians
    cos_lat = np.cos(lat)[None, :]                     # (1, nlat)
    sin_lat = np.sin(lat)[None, :]
    cos_dec = np.cos(dec)[:, None]                     # (ntime, 1)
    sin_dec = np.sin(dec)[:, None]

    arg = -np.tan(lat)[None, :] * np.tan(dec)[:, None]
    arg = np.clip(arg, -1.0, 1.0)
    H = np.arccos(arg)                                 # half-day angle (ntime, nlat)

    S = (S0 / np.pi) * (H * sin_lat * sin_dec + cos_lat * cos_dec * np.sin(H))
    return np.maximum(S, 0.0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_zarr(zarr_path: str) -> xr.Dataset:
    ds = xr.open_zarr(zarr_path, consolidated=False)
    print(f"\nLoaded: {zarr_path}")
    print(ds)
    return ds


def load_normalization(mean_path: str, std_path: str):
    mean_ds = xr.open_dataset(mean_path)
    std_ds  = xr.open_dataset(std_path)
    return mean_ds, std_ds


# ---------------------------------------------------------------------------
# Variable definitions
# ---------------------------------------------------------------------------

# Fields the ocean sends to the coupler (our model INPUTS)
OCEAN_INPUT_VARS = [
    "SST",      # or So_t — ocean surface temperature
    "ICEFRAC",  # or Sf_ifrac — sea ice fraction
]

# Atmospheric fields we need to predict (our model OUTPUTS)
# Surface / single-level only — no pressure columns
ATM_OUTPUT_VARS = [
    "FSDS_J",   # downwelling SW at surface [J/m2 per 6h]
    "FLDS_J",   # downwelling LW at surface [J/m2 per 6h]
    "FSUS",     # upwelling SW at surface
    "FLUS",     # upwelling LW at surface
    "FSUTOA",   # upwelling SW at TOA [W/m2]
    "FLUT",     # OLR at TOA [W/m2]
    "PRECT",    # total precipitation [m/s]
    "TS",       # surface skin temperature [K]
    "U10",      # 10m wind speed [m/s]
]

# Bottom-level 3D fields (we grab level index -1 = lowest model level)
ATM_3D_VARS = ["U", "V", "T", "Qtot", "PS"]


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------

def summarize_dataset(ds: xr.Dataset):
    print("\n=== Variables ===")
    for v in ds.data_vars:
        print(f"  {v:20s}  {str(ds[v].dims):40s}  {ds[v].shape}")
    print("\n=== Coordinates ===")
    for c in ds.coords:
        print(f"  {c:20s}  {ds[c].shape}")


def check_available_vars(ds: xr.Dataset, wanted: list[str]) -> tuple[list, list]:
    present = [v for v in wanted if v in ds]
    missing = [v for v in wanted if v not in ds]
    if missing:
        print(f"\nWARNING — not found in dataset: {missing}")
    return present, missing


def compute_correlations(ds: xr.Dataset, input_vars: list[str], output_vars: list[str],
                          n_time: int = 100) -> dict:
    """
    Pearson correlation between each input and output variable,
    computed over (time, lat, lon) flattened, on a small time subset.
    """
    results = {}
    for out_var in output_vars:
        if out_var not in ds:
            continue
        out_data = ds[out_var].isel(time=slice(0, n_time)).values.ravel()
        results[out_var] = {}
        for in_var in input_vars:
            if in_var not in ds:
                continue
            in_data = ds[in_var].isel(time=slice(0, n_time)).values.ravel()
            # mask NaNs
            mask = np.isfinite(in_data) & np.isfinite(out_data)
            if mask.sum() < 10:
                results[out_var][in_var] = np.nan
            else:
                results[out_var][in_var] = float(np.corrcoef(in_data[mask], out_data[mask])[0, 1])
    return results


def plot_correlations(corr: dict, out_path: str = "correlations.png"):
    out_vars = list(corr.keys())
    in_vars  = list(next(iter(corr.values())).keys())
    matrix   = np.array([[corr[o].get(i, np.nan) for i in in_vars] for o in out_vars])

    fig, ax = plt.subplots(figsize=(max(6, len(in_vars) * 1.5), max(4, len(out_vars) * 0.8)))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(in_vars)));  ax.set_xticklabels(in_vars, rotation=45, ha="right")
    ax.set_yticks(range(len(out_vars))); ax.set_yticklabels(out_vars)
    for i in range(len(out_vars)):
        for j in range(len(in_vars)):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Ocean inputs vs Atm outputs — Pearson correlation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved correlation plot → {out_path}")


def plot_sample_fields(ds: xr.Dataset, vars_to_plot: list[str], time_idx: int = 0,
                       out_path: str = "sample_fields.png"):
    present = [v for v in vars_to_plot if v in ds]
    if not present:
        print("No variables to plot.")
        return

    ncols = 3
    nrows = int(np.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
    axes = np.array(axes).ravel()

    for ax, v in zip(axes, present):
        data = ds[v].isel(time=time_idx)
        # if 3D grab bottom level
        if "lev" in data.dims:
            data = data.isel(lev=-1)
        data.plot(ax=ax, add_colorbar=True)
        ax.set_title(v)

    for ax in axes[len(present):]:
        ax.set_visible(False)

    plt.suptitle(f"Sample fields at time index {time_idx}", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved sample fields → {out_path}")


def add_toa_insolation(ds: xr.Dataset) -> xr.Dataset:
    """Compute TOA insolation and add as a new variable."""
    lat  = ds["latitude"].values
    time = ds["time"].values
    doy  = xr.DataArray(time).dt.dayofyear.values

    S = compute_toa_insolation(lat, doy)  # (ntime, nlat)

    # broadcast to (ntime, nlat, nlon)
    nlon = ds.dims["longitude"]
    S_full = np.broadcast_to(S[:, :, None], (len(doy), len(lat), nlon)).copy()

    ds = ds.assign(
        TOA_SW=xr.DataArray(S_full, dims=["time", "latitude", "longitude"],
                            attrs={"long_name": "TOA insolation (analytical)", "units": "W/m2"})
    )
    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fake atmosphere exploration")
    parser.add_argument("--zarr_path",  required=True,  help="Path to a single .zarr store")
    parser.add_argument("--mean_path",  default=None,   help="Normalization mean .nc (optional)")
    parser.add_argument("--std_path",   default=None,   help="Normalization std .nc (optional)")
    parser.add_argument("--out_dir",    default=".",    help="Output directory for plots")
    parser.add_argument("--n_time",     type=int, default=100,
                        help="Number of timesteps to use for correlation analysis")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    ds = load_zarr(args.zarr_path)
    summarize_dataset(ds)

    # 2. Add TOA insolation
    ds = add_toa_insolation(ds)
    print("\nAdded TOA_SW (analytical insolation)")

    # 3. Check what's available
    all_inputs  = OCEAN_INPUT_VARS + ["TOA_SW"]
    all_outputs = ATM_OUTPUT_VARS

    present_in,  missing_in  = check_available_vars(ds, all_inputs)
    present_out, missing_out = check_available_vars(ds, all_outputs)

    print(f"\nInputs  available: {present_in}")
    print(f"Outputs available: {present_out}")

    # 4. Sample field plots
    plot_sample_fields(ds, present_in + present_out,
                       out_path=str(out_dir / "sample_fields.png"))

    # 5. Correlation matrix
    if present_in and present_out:
        corr = compute_correlations(ds, present_in, present_out, n_time=args.n_time)
        plot_correlations(corr, out_path=str(out_dir / "correlations.png"))

        print("\n=== Correlation summary ===")
        for out_var, in_corrs in corr.items():
            for in_var, r in in_corrs.items():
                print(f"  {in_var:15s} → {out_var:15s}  r = {r:+.3f}")

    # 6. Normalization files (optional sanity check)
    if args.mean_path and args.std_path:
        mean_ds, std_ds = load_normalization(args.mean_path, args.std_path)
        print("\n=== Normalization mean vars ===")
        for v in mean_ds.data_vars:
            print(f"  {v}")


if __name__ == "__main__":
    main()
