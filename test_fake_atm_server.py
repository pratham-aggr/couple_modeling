"""
test_fake_atm_server.py
========================
Standalone test driver — plays CESM's role without actually running CESM.

Reads one year of SST/ICEFRAC from a zarr store, drives fake_atm_server.py
through the flag-file protocol, collects outputs, and plots diagnostics.

Usage (two terminals):
    Terminal 1:  python fake_atm_server.py --run_dir ./test_run --model_dir ./output_full
    Terminal 2:  python test_fake_atm_server.py --run_dir ./test_run --zarr_path <path>
"""

import argparse
import time
from pathlib import Path

import numpy as np
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


POLL_INTERVAL = 0.1
POLL_TIMEOUT  = 300   # 5 minutes per step (generous for first step which loads model)

N_LAT_T62, N_LON_T62 = 94, 192


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gauss_legendre_lats(n: int) -> np.ndarray:
    from numpy.polynomial.legendre import leggauss
    x, _ = leggauss(n)
    return np.degrees(np.arcsin(x[::-1]))


def t62_grid():
    lats = gauss_legendre_lats(N_LAT_T62)
    lons = np.linspace(0.0, 360.0, N_LON_T62, endpoint=False)
    return lats, lons


def wait_for_flag(flag: Path, timeout: float = POLL_TIMEOUT) -> bool:
    elapsed = 0.0
    while not flag.exists():
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
        if elapsed >= timeout:
            return False
    return True


def write_sst_in(path: Path, sst: np.ndarray, ifrac: np.ndarray, ymd: int, tod: int):
    with nc.Dataset(path, "w", format="NETCDF4") as f:
        f.createDimension("ncol", len(sst))
        v = f.createVariable("sst",   "f8", ("ncol",)); v[:] = sst
        v = f.createVariable("ifrac", "f8", ("ncol",)); v[:] = ifrac
        v = f.createVariable("ymd",   "i4", ()); v[:] = ymd
        v = f.createVariable("tod",   "i4", ()); v[:] = tod


def read_cam_out(path: Path) -> dict:
    fields = {}
    with nc.Dataset(path, "r") as f:
        for name in f.variables:
            if name != "ncol":
                fields[name] = f.variables[name][:].copy()
    return fields


def interp_zarr_to_t62(arr_2d: np.ndarray, src_lats: np.ndarray,
                        src_lons: np.ndarray, dst_lats: np.ndarray,
                        dst_lons: np.ndarray) -> np.ndarray:
    """Bilinear interpolation from zarr 1° grid to T62."""
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (src_lats, src_lons), arr_2d, method="linear",
        bounds_error=False, fill_value=None
    )
    lon2d, lat2d = np.meshgrid(dst_lons, dst_lats)
    pts = np.stack([lat2d.ravel(), lon2d.ravel()], axis=-1)
    return interp(pts)   # (18048,)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",  required=True)
    parser.add_argument("--zarr_path", required=True,
                        help="Path to a single zarr store (e.g. 1980)")
    parser.add_argument("--n_steps", type=int, default=None,
                        help="Limit number of timesteps (default: all)")
    parser.add_argument("--out_dir", default="./test_run_output")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flag_go    = run_dir / "camulator_go.flag"
    flag_done  = run_dir / "camulator_done.flag"
    flag_ready = run_dir / "camulator_server_ready.flag"
    sst_in     = run_dir / "camulator_sst_in.nc"
    cam_out    = run_dir / "camulator_cam_out.nc"

    # --- wait for server to be ready ---
    print(f"[driver] Waiting for server ready flag in {run_dir} ...")
    if not wait_for_flag(flag_ready, timeout=120):
        raise RuntimeError("Server did not become ready within 120 s. "
                           "Start fake_atm_server.py first.")
    print("[driver] Server is ready.")

    # --- load zarr ---
    print(f"[driver] Loading zarr: {args.zarr_path}")
    ds = xr.open_zarr(args.zarr_path, consolidated=False)
    times    = ds["time"].values
    src_lats = ds["latitude"].values
    src_lons = ds["longitude"].values

    dst_lats, dst_lons = t62_grid()

    n_steps = len(times) if args.n_steps is None else min(args.n_steps, len(times))
    print(f"[driver] Running {n_steps} timesteps ...")

    # Storage for diagnostics
    store = {k: [] for k in ["fsds", "flnsd", "prect", "tbot", "u10", "qbot", "zbot"]}

    for i in range(n_steps):
        t = times[i]
        ymd  = int(t.year) * 10000 + int(t.month) * 100 + int(t.day)
        tod  = int(t.hour) * 3600 + int(t.minute) * 60 + int(t.second)

        # Load SST and ICEFRAC from zarr, interpolate to T62
        sst_1deg  = ds["SST"].isel(time=i).values.astype(np.float64)
        ifrac_1deg = ds["ICEFRAC"].isel(time=i).values.astype(np.float64)

        sst_t62   = interp_zarr_to_t62(sst_1deg,  src_lats, src_lons, dst_lats, dst_lons)
        ifrac_t62 = interp_zarr_to_t62(ifrac_1deg, src_lats, src_lons, dst_lats, dst_lons)
        ifrac_t62 = np.clip(ifrac_t62, 0.0, 1.0)

        # Fill NaNs (land points in zarr SST)
        nan_mask = ~np.isfinite(sst_t62)
        sst_t62[nan_mask]   = 0.0   # server will replace < 270 K with 283 K
        ifrac_t62[nan_mask] = 0.0

        # Write input NC and trigger server
        write_sst_in(sst_in, sst_t62, ifrac_t62, ymd, tod)
        flag_done.unlink(missing_ok=True)
        flag_go.touch()

        # Wait for server
        if not wait_for_flag(flag_done, timeout=POLL_TIMEOUT):
            raise RuntimeError(f"Server timed out at step {i} (ymd={ymd})")

        # Read outputs
        fields = read_cam_out(cam_out)
        for k in store:
            if k in fields:
                store[k].append(float(fields[k].mean()))

        if i % 50 == 0:
            print(f"[driver] step {i+1:4d}/{n_steps}  ymd={ymd}  "
                  f"fsds={store['fsds'][-1]:.1f} W/m²  "
                  f"flnsd={store['flnsd'][-1]:.1f} W/m²  "
                  f"prect={store['prect'][-1]*86400*1000:.2f} mm/day")

    ds.close()
    print("[driver] Done. Plotting diagnostics ...")

    # --- sanity checks ---
    fsds  = np.array(store["fsds"])
    flnsd = np.array(store["flnsd"])
    prect = np.array(store["prect"])
    tbot  = np.array(store["tbot"])

    n_nan  = int(np.isnan(fsds).sum() + np.isnan(flnsd).sum() +
                 np.isnan(prect).sum() + np.isnan(tbot).sum())
    n_neg_sw = int((fsds < 0).sum())
    n_neg_lw = int((flnsd < 0).sum())
    n_bad_p  = int((prect > 1e-3).sum())
    n_bad_t  = int(((tbot < 230) | (tbot > 340)).sum())

    print(f"\n=== Sanity checks ===")
    print(f"  NaNs in output:        {n_nan}  {'PASS' if n_nan == 0 else 'FAIL'}")
    print(f"  Negative SW:           {n_neg_sw}  {'PASS' if n_neg_sw == 0 else 'FAIL'}")
    print(f"  Negative LW:           {n_neg_lw}  {'PASS' if n_neg_lw == 0 else 'FAIL'}")
    print(f"  Precip > 1e-3 m/s:     {n_bad_p}  {'PASS' if n_bad_p == 0 else 'WARN'}")
    print(f"  Tbot out of [230,340]: {n_bad_t}  {'PASS' if n_bad_t == 0 else 'WARN'}")

    # --- plot time series ---
    steps = np.arange(n_steps)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(steps, fsds,  label="fsds (SW↓)")
    axes[0].plot(steps, flnsd, label="flnsd (LW↓)")
    axes[0].set_ylabel("W m⁻²"); axes[0].legend(); axes[0].set_title("Global-mean radiative fluxes")

    axes[1].plot(steps, prect * 86400 * 1000)
    axes[1].set_ylabel("mm day⁻¹"); axes[1].set_title("Global-mean precipitation")

    axes[2].plot(steps, tbot)
    axes[2].set_ylabel("K"); axes[2].set_title("Global-mean surface temperature (tbot)")

    axes[3].plot(steps, np.array(store["u10"]))
    axes[3].set_ylabel("m s⁻¹"); axes[3].set_title("Global-mean |wind| (u10 component)")
    axes[3].set_xlabel("Timestep (6-hourly)")

    plt.tight_layout()
    fig.savefig(out_dir / "timeseries.png", dpi=120, bbox_inches="tight")
    print(f"Saved → {out_dir}/timeseries.png")
    plt.close()

    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
