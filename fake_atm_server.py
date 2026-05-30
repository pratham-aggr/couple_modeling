"""
fake_atm_server.py
==================
MLP inference server that acts as a fake atmosphere for CESM.

Implements the CAMulator file-based handshake protocol:
  - Polls for camulator_go.flag
  - Reads camulator_sst_in.nc  (SST, ICEFRAC on T62 grid, written by DATM)
  - Runs MLP inference
  - Writes camulator_cam_out.nc (fluxes/state on T62 grid, read by DATM)
  - Writes camulator_done.flag

Usage:
    python fake_atm_server.py --run_dir /path/to/cesm/run --model_dir ./output_full
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import netCDF4 as nc
import torch
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# Import from training script
sys.path.insert(0, str(Path(__file__).parent))
from train_fake_atm import FakeAtmMLP, Normalizer
from explore_fake_atm import compute_toa_insolation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATICS_PATH = (
    "/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/"
    "b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc"
)

# T62 Gaussian grid (94 lat × 192 lon = 18,048 points)
# Gauss-Legendre latitudes, evenly-spaced longitudes
N_LAT_T62, N_LON_T62 = 94, 192

# CAM6 L32 hybrid coordinate scale factor for reference height
Z_BOT_SCALE = 0.2187  # m / K  (derived from hybi[-2] = 0.98511219)

POLL_INTERVAL   = 0.1   # seconds between flag checks
POLL_TIMEOUT    = 10800 # 3 hours


# ---------------------------------------------------------------------------
# T62 grid construction
# ---------------------------------------------------------------------------

def gauss_legendre_lats(n: int) -> np.ndarray:
    """Return n Gauss-Legendre latitudes in degrees, south to north."""
    from numpy.polynomial.legendre import leggauss
    x, _ = leggauss(n)
    return np.degrees(np.arcsin(x[::-1]))  # ascending


def t62_grid():
    lats = gauss_legendre_lats(N_LAT_T62)                            # (94,)
    lons = np.linspace(0.0, 360.0, N_LON_T62, endpoint=False)        # (192,)
    return lats, lons


# ---------------------------------------------------------------------------
# LANDFRAC interpolation from 1° statics to T62
# ---------------------------------------------------------------------------

def load_landfrac_t62(lats_t62: np.ndarray, lons_t62: np.ndarray) -> np.ndarray:
    ds = xr.open_dataset(STATICS_PATH)
    lf   = ds["LANDFRAC"].values.astype(np.float32)   # (192, 288) lat×lon on 1° grid
    lat1 = ds["latitude"].values                       # (192,)
    lon1 = ds["longitude"].values                      # (288,)
    ds.close()

    interp = RegularGridInterpolator(
        (lat1, lon1), lf, method="linear", bounds_error=False, fill_value=None
    )
    lon2d, lat2d = np.meshgrid(lons_t62, lats_t62)     # (94, 192)
    pts = np.stack([lat2d.ravel(), lon2d.ravel()], axis=-1)
    lf_t62 = interp(pts).reshape(N_LAT_T62, N_LON_T62).astype(np.float32)
    return np.clip(lf_t62, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def e_sat(T: np.ndarray) -> np.ndarray:
    """Saturation vapor pressure [Pa] via Clausius-Clapeyron."""
    return 611.2 * np.exp(17.67 * (T - 273.16) / (T - 29.65))


def q_sat(T: np.ndarray, p: float = 101325.0) -> np.ndarray:
    """Saturated specific humidity [kg/kg]."""
    es = e_sat(T)
    return np.clip(0.622 * es / np.maximum(p - es, 1.0), 1e-9, None)


def ymd_to_doy(ymd: int) -> int:
    """Convert YYYYMMDD integer to approximate day-of-year (1-based, ignores leap)."""
    MONTH_START = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    m = (ymd // 100) % 100
    d = ymd % 100
    return MONTH_START[m - 1] + d


# ---------------------------------------------------------------------------
# NetCDF I/O
# ---------------------------------------------------------------------------

def read_sst_in(path: Path):
    with nc.Dataset(path, "r") as f:
        sst   = f.variables["sst"][:].astype(np.float64)
        ifrac = f.variables["ifrac"][:].astype(np.float64)
        ymd   = int(f.variables["ymd"][:])
        tod   = int(f.variables["tod"][:])
    return sst, ifrac, ymd, tod


def write_cam_out(path: Path, fields: dict):
    with nc.Dataset(path, "w", format="NETCDF4") as f:
        f.createDimension("ncol", N_LAT_T62 * N_LON_T62)
        for name, data in fields.items():
            v = f.createVariable(name, "f8", ("ncol",))
            v[:] = data.ravel().astype(np.float64)


# ---------------------------------------------------------------------------
# Flag file helpers
# ---------------------------------------------------------------------------

def wait_for_flag(flag: Path, interval: float = POLL_INTERVAL,
                  timeout: float = POLL_TIMEOUT) -> bool:
    elapsed = 0.0
    while not flag.exists():
        time.sleep(interval)
        elapsed += interval
        if elapsed >= timeout:
            return False
    return True


def write_flag(flag: Path):
    flag.touch()


def remove_flag(flag: Path):
    flag.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# MLP inference → coupler fields
# ---------------------------------------------------------------------------

OUTPUT_VARS = ["FSDS_J", "FLDS_J", "FSUS", "FLUS", "FSUTOA", "FLUT",
               "PRECT", "TS", "U10"]
INPUT_VARS  = ["SST", "ICEFRAC", "SOLIN", "LANDFRAC"]


def run_inference(sst_flat, ifrac_flat, solin_flat, landfrac_flat,
                  lats_flat, lons_flat, doy: int,
                  model: FakeAtmMLP, norm: Normalizer,
                  device: torch.device) -> dict:
    n = len(sst_flat)

    # Fill land-point SST (< 270 K set to climatological ocean value)
    sst_filled = sst_flat.copy()
    sst_filled[sst_filled < 270.0] = 283.0

    # Cyclic encodings
    doy_rad = 2 * np.pi * doy / 365.0
    lat_rad = np.deg2rad(lats_flat)
    lon_rad = np.deg2rad(lons_flat)

    X = np.stack([
        sst_filled.astype(np.float32),
        ifrac_flat.astype(np.float32),
        solin_flat.astype(np.float32),
        landfrac_flat.astype(np.float32),
        np.full(n, np.sin(doy_rad), dtype=np.float32),
        np.full(n, np.cos(doy_rad), dtype=np.float32),
        np.sin(lat_rad).astype(np.float32),
        np.cos(lat_rad).astype(np.float32),
        np.sin(lon_rad).astype(np.float32),
        np.cos(lon_rad).astype(np.float32),
    ], axis=-1)  # (n, 10)

    X_norm = norm.transform_x(X).astype(np.float32)
    with torch.no_grad():
        Y_norm = model(torch.from_numpy(X_norm).to(device)).cpu().numpy()
    Y = norm.inverse_y(Y_norm)  # (n, 9)

    # Clip non-negative outputs
    NON_NEG_IDX = [OUTPUT_VARS.index(v) for v in
                   ["FSDS_J", "FLDS_J", "FSUS", "FLUS", "FSUTOA", "FLUT", "PRECT", "U10"]]
    Y[:, NON_NEG_IDX] = np.maximum(Y[:, NON_NEG_IDX], 0.0)

    d = {v: Y[:, i] for i, v in enumerate(OUTPUT_VARS)}

    # Map to coupler fields
    pbot = np.full(n, 101325.0, dtype=np.float64)
    tbot = d["TS"].astype(np.float64)
    u10  = d["U10"].astype(np.float64)

    fields = {
        "fsds":  np.maximum(d["FSDS_J"] / 21600.0, 0.0),            # W/m² downwelling SW
        "swnet": np.maximum((d["FSDS_J"] - d["FSUS"]) / 21600.0, 0.0),  # W/m² net SW
        "flnsd": np.maximum(d["FLDS_J"] / 21600.0, 0.0),                # W/m² downwelling LW
        # PRECT in zarr is m per 6h step (accumulated) — divide by 21600 to get m/s rate
        "prect": np.clip(d["PRECT"] / 21600.0, 0.0, 2e-4), # m/s
        "tbot":  tbot,                                       # K
        "tref":  tbot,                                       # K (diagnostic)
        "u10":   u10 / np.sqrt(2.0),                        # m/s (zonal component)
        "v10":   u10 / np.sqrt(2.0),                        # m/s (meridional component)
        "qbot":  q_sat(tbot, p=101325.0),                   # kg/kg
        "pbot":  pbot,                                       # Pa
        "zbot":  np.clip(Z_BOT_SCALE * tbot, 20.0, 200.0), # m
    }
    return fields


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",   required=True,  help="CESM run directory (flag files live here)")
    parser.add_argument("--model_dir", default="./output_full", help="Directory with best_model.pt and normalizer.npz")
    args = parser.parse_args()

    run_dir   = Path(args.run_dir)
    model_dir = Path(args.model_dir)

    flag_go    = run_dir / "camulator_go.flag"
    flag_done  = run_dir / "camulator_done.flag"
    flag_ready = run_dir / "camulator_server_ready.flag"
    sst_in     = run_dir / "camulator_sst_in.nc"
    cam_out    = run_dir / "camulator_cam_out.nc"

    # --- load model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[server] Device: {device}")

    norm  = Normalizer.load(str(model_dir / "normalizer.npz"))
    ckpt  = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=True)

    import json
    cfg   = json.load(open(model_dir / "model_config.json"))
    model = FakeAtmMLP(cfg["n_in"], cfg["n_out"], cfg["hidden"], cfg["depth"]).to(device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"[server] Model loaded ({cfg['n_in']}→{cfg['n_out']}, hidden={cfg['hidden']}, depth={cfg['depth']})")

    # --- build T62 grid and interpolate LANDFRAC ---
    lats_t62, lons_t62 = t62_grid()
    landfrac = load_landfrac_t62(lats_t62, lons_t62)  # (94, 192)
    lon2d, lat2d = np.meshgrid(lons_t62, lats_t62)    # (94, 192)

    lats_flat     = lat2d.ravel()
    lons_flat     = lon2d.ravel()
    landfrac_flat = landfrac.ravel()
    print(f"[server] T62 grid: {N_LAT_T62}×{N_LON_T62} = {N_LAT_T62*N_LON_T62} points")

    # --- signal ready ---
    remove_flag(flag_done)
    write_flag(flag_ready)
    print(f"[server] Ready — waiting for {flag_go.name} in {run_dir}")

    step = 0
    while True:
        if not wait_for_flag(flag_go):
            print("[server] Timeout waiting for go flag — exiting.")
            break

        step += 1
        t0 = time.time()

        # Read SST/ICEFRAC from CESM
        sst, ifrac, ymd, tod = read_sst_in(sst_in)
        doy = ymd_to_doy(ymd)

        # Compute SOLIN analytically on T62 grid
        solin_2d = compute_toa_insolation(lats_t62, np.array([doy]))[0]  # (nlat,)
        solin_flat = np.broadcast_to(solin_2d[:, None], (N_LAT_T62, N_LON_T62)).ravel().copy()

        # Run MLP
        fields = run_inference(
            sst, ifrac, solin_flat, landfrac_flat,
            lats_flat, lons_flat, doy,
            model, norm, device
        )

        # Write outputs
        write_cam_out(cam_out, fields)

        # Handshake
        remove_flag(flag_go)
        write_flag(flag_done)

        elapsed = time.time() - t0
        print(f"[server] step {step:4d}  ymd={ymd}  doy={doy:3d}  "
              f"fsds={fields['fsds'].mean():.1f} W/m²  "
              f"flnsd={fields['flnsd'].mean():.1f} W/m²  "
              f"prect={fields['prect'].mean()*86400*1000:.2f} mm/day  "
              f"({elapsed:.2f}s)")


if __name__ == "__main__":
    main()
