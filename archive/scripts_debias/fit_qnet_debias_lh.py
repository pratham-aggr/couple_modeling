"""fit_qnet_debias_lh.py — fit a smooth-in-latitude ADDITIVE correction for the
pz220f0b sensreg net's NON-FSDS (SHFLX+LHFLX+FLDS combined) Qnet bias.

Motivation: --qnet_debias (fit_fsds_debias.py) only corrects FSDS. audit_qnet_bias_pz220.py
shows FSDS dominates but does NOT fully explain the regional total bias:
  SO   total +28.4  FSDS +28.8  non-FSDS residual -0.3
  NH   total +18.3  FSDS +22.3  non-FSDS residual -4.0
  Trop total -26.7  FSDS -19.2  non-FSDS residual -7.5
The debiascheck coupled test (job 7234206/7235665) confirmed correcting FSDS alone
still leaves a persistent ~+0.5-1 K/yr global-mean drift vs the live baseline --
this uncorrected non-FSDS residual is the leading suspect. This script fits and
zero-means (TRUE TAREA-weighted, not cos(lat)) an additive lookup table for that
residual, saved as (2,K) [lat_centers, bias] like fsds_debias_lat.npy, applied at
serving time (new --qnet_debias_lh flag) by subtracting it from LHFLX (the largest-
magnitude non-solar term, and where the earlier SO breakdown attributed most of the
non-FSDS bias: "LHFLX +13.2" in memo-so-flux-excess-diagnosed.md).
"""
import json
from pathlib import Path
import numpy as np
import torch, sys, xarray as xr
sys.path.insert(0, "camulator_ud/climate")
from train_unet import UNet

CACHE = "/glade/work/praggarwal/couple_cache_gx1v7_mem24h"
DT6H = 21600.0
MODEL_DIR = "output/output_unet_gx1v7_atm_aux_sensreg"
REF_POP_FILE = ("/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_sensreg50yr/"
                 "run/g.e21.MEMO_GIAF_v01.pop.h.1980-01.nc")
N_SAMPLES = 300
LAT_EDGES = np.arange(-90, 91, 5.0)
I_SH, I_LH, I_FLDS = 2, 3, 6

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ref = xr.open_dataset(REF_POP_FILE)
    TLAT = _ref.TLAT.values
    TAREA = np.nan_to_num(_ref.TAREA.values.astype(np.float64))
    # OPEN-WATER area (see fit_fsds_debias.py for the full rationale): the
    # correction is applied before cice_coupler.apply()'s (1-aice) ice masking, so
    # zero-meaning against raw TAREA over-credits delivery at icy high latitudes
    # and leaks net warming. Use the same baseline-run IFRAC climatology.
    _aice_clim = np.load(Path(MODEL_DIR) / "aice_clim_baseline.npy")
    _aice_annual = _aice_clim.mean(axis=0)
    OW_AREA = TAREA * np.clip(1.0 - _aice_annual, 0.0, 1.0)

    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    Y = np.load(f"{CACHE}/Y.npy", mmap_mode="r")           # TAUX,TAUY,SHFLX,LHFLX,QFLX
    Yr = np.load(f"{CACHE}/Y_rad.npy", mmap_mode="r")      # FSDS_J,FLDS_J
    mask = np.load(f"{CACHE}/mask.npy", mmap_mode="r")
    test_idx = np.load(f"{MODEL_DIR}/test_indices.npy")
    sel = np.sort(np.random.default_rng(1).choice(test_idx, size=min(N_SAMPLES, len(test_idx)), replace=False))
    print(f"device={device}  n={len(sel)} test samples")

    d = Path(MODEL_DIR); cfg = json.loads((d / "model_config.json").read_text())
    model = UNet(n_in=cfg["n_in"], n_out=cfg["n_out"], base=cfg.get("base", 64), dropout=0.0).to(device)
    model.load_state_dict(torch.load(d / "best_model.pt", map_location=device, weights_only=True))
    model.eval()
    nz = np.load(f"{MODEL_DIR}/normalizer.npz")
    xm = nz["x_mean"][None,:,None,None]; xs = nz["x_std"][None,:,None,None]
    ym = nz["y_mean"][None,:,None,None]; ys = nz["y_std"][None,:,None,None]

    m_ok = mask[sel] > 0.5
    raw = X[sel].astype(np.float64)
    dsst = ((raw[:, 0] - raw[:, 3]) / 86400.0)[:, None]
    x_phys = np.concatenate([raw[:, 0:5], dsst], axis=1)

    pred = np.empty((len(sel), 3, x_phys.shape[2], x_phys.shape[3]), np.float64)
    bs = 8
    with torch.no_grad():
        for s in range(0, len(sel), bs):
            xn = (x_phys[s:s+bs] - xm) / (xs + 1e-8)
            yn = model(torch.from_numpy(xn.astype(np.float32)).to(device)).cpu().numpy().astype(np.float64)
            y = yn * ys + ym
            for k, ch in enumerate([I_SH, I_LH, I_FLDS]):
                pred[s:s+bs, k] = y[:, ch] / DT6H

    truth = np.empty_like(pred)
    truth[:, 0] = Y[sel][:, 2].astype(np.float64) / DT6H   # SHFLX
    truth[:, 1] = Y[sel][:, 3].astype(np.float64) / DT6H   # LHFLX
    truth[:, 2] = Yr[sel][:, 1].astype(np.float64) / DT6H  # FLDS

    bias = (pred - truth).sum(axis=1)   # combined SH+LH+FLDS bias, (N,H,W)

    finite = np.isfinite(bias) & m_ok
    extreme = finite & (np.abs(bias) > 5000)
    m_ok = m_ok & finite & ~extreme
    print(f"excluded {int((mask[sel]>0.5).sum() - m_ok.sum())} non-finite/extreme cells")

    lat_b = np.broadcast_to(TLAT[None], m_ok.shape)

    centers, means, areas = [], [], []
    for i in range(len(LAT_EDGES) - 1):
        lo, hi = LAT_EDGES[i], LAT_EDGES[i+1]
        bm = m_ok & (lat_b >= lo) & (lat_b < hi)
        n = int(bm.sum())
        centers.append(0.5 * (lo + hi))
        means.append(float(bias[bm].mean()) if n > 0 else np.nan)
        band_mask_2d = (TLAT >= lo) & (TLAT < hi)
        areas.append(float(OW_AREA[band_mask_2d].sum()))
    centers = np.array(centers); means = np.array(means); areas = np.array(areas)

    valid = np.isfinite(means)
    means_filled = np.interp(centers, centers[valid], means[valid])
    kernel = np.array([1, 1, 1]) / 3.0
    smoothed = np.convolve(means_filled, kernel, mode="same")
    smoothed[0], smoothed[-1] = means_filled[0], means_filled[-1]

    print(f"\n{'lat_center':>10s}{'raw_bias':>12s}{'smoothed':>12s}")
    for c, mv, sv in zip(centers, means, smoothed):
        print(f"{c:>10.1f}{mv:>12.2f}{sv:>12.2f}")

    w = areas
    global_mean = float(np.sum(smoothed * w) / np.sum(w))
    smoothed_zeromean = smoothed - global_mean
    print(f"\narea-weighted (real TAREA) global mean of raw fit: {global_mean:+.2f} W/m2 -> removed")

    out_path = Path(MODEL_DIR) / "qnet_debias_lh_lat.npy"
    np.save(out_path, np.stack([centers, smoothed_zeromean]))
    print(f"\nwrote {out_path}")

    so = smoothed_zeromean[(centers >= -90) & (centers < -45)].mean()
    nh = smoothed_zeromean[(centers >= 45) & (centers <= 90)].mean()
    tr = smoothed_zeromean[(centers >= -30) & (centers < 30)].mean()
    print(f"\nFitted correction (subtract this from LHFLX): SO={so:+.2f}  NH={nh:+.2f}  Tropics={tr:+.2f} W/m2")

if __name__ == "__main__":
    main()
