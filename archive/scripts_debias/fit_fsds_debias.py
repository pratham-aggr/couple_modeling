"""fit_fsds_debias.py — fit a smooth-in-latitude ADDITIVE correction for the
pz220f0b sensreg net's FSDS bias (Step 2 of the tropics-cool/poles-warm root
cause plan). Confirmed via audit_qnet_bias_pz220.py that FSDS now drives ~100%
of the regional Qnet bias (SO +28.8, NH +22.3, Tropics -19.2 W/m2).

Method: bin the (pred-truth) FSDS bias by 5-degree latitude bands using the
SAME test samples/methodology as the audit script, smooth with a rolling
average, and save a (lat_centers, bias) lookup table. model_server.py's new
--qnet_debias flag will interpolate this onto the native grid and SUBTRACT it
from fsds at serving time (additive correction -- a bias in W/m2 is not
scale-proportional to FSDS magnitude, so additive is the correct form here,
and avoids blowing up during polar night when FSDS_raw~0).

Read-only diagnostic + one small output file. Does not touch model_server.py
or any coupled run.
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
I_FSDS = 5
N_SAMPLES = 300
LAT_EDGES = np.arange(-90, 91, 5.0)   # 5-degree bins

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ref = xr.open_dataset(REF_POP_FILE)
    TLAT = _ref.TLAT.values
    TAREA = np.nan_to_num(_ref.TAREA.values.astype(np.float64))
    # OPEN-WATER area, not total area: model_server.py applies this correction to
    # fsds/lhflx BEFORE cice_coupler.apply() multiplies the whole flux by (1-aice)
    # for ice-covered cells (fsds_o = ow*fsds, lhflx_o = ow*lhflx). A zero-mean fit
    # weighted by raw TAREA silently assumes 100% delivery everywhere; in reality the
    # high-lat heat-REMOVAL part of the correction is under-delivered wherever ice is
    # present while the tropical heat-ADDITION part (ice-free) is delivered in full,
    # leaking a net ~0.3-1.0 W/m2 of extra warming (confirmed 2026-08-25 against the
    # live coupled run's own IFRAC output -- this explains the +2.6K warm plateau).
    _aice_clim = np.load(Path(MODEL_DIR) / "aice_clim_baseline.npy")   # (12, nj, ni)
    _aice_annual = _aice_clim.mean(axis=0)
    OW_AREA = TAREA * np.clip(1.0 - _aice_annual, 0.0, 1.0)

    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    Yr = np.load(f"{CACHE}/Y_rad.npy", mmap_mode="r")
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

    pred_fsds = np.empty((len(sel), x_phys.shape[2], x_phys.shape[3]), np.float64)
    bs = 8
    with torch.no_grad():
        for s in range(0, len(sel), bs):
            xn = (x_phys[s:s+bs] - xm) / (xs + 1e-8)
            yn = model(torch.from_numpy(xn.astype(np.float32)).to(device)).cpu().numpy().astype(np.float64)
            y = yn * ys + ym
            pred_fsds[s:s+bs] = y[:, I_FSDS] / DT6H

    truth_fsds = Yr[sel][:, 0].astype(np.float64) / DT6H
    bias = pred_fsds - truth_fsds

    finite = np.isfinite(bias) & m_ok
    extreme = finite & (np.abs(bias) > 5000)
    m_ok = m_ok & finite & ~extreme
    print(f"excluded {int((mask[sel]>0.5).sum() - m_ok.sum())} non-finite/extreme cells "
          f"of {(mask[sel]>0.5).sum()}")

    lat_b = np.broadcast_to(TLAT[None], m_ok.shape)

    # real ocean area per band (TAREA sum over the actual gx1v7 mask, NOT
    # a cell-count or cos(lat) proxy) -- used both to report n and, below,
    # to zero-mean the fitted curve against the TRUE grid weighting.
    centers, means, counts, areas = [], [], [], []
    for i in range(len(LAT_EDGES) - 1):
        lo, hi = LAT_EDGES[i], LAT_EDGES[i+1]
        bm = m_ok & (lat_b >= lo) & (lat_b < hi)
        n = int(bm.sum())
        centers.append(0.5 * (lo + hi))
        counts.append(n)
        means.append(float(bias[bm].mean()) if n > 0 else np.nan)
        band_mask_2d = (TLAT >= lo) & (TLAT < hi)
        areas.append(float(OW_AREA[band_mask_2d].sum()))
    centers = np.array(centers); means = np.array(means); counts = np.array(counts)
    areas = np.array(areas)

    # fill any empty bins by linear interpolation over neighbors, then smooth
    valid = np.isfinite(means)
    means_filled = np.interp(centers, centers[valid], means[valid])
    # 3-bin (15-degree) centered rolling average for a smooth, non-overfit curve
    kernel = np.array([1, 1, 1]) / 3.0
    smoothed = np.convolve(means_filled, kernel, mode="same")
    smoothed[0], smoothed[-1] = means_filled[0], means_filled[-1]   # avoid edge under-smoothing

    print(f"\n{'lat_center':>10s}{'n_cells':>10s}{'raw_bias':>12s}{'smoothed':>12s}")
    for c, n, mv, sv in zip(centers, counts, means, smoothed):
        print(f"{c:>10.1f}{n:>10d}{mv:>12.2f}{sv:>12.2f}")

    # Zero-mean the curve so the correction is a pure redistribution and
    # injects no net global-mean heat once actually DELIVERED through the
    # ice-coupled pipeline. Weight by OPEN-WATER area (TAREA*(1-aice_clim)),
    # not raw TAREA: the earlier cos(lat)- and then TAREA-only zero-mean both
    # left a residual warm drift because the correction is applied before the
    # (1-aice) ice masking, so a raw-TAREA zero-mean over-credits delivery at
    # icy high latitudes.
    w = areas
    global_mean = float(np.sum(smoothed * w) / np.sum(w))
    smoothed_zeromean = smoothed - global_mean
    print(f"\nopen-water-area-weighted global mean of raw fit: {global_mean:+.2f} W/m2 -> removed (zero-sum redistribution only)")

    out_path = Path(MODEL_DIR) / "fsds_debias_lat.npy"
    np.save(out_path, np.stack([centers, smoothed_zeromean]))
    print(f"\nwrote {out_path}  (shape {np.stack([centers, smoothed]).shape}: [lat_centers, bias_to_subtract])")

    # quick sanity check: SO/NH/Tropics band means of the fitted curve
    so = smoothed_zeromean[(centers >= -90) & (centers < -45)].mean()
    nh = smoothed_zeromean[(centers >= 45) & (centers <= 90)].mean()
    tr = smoothed_zeromean[(centers >= -30) & (centers < 30)].mean()
    print(f"\nFitted correction (subtract this from FSDS): SO={so:+.2f}  NH={nh:+.2f}  Tropics={tr:+.2f} W/m2")

if __name__ == "__main__":
    main()
