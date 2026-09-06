"""validate_qnet_debias_combined.py — offline validation of BOTH the FSDS debias
(fsds_debias_lat.npy) and the new non-FSDS/LHFLX debias (qnet_debias_lh_lat.npy)
applied together, on a FRESH held-out sample. Reports the TOTAL Qnet bias
(SH+LH+FSDS+FLDS) per region, raw vs corrected.
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
N_SAMPLES = 200
SEED = 7   # same fresh seed as validate_fsds_debias.py
I_SH, I_LH, I_FSDS, I_FLDS = 2, 3, 5, 6

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    TLAT = xr.open_dataset(REF_POP_FILE).TLAT.values
    fsds_curve = np.load(f"{MODEL_DIR}/fsds_debias_lat.npy")
    lh_curve = np.load(f"{MODEL_DIR}/qnet_debias_lh_lat.npy")
    fsds_corr = np.interp(TLAT.ravel(), fsds_curve[0], fsds_curve[1]).reshape(TLAT.shape)
    lh_corr = np.interp(TLAT.ravel(), lh_curve[0], lh_curve[1]).reshape(TLAT.shape)

    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    Y = np.load(f"{CACHE}/Y.npy", mmap_mode="r")
    Yr = np.load(f"{CACHE}/Y_rad.npy", mmap_mode="r")
    mask = np.load(f"{CACHE}/mask.npy", mmap_mode="r")
    test_idx = np.load(f"{MODEL_DIR}/test_indices.npy")
    sel = np.sort(np.random.default_rng(SEED).choice(test_idx, size=min(N_SAMPLES, len(test_idx)), replace=False))
    print(f"device={device}  n={len(sel)} FRESH test samples (seed={SEED})")

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

    pred = np.empty((len(sel), 4, x_phys.shape[2], x_phys.shape[3]), np.float64)
    bs = 8
    with torch.no_grad():
        for s in range(0, len(sel), bs):
            xn = (x_phys[s:s+bs] - xm) / (xs + 1e-8)
            yn = model(torch.from_numpy(xn.astype(np.float32)).to(device)).cpu().numpy().astype(np.float64)
            y = yn * ys + ym
            for k, ch in enumerate([I_SH, I_LH, I_FSDS, I_FLDS]):
                pred[s:s+bs, k] = y[:, ch] / DT6H

    truth = np.empty_like(pred)
    truth[:, 0] = Y[sel][:, 2].astype(np.float64) / DT6H
    truth[:, 1] = Y[sel][:, 3].astype(np.float64) / DT6H
    truth[:, 2] = Yr[sel][:, 0].astype(np.float64) / DT6H
    truth[:, 3] = Yr[sel][:, 1].astype(np.float64) / DT6H

    qnet_raw = pred.sum(axis=1) - truth.sum(axis=1)
    pred_corr = pred.copy()
    pred_corr[:, 2] -= fsds_corr[None]   # local stack order [SH,LH,FSDS,FLDS] -> FSDS is index 2
    pred_corr[:, 1] -= lh_corr[None]     # LHFLX is index 1
    qnet_corr = pred_corr.sum(axis=1) - truth.sum(axis=1)

    finite = np.isfinite(qnet_raw) & np.isfinite(qnet_corr) & m_ok
    extreme = finite & (np.abs(qnet_raw) > 5000)
    m_ok2 = m_ok & finite & ~extreme

    lat_b = np.broadcast_to(TLAT[None], m_ok2.shape)
    bands = {
        "Southern Ocean (<-45)": lat_b < -45,
        "NH high-lat (>45)":     lat_b > 45,
        "Tropics (|lat|<30)":    np.abs(lat_b) < 30,
        "Global":                np.ones_like(m_ok2, dtype=bool),
    }
    print(f"\n{'Region':26s}{'raw Qnet':>12s}{'corrected':>12s}{'n_cells':>10s}")
    for bname, bmask in bands.items():
        sm = bmask & m_ok2
        n = int(sm.sum())
        if n == 0:
            print(f"{bname:26s}  no cells"); continue
        rb = float(qnet_raw[sm].mean()); cb = float(qnet_corr[sm].mean())
        print(f"{bname:26s}{rb:>+12.2f}{cb:>+12.2f}{n:>10d}")

if __name__ == "__main__":
    main()
