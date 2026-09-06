"""audit_qnet_bias_pz220.py — offline ABSOLUTE Qnet bias audit for pz220f0b sensreg,
broken down by latitude band (Southern Ocean / NH high-lat / Tropics / Global).

Root-cause step 1 of the tropics-cool/poles-warm drift debugging plan (2026-08-24):
memo-so-flux-excess-diagnosed.md measured a regional bias (SO +24.1, NH +24.2,
Tropics -5.2, Global +7.7 W/m2) for an earlier checkpoint. This script re-measures
the SAME quantity (raw U-Net output vs the actual training target, i.e. the CAM6/
CREDIT-derived truth on the gx1v7 cache -- no SST perturbation, no serving-time
--ocn_albedo/--flux_clip corrections) against the CURRENT deployed checkpoint
(output_unet_gx1v7_atm_aux_sensreg/best_model.pt, what run_pop_sensreg_50yr.pbs
actually loads) to confirm the defect is still live before designing a fix.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch, sys, xarray as xr
sys.path.insert(0, "camulator_ud/climate")
from train_unet import UNet

CACHE = "/glade/work/praggarwal/couple_cache_gx1v7_mem24h"
DT6H = 21600.0
REF_POP_FILE = ("/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_sensreg50yr/"
                 "run/g.e21.MEMO_GIAF_v01.pop.h.1980-01.nc")

def load_model(out_dir, device):
    d = Path(out_dir); cfg = json.loads((d / "model_config.json").read_text())
    m = UNet(n_in=cfg["n_in"], n_out=cfg["n_out"], base=cfg.get("base", 64), dropout=0.0).to(device)
    m.load_state_dict(torch.load(d / "best_model_snap.pt", map_location=device, weights_only=True))
    m.eval(); return m, cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model_dir", default="output/output_unet_gx1v7_atm_aux_sensreg")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    TLAT = xr.open_dataset(REF_POP_FILE).TLAT.values   # (384,320)

    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    Y = np.load(f"{CACHE}/Y.npy", mmap_mode="r")           # TAUX,TAUY,SHFLX,LHFLX,QFLX
    Yr = np.load(f"{CACHE}/Y_rad.npy", mmap_mode="r")      # FSDS_J,FLDS_J
    mask = np.load(f"{CACHE}/mask.npy", mmap_mode="r")
    test_idx = np.load(f"{args.model_dir}/test_indices.npy")
    sel = np.sort(np.random.default_rng(0).choice(test_idx, size=min(args.n, len(test_idx)), replace=False))
    print(f"device={device}  n={len(sel)} test samples  model_dir={args.model_dir}")

    model, cfg = load_model(args.model_dir, device)
    nz = np.load(f"{args.model_dir}/normalizer.npz")
    xm = nz["x_mean"][None,:,None,None]; xs = nz["x_std"][None,:,None,None]
    ym = nz["y_mean"][None,:,None,None]; ys = nz["y_std"][None,:,None,None]
    I_SH, I_LH, I_FSDS, I_FLDS = 2, 3, 5, 6

    m_ok = mask[sel] > 0.5
    raw = X[sel].astype(np.float64)          # SST,ICEFRAC,SOLIN,SST_prev,ICEFRAC_prev,SOLIN_prev
    # pz220f0b's 6th input channel is dSST_dt, NOT the raw cache's SOLIN_prev --
    # confirmed via audit_dqdsst_pz220.py's build_inputs(). Feeding raw SOLIN_prev
    # (scale ~0-1200) into a channel normalized around dSST_dt (scale ~1e-5) is
    # wildly out-of-distribution and was producing garbage billions-of-W/m2 output,
    # worst in the tropics (highest SOLIN) -- this was a bug in THIS script, not
    # a real model defect (verified 2026-08-24).
    dsst = ((raw[:, 0] - raw[:, 3]) / 86400.0)[:, None]
    x_phys = np.concatenate([raw[:, 0:5], dsst], axis=1)

    pred = np.zeros((len(sel), 4, x_phys.shape[2], x_phys.shape[3]), np.float64)
    bs = 8
    with torch.no_grad():
        for s in range(0, len(sel), bs):
            xn = (x_phys[s:s+bs] - xm) / (xs + 1e-8)
            yn = model(torch.from_numpy(xn.astype(np.float32)).to(device)).cpu().numpy().astype(np.float64)
            y = yn * ys + ym
            for k, ch in enumerate([I_SH, I_LH, I_FSDS, I_FLDS]):
                pred[s:s+bs, k] = y[:, ch] / DT6H

    truth = np.zeros_like(pred)
    truth[:, 0] = Y[sel][:, 2].astype(np.float64) / DT6H     # SHFLX
    truth[:, 1] = Y[sel][:, 3].astype(np.float64) / DT6H     # LHFLX
    truth[:, 2] = Yr[sel][:, 0].astype(np.float64) / DT6H    # FSDS_J
    truth[:, 3] = Yr[sel][:, 1].astype(np.float64) / DT6H    # FLDS_J

    bias = pred - truth   # (N,4,H,W): SH,LH,FSDS,FLDS bias, W/m2, +=MEMO over-delivers
    finite = np.isfinite(bias).all(axis=1) & m_ok
    n_bad = int((m_ok & ~finite).sum())
    if n_bad:
        print(f"WARNING: {n_bad} masked cells have non-finite pred/truth -- excluding from stats")
    extreme = m_ok & finite & (np.abs(bias).max(axis=1) > 5000)
    n_extreme = int(extreme.sum())
    if n_extreme:
        print(f"WARNING: {n_extreme} masked cells have |component bias| > 5000 W/m2 "
              f"(likely land-mask edge/coastal artifacts) -- excluding from stats "
              f"(max found: {np.abs(bias)[np.broadcast_to(extreme[:,None],bias.shape)].max():.1f})")
    m_ok = m_ok & finite & ~extreme

    lat_b = np.broadcast_to(TLAT[None], m_ok.shape)
    bands = {
        "Southern Ocean (<-45)": lat_b < -45,
        "NH high-lat (>45)":     lat_b > 45,
        "Tropics (|lat|<30)":    np.abs(lat_b) < 30,
        "Global":                np.ones_like(m_ok, dtype=bool),
    }
    comp_names = ["SHFLX", "LHFLX", "FSDS", "FLDS"]

    print(f"\n{'Region':26s}{'Qnet bias':>12s}" + "".join(f"{c:>10s}" for c in comp_names) + f"{'n_cells':>10s}")
    results = {}
    for bname, bmask in bands.items():
        sel_mask = bmask & m_ok
        n = sel_mask.sum()
        if n == 0:
            print(f"{bname:26s}  no cells"); continue
        comp_bias = [float(bias[:, k][sel_mask].mean()) for k in range(4)]
        qnet_bias = sum(comp_bias)
        results[bname] = dict(qnet=qnet_bias, SHFLX=comp_bias[0], LHFLX=comp_bias[1],
                               FSDS=comp_bias[2], FLDS=comp_bias[3], n_cells=int(n))
        print(f"{bname:26s}{qnet_bias:>+12.2f}" + "".join(f"{c:>+10.2f}" for c in comp_bias) + f"{n:>10d}")

    print("\nReference (memo-so-flux-excess-diagnosed.md, earlier checkpoint):")
    print("  Southern Ocean +24.1, NH high-lat +24.2, Tropics -5.2, Global +7.7 W/m2")
    print("  (SO component split: LHFLX +13.2, FSDS +9.8, SHFLX +4.1, FLDS -3.0)")

    out = {"model_dir": args.model_dir, "n_samples": int(len(sel)), "bands": results}
    outp = Path(args.model_dir) / "qnet_bias_audit.json"
    json.dump(out, open(outp, "w"), indent=2)
    print(f"\nwrote {outp}")

if __name__ == "__main__":
    main()
