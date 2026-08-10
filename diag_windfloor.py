"""diag_windfloor.py — isolate the +0.58 K climatology-wind floor at the flux level.

The residual coupled run converges to a +0.58 K warm floor; the SAME v15 flux net
with TRUE prescribed winds gives +0.002 K. So the floor is exactly the flux-level
consequence of feeding the net CLIMATOLOGY winds (clim_gx1v7[month]) instead of the
true synoptic winds (X_atm) at the same ocean state. This quantifies that:

  at TRUTH SST over all of 1980, run the net with (a) true winds, (b) clim winds,
  and compare Qnet=SH+LH+FSDS+FLDS -- to each other and to CREDIT truth (Y,Y_rad) --
  split by latitude band and by component.

  dQ_wind = Qnet(clim) - Qnet(true)   [+ = clim winds push MORE heat into ocean]

is the driver of the floor; bias_true = Qnet(net,true) - Qnet(truth) is the net's
own static emulation bias for reference. All in W/m2, area-weighted, over ocean.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, "/glade/u/home/praggarwal/couple")
sys.path.insert(0, "camulator_ud/climate")
from train_unet import UNet

CACHE = "/glade/work/praggarwal/couple_cache_gx1v7_mem24h"
SENSREG = "output/output_unet_gx1v7_sensreg"
CLIM = "output/output_atm_emulator_residual/clim_gx1v7.npy"
POPH = ("/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_v15atmemuResidual1yr/run/"
        "g.e21.MEMO_GIAF_v01.pop.h.1980-01.nc")
DT6H = 21600.0
ROWS_PER_YEAR = 1456          # 1460 windows - 4 memory-lag windows
# model output channels: 0 TAUX 1 TAUY 2 SHFLX 3 LHFLX 4 QFLX 5 FSDS 6 FLDS 7 PRECT
I_SH, I_LH, I_FSDS, I_FLDS = 2, 3, 5, 6
MONTH_LEN = np.array([31,28,31,30,31,30,31,31,30,31,30,31])  # noleap


def row_month(w):
    """0-based calendar month for within-year row index w (noleap, +4 lag)."""
    doy = (w + 4) // 4                      # 0-based day of year
    return int(np.searchsorted(np.cumsum(MONTH_LEN), doy, side="right"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=4, help="row stride within 1980")
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    import xarray as xr
    with xr.open_dataset(POPH, decode_times=False) as ds:
        tlat = ds["TLAT"].values.astype(np.float64)
        tarea = ds["TAREA"].values.astype(np.float64)

    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    Xa = np.load(f"{CACHE}/X_atm.npy", mmap_mode="r")
    Y = np.load(f"{CACHE}/Y.npy", mmap_mode="r")
    Yr = np.load(f"{CACHE}/Y_rad.npy", mmap_mode="r")
    mask = np.load(f"{CACHE}/mask.npy", mmap_mode="r")
    clim = np.load(CLIM).astype(np.float64)          # (12,5,H,W)

    d = Path(SENSREG)
    import json
    cfg = json.loads((d / "model_config.json").read_text())
    m = UNet(n_in=cfg["n_in"], n_out=cfg["n_out"], base=cfg.get("base", 64), dropout=0.0).to(dev)
    m.load_state_dict(torch.load(d / "best_model.pt", map_location=dev, weights_only=True))
    m.eval()
    nz = np.load(f"{SENSREG}/normalizer.npz")
    xm = nz["x_mean"][None, :, None, None]; xs = nz["x_std"][None, :, None, None]
    ym = nz["y_mean"][None, :, None, None]; ys = nz["y_std"][None, :, None, None]

    rows = np.arange(0, ROWS_PER_YEAR, args.stride)
    months = np.array([row_month(w) for w in rows])
    print(f"device={dev}  {len(rows)} rows of 1980 (stride {args.stride})")

    H, W = 384, 320
    # accumulators: sum over samples of per-cell component (W/m2), for true & clim
    def zero(): return {k: np.zeros((H, W)) for k in ["SH", "LH", "FSDS", "FLDS"]}
    acc_true, acc_clim, acc_truth = zero(), zero(), zero()
    nacc = np.zeros((H, W))                          # ocean-sample count per cell

    def run(xphys):
        yb = np.empty((xphys.shape[0], cfg["n_out"], H, W), np.float64)
        with torch.no_grad():
            for s in range(0, xphys.shape[0], args.bs):
                xn = (xphys[s:s+args.bs] - xm) / (xs + 1e-8)
                yn = m(torch.from_numpy(xn.astype(np.float32)).to(dev)).cpu().numpy().astype(np.float64)
                yb[s:s+args.bs] = yn * ys + ym
        return yb

    B = 40
    for i in range(0, len(rows), B):
        idx = rows[i:i+B]; mo = months[i:i+B]
        raw = X[idx].astype(np.float64)             # (n,6,H,W)
        a_true = Xa[idx].astype(np.float64)         # (n,5,H,W)
        a_clim = clim[mo]                           # (n,5,H,W)
        dsst = ((raw[:, 0] - raw[:, 3]) / 86400.0)[:, None]
        base = np.concatenate([raw[:, 0:5], dsst], axis=1)      # (n,6,H,W)
        x_true = np.concatenate([base, a_true], axis=1)
        x_clim = np.concatenate([base, a_clim], axis=1)
        y_true = run(x_true); y_clim = run(x_clim)
        # truth fluxes from cache (same units as model outputs: J/m2/6h -> W/m2)
        yt_sh = Y[idx, 2].astype(np.float64); yt_lh = Y[idx, 3].astype(np.float64)
        yt_fs = Yr[idx, 0].astype(np.float64); yt_fl = Yr[idx, 1].astype(np.float64)
        mk = (mask[idx] > 0.5)
        for n in range(len(idx)):
            w = mk[n]
            nacc += w
            for lbl, ch, tv in [("SH", I_SH, yt_sh[n]), ("LH", I_LH, yt_lh[n]),
                                ("FSDS", I_FSDS, yt_fs[n]), ("FLDS", I_FLDS, yt_fl[n])]:
                acc_true[lbl] += np.where(w, y_true[n, ch] / DT6H, 0.0)
                acc_clim[lbl] += np.where(w, y_clim[n, ch] / DT6H, 0.0)
                acc_truth[lbl] += np.where(w, tv / DT6H, 0.0)
        print(f"  {i+len(idx)}/{len(rows)}", end="\r")
    print()

    # per-cell time-means (ocean cells only)
    ok = nacc > 0
    def cellmean(acc):
        out = {k: np.full((H, W), np.nan) for k in acc}
        for k in acc:
            out[k][ok] = acc[k][ok] / nacc[ok]
        return out
    mt, mc, mtr = cellmean(acc_true), cellmean(acc_clim), cellmean(acc_truth)

    bands = [("GLOBAL", ok),
             ("SO <-45", ok & (tlat < -45)),
             ("SO <-60", ok & (tlat < -60)),
             ("TROP -45..45", ok & (tlat >= -45) & (tlat <= 45)),
             ("NH >45", ok & (tlat > 45))]

    def wmean(field, bmask):
        w = tarea[bmask]; v = field[bmask]
        good = ~np.isnan(v)
        return float(np.average(v[good], weights=w[good])) if good.any() else np.nan

    def qnet(md): return md["SH"] + md["LH"] + md["FSDS"] + md["FLDS"]
    q_true, q_clim, q_truth = qnet(mt), qnet(mc), qnet(mtr)

    print("=" * 78)
    print("Qnet = SH+LH+FSDS+FLDS  [W/m2, area-wtd over ocean, 1980 mean]  sign per model conv.")
    print(f"{'band':<14}{'net@true':>10}{'net@clim':>10}{'TRUTH':>10}"
          f"{'dQ_wind':>10}{'bias_true':>11}")
    print(f"{'':14}{'':>10}{'':>10}{'':>10}{'(clim-true)':>10}{'(true-truth)':>11}")
    for nm, bmask in bands:
        qt, qc, qr = wmean(q_true, bmask), wmean(q_clim, bmask), wmean(q_truth, bmask)
        print(f"{nm:<14}{qt:>10.2f}{qc:>10.2f}{qr:>10.2f}{qc-qt:>10.2f}{qt-qr:>11.2f}")

    print("\ndQ_wind (clim-true) by COMPONENT [W/m2, + = clim pushes more heat into ocean]:")
    print(f"{'band':<14}{'dSH':>9}{'dLH':>9}{'dFSDS':>9}{'dFLDS':>9}")
    for nm, bmask in bands:
        row = [wmean(mc[k] - mt[k], bmask) for k in ["SH", "LH", "FSDS", "FLDS"]]
        print(f"{nm:<14}" + "".join(f"{v:>9.2f}" for v in row))

    print("\nbias_true (net@true - TRUTH) by COMPONENT [W/m2, static emulation bias]:")
    print(f"{'band':<14}{'dSH':>9}{'dLH':>9}{'dFSDS':>9}{'dFLDS':>9}")
    for nm, bmask in bands:
        row = [wmean(mt[k] - mtr[k], bmask) for k in ["SH", "LH", "FSDS", "FLDS"]]
        print(f"{nm:<14}" + "".join(f"{v:>9.2f}" for v in row))


if __name__ == "__main__":
    main()
