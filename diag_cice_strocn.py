"""
diag_cice_strocn.py
===================
Spatial sanity check for the trained CICE emulator (output/output_cice_v1),
before we commit to the coupling wiring.  Runs best_model.pt on held-out
(test-tail) samples and maps predicted vs. truth ice->ocean stress
strocnx/strocny, with a zoom on Hudson Bay -- the region where the standalone
MEMO->POP run blows up.  The question this answers: does the emulator put the
momentum sink in the right place with the right magnitude?

Outputs (into out_dir):
    strocn_global_nh.png     global NH: |strocn| truth vs pred vs error
    strocn_hudson.png        Hudson Bay zoom: strocnx & strocny, truth/pred/err
    strocn_scatter.png       pred vs truth scatter over the Hudson box
    diag_stats.json          box-restricted R2/rmse/bias for strocnx/strocny
"""
import argparse, glob, json
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_unet import UNet, Normalizer

ICE_ROOT = "/glade/campaign/cesm/collections/CESM2-LE/ice/proc/tseries/month_1"

# Hudson Bay box (deg): lat 50-65 N, lon 265-285 E (=95-75 W)
HB_LAT = (50.0, 65.0)
HB_LON = (263.0, 285.0)


def load_grid():
    p = sorted(glob.glob(f"{ICE_ROOT}/strocnx/*LE2-1231.002*.nc"))[0]
    ds = xr.open_dataset(p, decode_timedelta=False)
    return ds["TLAT"].values, ds["TLON"].values      # (384,320) each


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="/glade/derecho/scratch/praggarwal/couple_cache_cice")
    ap.add_argument("--out_dir", default="output/output_cice_v1")
    ap.add_argument("--n_show", type=int, default=1, help="how many top-ice test samples to map")
    args = ap.parse_args()

    out = Path(args.out_dir); cache = Path(args.cache_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X = np.load(cache / "X.npy", mmap_mode="r")
    Y = np.load(cache / "Y.npy", mmap_mode="r")
    mask = np.load(cache / "mask.npy", mmap_mode="r")
    meta = json.loads((cache / "meta.json").read_text())
    ov = meta["output_vars"]
    ix, iy = ov.index("strocnx"), ov.index("strocny")
    N = X.shape[0]
    nva = int(0.9 * N)
    idx_te = np.arange(nva, N)                          # held-out tail
    print(f"N={N}  test tail = [{nva},{N})  ({len(idx_te)} months)")

    norm = Normalizer.load(out / "normalizer.npz")
    model = UNet(n_in=X.shape[1], n_out=Y.shape[1], base=meta.get("base", 64)).to(dev)
    model.load_state_dict(torch.load(out / "best_model.pt", map_location=dev))
    model.eval()

    TLAT, TLON = load_grid()
    hb = ((TLAT >= HB_LAT[0]) & (TLAT <= HB_LAT[1]) &
          (TLON >= HB_LON[0]) & (TLON <= HB_LON[1]))     # (384,320) bool
    print(f"Hudson box cells: {hb.sum()}")

    # --- run the model on the whole test tail, denormalise ---------------------
    preds, truths, masks = [], [], []
    with torch.no_grad():
        for i in idx_te:
            x = np.asarray(X[i], np.float32)
            xn = (x - norm.x_mean[:, None, None]) / (norm.x_std[:, None, None] + 1e-8)
            p = model(torch.from_numpy(xn[None]).to(dev))[0].cpu().numpy()
            p = p * norm.y_std[:, None, None] + norm.y_mean[:, None, None]   # denorm
            preds.append(p); truths.append(np.asarray(Y[i], np.float32))
            masks.append(np.asarray(mask[i], bool))
    P = np.stack(preds); T = np.stack(truths); M = np.stack(masks)   # (n,6,nj,ni)

    # pick the sample with the most ice-stress activity inside the Hudson box
    hb_mag = np.sqrt(T[:, ix]**2 + T[:, iy]**2)
    hb_score = (hb_mag * hb[None]).reshape(len(idx_te), -1).sum(1)
    order = np.argsort(hb_score)[::-1]
    s = int(order[0])
    print(f"most-active Hudson sample: local idx {s} (global {idx_te[s]}), score {hb_score[s]:.2f}")

    # ---- box-restricted skill (all test months, ocean+box cells) --------------
    boxm = M & hb[None]
    stats = {}
    for name, c in (("strocnx", ix), ("strocny", iy)):
        t = T[:, c][boxm]; p = P[:, c][boxm]
        ss_res = np.sum((t - p)**2); ss_tot = np.sum((t - t.mean())**2)
        stats[name] = {
            "r2_hudson": float(1 - ss_res / ss_tot),
            "rmse_hudson": float(np.sqrt(np.mean((t - p)**2))),
            "bias_hudson": float((p - t).mean()),
            "corr_hudson": float(np.corrcoef(t, p)[0, 1]),
            "truth_std": float(t.std()),
        }
    (out / "diag_stats.json").write_text(json.dumps(stats, indent=2))
    print("Hudson-box skill:", json.dumps(stats, indent=2))

    # ---------- helpers: mask land to NaN for clean plotting -------------------
    def masked(arr, mk): return np.where(mk, arr, np.nan)

    lo, hi = TLON.min(), TLON.max()

    # ==== FIG 1: global NH |strocn| truth / pred / error =======================
    magT = masked(np.sqrt(T[s, ix]**2 + T[s, iy]**2), M[s])
    magP = masked(np.sqrt(P[s, ix]**2 + P[s, iy]**2), M[s])
    vmax = np.nanpercentile(magT, 99.5)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for a, d, ttl in ((ax[0], magT, "truth"), (ax[1], magP, "pred")):
        im = a.imshow(d, origin="lower", vmin=0, vmax=vmax, cmap="viridis")
        a.set_title(f"|strocn| {ttl}"); plt.colorbar(im, ax=a, shrink=0.8)
    err = magP - magT
    e = np.nanpercentile(np.abs(err), 99)
    im = ax[2].imshow(err, origin="lower", vmin=-e, vmax=e, cmap="RdBu_r")
    ax[2].set_title("pred - truth"); plt.colorbar(im, ax=ax[2], shrink=0.8)
    for a in ax: a.set_xlabel("ni"); a.set_ylabel("nj")
    fig.suptitle(f"|strocn| N/m^2  (test sample global idx {idx_te[s]})")
    fig.tight_layout(); fig.savefig(out / "strocn_global_nh.png", dpi=110); plt.close(fig)

    # ==== FIG 2: Hudson Bay zoom, strocnx & strocny truth/pred/err =============
    # restrict index window to rows/cols touching the box for a tight crop
    rows = np.where(hb.any(1))[0]; cols = np.where(hb.any(0))[0]
    r0, r1, c0, c1 = rows.min(), rows.max()+1, cols.min(), cols.max()+1
    fig, ax = plt.subplots(2, 3, figsize=(14, 9))
    for row, (name, c) in enumerate((("strocnx", ix), ("strocny", iy))):
        t = masked(T[s, c], M[s])[r0:r1, c0:c1]
        p = masked(P[s, c], M[s])[r0:r1, c0:c1]
        v = np.nanpercentile(np.abs(t), 99); v = v if v > 0 else 1e-3
        for a, d, ttl in ((ax[row, 0], t, "truth"), (ax[row, 1], p, "pred")):
            im = a.imshow(d, origin="lower", vmin=-v, vmax=v, cmap="RdBu_r")
            a.set_title(f"{name} {ttl}"); plt.colorbar(im, ax=a, shrink=0.8)
        er = p - t
        im = ax[row, 2].imshow(er, origin="lower", vmin=-v, vmax=v, cmap="RdBu_r")
        ax[row, 2].set_title(f"{name} pred-truth"); plt.colorbar(im, ax=ax[row, 2], shrink=0.8)
    fig.suptitle(f"Hudson Bay strocn N/m^2 (global idx {idx_te[s]})")
    fig.tight_layout(); fig.savefig(out / "strocn_hudson.png", dpi=110); plt.close(fig)

    # ==== FIG 3: pred-vs-truth scatter over the Hudson box (all test months) ===
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, (name, c) in zip(ax, (("strocnx", ix), ("strocny", iy))):
        t = T[:, c][boxm]; p = P[:, c][boxm]
        a.hexbin(t, p, gridsize=60, bins="log", cmap="magma")
        lim = np.nanpercentile(np.abs(t), 99.5)
        a.plot([-lim, lim], [-lim, lim], "c--", lw=1)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
        a.set_xlabel("truth"); a.set_ylabel("pred")
        a.set_title(f"{name}  R2={stats[name]['r2_hudson']:.3f}  "
                    f"corr={stats[name]['corr_hudson']:.3f}")
    fig.suptitle("Hudson-box ice->ocean stress: pred vs truth (all test months)")
    fig.tight_layout(); fig.savefig(out / "strocn_scatter.png", dpi=110); plt.close(fig)

    print("wrote:", out / "strocn_global_nh.png", out / "strocn_hudson.png",
          out / "strocn_scatter.png", out / "diag_stats.json")


if __name__ == "__main__":
    main()
