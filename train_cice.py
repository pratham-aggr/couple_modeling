"""
train_cice.py
=============
Train the CICE sea-ice emulator on the gx1v7 monthly cache built by
preprocess_cice.py.  A trimmed companion to train_unet.py (MEMO): it REUSES the
grid-agnostic pieces from train_unet — UNet, Normalizer, masked_mse,
compute_metrics, ModelEMA — and only swaps in the CICE cache, channel counts,
and a temporal (leakage-free) split.

Model: same 5-level UNet as MEMO.  gx1v7 is 384 x 320, both divisible by 32, so
the architecture drops in unchanged; CircPad's circular wrap on the last axis is
correct for the gx1v7 `ni` longitude wrap.  No cos(lat) weighting is applied
(POP/CICE cells are already area-comparable and the loss is masked to ocean).

    IN  (7ch): SST, uatm, vatm, aice_prev, hi_prev, uvel_prev, vvel_prev
    OUT (6ch): aice, hi, uvel, vvel, strocnx, strocny

Usage:
    python train_cice.py --cache_dir /glade/derecho/scratch/praggarwal/couple_cache_cice \
        --out_dir output/output_cice_v1 --epochs 60
    python train_cice.py --cache_dir ... --epochs 2 --dry_run    # wiring check
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from train_unet import UNet, Normalizer, masked_mse, compute_metrics, ModelEMA


class CiceDataset(Dataset):
    """Yields (x_n, y_n, mask) normalised float32 tensors.  Optional circular
    longitude roll augment (gx1v7 `ni` is periodic), train-split only.

    res_pairs : list of (out_idx, in_prev_idx).  For those output channels the
    target is converted to a RESIDUAL y[oc] -= x_raw[ic] BEFORE normalization, so
    the model predicts the (near-zero-mean) monthly increment on top of
    persistence.  This anchors amplitude at persistence and stops MSE from
    collapsing the bimodal ice concentration to a damped climatology."""
    def __init__(self, X, Y, mask, norm, augment=False, res_pairs=()):
        self.X, self.Y, self.mask = X, Y, mask
        self.norm, self.augment = norm, augment
        self.res_pairs = list(res_pairs)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = np.asarray(self.X[i], dtype=np.float32)
        y = np.asarray(self.Y[i], dtype=np.float32)
        m = np.asarray(self.mask[i], dtype=np.float32)
        for oc, ic in self.res_pairs:                # residual target (raw units)
            y[oc] = y[oc] - x[ic]
        x = (x - self.norm.x_mean[:, None, None]) / (self.norm.x_std[:, None, None] + 1e-8)
        y = (y - self.norm.y_mean[:, None, None]) / (self.norm.y_std[:, None, None] + 1e-8)
        if self.augment:
            s = int(np.random.randint(0, x.shape[-1]))
            if s:
                x = np.roll(x, s, -1); y = np.roll(y, s, -1); m = np.roll(m, s, -1)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(np.ascontiguousarray(m)))


def compute_norm(X, Y, mask, idx, res_pairs=()):
    """Per-channel mean/std over ocean points of the TRAIN split only.  For
    residual output channels the y stats are computed on (Y - X_prev) so the
    normalizer matches the residual target the model actually learns."""
    Xi = np.asarray(X[idx]); Yi = np.asarray(Y[idx]).copy()
    for oc, ic in res_pairs:
        Yi[:, oc] -= Xi[:, ic]
    m = mask[idx].astype(bool)                       # (n, nj, ni)
    def stats(A):                                    # A: (n, C, nj, ni)
        C = A.shape[1]
        mu = np.empty(C, np.float32); sd = np.empty(C, np.float32)
        for c in range(C):
            v = A[:, c][m]
            mu[c] = v.mean(); sd[c] = v.std() + 1e-6
        return mu, sd
    xm, xs = stats(Xi); ym, ys = stats(Yi)
    return Normalizer(xm, xs, ym, ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir", default="output/output_cice_v1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--residual", default="aice,hi",
                    help="comma list of output vars predicted as a RESIDUAL on their "
                         "*_prev input (amplitude anchor); '' to disable")
    ap.add_argument("--dry_run", action="store_true", help="tiny subset, 2 epochs")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)

    X = np.load(cache / "X.npy", mmap_mode="r")      # (N,7,nj,ni)
    Y = np.load(cache / "Y.npy", mmap_mode="r")      # (N,6,nj,ni)
    mask = np.load(cache / "mask.npy", mmap_mode="r")
    meta = json.loads((cache / "meta.json").read_text())
    N, n_in = X.shape[0], X.shape[1]; n_out = Y.shape[1]
    in_vars, out_vars = meta["input_vars"], meta["output_vars"]
    print(f"cache N={N} n_in={n_in} n_out={n_out} grid={X.shape[2]}x{X.shape[3]}")

    # residual targets: map output var -> its <var>_prev input channel
    res_names = [v for v in args.residual.split(",") if v]
    res_pairs = []
    for v in res_names:
        oc = out_vars.index(v); ic = in_vars.index(f"{v}_prev")
        res_pairs.append((oc, ic))
    if res_pairs:
        print(f"residual targets: {[(out_vars[o], in_vars[i]) for o, i in res_pairs]}")

    # temporal split (data are chronological): 80/10/10 train/val/test.
    ntr, nva = int(0.8 * N), int(0.9 * N)
    idx_tr = np.arange(0, ntr)
    idx_va = np.arange(ntr, nva)
    idx_te = np.arange(nva, N)
    if args.dry_run:
        idx_tr, idx_va, idx_te = idx_tr[:32], idx_va[:16], idx_te[:16]
        args.epochs = 2

    norm = compute_norm(X, Y, mask, idx_tr, res_pairs)
    norm.save(out / "normalizer.npz")

    def loader(idx, aug, shuf):
        ds = CiceDataset(X[idx], Y[idx], mask[idx], norm, augment=aug, res_pairs=res_pairs)
        return DataLoader(ds, batch_size=args.batch, shuffle=shuf,
                          num_workers=4, pin_memory=(dev == "cuda"), drop_last=shuf)
    dl_tr = loader(idx_tr, args.augment, True)
    dl_va = loader(idx_va, False, False)
    dl_te = loader(idx_te, False, False)

    model = UNet(n_in=n_in, n_out=n_out, base=args.base, dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ema = ModelEMA(model) if args.ema else None

    best_va = float("inf")
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for x_n, y_n, m in dl_tr:
            x_n, y_n, m = x_n.to(dev), y_n.to(dev), m.to(dev)
            opt.zero_grad()
            loss = masked_mse(model(x_n), y_n, m)
            loss.backward(); opt.step()
            if ema: ema.update(model)
            tot += loss.item(); nb += 1
        sched.step()

        model.eval(); vtot = 0.0; vnb = 0
        with torch.no_grad():
            for x_n, y_n, m in dl_va:
                vtot += masked_mse(model(x_n.to(dev)), y_n.to(dev), m.to(dev)).item(); vnb += 1
        va = vtot / max(vnb, 1)
        print(f"ep {ep:3d}  train {tot/max(nb,1):.4f}  val {va:.4f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s")
        if va < best_va:
            best_va = va
            torch.save(model.state_dict(), out / "best_model.pt")

    # ---- final test metrics on the held-out tail (ABSOLUTE space) ---------------
    # The dataloader yields RESIDUAL targets for aice/hi, so compute_metrics would
    # report residual R^2 (not comparable).  Reconstruct absolute fields
    # (aice = aice_prev + dpred, clipped) and score those + the amplitude bias
    # ratio (mean pred / mean truth over ice cells) -- the number the SST-only v1
    # got wrong (~0.35).
    if (out / "best_model.pt").exists():
        model.load_state_dict(torch.load(out / "best_model.pt", map_location=dev))
    ov = meta["output_vars"]
    ym_t = torch.tensor(norm.y_mean[:, None, None], device=dev)
    ys_t = torch.tensor(norm.y_std[:, None, None], device=dev)
    res_out2in = {oc: ic for oc, ic in res_pairs}
    ss_res = np.zeros(n_out); ss_tot = np.zeros(n_out)
    sum_p = np.zeros(n_out); sum_t = np.zeros(n_out); n_ice = np.zeros(n_out)
    tmean = np.zeros(n_out); tcnt = 0.0
    model.eval()
    with torch.no_grad():
        for x_n, y_n, m in dl_te:
            x_n = x_n.to(dev)
            pred = model(x_n) * ys_t + ym_t                    # denorm (residual for aice/hi)
            # reconstruct absolute predictions + absolute truth
            xr_ = x_n * torch.tensor(norm.x_std[:, None, None], device=dev) \
                  + torch.tensor(norm.x_mean[:, None, None], device=dev)
            truth = y_n.to(dev) * ys_t + ym_t
            for oc, ic in res_out2in.items():
                pred[:, oc] = pred[:, oc] + xr_[:, ic]
                truth[:, oc] = truth[:, oc] + xr_[:, ic]
            pred = pred.cpu().numpy(); truth = truth.cpu().numpy()
            mm = m.numpy().astype(bool)                        # (b, nj, ni)
            for c in range(n_out):
                pc, tc = pred[:, c], truth[:, c]
                sel = mm
                ss_res[c] += ((pc[sel] - tc[sel]) ** 2).sum()
                # ice-cell amplitude bias (aice/hi only meaningful where truth>0.15/0.1)
                ic_sel = sel & (tc > (0.15 if ov[c] == "aice" else 0.1))
                sum_p[c] += pc[ic_sel].sum(); sum_t[c] += tc[ic_sel].sum(); n_ice[c] += ic_sel.sum()
                tmean[c] += tc[sel].sum()
            tcnt += mm.sum()
    tmean /= max(tcnt, 1)
    # second pass for ss_tot needs the global mean; approximate with tmean per channel
    with torch.no_grad():
        for x_n, y_n, m in dl_te:
            x_n = x_n.to(dev)
            xr_ = x_n * torch.tensor(norm.x_std[:, None, None], device=dev) \
                  + torch.tensor(norm.x_mean[:, None, None], device=dev)
            truth = y_n.to(dev) * ys_t + ym_t
            for oc, ic in res_out2in.items():
                truth[:, oc] = truth[:, oc] + xr_[:, ic]
            truth = truth.cpu().numpy(); mm = m.numpy().astype(bool)
            for c in range(n_out):
                tc = truth[:, c]
                ss_tot[c] += ((tc[mm] - tmean[c]) ** 2).sum()
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    rmse = np.sqrt(ss_res / max(tcnt, 1))
    bias = np.where(sum_t > 0, sum_p / (sum_t + 1e-12), np.nan)

    metrics = {v: {"r2_abs": float(r2[i]), "rmse_abs": float(rmse[i]),
                   "amp_bias_ratio": float(bias[i])} for i, v in enumerate(ov)}
    (out / "metrics_test.json").write_text(json.dumps(metrics, indent=2))
    (out / "r2_scores.json").write_text(
        json.dumps({v: float(r2[i]) for i, v in enumerate(ov)}, indent=2))
    (out / "model_config.json").write_text(json.dumps({
        "grid": meta.get("grid", "gx1v7"), "nj": X.shape[2], "ni": X.shape[3],
        "n_in": n_in, "n_out": n_out, "base": args.base,
        "input_vars": in_vars, "output_vars": ov,
        "residual_targets": {out_vars[o]: in_vars[i] for o, i in res_pairs},
        "memory_lag_steps": meta.get("memory_lag_steps", 1), "cadence": "monthly",
    }, indent=2))

    print("\nTest metrics (ABSOLUTE space, held-out tail):")
    for i, v in enumerate(ov):
        b = f"  amp_bias={bias[i]:.2f}" if v in ("aice", "hi") else ""
        print(f"  {v:8s} R2={r2[i]:+.3f}  rmse={rmse[i]:.3g}{b}")
    print("wrote", out / "metrics_test.json")


if __name__ == "__main__":
    main()
