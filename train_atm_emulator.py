"""
train_atm_emulator.py
=====================
EXPERIMENT (2026-07-30): train an autoregressive near-surface ATMOSPHERE emulator
so the coupled run can PREDICT the boundary-layer state instead of PRESCRIBING it
from blstate files.  Mirror of train_cice.py: reuses the grid-agnostic UNet /
Normalizer / masked_mse / ModelEMA from train_unet.

Task it learns (6-hourly, one step ahead):

    IN  (8ch): SST, ICEFRAC, SOLIN, Ubot, Vbot, Tbot, Qbot, PS      (all at time t)
    OUT (5ch): Ubot, Vbot, Tbot, Qbot, PS                          (at time t+1, +6 h)

At run time the AtmCoupler feeds the model its OWN previous prediction as the
Ubot..PS inputs (autoregressive), plus the live SST/ICEFRAC (from POP/ice) and
prescribed SOLIN, and hands the fresh atm state to the flux UNet in place of the
prescribed blstate.  This is the "turn down prescription" experiment.

Data path (no new cache):
  * SST/ICEFRAC/SOLIN at t_now come straight from the gx1v7 training zarrs.
  * Ubot..PS at t_now come from the existing aligned cache X_atm.npy
    (N,5,384,320), whose sample layout is 1456 rows/year with row j at
    zarr-time t_now = j + memory_lag_steps (=4).  The one-step-ahead target is
    simply row j+1 of the same year (drop the last row of each year).

Everything here is an isolated experiment: writes only to --out_dir; touches no
production checkpoint or the coupled server.

Usage (see scripts/submit_atm_emulator.pbs):
    python train_atm_emulator.py --out_dir output/output_atm_emulator \
        --train_years 1980-1991 --test_years 1992-1993 --stride 2 --epochs 40
"""
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, DataLoader

from train_unet import UNet, Normalizer, masked_mse, ModelEMA

GX_ZARR = "/glade/derecho/scratch/praggarwal/zarr_gx1v7/b.e21.CREDIT_gx1v7_{year}.zarr"
X_ATM   = "/glade/work/praggarwal/couple_cache_gx1v7_mem24h/X_atm.npy"
META    = "/glade/work/praggarwal/couple_cache_gx1v7_mem24h/meta.json"
OCEAN_VARS = ["SST", "ICEFRAC", "SOLIN"]          # from zarr, at t_now
ATM_VARS   = ["Ubot", "Vbot", "Tbot", "Qbot", "PS"]  # from X_atm, at t_now
IN_VARS  = OCEAN_VARS + ATM_VARS                   # 8 input channels
OUT_VARS = ATM_VARS                                # 5 output channels (t+1)


def _year_range(spec):
    """'1980-1991' or '1980,1981,1990' -> list[int]."""
    out = []
    for tok in spec.split(","):
        if "-" in tok:
            a, b = tok.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def load_years(years, lag, samples_per_year, stride):
    """Build (X, Y) for the given calendar years.  X:(P,8,H,W)  Y:(P,5,H,W)."""
    x_atm = np.load(X_ATM, mmap_mode="r")            # (N,5,384,320)
    Xs, Ys = [], []
    for yr in years:
        yidx = yr - 1980
        blk = x_atm[yidx * samples_per_year:(yidx + 1) * samples_per_year]   # (1456,5,H,W) atm@t_now
        zf = GX_ZARR.format(year=yr)
        ds = xr.open_zarr(zf, consolidated=False)
        # ocean at t_now: zarr time index = j + lag, j in [0, samples_per_year)
        oc = np.stack([ds[v].isel(time=slice(lag, lag + samples_per_year)).values
                       for v in OCEAN_VARS], axis=1).astype(np.float32)        # (1456,3,H,W)
        ds.close()
        n = min(len(blk), len(oc))
        # valid one-step pairs: j and j+1 in the SAME year  ->  j in [0, n-2]
        js = np.arange(0, n - 1, stride)
        x = np.concatenate([oc[js], np.asarray(blk[js], np.float32)], axis=1)  # (P,8,H,W)
        y = np.asarray(blk[js + 1], np.float32)                                # (P,5,H,W)  atm@t+1
        Xs.append(x); Ys.append(y)
        print(f"  {yr}: {len(js)} pairs")
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    # sanitise land/fill NaNs in the ocean input channels (atm fields are global)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    return X, Y


class ArrDataset(Dataset):
    def __init__(self, X, Y, norm, augment=False):
        self.X, self.Y, self.norm, self.augment = X, Y, norm, augment

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = (self.X[i] - self.norm.x_mean[:, None, None]) / (self.norm.x_std[:, None, None] + 1e-8)
        y = (self.Y[i] - self.norm.y_mean[:, None, None]) / (self.norm.y_std[:, None, None] + 1e-8)
        if self.augment:
            s = int(np.random.randint(0, x.shape[-1]))
            if s:
                x = np.roll(x, s, -1); y = np.roll(y, s, -1)
        return (torch.from_numpy(np.ascontiguousarray(x.astype(np.float32))),
                torch.from_numpy(np.ascontiguousarray(y.astype(np.float32))))


def compute_norm(X, Y):
    xm = X.mean(axis=(0, 2, 3)); xs = X.std(axis=(0, 2, 3))
    ym = Y.mean(axis=(0, 2, 3)); ys = Y.std(axis=(0, 2, 3))
    xs[xs < 1e-6] = 1.0; ys[ys < 1e-6] = 1.0
    return Normalizer(xm, xs, ym, ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_years", default="1980-1991")
    ap.add_argument("--test_years",  default="1992-1993")
    ap.add_argument("--stride", type=int, default=2, help="subsample every Nth one-step pair")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--base",   type=int, default=64)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads(Path(META).read_text())
    lag = int(meta["memory_lag_steps"]); spy = meta["n_samples"] // 35
    print(f"device={dev}  lag={lag}  samples/year={spy}")

    tr_years = _year_range(args.train_years); te_years = _year_range(args.test_years)
    print(f"TRAIN years {tr_years}"); Xtr, Ytr = load_years(tr_years, lag, spy, args.stride)
    print(f"TEST  years {te_years}"); Xte, Yte = load_years(te_years, lag, spy, args.stride)
    print(f"train {Xtr.shape} -> {Ytr.shape}   test {Xte.shape}")
    if args.dry_run:
        Xtr, Ytr, Xte, Yte = Xtr[:64], Ytr[:64], Xte[:32], Yte[:32]; args.epochs = 2

    norm = compute_norm(Xtr, Ytr); norm.save(out / "normalizer.npz")
    print("norm x_mean", np.round(norm.x_mean, 3)); print("norm y_mean", np.round(norm.y_mean, 3))

    # carve a small val slice off the end of train (temporal)
    nva = max(1, int(0.1 * len(Xtr))); ntr = len(Xtr) - nva
    dl_tr = DataLoader(ArrDataset(Xtr[:ntr], Ytr[:ntr], norm, args.augment),
                       batch_size=args.batch, shuffle=True, num_workers=4,
                       pin_memory=(dev == "cuda"), drop_last=True)
    dl_va = DataLoader(ArrDataset(Xtr[ntr:], Ytr[ntr:], norm),
                       batch_size=args.batch, shuffle=False, num_workers=2)
    dl_te = DataLoader(ArrDataset(Xte, Yte, norm),
                       batch_size=args.batch, shuffle=False, num_workers=2)

    model = UNet(n_in=len(IN_VARS), n_out=len(OUT_VARS), base=args.base, dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ema = ModelEMA(model) if args.ema else None
    ones = None
    best = float("inf")
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = nb = 0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            if ones is None or ones.shape[0] != x.shape[0]:
                ones = torch.ones(x.shape[0], x.shape[2], x.shape[3], device=dev)
            opt.zero_grad()
            loss = masked_mse(model(x), y, ones[:x.shape[0]])
            loss.backward(); opt.step()
            if ema: ema.update(model)
            tot += loss.item(); nb += 1
        sched.step()
        model.eval(); vt = vn = 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(dev), y.to(dev)
                m = torch.ones(x.shape[0], x.shape[2], x.shape[3], device=dev)
                vt += masked_mse(model(x), y, m).item(); vn += 1
        va = vt / max(vn, 1)
        print(f"ep {ep:3d}  train {tot/max(nb,1):.4f}  val {va:.4f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s", flush=True)
        if va < best:
            best = va
            torch.save(model.state_dict(), out / "best_model.pt")
            if ema: torch.save(ema.shadow, out / "best_model_ema.pt")

    # held-out test: per-variable RMSE (physical units) of the model AND of a
    # persistence baseline (atm@t unchanged), so we can see the model beats "do nothing".
    model.load_state_dict(torch.load(out / "best_model.pt", map_location=dev)); model.eval()
    se = np.zeros(len(OUT_VARS)); se_p = np.zeros(len(OUT_VARS)); cnt = 0
    ys_ = norm.y_std[None, :, None, None]; ym_ = norm.y_mean[None, :, None, None]
    with torch.no_grad():
        for x, y in dl_te:
            p = model(x.to(dev)).cpu().numpy()
            p_phys = p * ys_ + ym_
            y_phys = y.numpy() * ys_ + ym_
            # persistence = the physical atm@t inputs (channels OCEAN..end of x, denormalised)
            xin = x.numpy()[:, len(OCEAN_VARS):, :, :]
            pers_phys = xin * norm.x_std[len(OCEAN_VARS):][None, :, None, None] \
                            + norm.x_mean[len(OCEAN_VARS):][None, :, None, None]
            se   += ((p_phys - y_phys) ** 2).sum(axis=(0, 2, 3))
            se_p += ((pers_phys - y_phys) ** 2).sum(axis=(0, 2, 3))
            cnt  += p.shape[0] * p.shape[2] * p.shape[3]
    rmse = np.sqrt(se / cnt); rmse_p = np.sqrt(se_p / cnt)
    metrics = {v: {"rmse_model": float(rmse[i]), "rmse_persistence": float(rmse_p[i]),
                   "skill_vs_persistence": float(1.0 - rmse[i] / rmse_p[i])}
               for i, v in enumerate(OUT_VARS)}
    (out / "metrics_test.json").write_text(json.dumps(metrics, indent=2))
    (out / "model_config.json").write_text(json.dumps({
        "grid": "gx1v7", "nj": 384, "ni": 320,
        "n_in": len(IN_VARS), "n_out": len(OUT_VARS), "base": args.base, "dropout": args.dropout,
        "input_vars": IN_VARS, "output_vars": OUT_VARS, "cadence": "6-hourly",
        "note": "autoregressive BL-atm emulator (experiment); predicts Ubot..PS at t+1",
    }, indent=2))
    print("\nHeld-out test RMSE (physical units)  [model vs persistence]:")
    for i, v in enumerate(OUT_VARS):
        print(f"  {v:5s}  model={rmse[i]:.4g}  persist={rmse_p[i]:.4g}  "
              f"skill={1.0 - rmse[i]/rmse_p[i]:+.3f}")
    print("wrote", out / "model_config.json")


if __name__ == "__main__":
    main()
