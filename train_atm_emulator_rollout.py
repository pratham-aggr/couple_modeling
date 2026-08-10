"""
train_atm_emulator_rollout.py
=============================
ROLLOUT-TRAINED autoregressive near-surface ATMOSPHERE emulator.

Why this exists: the one-step emulator (train_atm_emulator.py, output_atm_emulator)
had good one-step skill (Ubot +0.66 vs persistence) but BLEW UP in closed loop —
coupled job 6969200 drove global-mean |wind| to 20.6 m/s in 30 days -> POP k.e.>100.
Good one-step prediction does NOT survive 120+ autoregressive steps: per-step bias
compounds into runaway. The fix is to TRAIN on the rollout it will actually run:
unroll the model K steps feeding its OWN predictions back, backprop through the whole
rollout (BPTT), with a curriculum that grows K over training.

SAME network as the v15 flux net: the grid-agnostic train_unet.UNet backbone,
base=64 (identical architecture/hyperparameters), just a different I/O head for the
wind task and trained as autoregressive rollouts instead of one-shot regression.

Task (6-hourly, gx1v7 384x320):
    IN  (8): SST, ICEFRAC, SOLIN   (LIVE ocean, from truth/POP each step)
             Ubot, Vbot, Tbot, Qbot, PS   (autoregressive: model's OWN prev output)
    OUT (5): Ubot, Vbot, Tbot, Qbot, PS   (at t+1)

Data: the aligned flux cache (no new preprocessing). Ocean inputs are X.npy[:,0:3]
(SST[K]/ICEFRAC/SOLIN at t_now); winds are X_atm.npy[:,0:5]; both are 1456 rows/yr
x 35 yr, consecutive 6-h steps within a year, so a window s..s+K is a valid rollout.
Temporal split (matches the flux-net training): train years <=2010, val 2011-2012,
test 2013-2014.

NaN convention matches the deployed AtmCoupler: SST land -> SST mean (->0 after
norm), ICEFRAC/SOLIN -> 0, wind fields are global.

Outputs (drop-in for AtmCoupler / --atm_emulator): best_model.pt, best_model_ema.pt,
model_config.json (n_in=8,n_out=5,base,input_vars,output_vars), normalizer.npz.
Isolated: writes only --out_dir; touches no production checkpoint.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from train_unet import UNet, Normalizer, masked_mse, ModelEMA

try:
    import wandb
    WANDB = True
except Exception:
    WANDB = False

CACHE = "/glade/work/praggarwal/couple_cache_gx1v7_mem24h"
X_NPY   = f"{CACHE}/X.npy"          # (N,6,H,W): 0=SST[K] 1=ICEFRAC 2=SOLIN 3=SST_prev ...
XATM    = f"{CACHE}/X_atm.npy"      # (N,5,H,W): Ubot,Vbot,Tbot,Qbot,PS
META    = f"{CACHE}/meta.json"
OCEAN_VARS = ["SST", "ICEFRAC", "SOLIN"]
ATM_VARS   = ["Ubot", "Vbot", "Tbot", "Qbot", "PS"]
IN_VARS  = OCEAN_VARS + ATM_VARS    # 8
OUT_VARS = ATM_VARS                 # 5
SPY = 1456                          # samples per year (noleap 6-hourly, cache layout)


# --------------------------------------------------------------------------- data
class RolloutWindows(Dataset):
    """Consecutive-in-year windows of length K+1 built from the aligned memmaps.

    Returns (physical units):
      ocean : (K, 3, H, W)  SST/ICEFRAC/SOLIN at rows s..s+K-1  (NaN-filled)
      atm   : (K+1, 5, H, W) winds at rows s..s+K  (atm[0]=seed, atm[1:]=targets)
    """
    def __init__(self, X_mm, Xa_mm, starts, K, sst_fill, mor=None):
        self.X, self.Xa = X_mm, Xa_mm
        self.starts = np.asarray(starts)
        self.K = int(K)
        self.sst_fill = float(sst_fill)
        self.mor = mor                                            # month-of-row table or None

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i]); K = self.K
        oc = np.array(self.X[s:s + K, 0:3], np.float32)            # (K,3,H,W) writable copy (mmap is read-only)
        oc[:, 0] = np.nan_to_num(oc[:, 0], nan=self.sst_fill)      # SST -> mean
        oc[:, 1:] = np.nan_to_num(oc[:, 1:], nan=0.0)              # ICEFRAC/SOLIN -> 0
        atm = np.nan_to_num(np.asarray(self.Xa[s:s + K + 1, 0:5], np.float32), nan=0.0)
        # month index for rows s..s+K (residual mode); zeros if unused (ignored downstream)
        if self.mor is not None:
            m = self.mor[(np.arange(s, s + K + 1)) % SPY].astype(np.int64)
        else:
            m = np.zeros(K + 1, np.int64)
        return (torch.from_numpy(np.ascontiguousarray(oc)),
                torch.from_numpy(np.ascontiguousarray(atm)),
                torch.from_numpy(m))


def year_starts(years, K):
    """Global start rows whose K-step window stays inside one calendar year."""
    st = []
    for yr in years:
        base = (yr - 1980) * SPY
        st += list(range(base, base + SPY - K))     # s+K <= year end
    return np.array(st, dtype=np.int64)


# ------------------------------------------------------- residual/climatology mode
def month_of_row_table():
    """month index (0..11) for each of the SPY cache rows in one noleap year.
    Matches build_blstate_clim.py: 6-hourly, day-per-month noleap; the cache drops
    the last memory_lag steps (1460->1456) which only trims late December."""
    dpm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    mos = np.concatenate([np.full(d * 4, m, np.int64) for m, d in enumerate(dpm)])  # 1460
    return mos[:SPY]                                                                # 1456


def compute_climatology(Xa_mm, years, mor):
    """Per-calendar-month mean of the 5 wind fields over `years`, on the gx1v7 grid,
    aligned exactly to the cache rows.  Because bilinear CAM->gx1v7 remap is LINEAR,
    this equals remap(monthly-mean-CAM) = the served --blstate_dir climatology, so
    the anomaly baseline used in training is bit-identical to the served floor when
    the server loads the saved clim_gx1v7.npy.  Returns (12, 5, H, W) float32."""
    H, W = Xa_mm.shape[2], Xa_mm.shape[3]
    ssum = np.zeros((12, 5, H, W), np.float64)
    scnt = np.zeros(12, np.int64)
    for yr in years:
        base = (yr - 1980) * SPY
        block = np.nan_to_num(np.asarray(Xa_mm[base:base + SPY, 0:5], np.float64), nan=0.0)
        for m in range(12):
            sel = (mor == m)
            ssum[m] += block[sel].sum(axis=0)
            scnt[m] += int(sel.sum())
    return (ssum / scnt[:, None, None, None]).astype(np.float32)                    # (12,5,H,W)


def compute_norm(X_mm, Xa_mm, rows, chunk=2048, clim=None, mor=None):
    """Streaming per-channel mean/std over training rows. SST uses nan-aware stats
    (land NaN excluded); ICEFRAC/SOLIN 0-filled; winds global.

    x stats are always over (ocean, FULL wind) -- the 8 input channels.  For the y
    (output) stats: if clim/mor are given (RESIDUAL mode) they are computed over the
    ANOMALY (wind - climatology[month]) so the net's anomaly output is unit-variance;
    otherwise they are the full-wind stats (original behavior)."""
    rows = np.asarray(rows)
    n_oc, n_at = 3, 5
    s1 = np.zeros(n_oc + n_at); s2 = np.zeros(n_oc + n_at); cnt = np.zeros(n_oc + n_at)
    ay1 = np.zeros(n_at); ay2 = np.zeros(n_at); acnt = np.zeros(n_at)   # anomaly y-stats
    for a in range(0, len(rows), chunk):
        r = rows[a:a + chunk]
        oc = np.asarray(X_mm[r, 0:3], np.float64)
        oc[:, 1:] = np.nan_to_num(oc[:, 1:], nan=0.0)             # non-SST land -> 0
        at = np.nan_to_num(np.asarray(Xa_mm[r, 0:5], np.float64), nan=0.0)
        block = np.concatenate([oc, at], axis=1)                  # (b,8,H,W)
        for c in range(n_oc + n_at):
            v = block[:, c]
            m = np.isfinite(v)                                    # excludes SST land NaN
            s1[c] += v[m].sum(); s2[c] += (v[m] ** 2).sum(); cnt[c] += m.sum()
        if clim is not None:
            anom = at - clim[mor[r % SPY]]                        # (b,5,H,W) wind anomaly
            for c in range(n_at):
                w = anom[:, c].ravel()
                ay1[c] += w.sum(); ay2[c] += (w ** 2).sum(); acnt[c] += w.size
    mean = s1 / cnt
    std = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 1e-12))
    std[std < 1e-6] = 1.0
    xm, xs = mean.astype(np.float32), std.astype(np.float32)      # 8
    if clim is not None:
        ym = (ay1 / acnt).astype(np.float32)
        ys = np.sqrt(np.maximum(ay2 / acnt - (ay1 / acnt) ** 2, 1e-12)).astype(np.float32)
        ys[ys < 1e-6] = 1.0                                       # 5 (wind ANOMALY)
    else:
        ym, ys = mean[n_oc:].astype(np.float32), std[n_oc:].astype(np.float32)  # 5 (winds)
    return Normalizer(xm, xs, ym, ys)


# ----------------------------------------------------------------------- rollouts
def variance_penalty(pred_n, tgt_n, eps=1e-4):
    """OPTION A (variance-preserving): penalize the emulator regressing to the
    conditional MEAN (the MSE-optimal but variance-collapsed solution that made the
    generated winds ~5-6 m/s vs truth 7.3 -> dead ventilation -> warm drift).

    Two RELATIVE terms in normalized (y) space, per-sample per-channel:
      (1) spatial-variance match  ((var(pred)-var(tgt))/var(tgt))^2  -- restores
          overall field amplitude/spread;
      (2) spatial-gradient-energy match on |grad|^2 -- restores the small-scale
          synoptic structure (high-wavenumber energy) MSE smooths away, which is
          what actually ventilates.
    Relative form makes it scale-free so --var_loss_weight has a stable meaning.
    pred_n, tgt_n: (B,C,H,W) normalized."""
    dims = (-2, -1)
    vp = pred_n.var(dim=dims, unbiased=False)                 # (B,C)
    vt = tgt_n.var(dim=dims, unbiased=False)
    var_term = (((vp - vt) / (vt + eps)) ** 2).mean()

    def grad_energy(z):
        gx = z[..., :, 1:] - z[..., :, :-1]
        gy = z[..., 1:, :] - z[..., :-1, :]
        return gx.pow(2).mean(dim=dims) + gy.pow(2).mean(dim=dims)   # (B,C)
    gp, gt = grad_energy(pred_n), grad_energy(tgt_n)
    grad_term = (((gp - gt) / (gt + eps)) ** 2).mean()
    return var_term + grad_term


def rollout_loss(model, oc, atm, K, xm, xs, ym, ys, ones, var_w=0.0,
                 clim=None, m_idx=None):
    """BPTT over K autoregressive steps. oc:(B,K,3,H,W) atm:(B,K+1,5,H,W) physical.
    Loss in normalized (y) space so all 5 fields weigh evenly; autoregression in
    physical units so the fed-back state has the right magnitude.  var_w>0 adds the
    OPTION-A variance/gradient-energy penalty at each step (training only; the val
    metric keeps var_w=0 so model selection stays pure-MSE / comparable across runs).

    RESIDUAL mode (clim/m_idx given): the net predicts the wind ANOMALY relative to
    the fixed monthly climatology.  Input is always the FULL physical wind (what the
    flux net receives); ym/ys are anomaly stats.  The next full wind fed back is
    clim[month] + predicted_anomaly, so the climatology is re-injected every step --
    the anchor that stops the compounding blowup and floors the worst case at the
    (bounded) climatology.  clim:(12,5,H,W), m_idx:(B,K+1) long."""
    residual = clim is not None
    a = atm[:, 0]                                   # (B,5,H,W) physical seed (truth full wind)
    loss = 0.0
    for k in range(K):
        x = torch.cat([oc[:, k], a], dim=1)         # (B,8,H,W) physical
        x_n = (x - xm) / xs
        pred_n = model(x_n)                         # (B,5,H,W) normalized (anomaly if residual)
        if residual:
            cl_next = clim[m_idx[:, k + 1]]         # (B,5,H,W) climatology at t+1
            tgt_n = (atm[:, k + 1] - cl_next - ym) / ys
        else:
            tgt_n = (atm[:, k + 1] - ym) / ys
        loss = loss + masked_mse(pred_n, tgt_n, ones)
        if var_w > 0.0:
            loss = loss + var_w * variance_penalty(pred_n, tgt_n)
        if residual:
            a = cl_next + (pred_n * ys + ym)        # reconstruct full wind (keep graph)
        else:
            a = pred_n * ys + ym                    # physical -> next input (keep graph)
    return loss / K


@torch.no_grad()
def stability_probe(model, X_mm, Xa_mm, starts, steps, xm, xs, ym, ys, dev, sst_fill,
                    clim=None, mor=None):
    """Free-run |wind| for `steps` from each start; return predicted vs truth mean
    wind speed at 30/60/120 steps. Directly measures the coupled blowup signature
    (truth ~7 m/s; the one-step model hit 20.6 m/s by step 120).  In RESIDUAL mode
    (clim/mor given) the fed-back wind is clim[month] + predicted anomaly."""
    residual = clim is not None
    was_training = model.training
    model.eval()                                    # running BN stats; robust to batch=1
    marks = [m for m in (30, 60, 120) if m <= steps]
    pred_sp = {m: [] for m in marks}; true_sp = {m: [] for m in marks}
    for s in starts:
        a = torch.from_numpy(np.nan_to_num(np.asarray(Xa_mm[s, 0:5], np.float32)))[None].to(dev)
        for k in range(1, steps + 1):
            oc = np.array(X_mm[s + k - 1, 0:3], np.float32)   # writable copy (mmap is read-only)
            oc[0] = np.nan_to_num(oc[0], nan=sst_fill); oc[1:] = np.nan_to_num(oc[1:], nan=0.0)
            ocb = torch.from_numpy(oc)[None].to(dev)
            x_n = (torch.cat([ocb, a], dim=1) - xm) / xs
            if residual:
                cl = clim[mor[(s + k) % SPY]][None]           # (1,5,H,W) climatology at t+1
                a = cl + (model(x_n) * ys + ym)
            else:
                a = model(x_n) * ys + ym
            if k in marks:
                sp = torch.sqrt(a[:, 0] ** 2 + a[:, 1] ** 2).mean().item()
                pred_sp[k].append(sp)
                at = np.nan_to_num(np.asarray(Xa_mm[s + k, 0:5], np.float32))
                true_sp[k].append(float(np.sqrt(at[0] ** 2 + at[1] ** 2).mean()))
    if was_training:
        model.train()
    return ({m: float(np.mean(pred_sp[m])) for m in marks},
            {m: float(np.mean(true_sp[m])) for m in marks})


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_years", default="1980-2010")
    ap.add_argument("--val_years",   default="2011-2012")
    ap.add_argument("--test_years",  default="2013-2014")
    ap.add_argument("--rollout", type=int, default=8, help="max BPTT rollout length K")
    ap.add_argument("--rollout_ramp_epochs", type=int, default=8,
                    help="epochs to grow K from --rollout_start to --rollout (curriculum)")
    ap.add_argument("--rollout_start", type=int, default=2, help="initial K at epoch 0")
    ap.add_argument("--stride", type=int, default=2, help="subsample every Nth window start")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch",  type=int, default=8)
    ap.add_argument("--base",   type=int, default=64)   # same as v15
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.1)  # same as v15
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--probe_steps", type=int, default=120, help="free-run steps for stability probe")
    ap.add_argument("--sst_perturb", type=float, default=0.0, metavar="P",
                    help="LEVER 1: augment training with a sustained WARM SST offset ~U(0,P) K "
                         "added to the input SST channel across all K rollout steps (targets "
                         "unchanged). Teaches the emulator to keep winds physical under coupled "
                         "warm/OOD SST drift (the job-7000015 blowup regime). 0 = off (default). "
                         "Applied in training only; val/probe stay on true SST.")
    ap.add_argument("--var_loss_weight", type=float, default=0.0, metavar="W",
                    help="OPTION A (variance-preserving emulator): weight of the "
                         "variance + spatial-gradient-energy matching penalty added to the "
                         "rollout MSE at each step. Fights the MSE mean-regression that "
                         "collapses generated winds to 5-6 m/s (vs truth 7.3) and starves "
                         "ventilation. Training only; val/model-selection stays pure MSE. "
                         "0 = off (default). Typical ~0.3-1.0.")
    ap.add_argument("--residual", action="store_true",
                    help="RESIDUAL (climatology-anchored) emulator: the net predicts the "
                         "wind ANOMALY on top of the fixed monthly climatology instead of the "
                         "full wind. Reconstructed wind = climatology[month] + anomaly is fed "
                         "back each step, so the climatology is re-injected as an anchor -> the "
                         "worst case degrades to the (bounded, +0.58 K) climatology floor "
                         "instead of compounding off-manifold. Saves clim_gx1v7.npy (12,5,H,W) "
                         "for the server to load as the exact same floor. Default off.")
    ap.add_argument("--clim_years", default="1980-2014",
                    help="Years averaged into the monthly climatology baseline (residual mode). "
                         "Default = all 35 (matches the served 35-yr --blstate_dir climatology).")
    ap.add_argument("--wandb_name", default="mem24h-atm-rollout")
    ap.add_argument("--wandb_project", default="couple-unet")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu = torch.cuda.device_count()
    print(f"device={dev}  visible GPUs={ngpu}", flush=True)

    def yr(spec):
        o = []
        for t in spec.split(","):
            if "-" in t: a, b = t.split("-"); o += list(range(int(a), int(b) + 1))
            else: o.append(int(t))
        return o
    tr_years, va_years, te_years = yr(args.train_years), yr(args.val_years), yr(args.test_years)

    X_mm  = np.load(X_NPY, mmap_mode="r")
    Xa_mm = np.load(XATM,  mmap_mode="r")
    Kmax = args.rollout

    tr_st = year_starts(tr_years, Kmax)[::args.stride]
    va_st = year_starts(va_years, Kmax)[::args.stride]
    print(f"  Train:  {len(tr_st):6d} windows  (years {tr_years[0]}-{tr_years[-1]})")
    print(f"  Val:    {len(va_st):6d} windows  (years {va_years[0]}-{va_years[-1]})")
    print(f"  Test:   {len(year_starts(te_years, Kmax)):6d} windows  (years {te_years[0]}-{te_years[-1]})", flush=True)

    # RESIDUAL mode: build the monthly climatology baseline (gx1v7, cache-aligned) and
    # save it so the server can load the identical floor. mor = month index per row.
    mor = clim = clim_dev = None
    if args.residual:
        mor = month_of_row_table()
        cl_years = yr(args.clim_years)
        print(f"Computing monthly wind climatology over {cl_years[0]}-{cl_years[-1]} "
              f"({len(cl_years)} yr) for residual baseline ...", flush=True)
        clim = compute_climatology(Xa_mm, cl_years, mor)          # (12,5,H,W) float32
        np.save(out / "clim_gx1v7.npy", clim)
        print("  clim |wind| by month:",
              np.round(np.sqrt(clim[:, 0] ** 2 + clim[:, 1] ** 2).mean(axis=(1, 2)), 2), flush=True)

    print("Computing normalizer (subsampled train rows) ...", flush=True)
    _norm_rows = np.unique(np.concatenate(
        [np.arange((y - 1980) * SPY, (y - 1980 + 1) * SPY) for y in tr_years]))[::10]
    norm = compute_norm(X_mm, Xa_mm, _norm_rows, clim=clim, mor=mor)
    norm.save(out / "normalizer.npz")
    sst_fill = float(norm.x_mean[0])
    print("  x_mean", np.round(norm.x_mean, 3), flush=True)
    if args.residual:
        print("  anomaly y_mean", np.round(norm.y_mean, 4),
              " y_std", np.round(norm.y_std, 4), flush=True)

    if args.dry_run:
        tr_st, va_st = tr_st[:64], va_st[:32]; args.epochs = 2; args.probe_steps = 20

    tr_ds = RolloutWindows(X_mm, Xa_mm, tr_st, Kmax, sst_fill, mor=mor)
    va_ds = RolloutWindows(X_mm, Xa_mm, va_st, Kmax, sst_fill, mor=mor)
    dl_tr = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=6,
                       pin_memory=(dev == "cuda"), drop_last=True, persistent_workers=True)
    dl_va = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=4,
                       persistent_workers=True)

    model = UNet(n_in=len(IN_VARS), n_out=len(OUT_VARS), base=args.base, dropout=args.dropout).to(dev)
    core = model
    if ngpu >= 2:
        model = nn.DataParallel(model)
        print(f"  DataParallel across {ngpu} GPUs", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ema = ModelEMA(core) if args.ema else None

    def to_dev(a):
        return a.to(dev, non_blocking=True)
    xm = torch.from_numpy(norm.x_mean).view(1, -1, 1, 1).to(dev)
    xs = torch.from_numpy(norm.x_std ).view(1, -1, 1, 1).to(dev)
    ym = torch.from_numpy(norm.y_mean).view(1, -1, 1, 1).to(dev)
    ys = torch.from_numpy(norm.y_std ).view(1, -1, 1, 1).to(dev)
    if args.residual:
        clim_dev = torch.from_numpy(clim).to(dev)                # (12,5,H,W) climatology floor

    use_wandb = WANDB and (not args.no_wandb)
    if use_wandb:
        run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
        (out / "wandb_run_id.txt").write_text(run.id)
        print(f"  wandb run {run.id}  ({run.get_url()})", flush=True)

    probe_starts = [(y - 1980) * SPY for y in va_years]      # a Jan-1 start per val year
    ones = None
    best = float("inf")
    for ep in range(args.epochs):
        # curriculum: grow rollout length K over the first ramp epochs
        if args.rollout_ramp_epochs > 0:
            frac = min(1.0, ep / max(1, args.rollout_ramp_epochs))
            K = int(round(args.rollout_start + frac * (Kmax - args.rollout_start)))
        else:
            K = Kmax
        K = max(1, min(K, Kmax))

        model.train(); t0 = time.time(); tot = nb = 0
        for oc, atm, m in dl_tr:
            oc, atm = to_dev(oc), to_dev(atm)
            m = to_dev(m) if args.residual else None
            if args.sst_perturb > 0.0:
                # LEVER 1: sustained per-window warm SST offset on the input SST channel
                # (index 0 of the 3 ocean chans), constant across the K rollout steps to
                # mimic a coupled warm drift. Targets (truth winds) unchanged -> the net
                # learns bounded winds under warmer-than-climatology SST.
                off = torch.rand(oc.shape[0], 1, 1, 1, device=dev) * args.sst_perturb
                oc = oc.clone()
                oc[:, :, 0] = oc[:, :, 0] + off
            if ones is None or ones.shape[0] != oc.shape[0]:
                ones = torch.ones(oc.shape[0], oc.shape[3], oc.shape[4], device=dev)
            opt.zero_grad()
            loss = rollout_loss(model, oc, atm, K, xm, xs, ym, ys, ones,
                                var_w=args.var_loss_weight, clim=clim_dev, m_idx=m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema: ema.update(core)
            tot += loss.item(); nb += 1
        sched.step()

        # validation: rollout loss at FULL Kmax (the metric we actually care about)
        model.eval(); vt = vn = 0
        with torch.no_grad():
            for oc, atm, m in dl_va:
                oc, atm = to_dev(oc), to_dev(atm)
                m = to_dev(m) if args.residual else None
                o1 = torch.ones(oc.shape[0], oc.shape[3], oc.shape[4], device=dev)
                vt += rollout_loss(model, oc, atm, Kmax, xm, xs, ym, ys, o1,
                                   clim=clim_dev, m_idx=m).item(); vn += 1
        va = vt / max(vn, 1)

        # long free-run stability probe (the blowup detector)
        psp, tsp = stability_probe(core, X_mm, Xa_mm, probe_starts, args.probe_steps,
                                   xm, xs, ym, ys, dev, sst_fill, clim=clim_dev, mor=mor)
        pk = max(psp) if psp else float("nan")   # worst predicted |wind| at any mark
        smark = sorted(psp)[-1] if psp else 0
        print(f"ep {ep:3d}  K={K}  train {tot/max(nb,1):.4f}  valK{Kmax} {va:.4f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  |wind|@{smark}: pred {psp.get(smark,float('nan')):.1f} "
              f"truth {tsp.get(smark,float('nan')):.1f} m/s  {time.time()-t0:.0f}s", flush=True)
        if use_wandb:
            log = {"epoch": ep, "K": K, "train_loss": tot / max(nb, 1), "val_rollout_loss": va,
                   "lr": sched.get_last_lr()[0]}
            for m in psp: log[f"wind_pred_step{m}"] = psp[m]; log[f"wind_truth_step{m}"] = tsp[m]
            wandb.log(log)

        if va < best:
            best = va
            torch.save(core.state_dict(), out / "best_model.pt")
            if ema: torch.save(ema.shadow, out / "best_model_ema.pt")

    (out / "model_config.json").write_text(json.dumps({
        "grid": "gx1v7", "nj": 384, "ni": 320,
        "n_in": len(IN_VARS), "n_out": len(OUT_VARS), "base": args.base, "dropout": args.dropout,
        "input_vars": IN_VARS, "output_vars": OUT_VARS, "cadence": "6-hourly",
        "training": "autoregressive BPTT rollout (curriculum), full 1980-2010 split",
        "sst_perturb": args.sst_perturb,
        "var_loss_weight": args.var_loss_weight,
        "residual": bool(args.residual),
        "clim_file": "clim_gx1v7.npy" if args.residual else None,
        "clim_years": args.clim_years if args.residual else None,
        "output_is_anomaly": bool(args.residual),
        "note": "rollout-trained BL-atm emulator; predicts Ubot..PS at t+1 (blowup-hardened)"
                + ("; RESIDUAL: output is the wind ANOMALY, served wind = "
                   "clim_gx1v7.npy[month] + anomaly (climatology-anchored floor)"
                   if args.residual else "")
                + (f"; SST-perturbed training U(0,{args.sst_perturb})K for warm-OOD robustness"
                   if args.sst_perturb > 0 else "")
                + (f"; variance/grad-energy loss w={args.var_loss_weight} for "
                   "variance-preserving winds (anti mean-regression)"
                   if args.var_loss_weight > 0 else ""),
    }, indent=2))

    # final stability report on the best model
    core.load_state_dict(torch.load(out / "best_model.pt", map_location=dev)); core.eval()
    psp, tsp = stability_probe(core, X_mm, Xa_mm,
                               [(y - 1980) * SPY for y in te_years], args.probe_steps,
                               xm, xs, ym, ys, dev, sst_fill, clim=clim_dev, mor=mor)
    print("\nFinal free-run stability (test-year starts):", flush=True)
    for m in sorted(psp):
        print(f"  step {m:3d}:  |wind| pred {psp[m]:.2f}  truth {tsp[m]:.2f} m/s", flush=True)
    (out / "stability_test.json").write_text(json.dumps(
        {"pred": psp, "truth": tsp, "probe_steps": args.probe_steps}, indent=2))
    print("wrote", out / "model_config.json", flush=True)
    if use_wandb: wandb.finish()


if __name__ == "__main__":
    main()
