"""
train_unified_flux_wind_rollout.py
==================================
UNIFIED autoregressive flux+wind model = the WORKING MEMO flux net, extended to
ALSO predict the next-step boundary-layer atmosphere so it can drive itself past
the 2014 forcing window.

Why this exists (user directive 2026-08-04): the working MEMO flux net (v15, n_in=11
incl. Ubot/Vbot/Tbot/Qbot/PS AS INPUT -> 8 fluxes) drifts only +0.002 K when fed the
prescribed CREDIT winds -- but those winds stop at 2014.  Two prior attempts failed:
  * kvjcofin: predicted winds+fluxes from OCEAN-ONLY inputs (no wind input) -> lost
    the flux quality, died when coupled.
  * standalone rollout emulator: a SEPARATE net for winds -> variance-collapsed weak
    winds (5-6 vs 7.3 m/s) -> +1.37 K coupled drift.

This model keeps the WINNING structure -- winds as INPUT, n_in=11, same UNet as the
working memo -- and simply ADDS the 5 next-step wind channels to the output head:

    IN  (11): SST, ICEFRAC, SOLIN, SST_prev, ICEFRAC_prev, SOLIN_prev  (ocean, truth
              each step -> from POP when coupled)
              Ubot, Vbot, Tbot, Qbot, PS @t  (autoregressive: model's OWN prev output)
    OUT (13): TAUX, TAUY, SHFLX, LHFLX, QFLX, FSDS_J, FLDS_J, PRECT  (fluxes @t)
              Ubot, Vbot, Tbot, Qbot, PS @t+1  (next BL atmosphere -> fed back)

Trained on the ROLLOUT it will actually run (BPTT, curriculum K 2->8) -- the piece
kvjcofin lacked.  At inference the fluxes drive POP and the 5 wind outputs become the
next step's wind inputs (seeded once from the real blstate IC).

Data: the aligned flux cache (no new preprocessing). X.npy[:,0:6]=ocean/forcing,
X_atm.npy[:,0:5]=winds@t_now, Y/Y_rad/Y_precip = the 8 flux targets@t.  Wind target@t+1
is just X_atm[s+k+1] (consecutive 6-h rows within a year).  Temporal split matches the
flux net: train<=2010, val 2011-2012, test 2013-2014.

=== 2026-08-10: OPTION 2 additions (default-off; behavior byte-identical when off) ===
The first unified run predicted ABSOLUTE winds with plain rollout MSE and was never
coupled-validated; measured offline it is the "full-wind" flavor (blur->weak winds or
OOD->blowup) with no dQ/dSST feedback shaping.  Three additive fixes, each mirroring
the pieces that already work in the two separate nets, so ONE net = 2 emulators total:

  --residual        the 5 wind outputs are ANOMALIES on a fixed monthly climatology;
                    fed back as clim[month]+anomaly every step (mirrors the WORKING
                    residual emulator, train_atm_emulator_rollout.py).  Worst case
                    (anomaly->0) degrades to the exact climatology floor, never a
                    runaway.  x wind stats stay full-wind (input is the real wind);
                    y wind stats are over the anomaly.  Saves clim_gx1v7.npy.
  --sst_sens_reg    a bulk-flux SST-sensitivity penalty on the FLUX outputs (SHFLX,
                    LHFLX) toward the physical dQ/dSST, radiation toward 0 (mirrors
                    the deployed v15 sensreg fine-tune, train_unet.py).  Applied at
                    rollout step 0 (truth winds), BN in eval -- the deployed feedback
                    is a single-step input->flux property.
  --init_from CKPT  partial warm-start from a validated v15 flux net: copies every
                    matching backbone/enc/dec weight and the first 8 flux-head
                    channels; the 5 wind-head channels start fresh.  Inherits the
                    +0.002 K flux quality/feedback instead of re-deriving it.

IMPORTANT: the 5 wind channels share the INPUT normalizer (absolute wind) between the
input slot and the fed-back full wind, so reconstruction re-normalizes cleanly; the
OUTPUT normalizer is anomaly stats in --residual mode.

Outputs (drop-in for the unified server path): best_model.pt, best_model_ema.pt,
model_config.json (n_in=11,n_out=13,input_vars,output_vars,n_flux=8,n_wind=5,
residual,clim_file), normalizer.npz, clim_gx1v7.npy (residual).  Isolated: writes only --out_dir.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from train_unet import UNet, masked_mse, ModelEMA, bulk_sens_target, bn_eval

try:
    import wandb
    WANDB = True
except Exception:
    WANDB = False

CACHE = "/glade/work/praggarwal/couple_cache_gx1v7_mem24h"
X_NPY  = f"{CACHE}/X.npy"          # (N,6,H,W) ocean/forcing inputs
XATM   = f"{CACHE}/X_atm.npy"      # (N,5,H,W) Ubot,Vbot,Tbot,Qbot,PS @t_now
Y_NPY  = f"{CACHE}/Y.npy"          # (N,5,H,W) TAUX,TAUY,SHFLX,LHFLX,QFLX
YRAD   = f"{CACHE}/Y_rad.npy"      # (N,2,H,W) FSDS_J,FLDS_J
YPREC  = f"{CACHE}/Y_precip.npy"   # (N,1,H,W) PRECT
MASK   = f"{CACHE}/mask.npy"       # (N,H,W)   ocean/ice mask

OCEAN_IN = ["SST", "ICEFRAC", "SOLIN", "SST_prev", "ICEFRAC_prev", "SOLIN_prev"]  # 6
WIND     = ["Ubot", "Vbot", "Tbot", "Qbot", "PS"]                                # 5
FLUX     = ["TAUX", "TAUY", "SHFLX", "LHFLX", "QFLX", "FSDS_J", "FLDS_J", "PRECT"]  # 8
IN_VARS  = OCEAN_IN + WIND       # 11  (== working memo inputs)
OUT_VARS = FLUX + WIND           # 13  (8 fluxes@t + 5 winds@t+1)
N_FLUX, N_WIND, N_OCEAN = 8, 5, 6
SPY = 1456                       # samples per year (noleap 6-hourly, cache layout)

# --- sensitivity-regularizer constants (match train_unet.py) ---
SENS_SCALE = 40.0                # W/m2/K, normalizes the penalty to O(1)
J2W        = 1.0 / 21600.0       # J/m2/6h -> W/m2 (fluxes stored per 6h)
SHFLX_I, LHFLX_I, FSDS_I, FLDS_I = 2, 3, 5, 6   # flux-output-channel indices


# ------------------------------------------------------- residual/climatology mode
def month_of_row_table():
    """month index (0..11) for each of the SPY cache rows in one noleap year.
    Matches train_atm_emulator_rollout.py / build_blstate_clim.py."""
    dpm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    mos = np.concatenate([np.full(d * 4, m, np.int64) for m, d in enumerate(dpm)])  # 1460
    return mos[:SPY]                                                                # 1456


def compute_climatology(Xa_mm, years, mor):
    """Per-calendar-month mean of the 5 wind fields over `years`, aligned to cache
    rows.  Bilinear CAM->gx1v7 remap is LINEAR, so this equals the served
    --blstate_dir climatology -> the anomaly baseline is bit-identical to the served
    floor.  Returns (12, 5, H, W) float32."""
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


# --------------------------------------------------------------------------- data
class UnifiedWindows(Dataset):
    """Consecutive-in-year rollout windows from the aligned memmaps (physical units).

    Returns:
      oc : (K, 6, H, W)   ocean/forcing at rows s..s+K-1
      wd : (K+1, 5, H, W) winds at rows s..s+K  (wd[0]=seed, wd[1:]=wind targets@t+1)
      fl : (K, 8, H, W)   flux targets at rows s..s+K-1
      mk : (K, H, W)      ocean mask at rows s..s+K-1
      mi : (K+1,)         month index (0..11) at rows s..s+K  (residual mode; else 0)
    """
    def __init__(self, starts, K, sst_fill, mor=None):
        self.X  = np.load(X_NPY,  mmap_mode="r")
        self.Xa = np.load(XATM,   mmap_mode="r")
        self.Y  = np.load(Y_NPY,  mmap_mode="r")
        self.Yr = np.load(YRAD,   mmap_mode="r")
        self.Yp = np.load(YPREC,  mmap_mode="r")
        self.M  = np.load(MASK,   mmap_mode="r")
        self.starts = np.asarray(starts)
        self.K = int(K)
        self.sst_fill = float(sst_fill)
        self.mor = mor                                # (SPY,) month-of-row or None

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i]); K = self.K
        oc = np.array(self.X[s:s + K, 0:6], np.float32)               # (K,6) writable
        oc[:, 0] = np.nan_to_num(oc[:, 0], nan=self.sst_fill)         # SST land -> mean
        oc[:, 1:] = np.nan_to_num(oc[:, 1:], nan=0.0)
        wd = np.nan_to_num(np.array(self.Xa[s:s + K + 1, 0:5], np.float32), nan=0.0)
        fl = np.concatenate([np.array(self.Y[s:s + K],  np.float32),
                             np.array(self.Yr[s:s + K], np.float32),
                             np.array(self.Yp[s:s + K], np.float32)], axis=1)   # (K,8)
        fl = np.nan_to_num(fl, nan=0.0)
        mk = np.nan_to_num(np.array(self.M[s:s + K], np.float32), nan=0.0)      # (K,H,W)
        if self.mor is not None:
            mi = self.mor[(np.arange(s, s + K + 1)) % SPY].astype(np.int64)     # (K+1,)
        else:
            mi = np.zeros(K + 1, np.int64)
        return (torch.from_numpy(np.ascontiguousarray(oc)),
                torch.from_numpy(np.ascontiguousarray(wd)),
                torch.from_numpy(np.ascontiguousarray(fl)),
                torch.from_numpy(np.ascontiguousarray(mk)),
                torch.from_numpy(np.ascontiguousarray(mi)))


def year_starts(years, K):
    st = []
    for yr in years:
        base = (yr - 1980) * SPY
        st += list(range(base, base + SPY - K))     # s+K <= year end
    return np.array(st, dtype=np.int64)


def compute_norm(rows, chunk=1024, clim=None, mor=None):
    """Streaming masked per-channel mean/std over training rows.
    x_mean/x_std (11): ocean(6) + FULL wind(5) -- inputs are always the real wind.
    y_mean/y_std (13): flux(8) + wind(5).  In RESIDUAL mode (clim/mor given) the wind
    y-stats are over the ANOMALY (wind - clim[month]); otherwise they equal the input
    (full-wind) stats, exactly as before."""
    X  = np.load(X_NPY,  mmap_mode="r"); Xa = np.load(XATM, mmap_mode="r")
    Y  = np.load(Y_NPY,  mmap_mode="r"); Yr = np.load(YRAD, mmap_mode="r")
    Yp = np.load(YPREC,  mmap_mode="r"); M  = np.load(MASK, mmap_mode="r")
    rows = np.asarray(rows)

    def acc(nc):
        return np.zeros(nc), np.zeros(nc), np.zeros(nc)
    s1o, s2o, cno = acc(N_OCEAN)      # ocean (6)
    s1w, s2w, cnw = acc(N_WIND)       # wind (5, full)
    s1f, s2f, cnf = acc(N_FLUX)       # flux (8)
    a1w, a2w, acw = acc(N_WIND)       # wind ANOMALY (5, residual only)
    for a in range(0, len(rows), chunk):
        r = rows[a:a + chunk]
        oc = np.asarray(X[r, 0:6], np.float64)
        oc[:, 1:] = np.nan_to_num(oc[:, 1:], nan=0.0)
        wd = np.nan_to_num(np.asarray(Xa[r, 0:5], np.float64), nan=0.0)
        fl = np.concatenate([np.asarray(Y[r], np.float64),
                             np.asarray(Yr[r], np.float64),
                             np.asarray(Yp[r], np.float64)], axis=1)
        fl = np.nan_to_num(fl, nan=0.0)
        for blk, s1, s2, cn in ((oc, s1o, s2o, cno), (wd, s1w, s2w, cnw), (fl, s1f, s2f, cnf)):
            for c in range(blk.shape[1]):
                v = blk[:, c]; m = np.isfinite(v)
                s1[c] += v[m].sum(); s2[c] += (v[m] ** 2).sum(); cn[c] += m.sum()
        if clim is not None:
            anom = wd - clim[mor[r % SPY]]           # (b,5,H,W) wind anomaly
            for c in range(N_WIND):
                v = anom[:, c].ravel()
                a1w[c] += v.sum(); a2w[c] += (v ** 2).sum(); acw[c] += v.size

    def finish(s1, s2, cn):
        mean = s1 / cn
        std = np.sqrt(np.maximum(s2 / cn - mean ** 2, 1e-12)); std[std < 1e-6] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)
    om, os_ = finish(s1o, s2o, cno)
    wm, ws  = finish(s1w, s2w, cnw)          # full-wind (INPUT) stats
    fm, fs  = finish(s1f, s2f, cnf)
    x_mean = np.concatenate([om, wm]); x_std = np.concatenate([os_, ws])   # 11
    if clim is not None:
        awm, aws = finish(a1w, a2w, acw)     # wind ANOMALY (OUTPUT) stats
        y_mean = np.concatenate([fm, awm]); y_std = np.concatenate([fs, aws])   # 13
    else:
        y_mean = np.concatenate([fm, wm]); y_std = np.concatenate([fs, ws])     # 13 (shared)
    return x_mean, x_std, y_mean, y_std


# ----------------------------------------------------------------------- rollouts
def variance_penalty(pred_n, tgt_n, eps=1e-4):
    """Relative spatial-variance + spatial-gradient-energy match (anti mean-regression);
    applied to the WIND output channels only.  See train_atm_emulator_rollout.py."""
    dims = (-2, -1)
    vp = pred_n.var(dim=dims, unbiased=False); vt = tgt_n.var(dim=dims, unbiased=False)
    var_term = (((vp - vt) / (vt + eps)) ** 2).mean()

    def ge(z):
        gx = z[..., :, 1:] - z[..., :, :-1]; gy = z[..., 1:, :] - z[..., :-1, :]
        return gx.pow(2).mean(dim=dims) + gy.pow(2).mean(dim=dims)
    gp, gt = ge(pred_n), ge(tgt_n)
    return var_term + (((gp - gt) / (gt + eps)) ** 2).mean()


def sens_penalty(model, x_n, sst_phys, wd_phys, mk, xs, ys, sens_delta, hybm,
                 lam_turb, lam_rad):
    """Bulk-flux SST-sensitivity penalty on the FLUX outputs (mirrors train_unet.py).
    x_n:(B,11,H,W) normalized step-0 input; sst_phys:(B,H,W); wd_phys:(B,5,H,W)
    physical (Ubot,Vbot,Tbot,Qbot,PS); mk:(B,H,W).  Returns a scalar loss.
    Net causal dQ/dSST via centered +-delta perturbation of SST(0)+SST_prev(3), BN in
    eval; target = shr_flux d(sen+lat)/dSST from the cell's own BL atm; radiation->0."""
    dnorm0 = float(sens_delta) / (float(xs[0, 0, 0, 0]) + 1e-8)
    dnorm3 = float(sens_delta) / (float(xs[0, 3, 0, 0]) + 1e-8)
    x_plus  = x_n.clone(); x_plus[:, 0]  += dnorm0; x_plus[:, 3]  += dnorm3
    x_minus = x_n.clone(); x_minus[:, 0] -= dnorm0; x_minus[:, 3] -= dnorm3
    B = x_n.shape[0]
    with bn_eval(model):
        pboth = model(torch.cat([x_plus, x_minus], dim=0))     # (2B,13,H,W)
    pp, pm = pboth[:B], pboth[B:]
    inv2d = 1.0 / (2.0 * float(sens_delta))
    ysf = ys[0, :, 0, 0]                                        # (13,)
    d_turb = ((pp[:, SHFLX_I] - pm[:, SHFLX_I]) * ysf[SHFLX_I]
              + (pp[:, LHFLX_I] - pm[:, LHFLX_I]) * ysf[LHFLX_I]) * (J2W * inv2d)   # (B,H,W)
    d_rad  = ((pp[:, FSDS_I] - pm[:, FSDS_I]) * ysf[FSDS_I]
              + (pp[:, FLDS_I] - pm[:, FLDS_I]) * ysf[FLDS_I]) * (J2W * inv2d)
    # target: per-cell bulk d(sen+lat)/dSST (numpy, no grad)
    npd = lambda t: t.detach().cpu().numpy().astype(np.float64)
    tgt = bulk_sens_target(npd(sst_phys), npd(wd_phys[:, 0]), npd(wd_phys[:, 1]),
                           npd(wd_phys[:, 2]), npd(wd_phys[:, 3]), npd(wd_phys[:, 4]),
                           float(sens_delta), float(hybm))       # (B,H,W) W/m2/K
    sens_t = torch.from_numpy(tgt).to(x_n.device, x_n.dtype)
    msum = mk.sum() + 1e-8
    l_turb = ((((d_turb - sens_t) / SENS_SCALE) ** 2) * mk).sum() / msum
    l_rad  = (((d_rad / SENS_SCALE) ** 2) * mk).sum() / msum
    return lam_turb * l_turb + lam_rad * l_rad


def rollout_loss(model, oc, wd, fl, mk, mi, K, xm, xs, ym, ys, var_w=0.0,
                 clim=None, sens_cfg=None):
    """BPTT over K steps. oc(B,K,6) wd(B,K+1,5) fl(B,K,8) mk(B,K,H,W) mi(B,K+1) physical.
    Each step: input [ocean@t (truth), winds@t (autoregressive)] -> [flux@t, winds@t+1].
    Loss in normalized space; winds fed back in physical units.

    RESIDUAL mode (clim given): the 5 wind outputs are anomalies on clim[month]; the
    wind fed back is clim[month]+anomaly (ym/ys are anomaly stats).  sens_cfg (dict)
    adds the bulk-flux SST-sensitivity penalty on the flux outputs at step 0."""
    residual = clim is not None
    a = wd[:, 0]                                    # (B,5) physical wind seed (truth)
    loss = 0.0
    for k in range(K):
        x = torch.cat([oc[:, k], a], dim=1)         # (B,11) physical
        x_n = (x - xm) / xs
        pred_n = model(x_n)                         # (B,13) normalized
        fl_tgt_n = (fl[:, k] - ym[:, :N_FLUX]) / ys[:, :N_FLUX]        # (B,8)
        if residual:
            cl_next = clim[mi[:, k + 1]]                               # (B,5,H,W) clim@t+1
            wd_tgt_n = (wd[:, k + 1] - cl_next - ym[:, N_FLUX:]) / ys[:, N_FLUX:]   # anomaly
        else:
            wd_tgt_n = (wd[:, k + 1] - ym[:, N_FLUX:]) / ys[:, N_FLUX:]    # (B,5) winds@t+1
        tgt_n = torch.cat([fl_tgt_n, wd_tgt_n], dim=1)                 # (B,13)
        loss = loss + masked_mse(pred_n, tgt_n, mk[:, k])
        if var_w > 0.0:
            loss = loss + var_w * variance_penalty(pred_n[:, N_FLUX:], wd_tgt_n)
        if sens_cfg is not None and k == 0:
            loss = loss + sens_penalty(model, x_n, oc[:, 0, 0], a, mk[:, 0], xs, ys,
                                       sens_cfg["delta"], sens_cfg["hybm"],
                                       sens_cfg["lam_turb"], sens_cfg["lam_rad"])
        if residual:
            a = cl_next + (pred_n[:, N_FLUX:] * ys[:, N_FLUX:] + ym[:, N_FLUX:])   # full wind
        else:
            a = pred_n[:, N_FLUX:] * ys[:, N_FLUX:] + ym[:, N_FLUX:]       # -> next wind input
    return loss / K


@torch.no_grad()
def stability_probe(model, starts, steps, xm, xs, ym, ys, dev, sst_fill,
                    clim=None, mor=None):
    """Free-run |wind| for `steps` from each start (ocean from truth); pred vs truth
    mean wind speed at 30/60/120.  RESIDUAL mode: fed-back wind = clim[month]+anomaly."""
    X = np.load(X_NPY, mmap_mode="r"); Xa = np.load(XATM, mmap_mode="r")
    residual = clim is not None
    was = model.training; model.eval()
    marks = [m for m in (30, 60, 120) if m <= steps]
    psp = {m: [] for m in marks}; tsp = {m: [] for m in marks}
    for s in starts:
        a = torch.from_numpy(np.nan_to_num(np.asarray(Xa[s, 0:5], np.float32)))[None].to(dev)
        for k in range(1, steps + 1):
            oc = np.array(X[s + k - 1, 0:6], np.float32)
            oc[0] = np.nan_to_num(oc[0], nan=sst_fill); oc[1:] = np.nan_to_num(oc[1:], nan=0.0)
            ocb = torch.from_numpy(oc)[None].to(dev)
            x_n = (torch.cat([ocb, a], dim=1) - xm) / xs
            pred_n = model(x_n)
            anom = pred_n[:, N_FLUX:] * ys[:, N_FLUX:] + ym[:, N_FLUX:]
            if residual:
                cl = torch.from_numpy(clim[mor[(s + k) % SPY]])[None].to(dev)
                a = cl + anom                                          # full wind@t+1
            else:
                a = anom                                              # winds@t+1 physical
            if k in marks:
                psp[k].append(torch.sqrt(a[:, 0] ** 2 + a[:, 1] ** 2).mean().item())
                at = np.nan_to_num(np.asarray(Xa[s + k, 0:5], np.float32))
                tsp[k].append(float(np.sqrt(at[0] ** 2 + at[1] ** 2).mean()))
    if was:
        model.train()
    return ({m: float(np.mean(psp[m])) for m in marks},
            {m: float(np.mean(tsp[m])) for m in marks})


def warm_start(core, path):
    """Partial load from a validated flux-net checkpoint: copy every matching backbone
    weight and the first N_FLUX head channels; leave the 5 wind-head channels fresh.
    NOTE: v15 input ch5 is dSST_dt while this net's ch5 is SOLIN_prev -- init_conv
    column 5 is copied anyway (a warm start, not exact); training adapts it."""
    sd = torch.load(path, map_location="cpu")
    own = core.state_dict()
    loaded = partial = 0
    for k, v in sd.items():
        if k not in own:
            continue
        if own[k].shape == v.shape:
            own[k].copy_(v); loaded += 1
        elif k in ("head.weight", "head.bias") and v.shape[0] == N_FLUX:
            own[k][:N_FLUX].copy_(v); partial += 1     # first 8 flux channels
    core.load_state_dict(own)
    return loaded, partial


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_years", default="1980-2010")
    ap.add_argument("--val_years",   default="2011-2012")
    ap.add_argument("--test_years",  default="2013-2014")
    ap.add_argument("--rollout", type=int, default=8)
    ap.add_argument("--rollout_ramp_epochs", type=int, default=8)
    ap.add_argument("--rollout_start", type=int, default=2)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch",  type=int, default=12)
    ap.add_argument("--base",   type=int, default=64)   # same as v15
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--probe_steps", type=int, default=120)
    ap.add_argument("--var_loss_weight", type=float, default=0.0, metavar="W",
                    help="variance/grad-energy penalty on the WIND outputs (anti mean-"
                         "regression). 0 = off. Set >0 (~0.1) if winds collapse.")
    ap.add_argument("--wind_loss_weight", type=float, default=1.0, metavar="W",
                    help="scale on the wind-channel MSE relative to flux MSE.")
    # --- OPTION 2 (default-off) ---
    ap.add_argument("--residual", action="store_true",
                    help="wind outputs are ANOMALIES on a fixed monthly climatology; "
                         "fed back as clim[month]+anomaly (the working residual trick).")
    ap.add_argument("--sst_sens_reg", action="store_true",
                    help="bulk-flux SST-sensitivity penalty on the flux outputs "
                         "(dQ/dSST toward physical; radiation toward 0), at rollout step 0.")
    ap.add_argument("--sens_lambda", type=float, default=0.2, help="turbulent sens weight")
    ap.add_argument("--sens_lambda_rad", type=float, default=0.1, help="radiative sens weight")
    ap.add_argument("--sens_delta", type=float, default=0.5, help="SST perturbation (K)")
    ap.add_argument("--hybm_bot", type=float, default=0.992556, help="pbot=hybm_bot*PS")
    ap.add_argument("--init_from", default=None,
                    help="partial warm-start from a validated flux-net best_model.pt "
                         "(backbone + first 8 flux-head channels).")
    ap.add_argument("--wandb_name", default="mem24h-unified-flux-wind")
    ap.add_argument("--wandb_project", default="couple-unet")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu = torch.cuda.device_count()
    print(f"device={dev}  visible GPUs={ngpu}  residual={args.residual}  "
          f"sst_sens_reg={args.sst_sens_reg}", flush=True)

    def yr(spec):
        o = []
        for t in spec.split(","):
            if "-" in t: a, b = t.split("-"); o += list(range(int(a), int(b) + 1))
            else: o.append(int(t))
        return o
    tr_years, va_years, te_years = yr(args.train_years), yr(args.val_years), yr(args.test_years)
    Kmax = args.rollout

    tr_st = year_starts(tr_years, Kmax)[::args.stride]
    va_st = year_starts(va_years, Kmax)[::args.stride]
    print(f"  Train:  {len(tr_st):6d} windows  ({tr_years[0]}-{tr_years[-1]})")
    print(f"  Val:    {len(va_st):6d} windows  ({va_years[0]}-{va_years[-1]})", flush=True)

    # --- residual: build monthly wind climatology (12,5,H,W) from the training years ---
    mor = clim_np = None
    if args.residual:
        print("Building monthly wind climatology (residual mode) ...", flush=True)
        mor = month_of_row_table()
        Xa_mm = np.load(XATM, mmap_mode="r")
        clim_np = compute_climatology(Xa_mm, tr_years, mor)          # (12,5,H,W)
        np.save(out / "clim_gx1v7.npy", clim_np)
        print(f"  clim {clim_np.shape}  |wind| clim-mean "
              f"{float(np.sqrt(clim_np[:,0]**2+clim_np[:,1]**2).mean()):.2f} m/s", flush=True)

    print("Computing normalizer (subsampled masked train rows) ...", flush=True)
    norm_rows = np.unique(np.concatenate(
        [np.arange((y - 1980) * SPY, (y - 1980 + 1) * SPY) for y in tr_years]))[::20]
    x_mean, x_std, y_mean, y_std = compute_norm(norm_rows, clim=clim_np, mor=mor)
    np.savez(out / "normalizer.npz", x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    sst_fill = float(x_mean[0])
    print("  x_mean", np.round(x_mean, 3), flush=True)
    print("  y_mean", np.round(y_mean, 3), flush=True)

    if args.dry_run:
        tr_st, va_st = tr_st[:48], va_st[:24]; args.epochs = 2; args.probe_steps = 20

    tr_ds = UnifiedWindows(tr_st, Kmax, sst_fill, mor=mor)
    va_ds = UnifiedWindows(va_st, Kmax, sst_fill, mor=mor)
    dl_tr = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=6,
                       pin_memory=(dev == "cuda"), drop_last=True, persistent_workers=True)
    dl_va = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=4,
                       persistent_workers=True)

    model = UNet(n_in=len(IN_VARS), n_out=len(OUT_VARS), base=args.base, dropout=args.dropout).to(dev)
    core = model
    if args.init_from:
        ld, pt = warm_start(core, args.init_from)
        print(f"  warm-start from {args.init_from}: {ld} full + {pt} partial(head) tensors", flush=True)
    if ngpu >= 2:
        model = nn.DataParallel(model); print(f"  DataParallel across {ngpu} GPUs", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ema = ModelEMA(core) if args.ema else None

    def to_dev(a): return a.to(dev, non_blocking=True)
    xm = torch.from_numpy(x_mean).view(1, -1, 1, 1).to(dev)
    xs = torch.from_numpy(x_std ).view(1, -1, 1, 1).to(dev)
    ym = torch.from_numpy(y_mean).view(1, -1, 1, 1).to(dev)
    ys = torch.from_numpy(y_std ).view(1, -1, 1, 1).to(dev)
    clim_t = torch.from_numpy(clim_np).to(dev) if args.residual else None   # (12,5,H,W)
    sens_cfg = None
    if args.sst_sens_reg:
        sens_cfg = {"delta": args.sens_delta, "hybm": args.hybm_bot,
                    "lam_turb": args.sens_lambda, "lam_rad": args.sens_lambda_rad}

    use_wandb = WANDB and (not args.no_wandb)
    if use_wandb:
        run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
        (out / "wandb_run_id.txt").write_text(run.id)
        print(f"  wandb run {run.id}  ({run.get_url()})", flush=True)

    probe_starts = [(y - 1980) * SPY for y in va_years]
    best = float("inf")
    for ep in range(args.epochs):
        if args.rollout_ramp_epochs > 0:
            frac = min(1.0, ep / max(1, args.rollout_ramp_epochs))
            K = int(round(args.rollout_start + frac * (Kmax - args.rollout_start)))
        else:
            K = Kmax
        K = max(1, min(K, Kmax))

        model.train(); t0 = time.time(); tot = nb = 0
        for oc, wd, fl, mk, mi in dl_tr:
            oc, wd, fl, mk, mi = to_dev(oc), to_dev(wd), to_dev(fl), to_dev(mk), to_dev(mi)
            opt.zero_grad()
            loss = rollout_loss(model, oc, wd, fl, mk, mi, K, xm, xs, ym, ys,
                                var_w=args.var_loss_weight, clim=clim_t, sens_cfg=sens_cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema: ema.update(core)
            tot += loss.item(); nb += 1
        sched.step()

        model.eval(); vt = vn = 0
        with torch.no_grad():
            for oc, wd, fl, mk, mi in dl_va:
                oc, wd, fl, mk, mi = to_dev(oc), to_dev(wd), to_dev(fl), to_dev(mk), to_dev(mi)
                # val = pure rollout MSE (no sens/var) so selection stays comparable
                vt += rollout_loss(model, oc, wd, fl, mk, mi, Kmax, xm, xs, ym, ys,
                                   clim=clim_t).item(); vn += 1
        va = vt / max(vn, 1)

        psp, tsp = stability_probe(core, probe_starts, args.probe_steps, xm, xs, ym, ys,
                                   dev, sst_fill, clim=clim_np, mor=mor)
        sm = sorted(psp)[-1] if psp else 0
        print(f"ep {ep:3d}  K={K}  train {tot/max(nb,1):.4f}  valK{Kmax} {va:.4f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  |wind|@{sm}: pred {psp.get(sm,float('nan')):.1f} "
              f"truth {tsp.get(sm,float('nan')):.1f} m/s  {time.time()-t0:.0f}s", flush=True)
        if use_wandb:
            log = {"epoch": ep, "K": K, "train_loss": tot/max(nb, 1), "val_rollout_loss": va,
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
        "input_vars": IN_VARS, "output_vars": OUT_VARS,
        "n_flux": N_FLUX, "n_wind": N_WIND, "n_ocean": N_OCEAN,
        "flux_vars": FLUX, "wind_vars": WIND, "ocean_vars": OCEAN_IN,
        "cadence": "6-hourly", "var_loss_weight": args.var_loss_weight,
        "residual": bool(args.residual),
        "clim_file": ("clim_gx1v7.npy" if args.residual else None),
        "sst_sens_reg": bool(args.sst_sens_reg),
        "sens_lambda": args.sens_lambda, "sens_lambda_rad": args.sens_lambda_rad,
        "sens_delta": args.sens_delta,
        "init_from": args.init_from,
        "training": "unified autoregressive BPTT rollout (curriculum K2->8); winds fed back, "
                    "fluxes diagnostic; wind channels share input normalizer"
                    + (", RESIDUAL winds on monthly climatology" if args.residual else "")
                    + (", flux dQ/dSST sensitivity-regularized" if args.sst_sens_reg else ""),
        "note": "working-memo flux net (n_in=11) + next-step BL-atm outputs -> self-driving "
                "past 2014. fluxes[0:8]@t drive POP; winds[8:13]@t+1 autoregress"
                + (" as clim[month]+anomaly." if args.residual else "."),
    }, indent=2))

    core.load_state_dict(torch.load(out / "best_model.pt", map_location=dev)); core.eval()
    psp, tsp = stability_probe(core, [(y - 1980) * SPY for y in te_years],
                               args.probe_steps, xm, xs, ym, ys, dev, sst_fill,
                               clim=clim_np, mor=mor)
    print("\nFinal free-run stability (test-year starts):", flush=True)
    for m in sorted(psp):
        print(f"  step {m:3d}:  |wind| pred {psp[m]:.2f}  truth {tsp[m]:.2f} m/s", flush=True)
    (out / "stability_test.json").write_text(json.dumps(
        {"pred": psp, "truth": tsp, "probe_steps": args.probe_steps}, indent=2))
    print("wrote", out / "model_config.json", flush=True)
    if use_wandb: wandb.finish()


if __name__ == "__main__":
    main()
