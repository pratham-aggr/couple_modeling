#!/usr/bin/env python
"""
Model-agnostic comparison of two MEMO flux emulators trained on DIFFERENT objectives.

  p5ge0uvh  (out: output_unet_mem24h_dsst_temporal_radprecip_ucast)
            MAE pretrain -> CRPS fine-tune w/ MC Dropout  => PREDICTIVE DISTRIBUTION
  vfcg1m61  (out: output_unet_mem24h_dsst_temporal_radprecip)
            MSE                                            => POINT prediction

R^2 is NOT a fair referee (it IS normalised MSE -> structurally favours vfcg1m61).
We compare on the SAME test cells / vars with:
  1. CRPS  (lead, proper score). Prob model: MC-dropout ensemble (N_MEMBERS passes).
           Point model: degenerate spike  => CRPS == MAE  (exact, fair).
  2. RMSE & MAE on equal footing  (prob model -> ENSEMBLE MEAN as its point pred).
  3. Calibration of the prob model: ensemble spread, rank/PIT histogram,
     90%/50% coverage, spread-skill correlation.

Reuses train_unet.py's UNet, Normalizer, enable_mc_dropout and the cache + each
model's own training-only normalizer + saved temporal test_indices.npy.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, "/glade/u/home/praggarwal/couple")
from train_unet import UNet, Normalizer, enable_mc_dropout

CACHE      = "/glade/work/praggarwal/couple_cache_mem24h"
ROOT       = "/glade/u/home/praggarwal/couple"
DIR_PROB   = f"{ROOT}/output/output_unet_mem24h_dsst_temporal_radprecip_ucast"   # p5ge0uvh
DIR_POINT  = f"{ROOT}/output/output_unet_mem24h_dsst_temporal_radprecip"          # vfcg1m61
OUT_JSON   = f"{ROOT}/output/compare_prob_vs_point.json"

VARS       = ["TAUX","TAUY","SHFLX","LHFLX","QFLX","FSDS_J","FLDS_J","PRECT"]
HEAT       = ["SHFLX","LHFLX","FLDS_J"]              # coupling-relevant heat fluxes
N_OUT      = 8
N_MEMBERS  = 40                                     # MC-dropout stochastic passes
BATCH      = 16
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
RANK_BINS  = N_MEMBERS + 1                          # ranks 0..M


def load_model(d, dropout):
    m = UNet(n_in=6, n_out=N_OUT, base=64, dropout=dropout).to(DEVICE)
    sd = torch.load(f"{d}/best_model.pt", map_location=DEVICE)
    sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    m.load_state_dict(sd)
    m.eval()
    return m


def build_inputs(idx):
    """Reconstruct the exact 6-channel input + 8-channel physical truth + mask
    for the given cache rows, matching train_unet.py (--memory --dsst_dt
    --with_rad --with_precip)."""
    X   = np.load(f"{CACHE}/X.npy",        mmap_mode="r")   # (N,6,H,W) [SST,ICE,SOLIN,SST_p,ICE_p,SOLIN_p]
    Y   = np.load(f"{CACHE}/Y.npy",        mmap_mode="r")   # (N,5,H,W) fluxes
    Yr  = np.load(f"{CACHE}/Y_rad.npy",    mmap_mode="r")   # (N,2,H,W) FSDS,FLDS
    Yp  = np.load(f"{CACHE}/Y_precip.npy", mmap_mode="r")   # (N,1,H,W) PRECT
    M   = np.load(f"{CACHE}/mask.npy",     mmap_mode="r")   # (N,H,W)
    xs, ys_, mk = [], [], []
    for i in idx:
        xi = X[i][[0,1,2,3,4]]                              # mem channels (prev_solin off)
        dsst = ((X[i][0] - X[i][3]) / 86400.0)[None]        # dSST_dt
        xs.append(np.concatenate([xi, dsst], 0).astype(np.float32))
        ys_.append(np.concatenate([Y[i], Yr[i], Yp[i]], 0).astype(np.float32))
        mk.append((M[i] > 0.5).astype(np.float32))
    return np.stack(xs), np.stack(ys_), np.stack(mk)


def main():
    print(f"device={DEVICE}  members={N_MEMBERS}")
    test_idx = np.load(f"{DIR_PROB}/test_indices.npy")
    print(f"test samples: {len(test_idx)}")

    norm_prob  = Normalizer.load(f"{DIR_PROB}/normalizer.npz")
    norm_point = Normalizer.load(f"{DIR_POINT}/normalizer.npz")
    model_prob  = load_model(DIR_PROB,  dropout=0.1)   # dropout layers exist -> MC dropout
    model_point = load_model(DIR_POINT, dropout=0.0)

    X, Ytrue, Mask = build_inputs(test_idx)
    print(f"X{X.shape} Ytrue{Ytrue.shape} Mask{Mask.shape}")

    def nrm(arr, n):  # physical -> normalized input
        return (arr - n.x_mean[None,:,None,None]) / (n.x_std[None,:,None,None] + 1e-8)

    ys_p = torch.tensor(norm_prob.y_std,  device=DEVICE).view(1,-1,1,1)
    ym_p = torch.tensor(norm_prob.y_mean, device=DEVICE).view(1,-1,1,1)
    ys_q = torch.tensor(norm_point.y_std, device=DEVICE).view(1,-1,1,1)
    ym_q = torch.tensor(norm_point.y_mean,device=DEVICE).view(1,-1,1,1)

    # ---- streaming accumulators (per variable, over ocean cells) ----
    nv = N_OUT
    W  = 0.0                                            # total ocean-cell count
    # CRPS
    crps_prob  = np.zeros(nv); crps_point = np.zeros(nv)
    # deterministic
    se_prob = np.zeros(nv); ae_prob = np.zeros(nv)      # ensemble-mean errors
    se_pt   = np.zeros(nv); ae_pt   = np.zeros(nv)      # point errors
    # spread
    spread_sum = np.zeros(nv)
    # rank histogram
    rank_hist = np.zeros((nv, RANK_BINS))
    # coverage
    cov90 = np.zeros(nv); cov50 = np.zeros(nv)
    # spread-skill correlation accumulators
    s_s = np.zeros(nv); s_e = np.zeros(nv); s_ss = np.zeros(nv); s_ee = np.zeros(nv); s_se = np.zeros(nv)

    sample_cell = None                                  # raw values for hand-check

    Xn_prob  = torch.tensor(nrm(X, norm_prob),  dtype=torch.float32)
    Xn_point = torch.tensor(nrm(X, norm_point), dtype=torch.float32)
    Yt       = torch.tensor(Ytrue, device=DEVICE)
    Mk       = torch.tensor(Mask,  device=DEVICE)

    N = len(test_idx)
    with torch.no_grad():
        for b0 in range(0, N, BATCH):
            b1 = min(b0+BATCH, N)
            xb_p = Xn_prob[b0:b1].to(DEVICE)
            xb_q = Xn_point[b0:b1].to(DEVICE)
            yt   = Yt[b0:b1]                              # (B,8,H,W) physical
            mk   = Mk[b0:b1].unsqueeze(1)                 # (B,1,H,W)

            # ----- prob model: MC-dropout ensemble -----
            enable_mc_dropout(model_prob)
            ens = torch.stack([model_prob(xb_p)*ys_p + ym_p for _ in range(N_MEMBERS)], 0)  # (M,B,8,H,W)
            model_prob.eval()
            # ----- point model -----
            pt  = model_point(xb_q)*ys_q + ym_q           # (B,8,H,W)

            mean = ens.mean(0)                            # (B,8,H,W)
            std  = ens.std(0)                             # population-ish (unbiased default n-1)

            # CRPS prob (unbiased estimator):  mean_i|x_i-y| - 1/(2M(M-1)) sum_ij|x_i-x_j|
            skill = (ens - yt.unsqueeze(0)).abs().mean(0)            # (B,8,H,W)
            # pairwise term via sort trick: sum_ij|xi-xj| = 2*sum_k (2k-M+1) x_(k)
            es, _ = torch.sort(ens, dim=0)                          # (M,B,8,H,W)
            kk = torch.arange(1, N_MEMBERS+1, device=DEVICE).view(-1,1,1,1,1).float()
            pair = (2*(2*kk - N_MEMBERS - 1) * es).sum(0)          # = sum_ij|xi-xj|
            crps_p = skill - pair / (2.0 * N_MEMBERS * (N_MEMBERS-1))
            # CRPS point = MAE
            crps_q = (pt - yt).abs()

            mk_b = mk                                              # (B,1,H,W)
            w = mk_b.sum().item()
            W += w
            def msum(t):   # masked sum over (B,1or8,H,W) -> per var (8,)
                return (t * mk_b).sum(dim=(0,2,3)).cpu().numpy()

            crps_prob  += msum(crps_p)
            crps_point += msum(crps_q)
            se_prob += msum((mean-yt)**2); ae_prob += msum((mean-yt).abs())
            se_pt   += msum((pt-yt)**2);   ae_pt   += msum((pt-yt).abs())
            spread_sum += msum(std)

            # rank histogram: #members strictly below truth (0..M)
            rank = (ens < yt.unsqueeze(0)).sum(0).clamp(0, N_MEMBERS)   # (B,8,H,W)
            mb = mk_b.squeeze(1) > 0.5                                  # (B,H,W)
            for v in range(nv):
                rv = rank[:,v,:,:][mb].cpu().numpy()
                rank_hist[v] += np.bincount(rv, minlength=RANK_BINS)

            # coverage via empirical quantiles
            q05 = torch.quantile(ens, 0.05, dim=0); q95 = torch.quantile(ens, 0.95, dim=0)
            q25 = torch.quantile(ens, 0.25, dim=0); q75 = torch.quantile(ens, 0.75, dim=0)
            in90 = ((yt>=q05)&(yt<=q95)).float(); in50 = ((yt>=q25)&(yt<=q75)).float()
            cov90 += msum(in90); cov50 += msum(in50)

            # spread-skill correlation: std vs |mean-truth| across cells
            err = (mean-yt).abs()
            s_s  += msum(std);       s_e  += msum(err)
            s_ss += msum(std**2);    s_ee += msum(err**2); s_se += msum(std*err)

            # grab one raw ocean cell (heat flux SHFLX=idx2) from first batch
            if sample_cell is None:
                bb = 0
                oc = torch.nonzero(mk_b[bb,0] > 0.5)
                if len(oc):
                    yy, xx = oc[len(oc)//2].tolist()
                    sample_cell = {
                        "var": "SHFLX", "cell_yx": [yy, xx],
                        "truth": float(yt[bb,2,yy,xx]),
                        "ens_samples": [float(ens[k,bb,2,yy,xx]) for k in range(N_MEMBERS)],
                        "ens_mean": float(mean[bb,2,yy,xx]),
                        "ens_std":  float(std[bb,2,yy,xx]),
                        "point_pred": float(pt[bb,2,yy,xx]),
                    }
            if b0 % (BATCH*20) == 0:
                print(f"  {b1}/{N}", flush=True)

    # ---- finalize ----
    res = {"vars": VARS, "n_test": int(N), "n_members": N_MEMBERS, "ocean_cells": float(W)}
    res["crps_prob"]   = (crps_prob  / W).tolist()
    res["crps_point"]  = (crps_point / W).tolist()
    res["rmse_prob"]   = np.sqrt(se_prob/W).tolist()
    res["rmse_point"]  = np.sqrt(se_pt /W).tolist()
    res["mae_prob"]    = (ae_prob/W).tolist()
    res["mae_point"]   = (ae_pt /W).tolist()
    res["mean_spread"] = (spread_sum/W).tolist()
    res["cov90"]       = (cov90/W).tolist()
    res["cov50"]       = (cov50/W).tolist()
    # spread-skill pearson
    n = W
    cov = s_se/n - (s_s/n)*(s_e/n)
    vs  = s_ss/n - (s_s/n)**2
    ve  = s_ee/n - (s_e/n)**2
    res["spread_skill_corr"] = (cov/np.sqrt(vs*ve + 1e-30)).tolist()
    res["rank_hist"] = rank_hist.tolist()
    res["sample_cell"] = sample_cell

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT_JSON}")

    # ---- pretty report ----
    def row(name, a, b):
        return f"  {name:8s}  prob={a:12.5g}   point={b:12.5g}   Δ(prob-point)={a-b:+.4g}"
    print("\n" + "="*78)
    print("1+2) CRPS / RMSE / MAE per variable  (lower=better; same test cells)")
    print("="*78)
    for i,v in enumerate(VARS):
        tag = " <-- HEAT" if v in HEAT else ""
        print(f"\n[{v}]{tag}")
        print(row("CRPS", res['crps_prob'][i],  res['crps_point'][i]))
        print(row("RMSE", res['rmse_prob'][i],  res['rmse_point'][i]))
        print(row("MAE",  res['mae_prob'][i],   res['mae_point'][i]))
    print("\n" + "="*78)
    print("3) Calibration of PROB model (p5ge0uvh)")
    print("="*78)
    print(f"{'var':8s} {'mean_spread':>12s} {'cov90%':>8s} {'cov50%':>8s} {'spr-skill r':>12s}")
    for i,v in enumerate(VARS):
        print(f"{v:8s} {res['mean_spread'][i]:12.4g} "
              f"{100*res['cov90'][i]:8.1f} {100*res['cov50'][i]:8.1f} "
              f"{res['spread_skill_corr'][i]:12.3f}")
    print("\nRank/PIT histogram (normalized, per var; flat=calibrated, U=overconfident, dome=underconfident)")
    for i,v in enumerate(VARS):
        h = np.array(res['rank_hist'][i]); h = h/h.sum()
        # compress to 11 super-bins for readability
        edges = np.linspace(0, RANK_BINS, 12).astype(int)
        comp = [h[edges[j]:edges[j+1]].sum() for j in range(11)]
        bar = "".join("█▇▆▅▄▃▂▁ "[min(8, max(0, 8-int(c*11*8)))] for c in comp)
        print(f"  {v:8s} |{bar}|  (uniform={1/11:.3f}; min={min(comp):.3f} max={max(comp):.3f})")
    print("\nSample raw cell (hand-check CRPS/coverage):")
    sc = res['sample_cell']
    if sc:
        s = np.array(sc['ens_samples'])
        print(f"  {sc['var']} cell {sc['cell_yx']}: truth={sc['truth']:.4g}  "
              f"point={sc['point_pred']:.4g}  ens_mean={sc['ens_mean']:.4g}  ens_std={sc['ens_std']:.4g}")
        print(f"  ens min/5%/50%/95%/max = {s.min():.4g}/{np.percentile(s,5):.4g}/"
              f"{np.median(s):.4g}/{np.percentile(s,95):.4g}/{s.max():.4g}")
        print(f"  first 8 samples: {np.round(s[:8],3).tolist()}")


if __name__ == "__main__":
    main()
