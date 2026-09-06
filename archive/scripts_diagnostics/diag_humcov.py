"""diag_humcov.py — chase the tropical residual gustiness leaves on the table.

diag_gustiness.py fixed the wind-SPEED second moment (|U|_eff = sqrt(<|U|^2>) =
RMS speed) and recovered 72-95% of the clim-wind latent deficit -- EXCEPT the
tropics (72%, residual dLH +3.30). The residual is the humidity CROSS-moment
cov(|U|, Qbot): synoptic storms/gust-fronts pair high wind with DRY air, so the
time-mean evaporation carries a -cov(|U|,Qbot) rectification term that speed
inflation cannot reach where the wind variance itself is small (the tropics).

Fix (same spirit as gustiness, a measured 2nd moment, NOT a tuned knob): serve an
EFFECTIVE boundary-layer humidity/temperature that folds the missing cross-moment
back in. The net's bulk latent sensitivity is ~ dLH/dQbot ~ -kappa*|U|_served, so
injecting a flux increment kappa*cov(|U|,Qbot) needs

    Qbot_eff = Qbot_clim + cov(|U|,Qbot) / |U|_served
    Tbot_eff = Tbot_clim + cov(|U|,Tbot) / |U|_served   (sensible analog)

cov(|U|,Qbot) < 0 (windy = dry) -> drier served air -> more evaporation. The
divide-by-|U|_served makes the injected FLUX = kappa*cov independent of whichever
speed we feed, so it composes cleanly with gustiness.

Configs compared at TRUTH SST over 1980 (net@true = +0.002 ceiling proxy):
  true          - true synoptic winds+T+Q            (target)
  clim          - monthly-climatology inputs         (the +0.58 floor)
  gust          - clim + |U|_eff gustiness           (current product)
  gust_hcovQ    - gust + Qbot_eff only               (humidity cross-moment)
  gust_hcov     - gust + Qbot_eff + Tbot_eff         (humidity + sensible)
  meanwind_hcov - Jensen-mean wind + Qbot/Tbot_eff   (clean decomposition, no gust)

Reports dLH/dSH vs net@true and Qnet vs CREDIT truth by band. Read-only; uses the
same X_atm/clim the net trained on. Prototype uses 1980; production = 1980-2014.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch, sys
sys.path.insert(0, "/glade/u/home/praggarwal/couple")
sys.path.insert(0, "camulator_ud/climate")
from train_unet import UNet

CACHE = "/glade/work/praggarwal/couple_cache_gx1v7_mem24h"
SENSREG = "output/output_unet_gx1v7_sensreg"
CLIM = "output/output_atm_emulator_residual/clim_gx1v7.npy"
GUST = "output/gustiness_gx1v7.npy"          # production g2, exactly as deployed
POPH = ("/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_v15atmemuResidual1yr/run/"
        "g.e21.MEMO_GIAF_v01.pop.h.1980-01.nc")
DT6H = 21600.0
ROWS_PER_YEAR = 1456
I_SH, I_LH, I_FSDS, I_FLDS = 2, 3, 5, 6
MONTH_LEN = np.array([31,28,31,30,31,30,31,31,30,31,30,31])


def row_month(w):
    return int(np.searchsorted(np.cumsum(MONTH_LEN), (w + 4) // 4, side="right"))


def moment_clim(Xa, H, W):
    """Per-month per-cell moments of the TRUE 1980 winds/T/Q needed for the fixes.

    returns dict of (12,H,W): meanspd=<|U|>, rms=sqrt(<|U|^2>), spdclim=|<U>|,
    covQ=cov(|U|,Qbot), covT=cov(|U|,Tbot).  (T,Q at atm-input channels 2,3.)
    """
    z = lambda: np.zeros((12, H, W))
    n = np.zeros(12)
    sU, sV = z(), z()                      # mean vector
    sS, sS2 = z(), z()                     # speed, speed^2
    sT, sQ = z(), z()                      # mean T, Q
    sST, sSQ = z(), z()                    # speed*T, speed*Q
    for w in range(ROWS_PER_YEAR):
        mo = row_month(w)
        u = Xa[w, 0].astype(np.float64); v = Xa[w, 1].astype(np.float64)
        t = Xa[w, 2].astype(np.float64); q = Xa[w, 3].astype(np.float64)
        s = np.sqrt(u*u + v*v)
        sU[mo]+=u; sV[mo]+=v; sS[mo]+=s; sS2[mo]+=s*s
        sT[mo]+=t; sQ[mo]+=q; sST[mo]+=s*t; sSQ[mo]+=s*q; n[mo]+=1
    out = {k: z() for k in ["meanspd","rms","spdclim","covQ","covT"]}
    for mo in range(12):
        c = n[mo]
        mU, mV = sU[mo]/c, sV[mo]/c
        mS = sS[mo]/c; mS2 = sS2[mo]/c
        out["meanspd"][mo] = mS
        out["rms"][mo] = np.sqrt(np.maximum(mS2, 0.0))
        out["spdclim"][mo] = np.sqrt(mU*mU + mV*mV)
        out["covQ"][mo] = sSQ[mo]/c - mS*(sQ[mo]/c)
        out["covT"][mo] = sST[mo]/c - mS*(sT[mo]/c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=4)
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
    g2 = np.load(GUST).astype(np.float64)            # (12,H,W) production gustiness
    H, W = 384, 320

    print("building moment climatology from 1980 true winds ...")
    M = moment_clim(Xa, H, W)
    print(f"  <|U|>={np.sqrt((M['meanspd']**2).mean()):.2f}  rms={np.sqrt((M['rms']**2).mean()):.2f}  "
          f"|<U>|={np.sqrt((M['spdclim']**2).mean()):.2f} m/s  "
          f"covQ mean={M['covQ'].mean():.3e}  covT mean={M['covT'].mean():.3e}")

    d = Path(SENSREG); cfg = json.loads((d/"model_config.json").read_text())
    m = UNet(n_in=cfg["n_in"], n_out=cfg["n_out"], base=cfg.get("base",64), dropout=0.0).to(dev)
    m.load_state_dict(torch.load(d/"best_model.pt", map_location=dev, weights_only=True)); m.eval()
    nz = np.load(f"{SENSREG}/normalizer.npz")
    xm = nz["x_mean"][None,:,None,None]; xs = nz["x_std"][None,:,None,None]
    ym = nz["y_mean"][None,:,None,None]; ys = nz["y_std"][None,:,None,None]

    def run(xphys):
        yb = np.empty((xphys.shape[0], cfg["n_out"], H, W), np.float64)
        with torch.no_grad():
            for s in range(0, xphys.shape[0], args.bs):
                xn = (xphys[s:s+args.bs]-xm)/(xs+1e-8)
                yn = m(torch.from_numpy(xn.astype(np.float32)).to(dev)).cpu().numpy().astype(np.float64)
                yb[s:s+args.bs] = yn*ys+ym
        return yb

    def add_humcov(a, mo, served_speed, do_T):
        """add Qbot_eff (and optionally Tbot_eff) covariance correction in place."""
        ss = np.maximum(served_speed, 1e-6)
        a[:,3] = a[:,3] + M["covQ"][mo]/ss          # Qbot (ch 3)
        if do_T:
            a[:,2] = a[:,2] + M["covT"][mo]/ss      # Tbot (ch 2)
        return a

    rows = np.arange(0, ROWS_PER_YEAR, args.stride)
    months = np.array([row_month(w) for w in rows])
    configs = ["true","clim","gust","gust_hcovQ","gust_hcov","meanwind_hcov"]
    accSH = {c: np.zeros((H,W)) for c in configs}
    accLH = {c: np.zeros((H,W)) for c in configs}
    accQ  = {c: np.zeros((H,W)) for c in configs}
    accLH_tr = np.zeros((H,W)); accQ_tr = np.zeros((H,W))
    nacc = np.zeros((H,W))

    B = 40
    for i in range(0, len(rows), B):
        idx = rows[i:i+B]; mo = months[i:i+B]
        raw = X[idx].astype(np.float64); a_true = Xa[idx].astype(np.float64)
        dsst = ((raw[:,0]-raw[:,3])/86400.0)[:,None]
        base = np.concatenate([raw[:,0:5], dsst], axis=1)
        a_clim = clim[mo].copy()
        meanspd = M["meanspd"][mo]

        # deployment-EXACT gust: eff = sqrt(|clim wind|^2 + g2), direction kept
        sp = np.sqrt(a_clim[:,0]**2 + a_clim[:,1]**2)
        eff = np.sqrt(sp**2 + g2[mo])
        sc = eff / np.maximum(sp, 1e-6)
        a_gust = a_clim.copy(); a_gust[:,0] *= sc; a_gust[:,1] *= sc

        a_gQ  = add_humcov(a_gust.copy(), mo, eff, do_T=False)
        a_gQT = add_humcov(a_gust.copy(), mo, eff, do_T=True)

        # clean decomposition: serve mean instantaneous speed + explicit covariance
        sc2 = meanspd / np.maximum(sp, 1e-6)
        a_mw = a_clim.copy(); a_mw[:,0] *= sc2; a_mw[:,1] *= sc2
        a_mw = add_humcov(a_mw, mo, meanspd, do_T=True)

        variants = {"true":a_true, "clim":a_clim, "gust":a_gust,
                    "gust_hcovQ":a_gQ, "gust_hcov":a_gQT, "meanwind_hcov":a_mw}

        mk = (mask[idx] > 0.5)
        yt_sh = Y[idx,2].astype(np.float64); yt_lh = Y[idx,3].astype(np.float64)
        yt_fs = Yr[idx,0].astype(np.float64); yt_fl = Yr[idx,1].astype(np.float64)
        for c in configs:
            y = run(np.concatenate([base, variants[c]], axis=1))
            for n in range(len(idx)):
                w = mk[n]
                accSH[c] += np.where(w, y[n,I_SH]/DT6H, 0.0)
                accLH[c] += np.where(w, y[n,I_LH]/DT6H, 0.0)
                accQ[c]  += np.where(w, (y[n,I_SH]+y[n,I_LH]+y[n,I_FSDS]+y[n,I_FLDS])/DT6H, 0.0)
        for n in range(len(idx)):
            w = mk[n]; nacc += w
            accLH_tr += np.where(w, yt_lh[n]/DT6H, 0.0)
            accQ_tr  += np.where(w, (yt_sh[n]+yt_lh[n]+yt_fs[n]+yt_fl[n])/DT6H, 0.0)
        print(f"  {i+len(idx)}/{len(rows)}", end="\r")
    print()

    ok = nacc > 0
    def cm(acc):
        o = np.full((H,W), np.nan); o[ok] = acc[ok]/nacc[ok]; return o
    bands = [("GLOBAL", ok), ("SO <-45", ok&(tlat<-45)),
             ("TROP -45..45", ok&(tlat>=-45)&(tlat<=45)), ("NH >45", ok&(tlat>45))]
    def wmean(f,b):
        wt=tarea[b]; v=f[b]; g=~np.isnan(v)
        return float(np.average(v[g],weights=wt[g])) if g.any() else np.nan

    lh = {c: cm(accLH[c]) for c in configs}
    sh = {c: cm(accSH[c]) for c in configs}
    q  = {c: cm(accQ[c])  for c in configs}
    q_tr = cm(accQ_tr)

    print("\ndLH vs net@true [W/m2, + = less cooling = the deficit]:")
    cols = ["clim","gust","gust_hcovQ","gust_hcov","meanwind_hcov"]
    print(f"{'band':<14}" + "".join(f"{c:>14}" for c in cols))
    for nm,b in bands:
        bt = wmean(lh["true"],b)
        print(f"{nm:<14}" + "".join(f"{wmean(lh[c],b)-bt:>14.2f}" for c in cols))

    print("\ndLH recovered vs the clim floor [%] (100 = matches net@true):")
    for nm,b in bands:
        bt = wmean(lh["true"],b); dclim = wmean(lh["clim"],b)-bt
        recs = []
        for c in cols[1:]:
            dg = wmean(lh[c],b)-bt
            recs.append((dclim-dg)/dclim*100 if abs(dclim)>1e-6 else np.nan)
        print(f"  {nm:<14}" + "".join(f"{c}={r:.0f}%  " for c,r in zip(cols[1:],recs)))

    print("\ndSH vs net@true [W/m2]:")
    print(f"{'band':<14}" + "".join(f"{c:>14}" for c in cols))
    for nm,b in bands:
        bt = wmean(sh["true"],b)
        print(f"{nm:<14}" + "".join(f"{wmean(sh[c],b)-bt:>14.2f}" for c in cols))

    print("\nQnet vs CREDIT truth [W/m2]  (want near net@true; TRUTH shown):")
    print(f"{'band':<14}" + "".join(f"{c:>13}" for c in configs) + f"{'TRUTH':>10}")
    for nm,b in bands:
        print(f"{nm:<14}" + "".join(f"{wmean(q[c],b):>13.2f}" for c in configs)
              + f"{wmean(q_tr,b):>10.2f}")


if __name__ == "__main__":
    main()
