"""diag_gustiness.py — option 2 prototype: recover the latent-heat rectification
that climatology winds lose, via an effective (gustiness-inflated) wind speed.

diag_windfloor.py showed the +0.58 K floor is a latent deficit: feeding the flux
net climatology winds vs true winds costs dQ_wind +14.2 W/m2 (dLH +9.9). Cause =
lost synoptic wind variance -> lost rectified evaporation. Fix (COARE-style
gustiness, generalized from convective to synoptic scale): replace the served wind
SPEED with an effective speed that folds the missing sub-monthly variance back in

    |U|_eff = sqrt(|U_clim|^2 + alpha * g^2),   g^2 = Var_submonthly(U)+Var(V)

direction preserved. g^2(month,cell) is a read-only climatology from the TRUE winds
(here 1980; production = 1980-2014). Feed |U|_eff to the net and measure how much of
the +9.9 dLH it recovers, plus an OOD check (eff speed vs training max).
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
POPH = ("/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_v15atmemuResidual1yr/run/"
        "g.e21.MEMO_GIAF_v01.pop.h.1980-01.nc")
DT6H = 21600.0
ROWS_PER_YEAR = 1456
I_SH, I_LH, I_FSDS, I_FLDS = 2, 3, 5, 6
MONTH_LEN = np.array([31,28,31,30,31,30,31,31,30,31,30,31])


def row_month(w):
    return int(np.searchsorted(np.cumsum(MONTH_LEN), (w + 4) // 4, side="right"))


def gustiness_clim(Xa, H, W):
    """g^2(12,H,W) = per-month per-cell Var(Ubot)+Var(Vbot) of TRUE winds, 1980."""
    s1u = np.zeros((12, H, W)); s2u = np.zeros((12, H, W))
    s1v = np.zeros((12, H, W)); s2v = np.zeros((12, H, W))
    cnt = np.zeros(12)
    for w in range(ROWS_PER_YEAR):
        mo = row_month(w)
        u = Xa[w, 0].astype(np.float64); v = Xa[w, 1].astype(np.float64)
        s1u[mo] += u; s2u[mo] += u*u; s1v[mo] += v; s2v[mo] += v*v; cnt[mo] += 1
    g2 = np.zeros((12, H, W))
    for mo in range(12):
        n = cnt[mo]
        varu = s2u[mo]/n - (s1u[mo]/n)**2
        varv = s2v[mo]/n - (s1v[mo]/n)**2
        g2[mo] = np.maximum(varu + varv, 0.0)
    return g2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.5, 1.0])
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
    clim = np.load(CLIM).astype(np.float64)
    H, W = 384, 320

    print("building gustiness climatology from 1980 true winds ...")
    g2 = gustiness_clim(Xa, H, W)
    print(f"  g (RMS submonthly wind std) global mean = {np.sqrt(g2.mean()):.2f} m/s  "
          f"max = {np.sqrt(g2.max()):.2f}")
    true_speed_max = float(np.sqrt((Xa[:ROWS_PER_YEAR, 0].astype(np.float64)**2 +
                                    Xa[:ROWS_PER_YEAR, 1].astype(np.float64)**2).max()))
    print(f"  training-year max |wind| = {true_speed_max:.1f} m/s (OOD reference)")

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

    rows = np.arange(0, ROWS_PER_YEAR, args.stride)
    months = np.array([row_month(w) for w in rows])

    configs = ["true", "clim"] + [f"gust{a}" for a in args.alpha]
    accLH = {c: np.zeros((H,W)) for c in configs}
    accQ  = {c: np.zeros((H,W)) for c in configs}
    accLH_truth = np.zeros((H,W)); accQ_truth = np.zeros((H,W))
    nacc = np.zeros((H,W))
    eff_speed_max = {f"gust{a}": 0.0 for a in args.alpha}

    B = 40
    for i in range(0, len(rows), B):
        idx = rows[i:i+B]; mo = months[i:i+B]
        raw = X[idx].astype(np.float64); a_true = Xa[idx].astype(np.float64)
        dsst = ((raw[:,0]-raw[:,3])/86400.0)[:,None]
        base = np.concatenate([raw[:,0:5], dsst], axis=1)
        a_clim = clim[mo].copy()
        variants = {"true": a_true, "clim": a_clim}
        sp_clim = np.sqrt(a_clim[:,0]**2 + a_clim[:,1]**2)
        for al in args.alpha:
            ag = a_clim.copy()
            eff = np.sqrt(sp_clim**2 + al*g2[mo])
            scale = eff/np.maximum(sp_clim, 1e-6)
            ag[:,0] *= scale; ag[:,1] *= scale
            variants[f"gust{al}"] = ag
            eff_speed_max[f"gust{al}"] = max(eff_speed_max[f"gust{al}"], float(eff.max()))
        mk = (mask[idx] > 0.5)
        yt_lh = Y[idx,3].astype(np.float64)
        yt = {"SH":Y[idx,2].astype(np.float64), "LH":yt_lh,
              "FSDS":Yr[idx,0].astype(np.float64), "FLDS":Yr[idx,1].astype(np.float64)}
        for c in configs:
            y = run(np.concatenate([base, variants[c]], axis=1))
            for n in range(len(idx)):
                w = mk[n]
                accLH[c] += np.where(w, y[n,I_LH]/DT6H, 0.0)
                accQ[c]  += np.where(w, (y[n,I_SH]+y[n,I_LH]+y[n,I_FSDS]+y[n,I_FLDS])/DT6H, 0.0)
        for n in range(len(idx)):
            w = mk[n]; nacc += w
            accLH_truth += np.where(w, yt["LH"][n]/DT6H, 0.0)
            accQ_truth  += np.where(w, (yt["SH"][n]+yt["LH"][n]+yt["FSDS"][n]+yt["FLDS"][n])/DT6H, 0.0)
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

    lh = {c: cm(accLH[c]) for c in configs}; lh_tr = cm(accLH_truth)
    q  = {c: cm(accQ[c])  for c in configs}; q_tr  = cm(accQ_truth)

    print("\nOOD check (effective wind speed vs training max %.1f m/s):" % true_speed_max)
    for al in args.alpha:
        print(f"  gust{al}: max eff |wind| = {eff_speed_max[f'gust{al}']:.1f} m/s  "
              f"{'OK' if eff_speed_max[f'gust{al}']<=true_speed_max else 'OOD!'}")

    print("\nLATENT HEAT dLH vs true winds [W/m2, + = less cooling than true=the deficit]:")
    hdr = f"{'band':<14}" + "".join(f"{c:>10}" for c in ["clim"]+[f'gust{a}' for a in args.alpha]) + f"{'|dLH_true':>10}"
    print(hdr)
    for nm,b in bands:
        base_true = wmean(lh["true"],b)
        vals = [wmean(lh[c],b)-base_true for c in ["clim"]+[f"gust{a}" for a in args.alpha]]
        print(f"{nm:<14}" + "".join(f"{v:>10.2f}" for v in vals) + f"{0.0:>10.2f}")

    print("\nfraction of the clim dLH deficit RECOVERED by gustiness (GLOBAL & bands):")
    for nm,b in bands:
        bt = wmean(lh["true"],b); dclim = wmean(lh["clim"],b)-bt
        for al in args.alpha:
            dg = wmean(lh[f"gust{al}"],b)-bt
            rec = (dclim-dg)/dclim*100 if abs(dclim)>1e-6 else np.nan
            print(f"  {nm:<14} alpha={al}: dLH {dclim:+.2f} -> {dg:+.2f}  recovered {rec:.0f}%")

    print("\nQnet vs TRUTH [W/m2]:  (true, clim, gust*, TRUTH)")
    print(f"{'band':<14}" + "".join(f"{c:>9}" for c in configs) + f"{'TRUTH':>9}")
    for nm,b in bands:
        print(f"{nm:<14}" + "".join(f"{wmean(q[c],b):>9.2f}" for c in configs) + f"{wmean(q_tr,b):>9.2f}")


if __name__ == "__main__":
    main()
