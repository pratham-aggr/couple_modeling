"""audit_dqdsst_pz220.py — offline dQnet/dSST feedback audit for pz220f0b (n_in=6).

Adapts audit_dqdsst.py to the 6-channel ocean-only net (SST,ICEFRAC,SOLIN,SST_prev,
ICEFRAC_prev,dSST_dt).  Compares the MSE base (output_unet_gx1v7_atm_aux) vs the
sensreg fine-tune (output_unet_gx1v7_atm_aux_sensreg, epoch 10) vs the shr_flux
bulk-physics turbulent target on the SAME 2013-2014 test samples.

GATE: base has the +7 W/m2/K correlation defect (wrong sign, runaway); sensreg should
be NEGATIVE (restoring, -10..-25 target).  Sustained SST perturbation (SST & SST_prev
shifted equally so dSST_dt stays 0 = the slow/equilibrium channel that governs drift).
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch, sys
sys.path.insert(0, "camulator_ud/climate")
from train_unet import UNet, bulk_sens_target

CACHE = "/glade/work/praggarwal/couple_cache_gx1v7_mem24h"
HYBM_BOT = 0.992556; SIGMA = 5.67e-8; EPS = 0.98; DT6H = 21600.0
I_SH, I_LH, I_FSDS, I_FLDS = 2, 3, 5, 6   # flux channel indices (first 8 of 13)

def load_model(out_dir, device):
    d = Path(out_dir); cfg = json.loads((d / "model_config.json").read_text())
    m = UNet(n_in=cfg["n_in"], n_out=cfg["n_out"], base=cfg.get("base", 64), dropout=0.0).to(device)
    m.load_state_dict(torch.load(d / "best_model.pt", map_location=device, weights_only=True))
    m.eval(); return m, cfg

def build_inputs(idxs, X, Xa):
    """6-channel ocean-only input (mem_ch[0:5] + dSST_dt). a = BL atm for bulk target."""
    raw = X[idxs].astype(np.float64)          # (N,6,H,W)
    a   = Xa[idxs].astype(np.float64)         # (N,5,H,W) BL atm (offline, for target only)
    dsst = ((raw[:, 0] - raw[:, 3]) / 86400.0)[:, None]
    x = np.concatenate([raw[:, 0:5], dsst], axis=1)   # (N,6,H,W)
    return x, raw, a

def fwd(model, x_phys, xm, xs, ym, ys, device, chans, bs=8):
    """Return summed physical output over `chans` (W/m2) per cell."""
    N = x_phys.shape[0]
    out = np.zeros((N, x_phys.shape[2], x_phys.shape[3]), np.float64)
    with torch.no_grad():
        for s in range(0, N, bs):
            xn = (x_phys[s:s+bs] - xm) / (xs + 1e-8)
            yn = model(torch.from_numpy(xn.astype(np.float32)).to(device)).cpu().numpy().astype(np.float64)
            y = yn * ys + ym
            out[s:s+bs] = sum(y[:, c] for c in chans) / DT6H
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--dS", type=float, default=0.5)
    ap.add_argument("--base_dir",    default="output/output_unet_gx1v7_atm_aux")
    ap.add_argument("--sensreg_dir", default="output/output_unet_gx1v7_atm_aux_sensreg")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  n={args.n}  dS={args.dS} K  (SUSTAINED SST & SST_prev shift)")

    X  = np.load(f"{CACHE}/X.npy",     mmap_mode="r")
    Xa = np.load(f"{CACHE}/X_atm.npy", mmap_mode="r")
    mask = np.load(f"{CACHE}/mask.npy", mmap_mode="r")
    test_idx = np.load(f"{args.base_dir}/test_indices.npy")
    sel = np.sort(np.random.default_rng(0).choice(test_idx, size=min(args.n, len(test_idx)), replace=False))
    print(f"selected {len(sel)} of {len(test_idx)} test rows")

    x0, raw, a = build_inputs(sel, X, Xa)
    m = (mask[sel] > 0.5); sst_phys = raw[:, 0]
    xp = x0.copy(); xp[:, 0] += args.dS; xp[:, 3] += args.dS
    xn_ = x0.copy(); xn_[:, 0] -= args.dS; xn_[:, 3] -= args.dS
    mm = lambda f: float(f[m].mean())

    results = {}
    for name, out_dir in [("MSE base", args.base_dir), ("sensreg (ep10)", args.sensreg_dir)]:
        model, cfg = load_model(out_dir, device)
        nz = np.load(f"{out_dir}/normalizer.npz")
        xm = nz["x_mean"][None,:,None,None]; xs = nz["x_std"][None,:,None,None]
        ym = nz["y_mean"][None,:,None,None]; ys = nz["y_std"][None,:,None,None]
        qp = fwd(model, xp,  xm,xs,ym,ys, device, [I_SH,I_LH,I_FSDS,I_FLDS])
        qn = fwd(model, xn_, xm,xs,ym,ys, device, [I_SH,I_LH,I_FSDS,I_FLDS])
        dq = mm((qp - qn) / (2*args.dS))
        comp = {}
        for lbl, ch in [("SH",I_SH),("LH",I_LH),("FSDS",I_FSDS),("FLDS",I_FLDS)]:
            cp = fwd(model, xp,  xm,xs,ym,ys, device, [ch])
            cn = fwd(model, xn_, xm,xs,ym,ys, device, [ch])
            comp[lbl] = mm((cp - cn) / (2*args.dS))
        lwup = mm(-4*EPS*SIGMA*(sst_phys**3))
        eff = dq + lwup
        results[name] = dict(total=dq, effective=eff, lwup=lwup, **comp)
        print(f"\n=== {name} ===  ({out_dir})")
        print(f"  dQnet/dSST (UNet): {dq:+.2f} W/m2/K")
        print(f"    dSH {comp['SH']:+.2f}  dLH {comp['LH']:+.2f}  dFSDS {comp['FSDS']:+.2f}  dFLDS {comp['FLDS']:+.2f}")
        print(f"  + POP LWUP {lwup:+.2f}  =>  EFFECTIVE coupled dQnet/dSST: {eff:+.2f} W/m2/K")

    dturb = np.empty_like(sst_phys)
    for k in range(len(sel)):
        dturb[k] = bulk_sens_target(sst_phys[k], a[k,0],a[k,1],a[k,2],a[k,3],a[k,4], args.dS, HYBM_BOT)
    tgt = mm(dturb)
    print(f"\n=== bulk-physics reference (shr_flux port) ===")
    print(f"  d(sen+lat)/dSST target: {tgt:+.2f} W/m2/K")

    b = results["MSE base"]; s = results["sensreg (ep10)"]
    print("\n---------------- VERDICT ----------------")
    print(f"  base total   {b['total']:+.2f} | sensreg total {s['total']:+.2f} | bulk target {tgt:+.2f}")
    print(f"  base eff     {b['effective']:+.2f} | sensreg eff   {s['effective']:+.2f}")
    verdict = "PASS" if (s['total'] < 0 and s['effective'] < 0) else "REVIEW"
    print(f"  GATE (sensreg feedback restoring, negative): {verdict}")
    json.dump({"base": b, "sensreg": s, "bulk_target": tgt, "dS": args.dS, "n": int(len(sel)), "verdict": verdict},
              open(f"{args.sensreg_dir}/dqdsst_audit.json", "w"), indent=2)
    print(f"wrote {args.sensreg_dir}/dqdsst_audit.json")

if __name__ == "__main__":
    main()
