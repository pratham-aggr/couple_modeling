"""
audit_dqdsst.py  — offline dQnet/dSST feedback audit (coupled-deploy gate).

CRPS Stage-2 fine-tuning dropped the SST-sensitivity regularizer, so before the
_crps checkpoint can be deployed in a coupled run we must confirm the PHYSICAL
damping the v15 sensreg model bought (dQnet/dSST ~ -30..-45 W/m2/K turbulent-
inclusive) SURVIVED, i.e. did NOT regress toward the old +7 W/m2/K correlation
defect (which was FLDS-driven -> radiation MUST be included, not just sen+lat).

Method: build the exact 11-channel input from the mem24h cache, apply a SUSTAINED
SST perturbation (shift SST and SST_prev by +-dS equally so dSST_dt stays 0 -- the
slow/equilibrium channel that governs drift), run the UNet, and finite-difference
the net downward heat flux Qnet = SHFLX + LHFLX + FSDS + FLDS (W/m2) over ocean.

Reports, per model, dQnet/dSST split by component, plus:
  - the bulk-physics turbulent target d(sen+lat)/dSST on the same samples,
  - the EFFECTIVE coupled feedback = UNet dQnet/dSST - 4*eps*sigma*SST^3 (the
    -sigma*SST^4 LWUP that POP adds itself), which is what actually damps drift.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, "camulator_ud/climate")   # bulk_flux lives beside model_server
from train_unet import UNet
import bulk_flux

CACHE = "/glade/work/praggarwal/couple_cache_gx1v7_mem24h"
HYBM_BOT = 0.992556
SIGMA = 5.67e-8
EPS = 0.98
DT6H = 21600.0

# y channel indices (TAUX,TAUY,SHFLX,LHFLX,QFLX,FSDS_J,FLDS_J,PRECT)
I_SH, I_LH, I_FSDS, I_FLDS = 2, 3, 5, 6


def load_model(out_dir, ckpt, device):
    d = Path(out_dir)
    cfg = json.loads((d / "model_config.json").read_text())
    m = UNet(n_in=cfg["n_in"], n_out=cfg["n_out"],
             base=cfg.get("base", 64), dropout=0.0).to(device)   # deterministic audit
    sd = torch.load(d / ckpt, map_location=device, weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    return m, cfg


def build_inputs(idxs, X, Xa):
    """Assemble the 11-channel PHYSICAL input for the given cache rows, mirroring
    CacheMmapDataset.__getitem__ (mem_ch=[0..4] + dSST_dt + 5 atm)."""
    raw = X[idxs].astype(np.float64)          # (N,6,H,W)
    a   = Xa[idxs].astype(np.float64)         # (N,5,H,W)
    dsst = ((raw[:, 0] - raw[:, 3]) / 86400.0)[:, None]
    x = np.concatenate([raw[:, 0:5], dsst, a], axis=1)   # (N,11,H,W)
    return x, raw, a


def qnet_wm2(model, x_phys, xm, xs, ym, ys, device, bs=8):
    """Run model on physical 11-ch input -> Qnet (SH+LH+FSDS+FLDS) in W/m2."""
    N = x_phys.shape[0]
    out = np.empty((N, x_phys.shape[2], x_phys.shape[3]), np.float64)
    with torch.no_grad():
        for s in range(0, N, bs):
            xb = x_phys[s:s+bs]
            xn = (xb - xm) / (xs + 1e-8)
            yn = model(torch.from_numpy(xn.astype(np.float32)).to(device)).cpu().numpy().astype(np.float64)
            y = yn * ys + ym                                   # physical, (b,8,H,W)
            q = (y[:, I_SH] + y[:, I_LH] + y[:, I_FSDS] + y[:, I_FLDS]) / DT6H
            out[s:s+bs] = q
    return out


def component_wm2(model, x_phys, xm, xs, ym, ys, device, ch, bs=8):
    N = x_phys.shape[0]
    out = np.empty((N, x_phys.shape[2], x_phys.shape[3]), np.float64)
    with torch.no_grad():
        for s in range(0, N, bs):
            xn = (x_phys[s:s+bs] - xm) / (xs + 1e-8)
            yn = model(torch.from_numpy(xn.astype(np.float32)).to(device)).cpu().numpy().astype(np.float64)
            y = yn * ys + ym
            out[s:s+bs] = y[:, ch] / DT6H
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=160, help="test samples to audit")
    ap.add_argument("--dS", type=float, default=0.5, help="SST perturbation (K)")
    ap.add_argument("--crps_dir", default="output/output_unet_gx1v7_sensreg_crps")
    ap.add_argument("--sensreg_dir", default="output/output_unet_gx1v7_sensreg")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  n={args.n}  dS={args.dS} K  (SUSTAINED: SST & SST_prev shifted equally)")

    X  = np.load(f"{CACHE}/X.npy",     mmap_mode="r")
    Xa = np.load(f"{CACHE}/X_atm.npy", mmap_mode="r")
    mask = np.load(f"{CACHE}/mask.npy", mmap_mode="r")

    test_idx = np.load(f"{args.crps_dir}/test_indices.npy")
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(test_idx, size=min(args.n, len(test_idx)), replace=False))
    print(f"selected {len(sel)} of {len(test_idx)} test rows")

    x0, raw, a = build_inputs(sel, X, Xa)
    m = (mask[sel] > 0.5)                       # (N,H,W) ocean/ice
    sst_phys = raw[:, 0]                        # (N,H,W) K

    # sustained perturbation: shift channels 0 (SST) and 3 (SST_prev) by +-dS;
    # dSST_dt (ch 5) is unchanged because both shift equally.
    xp = x0.copy(); xp[:, 0] += args.dS; xp[:, 3] += args.dS
    xm_ = x0.copy(); xm_[:, 0] -= args.dS; xm_[:, 3] -= args.dS

    def masked_mean(field):
        return float(field[m].mean())

    results = {}
    for name, out_dir in [("sensreg (production)", args.sensreg_dir),
                          ("crps (candidate)",     args.crps_dir)]:
        model, cfg = load_model(out_dir, "best_model.pt", device)
        nz = np.load(f"{out_dir}/normalizer.npz")
        xm = nz["x_mean"][None, :, None, None]; xs = nz["x_std"][None, :, None, None]
        ym = nz["y_mean"][None, :, None, None]; ys = nz["y_std"][None, :, None, None]

        qp = qnet_wm2(model, xp,  xm, xs, ym, ys, device)
        qn = qnet_wm2(model, xm_, xm, xs, ym, ys, device)
        dq = (qp - qn) / (2 * args.dS)                     # W/m2/K, per cell
        comp = {}
        for lbl, ch in [("SH", I_SH), ("LH", I_LH), ("FSDS", I_FSDS), ("FLDS", I_FLDS)]:
            cp = component_wm2(model, xp,  xm, xs, ym, ys, device, ch)
            cn = component_wm2(model, xm_, xm, xs, ym, ys, device, ch)
            comp[lbl] = masked_mean((cp - cn) / (2 * args.dS))
        dqm = masked_mean(dq)
        lwup = -4 * EPS * SIGMA * (sst_phys ** 3)
        eff = masked_mean(dq + lwup)
        results[name] = dict(total=dqm, effective=eff, **comp)
        print(f"\n=== {name} ===  ({out_dir})")
        print(f"  dQnet/dSST (UNet, W/m2/K):  {dqm:+.2f}")
        print(f"    by component:  dSH {comp['SH']:+.2f}  dLH {comp['LH']:+.2f}  "
              f"dFSDS {comp['FSDS']:+.2f}  dFLDS {comp['FLDS']:+.2f}")
        print(f"  + POP LWUP (-4 eps sig T^3): {masked_mean(lwup):+.2f}")
        print(f"  => EFFECTIVE coupled dQnet/dSST: {eff:+.2f} W/m2/K")

    # bulk-physics turbulent target on the same samples (held-fixed BL atm)
    dturb = np.empty_like(sst_phys)
    for k in range(len(sel)):
        dturb[k] = bulk_flux_sens(sst_phys[k], a[k], args.dS)
    tgt = float(dturb[m].mean())
    print(f"\n=== bulk-physics reference (shr_flux port) ===")
    print(f"  d(sen+lat)/dSST target:  {tgt:+.2f} W/m2/K  (radiation dR/dSST=0 physically)")

    print("\n---------------- VERDICT ----------------")
    s = results["sensreg (production)"]; c = results["crps (candidate)"]
    print(f"  sensreg total {s['total']:+.2f} | crps total {c['total']:+.2f} | bulk target {tgt:+.2f}")
    print(f"  sensreg eff   {s['effective']:+.2f} | crps eff   {c['effective']:+.2f}")
    verdict = "PASS" if c['total'] < 0 and c['effective'] < 0 and abs(c['total'] - s['total']) < 15 \
              else "REVIEW"
    print(f"  GATE: {verdict} — crps damping is {'negative & close to sensreg' if verdict=='PASS' else 'NOT confirmed; inspect'}")
    json.dump({"sensreg": s, "crps": c, "bulk_target": tgt, "dS": args.dS,
               "n": len(sel), "verdict": verdict},
              open(f"{args.crps_dir}/dqdsst_audit.json", "w"), indent=2)
    print(f"\nwrote {args.crps_dir}/dqdsst_audit.json")


def bulk_flux_sens(Ts, a, dS):
    """d(sen+lat)/dSST over ocean cells for one sample via the shr_flux port."""
    from train_unet import bulk_sens_target
    d = bulk_sens_target(Ts.astype(np.float64), a[0], a[1], a[2], a[3], a[4],
                         dS, HYBM_BOT)
    return d.astype(np.float64)


if __name__ == "__main__":
    main()
