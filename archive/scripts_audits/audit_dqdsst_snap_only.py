"""dQ/dSST feedback audit for the ACTUAL checkpoint driving the live coupled
run (best_model_snap.pt), not best_model.pt. Base-only, no sensreg comparison."""
import json
from pathlib import Path
import numpy as np
import torch, sys
sys.path.insert(0, "camulator_ud/climate")
from train_unet import UNet, bulk_sens_target

CACHE = "/glade/work/praggarwal/couple_cache_gx1v7_mem24h"
MODEL_DIR = "output/output_unet_gx1v7_atm_aux"
HYBM_BOT = 0.992556; SIGMA = 5.67e-8; EPS = 0.98; DT6H = 21600.0
I_SH, I_LH, I_FSDS, I_FLDS = 2, 3, 5, 6
N = 96; dS = 0.5

device = "cuda" if torch.cuda.is_available() else "cpu"
d = Path(MODEL_DIR); cfg = json.loads((d / "model_config.json").read_text())
model = UNet(n_in=cfg["n_in"], n_out=cfg["n_out"], base=cfg.get("base", 64), dropout=0.0).to(device)
model.load_state_dict(torch.load(d / "best_model_snap.pt", map_location=device, weights_only=True))
model.eval()
nz = np.load(f"{MODEL_DIR}/normalizer.npz")
xm = nz["x_mean"][None,:,None,None]; xs = nz["x_std"][None,:,None,None]
ym = nz["y_mean"][None,:,None,None]; ys = nz["y_std"][None,:,None,None]

X  = np.load(f"{CACHE}/X.npy",     mmap_mode="r")
Xa = np.load(f"{CACHE}/X_atm.npy", mmap_mode="r")
mask = np.load(f"{CACHE}/mask.npy", mmap_mode="r")
test_idx = np.load(f"{MODEL_DIR}/test_indices.npy")
sel = np.sort(np.random.default_rng(0).choice(test_idx, size=min(N, len(test_idx)), replace=False))
print(f"device={device} n={len(sel)} dS={dS}  checkpoint=best_model_snap.pt")

raw = X[sel].astype(np.float64); a = Xa[sel].astype(np.float64)
dsst = ((raw[:, 0] - raw[:, 3]) / 86400.0)[:, None]
x0 = np.concatenate([raw[:, 0:5], dsst], axis=1)
m = (mask[sel] > 0.5); sst_phys = raw[:, 0]
xp = x0.copy(); xp[:, 0] += dS; xp[:, 3] += dS
xn_ = x0.copy(); xn_[:, 0] -= dS; xn_[:, 3] -= dS
mm = lambda f: float(f[m].mean())

def fwd(x_phys, chans, bs=8):
    Nn = x_phys.shape[0]
    out = np.zeros((Nn, x_phys.shape[2], x_phys.shape[3]), np.float64)
    with torch.no_grad():
        for s in range(0, Nn, bs):
            xn = (x_phys[s:s+bs] - xm) / (xs + 1e-8)
            yn = model(torch.from_numpy(xn.astype(np.float32)).to(device)).cpu().numpy().astype(np.float64)
            y = yn * ys + ym
            out[s:s+bs] = sum(y[:, c] for c in chans) / DT6H
    return out

qp = fwd(xp, [I_SH,I_LH,I_FSDS,I_FLDS]); qn = fwd(xn_, [I_SH,I_LH,I_FSDS,I_FLDS])
dq = mm((qp - qn) / (2*dS))
comp = {}
for lbl, ch in [("SH",I_SH),("LH",I_LH),("FSDS",I_FSDS),("FLDS",I_FLDS)]:
    cp = fwd(xp, [ch]); cn = fwd(xn_, [ch])
    comp[lbl] = mm((cp - cn) / (2*dS))
lwup = mm(-4*EPS*SIGMA*(sst_phys**3))
eff = dq + lwup

print(f"\n=== best_model_snap.pt (ACTUAL running checkpoint) ===")
print(f"  dQnet/dSST (UNet): {dq:+.2f} W/m2/K")
print(f"    dSH {comp['SH']:+.2f}  dLH {comp['LH']:+.2f}  dFSDS {comp['FSDS']:+.2f}  dFLDS {comp['FLDS']:+.2f}")
print(f"  + POP LWUP {lwup:+.2f}  =>  EFFECTIVE coupled dQnet/dSST: {eff:+.2f} W/m2/K")

dturb = np.empty_like(sst_phys)
for k in range(len(sel)):
    dturb[k] = bulk_sens_target(sst_phys[k], a[k,0],a[k,1],a[k,2],a[k,3],a[k,4], dS, HYBM_BOT)
tgt = mm(dturb)
print(f"\nbulk-physics reference (shr_flux port): d(sen+lat)/dSST target: {tgt:+.2f} W/m2/K")
print(f"\nGATE: sign should be NEGATIVE (restoring) for stability; POSITIVE = runaway-prone")
print(f"  RESULT: {'POSITIVE (runaway-prone)' if eff > 0 else 'NEGATIVE (restoring)'}")
