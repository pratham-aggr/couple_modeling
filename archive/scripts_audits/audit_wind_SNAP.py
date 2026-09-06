import json
from pathlib import Path
import numpy as np
import torch, sys, xarray as xr
sys.path.insert(0, "camulator_ud/climate")
from train_unet import UNet

CACHE = "/glade/work/praggarwal/couple_cache_gx1v7_mem24h"
MODEL_DIR = "output/output_unet_gx1v7_atm_aux"
REF_POP_FILE = ("/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_sensreg50yr/"
                 "run/g.e21.MEMO_GIAF_v01.pop.h.1980-01.nc")
N_SAMPLES = 300

device = "cuda" if torch.cuda.is_available() else "cpu"
ref = xr.open_dataset(REF_POP_FILE)
TLAT = ref.TLAT.values; TLONG = ref.TLONG.values

X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
Y = np.load(f"{CACHE}/Y.npy", mmap_mode="r")   # TAUX,TAUY,SHFLX,LHFLX,QFLX
mask = np.load(f"{CACHE}/mask.npy", mmap_mode="r")
test_idx = np.load(f"{MODEL_DIR}/test_indices.npy")
sel = np.sort(np.random.default_rng(0).choice(test_idx, size=min(N_SAMPLES, len(test_idx)), replace=False))
print(f"device={device} n={len(sel)}")

d = Path(MODEL_DIR); cfg = json.loads((d / "model_config.json").read_text())
model = UNet(n_in=cfg["n_in"], n_out=cfg["n_out"], base=cfg.get("base", 64), dropout=0.0).to(device)
model.load_state_dict(torch.load(d / "best_model_snap.pt", map_location=device, weights_only=True))
model.eval()
nz = np.load(f"{MODEL_DIR}/normalizer.npz")
xm = nz["x_mean"][None,:,None,None]; xs = nz["x_std"][None,:,None,None]
ym = nz["y_mean"][None,:,None,None]; ys = nz["y_std"][None,:,None,None]

m_ok = mask[sel] > 0.5
raw = X[sel].astype(np.float64)
dsst = ((raw[:, 0] - raw[:, 3]) / 86400.0)[:, None]
x_phys = np.concatenate([raw[:, 0:5], dsst], axis=1)

pred = np.empty((len(sel), 2, x_phys.shape[2], x_phys.shape[3]), np.float64)
bs = 8
with torch.no_grad():
    for s in range(0, len(sel), bs):
        xn = (x_phys[s:s+bs] - xm) / (xs + 1e-8)
        yn = model(torch.from_numpy(xn.astype(np.float32)).to(device)).cpu().numpy().astype(np.float64)
        y = yn * ys + ym
        pred[s:s+bs, 0] = y[:, 0]   # TAUX (N/m2 already, no /DT6H for stress)
        pred[s:s+bs, 1] = y[:, 1]   # TAUY

truth = np.empty_like(pred)
truth[:, 0] = Y[sel][:, 0].astype(np.float64)
truth[:, 1] = Y[sel][:, 1].astype(np.float64)

bias_taux = pred[:,0] - truth[:,0]
bias_tauy = pred[:,1] - truth[:,1]
finite = np.isfinite(bias_taux) & m_ok
lat_b = np.broadcast_to(TLAT[None], m_ok.shape)
lon_b = np.broadcast_to(TLONG[None], m_ok.shape)

eq_pac = (lat_b>=-5)&(lat_b<=5)&(lon_b>=190)&(lon_b<=240) & finite
eq_all = (lat_b>=-5)&(lat_b<=5) & finite

for name, bm in [("Equatorial Pacific (Nino3.4 box)", eq_pac), ("Equatorial band (all lon)", eq_all)]:
    n = int(bm.sum())
    print(f"{name}: n={n}")
    print(f"  TAUX  pred_mean={pred[:,0][bm].mean():+.4f}  truth_mean={truth[:,0][bm].mean():+.4f}  bias={bias_taux[bm].mean():+.4f} N/m2")
    print(f"  TAUY  pred_mean={pred[:,1][bm].mean():+.4f}  truth_mean={truth[:,1][bm].mean():+.4f}  bias={bias_tauy[bm].mean():+.4f} N/m2")
