#!/usr/bin/env python3
"""Geographic-zone R^2 comparison: radprecip (no oro) vs radprecip+oro.

The cache mask is all-ones (the model is scored over the WHOLE globe, land
included), so we define zones from a real geographic land mask (LANDFRAC) and
orography (z_norm) in the statics file, and report per-variable R^2 over:
  - open ocean   (LANDFRAC<0.5, >K cells from land)
  - coastal ocean(LANDFRAC<0.5, within K cells of land)
  - all land     (LANDFRAC>0.5)
  - high terrain (land & z_norm>1, i.e. >1 std elevation)

If orography helps "near coasts/terrain", the (oro - no_oro) R^2 gain should be
concentrated in coastal / land / high-terrain zones, not open ocean.
Evaluated on the held-out temporal TEST set (2013-2014).
"""
import os
import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_unet as T

CACHE = Path("/glade/work/praggarwal/couple_cache_mem24h")
OUT   = Path("/glade/u/home/praggarwal/couple/output")
STATF = "/glade/derecho/scratch/wchapman/b_credit_runs/statics_b_credit_runs_f32_02.nc"
RUNS  = {
    "no_oro": OUT / "output_unet_mem24h_dsst_temporal_radprecip",
    "oro":    OUT / "output_unet_mem24h_dsst_temporal_radprecip_oro",
}
VARS  = ["TAUX", "TAUY", "SHFLX", "LHFLX", "QFLX", "FSDS_J", "FLDS_J", "PRECT"]
MEM_CH = [0, 1, 2, 3, 4]          # no prev_solin
K = 3                              # coastal band width (grid cells)
BATCH = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dilate(mask, k):
    m = mask.copy()
    for _ in range(k):
        up = np.roll(m, 1, 0);  up[0] = m[0]
        dn = np.roll(m, -1, 0); dn[-1] = m[-1]
        m = m | up | dn | np.roll(m, 1, 1) | np.roll(m, -1, 1)
    return m


# --- cache (mmap) ---
X_np  = np.load(CACHE / "X.npy",        mmap_mode="r")
Y_np  = np.load(CACHE / "Y.npy",        mmap_mode="r")
YR_np = np.load(CACHE / "Y_rad.npy",    mmap_mode="r")
YP_np = np.load(CACHE / "Y_precip.npy", mmap_mode="r")
H, W  = X_np.shape[2], X_np.shape[3]
statics = T.load_statics(STATF, ["z_norm", "LANDM_COSLAT"], H, W)   # for the oro model input

# --- geographic zone masks (static, H x W) ---
import xarray as xr
sds = xr.open_dataset(STATF)
landfrac = sds["LANDFRAC"].values.astype(np.float32)
znorm    = sds["z_norm"].values.astype(np.float32)
sds.close()
land   = landfrac > 0.5
ocean  = ~land
coast  = ocean & dilate(land, K)
openoc = ocean & ~coast
hiterr = land & (znorm > 1.0)
ZONES = {"open_ocean": openoc, "coast_ocean": coast, "land_all": land, "land_high": hiterr}
print("Zone point counts:", {k: int(v.sum()) for k, v in ZONES.items()})

test_idx = np.load(RUNS["oro"] / "test_indices.npy")
print(f"Test samples: {len(test_idx)}  grid {H}x{W}  coastal band K={K} cells")


def build_x(i, use_oro):
    x = X_np[i][MEM_CH]
    dsst = ((X_np[i][0] - X_np[i][3]) / 86400.0)[None]
    x = np.concatenate([x, dsst], axis=0)
    if use_oro:
        x = np.concatenate([x, statics], axis=0)
    return x


def load_model(run_dir):
    cfg  = json.load(open(run_dir / "model_config.json"))
    norm = T.Normalizer.load(run_dir / "normalizer.npz")
    m = T.UNet(n_in=cfg["n_in"], n_out=cfg["n_out"], base=cfg["base"]).to(device)
    m.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    m.eval()
    return m, norm, cfg["input_vars"]


def new_acc():
    return {z: {"n": np.zeros(8), "sy": np.zeros(8),
                "syy": np.zeros(8), "sres": np.zeros(8)} for z in ZONES}


def update(acc, pred, true):
    # pred,true: (B,8,H,W); zone masks are (H,W) -> broadcast over batch
    for z, zm in ZONES.items():
        for v in range(8):
            p = pred[:, v][:, zm]      # (B, Nz)
            t = true[:, v][:, zm]
            acc[z]["n"][v]    += t.size
            acc[z]["sy"][v]   += t.sum()
            acc[z]["syy"][v]  += (t * t).sum()
            acc[z]["sres"][v] += ((t - p) ** 2).sum()


def r2(acc):
    out = {}
    for z, a in acc.items():
        n, sy, syy, sres = a["n"], a["sy"], a["syy"], a["sres"]
        sstot = syy - sy * sy / np.maximum(n, 1)
        out[z] = 1.0 - sres / np.maximum(sstot, 1e-12)
    return out


results = {}
for name, run_dir in RUNS.items():
    model, norm, ivars = load_model(run_dir)
    use_oro = "z_norm" in ivars
    xm = torch.tensor(norm.x_mean, device=device).view(1, -1, 1, 1)
    xs = torch.tensor(norm.x_std,  device=device).view(1, -1, 1, 1)
    ym = torch.tensor(norm.y_mean, device=device).view(1, -1, 1, 1)
    ys = torch.tensor(norm.y_std,  device=device).view(1, -1, 1, 1)
    acc = new_acc()
    with torch.no_grad():
        for b in range(0, len(test_idx), BATCH):
            idx = test_idx[b:b + BATCH]
            xb = np.stack([build_x(i, use_oro) for i in idx])
            yb = np.stack([np.concatenate([Y_np[i], YR_np[i], YP_np[i]], 0) for i in idx])
            xt = (torch.tensor(xb, device=device) - xm) / xs
            pred = (model(xt) * ys + ym).cpu().numpy()
            update(acc, pred, yb)
    results[name] = r2(acc)
    print(f"  {name:7s}: loaded {run_dir.name}  (oro={use_oro})")

# --- report ---
zlist = list(ZONES)
print("\n" + "=" * 96)
print("Per-variable R^2  (Δ = oro − no_oro)")
hdr = f"{'var':<8s}"
for z in zlist:
    hdr += f" | {z:^22s}"
print(hdr)
sub = f"{'':<8s}"
for z in zlist:
    sub += f" | {'no_oro':>6s} {'oro':>6s} {'Δ':>7s}"
print(sub)
print("-" * 96)
for v, vn in enumerate(VARS):
    row = f"{vn:<8s}"
    for z in zlist:
        a, b = results['no_oro'][z][v], results['oro'][z][v]
        row += f" | {a:6.3f} {b:6.3f} {b-a:+7.4f}"
    print(row)
print("=" * 96)

# mean Δ per zone (avg over the 8 vars and over TAUX/TAUY only)
print("\nMean R^2 gain (oro − no_oro) by zone:")
for z in zlist:
    d = results['oro'][z] - results['no_oro'][z]
    print(f"  {z:<12s}  all-vars Δ={d.mean():+.4f}   TAUX/TAUY Δ={d[:2].mean():+.4f}")
