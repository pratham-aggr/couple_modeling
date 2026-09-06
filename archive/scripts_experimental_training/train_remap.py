"""
train_remap.py
==============
LEARNABLE REMAPPING for the direct MEMO<->POP gx1v7 coupling.

The production pipeline (model_server.py, --grid gx1v7, arch latlon) is

    POP SST (gx1v7) --ScatterToRegular--> MEMO 192x288 --frozen vfcg1m61 UNet-->
    8 fluxes 192x288 --RegularToScatter--> gx1v7 --> POP

with FIXED interpolators on both sides.  This script keeps the UNet FROZEN and
trains the two remaps end-to-end so that the flux error is minimised ON THE
GRID POP ACTUALLY CONSUMES:

    gx SST --EncoderRemap--> [SST, ICEFRAC, SOLIN, SST_prev, ICEFRAC_prev,
    dSST/dt] --frozen UNet--> y (8ch stored units) --DecoderRemap--> gx1v7
                                                       vs truth fluxes on gx1v7

Both learnable operators are initialised EXACTLY at the production fixed remap
(EncoderRemap = IDW k-NN, DecoderRemap = 4-pt bilinear + zero residual CNN),
so the epoch-0 validation numbers ARE the current production baseline, and any
improvement is attributable to the learned remapping.

Data: the two index-aligned caches (verified: gx zarr was built with the very
same bilinear operator; SOLIN roundtrip corr 1.000000, rmse ~1e-5):
    /glade/work/praggarwal/couple_cache_mem24h        X (N,6,192,288)  Y (N,5,...)
    /glade/work/praggarwal/couple_cache_gx1v7_mem24h  X (N,6,384,320)  Y+rad+precip
N = 50960 = 35 years x 1456 six-hourly samples (noleap, 4 memory steps trimmed
per year-zarr), so sample i belongs to year 1980 + i//1456.  Temporal split
matches vfcg1m61: train 1980-2010, val 2011-2012, test 2013-2014.

Inputs mirror deployment exactly: SST/SST_prev cross the grid through the
(learnable) encoder; ICEFRAC/SOLIN/ICEFRAC_prev are CAM-native (prescribed
forcing at deployment); dSST/dt is derived from the ENCODED fields.

Usage:
    python train_remap.py --out_dir output/output_remap_learnable \
        [--epochs 10] [--batch 16] [--lr 1e-3] [--no_dec_residual] [--amp]
    python train_remap.py --smoke          # tiny CPU shape/roundtrip test
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from train_unet import UNet  # import-safe (main under __name__ guard)
from learnable_remap import (build_learned_coupler, LearnedCoupler,
                             GX_NJ, GX_NI, CAM_NLAT, CAM_NLON,
                             GX1V7_DOMAIN, OUTPUT_VARS)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

CAM_CACHE = "/glade/work/praggarwal/couple_cache_mem24h"
GX_CACHE = "/glade/work/praggarwal/couple_cache_gx1v7_mem24h"
MEMO_DIR = "output/output_unet_mem24h_dsst_temporal_radprecip"   # vfcg1m61
SAMPLES_PER_YEAR = 1456
YEAR0 = 1980
DT_MEM = 86400.0                                                  # 24h memory


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class PairedRemapDataset(Dataset):
    """Index-aligned CAM-grid + gx1v7-grid sample pairs (mmap-backed).

    Returns per sample:
      sst_gx  (2, GX_NJ, GX_NI)  SST, SST_prev on gx1v7 [K]     (encoder input)
      cam_aux (3, CAM_NLAT, CAM_NLON)  ICEFRAC, SOLIN, ICEFRAC_prev (CAM-native)
      y_gx    (8, GX_NJ, GX_NI)  TAUX..PRECT truth on gx1v7 (stored zarr units)
    """

    def __init__(self, cam_cache, gx_cache, indices):
        self.cam_cache = Path(cam_cache)
        self.gx_cache = Path(gx_cache)
        self.indices = np.asarray(indices, dtype=np.int64)
        self._h = None  # lazy per-worker mmap handles

    def _handles(self):
        if self._h is None:
            self._h = dict(
                Xc=np.load(self.cam_cache / "X.npy", mmap_mode="r"),
                Xg=np.load(self.gx_cache / "X.npy", mmap_mode="r"),
                Yg=np.load(self.gx_cache / "Y.npy", mmap_mode="r"),
                Yr=np.load(self.gx_cache / "Y_rad.npy", mmap_mode="r"),
                Yp=np.load(self.gx_cache / "Y_precip.npy", mmap_mode="r"),
            )
        return self._h

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        h = self._handles()
        i = int(self.indices[k])
        # gx X channels: 0=SST, 3=SST_prev (K, remapped truth)
        sst_gx = np.ascontiguousarray(h["Xg"][i, [0, 3]], dtype=np.float32)
        # CAM X channels: 1=ICEFRAC, 2=SOLIN, 4=ICEFRAC_prev
        cam_aux = np.ascontiguousarray(h["Xc"][i, [1, 2, 4]], dtype=np.float32)
        y_gx = np.concatenate([np.ascontiguousarray(h["Yg"][i], dtype=np.float32),
                               np.ascontiguousarray(h["Yr"][i], dtype=np.float32),
                               np.ascontiguousarray(h["Yp"][i], dtype=np.float32)], axis=0)
        return (torch.from_numpy(sst_gx), torch.from_numpy(cam_aux),
                torch.from_numpy(y_gx))


def temporal_split(n_samples, val_years, test_years, rng, subsample=1.0):
    assert n_samples == 35 * SAMPLES_PER_YEAR, \
        f"expected 35yr x {SAMPLES_PER_YEAR} = {35*SAMPLES_PER_YEAR}, got {n_samples}"
    years = YEAR0 + np.arange(n_samples) // SAMPLES_PER_YEAR
    val = np.where((years >= val_years[0]) & (years <= val_years[1]))[0]
    test = np.where((years >= test_years[0]) & (years <= test_years[1]))[0]
    train = np.where(years < val_years[0])[0]
    if subsample < 1.0:
        train = rng.choice(train, size=int(len(train) * subsample), replace=False)
        train.sort()
    return train, val, test


def compute_gx_norm(gx_cache, train_idx, mask2d, out_path, n_probe=512, seed=7):
    """Per-channel mean/std of the 8 gx-grid targets over ocean cells (train only)."""
    if Path(out_path).exists():
        d = np.load(out_path)
        return d["y_mean"], d["y_std"]
    rng = np.random.default_rng(seed)
    probe = np.sort(rng.choice(train_idx, size=min(n_probe, len(train_idx)), replace=False))
    Yg = np.load(Path(gx_cache) / "Y.npy", mmap_mode="r")
    Yr = np.load(Path(gx_cache) / "Y_rad.npy", mmap_mode="r")
    Yp = np.load(Path(gx_cache) / "Y_precip.npy", mmap_mode="r")
    m = mask2d.astype(bool)
    s = np.zeros(8); ss = np.zeros(8); n = 0
    for i in probe:
        y = np.concatenate([np.asarray(Yg[i], dtype=np.float64),
                            np.asarray(Yr[i], dtype=np.float64),
                            np.asarray(Yp[i], dtype=np.float64)], axis=0)[:, m]
        s += y.sum(axis=1); ss += (y ** 2).sum(axis=1); n += y.shape[1]
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 1e-30))
    np.savez(out_path, y_mean=mean.astype(np.float32), y_std=std.astype(np.float32),
             probe_samples=probe)
    return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------------------
# Forward pipeline (mirrors model_server.py gx1v7 latlon path exactly)
# ---------------------------------------------------------------------------

def pipeline_forward(lc, unet, x_mean, x_std, y_mean, y_std, sst_gx, cam_aux):
    """gx SST + CAM-native aux -> decoded 8-channel fluxes on gx1v7 (stored units)."""
    sst_cam = lc.encoder(sst_gx)                       # (B,2,192,288) K
    dsst = (sst_cam[:, 0] - sst_cam[:, 1]) / DT_MEM
    x = torch.stack([sst_cam[:, 0], cam_aux[:, 0], cam_aux[:, 1],
                     sst_cam[:, 1], cam_aux[:, 2], dsst], dim=1)
    x_n = (x - x_mean) / (x_std + 1e-8)
    y_n = unet(x_n)                                    # frozen; grads flow to enc
    y = y_n * y_std + y_mean                           # stored units, CAM grid
    return lc.decoder(y)                               # (B,8,GX_NJ,GX_NI)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class R2Accumulator:
    """Streaming per-channel R^2 over ocean cells."""

    def __init__(self, n_ch, device):
        self.sse = torch.zeros(n_ch, dtype=torch.float64, device=device)
        self.s = torch.zeros(n_ch, dtype=torch.float64, device=device)
        self.ss = torch.zeros(n_ch, dtype=torch.float64, device=device)
        self.n = 0

    def update(self, pred, target, mask):
        # pred/target (B,C,H,W); mask (H,W) bool
        p = pred[:, :, mask].double()
        t = target[:, :, mask].double()
        self.sse += ((p - t) ** 2).sum(dim=(0, 2))
        self.s += t.sum(dim=(0, 2))
        self.ss += (t ** 2).sum(dim=(0, 2))
        self.n += p.shape[0] * p.shape[2]

    def r2(self):
        var = self.ss - self.s ** 2 / max(self.n, 1)
        return (1.0 - self.sse / torch.clamp(var, min=1e-30)).cpu().numpy()


@torch.no_grad()
def evaluate(lc, unet, loader, mask_t, norms, device, amp=False):
    lc.eval()
    acc = R2Accumulator(8, device)
    x_mean, x_std, y_mean, y_std = norms
    for sst_gx, cam_aux, y_gx in loader:
        sst_gx, cam_aux, y_gx = (t.to(device, non_blocking=True)
                                 for t in (sst_gx, cam_aux, y_gx))
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            pred = pipeline_forward(lc, unet, x_mean, x_std, y_mean, y_std,
                                    sst_gx, cam_aux)
        acc.update(pred.float(), y_gx, mask_t)
    return acc.r2()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cam_cache", default=CAM_CACHE)
    p.add_argument("--gx_cache", default=GX_CACHE)
    p.add_argument("--memo_dir", default=MEMO_DIR,
                   help="Frozen flux-UNet output dir (default vfcg1m61)")
    p.add_argument("--out_dir", default="output/output_remap_learnable")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_residual", type=float, default=None,
                   help="Separate LR for the decoder residual CNN (default = --lr)")
    p.add_argument("--k_enc", type=int, default=8,
                   help="Encoder k-NN neighbours. The production ScatterToRegular is "
                        "IDW k=4; neighbours 5..k init at ~zero weight so step 0 still "
                        "equals the fixed remap.")
    p.add_argument("--no_dec_residual", action="store_true",
                   help="Disable the decoder residual CNN (pure learnable weights)")
    p.add_argument("--hidden", type=int, default=48, help="Residual CNN width")
    p.add_argument("--depth", type=int, default=3, help="Residual CNN conv layers")
    p.add_argument("--sum1_w", type=float, default=1e-3,
                   help="Weight of the decoder sum-to-one penalty")
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--val_years", type=int, nargs=2, default=[2011, 2012])
    p.add_argument("--test_years", type=int, nargs=2, default=[2013, 2014])
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--domain", default=GX1V7_DOMAIN)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_name", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny CPU test: 64 train / 32 val samples, 4 steps, 1 epoch")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(42)
    print(f"device={device}  out_dir={out_dir}")

    # ---- frozen MEMO ----
    memo_dir = Path(args.memo_dir)
    mcfg = json.load(open(memo_dir / "model_config.json"))
    assert mcfg["n_in"] == 6 and mcfg["n_out"] == 8, \
        f"expected vfcg1m61-style 6-in/8-out model, got {mcfg}"
    unet = UNet(n_in=6, n_out=8, base=int(mcfg["base"]), dropout=0.0).to(device)
    sd = torch.load(memo_dir / "best_model.pt", map_location=device, weights_only=True)
    unet.load_state_dict(sd)
    unet.eval().requires_grad_(False)   # eval BN stats; grads still FLOW THROUGH
    nz = np.load(memo_dir / "normalizer.npz")
    x_mean = torch.from_numpy(nz["x_mean"].astype(np.float32))[None, :, None, None].to(device)
    x_std = torch.from_numpy(nz["x_std"].astype(np.float32))[None, :, None, None].to(device)
    y_mean = torch.from_numpy(nz["y_mean"].astype(np.float32))[None, :, None, None].to(device)
    y_std = torch.from_numpy(nz["y_std"].astype(np.float32))[None, :, None, None].to(device)
    norms = (x_mean, x_std, y_mean, y_std)
    print(f"frozen MEMO: {memo_dir}  (base={mcfg['base']}, in={mcfg['input_vars']})")

    # ---- split ----
    n_total = np.load(Path(args.gx_cache) / "X.npy", mmap_mode="r").shape[0]
    train_idx, val_idx, test_idx = temporal_split(
        n_total, args.val_years, args.test_years, rng, args.subsample)
    if args.smoke:
        train_idx = train_idx[:64]; val_idx = val_idx[:32]; test_idx = test_idx[:32]
        args.epochs, args.batch, args.num_workers = 1, 2, 0
    print(f"split: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # ---- gx target normalisation (for loss + residual CNN) ----
    from learnable_remap import load_gx1v7_domain
    gx_xc, gx_yc, gx_mask = load_gx1v7_domain(args.domain)
    gy_mean, gy_std = compute_gx_norm(args.gx_cache, train_idx, gx_mask,
                                      out_dir / "gx_norm.npz",
                                      n_probe=64 if args.smoke else 512)
    print("gx target std:", {v: f"{s:.3g}" for v, s in zip(OUTPUT_VARS, gy_std)})

    # ---- learnable coupler (init == production fixed remap) ----
    lc, _ = build_learned_coupler(args.domain, k_enc=args.k_enc,
                                  dec_residual=not args.no_dec_residual,
                                  y_mean=gy_mean, y_std=gy_std,
                                  hidden=args.hidden, depth=args.depth)
    lc = lc.to(device)
    n_par = sum(p.numel() for p in lc.parameters() if p.requires_grad)
    print(f"LearnedCoupler params: {n_par/1e6:.2f} M "
          f"(enc {lc.encoder.theta.numel()/1e6:.2f} M, "
          f"dec weights {lc.decoder.weight.numel()/1e6:.2f} M)")

    # ---- data ----
    def make_loader(idx, shuffle):
        return DataLoader(PairedRemapDataset(args.cam_cache, args.gx_cache, idx),
                          batch_size=args.batch, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                          persistent_workers=args.num_workers > 0, drop_last=shuffle)
    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)
    test_loader = make_loader(test_idx, False)

    mask_t = torch.from_numpy(gx_mask.astype(bool)).to(device)
    gy_std_t = torch.from_numpy(gy_std)[None, :, None, None].to(device)
    gy_mean_t = torch.from_numpy(gy_mean)[None, :, None, None].to(device)

    # ---- optimizer ----
    res_params = list(lc.decoder.residual.parameters()) if lc.decoder.residual else []
    table_params = [lc.encoder.theta, lc.decoder.weight]
    opt = torch.optim.Adam([
        {"params": table_params, "lr": args.lr},
        {"params": res_params, "lr": args.lr_residual or args.lr},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    use_wandb = WANDB_AVAILABLE and args.wandb_project is not None
    if use_wandb:
        wandb.init(project=args.wandb_project,
                   name=args.wandb_name or f"remap-k{args.k_enc}-lr{args.lr:.0e}"
                        f"{'-nores' if args.no_dec_residual else ''}",
                   config=vars(args))

    start_ep, best_val = 0, -np.inf
    ckpt_path = out_dir / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        lc.load_state_dict(ck["lc"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep, best_val = ck["epoch"] + 1, ck["best_val"]
        print(f"resumed at epoch {start_ep} (best_val={best_val:.4f})")

    # ---- epoch 0 = FIXED-REMAP BASELINE (init reproduces production) ----
    if start_ep == 0:
        r2b = evaluate(lc, unet, val_loader, mask_t, norms, device, args.amp)
        baseline = {v: float(r) for v, r in zip(OUTPUT_VARS, r2b)}
        json.dump(baseline, open(out_dir / "baseline_val_r2.json", "w"), indent=2)
        print("BASELINE (fixed remap) val R2:",
              " ".join(f"{v}={r:.4f}" for v, r in baseline.items()))
        if use_wandb:
            wandb.log({f"baseline/val_r2_{v}": r for v, r in baseline.items()}, step=0)
        best_val = float(np.mean(r2b))

    # ---- training ----
    def masked_loss(pred, target):
        pn = (pred - gy_mean_t) / gy_std_t
        tn = (target - gy_mean_t) / gy_std_t
        return ((pn - tn) ** 2)[:, :, mask_t].mean()

    steps_per_ep = 4 if args.smoke else len(train_loader)
    for ep in range(start_ep, args.epochs):
        lc.train(); unet.eval()
        t0, run_loss, nstep = time.time(), 0.0, 0
        for sst_gx, cam_aux, y_gx in train_loader:
            if nstep >= steps_per_ep:
                break
            sst_gx, cam_aux, y_gx = (t.to(device, non_blocking=True)
                                     for t in (sst_gx, cam_aux, y_gx))
            with torch.autocast("cuda", enabled=args.amp and device.type == "cuda",
                                dtype=torch.bfloat16):
                pred = pipeline_forward(lc, unet, *norms, sst_gx, cam_aux)
                loss = masked_loss(pred.float(), y_gx) \
                     + args.sum1_w * lc.decoder.sum1_penalty()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], 5.0)
            opt.step()
            run_loss += float(loss); nstep += 1
            if nstep % 200 == 0:
                print(f"  ep{ep} step {nstep}/{steps_per_ep} "
                      f"loss={run_loss/nstep:.4f} ({time.time()-t0:.0f}s)")
        sched.step()

        r2v = evaluate(lc, unet, val_loader, mask_t, norms, device, args.amp)
        mean_r2 = float(np.mean(r2v))
        print(f"epoch {ep}: train_loss={run_loss/max(nstep,1):.4f}  "
              f"val_R2 mean={mean_r2:.4f}  "
              + " ".join(f"{v}={r:.4f}" for v, r in zip(OUTPUT_VARS, r2v))
              + f"  ({time.time()-t0:.0f}s)")
        if use_wandb:
            wandb.log({"epoch": ep, "train_loss": run_loss / max(nstep, 1),
                       "val_r2_mean": mean_r2,
                       **{f"val_r2_{v}": float(r) for v, r in zip(OUTPUT_VARS, r2v)}})
        if mean_r2 > best_val:
            best_val = mean_r2
            lc.save(out_dir / "remap_best.pt")
            print(f"  -> new best (mean val R2 {best_val:.4f}); saved remap_best.pt")
        torch.save({"lc": lc.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best_val": best_val},
                   ckpt_path)

    # ---- test: baseline (fresh init) vs best learned ----
    print("\n=== TEST (2013-2014) ===")
    lc_base, _ = build_learned_coupler(args.domain, k_enc=args.k_enc,
                                       dec_residual=not args.no_dec_residual,
                                       y_mean=gy_mean, y_std=gy_std,
                                       hidden=args.hidden, depth=args.depth)
    lc_base = lc_base.to(device)
    r2_fixed = evaluate(lc_base, unet, test_loader, mask_t, norms, device, args.amp)
    best_path = out_dir / "remap_best.pt"
    if best_path.exists():
        lc_best = LearnedCoupler.load(best_path, device=device)
        r2_learn = evaluate(lc_best, unet, test_loader, mask_t, norms, device, args.amp)
    else:
        r2_learn = r2_fixed
    result = {"fixed_remap": {v: float(r) for v, r in zip(OUTPUT_VARS, r2_fixed)},
              "learned_remap": {v: float(r) for v, r in zip(OUTPUT_VARS, r2_learn)},
              "mean_fixed": float(np.mean(r2_fixed)),
              "mean_learned": float(np.mean(r2_learn))}
    json.dump(result, open(out_dir / "metrics_test.json", "w"), indent=2)
    print(json.dumps(result, indent=2))
    if use_wandb:
        wandb.log({f"test_r2_fixed_{v}": r for v, r in result["fixed_remap"].items()})
        wandb.log({f"test_r2_learned_{v}": r for v, r in result["learned_remap"].items()})
        wandb.finish()


if __name__ == "__main__":
    main()
