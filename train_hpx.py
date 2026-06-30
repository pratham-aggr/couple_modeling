"""
train_hpx.py
============
HEALPix-native MEMO flux emulator (DLWP-HPX style).  Mirrors train_unet.py's
recipe (MSE, cosine, base 64, temporal split, same 6->8 I/O) but runs on the
HEALPix mesh: the lat-lon CircPad UNet is replaced by a face-folded UNet with
cross-face HEALPix padding (location-invariant kernels).  Equal-area pixels =>
NO cos(lat) weighting and NO longitude-roll augmentation.

Reuses from train_unet.py: Normalizer, masked_mse (dimension-agnostic when
lat_w/var_w are None).

Usage (see scripts/submit_hpx64.pbs):
    python train_hpx.py --cache_dir /glade/work/praggarwal/couple_cache_hpx64 \
        --out_dir output/output_hpx64_mem24h --zarr_glob "..." \
        --dsst_dt --with_rad --with_precip --base 64 --batch 8 \
        --scheduler cosine --max_epochs 500 --patience 40
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import healpy as hp
from train_unet import Normalizer, masked_mse
from healpix_grid import face_reshape_index, build_pad_index

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

IN_NAMES  = ["SST", "ICEFRAC", "SOLIN", "SST_prev", "ICEFRAC_prev", "dSST_dt"]
TGT_BASE  = ["TAUX", "TAUY", "SHFLX", "LHFLX", "QFLX"]


# ---------------------------------------------------------------------------
# HEALPix face padding + UNet
# ---------------------------------------------------------------------------

def _fold(x):
    """(B,C,12,F,F) -> (B*12,C,F,F)."""
    B, C, nf, F1, F2 = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(B * nf, C, F1, F2)


def _unfold(x, B):
    """(B*12,C,F,F) -> (B,C,12,F,F)."""
    BF, C, F1, F2 = x.shape
    return x.reshape(B, BF // B, C, F1, F2).permute(0, 2, 1, 3, 4)


class HEALPixUNet(nn.Module):
    """5-level UNet on the 12-face HEALPix image with cross-face padding.

    Channel progression matches train_unet.py UNet (base=64):
      init 1x1: n_in -> b
      enc: b->2b->2b->4b->8b->8b   (stride-2; nside 64->32->16->8->4->2)
      dec: mirror with bilinear upsample + skip concat; ends nside 64
      head 1x1: 2b -> n_out
    """
    LEAK = 0.2

    def __init__(self, nside=64, n_in=6, n_out=8, base=64, dropout=0.0):
        super().__init__()
        self.nside = nside
        b = base

        # --- index buffers: flat(nested) <-> 12-face image ---
        img2pix = face_reshape_index(nside)             # (12,F,F) image-flat -> pixel
        flat2img = img2pix.reshape(-1)                  # len npix
        img2flat = np.argsort(flat2img)                 # pixel -> image-flat position
        self.register_buffer("flat2img", torch.as_tensor(flat2img, dtype=torch.long))
        self.register_buffer("img2flat", torch.as_tensor(img2flat, dtype=torch.long))

        # --- per-level padding gather indices (nside 64,32,16,8) ---
        self.pad_nsides = [nside, nside // 2, nside // 4, nside // 8]
        for ns in self.pad_nsides:
            pad_src = build_pad_index(ns, p=1).reshape(-1)   # (12*(F+2)^2,)
            self.register_buffer(f"pad_{ns}", torch.as_tensor(pad_src, dtype=torch.long))

        self.init_conv = nn.Conv2d(n_in, b, 1, bias=True)

        # encoder: (in, out, kernel, nside_in, use_pad)
        self.enc_cfg = [
            (b,   b*2, 4, nside,      True),
            (b*2, b*2, 4, nside//2,   True),
            (b*2, b*4, 4, nside//4,   True),
            (b*4, b*8, 4, nside//8,   True),
            (b*8, b*8, 2, nside//16,  False),   # nside 4 -> 2, no pad
        ]
        # encoder convs are stride-2 (downsample)
        self.enc = nn.ModuleList([nn.Conv2d(ci, co, k, stride=2, padding=0, bias=True)
                                  for ci, co, k, _, _ in self.enc_cfg])
        self.enc_bn = nn.ModuleList([nn.BatchNorm2d(co) for _, co, _, _, _ in self.enc_cfg])

        # decoder: (in, out, kernel, _, use_pad).  Upsample does the 2x; conv is
        # stride-1 with kernel (k-1), exactly like train_unet.py's decoder block.
        self.dec_cfg = [
            (b*8,        b*8, 2, 0, False),  # upsample nside2->4, 1x1 conv, no pad
            (b*8 + b*8,  b*4, 4, 0, True),   # ->8
            (b*4 + b*4,  b*2, 4, 0, True),   # ->16
            (b*2 + b*2,  b*2, 4, 0, True),   # ->32
            (b*2 + b*2,  b,   4, 0, True),   # ->64
        ]
        # upsample-target nside (= face size) for each dec block: 4,8,16,32,64
        self.dec_pad_ns = [nside//16, nside//8, nside//4, nside//2, nside]
        self.dec = nn.ModuleList([nn.Conv2d(ci, co, k - 1, stride=1, padding=0, bias=True)
                                  for ci, co, k, _, _ in self.dec_cfg])
        self.dec_bn = nn.ModuleList([nn.BatchNorm2d(co) for _, co, _, _, _ in self.dec_cfg])

        self.head = nn.Conv2d(b + b, n_out, 1, bias=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    # ---- flat(nested) <-> face image ----
    def to_img(self, x):
        """(B,C,npix) -> (B,C,12,F,F)."""
        B, C, _ = x.shape
        F1 = self.nside
        return x[..., self.flat2img].reshape(B, C, 12, F1, F1)

    def to_flat(self, x_img):
        """(B,C,12,F,F) -> (B,C,npix)."""
        B, C = x_img.shape[:2]
        return x_img.reshape(B, C, 12 * self.nside * self.nside)[..., self.img2flat]

    def hpx_pad(self, x_img, ns):
        """(B,C,12,F,F) -> (B,C,12,F+2,F+2) cross-face padding at resolution ns."""
        B, C, nf, Fn, _ = x_img.shape
        pad = getattr(self, f"pad_{ns}")
        flat = x_img.reshape(B, C, nf * Fn * Fn)
        out = flat[..., pad]                          # (B,C,12*(F+2)^2)
        return out.reshape(B, C, nf, Fn + 2, Fn + 2)

    def enc_block(self, x_img, i):
        ci, co, k, ns, use_pad = self.enc_cfg[i]
        B = x_img.shape[0]
        if use_pad:
            x_img = self.hpx_pad(x_img, ns)
        h = _fold(x_img)
        h = self.enc[i](h)
        h = self.enc_bn[i](h)
        h = F.leaky_relu(h, self.LEAK)
        h = self.drop(h)
        return _unfold(h, B)

    def dec_block(self, x_img, i):
        ci, co, k, _, use_pad = self.dec_cfg[i]
        ns = self.dec_pad_ns[i]
        B = x_img.shape[0]
        h = _fold(x_img)
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        x_img = _unfold(h, B)
        if use_pad:
            x_img = self.hpx_pad(x_img, ns)
        h = _fold(x_img)
        h = self.dec[i](h)
        h = self.dec_bn[i](h)
        h = F.relu(h)
        h = self.drop(h)
        return _unfold(h, B)

    def forward(self, x):
        x = self.to_img(x)                            # (B,n_in,12,F,F)
        x = _unfold(self.init_conv(_fold(x)), x.shape[0])
        r = x
        skips = []
        for i in range(len(self.enc)):
            x = self.enc_block(x, i)
            skips.append(x)
        x = skips.pop()
        for i in range(len(self.dec)):
            x = self.dec_block(x, i)
            if skips:
                x = torch.cat([x, skips.pop()], dim=1)
        x = torch.cat([x, r], dim=1)
        out = _unfold(self.head(_fold(x)), x.shape[0])
        return self.to_flat(out)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HPXDataset(Dataset):
    def __init__(self, idxs, X, Y, Yrad, Yprecip, mask, norm):
        self.idxs = idxs
        self.X, self.Y, self.Yrad, self.Yprecip, self.mask = X, Y, Yrad, Yprecip, mask
        self.norm = norm

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, k):
        i = self.idxs[k]
        xc = self.X[i][[0, 1, 2, 3, 4]].astype(np.float32)          # (5,npix)
        dsst = ((self.X[i][0] - self.X[i][3]) / 86400.0)[None].astype(np.float32)
        x = np.concatenate([xc, dsst], axis=0)                      # (6,npix)
        ys = [self.Y[i].astype(np.float32)]
        if self.Yrad is not None:
            ys.append(self.Yrad[i].astype(np.float32))
        if self.Yprecip is not None:
            ys.append(self.Yprecip[i].astype(np.float32))
        y = np.concatenate(ys, axis=0)                              # (8,npix)
        x = (x - self.norm.x_mean[:, None]) / (self.norm.x_std[:, None] + 1e-8)
        y = (y - self.norm.y_mean[:, None]) / (self.norm.y_std[:, None] + 1e-8)
        m = self.mask[i].astype(np.float32)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(np.ascontiguousarray(m)))


def compute_metrics_hpx(model, loader, norm, device, n_out):
    """Per-variable R^2, RMSE, Pearson corr over ocean pixels (HEALPix, 1D)."""
    model.eval()
    ys = torch.tensor(norm.y_std, device=device)
    ym = torch.tensor(norm.y_mean, device=device)
    pred_all, true_all = [], []
    with torch.no_grad():
        for x_n, y_n, mask in loader:
            pred = model(x_n.to(device)) * ys[None, :, None] + ym[None, :, None]
            true = y_n.to(device) * ys[None, :, None] + ym[None, :, None]
            m = (mask.to(device) > 0.5).reshape(-1)
            pred_all.append(pred.permute(0, 2, 1).reshape(-1, n_out)[m].cpu().numpy())
            true_all.append(true.permute(0, 2, 1).reshape(-1, n_out)[m].cpu().numpy())
    model.train()
    pred_all = np.concatenate(pred_all, 0)
    true_all = np.concatenate(true_all, 0)
    diff = true_all - pred_all
    ss_res = (diff ** 2).sum(0)
    ss_tot = ((true_all - true_all.mean(0)) ** 2).sum(0)
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)
    rmse = np.sqrt((diff ** 2).mean(0))
    tc = true_all - true_all.mean(0)
    pc = pred_all - pred_all.mean(0)
    corr = (tc * pc).sum(0) / (np.sqrt((tc**2).sum(0)) * np.sqrt((pc**2).sum(0)) + 1e-10)
    return {"r2": r2, "rmse": rmse, "corr": corr}


def _log_wandb_maps_hpx(model, val_ds, norm, device, tgt_vars, n_samples=3, step=None):
    """Log truth / predicted / error HEALPix Mollweide maps for a few val samples.
    Uses the eager `model` (not the compiled handle) to avoid a batch-size-1 recompile.
    Cache is NESTED ordering -> hp.mollview(nest=True)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(99)
    idxs = rng.choice(len(val_ds), size=min(n_samples, len(val_ds)), replace=False)
    ys = torch.tensor(norm.y_std); ym = torch.tensor(norm.y_mean)

    model.eval()
    # predict each sampled example once; reuse across all variables
    samples = []
    with torch.no_grad():
        for idx in idxs:
            x_n, y_n, mask = val_ds[idx]
            pred_n = model(x_n.unsqueeze(0).to(device)).squeeze(0).cpu()
            true = (y_n * ys[:, None] + ym[:, None]).numpy()
            pred = (pred_n * ys[:, None] + ym[:, None]).numpy()
            samples.append((true, pred, mask.numpy() > 0.5))

    log_dict = {}
    for vi, vname in enumerate(tgt_vars):
        fig = plt.figure(figsize=(12, 3 * n_samples))
        for di, (true_all, pred_all, m) in enumerate(samples):
            true = true_all[vi]
            pred = pred_all[vi]
            tr = np.where(m, true, np.nan)
            pr = np.where(m, pred, np.nan)
            df = pr - tr
            vmax = np.nanpercentile(np.abs(tr), 98) + 1e-12
            vmin = -vmax if np.nanmin(tr) < 0 else 0.0
            cmap = "RdBu_r" if vmin < 0 else "plasma"
            dmax = np.nanpercentile(np.abs(df), 98) + 1e-12
            hp.mollview(tr, nest=True, fig=fig.number, sub=(n_samples, 3, di * 3 + 1),
                        title=f"S{di+1} truth", min=vmin, max=vmax, cmap=cmap, cbar=True)
            hp.mollview(pr, nest=True, fig=fig.number, sub=(n_samples, 3, di * 3 + 2),
                        title=f"S{di+1} pred", min=vmin, max=vmax, cmap=cmap, cbar=True)
            hp.mollview(df, nest=True, fig=fig.number, sub=(n_samples, 3, di * 3 + 3),
                        title="error", min=-dmax, max=dmax, cmap="RdBu_r", cbar=True)
        fig.suptitle(vname, fontsize=12)
        log_dict[f"maps/val/{vname}"] = wandb.Image(fig)
        plt.close(fig)
    wandb.log(log_dict, step=step)
    model.train()


def build_year_labels(zarr_glob, mem_lag_steps, n_full):
    years_list = []
    for zp in sorted(glob.glob(zarr_glob)):
        ds = __import__("xarray").open_zarr(zp, consolidated=False)
        times = ds["time"].values
        ds.close()
        valid = times[mem_lag_steps:]
        years_list.extend([int(valid[0].year)] * len(valid))
    years = np.array(years_list, dtype=np.int32)
    assert len(years) == n_full, f"{len(years)} != {n_full}"
    return years


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="/glade/work/praggarwal/couple_cache_hpx64")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--zarr_glob", required=True)
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--max_epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--dsst_dt", action="store_true")
    ap.add_argument("--with_rad", action="store_true")
    ap.add_argument("--with_precip", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_compile", action="store_true",
                    help="Disable torch.compile (compile gives ~9x speedup on A100).")
    ap.add_argument("--wandb_project", type=str, default=None,
                    help="W&B project name (omit to disable wandb logging).")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_name", type=str, default=None)
    ap.add_argument("--val_years", type=int, nargs=2, default=[2011, 2012])
    ap.add_argument("--test_years", type=int, nargs=2, default=[2013, 2014])
    ap.add_argument("--max_steps_dryrun", type=int, default=0,
                    help="If >0, run only this many train steps/epoch (wiring test).")
    ap.add_argument("--resume_from", type=str, default=None,
                    help="Path to a saved state_dict to warm-start from (weights only).")
    ap.add_argument("--start_epoch", type=int, default=0,
                    help="Continue epoch counter / cosine schedule from here (for resume).")
    ap.add_argument("--wandb_id", type=str, default=None,
                    help="Resume an existing W&B run by id (resume='must').")
    ap.add_argument("--resume_best_val", type=float, default=float("inf"),
                    help="Seed best_val so a worse warm-start epoch doesn't clobber best_model.pt.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = json.load(open(Path(args.cache_dir) / "meta.json"))
    nside = meta["nside"]; npix = meta["npix"]; mem_lag = meta["memory_lag_steps"]
    assert nside == args.nside, f"cache nside {nside} != --nside {args.nside}"

    X = np.load(Path(args.cache_dir) / "X.npy", mmap_mode="r")
    Y = np.load(Path(args.cache_dir) / "Y.npy", mmap_mode="r")
    Yrad = np.load(Path(args.cache_dir) / "Y_rad.npy", mmap_mode="r") if args.with_rad else None
    Yprecip = np.load(Path(args.cache_dir) / "Y_precip.npy", mmap_mode="r") if args.with_precip else None
    mask = np.load(Path(args.cache_dir) / "mask.npy", mmap_mode="r")
    N = len(X)

    n_out = Y.shape[1] + (Yrad.shape[1] if Yrad is not None else 0) \
            + (Yprecip.shape[1] if Yprecip is not None else 0)
    tgt_vars = list(TGT_BASE) + (["FSDS_J", "FLDS_J"] if args.with_rad else []) \
               + (["PRECT"] if args.with_precip else [])
    n_in = 6
    print(f"HEALPix nside={nside} npix={npix}  N={N}  n_in={n_in} n_out={n_out}")
    print(f"targets: {tgt_vars}")

    # temporal split
    years = build_year_labels(args.zarr_glob, mem_lag, N)
    vs, ve = args.val_years; ts, te = args.test_years
    val_m = (years >= vs) & (years <= ve)
    test_m = (years >= ts) & (years <= te)
    train_m = ~val_m & ~test_m
    ti = np.where(train_m)[0]; vi = np.where(val_m)[0]; tei = np.where(test_m)[0]
    print(f"train={len(ti)} val={len(vi)} test={len(tei)}")
    np.save(out_dir / "test_indices.npy", tei)

    norm = Normalizer.load(out_dir / "normalizer.npz")

    mk = lambda idxs, shuf: DataLoader(
        HPXDataset(idxs, X, Y, Yrad, Yprecip, mask, norm),
        batch_size=args.batch, shuffle=shuf, num_workers=args.num_workers,
        pin_memory=True, drop_last=shuf)
    train_loader = mk(ti, True); val_loader = mk(vi, False)
    test_loader = mk(tei, False) if len(tei) else None
    val_ds = HPXDataset(vi, X, Y, Yrad, Yprecip, mask, norm)  # for map logging

    model = HEALPixUNet(nside=nside, n_in=n_in, n_out=n_out,
                        base=args.base, dropout=args.dropout).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model params: {nparam/1e6:.2f}M")

    # torch.compile fuses the gather-pad + small-conv chain -> ~9x faster/step on A100.
    # The compiled handle shares parameters with `model`, so we still save/load
    # `model.state_dict()` (no _orig_mod. prefix) and only use `fwd` for forward passes.
    if args.resume_from:
        model.load_state_dict(torch.load(args.resume_from, map_location=device))
        print(f"resumed weights from {args.resume_from} (warm-start, optimizer reset)",
              flush=True)

    if device == "cuda" and not args.no_compile:
        fwd = torch.compile(model)
        print("torch.compile enabled", flush=True)
    else:
        fwd = model

    json.dump({"nside": nside, "n_in": n_in, "n_out": n_out, "base": args.base,
               "dropout": args.dropout, "input_vars": IN_NAMES,
               "output_vars": tgt_vars}, open(out_dir / "model_config.json", "w"), indent=2)

    use_wandb = WANDB_AVAILABLE and args.wandb_project is not None
    if use_wandb:
        run_name = args.wandb_name or (
            f"hpx{nside}-base{args.base}-bs{args.batch}-lr{args.lr:.0e}"
            f"-do{args.dropout}-mem24h-dsst-radprecip")
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, name=run_name,
            id=args.wandb_id, resume=("must" if args.wandb_id else None),
            config={"grid": "healpix", "nside": nside, "npix": npix,
                    "n_in": n_in, "n_out": n_out, "base": args.base,
                    "batch": args.batch, "lr": args.lr, "dropout": args.dropout,
                    "weight_decay": args.weight_decay, "max_epochs": args.max_epochs,
                    "patience": args.patience, "params_M": round(nparam / 1e6, 2),
                    "compile": (device == "cuda" and not args.no_compile),
                    "input_vars": IN_NAMES, "target_vars": tgt_vars,
                    "train_n": int(len(ti)), "val_n": int(len(vi)), "test_n": int(len(tei))})
        wandb.define_metric("val/loss", summary="min")
        wandb.define_metric("train/loss", summary="min")
        for v in tgt_vars:
            wandb.define_metric(f"test/r2_{v}", summary="max")
        (out_dir / "wandb_run_id.txt").write_text(wandb.run.id)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.max_epochs, eta_min=args.lr * 1e-3)
    # Fast-forward the cosine schedule so LR continues from where the prior run stopped.
    for _ in range(args.start_epoch):
        sched.step()
    if args.start_epoch:
        print(f"resuming at epoch {args.start_epoch}, lr {sched.get_last_lr()[0]:.2e}",
              flush=True)

    best_val = args.resume_best_val; no_improve = 0; t0 = time.time()
    for epoch in range(args.start_epoch, args.max_epochs):
        model.train()
        tr_loss = 0.0; nb = 0
        for x_n, y_n, m in train_loader:
            x_n, y_n, m = x_n.to(device), y_n.to(device), m.to(device)
            opt.zero_grad()
            pred = fwd(x_n)
            loss = masked_mse(pred, y_n, m)
            loss.backward(); opt.step()
            tr_loss += loss.item(); nb += 1
            if args.max_steps_dryrun and nb >= args.max_steps_dryrun:
                break
        sched.step()

        # validation
        model.eval(); val_loss = 0.0; vnb = 0
        with torch.no_grad():
            for x_n, y_n, m in val_loader:
                x_n, y_n, m = x_n.to(device), y_n.to(device), m.to(device)
                val_loss += masked_mse(fwd(x_n), y_n, m).item(); vnb += 1
                if args.max_steps_dryrun and vnb >= args.max_steps_dryrun:
                    break
        model.train()
        val_loss /= max(vnb, 1)
        print(f"epoch {epoch:3d}  train {tr_loss/max(nb,1):.4f}  val {val_loss:.4f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s", flush=True)
        if use_wandb:
            wandb.log({"train/loss": tr_loss / max(nb, 1), "val/loss": val_loss,
                       "lr": sched.get_last_lr()[0], "epoch": epoch}, step=epoch)
            if epoch % 10 == 0:
                _log_wandb_maps_hpx(model, val_ds, norm, device, tgt_vars,
                                    n_samples=3, step=epoch)

        if val_loss < best_val:
            best_val = val_loss; no_improve = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            no_improve += 1
            if args.patience and no_improve >= args.patience:
                print(f"early stop at epoch {epoch}"); break

        if args.max_steps_dryrun and epoch >= 1:
            print("dry-run complete"); break

    # final eval on test
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    if test_loader is not None:
        mt = compute_metrics_hpx(fwd, test_loader, norm, device, n_out)
        out = {v: {"r2": float(mt["r2"][i]), "rmse": float(mt["rmse"][i]),
                   "corr": float(mt["corr"][i])} for i, v in enumerate(tgt_vars)}
        json.dump(out, open(out_dir / "metrics_test.json", "w"), indent=2)
        json.dump({v: float(mt["r2"][i]) for i, v in enumerate(tgt_vars)},
                  open(out_dir / "r2_scores.json", "w"), indent=2)
        print("TEST metrics:")
        for v in tgt_vars:
            print(f"  {v:8s} R2={out[v]['r2']:.4f} RMSE={out[v]['rmse']:.4e} corr={out[v]['corr']:.4f}")
        if use_wandb:
            wandb.log({f"test/r2_{v}": out[v]["r2"] for v in tgt_vars}
                      | {f"test/corr_{v}": out[v]["corr"] for v in tgt_vars}
                      | {"val/best_loss": best_val})

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
