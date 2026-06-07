"""
train_unet.py
=============
UNet emulator: predicts surface turbulent fluxes from ocean state.

Inputs  (time t, daily mean): SST, ICEFRAC, SOLIN
Outputs (time t+1, daily mean): TAUX, TAUY, SHFLX, LHFLX, QFLX

By default trains on 24-hour daily means (group 4 × 6h steps into one day),
then predicts the next day's fluxes. Use --sixhour for raw 6-hourly pairs.

The UNet uses circular padding along longitude (wraparound) and replicate
padding along latitude, so it respects the global spherical geometry.

Usage:
    python train_unet.py \\
        --zarr_glob "/glade/derecho/scratch/wchapman/b_credit_runs/*.zarr" \\
        --out_dir ./output_unet \\
        [--subsample 0.2] [--epochs 30] [--batch 8] [--base 64] [--sixhour]
"""

import argparse
import glob
import json
import time as time_module
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

TARGET_VARS = ["TAUX", "TAUY", "SHFLX", "LHFLX", "QFLX"]
INPUT_VARS  = ["SST", "ICEFRAC", "SOLIN"]
H, W = 192, 288


# ---------------------------------------------------------------------------
# Muon optimizer (Momentum + Newton-Schulz orthogonalization)
# Applied to 2D+ weight tensors; use AdamW for biases / norm params.
# Reference: Keller Jordan, github.com/KellerJordan/Muon
# ---------------------------------------------------------------------------

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _zeropower_via_newtonschulz5(G, steps=5):
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G.float()
        shape = X.shape
        X = X.view(shape[0], -1)
        transposed = X.shape[0] > X.shape[1]
        if transposed:
            X = X.T
        X = X / (X.norm() + 1e-7)
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if transposed:
            X = X.T
        return X.view(shape).to(G.dtype)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mom, nesterov = group["lr"], group["momentum"], group["nesterov"]
            ns_steps, wd = group["ns_steps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                if wd != 0.0:
                    g = g + wd * p.data.float()
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.mul_(mom).add_(g)
                u = g + mom * buf if nesterov else buf.clone()
                if u.ndim >= 2:
                    u = self._zeropower_via_newtonschulz5(u, steps=ns_steps)
                    scale = max(1, u.shape[0] / u.shape[1]) ** 0.5
                    u = u * scale
                p.data.add_(u.to(p.dtype), alpha=-lr)
        return loss


# ---------------------------------------------------------------------------
# Normalizer — channel-wise mean/std over the full spatial field
# ---------------------------------------------------------------------------

class Normalizer:
    def __init__(self, x_mean, x_std, y_mean, y_std):
        self.x_mean = x_mean.astype(np.float32)   # (n_in,)
        self.x_std  = x_std.astype(np.float32)
        self.y_mean = y_mean.astype(np.float32)   # (n_out,)
        self.y_std  = y_std.astype(np.float32)

    def save(self, path):
        np.savez(path, x_mean=self.x_mean, x_std=self.x_std,
                 y_mean=self.y_mean, y_std=self.y_std)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["x_mean"], d["x_std"], d["y_mean"], d["y_std"])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """
    Each sample: X (n_in, H, W), Y (n_out, H, W) — normalised float32.
    Also returns ocean_mask (H, W) bool for masked loss.
    """

    def __init__(self, X_list, Y_list, mask_list, norm: Normalizer,
                 augment: bool = False):
        self.X       = X_list      # list of (n_in,  H, W) float32 arrays
        self.Y       = Y_list      # list of (n_out, H, W) float32 arrays
        self.mask    = mask_list   # list of (H, W) bool arrays — True = ocean/ice
        self.norm    = norm
        self.augment = augment     # random longitudinal (W) roll — train only

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        m = self.mask[idx]
        x_n = (x - self.norm.x_mean[:, None, None]) / (self.norm.x_std[:, None, None] + 1e-8)
        y_n = (y - self.norm.y_mean[:, None, None]) / (self.norm.y_std[:, None, None] + 1e-8)
        if self.augment:
            # Circular shift along longitude (last axis). The grid is lon-periodic
            # (see CircPad), so this is a physically exact, leakage-free augmentation.
            # x_n/y_n share the same shift to stay aligned; mask too.
            s = int(np.random.randint(0, x_n.shape[-1]))
            if s:
                x_n = np.roll(x_n, shift=s, axis=-1)
                y_n = np.roll(y_n, shift=s, axis=-1)
                m   = np.roll(m,   shift=s, axis=-1)
        return (torch.from_numpy(np.ascontiguousarray(x_n)),
                torch.from_numpy(np.ascontiguousarray(y_n)),
                torch.from_numpy(np.ascontiguousarray(m)))


# ---------------------------------------------------------------------------
# UNet adapted from climatebench/src/models/unet_simple.py
#
# Key changes vs the original:
#   - Standalone (no LightningModule / hydra / omegaconf dependencies)
#   - 5 encoder + 5 decoder levels (original has 6; 192×288 is divisible by
#     2^5=32 but not 2^6=64, so 5 levels avoids padding)
#   - Circular padding along longitude before every conv, replicate along lat
#   - Weight init kept from original: N(0,0.02) for conv, N(1,0.02) for BN
# ---------------------------------------------------------------------------

class CircPad(nn.Module):
    """Circular pad along lon (last dim), replicate along lat (second-to-last)."""
    def __init__(self, p=1):
        super().__init__()
        self.p = p

    def forward(self, x):
        x = F.pad(x, (self.p, self.p, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.p, self.p), mode="replicate")
        return x


class UNetBlock(nn.Module):
    """
    Single encoder or decoder block from climatebench unet_simple.py.

    Encoder (transposed=False): CircPad → Conv2d(stride=2) → BN → LeakyReLU
    Decoder (transposed=True):  Bilinear upsample → CircPad → Conv2d → BN → ReLU
    """
    LEAK = 0.2

    def __init__(self, in_ch, out_ch, transposed=False, bn=True, relu=True,
                 kernel=4, pad=1, dropout=0.0):
        super().__init__()
        if not transposed:
            # Encoder: stride-2 conv with circular longitude padding
            self.ops = nn.Sequential(
                CircPad(pad),
                nn.Conv2d(in_ch, out_ch, kernel, stride=2, padding=0, bias=True),
                nn.BatchNorm2d(out_ch) if bn else nn.GroupNorm(8, out_ch),
            )
        else:
            # Decoder: bilinear upsample then conv (avoids checkerboard artifacts)
            self.ops = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                CircPad(pad),
                nn.Conv2d(in_ch, out_ch, kernel - 1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(out_ch) if bn else nn.GroupNorm(8, out_ch),
            )
        self.act  = nn.ReLU() if relu else nn.LeakyReLU(self.LEAK)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.ops(x)))


class UNet(nn.Module):
    """
    5-level UNet for (B, n_in, 192, 288) inputs.

    Architecture follows climatebench/src/models/unet_simple.py:
      - LeakyReLU in encoder, ReLU in decoder
      - Skip connections (concatenation) at every level
      - Weight init: N(0,0.02) for conv, N(1,0.02) for BN bias/weight
      - 5 levels (192 and 288 are both divisible by 32 = 2^5)

    Channel progression (base=64):
      Encoder: n_in → 64 → 128 → 128 → 256 → 512
      Bottleneck: 512 → 512
      Decoder:  1024→256 → 512→128 → 256→128 → 256→64 → 128→n_out
    """
    def __init__(self, n_in=3, n_out=5, base=64, dropout=0.0):
        super().__init__()
        b = base

        self.init_conv = nn.Conv2d(n_in, b, 1, bias=True)

        # Encoder (LeakyReLU, BN)
        self.enc = nn.ModuleList([
            UNetBlock(b,    b*2,  transposed=False, bn=True, relu=False, kernel=4, pad=1, dropout=dropout),
            UNetBlock(b*2,  b*2,  transposed=False, bn=True, relu=False, kernel=4, pad=1, dropout=dropout),
            UNetBlock(b*2,  b*4,  transposed=False, bn=True, relu=False, kernel=4, pad=1, dropout=dropout),
            UNetBlock(b*4,  b*8,  transposed=False, bn=True, relu=False, kernel=4, pad=1, dropout=dropout),
            UNetBlock(b*8,  b*8,  transposed=False, bn=True, relu=False, kernel=2, pad=0, dropout=dropout),
        ])

        # Decoder (ReLU, BN); input channels doubled by skip concatenation
        self.dec = nn.ModuleList([
            UNetBlock(b*8,       b*8, transposed=True,  bn=True, relu=True, kernel=2, pad=0, dropout=dropout),
            UNetBlock(b*8 + b*8, b*4, transposed=True,  bn=True, relu=True, kernel=4, pad=1, dropout=dropout),
            UNetBlock(b*4 + b*4, b*2, transposed=True,  bn=True, relu=True, kernel=4, pad=1, dropout=dropout),
            UNetBlock(b*2 + b*2, b*2, transposed=True,  bn=True, relu=True, kernel=4, pad=1, dropout=dropout),
            UNetBlock(b*2 + b*2, b,   transposed=True,  bn=True, relu=True, kernel=4, pad=1, dropout=dropout),
        ])

        self.head = nn.Conv2d(b + b, n_out, 1, bias=True)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            m.weight.data.normal_(0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        x = self.init_conv(x)
        r = x                         # residual for final skip (from climatebench)

        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)

        x = skips.pop()               # bottleneck output is first decoder input
        for block in self.dec:
            x = block(x)
            if skips:
                x = torch.cat([x, skips.pop()], dim=1)

        x = torch.cat([x, r], dim=1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Data loading — one zarr (one year) at a time to keep peak RAM bounded
# ---------------------------------------------------------------------------

def load_year(zarr_path: str, tgt_vars: list, daily: bool,
              subsample: float, rng: np.random.Generator,
              lag_steps: int = 1):
    """
    Returns lists of (X, Y, ocean_mask) arrays for one year.
    X shape: (3, H, W), Y shape: (n_out, H, W), mask shape: (H, W).
    lag_steps: how many time-steps ahead Y is relative to X (0 = simultaneous).
    """
    ds = xr.open_zarr(zarr_path, consolidated=False)
    T  = len(ds["time"])

    # Load all needed variables at once (one contiguous read per variable)
    sst_raw = ds["SST"].values.astype(np.float32)        # (T, H, W), NaN over land
    ifrac   = ds["ICEFRAC"].values.astype(np.float32)
    solin   = ds["SOLIN"].values.astype(np.float32)
    tgts    = {v: ds[v].values.astype(np.float32) for v in tgt_vars}
    ds.close()

    # Replace NaN with 0 so we can take means without nan-propagation
    def fill(a): return np.where(np.isfinite(a), a, 0.0)

    sst_f   = fill(sst_raw)
    ifrac_f = np.clip(fill(ifrac), 0.0, 1.0)
    solin_f = fill(solin)
    tgts_f  = {v: fill(tgts[v]) for v in tgt_vars}

    if daily:
        n_days = T // 4
        def dmean(a):
            return a[:n_days*4].reshape(n_days, 4, H, W).mean(axis=1)

        sst_t   = dmean(sst_f)
        ifrac_t = dmean(ifrac_f)
        solin_t = dmean(solin_f)
        tgts_t  = {v: dmean(tgts_f[v]) for v in tgt_vars}
        # Ocean mask: SST was finite (non-land) for majority of the 4 timesteps
        ocean_t = dmean(np.isfinite(sst_raw).astype(np.float32)) > 0.5  # (n_days, H, W)
        n_steps = n_days
    else:
        sst_t   = sst_f
        ifrac_t = ifrac_f
        solin_t = solin_f
        tgts_t  = tgts_f
        ocean_t = np.isfinite(sst_raw)
        n_steps = T

    # n_pairs: valid X indices are 0 .. n_steps-lag_steps-1 (or all if lag=0)
    n_pairs = n_steps - lag_steps if lag_steps > 0 else n_steps
    chosen = rng.choice(n_pairs, size=max(1, int(n_pairs * subsample)), replace=False)
    chosen.sort()

    X_list, Y_list, mask_list = [], [], []
    for pi in chosen:
        X = np.stack([sst_t[pi], ifrac_t[pi], solin_t[pi]], axis=0)            # (3, H, W)
        Y = np.stack([tgts_t[v][pi + lag_steps] for v in tgt_vars], axis=0)    # (n_out, H, W)
        mask = ocean_t[pi].astype(np.float32)                                   # (H, W)
        X_list.append(X)
        Y_list.append(Y)
        mask_list.append(mask)

    return X_list, Y_list, mask_list


# ---------------------------------------------------------------------------
# Normalizer computation from loaded samples
# ---------------------------------------------------------------------------

def compute_norm(X_list, Y_list) -> Normalizer:
    X_all = np.stack(X_list)   # (N, n_in,  H, W)
    Y_all = np.stack(Y_list)   # (N, n_out, H, W)

    def stats(arr):
        ax = (0, 2, 3)
        return arr.mean(axis=ax).astype(np.float32), arr.std(axis=ax).astype(np.float32)

    xm, xs = stats(X_all)
    ym, ys = stats(Y_all)
    return Normalizer(xm, xs, ym, ys)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def masked_mse(pred, target, mask, lat_w=None, var_w=None):
    """MSE over ocean/ice points only.
    lat_w: optional (1,1,H,1) cosine-latitude weights (--lat_weight).
    var_w: optional (n_out,) per-variable weights (--loss_weights).
    """
    m = mask.unsqueeze(1)
    if lat_w is not None:
        m = m * lat_w
    err = (pred - target) ** 2
    if var_w is not None:
        err = err * var_w[None, :, None, None]
    return (err * m).sum() / (m.sum() * pred.shape[1] + 1e-8)


def masked_mae(pred, target, mask, lat_w=None, var_w=None):
    """MAE over ocean/ice points only. Stage 1 deterministic loss (U-Cast §3.2)."""
    m = mask.unsqueeze(1)
    if lat_w is not None:
        m = m * lat_w
    err = torch.abs(pred - target)
    if var_w is not None:
        err = err * var_w[None, :, None, None]
    return (err * m).sum() / (m.sum() * pred.shape[1] + 1e-8)


def masked_crps(preds, target, mask, lat_w=None, var_w=None):
    """Unbiased CRPS estimator (Zamo & Naveau 2018) over ocean/ice points.

    preds: list of M (B, C, H, W) stochastic ensemble members via MC Dropout.
    CRPS = Skill - 0.5 * Spread  (U-Cast Eq. 3-5).
    """
    M = len(preds)
    m = mask.unsqueeze(1)
    if lat_w is not None:
        m = m * lat_w
    skill  = sum(torch.abs(p - target) for p in preds) / M
    spread = sum(torch.abs(preds[i] - preds[j])
                 for i in range(M) for j in range(M) if i != j) / (M * (M - 1))
    crps = skill - 0.5 * spread
    if var_w is not None:
        crps = crps * var_w[None, :, None, None]
    return (crps * m).sum() / (m.sum() * target.shape[1] + 1e-8)


def _load_months(zarr_paths, chosen, is_memory, mem_lag_steps):
    """Return int8 month (1-12) for each index in `chosen`, aligned with X_all/Y_all."""
    months_list = []
    for zp in zarr_paths:
        ds = xr.open_zarr(zp, consolidated=False)
        times = ds["time"].values  # cftime.DatetimeNoLeap — not pandas-compatible
        ds.close()
        valid_times = times[mem_lag_steps:] if is_memory else times
        months_list.extend([int(t.month) for t in valid_times])
    months_cache = np.array(months_list, dtype=np.int8)
    return months_cache[chosen]  # aligned with X_all / Y_all indexing


def _load_years(zarr_paths, chosen, is_memory, mem_lag_steps):
    """Return int32 year for each index in `chosen`, aligned with X_all/Y_all.
    All samples from a given zarr file share the same year, so we just repeat
    the first valid timestamp's year for each file."""
    years_list = []
    for zp in zarr_paths:
        ds = xr.open_zarr(zp, consolidated=False)
        times = ds["time"].values
        ds.close()
        valid_times = times[mem_lag_steps:] if is_memory else times
        yr = int(valid_times[0].year)
        years_list.extend([yr] * len(valid_times))
    years_cache = np.array(years_list, dtype=np.int32)
    return years_cache[chosen]


def compute_r2_with_clim(model, norm, device, n_out,
                          X_vi, Y_raw_vi, mask_vi, months_vi, clim):
    """R² on absolute (un-anomalied) fluxes for --anomaly experiments.
    Model outputs a normalised anomaly; we unnormalise and add back the
    monthly climatology before comparing to the raw target."""
    model.eval()
    ys    = torch.tensor(norm.y_std,  device=device)
    ym    = torch.tensor(norm.y_mean, device=device)
    clim_t = torch.from_numpy(clim).to(device)

    pred_all, true_all = [], []
    B = 8
    with torch.no_grad():
        for s in range(0, len(X_vi), B):
            e      = min(s + B, len(X_vi))
            x_b    = np.stack(X_vi[s:e])
            y_b    = np.stack(Y_raw_vi[s:e])
            m_b    = np.stack(mask_vi[s:e])
            mo_b   = months_vi[s:e]
            x_n    = (x_b - norm.x_mean[:,None,None]) / (norm.x_std[:,None,None] + 1e-8)
            pred_n = model(torch.from_numpy(x_n).to(device))
            pred   = pred_n * ys[None,:,None,None] + ym[None,:,None,None]
            for b, mo in enumerate(mo_b):
                pred[b] += clim_t[int(mo) - 1]
            pred_np   = pred.cpu().numpy()
            mf        = (m_b > 0.5).reshape(-1)
            pred_all.append(pred_np.transpose(0,2,3,1).reshape(-1, n_out)[mf])
            true_all.append(y_b.transpose(0,2,3,1).reshape(-1, n_out)[mf])

    model.train()
    pred_all = np.concatenate(pred_all)
    true_all = np.concatenate(true_all)
    ss_res = ((true_all - pred_all) ** 2).sum(0)
    ss_tot = ((true_all - true_all.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / (ss_tot + 1e-10)


def compute_r2(model, loader, norm, device, n_out):
    model.eval()
    norm_y_std  = torch.tensor(norm.y_std,  device=device)
    norm_y_mean = torch.tensor(norm.y_mean, device=device)

    pred_all, true_all = [], []
    with torch.no_grad():
        for x_n, y_n, mask in loader:
            pred_n = model(x_n.to(device))
            pred = pred_n * norm_y_std[None,:,None,None] + norm_y_mean[None,:,None,None]
            true = y_n.to(device) * norm_y_std[None,:,None,None] + norm_y_mean[None,:,None,None]
            m = mask.to(device) > 0.5   # (B, H, W)

            # Permute to (B, H, W, n_out) then flatten spatial so indexing with
            # (B*H*W,) mask gives (N_ocean, n_out) — one row per ocean point.
            pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, n_out)  # (B*H*W, n_out)
            true_flat = true.permute(0, 2, 3, 1).reshape(-1, n_out)
            m_flat    = m.reshape(-1)                                  # (B*H*W,)

            pred_all.append(pred_flat[m_flat].cpu().numpy())
            true_all.append(true_flat[m_flat].cpu().numpy())

    pred_all = np.concatenate(pred_all, axis=0)   # (N_ocean, n_out)
    true_all = np.concatenate(true_all, axis=0)

    ss_res = ((true_all - pred_all) ** 2).sum(axis=0)
    ss_tot = ((true_all - true_all.mean(axis=0)) ** 2).sum(axis=0)
    model.train()
    return 1.0 - ss_res / (ss_tot + 1e-10)


# ---------------------------------------------------------------------------
# Wandb map logging  (follows climatebench's create_wandb_figures pattern)
# ---------------------------------------------------------------------------

def _log_wandb_maps(model, val_ds, norm, device, tgt_vars, n_samples=4, step=None):
    """Log truth / predicted / error spatial maps for n_samples val examples."""
    rng_local = np.random.default_rng(99)
    idxs = rng_local.choice(len(val_ds), size=min(n_samples, len(val_ds)), replace=False)

    y_std  = torch.tensor(norm.y_std)
    y_mean = torch.tensor(norm.y_mean)

    model.eval()
    log_dict = {}

    with torch.no_grad():
        for vi, vname in enumerate(tgt_vars):
            fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples),
                                     constrained_layout=True)
            if n_samples == 1:
                axes = axes[None, :]

            for di, idx in enumerate(idxs):
                x_n, y_n, mask = val_ds[idx]
                pred_n = model(x_n.unsqueeze(0).to(device)).squeeze(0).cpu()

                true = (y_n    * y_std[:, None, None] + y_mean[:, None, None]).numpy()
                pred = (pred_n * y_std[:, None, None] + y_mean[:, None, None]).numpy()
                m = mask.numpy() > 0.5

                tr = np.where(m, true[vi], np.nan)
                pr = np.where(m, pred[vi], np.nan)
                df = pr - tr

                vmax = np.nanpercentile(np.abs(tr), 98)
                vmin = -vmax if np.nanmin(tr) < 0 else 0
                cmap = "RdBu_r" if vmin < 0 else "plasma"
                dmax = np.nanpercentile(np.abs(df), 98) + 1e-10

                im0 = axes[di, 0].imshow(tr, origin="lower", aspect="auto",
                                         cmap=cmap, vmin=vmin, vmax=vmax)
                im1 = axes[di, 1].imshow(pr, origin="lower", aspect="auto",
                                         cmap=cmap, vmin=vmin, vmax=vmax)
                im2 = axes[di, 2].imshow(df, origin="lower", aspect="auto",
                                         cmap="RdBu_r", vmin=-dmax, vmax=dmax)

                for ax, im, title in zip(axes[di], [im0, im1, im2],
                                         ["Truth", "Predicted", "Error (pred−truth)"]):
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title(f"Sample {di+1} — {title}", fontsize=8)
                    ax.set_xticks([]); ax.set_yticks([])

            fig.suptitle(vname, fontsize=12)
            log_dict[f"maps/val/{vname}"] = wandb.Image(fig)
            plt.close(fig)

    wandb.log(log_dict, step=step)
    model.train()


def _log_rmse_maps(model, val_loader, norm, device, tgt_vars, step=None):
    """Log per-pixel RMSE maps (ocean only) for each variable."""
    n_out  = len(tgt_vars)
    y_std_gpu  = torch.tensor(norm.y_std,  device=device)
    y_mean_gpu = torch.tensor(norm.y_mean, device=device)
    y_std_cpu  = torch.tensor(norm.y_std)
    y_mean_cpu = torch.tensor(norm.y_mean)

    sse   = np.zeros((n_out, H, W), dtype=np.float64)
    count = np.zeros((H, W),        dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for x_n, y_n, mask in val_loader:
            pred_n = model(x_n.to(device))
            pred = (pred_n * y_std_gpu[None,:,None,None] + y_mean_gpu[None,:,None,None]).cpu().numpy()
            true = (y_n    * y_std_cpu[None,:,None,None] + y_mean_cpu[None,:,None,None]).numpy()
            m = mask.numpy() > 0.5
            err2 = (pred - true) ** 2
            for b in range(pred.shape[0]):
                sse[:, m[b]] += err2[b][:, m[b]]
                count[m[b]]  += 1

    rmse = np.where(count[None] > 0, np.sqrt(sse / np.maximum(count[None], 1)), np.nan)

    log_dict = {}
    for vi, vname in enumerate(tgt_vars):
        data = rmse[vi]
        vmax = np.nanpercentile(data, 95)
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        im = ax.imshow(data, origin="lower", aspect="auto", cmap="plasma", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"RMSE — {vname}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        log_dict[f"rmse_maps/val/{vname}"] = wandb.Image(fig)
        plt.close(fig)

    wandb.log(log_dict, step=step)
    model.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_glob", required=True,
                        help="Glob pattern for zarr stores (e.g. '/path/*.zarr')")
    parser.add_argument("--out_dir",   default="./output_unet")
    parser.add_argument("--subsample", type=float, default=0.2,
                        help="Fraction of daily pairs to use per year (default 0.2)")
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--batch",     type=int,   default=8)
    parser.add_argument("--base",      type=int,   default=64,
                        help="UNet base channel width (default 64; use 32 for lighter model)")
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--sixhour",   action="store_true",
                        help="Use raw 6-hourly pairs instead of daily means")
    parser.add_argument("--lag",       type=int, default=24,
                        help="Prediction lag in hours (default 24). "
                             "Use 0 for simultaneous, 12 for 12-hour ahead. "
                             "Values other than 24 force 6-hourly mode and bypass the cache.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Path to precomputed data cache (from preprocess_data.py). "
                             "If set and cache exists, skips zarr loading entirely.")
    parser.add_argument("--with_co2", action="store_true",
                        help="Add co2vmr as an extra input channel (broadcast globally). "
                             "Requires co2.npy in the cache dir.")
    parser.add_argument("--memory", action="store_true",
                        help="Memory experiment: X = [state_t, state_{t-Nh}] → Y = fluxes_t. "
                             "Uses couple_cache_mem{N}h (6 channels stored; training selects subset).")
    parser.add_argument("--memory_lag", type=int, default=24,
                        help="Memory lag in hours (default 24). Only used with --memory.")
    parser.add_argument("--prev_solin", action="store_true",
                        help="Include SOLIN from prior timestep as an input channel (only with --memory).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint.pt in out_dir")
    parser.add_argument("--extra_epochs", type=int, default=0,
                        help="Extra epochs to train beyond --epochs when resuming "
                             "(uses flat LR = lr * 0.01 to avoid cosine restart)")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early-stopping patience in epochs (0 = disabled). "
                             "Training stops when val loss has not improved for this many epochs.")
    parser.add_argument("--max_epochs", type=int, default=0,
                        help="Hard cap on epochs when using early stopping (0 = use --epochs).")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name (omit to disable wandb logging)")
    parser.add_argument("--wandb_entity",  type=str, default=None,
                        help="W&B entity (team/username)")
    parser.add_argument("--wandb_name",    type=str, default=None,
                        help="W&B run name (auto-generated if omitted)")
    parser.add_argument("--anomaly", action="store_true",
                        help="Train on flux anomalies (Y - monthly climatology) instead of "
                             "raw fluxes. Climatology is computed from training samples only. "
                             "R² is reported on absolute fluxes (anomaly + clim vs raw Y).")
    parser.add_argument("--scheduler", choices=["sgdr", "cosine"], default="sgdr",
                        help="LR scheduler: sgdr (CosineAnnealingWarmRestarts T_0=30, default) "
                             "or cosine (single cosine decay over all epochs, no restarts).")
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw",
                        help="Optimizer: adamw (default) or muon (Muon applies to 2D+ weight "
                             "tensors; AdamW handles biases and 1D params). "
                             "U-Cast uses Muon + cosine decay.")
    parser.add_argument("--loss", choices=["mse", "mae", "crps"], default="mse",
                        help="Loss function: mae (stage 1, default), crps (stage 2 "
                             "probabilistic fine-tuning), or mse (legacy). "
                             "crps uses MC Dropout with --crps_members forward passes.")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="MC Dropout rate applied to all UNet blocks (default 0). "
                             "Set to 0.1 when using --loss crps for stochastic ensemble.")
    parser.add_argument("--crps_members", type=int, default=2,
                        help="Number of MC Dropout ensemble members for CRPS loss (default 2).")
    parser.add_argument("--lat_weight", action="store_true",
                        help="Weight loss by cos(lat): de-emphasises polar regions. "
                             "Precomputed as (1,1,H,1) tensor before the training loop.")
    parser.add_argument("--loss_weights", type=float, nargs=5,
                        default=[1.0, 1.0, 1.0, 1.0, 1.0],
                        metavar=("W_TAUX", "W_TAUY", "W_SHFLX", "W_LHFLX", "W_QFLX"),
                        help="Per-variable loss multipliers in TAUX TAUY SHFLX LHFLX QFLX "
                             "order (default: 1 1 1 1 1). Logged as val/wloss_{var}.")
    parser.add_argument("--dsst_dt", action="store_true",
                        help="Append (SST[t]-SST[t-lag])/86400 as an extra input channel. "
                             "Only meaningful with --memory. Stats computed from chosen samples.")
    parser.add_argument("--augment", action="store_true",
                        help="Random circular longitude (W) roll on the TRAINING set only. "
                             "Physically exact (grid is lon-periodic); fights overfitting. "
                             "Default off — old runs reproduce identically.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW/Muon weight decay (default 1e-4 — the previous "
                             "hardcoded value, so old runs are unchanged).")
    parser.add_argument("--val_split_mode", choices=["random", "temporal"], default="random",
                        help="Val split: random (default, seed-42 permutation) or temporal "
                             "(hold out last 10%% by cache position as time proxy). "
                             "Ignored when --split_mode temporal is set.")
    parser.add_argument("--split_mode", choices=["random", "temporal"], default="random",
                        help="random (default): existing 90/10 random split. "
                             "temporal: year-based split following CAMulator (Chapman 2025) — "
                             "train 1980-2010, val 2011-2012, test 2013-2014. "
                             "Normaliser computed on training portion only (no leakage). "
                             "Override year boundaries with --val_years / --test_years.")
    parser.add_argument("--val_years", type=int, nargs=2, default=None,
                        metavar=("START", "END"),
                        help="Year range for the validation set (e.g. 2010 2012). "
                             "Enables year-based temporal split.")
    parser.add_argument("--test_years", type=int, nargs=2, default=None,
                        metavar=("START", "END"),
                        help="Year range for the held-out test set (e.g. 2013 2014). "
                             "These samples are excluded from train/val; indices saved to "
                             "test_indices.npy. R² reported at end of training.")
    parser.add_argument("--eval_test", action="store_true",
                        help="Skip training: load best_model.pt and report R² on the test set "
                             "defined by test_indices.npy (requires a completed training run "
                             "with --test_years).")
    args = parser.parse_args()

    if args.dsst_dt and not args.memory:
        print("WARNING: --dsst_dt ignored (only meaningful with --memory)")
        args.dsst_dt = False

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    daily     = (not args.sixhour and args.lag == 24) if not args.memory else False
    lag_steps = (args.lag // 6 if not daily else 1)   if not args.memory else 0
    rng       = np.random.default_rng(42)

    # Channel selection for memory experiments
    if args.memory:
        mem_channels    = [0, 1, 2, 3, 4, 5] if args.prev_solin else [0, 1, 2, 3, 4]
        input_vars_base = (["SST", "ICEFRAC", "SOLIN", "SST_prev", "ICEFRAC_prev"] +
                           (["SOLIN_prev"] if args.prev_solin else []))
    else:
        mem_channels    = None
        input_vars_base = INPUT_VARS
    input_vars = (input_vars_base
                  + (["dSST_dt"] if args.dsst_dt else [])
                  + (["CO2"]     if args.with_co2 else []))

    use_wandb = WANDB_AVAILABLE and args.wandb_project is not None
    if use_wandb:
        ep_tag = f"es{args.patience}" if args.patience > 0 else f"ep{args.epochs}"
        run_name = args.wandb_name or (
            f"unet"
            f"-sub{args.subsample}"
            f"-{ep_tag}"
            f"-base{args.base}"
            f"-lr{args.lr:.0e}"
            f"-{'6h' if not daily else 'daily'}"
            f"{f'-mem{args.memory_lag}h' if args.memory else f'-lag{args.lag}h'}"
            f"{'-solin' if args.memory and args.prev_solin else ''}"
            f"{'-co2' if args.with_co2 else ''}"
            f"-{args.loss}"
            f"{'-anomaly' if args.anomaly else ''}"
            f"{'-cosine' if args.scheduler == 'cosine' else ''}"
            f"{'-muon' if args.optimizer == 'muon' else ''}"
            f"{'-latw' if args.lat_weight else ''}"
            f"{'-vw' if any(w != 1.0 for w in args.loss_weights) else ''}"
            f"{'-dsst' if args.dsst_dt else ''}"
            f"{'-tvalsplit' if args.val_split_mode == 'temporal' else ''}"
        )
        # Resume the same wandb run if resuming training
        resume_run_id = None
        run_id_file   = out_dir / "wandb_run_id.txt"
        if args.resume and run_id_file.exists():
            resume_run_id = run_id_file.read_text().strip()

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            id=resume_run_id,
            resume="allow" if resume_run_id else None,
            config={
                "subsample": args.subsample,
                "epochs":    args.epochs,
                "batch":     args.batch,
                "base":      args.base,
                "lr":        args.lr,
                "daily":     daily,
                "input_vars":  input_vars,
                "target_vars": TARGET_VARS,
            },
        )
    else:
        use_wandb = False

    if use_wandb:
        wandb.define_metric("val/loss",   summary="min")
        wandb.define_metric("train/loss", summary="min")
        for v in TARGET_VARS:
            wandb.define_metric(f"val/loss_{v}",  summary="min")
            wandb.define_metric(f"val/wloss_{v}", summary="min")
            wandb.define_metric(f"val/r2_{v}",    summary="max")
        (out_dir / "wandb_run_id.txt").write_text(wandb.run.id)

    zarr_paths = sorted(glob.glob(args.zarr_glob))
    if not zarr_paths:
        raise RuntimeError(f"No zarr stores found matching: {args.zarr_glob}")
    print(f"Found {len(zarr_paths)} zarr stores")
    print(f"Device: {device}")

    # Discover which target vars are actually present
    ds0 = xr.open_zarr(zarr_paths[0], consolidated=False)
    tgt_vars = [v for v in TARGET_VARS if v in ds0.data_vars]
    ds0.close()
    n_out = len(tgt_vars)
    print(f"Input vars:  {input_vars}")
    print(f"Target vars: {tgt_vars}")
    print(f"Mode: {'daily means' if daily else '6-hourly'}, lag={args.lag}h, "
          f"subsample={args.subsample}")

    # --- Load data ---
    # Auto-resolve cache dir: explicit arg wins; otherwise use mode-specific default path.
    # Daily mode (lag=24, no --sixhour) → couple_cache (original)
    # Any 6-hourly mode                 → couple_cache_lag{N}h
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    elif args.memory:
        cache_dir = Path(f"/glade/work/praggarwal/couple_cache_mem{args.memory_lag}h")
    elif daily:
        cache_dir = Path("/glade/work/praggarwal/couple_cache")
    else:
        cache_dir = Path(f"/glade/work/praggarwal/couple_cache_lag{args.lag}h")
    cache_ok = (cache_dir / "X.npy").exists()

    t0 = time_module.time()
    if cache_ok:
        print(f"\nLoading from cache: {cache_dir} ...")
        X_np    = np.load(cache_dir / "X.npy",    mmap_mode="r")  # (N, 3, H, W)
        Y_np    = np.load(cache_dir / "Y.npy",    mmap_mode="r")  # (N, n_out, H, W)
        mask_np = np.load(cache_dir / "mask.npy", mmap_mode="r")  # (N, H, W)
        co2_np  = None
        if args.with_co2:
            co2_path = cache_dir / "co2.npy"
            assert co2_path.exists(), "co2.npy not found — run add_co2_to_cache.py first"
            co2_np = np.load(co2_path, mmap_mode="r")              # (N,)
            print("  CO2 channel enabled")
        N_full  = len(X_np)
        # Apply subsample via random selection
        chosen = rng.choice(N_full, size=max(1, int(N_full * args.subsample)), replace=False)
        chosen.sort()
        X_all = []
        for i in chosen:
            x = X_np[i][mem_channels] if args.memory else X_np[i]
            if args.dsst_dt:
                # SST[t]=ch0, SST_prev=ch3 in the raw 6-channel memory cache
                dsst = ((X_np[i][0] - X_np[i][3]) / 86400.0)[None]   # (1, H, W)
                x = np.concatenate([x, dsst], axis=0)
            if args.with_co2:
                x = np.concatenate([x, np.full((1, H, W), co2_np[i], dtype=np.float32)], axis=0)
            X_all.append(x.copy())
        Y_all    = [Y_np[i].copy()    for i in chosen]
        mask_all = [mask_np[i].copy() for i in chosen]
        print(f"  Loaded {len(X_all)} / {N_full} samples in {time_module.time()-t0:.1f}s")
    else:
        print("\nLoading data from zarr ...")
        X_all, Y_all, mask_all = [], [], []
        years_all_list = []
        for i, zp in enumerate(zarr_paths):
            ds_yr = xr.open_zarr(zp, consolidated=False)
            yr = int(ds_yr["time"].values[0].year)
            ds_yr.close()
            Xy, Yy, My = load_year(zp, tgt_vars, daily, args.subsample, rng, lag_steps)
            X_all.extend(Xy); Y_all.extend(Yy); mask_all.extend(My)
            years_all_list.extend([yr] * len(Xy))
            if (i + 1) % 5 == 0 or i == len(zarr_paths) - 1:
                print(f"  {i+1:3d}/{len(zarr_paths)} years  "
                      f"{len(X_all):5d} samples  "
                      f"{time_module.time()-t0:.0f}s")

    print(f"\nTotal samples: {len(X_all)}")

    # Build year label for every sample in X_all (used for year-based splits)
    _mem_lag_steps = (args.memory_lag // 6) if args.memory else 0
    if cache_ok:
        years_all = _load_years(zarr_paths, chosen, args.memory, _mem_lag_steps)
    else:
        years_all = np.array(years_all_list, dtype=np.int32)

    # --- Train / val / test split (before normaliser — training subset used for stats) ---
    if args.split_mode == "temporal" or args.val_years or args.test_years:
        val_s,  val_e  = args.val_years  if args.val_years  else (2011, 2012)
        test_s, test_e = args.test_years if args.test_years else (2013, 2014)
        val_mask   = (years_all >= val_s)  & (years_all <= val_e)
        test_mask  = (years_all >= test_s) & (years_all <= test_e)
        train_mask = ~val_mask & ~test_mask
        ti       = np.where(train_mask)[0]
        vi       = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]
        np.save(out_dir / "test_indices.npy", test_idx)
        print(f"Temporal split  (CAMulator-style, no leakage):")
        print(f"  Train: {len(ti):6d} samples  (years ≤ {val_s - 1})")
        print(f"  Val:   {len(vi):6d} samples  (years {val_s}–{val_e})")
        print(f"  Test:  {len(test_idx):6d} samples  (years {test_s}–{test_e})")
    else:
        test_idx = np.array([], dtype=int)
        n_val = max(1, len(X_all) // 10)
        if args.val_split_mode == 'temporal':
            ti = np.arange(len(X_all) - n_val)
            vi = np.arange(len(X_all) - n_val, len(X_all))
            print(f"Temporal val split: train [0,{len(ti)}), val [{len(ti)},{len(X_all)})")
        else:
            idx = rng.permutation(len(X_all))
            vi, ti = idx[:n_val], idx[n_val:]

    # --- Normalizer ---
    # temporal split + precomputed out_dir/normalizer.npz → load it (no leakage, fast)
    # temporal split, no precomputed file               → compute from training samples
    # random split                                      → use all-years cache stats
    precomp_norm = out_dir / "normalizer.npz"
    cached_norm  = cache_dir / "normalizer.npz" if cache_ok else None
    if args.split_mode == "temporal" and precomp_norm.exists():
        print(f"Loading pre-computed training-only normaliser from {precomp_norm} ...")
        norm = Normalizer.load(precomp_norm)
    elif args.split_mode == "temporal":
        print("Computing normalisation stats from training samples ...")
        norm = compute_norm([X_all[k] for k in ti], [Y_all[k] for k in ti])
        norm.save(precomp_norm)
    else:
        pass  # fall through to cache-based block below

    use_cache_norm = (args.split_mode != "temporal" and
                      cached_norm is not None and cached_norm.exists())
    if use_cache_norm:
        print("Loading normalisation stats from cache ...")
        norm_full = Normalizer.load(cached_norm)
        if args.memory:
            norm = Normalizer(norm_full.x_mean[mem_channels], norm_full.x_std[mem_channels],
                              norm_full.y_mean, norm_full.y_std)
        else:
            norm = norm_full
        if args.dsst_dt:
            dsst_ch  = len(mem_channels)
            dsst_arr = np.stack([X_all[k][dsst_ch] for k in ti]).astype(np.float64)
            dsst_mean = np.array([dsst_arr.mean()],             dtype=np.float32)
            dsst_std  = np.array([max(dsst_arr.std(), 1e-8)],   dtype=np.float32)
            norm = Normalizer(
                np.concatenate([norm.x_mean, dsst_mean]),
                np.concatenate([norm.x_std,  dsst_std]),
                norm.y_mean, norm.y_std,
            )
            print(f"  {'dSST_dt':10s}: mean={dsst_mean[0]:.4e}  std={dsst_std[0]:.4e}")
        if args.with_co2:
            co2_vals = np.array([co2_np[chosen[k]] for k in ti], dtype=np.float32)
            co2_mean = np.array([co2_vals.mean()], dtype=np.float32)
            co2_std  = np.array([co2_vals.std() + 1e-8], dtype=np.float32)
            norm = Normalizer(
                np.concatenate([norm.x_mean, co2_mean]),
                np.concatenate([norm.x_std,  co2_std]),
                norm.y_mean, norm.y_std,
            )
    elif not use_cache_norm and args.split_mode != "temporal":
        print("Computing normalisation stats ...")
        norm = compute_norm(X_all, Y_all)
    if not precomp_norm.exists():
        norm.save(out_dir / "normalizer.npz")
    for i, v in enumerate(input_vars):
        print(f"  {v:10s}: mean={norm.x_mean[i]:.3f}  std={norm.x_std[i]:.3f}")
    for i, v in enumerate(tgt_vars):
        print(f"  {v:10s}: mean={norm.y_mean[i]:.4e}  std={norm.y_std[i]:.4e}")

    # --- Anomaly prediction (opt-in) ---
    clim        = None   # (12, n_out, H, W) monthly climatology
    clim_torch  = None
    Y_all_raw_vi = None  # raw val Y for R² (only used when --anomaly)
    months_vi   = None

    if args.anomaly:
        print("\nComputing monthly climatology for anomaly prediction ...")
        months_chosen = _load_months(zarr_paths, chosen,
                                     args.memory, args.memory_lag // 6)

        # Climatology from training samples only (no leakage from val)
        clim = np.zeros((12, n_out, H, W), dtype=np.float32)
        cnt  = np.zeros(12, dtype=np.int32)
        for k in ti:
            m = int(months_chosen[k]) - 1
            clim[m] += Y_all[k]
            cnt[m]  += 1
        for m in range(12):
            if cnt[m] > 0:
                clim[m] /= cnt[m]
        np.save(out_dir / "climatology.npy", clim)
        print(f"  Climatology saved  (shape={clim.shape}, "
              f"counts per month: {cnt.tolist()})")

        # Save raw val Y before replacing with anomalies
        Y_all_raw_vi = [Y_all[k].copy() for k in vi]
        months_vi    = months_chosen[vi]

        # Replace Y_all entries with anomalies
        Y_all = [Y_all[k] - clim[int(months_chosen[k]) - 1]
                 for k in range(len(Y_all))]

        # Recompute Y normalizer from training anomalies only
        y_trn_anom = np.stack([Y_all[k] for k in ti])   # (N_trn, n_out, H, W)
        anom_mean  = y_trn_anom.mean(axis=(0, 2, 3)).astype(np.float32)
        anom_std   = (y_trn_anom.std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)
        norm = Normalizer(norm.x_mean, norm.x_std, anom_mean, anom_std)
        print("  Anomaly normalizer (Y):")
        for i, v in enumerate(tgt_vars):
            print(f"    {v:10s}: mean={anom_mean[i]:.4e}  std={anom_std[i]:.4e}")

        clim_torch = torch.from_numpy(clim).to(device)

    trn_ds = PairDataset([X_all[i] for i in ti], [Y_all[i] for i in ti],
                         [mask_all[i] for i in ti], norm, augment=args.augment)
    val_ds = PairDataset([X_all[i] for i in vi], [Y_all[i] for i in vi],
                         [mask_all[i] for i in vi], norm)

    trn_loader = DataLoader(trn_ds, batch_size=args.batch, shuffle=True,
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=4, pin_memory=True)

    test_loader = None
    if len(test_idx) > 0:
        test_ds = PairDataset([X_all[i] for i in test_idx], [Y_all[i] for i in test_idx],
                              [mask_all[i] for i in test_idx], norm)
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                                 num_workers=4, pin_memory=True)

    # --- Model ---
    model = UNet(n_in=len(input_vars), n_out=n_out, base=args.base,
                 dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nUNet: {n_params:,} parameters  (base={args.base}, "
          f"n_in={len(input_vars)}, n_out={n_out}, dropout={args.dropout})")
    print(f"Loss: {args.loss}"
          + (f" (M={args.crps_members} members)" if args.loss == 'crps' else ""))

    json.dump({"n_in": len(input_vars), "n_out": n_out, "base": args.base,
               "input_vars": input_vars, "output_vars": tgt_vars, "daily": daily},
              open(out_dir / "model_config.json", "w"), indent=2)

    # --- eval_test mode: skip training, report R² on held-out test set ---
    if args.eval_test:
        assert test_loader is not None, \
            "--eval_test requires --test_years and test_indices.npy to exist"
        print("\n[eval_test] Loading best checkpoint ...")
        model.load_state_dict(torch.load(out_dir / "best_model.pt",
                                         map_location=device, weights_only=True))
        print("[eval_test] Computing R² on test set ...")
        r2_test = compute_r2(model, test_loader, norm, device, n_out)
        r2_test_dict = {v: float(r2_test[i]) for i, v in enumerate(tgt_vars)}
        json.dump(r2_test_dict, open(out_dir / "r2_scores_test.json", "w"), indent=2)
        print("Test R² scores (ocean/ice points):")
        for v, s in r2_test_dict.items():
            print(f"  {v:10s}: {s:.4f}")
        if use_wandb:
            wandb.log({f"test/r2_{v}": s for v, s in r2_test_dict.items()})
            wandb.finish()
        return

    # Precompute latitude weights (1, 1, H, 1) — cos(lat) over 192-pt regular grid
    if args.lat_weight:
        lats_rad = np.deg2rad(np.linspace(-90.0, 90.0, H))
        lat_w = torch.from_numpy(np.cos(lats_rad).astype(np.float32)).to(device).view(1, 1, H, 1)
        print(f"Latitude weighting: cos(lat) in [{lat_w.min():.3f}, {lat_w.max():.3f}]")
    else:
        lat_w = None

    # Per-variable loss weights — None when all ones (identity, existing behavior)
    _lw   = args.loss_weights[:n_out]
    var_w = (torch.tensor(_lw, dtype=torch.float32, device=device)
             if any(w != 1.0 for w in _lw) else None)
    if var_w is not None:
        print(f"Variable weights: { {v: w for v, w in zip(tgt_vars, _lw)} }")

    if args.optimizer == "muon":
        # Muon for 2D+ weight tensors (conv kernels, linear weights)
        # AdamW for biases, norm weights/biases (1D params)
        muon_params  = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        adamw_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
        optimizer = torch.optim.AdamW(adamw_params, lr=args.lr * 0.1, weight_decay=args.weight_decay)
        muon_opt  = Muon(muon_params, lr=args.lr, momentum=0.95, weight_decay=args.weight_decay)
        print(f"Muon: {len(muon_params)} 2D+ tensors, "
              f"AdamW: {len(adamw_params)} 1D tensors")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        muon_opt  = None
    total_epochs_for_sched = args.max_epochs if args.max_epochs > 0 else args.epochs
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs_for_sched, eta_min=args.lr * 1e-3
        )
    else:
        # SGDR with T_0=30 cycles indefinitely — safe for both fixed and early-stop runs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=1, eta_min=args.lr * 1e-3
        )

    start_epoch = 1
    history     = []
    best_val    = float("inf")

    ckpt_path = out_dir / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        print(f"Resuming from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if muon_opt is not None and "muon_opt" in ckpt:
            muon_opt.load_state_dict(ckpt["muon_opt"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt["best_val"]
        history     = ckpt["history"]
        print(f"  Resumed at epoch {start_epoch}/{args.epochs}, best_val={best_val:.4f}")

    # --- Training loop ---
    total_epochs = args.max_epochs if args.max_epochs > 0 else (args.epochs + args.extra_epochs)
    early_stop   = args.patience > 0
    no_improve   = 0   # epochs without val loss improvement

    if args.extra_epochs > 0 and args.resume:
        # Cosine warm restart from lr*0.1 over the extension period.
        extension_lr = args.lr * 0.1
        for pg in optimizer.param_groups:
            pg["lr"] = extension_lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.extra_epochs, T_mult=1, eta_min=0.0
        )
        print(f"Extension mode: {args.extra_epochs} extra epochs, "
              f"SGDR restart from lr={extension_lr:.1e}")

    if early_stop:
        print(f"\nTraining with early stopping (patience={args.patience}, "
              f"max_epochs={total_epochs}) ...")
    else:
        print(f"\nTraining for {total_epochs} epochs ...")

    vi_indices = np.array(vi)  # save before inner loops shadow `vi`

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        trn_loss = 0.0
        t_ep = time_module.time()

        for x_n, y_n, mask in trn_loader:
            x_n  = x_n.to(device)
            y_n  = y_n.to(device)
            mask = mask.to(device)
            if args.loss == 'crps':
                preds = [model(x_n) for _ in range(args.crps_members)]
                loss  = masked_crps(preds, y_n, mask, lat_w=lat_w, var_w=var_w)
            elif args.loss == 'mae':
                loss = masked_mae(model(x_n), y_n, mask, lat_w=lat_w, var_w=var_w)
            else:
                loss = masked_mse(model(x_n), y_n, mask, lat_w=lat_w, var_w=var_w)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if muon_opt is not None:
                muon_opt.step()
            trn_loss += loss.item()

        val_loss    = 0.0
        val_loss_pv = np.zeros(n_out)   # per-variable accumulator
        with torch.no_grad():
            if args.loss == 'crps':
                # Val CRPS: M stochastic passes with dropout active
                model.train()
                for x_n, y_n, mask in val_loader:
                    x_n, y_n, mask = x_n.to(device), y_n.to(device), mask.to(device)
                    M     = args.crps_members
                    preds = [model(x_n) for _ in range(M)]
                    val_loss += masked_crps(preds, y_n, mask, lat_w=lat_w, var_w=var_w).item()
                    m     = mask.unsqueeze(1)
                    m_eff = m * lat_w if lat_w is not None else m
                    n_pts = m_eff.sum() + 1e-8
                    for vi in range(n_out):
                        skill_v  = sum(torch.abs(p[:,vi:vi+1] - y_n[:,vi:vi+1]) for p in preds) / M
                        spread_v = sum(torch.abs(preds[i][:,vi:vi+1] - preds[j][:,vi:vi+1])
                                       for i in range(M) for j in range(M) if i != j) / (M * (M - 1))
                        val_loss_pv[vi] += ((skill_v - 0.5 * spread_v) * m_eff).sum().item() / n_pts.item()
            else:
                model.eval()
                for x_n, y_n, mask in val_loader:
                    x_n, y_n, mask = x_n.to(device), y_n.to(device), mask.to(device)
                    pred  = model(x_n)
                    m     = mask.unsqueeze(1)
                    m_eff = m * lat_w if lat_w is not None else m
                    n_pts = m_eff.sum() + 1e-8
                    if args.loss == 'mae':
                        val_loss += masked_mae(pred, y_n, mask, lat_w=lat_w, var_w=var_w).item()
                        for vi in range(n_out):
                            val_loss_pv[vi] += (torch.abs(pred[:,vi:vi+1] - y_n[:,vi:vi+1]) * m_eff).sum().item() / n_pts.item()
                    else:  # mse
                        val_loss += masked_mse(pred, y_n, mask, lat_w=lat_w, var_w=var_w).item()
                        for vi in range(n_out):
                            val_loss_pv[vi] += (((pred[:,vi:vi+1] - y_n[:,vi:vi+1])**2) * m_eff).sum().item() / n_pts.item()

        trn_loss    /= len(trn_loader)
        val_loss    /= max(1, len(val_loader))
        val_loss_pv /= max(1, len(val_loader))
        history.append((trn_loss, val_loss))
        scheduler.step()

        improved = val_loss < best_val
        if improved:
            best_val   = val_loss
            no_improve = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            no_improve += 1

        # SGDR warm restart resets the LR to its peak — don't penalise the
        # transient val spike that follows; reset patience at each restart.
        if early_stop and args.scheduler == "sgdr" and epoch % 30 == 1 and epoch > 1:
            no_improve = 0

        # Full checkpoint for resuming
        ckpt_dict = {
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "best_val":   best_val,
            "history":    history,
            "no_improve": no_improve,
        }
        if muon_opt is not None:
            ckpt_dict["muon_opt"] = muon_opt.state_dict()
        torch.save(ckpt_dict, out_dir / "checkpoint.pt")

        es_tag = f"  no_improve={no_improve}/{args.patience}" if early_stop else ""
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}/{total_epochs}  "
              f"trn={trn_loss:.4f}  val={val_loss:.4f}  "
              f"lr={lr_now:.1e}  "
              f"({'*' if improved else ' '}) "
              f"({time_module.time()-t_ep:.0f}s){es_tag}")

        if use_wandb:
            log_dict = {
                "train/loss": trn_loss,
                "val/loss":   val_loss,
                "lr":         lr_now,
            }
            if early_stop:
                log_dict["early_stop/no_improve"] = no_improve
            for vi, vname in enumerate(tgt_vars):
                log_dict[f"val/loss_{vname}"] = val_loss_pv[vi]
                if var_w is not None:
                    log_dict[f"val/wloss_{vname}"] = args.loss_weights[vi] * val_loss_pv[vi]
            wandb.log(log_dict, step=epoch)
            if epoch % 10 == 0:
                _log_wandb_maps(model, val_ds, norm, device, tgt_vars, n_samples=4, step=epoch)

        if early_stop and no_improve >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs).")
            if use_wandb:
                wandb.log({"early_stop/stopped_epoch": epoch}, step=epoch)
            break

    # --- R² on validation set ---
    print("\nComputing R² scores on validation set (ocean points only) ...")
    model.load_state_dict(torch.load(out_dir / "best_model.pt",
                                     map_location=device, weights_only=True))
    if args.anomaly:
        # R² on absolute fluxes: unnormalise anomaly prediction + add monthly clim
        X_vi_list    = [X_all[k] for k in vi_indices]
        mask_vi_list = [mask_all[k] for k in vi_indices]
        r2 = compute_r2_with_clim(model, norm, device, n_out,
                                   X_vi_list, Y_all_raw_vi, mask_vi_list,
                                   months_vi, clim)
    else:
        r2 = compute_r2(model, val_loader, norm, device, n_out)
    r2_dict = {v: float(r2[i]) for i, v in enumerate(tgt_vars)}
    json.dump(r2_dict, open(out_dir / "r2_scores.json", "w"), indent=2)
    print("R² scores (ocean/ice points):")
    for v, s in r2_dict.items():
        print(f"  {v:10s}: {s:.4f}")

    if use_wandb:
        wandb.log({f"val/r2_{v}": s for v, s in r2_dict.items()})

    # --- R² on held-out test set ---
    if test_loader is not None:
        print("\nComputing R² scores on test set (ocean points only) ...")
        r2_test = compute_r2(model, test_loader, norm, device, n_out)
        r2_test_dict = {v: float(r2_test[i]) for i, v in enumerate(tgt_vars)}
        json.dump(r2_test_dict, open(out_dir / "r2_scores_test.json", "w"), indent=2)
        print("Test R² scores (ocean/ice points):")
        for v, s in r2_test_dict.items():
            print(f"  {v:10s}: {s:.4f}")
        if use_wandb:
            wandb.log({f"test/r2_{v}": s for v, s in r2_test_dict.items()})
        _log_rmse_maps(model, val_loader, norm, device, tgt_vars, step=epoch)

    # --- Plots ---
    trn_h = [h[0] for h in history]
    val_h = [h[1] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(trn_h, label="train")
    axes[0].plot(val_h, label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(f"Masked {args.loss.upper()} (normalised)")
    axes[0].set_title("Loss curve")
    axes[0].legend()

    axes[1].barh(list(r2_dict.keys()), list(r2_dict.values()))
    axes[1].set_xlim(0, 1)
    axes[1].axvline(0.9, color="gray", linestyle="--", alpha=0.5, label="R²=0.9")
    axes[1].set_xlabel("R²")
    axes[1].set_title("Validation R² (ocean/ice points)")
    axes[1].legend()

    plt.tight_layout()
    summary_path = out_dir / "training_summary.png"
    fig.savefig(summary_path, dpi=120, bbox_inches="tight")
    plt.close()

    if use_wandb:
        wandb.log({
            "charts/training_summary": wandb.Image(str(summary_path)),
            "charts/best_val_loss": best_val,
        })
        wandb.finish()

    print(f"\nDone. Outputs saved to {out_dir}/")
    print("  best_model.pt, normalizer.npz, model_config.json,")
    print("  r2_scores.json, training_summary.png")


if __name__ == "__main__":
    main()
