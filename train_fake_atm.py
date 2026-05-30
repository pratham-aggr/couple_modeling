"""
Fake Atmosphere MLP
====================
Pointwise regression: (SST, ICEFRAC, SOLIN, sin/cos(DOY), lat, lon, LANDFRAC)
                    → (FSDS_J, FLDS_J, FSUS, FLUS, FSUTOA, FLUT, PRECT, TS, U10)

Each grid cell × timestep is an independent sample — no spatial structure.

Usage:
    python train_fake_atm.py \
        --zarr_glob "/glade/derecho/scratch/wchapman/b_credit_runs/b.e21.CREDIT_climate_branch_1980_????_zmdata_ERA5scaled_zmdata_Qtot.zarr" \
        --out_dir ./output
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import xarray as xr
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_VARS  = ["SST", "ICEFRAC", "SOLIN", "LANDFRAC"]
OUTPUT_VARS = ["FSDS_J", "FLDS_J", "FSUS", "FLUS", "FSUTOA", "FLUT", "PRECT", "TS", "U10"]

# Outputs that must be >= 0 (enforced at inference)
NON_NEGATIVE = {"FSDS_J", "FLDS_J", "FSUS", "FLUS", "FSUTOA", "FLUT", "PRECT", "U10"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_zarr(path: str) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=False)


def build_features(ds: xr.Dataset, subsample: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, Y) arrays from one zarr store.

    X shape: (N, n_inputs)   where N = time * lat * lon
    Y shape: (N, n_outputs)

    Features:
        SST, ICEFRAC, SOLIN, LANDFRAC,
        sin(DOY*2pi/365), cos(DOY*2pi/365),
        sin(lat), cos(lat),
        sin(lon), cos(lon)
    """
    time  = ds["time"].values
    lat   = ds["latitude"].values    # (nlat,)
    lon   = ds["longitude"].values   # (nlon,)
    ntime, nlat, nlon = len(time), len(lat), len(lon)

    doy = xr.DataArray(time).dt.dayofyear.values.astype(np.float32)  # (ntime,)

    # --- cyclic encodings ---
    doy_sin = np.sin(2 * np.pi * doy / 365)[:, None, None] * np.ones((1, nlat, nlon))
    doy_cos = np.cos(2 * np.pi * doy / 365)[:, None, None] * np.ones((1, nlat, nlon))

    lat_sin = np.sin(np.deg2rad(lat))[None, :, None] * np.ones((ntime, 1, nlon))
    lat_cos = np.cos(np.deg2rad(lat))[None, :, None] * np.ones((ntime, 1, nlon))

    lon_sin = np.sin(np.deg2rad(lon))[None, None, :] * np.ones((ntime, nlat, 1))
    lon_cos = np.cos(np.deg2rad(lon))[None, None, :] * np.ones((ntime, nlat, 1))

    # --- zarr variables (load to memory in chunks) ---
    print("  Loading input variables...")
    input_arrays = []
    for v in INPUT_VARS:
        arr = ds[v].values.astype(np.float32)  # (time, lat, lon)
        input_arrays.append(arr)

    print("  Loading output variables...")
    output_arrays = []
    for v in OUTPUT_VARS:
        arr = ds[v].values.astype(np.float32)
        output_arrays.append(arr)

    # --- stack and flatten ---
    X = np.stack(input_arrays + [
        doy_sin.astype(np.float32),
        doy_cos.astype(np.float32),
        lat_sin.astype(np.float32),
        lat_cos.astype(np.float32),
        lon_sin.astype(np.float32),
        lon_cos.astype(np.float32),
    ], axis=-1)  # (time, lat, lon, n_features)

    Y = np.stack(output_arrays, axis=-1)  # (time, lat, lon, n_outputs)

    # flatten spatial dims
    X = X.reshape(-1, X.shape[-1])
    Y = Y.reshape(-1, Y.shape[-1])

    # drop rows with NaNs (ocean-only vars like SST are NaN over land)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X, Y = X[valid], Y[valid]

    if subsample < 1.0:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=int(len(X) * subsample), replace=False)
        X, Y = X[idx], Y[idx]

    print(f"  Valid samples: {len(X):,} / {valid.size:,}")
    return X, Y


def feature_names() -> list[str]:
    return INPUT_VARS + ["doy_sin", "doy_cos", "lat_sin", "lat_cos", "lon_sin", "lon_cos"]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class Normalizer:
    def __init__(self):
        self.x_mean = self.x_std = None
        self.y_mean = self.y_std = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.x_mean = X.mean(axis=0)
        self.x_std  = X.std(axis=0) + 1e-8
        self.y_mean = Y.mean(axis=0)
        self.y_std  = Y.std(axis=0) + 1e-8

    def transform_x(self, X): return (X - self.x_mean) / self.x_std
    def transform_y(self, Y): return (Y - self.y_mean) / self.y_std
    def inverse_y(self, Y):   return Y * self.y_std + self.y_mean

    def save(self, path: str):
        np.savez(path,
                 x_mean=self.x_mean, x_std=self.x_std,
                 y_mean=self.y_mean, y_std=self.y_std)

    @classmethod
    def load(cls, path: str):
        n = cls()
        d = np.load(path)
        n.x_mean, n.x_std = d["x_mean"], d["x_std"]
        n.y_mean, n.y_std = d["y_mean"], d["y_std"]
        return n


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FakeAtmMLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int = 256, depth: int = 4):
        super().__init__()
        layers = []
        in_dim = n_in
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU()]
            in_dim = hidden
        layers.append(nn.Linear(hidden, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
          n_epochs: int, lr: float, device: torch.device, out_dir: Path) -> list[float]:

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(1, n_epochs + 1):
        # train
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        train_loss = running / len(train_loader.dataset)

        # val
        model.eval()
        running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                running += criterion(model(xb), yb).item() * len(xb)
        val_loss = running / len(val_loader.dataset)

        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "best_model.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{n_epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

    return train_losses, val_losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, loader: DataLoader, norm: Normalizer,
             device: torch.device) -> dict[str, float]:
    """Per-variable R² on denormalized predictions."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            targets.append(yb.numpy())

    preds   = norm.inverse_y(np.concatenate(preds))
    targets = norm.inverse_y(np.concatenate(targets))

    r2 = {}
    for i, v in enumerate(OUTPUT_VARS):
        ss_res = ((targets[:, i] - preds[:, i]) ** 2).sum()
        ss_tot = ((targets[:, i] - targets[:, i].mean()) ** 2).sum()
        r2[v] = float(1 - ss_res / ss_tot)
    return r2


def plot_losses(train_losses, val_losses, out_path: str):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses,   label="val")
    plt.xlabel("Epoch"); plt.ylabel("MSE (normalized)"); plt.legend()
    plt.title("Training curve"); plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved loss curve → {out_path}")


def plot_r2(r2: dict, out_path: str):
    vars_ = list(r2.keys())
    vals  = [r2[v] for v in vars_]
    colors = ["steelblue" if v >= 0.5 else "tomato" for v in vals]

    plt.figure(figsize=(10, 4))
    plt.bar(vars_, vals, color=colors)
    plt.axhline(0.5, color="k", linestyle="--", linewidth=0.8, label="R²=0.5")
    plt.ylim(-0.1, 1.05); plt.ylabel("R²"); plt.title("Per-variable R² (test set)")
    plt.xticks(rotation=30, ha="right"); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved R² plot → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_glob", required=True,
                        help='Glob pattern for zarr stores, e.g. "/path/to/*.zarr"')
    parser.add_argument("--out_dir",   default="./output")
    parser.add_argument("--epochs",    type=int,   default=50)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--batch",     type=int,   default=4096)
    parser.add_argument("--hidden",    type=int,   default=256)
    parser.add_argument("--depth",     type=int,   default=4)
    parser.add_argument("--val_frac",  type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--subsample",  type=float, default=0.05,
                        help="Fraction of valid samples to keep per zarr (default 0.05 = 5%%)")
    parser.add_argument("--max_years", type=int,   default=None,
                        help="Cap number of zarr stores loaded (useful for quick tests)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- load data ---
    paths = sorted(glob.glob(args.zarr_glob))
    if not paths:
        raise FileNotFoundError(f"No zarr stores matched: {args.zarr_glob}")
    if args.max_years:
        paths = paths[:args.max_years]
    print(f"Loading {len(paths)} zarr store(s)...")

    X_all, Y_all = [], []
    for p in paths:
        print(f"  {p}")
        ds = load_zarr(p)
        X, Y = build_features(ds, subsample=args.subsample)
        X_all.append(X)
        Y_all.append(Y)
        ds.close()

    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)
    print(f"Total samples: {len(X_all):,}  |  features: {X_all.shape[1]}  |  outputs: {Y_all.shape[1]}")

    # --- normalize ---
    norm = Normalizer()
    norm.fit(X_all, Y_all)
    norm.save(str(out_dir / "normalizer.npz"))
    X_norm = norm.transform_x(X_all).astype(np.float32)
    Y_norm = norm.transform_y(Y_all).astype(np.float32)

    # --- split ---
    n      = len(X_norm)
    n_test = int(n * args.test_frac)
    n_val  = int(n * args.val_frac)
    n_train= n - n_val - n_test

    dataset = TensorDataset(torch.from_numpy(X_norm), torch.from_numpy(Y_norm))
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test],
                                              generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {n_train:,}  Val: {n_val:,}  Test: {n_test:,}")

    # --- model ---
    n_in  = X_norm.shape[1]
    n_out = Y_norm.shape[1]
    model = FakeAtmMLP(n_in, n_out, hidden=args.hidden, depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # --- train ---
    train_losses, val_losses = train(model, train_loader, val_loader,
                                     args.epochs, args.lr, device, out_dir)
    plot_losses(train_losses, val_losses, str(out_dir / "loss_curve.png"))

    # --- evaluate ---
    model.load_state_dict(torch.load(out_dir / "best_model.pt", weights_only=True))
    r2 = evaluate(model, test_loader, norm, device)
    plot_r2(r2, str(out_dir / "r2_scores.png"))

    print("\n=== R² scores (test set) ===")
    for v, r in r2.items():
        flag = "✓" if r >= 0.5 else "✗"
        print(f"  {flag} {v:12s}  R² = {r:.3f}")

    with open(out_dir / "r2_scores.json", "w") as f:
        json.dump(r2, f, indent=2)

    # save model config for inference
    config = dict(n_in=n_in, n_out=n_out, hidden=args.hidden, depth=args.depth,
                  input_vars=feature_names(), output_vars=OUTPUT_VARS)
    with open(out_dir / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()