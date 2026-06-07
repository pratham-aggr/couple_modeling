#!/usr/bin/env python
"""Build a self-contained interactive demo gallery for the MEMO emulator.

For a handful of held-out test timesteps (2013-2014) it runs three trained
models on the *same* timestep and renders, per (model, flux), a composite map
[ SST input | model prediction | CESM truth | error ]. All three models are
driven from the single mem24h cache (its channels 0,1,2 = SST, ICEFRAC, SOLIN
are exactly the memory-free model's inputs), so every model sees the identical
timestep.

Outputs:
  demo/index.html              self-contained viewer (reads manifest + PNGs)
  demo/manifest.json           dates / fluxes / models / scores
  demo/assets/<model>__<flux>__<k>.png

Run on a node with the `atm` conda env (CPU is fine -- ~36 forward passes).
"""
import json
import glob
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_unet import UNet, Normalizer  # noqa: E402

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
CACHE   = Path("/glade/campaign/work/praggarwal/couple_cache_mem24h")
ZGLOB   = "/glade/derecho/scratch/wchapman/b_credit_runs/b.e21.CREDIT_climate_branch_1980_????_zmdata_ERA5scaled_zmdata_Qtot.zarr"
MEM_LAG_STEPS = 4          # mem24h preprocessing drops the first 24h (4 steps) per year
TEST_YEARS    = (2013, 2014)
N_PICKS       = 12         # number of test timesteps to showcase
H, W = 192, 288

FLUXES = ["TAUX", "TAUY", "SHFLX", "LHFLX", "QFLX"]
# Descriptive names (the cache stores model-native units, so we avoid asserting SI units).
UNITS  = {"TAUX": "zonal wind stress",   "TAUY": "meridional wind stress",
          "SHFLX": "sensible heat flux", "LHFLX": "latent heat flux",
          "QFLX": "moisture flux"}

# (key, label, out_dir, channel-builder, dropout, color)
MODELS = [
    ("memfree", "Memory-free (lag 0)", "output_unet_lag0h_temporal",            "memfree", 0.0,  "#9e9e9e"),
    ("memo",    "MEMO (mem24h)",        "output_unet_mem24h_temporal",           "mem",     0.0,  "#4c72b0"),
    ("drop",    "MEMO + dropout (best)","output_unet_mem24h_dsst_temporal_drop", "dsst",    0.1,  "#c44e52"),
]

OUT      = ROOT / "demo"
ASSETS   = OUT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

device = torch.device("cpu")


def build_input(x6, kind):
    """x6: (6,H,W) raw mem cache channels [SST,ICEFRAC,SOLIN,SST_prev,ICEFRAC_prev,SOLIN_prev]."""
    if kind == "memfree":
        return x6[[0, 1, 2]]
    if kind == "mem":
        return x6[[0, 1, 2, 3, 4]]
    if kind == "dsst":
        dsst = ((x6[0] - x6[3]) / 86400.0)[None]
        return np.concatenate([x6[[0, 1, 2, 3, 4]], dsst], axis=0)
    raise ValueError(kind)


def load_model(out_dir, kind, dropout):
    cfg = json.loads((ROOT / out_dir / "model_config.json").read_text())
    m = UNet(n_in=cfg["n_in"], n_out=cfg["n_out"], base=cfg["base"], dropout=dropout)
    sd = torch.load(ROOT / out_dir / "best_model.pt", map_location=device, weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    norm = Normalizer.load(ROOT / out_dir / "normalizer.npz")
    return m, norm


def masked_r2(pred, truth, mask):
    m = mask.astype(bool)
    y, yh = truth[m], pred[m]
    ss_res = np.sum((y - yh) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


# ----------------------------------------------------------------------------
# 1. Reconstruct per-sample datetimes (aligned with cache rows) and find test rows
# ----------------------------------------------------------------------------
print("Reconstructing sample datetimes from zarr time axes ...")
zpaths = sorted(glob.glob(ZGLOB))
times_all = []
for zp in zpaths:
    ds = xr.open_zarr(zp, consolidated=False)
    t = ds["time"].values
    ds.close()
    times_all.extend(list(t[MEM_LAG_STEPS:]))   # mem preprocessing drops first 4 steps/yr
times_all = np.array(times_all, dtype=object)
print(f"  reconstructed {len(times_all)} datetimes")

years = np.array([int(t.year) for t in times_all])
test_rows = np.where((years >= TEST_YEARS[0]) & (years <= TEST_YEARS[1]))[0]
print(f"  test rows ({TEST_YEARS[0]}-{TEST_YEARS[1]}): {len(test_rows)}")

picks = test_rows[np.linspace(0, len(test_rows) - 1, N_PICKS).round().astype(int)]
pick_dates = [
    f"{times_all[i].year:04d}-{times_all[i].month:02d}-{times_all[i].day:02d} "
    f"{times_all[i].hour:02d}:00" for i in picks
]
for k, (i, d) in enumerate(zip(picks, pick_dates)):
    print(f"  pick {k:2d}: row {i}  {d}")

# ----------------------------------------------------------------------------
# 2. Lat/lon extent for nicer maps
# ----------------------------------------------------------------------------
ds0 = xr.open_zarr(zpaths[-1], consolidated=False)
lat = ds0["lat"].values if "lat" in ds0 else np.linspace(-90, 90, H)
lon = ds0["lon"].values if "lon" in ds0 else np.linspace(0, 360, W)
ds0.close()
extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

# ----------------------------------------------------------------------------
# 3. Load cache rows for the picks
# ----------------------------------------------------------------------------
print("Loading cache rows ...")
X_np    = np.load(CACHE / "X.npy",    mmap_mode="r")
Y_np    = np.load(CACHE / "Y.npy",    mmap_mode="r")
mask_np = np.load(CACHE / "mask.npy", mmap_mode="r")

X_pick = np.stack([np.asarray(X_np[i],    dtype=np.float32) for i in picks])   # (P,6,H,W)
Y_pick = np.stack([np.asarray(Y_np[i],    dtype=np.float32) for i in picks])   # (P,5,H,W)
M_pick = np.stack([np.asarray(mask_np[i], dtype=np.float32) for i in picks])   # (P,H,W)

# ----------------------------------------------------------------------------
# 4. Run every model on every pick, render composites
# ----------------------------------------------------------------------------
loaded = {key: load_model(out_dir, kind, dp)
          for key, _, out_dir, kind, dp, _ in MODELS}

# per-(model,flux,pick) R^2 for display
per_r2 = {key: {f: [] for f in FLUXES} for key, *_ in MODELS}


def render(sst, pred, truth, mask, flux, model_label, date, r2, path):
    land = ~mask.astype(bool)
    sstm   = np.where(land, np.nan, sst)
    predm  = np.where(land, np.nan, pred)
    truthm = np.where(land, np.nan, truth)
    errm   = np.where(land, np.nan, pred - truth)

    vlim = np.nanpercentile(np.abs(truthm), 98)
    if not np.isfinite(vlim) or vlim == 0:
        vlim = np.nanmax(np.abs(truthm)) or 1.0
    elim = np.nanpercentile(np.abs(errm), 98)
    if not np.isfinite(elim) or elim == 0:
        elim = vlim * 0.5 or 1.0

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 2.5))
    panels = [
        (sstm,   "SST input (K)", "coolwarm", None,  None),
        (predm,  f"{model_label}\nprediction", "RdBu_r", -vlim, vlim),
        (truthm, "CESM truth",        "RdBu_r", -vlim, vlim),
        (errm,   "error (pred − truth)", "PuOr", -elim, elim),
    ]
    for ax, (data, title, cmap, vmn, vmx) in zip(axes, panels):
        im = ax.imshow(data, origin="lower", extent=extent, aspect="auto",
                       cmap=cmap, vmin=vmn, vmax=vmx)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#dddddd")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    fig.suptitle(f"{flux} — {UNITS[flux]}    ·    {date} UTC    ·    "
                 f"sample $R^2$ = {r2:.3f}", fontsize=11, y=1.06)
    fig.tight_layout()
    fig.savefig(path, dpi=80, bbox_inches="tight",
                pil_kwargs={"quality": 85, "optimize": True})
    plt.close(fig)


print("Running inference + rendering ...")
with torch.no_grad():
    for key, label, out_dir, kind, dp, color in MODELS:
        model, norm = loaded[key]
        for k in range(len(picks)):
            xin = build_input(X_pick[k], kind)                       # (n_in,H,W)
            xn  = (xin - norm.x_mean[:, None, None]) / (norm.x_std[:, None, None] + 1e-8)
            xt  = torch.from_numpy(xn[None].astype(np.float32))
            yn  = model(xt)[0].numpy()                               # (5,H,W) normalized
            yhat = yn * norm.y_std[:, None, None] + norm.y_mean[:, None, None]
            for fi, flux in enumerate(FLUXES):
                pred  = yhat[fi]
                truth = Y_pick[k, fi]
                mask  = M_pick[k]
                r2 = masked_r2(pred, truth, mask)
                per_r2[key][flux].append(r2)
                render(X_pick[k, 0], pred, truth, mask, flux, label,
                       pick_dates[k], r2, ASSETS / f"{key}__{flux}__{k}.jpg")
        print(f"  done {label}")

# ----------------------------------------------------------------------------
# 5. Manifest
# ----------------------------------------------------------------------------
def test_scores(out_dir):
    p = ROOT / out_dir / "r2_scores_test.json"
    return json.loads(p.read_text()) if p.exists() else {}

manifest = {
    "fluxes": FLUXES,
    "units":  UNITS,
    "test_years": list(TEST_YEARS),
    "models": [
        {"key": key, "label": label, "color": color,
         "scores": test_scores(out_dir)}
        for key, label, out_dir, kind, dp, color in MODELS
    ],
    "picks": [{"k": k, "date": d} for k, d in enumerate(pick_dates)],
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"wrote {OUT/'manifest.json'}")
print("DONE. Open demo/index.html")
