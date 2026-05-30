"""
plot_lag_results.py
===================
Produce two clean figures for PI presentation:

  Figure 1 — R² bar chart comparing all 6 experiments
  Figure 2 — Time-mean validation maps (truth / prediction / bias)
             for the best model (lag=0h, no CO2)

Output: results/r2_summary.png  and  results/val_maps_<exp>.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from train_unet import UNet, Normalizer, PairDataset, H, W

# ---------------------------------------------------------------------------
CACHE_BASE  = Path("/glade/work/praggarwal/couple_cache")
COUPLE_DIR  = Path("/glade/u/home/praggarwal/couple")
RESULTS_DIR = COUPLE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EXPERIMENTS = [
    {"label": "lag=0h",              "out_dir": "output_unet_lag0h",           "cache": "couple_cache_lag0h",    "co2": False},
    {"label": "lag=12h",             "out_dir": "output_unet_lag12h",          "cache": "couple_cache_lag12h",   "co2": False},
    {"label": "lag=24h",             "out_dir": "output_unet_lag24h",          "cache": "couple_cache_lag24h",   "co2": False},
    {"label": "lag=0h+CO2",          "out_dir": "output_unet_lag0h_co2",       "cache": "couple_cache_lag0h",    "co2": True},
    {"label": "lag=12h+CO2",         "out_dir": "output_unet_lag12h_co2",      "cache": "couple_cache_lag12h",   "co2": True},
    {"label": "lag=24h+CO2",         "out_dir": "output_unet_lag24h_co2",      "cache": "couple_cache_lag24h",   "co2": True},
    {"label": "mem24h",              "out_dir": "output_unet_mem24h",          "cache": "couple_cache_mem24h",   "co2": False},
    {"label": "mem24h+solin",        "out_dir": "output_unet_mem24h_solin",    "cache": "couple_cache_mem24h",   "co2": False},
    {"label": "mem24h+CO2",          "out_dir": "output_unet_mem24h_co2",      "cache": "couple_cache_mem24h",   "co2": True},
    {"label": "mem24h+solin+CO2",    "out_dir": "output_unet_mem24h_solin_co2","cache": "couple_cache_mem24h",   "co2": True},
]

# Per-experiment colors: blues=lag no-CO2, reds=lag+CO2, greens=mem no-CO2, purples=mem+CO2
EXP_COLOR_MAP = {
    "lag=0h":           "#2166ac",
    "lag=12h":          "#4393c3",
    "lag=24h":          "#92c5de",
    "lag=0h+CO2":       "#b2182b",
    "lag=12h+CO2":      "#d6604d",
    "lag=24h+CO2":      "#f4a582",
    "mem24h":           "#1a9641",
    "mem24h+solin":     "#a6d96a",
    "mem24h+CO2":       "#762a83",
    "mem24h+solin+CO2": "#c2a5cf",
}

VAR_LABELS = {
    "TAUX":  "Zonal\nWind Stress",
    "TAUY":  "Meridional\nWind Stress",
    "SHFLX": "Sensible\nHeat Flux",
    "LHFLX": "Latent\nHeat Flux",
    "QFLX":  "Water\nFlux",
}
UNITS = {
    "TAUX":  "N m⁻²",
    "TAUY":  "N m⁻²",
    "SHFLX": "W m⁻²",
    "LHFLX": "W m⁻²",
    "QFLX":  "kg m⁻² s⁻¹",
}

# SHFLX and LHFLX are stored as cumulative J m⁻² per 6h step; divide to get W m⁻²
UNIT_SCALE = {
    "TAUX":  1.0,
    "TAUY":  1.0,
    "SHFLX": 1.0 / 21600.0,
    "LHFLX": 1.0 / 21600.0,
    "QFLX":  1.0,
}

BATCH   = 8
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Channel order in the memory cache (6 channels)
_MEM_CACHE_VARS = ["SST", "ICEFRAC", "SOLIN", "SST_prev", "ICEFRAC_prev", "SOLIN_prev"]
_MEM_CACHE_IDX  = {v: i for i, v in enumerate(_MEM_CACHE_VARS)}


class MmapValDataset(torch.utils.data.Dataset):
    """Streams val samples directly from mmap files — no full-dataset copy in RAM."""
    def __init__(self, X_mm, Y_mm, mask_mm, val_indices, norm, co2_mm=None, mem_channels=None):
        self.X_mm        = X_mm
        self.Y_mm        = Y_mm
        self.mask_mm     = mask_mm
        self.vi          = val_indices
        self.norm        = norm
        self.co2_mm      = co2_mm
        self.mem_channels = mem_channels  # list of channel indices to select from X_mm, or None

    def __len__(self):
        return len(self.vi)

    def __getitem__(self, i):
        idx = self.vi[i]
        x = self.X_mm[idx].copy().astype(np.float32)
        if self.mem_channels is not None:
            x = x[self.mem_channels]
        if self.co2_mm is not None:
            x = np.concatenate(
                [x, np.full((1, H, W), float(self.co2_mm[idx]), dtype=np.float32)],
                axis=0,
            )
        y    = self.Y_mm[idx].copy().astype(np.float32)
        mask = self.mask_mm[idx].copy().astype(np.float32)
        x_n = (x - self.norm.x_mean[:, None, None]) / (self.norm.x_std[:, None, None] + 1e-8)
        y_n = (y - self.norm.y_mean[:, None, None]) / (self.norm.y_std[:, None, None] + 1e-8)
        return torch.from_numpy(x_n), torch.from_numpy(y_n), torch.from_numpy(mask)


# ---------------------------------------------------------------------------
def load_val_data(exp):
    """Return a streaming val dataset — no full-dataset copy in RAM."""
    cache_dir = Path("/glade/work/praggarwal") / exp["cache"]
    out_dir   = COUPLE_DIR / exp["out_dir"]

    X_mm    = np.load(cache_dir / "X.npy",    mmap_mode="r")
    Y_mm    = np.load(cache_dir / "Y.npy",    mmap_mode="r")
    mask_mm = np.load(cache_dir / "mask.npy", mmap_mode="r")
    co2_mm  = np.load(cache_dir / "co2.npy",  mmap_mode="r") if exp["co2"] else None
    N_full  = len(X_mm)

    # Reproduce exact train/val split from training (seed 42)
    rng    = np.random.default_rng(42)
    chosen = rng.choice(N_full, size=N_full, replace=False)
    idx    = rng.permutation(N_full)
    n_val  = max(1, N_full // 10)
    vi     = chosen[idx[:n_val]]   # val indices into the original mmap

    norm     = Normalizer.load(out_dir / "normalizer.npz")
    cfg      = json.load(open(out_dir / "model_config.json"))
    tgt_vars = cfg["output_vars"]

    # For memory experiments the cache has 6 channels; select only what this model used
    base_vars = [v for v in cfg["input_vars"] if v != "CO2"]
    mem_channels = (
        [_MEM_CACHE_IDX[v] for v in base_vars]
        if any(v in _MEM_CACHE_IDX for v in base_vars) and len(base_vars) < X_mm.shape[1]
        else None
    )

    val_ds = MmapValDataset(X_mm, Y_mm, mask_mm, vi, norm, co2_mm, mem_channels)
    return val_ds, norm, tgt_vars, cfg


def run_inference(exp):
    """Return (truth_mean, pred_mean, bias_mean) each (n_out, H, W), NaN over land."""
    out_dir = COUPLE_DIR / exp["out_dir"]
    val_ds, norm, tgt_vars, cfg = load_val_data(exp)

    model = UNet(n_in=cfg["n_in"], n_out=cfg["n_out"], base=cfg["base"]).to(DEVICE)
    model.load_state_dict(
        torch.load(out_dir / "best_model.pt", map_location=DEVICE, weights_only=True)
    )
    model.eval()

    y_mean_t = torch.tensor(norm.y_mean, device=DEVICE)
    y_std_t  = torch.tensor(norm.y_std,  device=DEVICE)
    y_mean_c = torch.tensor(norm.y_mean)
    y_std_c  = torch.tensor(norm.y_std)

    loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    sum_true = np.zeros((cfg["n_out"], H, W), dtype=np.float64)
    sum_pred = np.zeros((cfg["n_out"], H, W), dtype=np.float64)
    count    = np.zeros((H, W),                dtype=np.float64)

    with torch.no_grad():
        for x_n, y_n, mask in loader:
            pred_n = model(x_n.to(DEVICE))
            pred = (pred_n * y_std_t[None,:,None,None] + y_mean_t[None,:,None,None]).cpu().numpy()
            true = (y_n    * y_std_c[None,:,None,None] + y_mean_c[None,:,None,None]).numpy()
            m = mask.numpy() > 0.5
            for b in range(pred.shape[0]):
                sum_true[:, m[b]] += true[b][:, m[b]]
                sum_pred[:, m[b]] += pred[b][:, m[b]]
                count[m[b]]       += 1

    # Free val dataset memory before returning
    del val_ds

    cnt = np.where(count > 0, count, np.nan)
    truth_mean = sum_true / cnt[None]
    pred_mean  = sum_pred / cnt[None]
    bias_mean  = pred_mean - truth_mean

    # Convert cumulative fluxes to rates (W m⁻²)
    scales = np.array([UNIT_SCALE[v] for v in tgt_vars], dtype=np.float64)
    truth_mean *= scales[:, None, None]
    pred_mean  *= scales[:, None, None]
    bias_mean  *= scales[:, None, None]

    return truth_mean, pred_mean, bias_mean, tgt_vars


# ---------------------------------------------------------------------------
# Figure 1: R² summary bar chart
# ---------------------------------------------------------------------------
def plot_r2_summary():
    print("Building R² summary chart ...")
    r2_data = {}
    for exp in EXPERIMENTS:
        out_dir = COUPLE_DIR / exp["out_dir"]
        r2_path = out_dir / "r2_scores.json"
        if not r2_path.exists():
            print(f"  Skipping {exp['label']} — no r2_scores.json")
            continue
        r2_data[exp["label"]] = json.load(open(r2_path))

    if not r2_data:
        print("No R² scores found.")
        return

    tgt_vars = list(next(iter(r2_data.values())).keys())
    n_vars   = len(tgt_vars)
    labels   = list(r2_data.keys())
    n_exp    = len(labels)

    bar_colors = [EXP_COLOR_MAP.get(lbl, "#888888") for lbl in labels]

    # One subplot per variable — horizontal bars avoid label collision
    fig, axes = plt.subplots(1, n_vars, figsize=(3.2 * n_vars, 0.55 * n_exp + 2.5),
                             sharey=True, constrained_layout=True)

    height = 0.55
    y      = np.arange(n_exp)

    for col, vname in enumerate(tgt_vars):
        ax   = axes[col]
        vals = [r2_data[lbl][vname] for lbl in labels]
        bars = ax.barh(y, vals, height, color=bar_colors,
                       edgecolor="white", linewidth=0.5)
        ax.set_xlim(0.55, 1.0)
        ax.set_title(f"{VAR_LABELS[vname]}\n({UNITS[vname]})",
                     fontsize=9, fontweight="bold")
        ax.axvline(1.0, color="k", lw=0.5, ls="--", alpha=0.3)
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("R²", fontsize=9)
        for bar, val in zip(bars, vals):
            ax.text(val - 0.004, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="right",
                    fontsize=7.5, color="white", fontweight="bold")
        if col == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=9)
        else:
            ax.set_yticks([])

    fig.suptitle("R² across lag, memory, and CO2 experiments\n"
                 "(validation set, ocean/ice only)",
                 fontsize=12, fontweight="bold")

    # Legend: one patch per experiment
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=bar_colors[i], label=labels[i]) for i in range(n_exp)]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=n_exp, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.08))

    out = RESULTS_DIR / "r2_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: Spatial maps for one experiment
# ---------------------------------------------------------------------------
def plot_val_maps(exp_label="lag=0h"):
    exp = next(e for e in EXPERIMENTS if e["label"] == exp_label)
    out_dir = COUPLE_DIR / exp["out_dir"]
    if not (out_dir / "best_model.pt").exists():
        print(f"Skipping {exp_label} — no best_model.pt")
        return

    print(f"Computing val maps for {exp_label} ...")
    truth_mean, pred_mean, bias_mean, tgt_vars = run_inference(exp)
    n_out = len(tgt_vars)

    fig, axes = plt.subplots(3, n_out, figsize=(3.5 * n_out, 8),
                             constrained_layout=True)

    r2 = json.load(open(out_dir / "r2_scores.json"))

    row_labels = ["Truth (val mean)", "Prediction (val mean)", "Bias (pred − truth)"]
    cmaps      = ["viridis", "viridis", "RdBu_r"]

    for col, vname in enumerate(tgt_vars):
        rows_data = [truth_mean[col], pred_mean[col], bias_mean[col]]

        # Shared color limits for truth/pred rows; symmetric for bias
        vmax_tp  = np.nanpercentile(np.abs(truth_mean[col]), 98)
        bias_abs = np.nanpercentile(np.abs(bias_mean[col]), 98)

        vlims = [(-vmax_tp, vmax_tp), (-vmax_tp, vmax_tp), (-bias_abs, bias_abs)]

        for row in range(3):
            ax   = axes[row, col]
            data = rows_data[row]
            vmin, vmax = vlims[row]
            im = ax.imshow(data, origin="lower", aspect="auto",
                           cmap=cmaps[row], vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            ax.set_facecolor("#cccccc")  # gray for land
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                         label=UNITS.get(vname, ""))
            if row == 0:
                ax.set_title(f"{vname}\nR²={r2[vname]:.3f}", fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)

    fig.suptitle(f"Validation maps — {exp_label}  (time-mean over val set)",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, -0.01,
             "Note: SHFLX and LHFLX are stored as cumulative energy (J m⁻²) per 6-hour step "
             "in the CESM output and have been divided by 21,600 s to convert to W m⁻².",
             ha="center", fontsize=7.5, color="gray", style="italic")

    out = RESULTS_DIR / f"val_maps_{exp_label.replace('=','').replace('+','_')}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figures 1–5: Cross-experiment summary panels
# ---------------------------------------------------------------------------

_FLUX_NOTE = (
    "SHFLX and LHFLX converted from cumulative J m⁻² (6-hour step) to W m⁻² "
    "by dividing by 21,600 s."
)


def _add_flux_note(fig):
    fig.text(0.5, -0.01, _FLUX_NOTE, ha="center", fontsize=7, color="gray", style="italic")


def _collect_all_inference():
    """Run inference for every experiment that has a trained model."""
    results  = {}
    tgt_vars = None
    for exp in EXPERIMENTS:
        out_dir = COUPLE_DIR / exp["out_dir"]
        if not (out_dir / "best_model.pt").exists():
            print(f"  Skipping {exp['label']} — no model")
            continue
        print(f"  Inference: {exp['label']}")
        truth, pred, bias, tvars = run_inference(exp)
        results[exp["label"]] = {"truth": truth, "pred": pred, "bias": bias}
        if tgt_vars is None:
            tgt_vars = tvars
    return results, tgt_vars


def _load_r2_all():
    r2_all = {}
    for exp in EXPERIMENTS:
        p = COUPLE_DIR / exp["out_dir"] / "r2_scores.json"
        if p.exists():
            r2_all[exp["label"]] = json.load(open(p))
    return r2_all


def _panel(ax, data, cmap, vmin, vmax, unit="", title="", row_label=""):
    im = ax.imshow(data, origin="lower", aspect="auto",
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_facecolor("#cccccc")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=unit)
    if title:
        ax.set_title(title, fontsize=8.5, fontweight="bold")
    if row_label:
        ax.set_ylabel(row_label, fontsize=9)
    return im


def plot_fig1_truth(results, tgt_vars):
    """Fig 1 — Ground truth time-mean (one reference run), 1 row × 5 vars."""
    print("Fig 1: ground truth ...")
    ref = results[next(iter(results))]["truth"]
    n_vars = len(tgt_vars)
    fig, axes = plt.subplots(1, n_vars, figsize=(3.5 * n_vars, 3.2), constrained_layout=True)
    for col, vname in enumerate(tgt_vars):
        data = ref[col]
        vmax = np.nanpercentile(np.abs(data), 98)
        _panel(axes[col], data, "RdBu_r", -vmax, vmax,
               unit=UNITS[vname],
               title=f"{VAR_LABELS[vname]}\n({UNITS[vname]})",
               row_label="Ground truth\n(val mean)" if col == 0 else "")
    fig.suptitle("Ground truth — time-mean over validation set", fontsize=12, fontweight="bold")
    _add_flux_note(fig)
    out = RESULTS_DIR / "fig1_truth.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_fig2_mean_std(results, tgt_vars):
    """Fig 2 — Mean prediction and inter-run std, 2 rows × 5 vars."""
    print("Fig 2: mean/std of predictions ...")
    labels = list(results.keys())
    n_vars = len(tgt_vars)
    preds       = np.stack([results[l]["pred"] for l in labels], axis=0)  # (n_exp, 5, H, W)
    pred_mean   = np.nanmean(preds, axis=0)
    pred_std    = np.nanstd(preds,  axis=0)
    fig, axes = plt.subplots(2, n_vars, figsize=(3.5 * n_vars, 6), constrained_layout=True)
    for col, vname in enumerate(tgt_vars):
        vmax     = np.nanpercentile(np.abs(pred_mean[col]), 98)
        vmax_std = np.nanpercentile(pred_std[col], 98)
        _panel(axes[0, col], pred_mean[col], "RdBu_r", -vmax, vmax,
               unit=UNITS[vname],
               title=f"{VAR_LABELS[vname]}\n({UNITS[vname]})",
               row_label="Mean prediction\n(across all runs)" if col == 0 else "")
        _panel(axes[1, col], pred_std[col], "plasma", 0, vmax_std,
               unit=UNITS[vname],
               row_label="Inter-run std\n(prediction)" if col == 0 else "")
    fig.suptitle(f"Mean and spread of predictions across {len(labels)} experiments",
                 fontsize=12, fontweight="bold")
    _add_flux_note(fig)
    out = RESULTS_DIR / "fig2_mean_std.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_fig3_mean_abs_error(results, tgt_vars):
    """Fig 3 — Mean |bias| averaged across all runs, 1 row × 5 vars."""
    print("Fig 3: mean absolute error ...")
    labels = list(results.keys())
    n_vars = len(tgt_vars)
    abs_biases    = np.stack([np.abs(results[l]["bias"]) for l in labels], axis=0)
    mean_abs_bias = np.nanmean(abs_biases, axis=0)
    fig, axes = plt.subplots(1, n_vars, figsize=(3.5 * n_vars, 3.2), constrained_layout=True)
    for col, vname in enumerate(tgt_vars):
        data = mean_abs_bias[col]
        vmax = np.nanpercentile(data, 98)
        _panel(axes[col], data, "YlOrRd", 0, vmax,
               unit=UNITS[vname],
               title=f"{VAR_LABELS[vname]}\n({UNITS[vname]})",
               row_label="Mean |bias|\n(across all runs)" if col == 0 else "")
    fig.suptitle(f"Mean absolute bias across all {len(labels)} experiments",
                 fontsize=12, fontweight="bold")
    _add_flux_note(fig)
    out = RESULTS_DIR / "fig3_mean_abs_error.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_fig4_best_worst(results, tgt_vars, r2_all):
    """Fig 4 — Best/worst run per variable: 4 rows × 5 vars.
       Rows: best prediction | best bias | worst prediction | worst bias.
    """
    print("Fig 4: best vs worst per variable ...")
    n_vars = len(tgt_vars)
    row_labels_left = [
        "Best run\nPrediction",
        "Best run\nBias (pred−truth)",
        "Worst run\nPrediction",
        "Worst run\nBias (pred−truth)",
    ]

    # Per-variable best/worst by R²
    best_lbl, worst_lbl = [], []
    for vname in tgt_vars:
        scores = {l: r2_all[l][vname] for l in results if l in r2_all}
        best_lbl.append(max(scores, key=scores.get))
        worst_lbl.append(min(scores, key=scores.get))

    fig, axes = plt.subplots(4, n_vars, figsize=(3.5 * n_vars, 11), constrained_layout=True)

    for col, vname in enumerate(tgt_vars):
        bl, wl = best_lbl[col], worst_lbl[col]
        r2b    = r2_all[bl][vname]
        r2w    = r2_all[wl][vname]

        # Consistent scales for pred and bias within this variable
        ref_truth = results[bl]["truth"][col]
        vmax_tp   = np.nanpercentile(np.abs(ref_truth), 98)
        bias_abs  = max(
            np.nanpercentile(np.abs(results[bl]["bias"][col]),  98),
            np.nanpercentile(np.abs(results[wl]["bias"][col]), 98),
        )

        panel_cfg = [
            (results[bl]["pred"][col],  "RdBu_r", -vmax_tp,  vmax_tp),
            (results[bl]["bias"][col],  "RdBu_r", -bias_abs, bias_abs),
            (results[wl]["pred"][col],  "RdBu_r", -vmax_tp,  vmax_tp),
            (results[wl]["bias"][col],  "RdBu_r", -bias_abs, bias_abs),
        ]
        col_titles = [
            f"{VAR_LABELS[vname]}\n({UNITS[vname]})\n{bl}  R²={r2b:.3f}",
            "",
            f"{wl}  R²={r2w:.3f}",
            "",
        ]

        for row, (data, cmap, vmin, vmax) in enumerate(panel_cfg):
            _panel(axes[row, col], data, cmap, vmin, vmax,
                   unit=UNITS[vname],
                   title=col_titles[row],
                   row_label=row_labels_left[row] if col == 0 else "")

    fig.suptitle("Best vs worst run per variable (ranked by R²)",
                 fontsize=12, fontweight="bold")
    _add_flux_note(fig)
    out = RESULTS_DIR / "fig4_best_worst.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_fig5_best_overall(results, tgt_vars, r2_all):
    """Fig 5 — Bias maps for the best overall run (highest mean R²), 1 row × 5 vars."""
    print("Fig 5: best overall run ...")
    n_vars = len(tgt_vars)
    mean_r2      = {l: np.mean(list(r2_all[l].values())) for l in results if l in r2_all}
    best_overall = max(mean_r2, key=mean_r2.get)
    mean_r2_val  = mean_r2[best_overall]
    print(f"  Best overall: {best_overall}  (mean R²={mean_r2_val:.3f})")

    fig, axes = plt.subplots(1, n_vars, figsize=(3.5 * n_vars, 3.2), constrained_layout=True)
    for col, vname in enumerate(tgt_vars):
        data = results[best_overall]["bias"][col]
        vmax = np.nanpercentile(np.abs(data), 98)
        r2v  = r2_all[best_overall][vname]
        _panel(axes[col], data, "RdBu_r", -vmax, vmax,
               unit=UNITS[vname],
               title=f"{VAR_LABELS[vname]}\n({UNITS[vname]})\nR²={r2v:.3f}",
               row_label="Bias (pred−truth)" if col == 0 else "")
    fig.suptitle(f"Best overall run: {best_overall}  (mean R²={mean_r2_val:.3f})",
                 fontsize=12, fontweight="bold")
    _add_flux_note(fig)
    out = RESULTS_DIR / "fig5_best_overall_error.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_summary_figures():
    """Orchestrate Figs 1–5."""
    print("=== Collecting inference for all experiments ===")
    results, tgt_vars = _collect_all_inference()
    r2_all = _load_r2_all()
    plot_fig1_truth(results, tgt_vars)
    plot_fig2_mean_std(results, tgt_vars)
    plot_fig3_mean_abs_error(results, tgt_vars)
    plot_fig4_best_worst(results, tgt_vars, r2_all)
    plot_fig5_best_overall(results, tgt_vars, r2_all)
    print("\nAll summary figures saved to", RESULTS_DIR)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Which experiment labels to plot (default: all). "
                             "Example: --labels 'lag=0h+CO2' 'lag=12h+CO2' 'lag=24h+CO2'")
    parser.add_argument("--skip_r2", action="store_true",
                        help="Skip regenerating the R² summary chart.")
    parser.add_argument("--summary", action="store_true",
                        help="Generate the 5 cross-experiment summary figures (Figs 1–5).")
    cli = parser.parse_args()

    if not cli.skip_r2:
        plot_r2_summary()

    if cli.summary:
        plot_summary_figures()
    else:
        run_labels = cli.labels or [e["label"] for e in EXPERIMENTS]
        for exp in EXPERIMENTS:
            if exp["label"] in run_labels:
                plot_val_maps(exp["label"])
    print("\nDone.")
