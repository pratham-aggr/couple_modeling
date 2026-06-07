#!/usr/bin/env python
"""2×2 panel plot of training curves (train vs val MSE loss) for the four key
temporal-split models, illustrating the progression from severe overfitting
(memory-free) to a well-regularised run (dropout).

Output: papers/fig_training_curves.{pdf,png}
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    ("(a) Memory-free (lag 0)",  "output_unet_lag0h_temporal",
     "#9e9e9e", "#555555"),
    ("(b) MEMO (mem24h)",         "output_unet_mem24h_temporal",
     "#4c72b0", "#1a3a6e"),
    (r"(c) + $dT_s/dt$",          "output_unet_mem24h_dsst_temporal",
     "#55a868", "#2a6637"),
    ("(d) + dropout ($p=0.1$)",   "output_unet_mem24h_dsst_temporal_drop",
     "#c44e52", "#7a1a1e"),
]


def load_history(out_dir):
    ckpt_path = ROOT / out_dir / "checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    h = ckpt["history"]
    trn = [x[0] for x in h]
    val = [x[1] for x in h]
    return trn, val


fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=False)
axes = axes.flat

for ax, (title, out_dir, c_trn, c_val) in zip(axes, MODELS):
    trn, val = load_history(out_dir)
    epochs = np.arange(1, len(trn) + 1)

    # find best-val epoch
    best_ep = int(np.argmin(val)) + 1

    ax.plot(epochs, trn, color=c_trn, lw=1.2, label="train")
    ax.plot(epochs, val, color=c_val, lw=1.2, ls="--", label="val")
    ax.axvline(best_ep, color="black", lw=0.7, ls=":", alpha=0.6)
    ax.text(best_ep + max(1, len(epochs) * 0.02),
            ax.get_ylim()[1] if ax.get_ylim()[1] > 0.1 else 0.3,
            f"best\nep {best_ep}", fontsize=7, va="top", color="black")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("MSE loss", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False)

fig.suptitle("Training and validation loss — temporal split (1980–2010 train)",
             fontsize=10, y=1.01)
fig.tight_layout()

for ext in ("pdf", "png"):
    out = ROOT / "papers" / f"fig_training_curves.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
