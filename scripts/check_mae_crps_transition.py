#!/usr/bin/env python
"""
check_mae_crps_transition.py
============================
Diagnostic: did --resume (instead of --resume_weights_only) hurt the
Stage 1 (MAE) → Stage 2 (CRPS) fine-tune in output_unet_mem24h_mae?

Usage (run from the couple/ repo root on Glade):
    python scripts/check_mae_crps_transition.py

Outputs
-------
  papers/fig_mae_crps_transition_check.pdf / .png  — loss curve with boundary
  stdout                                           — verdict + auditable numbers
"""

import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
CKPT     = ROOT / "output" / "output_unet_mem24h_mae" / "checkpoint.pt"
LOG_S1   = ROOT / "output" / "logs" / "unet_mem24h_mae_s1.log"
LOG_S2   = ROOT / "output" / "logs" / "unet_mem24h_mae_s2.log"
OUT_PDF  = ROOT / "papers" / "fig_mae_crps_transition_check.pdf"
OUT_PNG  = ROOT / "papers" / "fig_mae_crps_transition_check.png"

TREND_WINDOW = 10   # epochs on each side for slope analysis

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def linear_slope(ys):
    """Least-squares slope of ys (1-indexed x)."""
    x = np.arange(1, len(ys) + 1, dtype=float)
    return float(np.polyfit(x, ys, 1)[0])


def epoch_deltas(ys):
    """Epoch-over-epoch differences."""
    return np.diff(ys)


# ---------------------------------------------------------------------------
# Step 1: locate the Stage 1 → Stage 2 boundary epoch
# ---------------------------------------------------------------------------

print("=" * 70)
print("STEP 1: locating Stage 1 → Stage 2 boundary epoch")
print("=" * 70)

boundary_epoch  = None
boundary_method = None

# Primary: look for the "Resumed at epoch N" line that Stage 2 prints
RESUME_RE = re.compile(r"Resumed at epoch\s+(\d+)/")
if LOG_S2.exists():
    for line in LOG_S2.read_text().splitlines():
        m = RESUME_RE.search(line)
        if m:
            boundary_epoch  = int(m.group(1))
            boundary_method = f"parsed 'Resumed at epoch' from {LOG_S2}"
            print(f"  Found in Stage-2 log: {line.strip()}")
            break
    if boundary_epoch is None:
        print(f"  Stage-2 log exists but 'Resumed at epoch' line not found in {LOG_S2}")
else:
    print(f"  Stage-2 log not found: {LOG_S2}")

# Fallback: last printed epoch in Stage-1 log
if boundary_epoch is None:
    EPOCH_RE = re.compile(r"Epoch\s+(\d+)/\d+")
    if LOG_S1.exists():
        last_ep = None
        for line in LOG_S1.read_text().splitlines():
            m = EPOCH_RE.search(line)
            if m:
                last_ep = int(m.group(1))
        if last_ep is not None:
            boundary_epoch  = last_ep + 1
            boundary_method = (f"fallback: Stage-1 final epoch {last_ep} from {LOG_S1}; "
                               f"boundary = {last_ep} + 1 = {boundary_epoch}")
            print(f"  Stage-1 final epoch from log: {last_ep}")
        else:
            print(f"  Stage-1 log exists but no 'Epoch N/...' lines found in {LOG_S1}")
    else:
        print(f"  Stage-1 log not found: {LOG_S1}")

if boundary_epoch is None:
    print("\nERROR: cannot determine boundary epoch — neither log file yielded "
          "usable information.  Cannot proceed.  Stopping.")
    sys.exit(1)

print(f"\n  boundary_epoch = {boundary_epoch}")
print(f"  method         = {boundary_method}")

# ---------------------------------------------------------------------------
# Step 2: load checkpoint history
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2: loading checkpoint history")
print("=" * 70)

if not CKPT.exists():
    print(f"ERROR: checkpoint not found at {CKPT}.  Stopping.")
    sys.exit(1)

ckpt    = torch.load(CKPT, map_location="cpu", weights_only=False)
history = ckpt["history"]   # list of (train_loss, val_loss) tuples
n_total = len(history)
best_val_in_ckpt = ckpt.get("best_val", float("nan"))
ckpt_epoch       = ckpt.get("epoch", "?")

print(f"  checkpoint epoch : {ckpt_epoch}")
print(f"  history length   : {n_total} epochs total")
print(f"  best_val in ckpt : {best_val_in_ckpt:.6f}")

trn_all = np.array([x[0] for x in history])
val_all = np.array([x[1] for x in history])

# Split (history is 0-indexed; epoch N maps to history[N-1])
boundary_idx = boundary_epoch - 1   # first Stage-2 entry in history[]

if boundary_idx <= 0 or boundary_idx >= n_total:
    print(f"\nERROR: boundary_idx={boundary_idx} is out of range for history "
          f"of length {n_total}.  boundary_epoch={boundary_epoch} is inconsistent "
          f"with the checkpoint.  Stopping.")
    sys.exit(1)

stage1_val = val_all[:boundary_idx]
stage2_val = val_all[boundary_idx:]
stage1_trn = trn_all[:boundary_idx]
stage2_trn = trn_all[boundary_idx:]

print(f"\n  Stage 1 epochs : 1 – {boundary_idx}  ({len(stage1_val)} epochs, MAE loss)")
print(f"  Stage 2 epochs : {boundary_epoch} – {n_total}  ({len(stage2_val)} epochs, CRPS loss)")

# ---------------------------------------------------------------------------
# Step 3: auditable numbers around the boundary
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3: val-loss values around the Stage 1 → Stage 2 boundary")
print("=" * 70)

WINDOW = 5
# Stage 1 tail
s1_start = max(0, boundary_idx - WINDOW)
s1_tail  = list(zip(
    range(s1_start + 1, boundary_idx + 1),
    trn_all[s1_start:boundary_idx],
    val_all[s1_start:boundary_idx],
))
# Stage 2 head
s2_end  = min(n_total, boundary_idx + WINDOW)
s2_head = list(zip(
    range(boundary_epoch, s2_end + 1),
    trn_all[boundary_idx:s2_end],
    val_all[boundary_idx:s2_end],
))

print(f"\n  {'Epoch':>6}  {'train':>10}  {'val':>10}  {'stage'}")
print(f"  {'------':>6}  {'----------':>10}  {'----------':>10}  {'-------'}")
for ep, t, v in s1_tail:
    print(f"  {ep:>6}  {t:>10.6f}  {v:>10.6f}  Stage 1 (MAE)")
print(f"  {'------':>6}  {'----------':>10}  {'----------':>10}  {'-------'}")
for ep, t, v in s2_head:
    print(f"  {ep:>6}  {t:>10.6f}  {v:>10.6f}  Stage 2 (CRPS)")

# ---------------------------------------------------------------------------
# Step 4: trend analysis
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4: trend analysis")
print("=" * 70)

tw = TREND_WINDOW

# (a) Last N epochs of Stage 1
s1_window = val_all[max(0, boundary_idx - tw) : boundary_idx]
# (b) First N epochs of Stage 2
s2_early  = val_all[boundary_idx : min(n_total, boundary_idx + tw)]
# (c) Remainder of Stage 2
s2_rest   = val_all[min(n_total, boundary_idx + tw):]

slope_s1_tail  = linear_slope(s1_window)  if len(s1_window)  >= 2 else float("nan")
slope_s2_early = linear_slope(s2_early)   if len(s2_early)   >= 2 else float("nan")
slope_s2_rest  = linear_slope(s2_rest)    if len(s2_rest)    >= 2 else float("nan")

deltas_s2_early = epoch_deltas(s2_early)

n_rising_early = int(np.sum(deltas_s2_early > 0)) if len(deltas_s2_early) > 0 else 0
n_consec_rise  = 0
for d in deltas_s2_early:
    if d > 0:
        n_consec_rise += 1
    else:
        break

print(f"\n  (a) Stage-1 tail  (last {len(s1_window)} epochs):")
print(f"      val range : {s1_window.min():.6f} – {s1_window.max():.6f}")
print(f"      slope     : {slope_s1_tail:+.6f} per epoch")

print(f"\n  (b) Stage-2 early (first {len(s2_early)} epochs):")
if len(s2_early):
    print(f"      val range : {s2_early.min():.6f} – {s2_early.max():.6f}")
    print(f"      slope     : {slope_s2_early:+.6f} per epoch")
    print(f"      rising epochs (val > prev): {n_rising_early}/{len(deltas_s2_early)}")
    print(f"      consecutive rising at start: {n_consec_rise}")
else:
    print("      (insufficient Stage-2 epochs for window)")

print(f"\n  (c) Stage-2 remainder ({len(s2_rest)} epochs):")
if len(s2_rest) >= 2:
    print(f"      val range : {s2_rest.min():.6f} – {s2_rest.max():.6f}")
    print(f"      slope     : {slope_s2_rest:+.6f} per epoch")
else:
    print("      (too few epochs for slope)")

# ---------------------------------------------------------------------------
# Step 5: verdict
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 5: verdict")
print("=" * 70)

# Classify the transition quality
drop_at_boundary = (val_all[boundary_idx] - val_all[boundary_idx - 1]
                    if boundary_idx < n_total else float("nan"))

problem_signals = []
clean_signals   = []

if n_consec_rise >= 3:
    problem_signals.append(
        f"{n_consec_rise} consecutive rising val-loss epochs right after resume "
        f"(suggests optimizer warm-up friction from carried-over moments)"
    )
if not np.isnan(slope_s2_early) and slope_s2_early > 0 and len(s2_early) >= 5:
    problem_signals.append(
        f"Stage-2 early slope is positive ({slope_s2_early:+.6f}/epoch), "
        f"indicating net val-loss increase over the first {len(s2_early)} epochs"
    )
if n_rising_early >= len(deltas_s2_early) * 0.6 and len(deltas_s2_early) >= 4:
    problem_signals.append(
        f"{n_rising_early}/{len(deltas_s2_early)} early Stage-2 epochs are rising "
        f"(majority non-improving)"
    )

if not np.isnan(drop_at_boundary) and drop_at_boundary < 0:
    clean_signals.append(
        f"val loss dropped at the first Stage-2 epoch "
        f"({val_all[boundary_idx-1]:.6f} → {val_all[boundary_idx]:.6f}); "
        f"note: a mechanical drop is expected because CRPS < MAE when spread≈0"
    )
if not np.isnan(slope_s2_early) and slope_s2_early < 0:
    clean_signals.append(
        f"Stage-2 early slope is negative ({slope_s2_early:+.6f}/epoch), "
        f"val loss continued improving"
    )
if n_consec_rise == 0 and len(deltas_s2_early) > 0:
    clean_signals.append(
        "no consecutive rising epochs at the start of Stage 2"
    )

print()
if problem_signals:
    verdict = "PROBLEM SIGNALS DETECTED"
    print(f"  Verdict: {verdict}")
    print(f"  boundary_epoch = {boundary_epoch}  (method: {boundary_method})")
    for s in problem_signals:
        print(f"  [!] {s}")
    for s in clean_signals:
        print(f"  [ok] {s}")
    print()
    print("  Interpretation: the carried-over optimizer state and/or best_val")
    print("  appear to have caused wasted epochs early in Stage 2.  Consider")
    print("  relaunching Stage 2 with --resume_weights_only.")
elif clean_signals:
    verdict = "TRANSITION LOOKS CLEAN"
    print(f"  Verdict: {verdict}")
    print(f"  boundary_epoch = {boundary_epoch}  (method: {boundary_method})")
    for s in clean_signals:
        print(f"  [ok] {s}")
    print()
    print("  Interpretation: val loss decreased (or stayed flat) through the")
    print("  boundary with no multi-epoch rising stretch.  The mechanical drop")
    print("  at epoch {} is expected (CRPS ≈ MAE when spread≈0 at Stage-2 start).".format(boundary_epoch))
    print("  No evidence that --resume hurt Stage 2.  Relaunching is not needed.")
else:
    verdict = "AMBIGUOUS — insufficient Stage-2 data"
    print(f"  Verdict: {verdict}")
    print(f"  boundary_epoch = {boundary_epoch}  (method: {boundary_method})")
    print(f"  Stage 2 has only {len(stage2_val)} epoch(s); need ≥5 to judge the trend.")

# ---------------------------------------------------------------------------
# Step 6: plot
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 6: writing plot")
print("=" * 70)

epochs = np.arange(1, n_total + 1)

fig, ax = plt.subplots(figsize=(9, 4))

ax.plot(epochs, trn_all, color="#9e9e9e", lw=1.2, label="train loss")
ax.plot(epochs, val_all, color="#4c72b0", lw=1.4, ls="--", label="val loss")

ax.axvline(boundary_epoch - 0.5, color="#c44e52", lw=1.5, ls="--", alpha=0.85,
           label=f"Stage 1 (MAE) → Stage 2 (CRPS)  [ep {boundary_epoch}]")

# best val marker
best_ep  = int(np.argmin(val_all)) + 1
best_val = val_all[best_ep - 1]
ax.scatter([best_ep], [best_val], color="#c44e52", zorder=5, s=40)
ax.text(best_ep + max(1, n_total * 0.01), best_val,
        f"best ep {best_ep}\n({best_val:.4f})", fontsize=7, va="center", color="#c44e52")

# annotate stage labels
mid_s1 = (1 + (boundary_epoch - 1)) / 2
mid_s2 = (boundary_epoch + n_total) / 2
ylims  = ax.get_ylim()
ypos   = ylims[1] - (ylims[1] - ylims[0]) * 0.05
ax.text(mid_s1, ypos, "Stage 1\nMAE", ha="center", va="top",
        fontsize=8, color="#555555", style="italic")
if len(stage2_val) > 0:
    ax.text(mid_s2, ypos, "Stage 2\nCRPS", ha="center", va="top",
            fontsize=8, color="#555555", style="italic")

ax.set_xlabel("Epoch", fontsize=9)
ax.set_ylabel("Loss (normalised)", fontsize=9)
ax.set_title(
    f"MAE → CRPS transition check  |  {verdict}\n"
    f"boundary ep {boundary_epoch}  •  best val ep {best_ep} ({best_val:.4f})",
    fontsize=9,
)
ax.legend(fontsize=8, frameon=False)
ax.yaxis.grid(True, linestyle="--", alpha=0.35)
ax.set_axisbelow(True)
ax.tick_params(labelsize=8)

fig.tight_layout()
for path in (OUT_PDF, OUT_PNG):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  wrote {path}")

print("\nDone.")
