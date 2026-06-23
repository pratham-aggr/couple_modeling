"""
diag_icefrac_p5.py — Same ICEFRAC-stratified flux diagnostic as diag_icefrac.py
but using p5ge0uvh (CRPS probabilistic model) with 40 MC-dropout ensemble mean.

Hard-codes vfcg1m61 ratios from the previous run for direct comparison.
"""

import glob, sys
import numpy as np
import xarray as xr
import torch
from pathlib import Path
sys.path.insert(0, str(Path("/glade/u/home/praggarwal/couple")))
from train_unet import UNet

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR  = Path("/glade/u/home/praggarwal/couple/output/output_unet_mem24h_dsst_temporal_radprecip_ucast")
ZARR_GLOB  = "/glade/derecho/scratch/wchapman/b_credit_runs/b.e21.CREDIT_climate_branch_1980_*.zarr"
J_TO_W     = 1.0 / 21600.0
MEMORY_LAG = 4
N_MEMBERS  = 40

# ── load model with MC-dropout ─────────────────────────────────────────────────
import json
cfg   = json.loads((MODEL_DIR / "model_config.json").read_text())
model = UNet(n_in=cfg["n_in"], n_out=cfg["n_out"], base=cfg["base"], dropout=cfg.get("dropout", 0.1))
ckpt  = torch.load(MODEL_DIR / "best_model.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt.get("model_state_dict", ckpt))

# enable MC-dropout: eval mode (freeze BN) but re-enable dropout layers
model.eval()
for m in model.modules():
    if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
        m.train()

norm   = np.load(MODEL_DIR / "normalizer.npz")
x_mean = norm["x_mean"].astype(np.float32)
x_std  = norm["x_std"].astype(np.float32)
y_mean = norm["y_mean"].astype(np.float32)
y_std  = norm["y_std"].astype(np.float32)

OUT_VARS = cfg["output_vars"]

def run_memo_ens_mean(x_np):
    """x_np: (6,192,288) → ensemble mean (8,192,288) in physical units."""
    xn = (x_np - x_mean[:, None, None]) / (x_std[:, None, None] + 1e-8)
    xt = torch.tensor(xn[None]).float()
    members = []
    with torch.no_grad():
        for _ in range(N_MEMBERS):
            yn = model(xt)[0].numpy()
            members.append(yn)
    ens_mean = np.stack(members, 0).mean(0)   # (8,192,288) normalised
    return ens_mean * y_std[:, None, None] + y_mean[:, None, None]

# ── zarr setup ─────────────────────────────────────────────────────────────────
zarr_paths = sorted(glob.glob(ZARR_GLOB))
print(f"Found {len(zarr_paths)} zarr files")
YEARS_USE   = zarr_paths[:3]
STEP_STRIDE = 28

BINS       = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.001]
BIN_LABELS = ["0.0–0.1","0.1–0.3","0.3–0.5","0.5–0.7","0.7–0.9","0.9–1.0"]
N_BINS     = len(BINS) - 1

counts        = np.zeros(N_BINS, dtype=np.int64)
sum_tau_truth = np.zeros(N_BINS)
sum_tau_memo  = np.zeros(N_BINS)
sum_shf_truth = np.zeros(N_BINS)
sum_shf_memo  = np.zeros(N_BINS)
sum_lhf_truth = np.zeros(N_BINS)
sum_lhf_memo  = np.zeros(N_BINS)

total_steps = 0
for zp in YEARS_USE:
    ds     = xr.open_zarr(zp, consolidated=False)
    n_time = ds.sizes.get("time", ds.sizes.get("Time", 0))
    steps  = list(range(MEMORY_LAG, n_time, STEP_STRIDE))
    print(f"\n{zp.split('/')[-1]}: {n_time} steps, sampling {len(steps)} steps", flush=True)

    for t in steps:
        sst_now   = ds["SST"].isel(time=t).values.astype(np.float32)
        icef_now  = ds["ICEFRAC"].isel(time=t).values.astype(np.float32)
        solin     = ds["SOLIN"].isel(time=t).values.astype(np.float32)
        sst_prev  = ds["SST"].isel(time=max(0, t - MEMORY_LAG)).values.astype(np.float32)
        icef_prev = ds["ICEFRAC"].isel(time=max(0, t - MEMORY_LAG)).values.astype(np.float32)
        dsst_dt   = (sst_now - sst_prev) / (MEMORY_LAG * 6 * 3600.0)

        x      = np.stack([sst_now, icef_now, solin, sst_prev, icef_prev, dsst_dt], axis=0)
        y_memo = run_memo_ens_mean(x)   # (8,192,288)

        taux_truth = ds["TAUX"].isel(time=t).values.astype(np.float32)
        tauy_truth = ds["TAUY"].isel(time=t).values.astype(np.float32)
        shf_truth  = ds["SHFLX"].isel(time=t).values.astype(np.float32) * J_TO_W
        lhf_truth  = ds["LHFLX"].isel(time=t).values.astype(np.float32) * J_TO_W

        taux_memo = y_memo[OUT_VARS.index("TAUX")]
        tauy_memo = y_memo[OUT_VARS.index("TAUY")]
        shf_memo  = y_memo[OUT_VARS.index("SHFLX")] * J_TO_W
        lhf_memo  = y_memo[OUT_VARS.index("LHFLX")] * J_TO_W

        tau_truth_mag = np.sqrt(taux_truth**2 + tauy_truth**2)
        tau_memo_mag  = np.sqrt(taux_memo**2  + tauy_memo**2)

        icef_flat = icef_now.ravel()
        for b in range(N_BINS):
            mask = (icef_flat >= BINS[b]) & (icef_flat < BINS[b+1])
            if not mask.any():
                continue
            counts[b]        += mask.sum()
            sum_tau_truth[b] += tau_truth_mag.ravel()[mask].sum()
            sum_tau_memo[b]  += tau_memo_mag.ravel()[mask].sum()
            sum_shf_truth[b] += np.abs(shf_truth.ravel()[mask]).sum()
            sum_shf_memo[b]  += np.abs(shf_memo.ravel()[mask]).sum()
            sum_lhf_truth[b] += np.abs(lhf_truth.ravel()[mask]).sum()
            sum_lhf_memo[b]  += np.abs(lhf_memo.ravel()[mask]).sum()

        total_steps += 1
        if total_steps % 10 == 0:
            print(f"  ... {total_steps} steps processed", flush=True)

print(f"\nTotal steps: {total_steps}  Total cells: {counts.sum():,}")

# ── vfcg1m61 ratios from previous run (hardcoded for comparison) ───────────────
# Source: diag_icefrac.log
vfcg_tau_ratio = [0.897, 0.876, 0.875, 0.871, 0.859, 0.823]
vfcg_shf_ratio = [0.931, 0.839, 0.842, 0.821, 0.769, 0.595]
vfcg_lhf_ratio = [0.995, 0.925, 0.921, 0.924, 0.921, 1.070]

# raw vfcg means for full-ice bin (0.9-1.0) for hand-verification
vfcg_tau_fullice = 0.0864
vfcg_shf_fullice = 6.905
vfcg_lhf_fullice = 4.167

# ── print results ──────────────────────────────────────────────────────────────
print()
print("=" * 110)
print("p5ge0uvh (CRPS, 40-member ens mean) vs Truth — ICEFRAC-stratified, with vfcg1m61 comparison")
print("=" * 110)

# Header
print(f"{'ICEFRAC bin':12s}  {'N cells':>10s}  "
      f"{'ratio_truth_p5':>14s}  {'ratio_truth_vfcg':>16s}  {'ratio_p5_vs_vfcg':>17s}  {'verdict':>12s}"
      f"   (var: |τ|)")
print("-" * 110)

for b in range(N_BINS):
    if counts[b] == 0:
        continue
    n    = counts[b]
    tau_t = sum_tau_truth[b] / n
    tau_m = sum_tau_memo[b]  / n
    r_p5  = tau_m / max(tau_t, 1e-9)
    r_vf  = vfcg_tau_ratio[b]
    r_cmp = tau_m / max(vfcg_tau_ratio[b] * tau_t, 1e-9)   # p5 / vfcg absolute
    # simpler: ratio of MEMO magnitudes
    r_cmp2 = r_p5 / r_vf
    if r_cmp2 > 1.05:   verdict = "p5 better"
    elif r_cmp2 < 0.95: verdict = "p5 worse"
    else:               verdict = "similar"
    print(f"{BIN_LABELS[b]:12s}  {n:>10,}  "
          f"{r_p5:>14.3f}  {r_vf:>16.3f}  {r_cmp2:>17.3f}  {verdict:>12s}")

print()
print(f"{'ICEFRAC bin':12s}  {'N cells':>10s}  "
      f"{'ratio_truth_p5':>14s}  {'ratio_truth_vfcg':>16s}  {'ratio_p5_vs_vfcg':>17s}  {'verdict':>12s}"
      f"   (var: |SHF|)")
print("-" * 110)

for b in range(N_BINS):
    if counts[b] == 0:
        continue
    n    = counts[b]
    shf_t = sum_shf_truth[b] / n
    shf_m = sum_shf_memo[b]  / n
    r_p5  = shf_m / max(shf_t, 1e-9)
    r_vf  = vfcg_shf_ratio[b]
    r_cmp2 = r_p5 / r_vf
    if r_cmp2 > 1.05:   verdict = "p5 better"
    elif r_cmp2 < 0.95: verdict = "p5 worse"
    else:               verdict = "similar"
    print(f"{BIN_LABELS[b]:12s}  {n:>10,}  "
          f"{r_p5:>14.3f}  {r_vf:>16.3f}  {r_cmp2:>17.3f}  {verdict:>12s}")

print()
print(f"{'ICEFRAC bin':12s}  {'N cells':>10s}  "
      f"{'ratio_truth_p5':>14s}  {'ratio_truth_vfcg':>16s}  {'ratio_p5_vs_vfcg':>17s}  {'verdict':>12s}"
      f"   (var: |LHF|)")
print("-" * 110)

for b in range(N_BINS):
    if counts[b] == 0:
        continue
    n    = counts[b]
    lhf_t = sum_lhf_truth[b] / n
    lhf_m = sum_lhf_memo[b]  / n
    r_p5  = lhf_m / max(lhf_t, 1e-9)
    r_vf  = vfcg_lhf_ratio[b]
    r_cmp2 = r_p5 / r_vf
    if r_cmp2 > 1.05:   verdict = "p5 better"
    elif r_cmp2 < 0.95: verdict = "p5 worse"
    else:               verdict = "similar"
    print(f"{BIN_LABELS[b]:12s}  {n:>10,}  "
          f"{r_p5:>14.3f}  {r_vf:>16.3f}  {r_cmp2:>17.3f}  {verdict:>12s}")

# ── raw values for full-ice bin (0.9–1.0) for hand-verification ───────────────
b_ice  = 5
b_open = 0
print()
print("=" * 70)
print("RAW mean flux values — ICEFRAC 0.9–1.0 bin (for hand-verification)")
print("=" * 70)
n_ice  = counts[b_ice]
n_open = counts[b_open]
tau_t_ice = sum_tau_truth[b_ice] / n_ice;  tau_m_ice = sum_tau_memo[b_ice]  / n_ice
shf_t_ice = sum_shf_truth[b_ice] / n_ice;  shf_m_ice = sum_shf_memo[b_ice]  / n_ice
lhf_t_ice = sum_lhf_truth[b_ice] / n_ice;  lhf_m_ice = sum_lhf_memo[b_ice]  / n_ice
print(f"  |τ|   truth={tau_t_ice:.4f} N/m²   p5={tau_m_ice:.4f}   vfcg={vfcg_tau_fullice:.4f}")
print(f"  |SHF| truth={shf_t_ice:.3f} W/m²   p5={shf_m_ice:.3f}   vfcg={vfcg_shf_fullice:.3f}")
print(f"  |LHF| truth={lhf_t_ice:.3f} W/m²   p5={lhf_m_ice:.3f}   vfcg={vfcg_lhf_fullice:.3f}")

# ── ice suppression summary ────────────────────────────────────────────────────
print()
print("=" * 70)
print("Ice suppression summary (full-ice / open-ocean ratio)")
print("=" * 70)

tau_t_open = sum_tau_truth[b_open]/n_open;  tau_m_open = sum_tau_memo[b_open]/n_open
shf_t_open = sum_shf_truth[b_open]/n_open;  shf_m_open = sum_shf_memo[b_open]/n_open
lhf_t_open = sum_lhf_truth[b_open]/n_open;  lhf_m_open = sum_lhf_memo[b_open]/n_open

ts_truth = tau_t_ice / tau_t_open;  ts_p5 = tau_m_ice / tau_m_open
ss_truth = shf_t_ice / shf_t_open;  ss_p5 = shf_m_ice / shf_m_open
ls_truth = lhf_t_ice / lhf_t_open;  ls_p5 = lhf_m_ice / lhf_m_open

# vfcg suppression factors from log
ts_vfcg = 0.812;  ss_vfcg = 0.303;  ls_vfcg = None  # not printed in original log

print(f"              {'truth':>10s}   {'p5ge0uvh':>10s}   {'vfcg1m61':>10s}")
print(f"  |τ|  supp:  {ts_truth:>10.3f}×  {ts_p5:>10.3f}×  {ts_vfcg:>10.3f}×")
print(f"  |SHF| supp: {ss_truth:>10.3f}×  {ss_p5:>10.3f}×  {ss_vfcg:>10.3f}×")
print(f"  |LHF| supp: {ls_truth:>10.3f}×  {ls_p5:>10.3f}×  {'N/A':>10s}")

# ── go/no-go ──────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("GO / NO-GO recommendation")
print("=" * 70)
print(f"  SHF ice suppression ratio: truth={ss_truth:.3f}  p5={ss_p5:.3f}  vfcg={ss_vfcg:.3f}")
if ss_p5 > 0.45:
    print("  >> GO: p5 SHF suppression >0.45, materially closer to truth than vfcg.")
elif ss_p5 > 0.40:
    print("  >> AMBIGUOUS: p5 SHF suppression 0.40–0.45; improvement over vfcg but still short of truth.")
else:
    print("  >> NO-GO: p5 SHF suppression ≤0.40, similar to vfcg; structural ice failure persists.")
