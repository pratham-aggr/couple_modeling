"""
compare_hpx_vs_latlon.py
========================
Side-by-side per-variable skill of the HEALPix-native model vs the lat-lon MSE
MEMO point model.  Both report global ocean R2/RMSE/corr from metrics_test.json
on their own held-out test set (2013-14); HEALPix metrics are on equal-area
HEALPix pixels, lat-lon on the 192x288 grid (the one comparison caveat).

Usage:
    python scripts/compare_hpx_vs_latlon.py
"""
import json
from pathlib import Path

HPX     = Path("output/output_hpx64_mem24h/metrics_test.json")
LATLON  = Path("output/output_unet_mem24h_dsst_temporal_radprecip/metrics_test.json")

hpx = json.load(open(HPX))
ll  = json.load(open(LATLON))

vars_ = ["TAUX", "TAUY", "SHFLX", "LHFLX", "QFLX", "FSDS_J", "FLDS_J", "PRECT"]
print(f"{'var':8s} | {'R2_hpx':>8s} {'R2_ll':>8s} {'dR2':>7s} | "
      f"{'corr_hpx':>9s} {'corr_ll':>8s}")
print("-" * 60)
for v in vars_:
    if v not in hpx or v not in ll:
        continue
    h, l = hpx[v], ll[v]
    dr2 = h["r2"] - l["r2"]
    flag = "  <-- wind" if v in ("TAUX", "TAUY") else ""
    print(f"{v:8s} | {h['r2']:8.4f} {l['r2']:8.4f} {dr2:+7.4f} | "
          f"{h['corr']:9.4f} {l['corr']:8.4f}{flag}")

print("\nNote: R2 on each model's native grid (HEALPix equal-area vs lat-lon). "
      "Both are global ocean skill; not pixel-identical test sets.")
