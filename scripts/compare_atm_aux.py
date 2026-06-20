#!/usr/bin/env python3
"""Compare the atm-aux experiment against the matched flux-only baseline.

Reads metrics_test.json (per-variable R^2 / RMSE / corr on the 2013-2014 test
set) from both run dirs and prints:
  - a side-by-side table of the 8 shared flux variables (TAUX/TAUY highlighted)
    with baseline vs experiment and Δ for each metric, and
  - the experiment's metrics for the 5 auxiliary atm targets (how well the aux
    head reconstructs Ubot/Vbot/Tbot/Qbot/PS).

Usage:
    python scripts/compare_atm_aux.py
"""
import json
from pathlib import Path

# Baseline = existing flux-only MSE run vfcg1m61 (output_unet_mem24h_dsst_temporal_radprecip):
# config-identical to the experiment minus the 5 aux atm outputs. Run train_unet.py
# --eval_test on it once to emit metrics_test.json (R2/RMSE/corr) if not already present.
OUT = Path("/glade/u/home/praggarwal/couple/output")
BASE = OUT / "output_unet_mem24h_dsst_temporal_radprecip" / "metrics_test.json"
EXP  = OUT / "output_unet_mem24h_atm_aux"                  / "metrics_test.json"

FLUX = ["TAUX", "TAUY", "SHFLX", "LHFLX", "QFLX", "FSDS_J", "FLDS_J", "PRECT"]
ATM  = ["Ubot", "Vbot", "Tbot", "Qbot", "PS"]


def load(p):
    if not p.exists():
        raise SystemExit(f"Not found: {p}\n(has that run finished and written metrics_test.json?)")
    return json.load(open(p))


def main():
    base, exp = load(BASE), load(EXP)

    print("=" * 92)
    print("Flux targets — matched baseline vs atm-aux experiment (test set, ocean points)")
    print(f"{'var':8s} | {'R2 base':>8s} {'R2 exp':>8s} {'ΔR2':>8s} | "
          f"{'RMSE base':>10s} {'RMSE exp':>10s} | {'r base':>7s} {'r exp':>7s} {'Δr':>7s}")
    print("-" * 92)
    for v in FLUX:
        b, e = base.get(v), exp.get(v)
        if b is None or e is None:
            print(f"{v:8s} | (missing in one run)")
            continue
        star = " *" if v in ("TAUX", "TAUY") else "  "
        print(f"{v:6s}{star} | {b['r2']:8.4f} {e['r2']:8.4f} {e['r2']-b['r2']:+8.4f} | "
              f"{b['rmse']:10.3e} {e['rmse']:10.3e} | "
              f"{b['corr']:7.4f} {e['corr']:7.4f} {e['corr']-b['corr']:+7.4f}")
    print("=" * 92)
    # Headline: did the wind-stress R^2 improve?
    for v in ("TAUX", "TAUY"):
        if v in base and v in exp:
            d = exp[v]["r2"] - base[v]["r2"]
            verdict = "IMPROVED" if d > 0 else ("worse" if d < 0 else "no change")
            print(f"  {v}: ΔR² = {d:+.4f}  ->  {verdict}")

    print("\nAuxiliary atm targets (experiment only) — reconstruction skill at t-1:")
    print(f"  {'var':6s} {'R2':>8s} {'RMSE':>12s} {'corr':>8s}")
    for v in ATM:
        e = exp.get(v)
        if e is None:
            print(f"  {v:6s} (missing)")
        else:
            print(f"  {v:6s} {e['r2']:8.4f} {e['rmse']:12.3e} {e['corr']:8.4f}")


if __name__ == "__main__":
    main()
