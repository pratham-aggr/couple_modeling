"""Is FASST's residual trend distinguishable from unforced internal variability?

Computes the distribution of all overlapping N-year linear trends in the
1200-year CESM2 piControl global-mean SST record, and locates FASST's
observed trends within it.
"""
import numpy as np

PICTL_CACHE = "output/picontrol_gmsst_annual.npz"
FASST_TRENDS = {20: -1.78, 108: -0.66}   # K/century, from compare_unft_vs_sensreg.py

pic = np.load(PICTL_CACHE)["annual"]
print(f"piControl: {len(pic)} yr, mean={pic.mean():.3f} std={pic.std():.3f} degC\n")

def all_trends(series, win):
    """All overlapping `win`-year linear trends, in K/century."""
    out = []
    x = np.arange(win, dtype=float)
    A = np.stack([x - x.mean(), np.ones(win)], axis=1)
    for i in range(len(series) - win + 1):
        slope = np.linalg.lstsq(A, series[i:i + win], rcond=None)[0][0]
        out.append(slope * 100.0)
    return np.array(out)

for win, fasst in FASST_TRENDS.items():
    if len(pic) < win:
        print(f"{win}-yr: piControl too short"); continue
    t = all_trends(pic, win)
    pct = (t < fasst).mean() * 100.0
    inside = np.percentile(t, 2.5) <= fasst <= np.percentile(t, 97.5)
    print(f"=== {win}-year trends in piControl ({len(t)} overlapping windows) ===")
    print(f"  piControl trend distribution: mean {t.mean():+.2f}, sd {t.std():.2f} K/century")
    print(f"    2.5th pct {np.percentile(t,2.5):+.2f} | median {np.median(t):+.2f} "
          f"| 97.5th pct {np.percentile(t,97.5):+.2f}")
    print(f"    min {t.min():+.2f} | max {t.max():+.2f}")
    print(f"  FASST observed: {fasst:+.2f} K/century")
    print(f"    -> percentile within piControl: {pct:.1f}%")
    print(f"    -> inside piControl 95% range? {'YES' if inside else 'NO'}")
    z = (fasst - t.mean()) / t.std()
    print(f"    -> z-score vs piControl trend distribution: {z:+.2f}")
    print()
