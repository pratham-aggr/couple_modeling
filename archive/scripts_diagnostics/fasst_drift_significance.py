"""Rigorous drift test for FASST, post-transient only.

The full-run trend is meaningless as a drift measure (it includes the
spin-down transient). This computes FASST's trend over the post-transient
window only, and compares it against the distribution of same-length
trends in the 1200-yr piControl record.

Caches FASST's annual global-mean SST series so later analyses are fast.
"""
import glob, os, re
import numpy as np, xarray as xr

FASST_RUN = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw_iceloop/run"
PICTL_CACHE = "output/picontrol_gmsst_annual.npz"
FASST_CACHE = "output/fasst_gmsst_annual.npz"
STAB_YEAR = 2011      # from the 2-panel rolling-slope detector

def ym(f):
    m = re.search(r"\.(\d{4})-(\d{2})\.nc$", f)
    return int(m.group(1)), int(m.group(2))

def gmean(sst2d, w):
    s = np.asarray(sst2d, np.float64); m = np.isfinite(s)
    return float(np.nansum(np.where(m, s, 0) * w) / (w * m).sum())

mfiles = sorted(glob.glob(f"{FASST_RUN}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
mkeys = [ym(f) for f in mfiles]
need_rebuild = True
if os.path.exists(FASST_CACHE):
    c = np.load(FASST_CACHE)
    if int(c["n_months"]) == len(mkeys):
        years, ann = c["years"], c["annual"]; need_rebuild = False
        print(f"loaded cached FASST series ({len(years)} yr)")
if need_rebuild:
    print(f"computing FASST annual series from {len(mfiles)} monthly files ...")
    g = xr.open_dataset(mfiles[0])
    TAREA = np.nan_to_num(g["TAREA"].values.astype(np.float64))
    msst = {k: gmean(xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values, TAREA)
            for f, k in zip(mfiles, mkeys)}
    have = set(mkeys)
    years = np.array([y for y in sorted({y for y, m in mkeys})
                      if all((y, m) in have for m in range(1, 13))])
    ann = np.array([np.mean([msst[(y, m)] for m in range(1, 13)]) for y in years])
    np.savez(FASST_CACHE, years=years, annual=ann, n_months=len(mkeys))
    print(f"cached -> {FASST_CACHE}")

pic = np.load(PICTL_CACHE)["annual"]

def trend_kpc(x, v):
    x = np.asarray(x, float)
    A = np.stack([x - x.mean(), np.ones(len(x))], axis=1)
    return float(np.linalg.lstsq(A, v, rcond=None)[0][0]) * 100.0

def all_trends(series, win):
    x = np.arange(win, dtype=float)
    A = np.stack([x - x.mean(), np.ones(win)], axis=1)
    return np.array([np.linalg.lstsq(A, series[i:i+win], rcond=None)[0][0] * 100.0
                     for i in range(len(series) - win + 1)])

post = years >= STAB_YEAR
yp, ap = years[post], ann[post]
win = len(yp)
obs = trend_kpc(yp, ap)

print(f"\n=== FASST post-transient ({yp[0]}-{yp[-1]}, {win} yr) ===")
print(f"  mean {ap.mean():.3f} +/- {ap.std():.3f} degC")
print(f"  trend {obs:+.2f} K/century")
print(f"\n=== piControl reference ===")
print(f"  mean {pic.mean():.3f} +/- {pic.std():.3f} degC  ({len(pic)} yr)")
print(f"  FASST mean offset: {ap.mean()-pic.mean():+.3f} K")
print(f"  variability ratio (FASST sd / piControl sd): {ap.std()/pic.std():.2f}x")

t = all_trends(pic, win)
pct = (t < obs).mean() * 100.0
lo, hi = np.percentile(t, [2.5, 97.5])
print(f"\n=== {win}-yr trends in piControl ({len(t)} overlapping windows) ===")
print(f"  mean {t.mean():+.3f}, sd {t.std():.3f} K/century")
print(f"  95% range [{lo:+.3f}, {hi:+.3f}] | min {t.min():+.3f} max {t.max():+.3f}")
print(f"  FASST {obs:+.2f} -> percentile {pct:.2f}%, z={(obs-t.mean())/t.std():+.2f}, "
      f"inside 95%? {'YES' if lo <= obs <= hi else 'NO'}")

# also: is the MEAN offset within piControl's own spread of window-means?
wm = np.array([pic[i:i+win].mean() for i in range(len(pic) - win + 1)])
wlo, whi = np.percentile(wm, [2.5, 97.5])
print(f"\n=== {win}-yr window MEANS in piControl ===")
print(f"  mean {wm.mean():.3f}, sd {wm.std():.3f} | 95% range [{wlo:.3f}, {whi:.3f}]")
print(f"  FASST {ap.mean():.3f} -> inside 95%? "
      f"{'YES' if wlo <= ap.mean() <= whi else 'NO'}  "
      f"(z={(ap.mean()-wm.mean())/wm.std():+.2f})")
