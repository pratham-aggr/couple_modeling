"""Controlled ablation: unfinetuned (FASST) vs sens-reg flux net.
Identical coupled config; only the flux-net checkpoint differs."""
import glob, re
import numpy as np, xarray as xr

RUNS = {
    "FASST (unft, no sens_reg)": "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw_iceloop/run",
    "sensreg (dQ/dSST constrained)": "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_sensreg50yr/run",
}
PICTL_CACHE = "output/picontrol_gmsst_annual.npz"

def ym(f):
    m = re.search(r"\.(\d{4})-(\d{2})\.nc$", f)
    return int(m.group(1)), int(m.group(2))

def gmean(sst2d, w):
    s = np.asarray(sst2d, np.float64); m = np.isfinite(s)
    return float(np.nansum(np.where(m, s, 0) * w) / (w * m).sum())

z = np.load(PICTL_CACHE)
pic = z["annual"]
print(f"piControl: mean={pic.mean():.3f} std={pic.std():.3f} degC "
      f"({len(pic)} yr available)\n")

for label, run in RUNS.items():
    mfiles = sorted(glob.glob(f"{run}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
    if not mfiles:
        print(f"{label}: no data"); continue
    mkeys = [ym(f) for f in mfiles]
    g = xr.open_dataset(mfiles[0])
    TAREA = np.nan_to_num(g["TAREA"].values.astype(np.float64))
    msst = {k: gmean(xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values, TAREA)
            for f, k in zip(mfiles, mkeys)}
    have = set(mkeys)
    yrs = [y for y in sorted({y for y, m in mkeys})
           if all((y, m) in have for m in range(1, 13))]
    ann = np.array([np.mean([msst[(y, m)] for m in range(1, 13)]) for y in yrs])
    n = min(len(ann), len(pic))
    p = pic[:n]

    # linear trend over the full run and over the last 20 yr
    def trend(x, v):
        A = np.stack([np.asarray(x, float), np.ones(len(x))], axis=1)
        return float(np.linalg.lstsq(A, v, rcond=None)[0][0])

    last10 = ann[-10:]
    print(f"--- {label} ---")
    print(f"  span            : {yrs[0]}-{yrs[-1]} ({len(yrs)} yr)")
    print(f"  yr1 -> final     : {ann[0]:.3f} -> {ann[-1]:.3f} degC  "
          f"(net {ann[-1]-ann[0]:+.3f} K)")
    print(f"  full-run trend  : {trend(yrs, ann)*100:+.2f} K/century")
    print(f"  last-20yr trend : {trend(yrs[-20:], ann[-20:])*100:+.2f} K/century")
    print(f"  last-10yr mean  : {last10.mean():.3f} degC  "
          f"(vs piControl {pic.mean():.3f} -> offset {last10.mean()-pic.mean():+.3f} K)")
    print(f"  overlap-window  : sim {ann[:n].mean():.3f} +/- {ann[:n].std():.3f} | "
          f"piCtl {p.mean():.3f} +/- {p.std():.3f}  ({n} yr)")
    print()
