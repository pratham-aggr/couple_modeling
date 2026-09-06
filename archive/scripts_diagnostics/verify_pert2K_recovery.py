"""Verify the +2 K perturbation-recovery numbers claimed for the paper.

Pairs (each pert vs its OWN matching control, same ice configuration):
  interactive ice (= FASST config, --ice_input_from_emu):
      pert = diagfw_iceemu_pert2K   ctrl = diagfw_iceloop (FASST production 1980-)
  climatology ice input (no --ice_input_from_emu):
      pert = diagfw_pert2K          ctrl = diagfw1yr
Anomaly = area-weighted global-mean SST(pert) - SST(ctrl), annual means.
"""
import glob, re
import numpy as np, xarray as xr

S = "/glade/derecho/scratch/praggarwal"
PAIRS = {
    "interactive ice (FASST config)": (
        f"{S}/memo_pop_standalone_gx1v7_diagfw_iceemu_pert2K/run",
        f"{S}/memo_pop_standalone_gx1v7_diagfw_iceloop/run"),
    "climatology ice input": (
        f"{S}/memo_pop_standalone_gx1v7_diagfw_pert2K/run",
        f"{S}/memo_pop_standalone_gx1v7_diagfw1yr/run"),
}

def ym(f):
    m = re.search(r"\.(\d{4})-(\d{2})\.nc$", f); return int(m.group(1)), int(m.group(2))

def annual(rundir, years):
    fs = sorted(glob.glob(f"{rundir}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
    if not fs: return None
    g = xr.open_dataset(fs[0]); W = np.nan_to_num(g["TAREA"].values.astype(np.float64))
    by = {}
    for f in fs:
        y, m = ym(f)
        if y not in years: continue
        s = xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values.astype(np.float64)
        ok = np.isfinite(s)
        by[(y, m)] = float(np.nansum(np.where(ok, s, 0) * W) / (W * ok).sum())
    out = {}
    for y in years:
        if all((y, m) in by for m in range(1, 13)):
            out[y] = np.mean([by[(y, m)] for m in range(1, 13)])
    return out

YEARS = list(range(1980, 1985))
for label, (pd_, cd) in PAIRS.items():
    p = annual(pd_, YEARS); c = annual(cd, YEARS)
    print(f"\n=== {label} ===")
    print(f"  pert: {pd_.split('gx1v7_')[-1]}")
    print(f"  ctrl: {cd.split('gx1v7_')[-1]}")
    if not p or not c:
        print("  MISSING DATA"); continue
    common = [y for y in YEARS if y in p and y in c]
    print(f"  {'yr':<6}{'model_yr':>9}{'pert':>9}{'ctrl':>9}{'anom K':>9}")
    anoms = []
    for i, y in enumerate(common):
        a = p[y] - c[y]; anoms.append(a)
        print(f"  {y:<6}{i:>9}{p[y]:>9.3f}{c[y]:>9.3f}{a:>+9.3f}")
    if anoms:
        print(f"  -> model yr 0 anomaly : {anoms[0]:+.2f} K")
        print(f"  -> model yr {len(anoms)-1} anomaly : {anoms[-1]:+.2f} K")
        print(f"  -> decay             : {anoms[0]-anoms[-1]:+.2f} K "
              f"({100*(1-anoms[-1]/anoms[0]):.0f}% of the initial anomaly removed)")
