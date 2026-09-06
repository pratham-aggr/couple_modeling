"""perturb_pop_ic.py — build a perturbed POP initial restart for the coupler-
resilience experiment (user 2026-08-16): add a uniform upper-ocean warm anomaly to
the LE2 1980-01-01 restart, then let the pz220f0b + ML-CICE coupler run and see
whether its air-sea feedback damps the anomaly back toward the control over years.

PERTURBATION (documented for the paper):
  +2.0 K added to TEMP over the TOP 100 m (POP levels k=0..9, midpoints 5-105 m),
  OCEAN cells only (k < KMT), applied to BOTH leapfrog time levels (TEMP_CUR and
  TEMP_OLD). Everything else (salinity, velocity, ice, forcing, model) is identical
  to the control -> the ONLY difference from the control run is this IC warm kick.

Output restart keeps the LE2 filename so rpointer/POP read it unchanged; it is
staged into the isolated perturbed rundir by the PBS.
"""
import argparse
import numpy as np, xarray as xr, sys

LE2 = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagwind1yr/run/b.e21.BHISTcmip6.f09_g17.LE2-1231.002.pop.r.1980-01-01-00000.nc"
KMT_SRC = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw_iceemu1yr/run/g.e21.MEMO_GIAF_v01.pop.h.1980-01.nc"
KTOP  = 10           # perturb levels k=0..9  (top ~100 m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtemp", type=float, default=2.0, help="warm anomaly, K")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    global DTEMP, OUT
    DTEMP = args.dtemp
    OUT = args.out or f"/glade/derecho/scratch/praggarwal/pop_ic_pert{DTEMP:g}K_top100m.nc"
    ds = xr.open_dataset(LE2, decode_times=False)
    kmt = xr.open_dataset(KMT_SRC, decode_times=False)["KMT"].values.astype(int)  # (j,i) # of wet levels
    nk, nj, ni = ds["TEMP_CUR"].shape
    # wet(k,j,i) = k < KMT(j,i)
    kk = np.arange(nk)[:, None, None]
    wet = kk < kmt[None, :, :]                      # (nk,nj,ni) bool
    lev = np.zeros(nk, bool); lev[:KTOP] = True     # top-100m levels
    pmask = wet & lev[:, None, None]                # cells to perturb
    npert = int(pmask.sum())

    for tv in ("TEMP_CUR", "TEMP_OLD"):
        t = ds[tv].values.copy()
        pre = t[0][wet[0]].mean()
        t[pmask] += DTEMP
        ds[tv].values[...] = t
        post = ds[tv].values[0][wet[0]].mean()
        print(f"  {tv}: k=0 ocean-mean {pre:.3f} -> {post:.3f} degC")

    ds.attrs["perturbation"] = (f"+{DTEMP} K on TEMP_CUR/TEMP_OLD, levels k=0..{KTOP-1} "
                                f"(top ~100m), ocean only (k<KMT); coupler-resilience IC kick")
    ds.to_netcdf(OUT)
    # copy the .ro overflow file unchanged is handled by the PBS (perturbation is TEMP only)
    print(f"\nwrote {OUT}")
    print(f"perturbed {npert} ocean cells over top {KTOP} levels (+{DTEMP} K)")
    print(f"  = a clean upper-ocean warm anomaly; SST(k=0) rises +{DTEMP} K on every ocean column")

if __name__ == "__main__":
    main()
