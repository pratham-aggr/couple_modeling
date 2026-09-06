"""Equatorial zonal cross-section: SST vs longitude (pz220f0b, piControl, error),
averaged over a 5S-5N band on the NATIVE gx1v7 grid (no regrid interpolation
near the equator -- avoids any regrid-smoothing artifact right where ENSO
signal lives), binned into 1-deg longitude bins, TAREA-weighted within each bin.

Grid identity (TLAT/TLONG/TAREA/KMT bit-identical between the two datasets) was
already verified in plot_maps_pz220f0b_vs_picontrol.py -- reused here unchanged.
"""
import glob, re
import numpy as np, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PZ_DIR = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw_iceloop/run"
PI_FILE = ("/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/"
           "b.e21.B1850.f09_g17.CMIP6-piControl.001/ocn/proc/tseries/month_1/"
           "b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.SST.000101-009912.nc")
YEAR_MIN, YEAR_MAX = 2000, 2045
EQ_BAND = (-5, 5)

mfiles = sorted(glob.glob(f"{PZ_DIR}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
mfiles_win = [f for f in mfiles
              if YEAR_MIN <= int(re.search(r"\.(\d{4})-\d{2}\.nc$", f).group(1)) <= YEAR_MAX]
g0 = xr.open_dataset(mfiles_win[0])
TLAT, TLONG = g0["TLAT"].values.astype("f8"), g0["TLONG"].values.astype("f8")
TAREA = np.nan_to_num(g0["TAREA"].values.astype("f8"))
KMT = g0["KMT"].values

eq = (TLAT >= EQ_BAND[0]) & (TLAT <= EQ_BAND[1]) & (KMT > 0)
print(f"equatorial band {EQ_BAND} deg: {eq.sum()} ocean cells")

n_needed = len(mfiles_win)
sst_sum = np.zeros_like(TLAT)
for f in mfiles_win:
    sst_sum += xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values.astype("f8")
pz_mean = sst_sum / n_needed
print(f"pz220f0b: {n_needed} months averaged ({YEAR_MIN}-{YEAR_MAX})")

ds_pi = xr.open_dataset(PI_FILE)
assert np.allclose(TLONG, ds_pi["TLONG"].values.astype("f8")) and np.array_equal(KMT, ds_pi["KMT"].values), \
    "grid mismatch -- do not proceed"
pi_mean = np.nanmean(ds_pi["SST"].isel(z_t=0, time=slice(0, n_needed)).values.astype("f8"), axis=0)
print(f"piControl: {n_needed} months averaged (elapsed model yr 1-{n_needed // 12})")

# --- bin into 1-deg longitude bins within the equatorial band, TAREA-weighted ---
lon_bins = np.arange(0, 361, 1.0)
lon_ctr = 0.5 * (lon_bins[:-1] + lon_bins[1:])
lon_idx = np.digitize(TLONG[eq], lon_bins) - 1
w = TAREA[eq]
pz_v, pi_v = pz_mean[eq], pi_mean[eq]

def bin_weighted(vals, idx, w, nb):
    num = np.bincount(idx, weights=vals * w, minlength=nb)
    den = np.bincount(idx, weights=w, minlength=nb)
    out = np.full(nb, np.nan)
    m = den > 0
    out[m] = num[m] / den[m]
    return out

nb = len(lon_ctr)
valid = (lon_idx >= 0) & (lon_idx < nb)
pz_binned = bin_weighted(pz_v[valid], lon_idx[valid], w[valid], nb)
pi_binned = bin_weighted(pi_v[valid], lon_idx[valid], w[valid], nb)
err_binned = pz_binned - pi_binned

print(f"equatorial error: min={np.nanmin(err_binned):+.2f}  max={np.nanmax(err_binned):+.2f}  "
      f"mean={np.nanmean(err_binned):+.2f}  degC")
worst = np.nanargmax(np.abs(err_binned))
print(f"  worst at lon={lon_ctr[worst]:.0f}E: pz220f0b={pz_binned[worst]:.2f}  "
      f"piControl={pi_binned[worst]:.2f}  error={err_binned[worst]:+.2f}")

# ---------- figure ----------
fig, (a, b) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[1.3, 1])

a.plot(lon_ctr, pi_binned, color="#5b6167", lw=1.8, label="CESM2 piControl")
a.plot(lon_ctr, pz_binned, color="tab:orange", lw=1.8,
       label=f"pz220f0b unfinetuned ({YEAR_MIN}-{YEAR_MAX})")
a.set_ylabel("SST (degC)")
a.set_title(f"Equatorial ({EQ_BAND[0]} to {EQ_BAND[1]} deg lat) SST vs longitude")
a.legend()
a.grid(alpha=0.3)

b.axhline(0, color="k", lw=0.8)
b.plot(lon_ctr, err_binned, color="tab:red", lw=1.8)
b.fill_between(lon_ctr, 0, err_binned, color="tab:red", alpha=0.15)
b.set_ylabel("error (pz220f0b - piControl), degC")
b.set_xlabel("longitude (deg E)")
b.set_xlim(0, 360)
b.set_ylim(np.nanmin(err_binned) - 1, np.nanmax(err_binned) + 1)
for lon, name in [(190, "Nino3.4 west edge (170W)"), (240, "Nino3.4 east edge (120W)")]:
    b.axvline(lon, color="gray", ls="--", lw=0.8)
    b.text(lon, b.get_ylim()[0] + 0.3, name, fontsize=7, rotation=90, va="bottom", color="gray")
b.grid(alpha=0.3)

fig.tight_layout()
out = "output/equator_zonal_error.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")
