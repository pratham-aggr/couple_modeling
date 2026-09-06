"""analyze_diagfw_post2000.py -- redo the diagfw_iceloop NeurIPS-readiness
analysis restricted to YEARS >= 2000, excluding the ~20-yr IC-adjustment
transient (the 1980s-90s "hump"), to characterize the run's TRUE quasi-
equilibrium behavior. Fixes two issues from the first pass:
  - ice-extent unit bug (cm^2 -> million km^2 needs /1e16, not /1e12)
  - Nino3.4 anomaly now LINEARLY DETRENDED before computing std/autocorr,
    so slow drift doesn't leak into the "interannual variability" number.
"""
import glob, re, os, json
import numpy as np, xarray as xr

RUN = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw_iceloop/run"
PICTL_CACHE = "output/picontrol_gmsst_annual.npz"
OUT_JSON = "output/diagfw_iceloop_post2000_analysis.json"
Y0 = 2000

mfiles_all = sorted(glob.glob(f"{RUN}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
def ym(f):
    m = re.search(r"\.(\d{4})-(\d{2})\.nc$", f)
    return int(m.group(1)), int(m.group(2))
mkeys_all = [ym(f) for f in mfiles_all]
fmap = {k: f for f, k in zip(mfiles_all, mkeys_all)}
mkeys = [k for k in mkeys_all if k[0] >= Y0]
have = set(mkeys)
years_avail = sorted({y for (y, m) in mkeys})
full_years = [y for y in years_avail if all((y, m) in have for m in range(1, 13))]
print(f"post-{Y0} years: {full_years[0]}-{full_years[-1]} ({len(full_years)} complete years)")

g0 = xr.open_dataset(fmap[mkeys[0]])
TAREA = np.nan_to_num(g0["TAREA"].values.astype(np.float64))
TLAT = g0["TLAT"].values
TLONG = g0["TLONG"].values

regions = {"Tropics": (TLAT >= -30) & (TLAT <= 30), "SO": TLAT < -45, "NH": TLAT > 45}
nino34_mask = (TLAT >= -5) & (TLAT <= 5) & (TLONG >= 190) & (TLONG <= 240)
reg_w = {r: TAREA * m for r, m in regions.items()}
n34_w = TAREA * nino34_mask

monthly_global, monthly_nino34 = {}, {}
monthly_regional = {r: {} for r in regions}
for i, k in enumerate(mkeys):
    f = fmap[k]
    sst = xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values
    s = np.asarray(sst, np.float64); finite = np.isfinite(s)
    monthly_global[k] = float(np.nansum(np.where(finite, s, 0) * TAREA) / (TAREA * finite).sum())
    for r, w in reg_w.items():
        monthly_regional[r][k] = float(np.nansum(np.where(finite, s, 0) * w) / (w * finite).sum())
    monthly_nino34[k] = float(np.nansum(np.where(finite, s, 0) * n34_w) / (n34_w * finite).sum())
    if i % 100 == 0:
        print(f"  {i}/{len(mkeys)}  {k[0]}-{k[1]:02d}  global={monthly_global[k]:.3f}")

def annualize(d):
    return np.array([np.mean([d[(y, m)] for m in range(1, 13)]) for y in full_years])
ann_global = annualize(monthly_global)
ann_reg = {r: annualize(monthly_regional[r]) for r in regions}

yrs = np.array(full_years, dtype=np.float64)
A = np.stack([yrs, np.ones_like(yrs)], axis=1)
slope, intercept = np.linalg.lstsq(A, ann_global, rcond=None)[0]
resid_sd = float((ann_global - (A @ [slope, intercept])).std())

if os.path.exists(PICTL_CACHE):
    z = np.load(PICTL_CACHE)
    pic_mean, pic_sd = float(z["annual"].mean()), float(z["annual"].std())
else:
    pic_mean = pic_sd = None

# --- Nino3.4: build monthly series in order, remove seasonal clim, THEN linearly detrend ---
n34_series = np.array([monthly_nino34[(y, m)] for y in full_years for m in range(1, 13)])
n34_time = np.array([y + (m - 0.5) / 12.0 for y in full_years for m in range(1, 13)])
clim_by_month = {m: np.mean([monthly_nino34[(y, m)] for y in full_years]) for m in range(1, 13)}
n34_deseason = np.array([monthly_nino34[(y, m)] - clim_by_month[m] for y in full_years for m in range(1, 13)])
# linear detrend on top of deseasonalized series
Ad = np.stack([n34_time, np.ones_like(n34_time)], axis=1)
s2, i2 = np.linalg.lstsq(Ad, n34_deseason, rcond=None)[0]
n34_detrended = n34_deseason - (Ad @ [s2, i2])
n34_std_raw = float(n34_deseason.std())
n34_std_detrended = float(n34_detrended.std())
n34_ac1_detrended = float(np.corrcoef(n34_detrended[:-1], n34_detrended[1:])[0, 1])

def ice_extent_year(year):
    files = [fmap.get((year, m)) for m in range(1, 13)]
    if any(f is None for f in files):
        return None
    nh_ext, sh_ext = [], []
    for f in files:
        ds = xr.open_dataset(f)
        if "IFRAC" not in ds.variables:
            return None
        ic = np.nan_to_num(ds["IFRAC"].isel(time=0).values, nan=0.0)
        icy = ic > 0.15
        nh_ext.append(float(TAREA[icy & (TLAT > 0)].sum()) / 1e16)   # cm2 -> million km2
        sh_ext.append(float(TAREA[icy & (TLAT < 0)].sum()) / 1e16)
    return {"NH_max": max(nh_ext), "NH_min": min(nh_ext), "SH_max": max(sh_ext), "SH_min": min(sh_ext)}

ice_samples = {}
for y in [full_years[0], full_years[len(full_years)//2], full_years[-1]]:
    ie = ice_extent_year(y)
    if ie:
        ice_samples[y] = ie

print("\n=== POST-2000 SUMMARY (excludes the IC-adjustment transient) ===")
print(f"Years: {full_years[0]}-{full_years[-1]} ({len(full_years)} yr)")
print(f"Global-mean SST: mean={ann_global.mean():.3f}, trend={slope*100:+.2f} K/century, "
      f"residual (detrended) SD={resid_sd:.3f} degC")
if pic_mean is not None:
    print(f"piControl: mean={pic_mean:.3f}+/-{pic_sd:.3f} -> post-2000 sim offset = "
          f"{(ann_global.mean()-pic_mean)/pic_sd:+.2f} sigma")
print(f"Regional annual means -- Tropics: {ann_reg['Tropics'].mean():.3f}  "
      f"SO: {ann_reg['SO'].mean():.3f}  NH: {ann_reg['NH'].mean():.3f}")
print(f"Regional trends (K/century) -- Tropics: {np.polyfit(yrs, ann_reg['Tropics'], 1)[0]*100:+.2f}  "
      f"SO: {np.polyfit(yrs, ann_reg['SO'], 1)[0]*100:+.2f}  "
      f"NH: {np.polyfit(yrs, ann_reg['NH'], 1)[0]*100:+.2f}")
print(f"Nino3.4 anomaly std: RAW(deseasoned only)={n34_std_raw:.3f} degC, "
      f"DETRENDED={n34_std_detrended:.3f} degC (real-world ~0.8-1.0)")
print(f"Nino3.4 lag-1 autocorr (detrended): {n34_ac1_detrended:.3f} (real-world ENSO ~0.85-0.95)")
print("Ice extent (M km2, IFRAC>0.15):")
for y, ie in ice_samples.items():
    print(f"  {y}: NH {ie['NH_min']:.2f}-{ie['NH_max']:.2f}  SH {ie['SH_min']:.3f}-{ie['SH_max']:.2f}")

result = {
    "years": [full_years[0], full_years[-1]], "n_years": len(full_years),
    "global_mean": float(ann_global.mean()), "trend_K_per_century": float(slope * 100),
    "detrended_residual_sd": resid_sd,
    "piControl_mean": pic_mean, "piControl_sd": pic_sd,
    "offset_from_piControl_sigma": float((ann_global.mean() - pic_mean) / pic_sd) if pic_mean else None,
    "regional_means": {r: float(ann_reg[r].mean()) for r in regions},
    "regional_trends_K_per_century": {r: float(np.polyfit(yrs, ann_reg[r], 1)[0] * 100) for r in regions},
    "nino34_std_raw": n34_std_raw, "nino34_std_detrended": n34_std_detrended,
    "nino34_lag1_autocorr_detrended": n34_ac1_detrended,
    "ice_extent_Mkm2": ice_samples,
    "global_annual": ann_global.tolist(), "full_years_list": full_years,
}
with open(OUT_JSON, "w") as fh:
    json.dump(result, fh, indent=2)
print(f"\nwrote {OUT_JSON}")
