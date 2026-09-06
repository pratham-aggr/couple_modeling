"""analyze_diagfw_iceloop_neurips.py -- comprehensive readiness analysis of the
unfinetuned pz220f0b + interactive-ice diagfw_iceloop run (job stream
7239950->7241650), for NeurIPS CCAI workshop submission assessment.

Computes, over the FULL run history (1980-present):
  1. Global + regional (Tropics/SO/NH) annual-mean SST, TAREA-weighted
  2. Linear drift trend (full run + last-20yr window)
  3. Comparison to CESM2 piControl natural variability envelope
  4. NH/SH sea-ice extent annual cycle (IFRAC>0.15) at start/mid/end decades
  5. Nino3.4 monthly anomaly timeseries + interannual variability (std, lag-1 autocorr)
  6. Full blowup/error scan across every job log for this run
  7. Wall-clock efficiency (min/sim-yr) across all completed segments
"""
import glob, re, os, json
import numpy as np, xarray as xr

RUN = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw_iceloop/run"
LOGDIR = "/glade/u/home/praggarwal/couple/output/logs"
PICTL_CACHE = "output/picontrol_gmsst_annual.npz"
OUT_JSON = "output/diagfw_iceloop_neurips_analysis.json"

def gmean(sst2d, w):
    s = np.asarray(sst2d, np.float64)
    m = np.isfinite(s)
    return float(np.nansum(np.where(m, s, 0) * w) / (w * m).sum())

print("=== loading grid ===")
mfiles_all = sorted(glob.glob(f"{RUN}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
def ym(f):
    m = re.search(r"\.(\d{4})-(\d{2})\.nc$", f)
    return int(m.group(1)), int(m.group(2))
mkeys = [ym(f) for f in mfiles_all]
g0 = xr.open_dataset(mfiles_all[0])
TAREA = np.nan_to_num(g0["TAREA"].values.astype(np.float64))
TLAT = g0["TLAT"].values
TLONG = g0["TLONG"].values
years_avail = sorted({y for (y, m) in mkeys})
full_years = [y for y in years_avail if all((y, m) in dict.fromkeys(mkeys) or True for m in range(1, 13))]
# proper full-year filter
have = set(mkeys)
full_years = [y for y in years_avail if all((y, m) in have for m in range(1, 13))]
print(f"years available: {full_years[0]}-{full_years[-1]} ({len(full_years)} complete years)")

regions = {
    "Tropics": (TLAT >= -30) & (TLAT <= 30),
    "SO": TLAT < -45,
    "NH": TLAT > 45,
}
nino34_mask = (TLAT >= -5) & (TLAT <= 5) & (((TLONG >= 190) & (TLONG <= 240)))  # 170W-120W = 190E-240E

print("=== pass 1: monthly global+regional SST, monthly Nino3.4 ===")
fmap = {ym(f): f for f in mfiles_all}
monthly_global = {}
monthly_regional = {r: {} for r in regions}
monthly_nino34 = {}
n34_w = TAREA * nino34_mask
n34_wsum = n34_w.sum()
reg_w = {r: TAREA * m for r, m in regions.items()}
reg_wsum = {r: w.sum() for r, w in reg_w.items()}

for i, (y, m) in enumerate(mkeys):
    f = fmap[(y, m)]
    sst = xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values
    s = np.asarray(sst, np.float64)
    finite = np.isfinite(s)
    monthly_global[(y, m)] = float(np.nansum(np.where(finite, s, 0) * TAREA) / (TAREA * finite).sum())
    for r, w in reg_w.items():
        monthly_regional[r][(y, m)] = float(np.nansum(np.where(finite, s, 0) * w) / (w * finite).sum())
    monthly_nino34[(y, m)] = float(np.nansum(np.where(finite, s, 0) * n34_w) / (n34_w * finite).sum())
    if i % 100 == 0:
        print(f"  {i}/{len(mkeys)}  {y}-{m:02d}  global={monthly_global[(y,m)]:.3f}")

print("=== annual aggregation ===")
def annualize(d):
    return np.array([np.mean([d[(y, m)] for m in range(1, 13)]) for y in full_years])

ann_global = annualize(monthly_global)
ann_reg = {r: annualize(monthly_regional[r]) for r in regions}
ann_nino34 = annualize(monthly_nino34)   # annual mean nino3.4 (not anomaly)

# --- Nino3.4 monthly anomaly relative to the run's OWN monthly climatology ---
clim_by_month = {m: np.mean([monthly_nino34[(y, m)] for y in full_years]) for m in range(1, 13)}
n34_anom = np.array([monthly_nino34[(y, m)] - clim_by_month[m] for y in full_years for m in range(1, 13)])
n34_std = float(n34_anom.std())
n34_ac1 = float(np.corrcoef(n34_anom[:-1], n34_anom[1:])[0, 1])

print("=== drift trend ===")
yrs = np.array(full_years, dtype=np.float64)
def trend(y, v):
    A = np.stack([y, np.ones_like(y)], axis=1)
    slope, intercept = np.linalg.lstsq(A, v, rcond=None)[0]
    return float(slope)

trend_full = trend(yrs, ann_global)
trend_last20 = trend(yrs[-20:], ann_global[-20:]) if len(yrs) >= 20 else None
trend_first20 = trend(yrs[:20], ann_global[:20]) if len(yrs) >= 20 else None

print("=== piControl comparison ===")
if os.path.exists(PICTL_CACHE):
    z = np.load(PICTL_CACHE)
    pic_years, pic_annual = z["years"], z["annual"]
    pic_mean, pic_sd = float(pic_annual.mean()), float(pic_annual.std())
else:
    pic_mean = pic_sd = None

sim_mean = float(ann_global.mean())
sim_final20_mean = float(ann_global[-20:].mean())
offset_from_pic_sigma = (sim_final20_mean - pic_mean) / pic_sd if pic_mean is not None else None

print("=== ice extent (NH/SH, IFRAC>0.15) at start/mid/end decades ===")
def ice_extent_year(year):
    files = [fmap.get((year, m)) for m in range(1, 13)]
    if any(f is None for f in files):
        return None
    nh_ext, sh_ext = [], []
    for f in files:
        ifrac = xr.open_dataset(f)["IFRAC"].isel(time=0).values if "IFRAC" in xr.open_dataset(f).variables else None
        if ifrac is None:
            return None
        ic = np.nan_to_num(ifrac, nan=0.0)
        icy = ic > 0.15
        nh_ext.append(float(TAREA[icy & (TLAT > 0)].sum()) / 1e12)  # million km^2
        sh_ext.append(float(TAREA[icy & (TLAT < 0)].sum()) / 1e12)
    return {"NH_max": max(nh_ext), "NH_min": min(nh_ext), "SH_max": max(sh_ext), "SH_min": min(sh_ext)}

ice_samples = {}
for y in [full_years[0], full_years[len(full_years)//2], full_years[-1]]:
    ie = ice_extent_year(y)
    if ie:
        ice_samples[y] = ie
        print(f"  {y}: NH {ie['NH_min']:.1f}-{ie['NH_max']:.1f}  SH {ie['SH_min']:.1f}-{ie['SH_max']:.1f} (M km2)")

print("=== full blowup/error scan across ALL logs for this run ===")
log_files = sorted(glob.glob(f"{LOGDIR}/*.OU")) + sorted(glob.glob(f"{LOGDIR}/*diagfw_iceloop*"))
log_files = sorted(set(log_files))
hits = []
seg_times = []  # (elapsed_s, seg_years)
for lf in log_files:
    try:
        txt = open(lf, errors="ignore").read()
    except Exception:
        continue
    if "diagfw_iceloop" not in txt and "diagfw_iceloop" not in lf:
        continue
    for pat in ["blow", "NaN", "nan detected", "abort", "Non-finite", "fatal", "CFL", "Traceback", "ERROR"]:
        if pat.lower() in txt.lower() and pat != "nan":  # "nan" alone too noisy (netcdf msgs)
            for line in txt.splitlines():
                if pat.lower() in line.lower():
                    hits.append((os.path.basename(lf), line.strip()[:160]))
    m = re.search(r"POP_RC=(-?\d+)\s+elapsed=(\d+)s for (\d+) yr", txt)
    if m and int(m.group(1)) == 0:
        seg_times.append((int(m.group(2)), int(m.group(3))))

print(f"  scanned {len(log_files)} log files; {len(hits)} suspicious lines found")
for fn, line in hits[:20]:
    print(f"    !! {fn}: {line}")

if seg_times:
    minyr_list = [e/60.0/y for e, y in seg_times]
    print(f"  {len(seg_times)} clean (rc=0) segments; min/sim-yr: mean={np.mean(minyr_list):.1f} "
          f"min={np.min(minyr_list):.1f} max={np.max(minyr_list):.1f}")

result = {
    "years": [full_years[0], full_years[-1]],
    "n_years": len(full_years),
    "global_mean_full": sim_mean,
    "global_mean_last20": sim_final20_mean,
    "trend_full_K_per_yr": trend_full,
    "trend_first20_K_per_yr": trend_first20,
    "trend_last20_K_per_yr": trend_last20,
    "piControl_mean": pic_mean,
    "piControl_sd": pic_sd,
    "offset_from_piControl_in_sigma": offset_from_pic_sigma,
    "nino34_monthly_anomaly_std": n34_std,
    "nino34_lag1_autocorr": n34_ac1,
    "regional_annual": {r: ann_reg[r].tolist() for r in regions},
    "global_annual": ann_global.tolist(),
    "full_years_list": full_years,
    "ice_extent_samples_Mkm2": ice_samples,
    "n_suspicious_log_lines": len(hits),
    "suspicious_log_lines": hits[:50],
    "clean_segments": len(seg_times),
    "min_per_simyr_mean": float(np.mean([e/60.0/y for e, y in seg_times])) if seg_times else None,
}
with open(OUT_JSON, "w") as fh:
    json.dump(result, fh, indent=2)
print(f"\nwrote {OUT_JSON}")

print("\n=== SUMMARY ===")
print(f"Run span: {full_years[0]}-{full_years[-1]} ({len(full_years)} yr)")
print(f"Global-mean SST: full-run mean={sim_mean:.3f}, last-20yr mean={sim_final20_mean:.3f} degC")
print(f"Drift trend: full-run={trend_full*100:+.2f} K/century, "
      f"first-20yr={trend_first20*100:+.2f} K/century, last-20yr={trend_last20*100:+.2f} K/century" if trend_first20 else "")
if pic_mean is not None:
    print(f"piControl: mean={pic_mean:.3f}+/-{pic_sd:.3f} degC -> sim offset = {offset_from_pic_sigma:+.2f} sigma")
print(f"Nino3.4 monthly anomaly std={n34_std:.3f} degC, lag-1 autocorr={n34_ac1:.3f}")
print(f"Blowup/error scan: {len(hits)} suspicious lines across {len(log_files)} logs")
