"""Simple ENSO + SST comparison: pz220f0b UNet-coupled POP run vs CESM2 piControl.

Three panels:
  1. Nino3.4 SST anomaly timeseries (monthly anomaly relative to each run's own
     12-month climatology)
  2. Nino3.4 mean annual cycle (climatological Jan-Dec seasonal cycle)
  3. Global-mean SST, annual-mean timeseries
"""
import glob, re
import numpy as np, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PZ_DIR = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw_iceloop/run"  # 67-yr unfinetuned pz220f0b, interactive ice (job 7239950)
PI_FILE = ("/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/"
           "b.e21.B1850.f09_g17.CMIP6-piControl.001/ocn/proc/tseries/month_1/"
           "b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.SST.000101-009912.nc")

NINO34_LAT = (-5, 5)
NINO34_LON = (190, 240)  # 170W-120W, 0-360 convention

# Restrict the pz220f0b analysis window to exclude the ~20-yr IC-adjustment
# transient (the 1980s-90s "hump") and characterize the true quasi-equilibrium
# state instead.
YEAR_MIN, YEAR_MAX = 2000, 2045

def gmean(sst2d, w):
    s = np.asarray(sst2d, np.float64)
    m = np.isfinite(s)
    return float(np.nansum(np.where(m, s, 0) * w) / (w * m).sum())

def nino34_mask(tlat, tlon):
    return ((tlat >= NINO34_LAT[0]) & (tlat <= NINO34_LAT[1]) &
             (tlon >= NINO34_LON[0]) & (tlon <= NINO34_LON[1]))

# ---------- pz220f0b (POP, monthly files) ----------
mfiles = sorted(glob.glob(f"{PZ_DIR}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
def _in_window(f):
    m = re.search(r"\.(\d{4})-(\d{2})\.nc$", f)
    return YEAR_MIN <= int(m.group(1)) <= YEAR_MAX
mfiles = [f for f in mfiles if _in_window(f)]
g = xr.open_dataset(mfiles[0])
TAREA = np.nan_to_num(g["TAREA"].values.astype(np.float64))
TLAT, TLONG = g["TLAT"].values, g["TLONG"].values
nmask = nino34_mask(TLAT, TLONG)
n34_area = TAREA * nmask

pz_months, pz_gsst, pz_n34 = [], [], []
for f in mfiles:
    m = re.search(r"\.(\d{4})-(\d{2})\.nc$", f)
    y, mo = int(m.group(1)), int(m.group(2))
    sst = xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values
    pz_months.append((y, mo))
    pz_gsst.append(gmean(sst, TAREA))
    pz_n34.append(gmean(sst, n34_area))
pz_gsst, pz_n34 = np.array(pz_gsst), np.array(pz_n34)
pz_years = np.array([y + (mo - 0.5) / 12 for (y, mo) in pz_months])

# pz220f0b climatology + anomaly
pz_clim = np.array([pz_n34[[i for i, (_, mo) in enumerate(pz_months) if mo == m]].mean()
                     for m in range(1, 13)])
pz_anom = np.array([pz_n34[i] - pz_clim[mo - 1] for i, (_, mo) in enumerate(pz_months)])

pz_full_years = sorted({y for (y, mo) in pz_months
                         if sum(1 for (yy, mm) in pz_months if yy == y) == 12})
pz_ann_years, pz_ann_gsst = [], []
for y in pz_full_years:
    idx = [i for i, (yy, mo) in enumerate(pz_months) if yy == y]
    pz_ann_years.append(y)
    pz_ann_gsst.append(pz_gsst[idx].mean())

print(f"pz220f0b [{YEAR_MIN}-{YEAR_MAX} window]: {pz_months[0]} - {pz_months[-1]} "
      f"({len(pz_months)} months, {len(pz_full_years)} full years)")

# ---------- CESM2 piControl (100-yr block, same length as pz220f0b) ----------
n_months_needed = len(pz_months)
ds = xr.open_dataset(PI_FILE)
PI_TAREA = np.nan_to_num(ds["TAREA"].values.astype(np.float64))
PI_TLAT, PI_TLONG = ds["TLAT"].values, ds["TLONG"].values
pi_nmask = nino34_mask(PI_TLAT, PI_TLONG)
pi_n34_area = PI_TAREA * pi_nmask

sst_all = ds["SST"].isel(z_t=0).values[:n_months_needed]  # (time, nlat, nlon)
pi_gsst = np.array([gmean(sst_all[i], PI_TAREA) for i in range(n_months_needed)])
pi_n34 = np.array([gmean(sst_all[i], pi_n34_area) for i in range(n_months_needed)])
pi_months = [(1 + i // 12, 1 + i % 12) for i in range(n_months_needed)]  # model yr 1..

pi_clim = np.array([pi_n34[[i for i, (_, mo) in enumerate(pi_months) if mo == m]].mean()
                     for m in range(1, 13)])
pi_anom = np.array([pi_n34[i] - pi_clim[mo - 1] for i, (_, mo) in enumerate(pi_months)])
pi_years = np.array([y + (mo - 0.5) / 12 for (y, mo) in pi_months])

pi_full_years = sorted({y for (y, mo) in pi_months
                         if sum(1 for (yy, mm) in pi_months if yy == y) == 12})
pi_ann_years, pi_ann_gsst = [], []
for y in pi_full_years:
    idx = [i for i, (yy, mo) in enumerate(pi_months) if yy == y]
    pi_ann_years.append(y)
    pi_ann_gsst.append(pi_gsst[idx].mean())

print(f"piControl (model yr 1-{pi_months[-1][0]}): {n_months_needed} months")
print(f"Nino3.4 std: pz220f0b={pz_anom.std():.3f} K, piControl={pi_anom.std():.3f} K")

# align x-axis on "model year" (elapsed years from run start) for both, since
# piControl's calendar year is arbitrary
pz_elapsed = pz_years - pz_years[0]
pi_elapsed = pi_years - pi_years[0]
pz_ann_elapsed = np.array(pz_ann_years) - pz_years[0]
pi_ann_elapsed = np.array(pi_ann_years) - pi_years[0]

# ---------- figure ----------
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

ax = axes[0]
ax.axhline(0, color="k", lw=0.8)
ax.plot(pi_elapsed, pi_anom, color="#5b6167", lw=1.3,
        label=f"CESM2 piControl (mean {pi_anom.mean():+.3f}, std {pi_anom.std():.3f} degC)")
ax.plot(pz_elapsed, pz_anom, color="tab:orange", lw=1.2,
        label=f"pz220f0b unfinetuned, 7239950 (mean {pz_anom.mean():+.3f}, std {pz_anom.std():.3f} degC)")
ax.set_ylabel("Nino3.4 SST anomaly (degC)")
ax.set_xlabel("elapsed model year")
ax.set_title(f"Nino3.4 SST anomaly timeseries (pz220f0b {YEAR_MIN}-{YEAR_MAX}, post-transient)")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
months = np.arange(1, 13)
ax.plot(months, pi_clim, "o-", color="#5b6167",
        label=f"CESM2 piControl (mean {pi_clim.mean():.3f}, std {pi_clim.std():.3f} degC)")
ax.plot(months, pz_clim, "o-", color="tab:orange",
        label=f"pz220f0b unfinetuned, 7239950 (mean {pz_clim.mean():.3f}, std {pz_clim.std():.3f} degC)")
ax.set_xticks(months)
ax.set_xlabel("month")
ax.set_ylabel("Nino3.4 SST (degC)")
ax.set_title(f"Nino3.4 mean annual cycle (climatology, pz220f0b {YEAR_MIN}-{YEAR_MAX})")
ax.legend()
ax.grid(alpha=0.3)

pi_ann_gsst_a = np.array(pi_ann_gsst); pz_ann_gsst_a = np.array(pz_ann_gsst)
ax = axes[2]
ax.plot(pi_ann_elapsed, pi_ann_gsst, "o-", color="#5b6167",
        label=f"CESM2 piControl (mean {pi_ann_gsst_a.mean():.3f}, std {pi_ann_gsst_a.std():.3f} degC)")
ax.plot(pz_ann_elapsed, pz_ann_gsst, "d-", color="tab:orange", ms=3.5,
        label=f"pz220f0b unfinetuned, 7239950 (mean {pz_ann_gsst_a.mean():.3f}, std {pz_ann_gsst_a.std():.3f} degC)")
ax.set_xlabel("elapsed model year")
ax.set_ylabel("Global-mean SST (degC)")
ax.set_title(f"Global-mean SST, annual mean (pz220f0b {YEAR_MIN}-{YEAR_MAX})")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
out = "output/enso_pz220f0b_vs_picontrol.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")
