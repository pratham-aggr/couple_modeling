"""Diagnose WHY pz220f0b's Nino3.4/global-SST diverge from piControl: amplitude,
persistence, spectral peak, and drift-trend contamination."""
import glob, re
import numpy as np, xarray as xr

PZ_DIR = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw_iceloop/run"
PI_FILE = ("/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/"
           "b.e21.B1850.f09_g17.CMIP6-piControl.001/ocn/proc/tseries/month_1/"
           "b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.SST.000101-009912.nc")
NINO34_LAT, NINO34_LON = (-5, 5), (190, 240)

def gmean(sst2d, w):
    s = np.asarray(sst2d, np.float64); m = np.isfinite(s)
    return float(np.nansum(np.where(m, s, 0) * w) / (w * m).sum())

def nino34_mask(tlat, tlon):
    return ((tlat >= NINO34_LAT[0]) & (tlat <= NINO34_LAT[1]) &
             (tlon >= NINO34_LON[0]) & (tlon <= NINO34_LON[1]))

mfiles = sorted(glob.glob(f"{PZ_DIR}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
g = xr.open_dataset(mfiles[0])
TAREA = np.nan_to_num(g["TAREA"].values.astype(np.float64))
nmask = nino34_mask(g["TLAT"].values, g["TLONG"].values)
n34_area = TAREA * nmask

pz_months, pz_gsst, pz_n34 = [], [], []
for f in mfiles:
    mre = re.search(r"\.(\d{4})-(\d{2})\.nc$", f)
    y, mo = int(mre.group(1)), int(mre.group(2))
    sst = xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values
    pz_months.append((y, mo)); pz_gsst.append(gmean(sst, TAREA)); pz_n34.append(gmean(sst, n34_area))
pz_gsst, pz_n34 = np.array(pz_gsst), np.array(pz_n34)

n = len(pz_months)
ds = xr.open_dataset(PI_FILE)
PI_TAREA = np.nan_to_num(ds["TAREA"].values.astype(np.float64))
pi_nmask = nino34_mask(ds["TLAT"].values, ds["TLONG"].values)
pi_area = PI_TAREA * pi_nmask
sst_all = ds["SST"].isel(z_t=0).values[:n]
pi_gsst = np.array([gmean(sst_all[i], PI_TAREA) for i in range(n)])
pi_n34 = np.array([gmean(sst_all[i], pi_area) for i in range(n)])

def clim_anom(x, months):
    clim = np.array([x[[i for i,(_,mo) in enumerate(months) if mo==m]].mean() for m in range(1,13)])
    return np.array([x[i]-clim[mo-1] for i,(_,mo) in enumerate(months)]), clim

pz_months_pi = [(1+i//12, 1+i%12) for i in range(n)]
pz_anom, pz_clim = clim_anom(pz_n34, pz_months)
pi_anom, pi_clim = clim_anom(pi_n34, pz_months_pi)

print("=== amplitude ===")
print(f"pz220f0b Nino3.4 std: {pz_anom.std():.3f} K   max |anom|: {np.abs(pz_anom).max():.3f} K")
print(f"piControl Nino3.4 std: {pi_anom.std():.3f} K   max |anom|: {np.abs(pi_anom).max():.3f} K")
print(f"variance ratio pz/pi: {(pz_anom.var()/pi_anom.var()):.2f}x")

print("\n=== persistence (lag-1 monthly autocorrelation) ===")
def ac1(x):
    return float(np.corrcoef(x[:-1], x[1:])[0,1])
print(f"pz220f0b: {ac1(pz_anom):.3f}   piControl: {ac1(pi_anom):.3f}")

print("\n=== teleconnection: Nino3.4 anomaly vs GLOBAL-mean SST anomaly correlation ===")
pz_gsst_anom, _ = clim_anom(pz_gsst, pz_months)
pi_gsst_anom, _ = clim_anom(pi_gsst, pz_months_pi)
print(f"pz220f0b corr(n34,global): {np.corrcoef(pz_anom, pz_gsst_anom)[0,1]:.3f}")
print(f"piControl corr(n34,global): {np.corrcoef(pi_anom, pi_gsst_anom)[0,1]:.3f}")

print("\n=== dominant spectral period (annual-mean N34, Lomb-free simple FFT) ===")
def dominant_period_years(x_monthly):
    x = x_monthly - x_monthly.mean()
    f = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1/12.0)  # cycles/yr
    p = np.abs(f)**2
    p[0] = 0
    k = np.argmax(p[1:]) + 1
    return 1.0/freqs[k] if freqs[k] > 0 else np.inf, p[k]/p.sum()
per_pz, frac_pz = dominant_period_years(pz_anom)
per_pi, frac_pi = dominant_period_years(pi_anom)
print(f"pz220f0b dominant period: {per_pz:.2f} yr (power frac {frac_pz:.3f})")
print(f"piControl dominant period: {per_pi:.2f} yr (power frac {frac_pi:.3f})")

print("\n=== first-decade vs last-decade drift (global mean, contaminates seasonal 'anomaly') ===")
yrs = np.array([y for y,_ in pz_months])
def decade_mean(x, y0, y1):
    idx = [i for i,(y,_) in enumerate(pz_months) if y0<=y<y1]
    return x[idx].mean()
print(f"pz220f0b global SST: 1980-1990 mean {decade_mean(pz_gsst,1980,1990):.3f}, "
      f"2010-2020 mean {decade_mean(pz_gsst,2010,2020):.3f}, "
      f"2037-2047 mean {decade_mean(pz_gsst,2037,2047):.3f}")
print(f"piControl (elapsed) global SST: yr1-10 mean {pi_gsst[:120].mean():.3f}, "
      f"yr30-40 mean {pi_gsst[360:480].mean():.3f}, yr57-67 mean {pi_gsst[684:804].mean():.3f}")
