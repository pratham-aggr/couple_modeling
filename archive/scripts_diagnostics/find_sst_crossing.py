import glob, re
import numpy as np, xarray as xr

FASST_RUN = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw_iceloop/run"
PICTL_CACHE = "output/picontrol_gmsst_annual.npz"

def ym(f):
    m = re.search(r"\.(\d{4})-(\d{2})\.nc$", f)
    return int(m.group(1)), int(m.group(2))

def gmean(sst2d, w):
    s = np.asarray(sst2d, np.float64); m = np.isfinite(s)
    return float(np.nansum(np.where(m, s, 0) * w) / (w * m).sum())

mfiles = sorted(glob.glob(f"{FASST_RUN}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
mkeys = [ym(f) for f in mfiles]
g = xr.open_dataset(mfiles[0])
TAREA = np.nan_to_num(g["TAREA"].values.astype(np.float64))
msst_by_ym = {k: gmean(xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values, TAREA)
              for f, k in zip(mfiles, mkeys)}
years_avail = sorted({y for (y, m) in mkeys})
full_years = [y for y in years_avail if all((y, m) in msst_by_ym for m in range(1, 13))]
m_annual = np.array([np.mean([msst_by_ym[(y, m)] for m in range(1, 13)]) for y in full_years])

z = np.load(PICTL_CACHE)
pic_annual = z["annual"]
n = min(len(pic_annual), len(full_years))
diff = m_annual[:n] - pic_annual[:n]
years = np.array(full_years[:n])

print("year   FASST   piCtl   diff")
for i in range(min(15, n)):
    print(f"{years[i]}  {m_annual[i]:.3f}  {pic_annual[i]:.3f}  {diff[i]:+.3f}")

sign0 = np.sign(diff[0])
cross_idx = None
for i in range(1, n):
    if np.sign(diff[i]) != sign0 and np.sign(diff[i]) != 0:
        cross_idx = i
        break
if cross_idx:
    print(f"\nFirst crossing between {years[cross_idx-1]} (diff {diff[cross_idx-1]:+.3f}) "
          f"and {years[cross_idx]} (diff {diff[cross_idx]:+.3f})")
else:
    print("\nNo sign change found in the overlapping window")

for start in (1998, 1999):
    idx = np.where(years == start)[0]
    if len(idx) == 0:
        continue
    i0 = idx[0]
    f_seg = m_annual[i0:n]; p_seg = pic_annual[i0:n]
    print(f"\nFrom {start} onward ({n-i0} yr, through {years[n-1]}):")
    print(f"  FASST      mean={f_seg.mean():.3f}  std={f_seg.std():.3f}")
    print(f"  piControl  mean={p_seg.mean():.3f}  std={p_seg.std():.3f}")
