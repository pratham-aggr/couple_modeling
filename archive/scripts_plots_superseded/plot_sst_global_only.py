import glob, numpy as np, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_debiascheck/run"
LIVE = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_sensreg50yr/run"

ref = xr.open_dataset(sorted(glob.glob(f"{RUN}/*.pop.h.[0-9][0-9][0-9][0-9]-01.nc"))[0])
TAREA = np.nan_to_num(ref["TAREA"].values.astype(np.float64))

def wmean(sst):
    s = np.asarray(sst, np.float64)
    m = np.isfinite(s)
    w = TAREA * m
    return float((s[m]*w[m]).sum()/w[m].sum())

def year_mean(rundir, year):
    files = sorted(glob.glob(f"{rundir}/*.pop.h.{year}-[0-9][0-9].nc"))
    if len(files) < 12: return None
    arrs = [xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values for f in files]
    return wmean(np.nanmean(np.stack(arrs), axis=0))

years_bc = sorted(set(int(f.split(".pop.h.")[1][:4]) for f in glob.glob(f"{RUN}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc")))
years_live = sorted(set(int(f.split(".pop.h.")[1][:4]) for f in glob.glob(f"{LIVE}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc")))

bc = {y: year_mean(RUN, y) for y in years_bc}
bc = {y: v for y, v in bc.items() if v is not None}
lv = {y: year_mean(LIVE, y) for y in years_live}
lv = {y: v for y, v in lv.items() if v is not None}

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(sorted(bc), [bc[y] for y in sorted(bc)], 'o-', color='tab:red', ms=4, label='bias-corrected (7237474)')
ax.plot(sorted(lv), [lv[y] for y in sorted(lv)], 's-', color='tab:blue', ms=3, label='live baseline sensreg50yr (7232171)')
ax.set_xlabel("year")
ax.set_ylabel("Global-mean SST (degC)")
ax.set_title("Global-mean SST timeseries")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("output/sst_global_timeseries.png", dpi=130)
print("wrote output/sst_global_timeseries.png")
