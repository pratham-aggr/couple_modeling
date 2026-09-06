import glob, numpy as np, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_debiascheck/run"
LIVE = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_sensreg50yr/run"

def load_ref(rundir):
    fs = sorted(glob.glob(f"{rundir}/*.pop.h.[0-9][0-9][0-9][0-9]-01.nc"))
    return xr.open_dataset(fs[0])

ref = load_ref(RUN)
TAREA = np.nan_to_num(ref["TAREA"].values.astype(np.float64))
TLAT = ref["TLAT"].values

def wmean(sst, mask):
    s = np.asarray(sst, np.float64)
    m = np.isfinite(s) & mask
    w = TAREA * m
    return float((s[m]*w[m]).sum()/w[m].sum())

regions = {"Tropics (30S-30N)": (TLAT>=-30)&(TLAT<=30),
           "Southern Ocean (<-45)": TLAT<-45,
           "NH high-lat (>45)": TLAT>45,
           "Global": np.ones_like(TLAT, dtype=bool)}

def year_means(rundir, year):
    files = sorted(glob.glob(f"{rundir}/*.pop.h.{year}-[0-9][0-9].nc"))
    if len(files) < 12: return None
    arrs = [xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values for f in files]
    mean = np.nanmean(np.stack(arrs), axis=0)
    return {n: wmean(mean, m) for n, m in regions.items()}

years_bc = sorted(set(int(f.split(".pop.h.")[1][:4])
                for f in glob.glob(f"{RUN}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc")))
years_live = sorted(set(int(f.split(".pop.h.")[1][:4])
                for f in glob.glob(f"{LIVE}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc")))

data_bc, data_live = {}, {}
for y in years_bc:
    d = year_means(RUN, y)
    if d: data_bc[y] = d
for y in years_live:
    l = year_means(LIVE, y)
    if l: data_live[y] = l

common = sorted(set(data_bc) & set(data_live))
print(f"bias-corrected years: {years_bc[0]}-{years_bc[-1]} ({len(data_bc)})")
print(f"live baseline years : {years_live[0]}-{years_live[-1]} ({len(data_live)})")
print(f"common years for offset: {len(common)}")

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
panels = ["Tropics (30S-30N)", "Southern Ocean (<-45)", "NH high-lat (>45)", "Global"]
for ax, region in zip(axes.flat, panels):
    yrs_bc = sorted(data_bc.keys())
    yrs_lv = sorted(data_live.keys())
    ax.plot(yrs_bc, [data_bc[y][region] for y in yrs_bc], 'o-', color='tab:red', label='bias-corrected (7237474)', ms=3)
    ax.plot(yrs_lv, [data_live[y][region] for y in yrs_lv], 's-', color='tab:blue', label='live baseline sensreg50yr (7232171)', ms=3)
    ax.set_title(region)
    ax.set_xlabel("year")
    ax.set_ylabel("SST (degC)")
    ax.grid(alpha=0.3)
axes.flat[0].legend(fontsize=8, loc='best')
fig.suptitle(f"MEMO coupled: bias-corrected (Qnet FSDS+LHFLX debias) vs live sensreg50yr baseline\n"
             f"bias-corr through {years_bc[-1]}, baseline through {years_live[-1]}")
fig.tight_layout()
out = "output/biascorr50_vs_baseline_progress.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")

# offset plot
fig2, ax2 = plt.subplots(figsize=(8,5))
offs = [data_bc[y]["Global"] - data_live[y]["Global"] for y in common]
ax2.plot(common, offs, 'o-', color='tab:purple')
ax2.axhline(0, color='k', lw=0.5)
ax2.set_xlabel("year"); ax2.set_ylabel("Global SST offset, bias-corr minus baseline (K)")
ax2.set_title("Bias-corrected run's global-mean offset from baseline over time")
ax2.grid(alpha=0.3)
fig2.tight_layout()
out2 = "output/biascorr50_offset_progress.png"
fig2.savefig(out2, dpi=130)
print(f"wrote {out2}")
