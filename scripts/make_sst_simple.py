"""Simple single-panel global SST animation for the crashed standalone run."""
import glob
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

RUN = "/glade/derecho/scratch/praggarwal/memo_pop_standalone/run/archive_run1"
OUT = "/glade/u/home/praggarwal/couple/output/memo_pop_sst_simple.gif"
FILL = 1e30
STRIDE = 2

files = sorted(glob.glob(f"{RUN}/g.e21.MEMO_GIAF_v01.pop.h.nday1.*.nc"))
S, TM = [], []
for f in files:
    d = xr.open_dataset(f, decode_timedelta=False)
    S.append(d["SST"].values.astype("float32")); TM.append(d["time"].values); d.close()
sst = np.concatenate(S); tm = np.concatenate(TM)
o = np.argsort(tm); sst = sst[o]; tm = tm[o]
_, k = np.unique(tm, return_index=True); k = np.sort(k); sst = sst[k]; tm = tm[k]
sst = np.where(sst > FILL, np.nan, sst)
fr = np.arange(0, sst.shape[0], STRIDE)
print(f"{len(fr)} frames {str(tm[0])[:10]}..{str(tm[-1])[:10]}", flush=True)

fig, ax = plt.subplots(figsize=(8, 4.5))
im = ax.imshow(sst[0], origin="lower", cmap="turbo", vmin=-2, vmax=32, aspect="auto")
ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02); cb.set_label("SST (°C)")
ttl = ax.set_title("")

def update(kk):
    i = fr[kk]
    im.set_data(sst[i])
    mx = np.nanmax(sst[i])
    ttl.set_text(f"{str(tm[i])[:10]}   (max {mx:.1f}°C)")
    return im, ttl

ani = animation.FuncAnimation(fig, update, frames=len(fr), blit=False)
ani.save(OUT, writer=animation.PillowWriter(fps=10), dpi=85)
print("wrote", OUT, flush=True)
