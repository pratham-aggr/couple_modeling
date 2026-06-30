"""SST map + global-mean SST timeseries animation (two panels).

Usage:
    python make_sst_with_ts.py <glob> <out.gif> <label> [stride]
"""
import sys, glob, numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

GLOB, OUT, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
STRIDE = int(sys.argv[4]) if len(sys.argv) > 4 else 2
FILL = 1e30

files = sorted(glob.glob(GLOB))
S, TM = [], []; area = None
for f in files:
    d = xr.open_dataset(f, decode_timedelta=False)
    S.append(d["SST"].values.astype("float32")); TM.append(d["time"].values)
    if area is None and "TAREA" in d:
        area = d["TAREA"].values.astype("float64")
    d.close()
sst = np.concatenate(S); tm = np.concatenate(TM)
o = np.argsort(tm); sst = sst[o]; tm = tm[o]
_, k = np.unique(tm, return_index=True); k = np.sort(k); sst = sst[k]; tm = tm[k]
sst = np.where((sst > FILL) | (sst < -1e29), np.nan, sst)
# area-weighted global-mean SST(t) (falls back to unweighted if no TAREA)
if area is not None:
    valid = np.isfinite(sst)
    w = np.where(valid, area[None], 0.0)
    gmean = np.nansum(np.where(valid, sst, 0.0) * area[None], axis=(1, 2)) / w.sum(axis=(1, 2))
else:
    gmean = np.nanmean(sst.reshape(sst.shape[0], -1), axis=1)
gmax = np.nanmax(sst.reshape(sst.shape[0], -1), axis=1)          # max SST(t) — spikes at a blowup
tnum = np.arange(sst.shape[0])
fr = np.arange(0, sst.shape[0], STRIDE)
# detect the blowup frame (first time max SST leaves the physical range)
crash_ix = next((j for j in range(len(gmax)) if gmax[j] > 60.0), None)
print(f"{len(fr)} frames {str(tm[0])[:10]}..{str(tm[-1])[:10]}  crash_ix={crash_ix}", flush=True)

fig, (axm, axt) = plt.subplots(2, 1, figsize=(8, 6.4),
                               gridspec_kw={"height_ratios": [3, 1.1]})
im = axm.imshow(sst[0], origin="lower", cmap="turbo", vmin=-2, vmax=32, aspect="auto")
axm.set_xticks([]); axm.set_yticks([])
cb = fig.colorbar(im, ax=axm, shrink=0.85, pad=0.02); cb.set_label("SST (°C)")
ttl = axm.set_title("")

# single LOG axis showing BOTH the area-wtd mean (blue, ~stable) and max SST (red),
# so the blowup shows as a vertical red spike that dwarfs the flat mean.
axt.plot(tnum, gmean, color="0.8", lw=0.8)
axt.plot(tnum, gmax, color="0.85", lw=0.8)
lnm, = axt.plot([], [], color="C0", lw=1.8, label="area-wtd mean SST")
lnx, = axt.plot([], [], color="C3", lw=1.8, label="max SST")
pt,  = axt.plot([], [], "o", color="C0", ms=4)
axt.set_xlim(0, sst.shape[0] - 1)
# scale y to the *physical* trend (ignore the blowup spike so the normal
# mean/max curves stay visible); the crash spike just shoots off the top.
phys_max = np.nanmax(gmax[:crash_ix]) if crash_ix else np.nanmax(gmax)
axt.set_ylim(0, 45.0)   # fixed scale so standalone & full-coupler are directly comparable
axt.set_ylabel("SST (°C)"); axt.set_xlabel("time step")
axt.grid(alpha=0.3, which="both")
axt.legend(loc="upper left", fontsize=8, ncol=2)
if crash_ix is not None:
    axt.axvline(crash_ix, color="C3", ls="--", lw=1.0, alpha=0.7)
    ytop = axt.get_ylim()[1]
    axt.annotate(f"CRASH {str(tm[crash_ix])[:10]}\n(max {gmax[crash_ix]:.0f}°C, off-scale)",
                 xy=(crash_ix, ytop * 0.95), xytext=(-8, 0),
                 textcoords="offset points", ha="right", va="top",
                 color="C3", fontsize=8, fontweight="bold")
fig.suptitle(LABEL, fontsize=12, y=0.98)

def update(kk):
    i = fr[kk]
    im.set_data(sst[i])
    ttl.set_text(f"{str(tm[i])[:10]}   (max {np.nanmax(sst[i]):.1f}°C)")
    lnm.set_data(tnum[:i+1], gmean[:i+1]); pt.set_data([i], [gmean[i]])
    lnx.set_data(tnum[:i+1], np.maximum(gmax[:i+1], 1.0))
    return im, ttl, lnm, pt, lnx

ani = animation.FuncAnimation(fig, update, frames=len(fr), blit=False)
fig.tight_layout(rect=[0, 0, 1, 0.96])
ani.save(OUT, writer=animation.PillowWriter(fps=10), dpi=85)
print("wrote", OUT, flush=True)
