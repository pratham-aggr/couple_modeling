"""
Animate the standalone MEMO->POP run that blew up ~9 months in (memo_pop_standalone).
Shows the SLOW drift that precedes the crash, not just the final spike:
  - SST anomaly vs start (diverging, tight +/-3C)  -> reveals where heat accumulates
  - tight zoom on the crash cell neighborhood
  - max-SST line (symlog) with a marker
The hottest cell each frame is marked with an X on both maps.
"""
import glob
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

RUN = "/glade/derecho/scratch/praggarwal/memo_pop_standalone/run"
OUT = "/glade/u/home/praggarwal/couple/output/memo_pop_blowup.gif"
FILL = 1e30
STRIDE = 2
ZHALF = 30          # half-width (cells) of the crash zoom window

files = sorted(glob.glob(f"{RUN}/g.e21.MEMO_GIAF_v01.pop.h.nday1.*.nc"))
sst_list, time_list, tlat = [], [], None
for f in files:
    d = xr.open_dataset(f, decode_timedelta=False)
    sst_list.append(d["SST"].values.astype("float32"))
    time_list.append(d["time"].values)
    if tlat is None:
        tlat = d["TLAT"].values
    d.close()
sst = np.concatenate(sst_list, axis=0)
times = np.concatenate(time_list, axis=0)
order = np.argsort(times); sst = sst[order]; times = times[order]
_, keep = np.unique(times, return_index=True); keep = np.sort(keep)
sst = sst[keep]; times = times[keep]
sst = np.where(sst > FILL, np.nan, sst)
T, NJ, NI = sst.shape

anom = sst - sst[0]                                   # drift relative to start
maxsst = np.nanmax(sst.reshape(T, -1), axis=1)
day = np.arange(T)

# crash cell = location of the global max on the last finite frame
last_ok = max(i for i in range(T) if np.isfinite(sst[i]).any())
cj, ci = np.unravel_index(np.nanargmax(np.nan_to_num(sst[last_ok], nan=-1e9)), (NJ, NI))
cell_sst = sst[:, cj, ci]                              # the crash cell's own SST(t)
print(f"crash cell j={cj} i={ci} lat={tlat[cj,ci]:.1f}  peak {maxsst[last_ok]:.0f}C "
      f"{str(times[last_ok])[:10]}", flush=True)
j0, j1 = max(0, cj - ZHALF), min(NJ, cj + ZHALF)
i0, i1 = max(0, ci - ZHALF), min(NI, ci + ZHALF)

# per-frame hottest-cell location (for the moving marker)
flat_arg = np.array([np.nanargmax(np.nan_to_num(sst[t], nan=-1e9)) for t in range(T)])
hj, hi = np.unravel_index(flat_arg, (NJ, NI))

fr = np.arange(0, T, STRIDE)
print(f"{len(fr)} frames", flush=True)

AV = 1.5   # anomaly color range +/- 1.5 C (tight: surfaces the slow drift)
fig = plt.figure(figsize=(9.5, 6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.34, wspace=0.24)
ax_g = fig.add_subplot(gs[0, :])
ax_z = fig.add_subplot(gs[1, 0])
ax_t = fig.add_subplot(gs[1, 1])

m_g = ax_g.imshow(anom[0], origin="lower", cmap="RdBu_r", vmin=-AV, vmax=AV, aspect="auto")
ax_g.set_title("SST anomaly vs. day 0  (field stays quiet — no precursor drift)")
ax_g.set_xticks([]); ax_g.set_yticks([])
cb = fig.colorbar(m_g, ax=ax_g, pad=0.01, shrink=0.85); cb.set_label("ΔSST (°C)")
mk_g, = ax_g.plot([], [], "kx", ms=8, mew=2)            # hottest cell
# box around the zoom region
ax_g.add_patch(plt.Rectangle((i0, j0), i1 - i0, j1 - j0, fill=False, ec="lime", lw=1.5))

m_z = ax_z.imshow(anom[0, j0:j1, i0:i1], origin="lower", cmap="RdBu_r",
                  vmin=-AV, vmax=AV, aspect="auto", extent=[i0, i1, j0, j1])
ax_z.set_title(f"Zoom on crash cell (lat≈{tlat[cj,ci]:.0f}°)")
ax_z.set_xticks([]); ax_z.set_yticks([])
mk_z, = ax_z.plot([], [], "kx", ms=10, mew=2)

ax_t.plot(day, cell_sst, color="C3", lw=1.6, label="crash cell")
ax_t.plot(day, maxsst, color="0.7", lw=1.0, label="global max")
ax_t.axhline(33, color="k", ls=":", lw=0.8); ax_t.text(2, 36, "physical ~33°C", fontsize=8)
ax_t.set_yscale("symlog", linthresh=40)
ax_t.legend(fontsize=7, loc="upper left")
dot, = ax_t.plot([], [], "ko", ms=6)               # marker rides the crash-cell line
vln = ax_t.axvline(0, color="r", lw=1)
ax_t.set_xlabel("day"); ax_t.set_ylabel("SST (°C, symlog)")
ax_t.set_title(f"Crash cell ({tlat[cj,ci]:.0f}°N): flat ~12°C, then 1-step blow-up")

sup = fig.suptitle("", fontsize=12, y=0.98)
fig.text(0.5, 0.005, "daily-mean output — the sub-daily onset (tracer-CFL) is not resolved; "
         "cell = where the final overflow lands, not necessarily the physical trigger",
         ha="center", fontsize=7, color="0.4")

def update(k):
    i = fr[k]
    m_g.set_data(anom[i]); m_z.set_data(anom[i, j0:j1, i0:i1])
    mk_g.set_data([hi[i]], [hj[i]]); mk_z.set_data([hi[i]], [hj[i]])
    dot.set_data([day[i]], [cell_sst[i]]); vln.set_xdata([day[i], day[i]])
    sup.set_text(f"MEMO→POP standalone: stable ~9 months, then sudden 1-step blow-up | "
                 f"{str(times[i])[:10]} (day {i}) | max SST = {maxsst[i]:.1f}°C")
    return m_g, m_z, mk_g, mk_z, dot, vln, sup

ani = animation.FuncAnimation(fig, update, frames=len(fr), blit=False)
ani.save(OUT, writer=animation.PillowWriter(fps=10), dpi=80)
print("wrote", OUT, flush=True)
