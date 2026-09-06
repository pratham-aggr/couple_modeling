"""SST gif for the CESM2 piControl run, styled to match gen_sst_gif_6yr.py /
gen_sst_gif_cam6.py (RdYlBu_r, ocean-only kNN regrid + land mask, TAREA-weighted
global-mean/max trace underneath).

piControl `tos` is on the native gn grid = gx1v7 (384x320 curvilinear), the same
grid the coupled sims run on, so it uses the SAME ScatterToRegular kNN regrid as
the sim gifs -- not the direct imshow the CAM6 one uses (that field is a regular
lat/lon grid).  Land/mask comes from KMT on a sim history file, same grid.

The record is 1200 model years; the default window is a mid-record, fully
equilibrated 35 years, matching cam6_sst.gif's 420 frames.

Optional 5th arg LABEL_START remaps the displayed year so the animation lines up
with a sim's calendar (e.g. 1980) while the true model year stays in the frame
title -- piControl is unforced and on model years, so the mapping is elapsed-year
alignment only, not a claim of shared dates.

Usage: python gen_sst_gif_picontrol.py <out.gif> "<title>" [y0] [y1] [label_start]
"""
import sys, os, io, re, glob
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PIL import Image
sys.path.insert(0, "/glade/u/home/praggarwal/couple/camulator_ud/climate")
from model_server import ScatterToRegular

PICTL_DIR = ("/glade/collections/cmip/CMIP6/CMIP/NCAR/CESM2/piControl/"
             "r1i1p1f1/Omon/tos/gn/latest")
GRID_REF = ("/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_sensreg50yr/"
            "run/g.e21.MEMO_GIAF_v01.pop.h.1980-01.nc")

OUT   = sys.argv[1]
TITLE = sys.argv[2]
Y0    = int(sys.argv[3]) if len(sys.argv) > 3 else 501     # mid-record, equilibrated
Y1    = int(sys.argv[4]) if len(sys.argv) > 4 else 535
LBL0  = int(sys.argv[5]) if len(sys.argv) > 5 else None   # display-year for Y0

DST_NLAT, DST_NLON = 360, 720
DST_LATS = np.linspace(-89.75, 89.75, DST_NLAT)
DST_LONS = np.linspace(0.25, 359.75, DST_NLON)

def uxyz(lon, lat):
    lon = np.radians(lon); lat = np.radians(lat)
    return np.stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)], axis=-1)

g0 = xr.open_dataset(GRID_REF, decode_times=False)
tlon = g0["TLONG"].values.astype("f8"); tlat = g0["TLAT"].values.astype("f8")
kmt = g0["KMT"].values; tarea = g0["TAREA"].values.astype("f8")
ocean = (kmt > 0).astype(np.int32); ob = ocean.astype(bool); aw = tarea[ob]
s2r = ScatterToRegular(tlon, tlat, ocean, DST_LATS, DST_LONS, k=4)
tree = cKDTree(uxyz(tlon.ravel(), tlat.ravel()))
dl, dn = np.meshgrid(DST_LATS, DST_LONS, indexing="ij")
_, ni = tree.query(uxyz(dn.ravel(), dl.ravel()), k=1)
dst_ocean = (ocean.ravel()[ni] > 0).reshape(DST_NLAT, DST_NLON)

# --- collect the (file, time-index) pairs covering [Y0, Y1] ---
recs = []
for f in sorted(glob.glob(f"{PICTL_DIR}/*.nc")):
    m = re.search(r"_gn_(\d{4})\d\d-(\d{4})\d\d\.nc$", f)
    fy0, fy1 = int(m.group(1)), int(m.group(2))
    if fy1 < Y0 or fy0 > Y1:
        continue
    for y in range(max(fy0, Y0), min(fy1, Y1) + 1):
        for mo in range(1, 13):
            recs.append((f, ((y - fy0) * 12) + (mo - 1), y, mo))
print(f"{len(recs)} monthly frames  [{Y0}-{Y1}] from {len({r[0] for r in recs})} file(s)")

months, gm, gmax, frames = [], [], [], []
cmap = plt.get_cmap("RdYlBu_r").copy(); cmap.set_bad("dimgray")
open_f, ds = None, None
for f, i, y, mo in recs:
    if f != open_f:
        if ds is not None: ds.close()
        ds = xr.open_dataset(f, decode_times=False); open_f = f
    temp = ds["tos"].isel(time=i).values.astype("f8")
    if LBL0 is None:
        tag = f"{y:04d}-{mo:02d}"
        lyr = y
    else:
        lyr = LBL0 + (y - Y0)
        tag = f"{lyr:04d}-{mo:02d}  (model yr {y:04d})"
    months.append(f"{lyr:04d}-{mo:02d}")
    reg = s2r(np.nan_to_num(np.where(kmt > 0, temp, np.nan), nan=0.0))
    reg = np.ma.masked_where(~dst_ocean, reg)
    gm.append(float(np.average(np.nan_to_num(temp[ob]), weights=aw)))
    gmax.append(float(np.nanmax(temp[ob])))

    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 7.2), height_ratios=[3, 1.4],
                              facecolor="black")
    a.set_facecolor("dimgray")
    im = a.imshow(reg, origin="lower", extent=[0, 360, -90, 90],
                  vmin=-2, vmax=32, cmap=cmap, aspect="auto")
    a.set_title(f"{TITLE}  {tag}", color="white", fontsize=12)
    a.set_xticks([]); a.set_yticks([])
    cb = fig.colorbar(im, ax=a, fraction=0.025, pad=0.02)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.get_yticklabels(), color="white")
    b.set_facecolor("black"); x = np.arange(len(months))
    b.plot(x, gm, color="tab:blue", label="global mean")
    b.plot(x, gmax, color="tab:red", label="max")
    b.set_xlim(0, len(recs) - 1); b.set_ylim(10, 50)
    xt = [k for k, mm in enumerate(months) if mm.endswith("-01") and int(mm[:4]) % 5 == 0]
    b.set_xticks(xt); b.set_xticklabels([months[k][:4] for k in xt], color="white")
    b.tick_params(colors="white"); b.set_ylabel("degC", color="white")
    for sp in b.spines.values(): sp.set_color("white")
    b.legend(loc="upper left", facecolor="black", labelcolor="white",
             fontsize=8, frameon=False)
    fig.tight_layout(); buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, facecolor=fig.get_facecolor())
    plt.close(fig); buf.seek(0)
    frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
if ds is not None: ds.close()

frames[0].save(OUT, save_all=True, append_images=frames[1:],
               duration=180, loop=0, optimize=False)
print(f"wrote {OUT} ({len(frames)} frames)")
print(f"  global-mean SST {np.mean(gm):.3f} degC   max {np.max(gmax):.2f} degC")
