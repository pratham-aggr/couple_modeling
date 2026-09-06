"""SST gif for the CAM6/CREDIT run, styled to match gen_sst_gif_6yr.py output
(RdYlBu_r, land masked dimgray, area-weighted global mean + max trace).

CAM6 SST is a regular 192x288 lat/lon field in K, filled with 273.0 over land
and ice, so land is masked with LANDFRAC<0.5 -- no regrid is needed.

Usage: python gen_sst_gif_cam6.py <out.gif> "<title>" [y0] [y1]
"""
import sys, os, io, numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image

CAM6_DIR = "/glade/derecho/scratch/wchapman/b_credit_runs"
OUT   = sys.argv[1]
TITLE = sys.argv[2]
Y0    = int(sys.argv[3]) if len(sys.argv) > 3 else 1980
Y1    = int(sys.argv[4]) if len(sys.argv) > 4 else 2014

files = {y: os.path.join(CAM6_DIR,
         f"b.e21.CREDIT_climate_branch_1980_{y}_zmdata_ERA5scaled_zmdata_Qtot.zarr")
         for y in range(Y0, Y1 + 1)}
files = {y: f for y, f in files.items() if os.path.isdir(f)}
print(f"{len(files)} annual zarrs  [{min(files)}-{max(files)}]")

d0 = xr.open_zarr(next(iter(files.values())), consolidated=False)
lat, lon = d0["latitude"].values, d0["longitude"].values
lf = d0["LANDFRAC"].isel(time=0).values
ocean = lf < 0.5                                   # time-invariant land mask
latm, _ = np.meshgrid(lat, lon, indexing="ij")
w = (np.cos(np.deg2rad(latm)) * ocean)[ocean]      # area weights on ocean points

months, gm, gmax, frames = [], [], [], []
cmap = plt.get_cmap("RdYlBu_r").copy(); cmap.set_bad("dimgray")
nframes = 12 * len(files)

for y in sorted(files):
    ds = xr.open_zarr(files[y], consolidated=False)["SST"]
    mon = np.array([int(str(t)[5:7]) for t in ds["time"].values])
    for m in range(1, 13):
        sel = np.where(mon == m)[0]
        if not sel.size: continue
        fld = ds.isel(time=slice(sel[0], sel[-1] + 1)).mean("time").values - 273.15
        tag = f"{y}-{m:02d}"
        reg = np.ma.masked_where(~ocean, fld)
        months.append(tag)
        gm.append(float(np.average(fld[ocean], weights=w)))
        gmax.append(float(fld[ocean].max()))

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
        b.set_xlim(0, nframes - 1); b.set_ylim(10, 50)
        xt = [i for i, mm in enumerate(months)
              if mm.endswith("-01") and int(mm[:4]) % 5 == 0]
        b.set_xticks(xt); b.set_xticklabels([months[i][:4] for i in xt], color="white")
        b.tick_params(colors="white"); b.set_ylabel("degC", color="white")
        for sp in b.spines.values(): sp.set_color("white")
        b.legend(loc="upper left", facecolor="black", labelcolor="white",
                 fontsize=8, frameon=False)
        fig.tight_layout(); buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, facecolor=fig.get_facecolor())
        plt.close(fig); buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))

frames[0].save(OUT, save_all=True, append_images=frames[1:],
               duration=180, loop=0, optimize=False)
print(f"wrote {OUT} ({len(frames)} frames)")
