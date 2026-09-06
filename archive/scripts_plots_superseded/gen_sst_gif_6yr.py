"""6-yr SST gif for a given rundir (RdYlBu_r, ocean-only kNN regrid + land mask +
TAREA-weighted stats), matching v15climwindGust_sst_5yr.gif style. Usage:
  python gen_sst_gif_6yr.py <rundir> <out.gif> "<title>"
"""
import sys, glob, io, re, numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PIL import Image
sys.path.insert(0,"/glade/u/home/praggarwal/couple/camulator_ud/climate")
from model_server import ScatterToRegular

RUNDIR, OUT, TITLE = sys.argv[1], sys.argv[2], sys.argv[3]
DST_NLAT,DST_NLON=360,720
DST_LATS=np.linspace(-89.75,89.75,DST_NLAT); DST_LONS=np.linspace(0.25,359.75,DST_NLON)
def uxyz(lon,lat):
    lon=np.radians(lon);lat=np.radians(lat)
    return np.stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)],axis=-1)
files=sorted([f for f in glob.glob(f"{RUNDIR}/*.pop.h.*.nc") if re.search(r"\.pop\.h\.\d{4}-\d{2}\.nc$",f)])
print(f"{len(files)} monthly files")
d0=xr.open_dataset(files[0],decode_times=False)
tlon=d0["TLONG"].values.astype("f8");tlat=d0["TLAT"].values.astype("f8")
kmt=d0["KMT"].values;tarea=d0["TAREA"].values.astype("f8")
ocean=(kmt>0).astype(np.int32);ob=ocean.astype(bool);aw=tarea[ob]
s2r=ScatterToRegular(tlon,tlat,ocean,DST_LATS,DST_LONS,k=4)
tree=cKDTree(uxyz(tlon.ravel(),tlat.ravel()))
dl,dn=np.meshgrid(DST_LATS,DST_LONS,indexing="ij")
_,ni=tree.query(uxyz(dn.ravel(),dl.ravel()),k=1)
dst_ocean=(ocean.ravel()[ni]>0).reshape(DST_NLAT,DST_NLON)
months=[];gm=[];gmax=[];frames=[]
cmap=plt.get_cmap("RdYlBu_r").copy();cmap.set_bad("dimgray")
for f in files:
    tag=f.split(".pop.h.")[-1].replace(".nc","")
    with xr.open_dataset(f,decode_times=False) as ds:
        temp=ds["TEMP"].isel(time=0,z_t=0).values.astype("f8")
    reg=s2r(np.nan_to_num(np.where(kmt>0,temp,np.nan),nan=0.0))
    reg=np.ma.masked_where(~dst_ocean,reg)
    months.append(tag);gm.append(float(np.average(temp[ob],weights=aw)));gmax.append(float(temp[ob].max()))
    fig,(a,b)=plt.subplots(2,1,figsize=(9,7.2),height_ratios=[3,1.4],facecolor="black")
    a.set_facecolor("dimgray")
    im=a.imshow(reg,origin="lower",extent=[0,360,-90,90],vmin=-2,vmax=32,cmap=cmap,aspect="auto")
    a.set_title(f"{TITLE}  {tag}",color="white",fontsize=12);a.set_xticks([]);a.set_yticks([])
    cb=fig.colorbar(im,ax=a,fraction=0.025,pad=0.02);cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.get_yticklabels(),color="white")
    b.set_facecolor("black");x=np.arange(len(months))
    b.plot(x,gm,color="tab:blue",label="global mean");b.plot(x,gmax,color="tab:red",label="max")
    b.set_xlim(0,len(files)-1);b.set_ylim(10,50)
    xt=[i for i,m in enumerate(months) if m.endswith("-01")]
    b.set_xticks(xt);b.set_xticklabels([months[i][:4] for i in xt],color="white")
    b.tick_params(colors="white");b.set_ylabel("degC",color="white")
    for sp in b.spines.values():sp.set_color("white")
    b.legend(loc="upper left",facecolor="black",labelcolor="white",fontsize=8,frameon=False)
    fig.tight_layout();buf=io.BytesIO();fig.savefig(buf,format="png",dpi=90,facecolor=fig.get_facecolor())
    plt.close(fig);buf.seek(0);frames.append(Image.open(buf).convert("P",palette=Image.ADAPTIVE))
frames[0].save(OUT,save_all=True,append_images=frames[1:],duration=180,loop=0,optimize=False)
print(f"wrote {OUT} ({len(frames)} frames)")
