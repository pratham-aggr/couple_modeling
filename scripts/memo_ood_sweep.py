import numpy as np, torch, json, sys, xarray as xr, cftime
sys.path.insert(0,"camulator_ud/climate")
import model_server as ms
OUT="/glade/u/home/praggarwal/couple/output/output_unet_mem24h_dsst_temporal_radprecip"
mcfg=json.load(open(f"{OUT}/model_config.json"))
norm=ms.Normalizer.load(f"{OUT}/normalizer.npz")
model=ms.UNet(n_in=mcfg["n_in"],n_out=mcfg["n_out"],base=mcfg["base"])
model.load_state_dict(torch.load(f"{OUT}/best_model.pt",map_location="cpu",weights_only=True)); model.eval()
F="/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/b.e21.CREDIT_climate_branch_1980_2014.nc"
d=xr.open_dataset(F,decode_timedelta=False)
lat=d["latitude"].values; lon=d["longitude"].values
jj=int(np.argmin(np.abs(lat-61.72))); ii=int(np.argmin(np.abs(lon-278.37)))
tv=d["time"].values; ix=int(np.argmin([abs((t-cftime.DatetimeNoLeap(1980,10,16)).total_seconds()) for t in tv[:1400]]))
sst0=d["SST"].isel(time=ix).values.astype("float32")
ice0=d["ICEFRAC"].isel(time=ix).values.astype("float32")
sol0=d["SOLIN"].isel(time=ix).values.astype("float32")
K = norm.x_mean[0]>100
print("SST chan Kelvin:",K," train SST mean/std=%.2f/%.2f"%(norm.x_mean[0],norm.x_std[0]))
def predict(f):
    x=np.stack([f,ice0,sol0,f,ice0,np.zeros_like(f)],0)[None]
    xn=(x-norm.x_mean[:,None,None])/norm.x_std[:,None,None]
    with torch.no_grad(): y=model(torch.from_numpy(xn.astype("float32"))).numpy()[0]
    return y*norm.y_std[:,None,None]+norm.y_mean[:,None,None]
print("\n=== #2 offline MEMO flux sweep at Hudson Bay cell as SST -> 13C ===")
print("                "+"  ".join(f"{n:>8s}" for n in ms.OUT_VARS))
for sstC in [2,4,6,8,10,11,12,13,14,16,18]:
    f=sst0.copy(); f[jj,ii]=(sstC+273.15) if K else sstC
    v=predict(f)[:,jj,ii]
    print(f"SST={sstC:5.1f}C | "+"  ".join(f"{x:8.2f}" for x in v))
