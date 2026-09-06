"""Monthly NH/SH ice extent + area for CESM2 piControl (same 67-yr-length slice
used elsewhere), for direct comparison against the pz220f0b coupled run."""
import numpy as np, xarray as xr

PI_FILE = ("/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/"
           "b.e21.B1850.f09_g17.CMIP6-piControl.001/ice/proc/tseries/month_1/"
           "b.e21.B1850.f09_g17.CMIP6-piControl.001.cice.h.aice.000101-009912.nc")
N_MONTHS = 812  # match pz220f0b's available length

ds = xr.open_dataset(PI_FILE)
aice = ds["aice"].isel(time=slice(0, N_MONTHS)).values.astype(np.float64)  # (t,nj,ni)
tarea_m2 = ds["tarea"].values.astype(np.float64) if "tarea" in ds else None
if tarea_m2 is None:
    ds2 = xr.open_dataset(PI_FILE, decode_times=False)
    tarea_m2 = ds2["tarea"].values.astype(np.float64)
TAREA_km2 = np.nan_to_num(tarea_m2) * 1e-6   # cice tarea is m^2 -> km^2
TLAT = ds["TLAT"].values.astype(np.float64)
nh = TLAT > 0
sh = TLAT < 0

months = ds["time"].values[:N_MONTHS]
recs = []
for i, t in enumerate(months):
    a = np.nan_to_num(aice[i])
    ext_nh = float(TAREA_km2[nh & (a >= 0.15)].sum()) / 1e6
    ext_sh = float(TAREA_km2[sh & (a >= 0.15)].sum()) / 1e6
    area_nh = float((TAREA_km2 * a)[nh].sum()) / 1e6
    area_sh = float((TAREA_km2 * a)[sh].sum()) / 1e6
    recs.append((t.year, t.month, ext_nh, ext_sh, area_nh, area_sh))

recs = np.array(recs)
np.savez("output/picontrol_ice_extent_monthly.npz",
         year=recs[:, 0].astype(int), month=recs[:, 1].astype(int),
         ext_nh=recs[:, 2], ext_sh=recs[:, 3], area_nh=recs[:, 4], area_sh=recs[:, 5])
print(f"wrote output/picontrol_ice_extent_monthly.npz ({len(recs)} months)")
print("first 3:", recs[:3])
print("TAREA sum check (should be ~360 x1e6 km2 total sphere-ish ocean):",
      TAREA_km2.sum() / 1e6)
