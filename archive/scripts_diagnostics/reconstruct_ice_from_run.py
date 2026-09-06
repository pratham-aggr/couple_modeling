"""Offline reconstruction of the CICE emulator's monthly ice-fraction TARGET
field for the pz220f0b interactive-ice coupled run, driven by the run's own
completed monthly-mean SST (from pop.h). The full spatial aice history was
never written to disk online (only a scalar log line + a rolling restart
snapshot), so this replays the emulator deterministically off the ALREADY-
COMPLETED SST trajectory to recover it, month by month, for the whole run.

Outputs monthly NH/SH ice extent (area where aice>=0.15) and ice area
(aice-weighted) in 10^6 km^2, saved to a npz for comparison against piControl.
"""
import glob, re, sys
from datetime import datetime
import numpy as np, xarray as xr
import netCDF4 as nc

sys.path.insert(0, "/glade/u/home/praggarwal/couple/camulator_ud/climate")
from cice_coupler import CiceCoupler

PZ_DIR = "/glade/derecho/scratch/praggarwal/memo_pop_standalone_gx1v7_diagfw_iceloop/run"
GX1V7_DOMAIN = "/glade/campaign/cesm/cesmdata/inputdata/share/domains/domain.ocn.gx1v7.210716.nc"
CICE_OUT = "/glade/u/home/praggarwal/couple/output/output_cice_solin"
CICE_IC = "/glade/derecho/scratch/praggarwal/couple_cache_cice_nowind/ice_ic_1980-01.npz"

with nc.Dataset(GX1V7_DOMAIN) as ds:
    xc = ds["xc"][:].data.astype(np.float64)
    yc = ds["yc"][:].data.astype(np.float64)
    mask = ds["mask"][:].data.astype(np.int32)

cice = CiceCoupler(CICE_OUT, "best_model.pt", CICE_IC, ocean_mask=(mask > 0),
                    device="cpu", melth_cap=200.0, lat=yc, ramp_days=20.0,
                    virtual_ice=True)

mfiles = sorted(glob.glob(f"{PZ_DIR}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
g0 = xr.open_dataset(mfiles[0])
TAREA_cm2 = np.nan_to_num(g0["TAREA"].values.astype(np.float64))   # cm^2
TAREA_km2 = TAREA_cm2 * 1e-10                                       # cm^2 -> km^2
TLAT = g0["TLAT"].values

nh = TLAT > 0
sh = TLAT < 0

recs = []
for f in mfiles:
    m = re.search(r"\.(\d{4})-(\d{2})\.nc$", f)
    y, mo = int(m.group(1)), int(m.group(2))
    sst = xr.open_dataset(f)["TEMP"].isel(time=0, z_t=0).values.astype(np.float64)
    now_dt = datetime(y, mo, 15)
    stepped = cice.maybe_step(sst, now_dt)
    if stepped:
        aice = cice._tgt["aice"]
        # this step's target is for month mo+1 (see cice_coupler.py mid-month logic)
        tgt_y, tgt_mo = (y + (mo == 12), mo % 12 + 1)
        ext_nh = float(TAREA_km2[nh & (aice >= 0.15)].sum()) / 1e6
        ext_sh = float(TAREA_km2[sh & (aice >= 0.15)].sum()) / 1e6
        area_nh = float((TAREA_km2 * aice)[nh].sum()) / 1e6
        area_sh = float((TAREA_km2 * aice)[sh].sum()) / 1e6
        recs.append((tgt_y, tgt_mo, ext_nh, ext_sh, area_nh, area_sh))
        if len(recs) <= 3 or len(recs) % 120 == 0:
            print(f"{tgt_y}-{tgt_mo:02d}: NH ext={ext_nh:.2f} SH ext={ext_sh:.2f} "
                  f"(x1e6 km2)  [{len(recs)} months done]")

recs = np.array(recs)
np.savez("output/pz220f0b_ice_extent_monthly.npz",
         year=recs[:, 0].astype(int), month=recs[:, 1].astype(int),
         ext_nh=recs[:, 2], ext_sh=recs[:, 3], area_nh=recs[:, 4], area_sh=recs[:, 5])
print(f"wrote output/pz220f0b_ice_extent_monthly.npz ({len(recs)} months)")
print(f"first 3 months (timing check): {recs[:3]}")
