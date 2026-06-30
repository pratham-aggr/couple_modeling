"""
Locate the EXACT crash cell + onset hour from the hourly stream-4 output
(g.e21.MEMO_GIAF_v01.pop.h.nhour1.*.nc) of the diagnostic re-run.

Answers, from data (not from POP printing a location):
  - which (i,j) -> lat/lon first goes unphysical, and at what hour
  - the SST trajectory of that cell in the final hours (sudden vs ramp)
  - whether the incoming MEMO fluxes (SHF/TAUX/TAUY) spiked just before
"""
import glob, sys
import numpy as np
import xarray as xr

RUN = "/glade/derecho/scratch/praggarwal/memo_pop_standalone/run"
THR = 40.0          # degC: clearly unphysical SST
FILL = 1e30

files = sorted(glob.glob(f"{RUN}/g.e21.MEMO_GIAF_v01.pop.h.nhour1.*.nc"))
if not files:
    sys.exit("no nhour1 files yet — run hasn't produced hourly output")
print(f"{len(files)} hourly files", flush=True)

S, SHF, TX, TY, TM = [], [], [], [], []
tlat = tlon = None
for f in files:
    d = xr.open_dataset(f, decode_timedelta=False)
    S.append(d["SST"].values.astype("float32"))
    SHF.append(d["SHF"].values.astype("float32") if "SHF" in d else None)
    TX.append(d["TAUX"].values.astype("float32") if "TAUX" in d else None)
    TY.append(d["TAUY"].values.astype("float32") if "TAUY" in d else None)
    TM.append(d["time"].values)
    if tlat is None:
        tlat, tlon = d["TLAT"].values, d["TLONG"].values
    d.close()
sst = np.concatenate(S); tm = np.concatenate(TM)
sst = np.where(sst > FILL, np.nan, sst)
T, NJ, NI = sst.shape

mx = np.nanmax(sst.reshape(T, -1), axis=1)
t0 = next((t for t in range(T) if mx[t] > THR), None)
if t0 is None:
    print(f"max SST never exceeded {THR}C (peak {np.nanmax(mx):.1f}). "
          "Run may not have reached the crash."); sys.exit(0)

j, i = np.unravel_index(np.nanargmax(np.nan_to_num(sst[t0], nan=-1e9)), (NJ, NI))
lon = tlon[j, i] - 360 if tlon[j, i] > 180 else tlon[j, i]
print(f"\n*** ONSET: first SST>{THR}C at {str(tm[t0])[:13]}h  "
      f"cell (j={j}, i={i})  lat={tlat[j,i]:.2f}  lon={lon:.2f} ***\n")

print("crash-cell SST, final 12 hours before onset:")
for t in range(max(0, t0 - 12), t0 + 1):
    line = f"  {str(tm[t])[:13]}h  SST={sst[t,j,i]:8.2f}"
    if SHF[0] is not None:
        shf = np.concatenate([x for x in SHF])[t, j, i]
        line += f"  SHF={shf:9.1f} W/m2"
    print(line)
print("\n-> sudden jump = single-step CFL blow-up; gradual = thermal drift")
