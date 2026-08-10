# Standalone MEMO + POP2 (v8): what made it work

Goal: run POP2 with the MEMO flux emulator directly (cpl_none driver, no CPL7, no CAM/CICE/DATM),
stable for multiple years. **v8 = 5 full years 1980–1984, zero aborts** (jobs 6675299/6675300).
Everything below is server-side (`camulator_ud/climate/model_server.py` + `cice_coupler.py`)
except the timestep and Robert filter; POP source is untouched.

## The fix stack (each one is load-bearing — ablations confirmed)

| # | Fix | Failure it removed |
|---|-----|--------------------|
| 1 | **POP dt 1 h → 30 min** (`dt_count=48`) | Vertical tracer-advection CFL blowup (Courant ~66) in week 1 |
| 2 | **Clamp the dSST/dt input channel to ±0.1 K/day** | Exponential warm-cell runaway: emulator Qnet gain is ~+260 W/m² per (K/day), so one warm cell self-heats to >10³ °C. Clamp = ~1.6σ of training values; bounds the feedback |
| 3 | **Ice-flux masking**: all 8 fluxes × (1−aice); FLDS blended to σT⁴ under ice | Full shortwave/turbulent fluxes were hitting ice-covered cells (2–4× over-forced). FLDS must be *blended*, not zeroed, because POP computes LWUP internally from live SST — zeroing leaves a −300 W/m² drain (cold detonation) |
| 4 | **Virtual ice**: open-water weight × clip((SST+1.8)/0.3, 0, 1) | Cells at freezing kept losing heat where the ice product said "ice-free" → 55× frazil rate → brine-driven bottom convection (Greenland Sea, Feb) |
| 5 | **Live ML CICE emulator** with mid-month anchoring (see below) | Prescribed climatological aice can't respond to POP's actual state; the winter sill-convection wall (Dec/Feb, hmix_gm TLT abort) only fell with responsive ice at the ice edge |
| 6 | **10-day flux ramp** at cold start | First-step forcing shock on the LE2 restart (NW-Corner detonation on day 12 without it) |
| 7 | **Global Qnet + P−E budget closures** (`--qnet_balance --pe_balance`), anchored to the *anomaly* vs training climatology (`budget_clim_CREDIT_1980-84.nc`), ramp-gated, 3-day-EMA smoothed, delivered over low-lat open water | Secular warm drift +1.0 °C/month → +0.23 °C/month. Two delivery rules learned the hard way: never let a closure run inside the flux ramp, and correct the low-frequency drift (EMA), never the instantaneous anomaly |
| 8 | **NO local SST restoring** | Restoring was the August killer (grid-scale SST/SSS computational mode in Hudson Bay); the run without it finished the year |
| 9 | **robert_nu 0.1 → 0.3** (Robert–Asselin filter) | The year-2 winter wall: a 2Δt leapfrog oscillation in the Labrador top layer each winter. GIAF's 0.1 is tuned for dt_count=24; at 48 the 2Δt mode halves its period and 0.1 under-damps. 0.3 lets the cell convect physically instead of detonating |

Ops: split into ≤3-year segments (7.5 h walltime) chained with `-W depend=afterok` —
1-node jobs over ~10 h starve in Derecho's queue. Continuation via `ccsm_branch` + rpointer.

**Open problem (not stability): fidelity.** v8 drifts ~+4 °C warm vs coupled GIAF by year 3.
Diagnosed cause: the emulator reads winter storms from the 24-h SST-tendency channel, which
the standalone loop cannot supply (no atmosphere to imprint storm wakes) → wind stress
collapses → no subpolar ventilation → surface heat trapping. v9 (τ climatology anchoring +
stochastic gusts) fixed the winds but not the mixed layer — the subpolar buoyant cap is
thermal, so surface heat removal there is the next lever.

## The ML CICE model (`train_cice.py` → `output/output_cice_solin`)

Same 5-level UNet as MEMO (base 64, circular longitude padding; gx1v7 is 384×320, divisible
by 32, so the architecture drops in unchanged). Trained on a monthly gx1v7 cache from
CESM2-LE (LE2-1231.002).

**Deployed (wind-free) variant:**

```
IN  (6ch): SST, SOLIN, aice_prev, hi_prev, uvel_prev, vvel_prev
OUT (6ch): aice, hi, uvel, vvel, strocnx, strocny
```

aice and hi are **residual targets**: the net predicts the month-to-month *change*, added to
the previous state. Cadence is monthly: month-m mean state → month-(m+1) mean, with the
target month's analytic SOLIN (`solar.py`).

**Yes, it is autoregressive.** At run time (`cice_coupler.py`) its own previous predictions
(aice, hi, uvel, vvel) are fed back as the `_prev` inputs for the next step; the only
external inputs are live SST from POP and analytic SOLIN. It is seeded once from the
LE2 Jan-1980 ice state (`ice_ic_1980-01.npz`) and then free-runs for the whole simulation —
so ice errors can compound, but SST coupling keeps it anchored (POP warms → emulator melts
ice → melt-heat sink cools POP: the negative feedback prescribed climatology lacks).

**Timing convention (bug-prone, twice):** the IC equals the January *monthly mean*, and the
model maps mean(m)→mean(m+1). Naive scheduling gives a one-month lag or lead. The fix is
mid-month anchoring: hold the IC to Jan 15, predict the next month's mean, and glide the
*applied* fields linearly from mid-month to mid-month. Applied ice is therefore a
time-interpolated monthly state — no step shocks (an un-ramped aice snap of 0.57→0.27 in
one 6-h step once detonated a Weddell dense-water cell).

**How it touches POP** (all through the existing `memo_cam_out.nc` flux fields, no Fortran):
1. live aice → IFRAC and the (1−aice) flux weighting of fix #3;
2. ice–ocean stress: τ_ocn = (1−aice)·τ_MEMO + aice·strocn;
3. melt heat: Q = ρ_ice·L_f·d(hi)/dt folded into the sensible-heat channel (physically
   equivalent to MELTH_F — both enter POP's SHF_COMP additively).

Known gap: no brine rejection / melt freshwater flux (SALT_F, MELT_F) — the standalone polar
salinity carries a ~+1.5 psu bias partly for this reason.


are we taking eneregy change into account? looks like it's leaking energy 
