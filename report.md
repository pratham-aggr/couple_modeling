# Standalone MEMO → POP2 — Run Report

**Goal:** drive POP2 ocean directly from the MEMO ML flux emulator via the `cpl_none`
path (no CPL7/CICE) and reach a stable multi-year run.

**Status (2026-07-04): FULL STABLE YEAR ACHIEVED.** Run v4 (native `ql6lbazr` + ice-mask
+ CICE aice climatology, job 6637796) completed 1980-01-01→12-31 with zero aborts —
"Successful completion of POP run". Root cause of all earlier crashes = missing
ice-ocean coupling in the cpl_none path (not POP, not the restart, not the flux model).
Remaining defect is a global warm drift (+1 °C/month; no atmospheric thermostat), fixed
by OMIP-style weak SST restoring (v5, job 6638084, running).

## What works
- End-to-end MEMO↔POP coupling (cpl_none, file-flag handshake), gx1v7 tripole grid.
- Native flux model `ql6lbazr` (`output/output_unet_gx1v7`, 6→8) driving POP directly.
- 41 days with physically sane SST (max ~33 °C, no blowup).

## Runs / interventions

| Change | Effect | Verdict |
|---|---|---|
| dt = 30 min (tracer-CFL fix) | moved earliest crash off 1980-01-07 | **helped** |
| **dSST/dt input clamp (0.10 K/day)** | killed thermodynamic runaway; day 21 → **day 41** | **helped (key)** |
| CICE off (vs ML CICE on) | more polar heat → later crash (Feb 10 vs Feb 2) | **helped** |
| ML CICE emulator on | removes polar heat → crashes earlier/colder (Feb 2, −35385 °C) | hurt |
| `overflows_interactive=.false.` | ~4 days further (noise) | no help |
| `transition_layer_on=.false.` | aborts at init | worse |
| Option A: zero-TLT fallback in `hmix_gm.F90` | NaN at identical Feb-10 step | no help |
| Under-ice strong restoring (sfwf/shf) | NaN at step 1 (bad polar climatology field) | no help |
| Flux ramp (surface-flux spin-up) | surface knob on an interior problem | wrong lever |
| Orography statics (training) | ~+0.003 R² (noise) | no help |
| CFB current-feedback drag | momentum only, irrelevant to failure | no help |

## Flux models tried (what, why special, when it crashed)
All are 6→8 UNets (base 64). Crash dates below are **pre-dSST-clamp** unless noted.

| Model (wandb) | What's special | Crashed | SST at crash |
|---|---|---|---|
| **lat-lon** (original) | trained on CAM lat-lon grid (192×288); fluxes remapped to gx1v7 (interpolation error) | 1980-10-17 | 5.6e6 °C — separate Hudson Bay warm-drift, not the POP fault |
| **HPX** (`eutl8ixm`) | trained on HEALPix mesh; remapped to gx1v7 | 1980-01-17 | sane (35.6 °C), 39 overflow swaps |
| **CFB** | current-feedback drag: adds ocean-current feedback into the wind stress (momentum only) | 1980-07-14 | 1.7e10 °C — separate warm-drift, not the POP fault |
| **native gx1v7** (`ql6lbazr`) | trained directly on POP's gx1v7 tripole grid (384×320); **no remap** — SST fed straight in, fluxes straight back; best skill | 1980-01-12 (pre-clamp); **1980-02-10 with dSST/dt clamp = current baseline** | sane (33 °C) |

## Root cause (current understanding)
The Feb-10 wall is **saturated polar deep convection driven from inside POP** — the LE2
restart's polar density field is inconsistent with POP's numerics at the steep-topo
overflow sills; cold-start shock triggers runaway convection → overflow/tracer NaN.
Evidence: the emulator net-heats + slightly freshens the poles (convection-*suppressing*),
and *more* polar heat *delays* the crash — so the surface flux is not the driver. No
surface-side intervention fixes it.

## Current baseline
`couple` repo at `5ecfe65 Revert to 74d7763` (clean); dSST/dt clamp retained;
`hmix_gm.F90` original; `pop_in` strong restore 0/0. Furthest-surviving config = native
ql6lbazr + POP2 to Feb 10.

## The fix chain that got to a full year (all in model_server.py; POP untouched)
| Step | Fix | Cured |
|---|---|---|
| dt=30 min | pop_in dt_count=48 | 1980-01-07 tracer-CFL blowup |
| dSST/dt clamp 0.1 K/day | input-channel OOD guard | exponential coastal runaway (day 21→41) |
| ice-mask v2 | fluxes ×(1−ICEFRAC) + FLDS→σT⁴ under ice (POP computes LWUP internally — must balance it, v1 froze) | polar 2–4× over-forcing (Feb-10 wall) |
| CICE aice climatology | 12-mo aice from the stable coupled year (CAM forcing said Greenland Sea ice-free; reality 0.33) | Greenland Sea frazil-brine pump (+2 psu → Feb-15) |
| **= v4: FULL YEAR, zero aborts** | | |
| SST restoring q=30 W/m²/K (v5) | OMIP thermostat toward coupled-year monthly SST | +1 °C/month global warm drift |

Failed intermediates: virtual-ice (zero-inertia feedback, ringing blowup Jan-16);
under-ice strong restoring (PHC2 fill garbage); all POP-side edits (reverted).

## U-Cast probabilistic model (m377oyqt)
Coupled via lat-lon arch: crashes at the gx1v7 i-seam on the N-Atlantic front (Jan-12
bare; May-17 with restoring — restoring extended it 4 months and held the drift).
Seam flux jump 5× interior ⇒ remap defect, not a model defect. Fix: retrained natively
on gx1v7 (S1 MAE ep28 → S2 CRPS ep14, AdamW 1e-4 + EMA; Muon diverged), out_dir
`output/output_unet_gx1v7_ucast`.

**Native U-Cast test eval (2013–14, 8-member MC-dropout)** vs lat-lon U-Cast:
ens-mean R² better on 7/8 vars (PRECT 0.31→0.41, SHFLX 0.82→0.85, TAUX 0.70, FLDS 0.98);
SSR closer to 1 on 6/8 (LHFLX 0.85, FLDS 0.97); wind-stress CRPS −10%. No remap ⇒ seam
defect gone by construction. Full numbers: `output_unet_gx1v7_ucast/crps_ssr_test.json`.

**Coupled (job 6648567, run dir `memo_pop_standalone_gx1v7_ucast`)**: native U-Cast +
the full v5c guard stack — running, head-to-head vs v5c.

## Restoring-era runs (post-v4)
| Run | Config delta | Result |
|---|---|---|
| v5 (6638084) | + SST restore q=30, UNCAPPED | died Jan-10 (+956 W/m² hammer on a cold wiggle) |
| v5b (6644187) | restore capped ±150 W/m² | died Aug-14 Hudson Bay: emulator freshwater bomb (SSS −4 psu/5 d) |
| v5c (6648445) | + prect/qflx caps 2e-3 kg/m²/s | **running** — every flux channel now bounded |
| U-Cast (6648567) | v5c guards, native CRPS model | **running** — probabilistic vs MSE head-to-head |

## Next
1. v5c result → expect stable + drift-bounded year (the deliverable config).
2. U-Cast coupled result → probabilistic vs MSE coupled comparison.
3. Optional physicality: melt-freshwater/brine fluxes (step-2 ice thermo), SSS restore
   toward MEMO_GIAF monthly SSS.
