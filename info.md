# MEMO–POP Coupling: Experiments & Diagnosis

All standalone runs start from the CESM2-LE 1980-01-01 ocean restart.

---

## Experiments

### Standalone (no CPL7 coupler)

POP runs as a standalone executable; MEMO exchanges fluxes via file-flag handshake. ICEFRAC is prescribed from the CREDIT forcing file — no live CICE.

**MSE model** (`output_unet_mem24h_dsst_temporal_radprecip`)

| Date | Config | Crash day | Crash date | Cause |
|---|---|---|---|---|
| 2026-06-?? | No damping | 7 | 1980-01-08 | TLT abort |
| 2026-06-?? | Flux ramp + tau cap | 133 | 1980-05-13 | Tracer-CFL blowup |
| 2026-06-?? | Flux ramp + tau cap + smaller timestep | 225 | 1980-08-13 | Tracer-CFL blowup |
| 2026-06-23 | Flux ramp + tau cap (rerun) | ~224 | 1980-08-13 | Tracer-CFL blowup |

**CRPS model** (`output_unet_mem24h_dsst_temporal_radprecip_ucast`, run p5ge0uvh)

| Date | Config | Crash day | Crash date | Cause |
|---|---|---|---|---|
| 2026-06-23 | Flux ramp + tau cap | ~15 | 1980-01-15 | Tracer-CFL blowup |
| 2026-06-23 | No damping | ~8 | 1980-01-09 | TLT abort |

- **Flux ramp**: fluxes scale 0→1 linearly over 10 days to avoid cold-start shock
- **Tau cap**: wind stress clipped to 0.5 N/m² per cell to suppress MEMO outliers

### Coupled (full CPL7 coupler)

Full CESM2: MEMO as atmosphere, POP as ocean, CICE as sea ice. ICEFRAC comes from live CICE.  
Case: `g.e21.MEMO_GIAF_v01` — **MSE model**, flux ramp + tau cap.

| Date | Run length | End date | Outcome |
|---|---|---|---|
| 2026-06-22 | 15 days | 1980-01-16 | Completed |
| 2026-06-23 | 15 days | 1980-01-16 | Completed |
| 2026-06-23 | 1 year | in progress | — |

The coupled run does not crash — CICE is present and handles ice-ocean stress correctly.

---

## What the standalone run is missing vs the full coupler

In a full CESM run, the coupler merges atmosphere and ice contributions before passing to POP. In standalone, MEMO fills in for the atmosphere but CICE is absent.

| Input to POP | With coupler | Standalone |
|---|---|---|
| Wind stress | `(1−f)*tau_atm + f*tau_ice` | MEMO only, full strength everywhere |
| Sensible heat | Atmosphere | MEMO |
| Longwave down | Atmosphere | MEMO |
| Shortwave | Atmosphere, zeroed under ice | MEMO, **not zeroed under ice** |
| Evaporation | Atmosphere | MEMO |
| Precipitation | Atmosphere | MEMO |
| Upwelling longwave | From SST (Fortran) | From SST (Fortran) |
| ICEFRAC | Live CICE | Prescribed from forcing file |
| Ice melt heat/freshwater | CICE | **Missing — zero** |
| River runoff | MOSART | **Missing — zero** |
| Sea level pressure | Atmosphere | **Missing — zero** |

---

## Failure modes

| Mode | What happens | When |
|---|---|---|
| **TLT abort** | `hmix_gm.F90:3393`: KPP boundary-layer depth goes NaN under abrupt full-strength fluxes; GM transition-layer solver aborts | Day 7–8, undamped |
| **Tracer-CFL blowup** | Vertical advection Courant number `w·Δt/Δz > 1` at steep Antarctic shelf cells; SST diverges to ±10⁶°C in one step | Day 15–225, damped; timing varies |

---

## Root cause and suspected bugs

**Root cause:** MEMO's turbulent heat flux feedback is 20–250× too weak (SHFLX at 0.4%, LHFLX at 5.4% of bulk-formula truth). SST anomalies at Antarctic shelf cells grow unchecked until POP's numerics fail. Core missing inputs: wind speed (`Ubot`) and boundary-layer air temperature (`Tbot`).

**Bug 1 — Wind stress not weighted by ice fraction.**  
The coupler applies `tau_ocean = (1−f)*tau_atm + f*tau_ice`. Standalone applies MEMO's stress at 100% everywhere. In summer when ice fraction drops rapidly, newly exposed near-freezing cells receive sudden large stress, driving horizontal divergence → large `w` → tracer-CFL.  
Fix: multiply TAUX/TAUY by `(1−ICEFRAC)` before passing to POP.

**Bug 2 — Shortwave not zeroed under ice.**  
The coupler zeros shortwave under ice. `pop_set_coupled_forcing` weights heat/freshwater by `(1−f)` but shortwave bypasses this and enters `SHF_QSW` unmasked. MEMO's ~200 W/m² shortwave heats water under ice, creating unstable stratification.  
Fix: multiply shortwave by `(1−ICEFRAC)` before writing to `SHF_QSW`.

Both bugs are worst in SH winter (July–September) when Antarctic ICEFRAC transitions rapidly.
