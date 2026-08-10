# MEMO + FESOM2-in-JAX: replacing POP as the ocean

The MEMO flux U-Net, coupled to [`fesom_jax`](https://github.com/koldunovn/fesom_jax) (a JAX
port of FESOM2) instead of POP2. Same product, different ocean.

## Why this is not a drop-in swap

Two structural differences drove the whole design:

**1. POP takes fluxes; fesom_jax takes an atmosphere.** POP's `forcing_memo_mod` accepts the
eight MEMO fluxes directly. `fesom_jax` instead takes atmospheric *state* (10 m wind, humidity,
SW/LW, Tair, precip) and runs its own L&Y09 bulk formula
(`surface_forcing.compute_surface_fluxes`). MEMO does not predict a state, so the bulk formula
had to be *replaced*, not fed.

**2. The v15 U-Net is a CNN on POP's grid.** It runs `--arch gx1v7_native`: its convolutions act
on the structured 384×320 tripole grid with circular longitude padding. A CNN cannot be evaluated
on an unstructured triangular mesh. So the U-Net **stays on gx1v7** and the grid change is
absorbed at the coupling boundary:

```
FESOM nodes --kNN--> gx1v7 (SST) -> [UNCHANGED MEMO server] -> gx1v7 (8 fluxes) --kNN--> FESOM nodes
```

That costs one interpolation hop each way (measured round-trip error on a smooth unit-amplitude
field: 0.15 % mean, 4 % max). The alternative — retraining MEMO on the mesh — means replacing the
CNN with a graph network, i.e. a different product.

**The MEMO server is not modified.** `run_memo_fesom.py` plays POP's role in the existing
flag-file handshake, so every server-side lever proven in the POP work (`--tau_sign_fix`,
`--ocn_albedo`, `--ice_couple`, `--ice_virtual`, the flux ramp, the budget closures) applies
verbatim.

## Files

| file | what it is |
|---|---|
| `fesom_jax/fesom_jax/memo_forcing.py` | the bulk formula's replacement: MEMO fluxes → FESOM `SurfaceFluxes` |
| `fesom_jax/fesom_jax/step.py` | one added branch dispatching on `MemoStepForcing` |
| `camulator_ud/climate/run_memo_fesom.py` | the ocean-side driver (POP's role in the handshake) |
| `fesom_jax/fesom_jax/ice_thermo.py` | `rhow_ext`/`evap_ext`: the open-water override seam |
| `fesom_jax/fesom_jax/ice_step.py` | detects `MemoIceStepForcing`, swaps open water only |
| `fesom_jax/fesom_jax/tests/test_memo_ice.py` | the bulk-equivalence proof of that seam |
| `camulator_ud/climate/core2_memo.yaml` | CORE2 physics, `ice: null` (static ice) |
| `camulator_ud/climate/core2_memo_ice.yaml` | CORE2 physics, `ice: {whichEVP: 1}` (prognostic) |
| `camulator_ud/climate/run_memo_fesom.pbs` | static-ice server (CPU) + ocean (A100) job |
| `camulator_ud/climate/run_memo_fesom_ice.pbs` | prognostic-ice job |
| `camulator_ud/climate/compare_fesom_pop.py` | area-weighted scoring |

Environment: `/glade/work/praggarwal/conda-envs/fesomjax` (JAX 0.11 + CUDA 12).
Data: `/glade/work/praggarwal/fesom-data/core2_mesh_ic` (CORE2 mesh, PHC IC, SSS, runoff, chl).

## The flux translation

FESOM's `heat_flux` is **+up = ocean loses**, pre-shortwave-penetration. MEMO delivers
**+into-ocean**. The mapping mirrors `forcing.obudget` term for term:

| obudget term | bulk source | MEMO source |
|---|---|---|
| `hfswrow` (qsr) | `(1-albw)·shortwave` | `fsds` (net, when server runs `--ocn_albedo`) |
| `hflwrow` | `longwave` | `flds` |
| `hflwrdout` (Planck) | `-εσ(SST+273.15)⁴` | `-εσ(SST+273.15)⁴` ← **recomputed from live SST** |
| `hfsenow` | bulk sensible | `shflx` |
| `hflatow` | bulk latent | `lhflx` |

Everything downstream (SSS/runoff, ice stress blend, shortwave penetration, surface BCs) is the
*unchanged* FESOM code.

The Planck term is deliberately recomputed rather than taken from MEMO — the same choice POP makes
(it computes LWUP internally, which is why the server ships *downwelling* `flds`). It is the only
explicit `d(heat_flux)/d(SST)` term in the surface budget.

**Verification.** `jax.grad` of the summed heat flux w.r.t. SST gives **+4.948 W/m²/K** against the
analytic Planck `4εσT³` = **+4.925 W/m²/K** at the same mean temperature. This both confirms the
translation and quantifies the within-window feedback: Planck is the *only* SST damping inside a
coupling window; the turbulent damping acts only through MEMO's inputs at the *next* window. That
is your documented dQ/dSST deficit, now directly measurable — something POP could not give you.

## Prognostic sea ice — the fix for the +4.44 K

The static-ice result above forced the ice question. FESOM ships full prognostic ice
(`ice_evp.py`, `ice_mevp.py`, `ice_thermo.py`, `ice_step.py`, `ice_adv.py`) and — the part that
made this cheap — **everything it needs to run is already on disk**: `blstate_<year>.nc` carries
`Ubot, Vbot` (rheology), `Tbot, Qbot, PS` (thermodynamics), `FSDS, FLDS, PRECT`, 6-hourly at
1460 records/yr, i.e. exactly one per coupling window. That is the same file the server reads for
`--with_atm_in`, so the ice is driven by the atmosphere the U-Net already sees as input.

**The split: MEMO owns the open water, physics owns the ice.**

| quantity | source |
|---|---|
| open-water growth rate + evaporation (`ice_thermo._obudget`) | **MEMO** fluxes |
| open-water wind stress | **MEMO** `taux/tauy` |
| ice skin-temperature budget (`_budget`), sublimation | physics, from `blstate` |
| mEVP rheology, ice advection | physics, from `blstate` winds |
| ocean→ice heat flux `o2ihf`, freshwater/brine | physics — **the missing melt sink** |

The seam is one line in `ice_thermo.therm_ice_cell`: `rhow, evap = _obudget(...)` becomes an
optional override. Because `_obudget` builds `hftot = qsr + flo + hflwrd + hfsen + hflat` and
returns `-hftot/cl`, and `compute_memo_surface_fluxes` already forms that identical sum as
`heat_flux`, the conversion is exactly **`rhow = heat_flux/cl`** — the same quantity already
verified against `jax.grad`. No new physics.

Two traps this navigates, both easy to get silently wrong:

- **Two different shortwaves.** `fsds` is net of the *ocean* albedo (`--ocn_albedo`), but the ice
  budget applies its own ice/snow albedo (0.70–0.81) to whatever it is handed. So
  `MemoIceStepForcing` carries `fsds_dn` (true downwelling) for the ice and `fsds` for the water.
  Collapsing them would double-count an albedo and under-melt the ice.
- **Double area weighting.** `therm_ice_cell` already does `sh = rhow*(1−A) + rhice*A`, and
  `--ice_couple` weights delivered fluxes by `(1−aice)`. Running both masks twice; the server now
  refuses the combination outright.

**Verification.** `fesom_jax/tests/test_memo_ice.py` — fed the bulk formula's *own* open-water
fluxes, the MEMO path reproduces the bulk ice step to 1e-10 across 13 fields (`bc_T`, `bc_S`,
`heat_flux`, `water_flux`, `a_ice`, `m_ice`, `m_snow`, `u_ice`, `v_ice`, `t_skin`, `stress_surf`,
`virtual_salt`, `relax_salt`). Full suite: 555 passed, 397 skipped.

**Ice source decision (2026-07-29).** FESOM's physical ice feeds the server's ice input via the
new default-off `--ice_input_from_ocn`, and the ML CICE emulator drops out of the ocean loop —
one ice model rather than two that disagree. Run it with `run_memo_fesom_ice.pbs` /
`core2_memo_ice.yaml`.

## Sea ice under `ice: null` — the caveat this replaces

`ice: null`. FESOM's prognostic mEVP sea ice is driven by an atmospheric *state* (winds for the
rheology, Tair/humidity/radiation for the thermodynamics), which MEMO does not predict. So this
runs the static-ice path with `a_ice` prescribed from the server's ICEFRAC each window — the same
arrangement the POP runs used.

The melt-heat sink **is** included: `cice_coupler.py:338` folds `melth` into `shflx` before
delivery. But it is driven by the emulator's climatological `dhi/dt` and capped at ±200 W/m², so
it cannot respond to the ocean's actual heat excess. There is therefore no negative feedback
holding water near freezing under ice: nothing stops an ice-covered cell from warming.

Consequences to watch: no ice-ocean brine/freshwater flux (the gap flagged in the ice-FW memo),
and Antarctic shelf cells that in reality would consume summer shortwave as melt instead warm the
water.

## Running it

```bash
qsub -v DAYS=365,TAG=v1yr,START=1980-01-01 camulator_ud/climate/run_memo_fesom.pbs
```

Cost: ~4–6 s per 6-hour coupling window on one A100 ⇒ **a year in ~2–3 h**, single GPU.

## Result: 1980 full year (job 5397341, rc=0)

1460/1460 coupling windows, **1 h 47 m on one A100**, no NaN, numerically bounded. The swap
works. The ice coupling does not.

| band | day 1 | day 365 | drift |
|---|---|---|---|
| SO (<−45) | +4.21 | +8.66 | **+4.44** |
| STH (−45..−15) | +21.31 | +22.85 | +1.55 |
| TROP (−15..15) | +27.98 | +27.53 | −0.45 |
| STN (15..45) | +19.90 | +19.73 | −0.17 |
| NH (>45) | +2.42 | +2.95 | +0.54 |
| **global** | +18.26 | +19.33 | **+1.07** |

**Do not read this run's health off the global mean.** The global +1.07 K is a sum of canceling
errors: a large Southern Ocean warming partly masked by tropical cooling. The seasonal *phase* is
also correct (peak March 19.24, trough September 18.62, right for a SH-area-dominated global mean)
— and that too is not evidence of health. Band it first.

The SO warming is **broad**, not a few cells: per-node ΔT median +4.93 K, IQR 2.64–8.13 K. This is
the `ice: null` limitation below, now quantified.

Separately, **1921 nodes (0.41 % of area) are numerically garbage** — SST −52 .. +133 °C, already
broken by day 30. They are shallow shelf cells (median 17 wet levels vs 36 globally, control volumes
0.36× median), so very low heat capacity. These are **not** the drift: excluding all of them moves
the global number only +1.069 → +1.033 K and leaves SO at +4.44 K. Two independent problems.

## Known limitations

- **Single device.** The Zenodo record ships no `dist_N` partitions, so multi-GPU is unavailable
  until partitions are generated. CORE2 anti-scales past 4 GPUs anyway, so the ceiling is modest.
- **No JRA-forced control at 1980.** The Zenodo forcing is **1958 only**. A physics-forced baseline
  on the same mesh needs JRA55-do v1.4.0 from ESGF (`input4MIPs`, `MRI-JRA55-do-1-4-0`), renamed to
  `{var}.{year}.nc`.
- **Cold start from PHC climatology**, not a spun-up restart. The first ~2 weeks are an adjustment
  transient (global-mean SST rises then flattens); do not read drift from the first days.
- Comparisons against POP must be **area-weighted**. Unstructured control volumes vary by orders
  of magnitude — a plain node mean read 10.86 °C where the area-weighted value was 18.26 °C.
- **The MEMO server speaks CESM's NoLeap 365-day calendar.** A driver built on Python `datetime`
  (proleptic Gregorian) will ask for Feb 29 in a leap year; the server dies inside `cftime` with
  `invalid day number` and the ocean then blocks for its full reply timeout. `run_memo_fesom.py`
  uses its own `NoLeapDate` for this reason — do not swap it back for `datetime`.
