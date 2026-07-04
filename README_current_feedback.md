# Current-feedback drag (MEMO→POP standalone)

## Outcome (concluded — did NOT help; reverted)

Ran 1 yr on Derecho cpudev (job 6607174). It crashed at **1980-07-14 (~day 195)** —
*earlier* than the baseline (~day 290) — via the same failure as the HPX run: a slow
SST warm drift (31 °C Jan → 40 °C May → detonation to 1.7e10 °C Jul 14) that spikes the
overflow product index (`prd set old/new = 5 1`) and then faults GM's transition-layer
calc (`ERROR: Incorrect TLT computations`, hmix_gm.F90:1245).

Verdict: the drag targets the wrong mechanism. The README hypothesis was a *momentum*
runaway (barotropic → tracer-CFL), but the actual standalone failure is *thermodynamic*
— unbounded warm drift in poorly-ventilated / enclosed basins with no heat sink. A
momentum drag can't touch a stratification blowup, so it bought nothing.

Reverted exactly by setting `lcurrent_feedback = .false.` in the shared driver source
(the code is kept, gated off, as a reference). Next real fix to try: a sea-ice heat sink
(CICE in the direct path) to cap high-latitude SST; cheaper diagnostic stopgap is
disabling `overflows_interactive` / `transition_layer_on` to confirm the overflow/GM
path is the proximate detonator.

---

Experiment: restore the air-sea momentum damping a one-way emulator drops, without
changing POP2 physics. Tests whether it prevents the standalone crash (baseline
died 9.5 mo in at Hudson Bay via a barotropic→vertical-tracer-CFL runaway: surface
currents accelerated 9→33→99 cm/s with nothing braking them, because MEMO's wind
stress never sees the ocean current).

## What changed (coupling shim only — no POP physics, no namelist)

`components/pop/drivers/cpl_none/forcing_memo_mod.F90`, in `memo_exchange` right
after `rotate_wind_stress`:

    tau_eff = tau_MEMO - sdrag * U_ocn ,   sdrag = sqrt(rho_air_cfb * cdrag_cfb * |tau|)

- Applied to `SMF` (dyne/cm², U-grid), paired with surface `UVEL/VVEL` (cm/s) — same
  grid-aligned basis. Over the full haloed array, so halos stay consistent.
- `sdrag = sqrt(rho_air*Cd*|tau
|)` is the relative-wind linearization (depends only on
  |tau| + constants; POP supplies U_ocn locally).
- Toggle/params (module level): `lcurrent_feedback=.true.`, `rho_air_cfb=1.2e-3` g/cm³,
  `cdrag_cfb=1.5e-3`. Set `lcurrent_feedback=.false.` to revert exactly.
- First exchange logs `lcfb, rho_air_cfb, cdrag_cfb, sdrag[min>0,max]` via `document`.

## Run it

    bash stage_pop_cfb.sh                       # stage isolated rundir (1-yr stop)
    qsub -q main@desched1 run_pop_cfb_experiment.pbs   # builds driver + runs 1 yr

- Rundir: `memo_pop_standalone_cfb/run` (baseline `memo_pop_standalone` untouched).
- Same config as baseline (pure MEMO fluxes: `--tau_cap 0 --flux_ramp_days 0`), so the
  drag is the only difference. Survive past ~day 290 ⇒ it helped.
- Build (Cray `ftn`) runs on Derecho inside the job; server (conda) launches first in a
  clean env, before the module load.

## Files

- `forcing_memo_mod.F90` — the drag + diagnostic (above).
- `stage_pop_cfb.sh` — clone of `stage_pop_standalone.sh`, own dir/id, `stop_count=1`.
- `run_pop_cfb_experiment.pbs` — build driver → server → 1-yr POP, isolated.
- `build_pop_standalone_driver.sh` — unchanged; recompiles the cpl_none driver.


 CHANGED overflows_interactive=.false. TO true
 