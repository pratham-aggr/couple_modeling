# Standalone MEMO → POP2 — the changes that made it run

**Goal:** drive the POP2 ocean directly from the MEMO ML flux emulator (no CPL7, no
real CICE) and get a stable, physically correct year.

**Result:** v7b (job 6664241) ran 1980-01-01 → 1981-01-01 with zero aborts, no SST
restoring, and a global drift of +0.23 °C/month (5× better than the first stable
year). Full run-by-run history: `report_detailed.md`.

## The changes, in the order they were needed

1. **Standalone driver.** Built POP with the `cpl_none` driver and a small PIO-init
   fix in POP.F90 (the driver never initialized shr_pio, so restart reads crashed).
   MEMO runs as a Python server; POP and the server exchange SST/fluxes through
   files every 6 model hours.

2. **Train the flux model on POP's own grid.** The original model ran on the CAM
   lat-lon grid and its fluxes were remapped to POP's gx1v7 tripole grid. The remap
   injected a flux discontinuity at the grid seam that blew up the North Atlantic.
   Retraining the UNet natively on gx1v7 (SST in, 8 fluxes out, no remap) removed
   the problem by construction.

3. **Halve the ocean time step** (`dt_count` 24 → 48, i.e. 30 min). Fixed a
   vertical tracer-advection CFL blowup on day 7.

4. **Clamp the dSST/dt input channel to 0.1 K/day.** The emulator turned strong
   warming into even more heating (an unbounded positive feedback discovered at one
   Caribbean coastal cell, 16–400σ outside its training range). Clamping the input
   broke the loop without touching any other channel.

5. **Give the ocean sea ice — POP saw none.** `cpl_none` delivers no ice effects at
   all: full sunlight into 98%-ice cells, no insulation, no melt/brine fluxes. This
   was the root cause of the recurring polar crash (runaway deep convection at the
   overflow sills every winter). Fixes, in steps:
   - weight every air–sea flux by (1 − ice fraction);
   - blend downward longwave toward σT⁴ under ice — POP computes upward longwave
     internally at full strength, so simply zeroing FLDS freezes the poles;
   - use a trustworthy ice fraction: first a monthly climatology from a stable
     coupled year (got the first full year, v4), finally the live ML ice below.

6. **Remove SST restoring entirely.** Weak restoring toward climatology looked like
   the cure for drift, but three different runs died identically in Hudson Bay in
   August — the restoring itself destabilizes POP's vertical tracer advection. No
   per-cell forcing toward climatology anywhere in the final setup.

7. **Fix the drift with global budget closures instead.** Each step, compare the
   emulator's global net heat flux and P−E against the training data's monthly
   climatology and remove only the difference, spread over low-latitude open water.
   Two delivery details were essential: hold the closures off during the 10-day
   flux spin-up ramp, and smooth the correction with a 3-day running average so it
   removes the slow drift, not the weather. Drift fell from +1.0 to +0.23 °C/month.

8. **Couple the ML sea-ice emulator with correct timing.** A monthly UNet predicts
   the ice state from POP's live SST (so ice responds to the ocean instead of
   following a fixed climatology — this is what finally survived the December
   winter). Its predictions are *monthly means*, which are valid mid-month, so the
   coupler predicts next month's state at each mid-month and glides linearly to the
   next mid-month. Both naive schedulings failed: updating at month boundaries ran
   the ice ~3 weeks late (crash in January), and predicting on day 1 applied
   February's ice in early January (Antarctic melted a month early).

## What runs it

- `camulator_ud/climate/run_pop_gx1v7_v7_experiment.pbs` — the winning stack
  (Derecho `develop` queue, ~2 h wallclock per model year).
- Server flags: `--arch gx1v7_native --tau_cap 0 --flux_ramp_days 10 --ice_couple
  --qnet_balance --pe_balance` (+ model/IC/climatology file paths; see the PBS).
- `run_pop_gx1v7_v7_ucast_experiment.pbs` — same stack with the probabilistic
  U-Cast flux model (job 6665177 — died Jan-13; the eval-mode median of a
  CRPS/MC-dropout model is not a feedback-consistent forecast, see
  `report_detailed.md`).
- `run_pop_gx1v7_a{1,2,3}_ablation.pbs` — remove-one-guardrail ablations. All
  three failed (A1 no dSST clamp → Feb-07; A2 no flux ramp → Jan-12; A3 no P−E
  closure → silent NaN Dec-19). A1/A2 verdicts stand: clamp and ramp are
  load-bearing. A3's post-mortem revealed something bigger — see below.
- **Logs:** everything ends up in `output/logs/` — the PBS job output lands there
  directly, and at job end `memo_server.log` / `pop.stdout.log` / `build.log` are
  copied there as `<jobname>_<jobid>_*.log` (live originals stay in the scratch
  run dir while the job runs).

## Known remaining issues

- **The year-2 wall (found in the A3 post-mortem, 2026-07-08): v7b barely made
  it.** Its 1981-01-01 restart is already mid-blowup — a 2Δt leapfrog
  oscillation in the top 3 levels of Labrador Sea cells (j=352–354, i=303),
  top cell flip-flopping −2 ↔ +14 °C per step. A3 died of the *same*
  instability on Dec-19; its missing P−E closure only moved convection onset
  ~2 weeks earlier (saltier surface). Root cause: the year's drift heat is
  stored in the subpolar *subsurface* (+8–9 °C at depth at 60°N Labrador vs
  ~3–4 real, already by Dec-22) — when winter convection finally starts, the
  surface taps that warm reservoir and POP's surface layer goes numerically
  unstable. MEMO's fluxes at those cells were sane winter values; live ice was
  negligible there. So the residual drift is not cosmetic: it is the year-2
  killer, and multi-year cannot proceed from the 1981-01-01 restart.
  **FIX VALIDATED (2026-07-08, v7c/v7d):** restarting from v7b's clean Dec-22
  state, a control with the identical stack reproduced the blowup (v7c died
  Jan-17, same Labrador mode → the wall is state-driven, not path-driven),
  while the single-variable fix **robert_nu 0.1→0.3** (v7d) completed the full
  45 days to 1981-02-05 with max |dSST/dt| 3.33 K/day and the Labrador column
  convecting *physically* (+7 → +0.1 °C under −250..−450 W/m² winter cooling).
  Rationale: GIAF uses nu=0.1 at a 1-h step; standalone runs 30-min, so the
  2Δt mode needs ~2–3× the damping. robert_nu=0.3 is the new baseline; the
  numerical detonation is solved, but the subsurface drift reservoir itself
  (next bullet) remains the physical-quality issue for multi-year.
- Residual drift +0.23 °C/month, accelerating slightly in autumn; salinity also
  drifts +0.13 psu/yr even with the P−E closure (closed to the training
  climatology, not to zero). The heat closure removes drift only through the
  global mean delivered at low latitudes — the subpolar gyre keeps its local
  surplus, which is exactly what loads the Labrador subsurface (above).
- Regional bias: east-Pacific cold tongue reaches ~34 °C even with the global
  budget correct (needs a training-side fix, not more forcing).
- Multi-year continuation mechanics are now validated (v7c/v7d: rpointer.ocn →
  restart file, ccsm_branch + namelist clock, flux_ramp 0, server cold-starts
  its EMA/memory state — see run_pop_gx1v7_v7d_dec22_robertfix.pbs); runs
  remain capped at 2014 by the forcing file.

## v8: the 5-year milestone (2026-07-09)

A cold-start **5-year run (1980–1984) completed with no blowup** — jobs
6675299 (yrs 1–3) + 6675300 (yrs 4–5, chained continuation), v7d stack
(v7b guardrails + robert_nu=0.3). All five winters crossed; the Labrador
cells that used to detonate now convect physically every year. Outputs in
`memo_pop_standalone_gx1v7_ice/run/v8_output/` (60 monthly pop.h, annual
restarts incl. r.1985-01-01, logs). Stability goal: **achieved**.

Fidelity is now the problem: global SST drifts +4 °C warm over the 5 years
(decelerating toward a warm quasi-equilibrium). A full band/box diagnosis
against the coupled GIAF reference (see `report_detailed.md`) overturned the
old "subpolar Qnet gap" hypothesis: global heat input is nearly right
(+2–4 W/m²), but the emulator's **mid-latitude wind stress collapses to ~25%
of truth after year 1** (input-drift feedback), so winter mixed layers never
form (20–40 m vs 80–190 m in GIAF from winter 1) and the heat surplus is
trapped in a thin surface film (+5 °C SST over only ±1.5 °C of 0–500 m heat
content). Salinity-pattern errors (fresh tropics, +1.5 psu poles, fresh caps
at the convection sites) supply the old detonation mechanism; AMOC is fine.
Fix ranking: (1) momentum-flux anchoring to the training τ climatology,
(2) raise the dSST/dt clamp to ~1–1.5 K/day, (3) freshwater/ice-FW fidelity,
(4) band-resolved Qnet delivery (demoted — symptom, not cause).
