# Standalone MEMO → POP2 — Run Report

**Goal:** drive POP2 ocean directly from the MEMO ML flux emulator via the `cpl_none`
path (no CPL7/CICE) and reach a stable multi-year run.

**Status (2026-07-07): FULL PHYSICAL YEAR ACHIEVED — v7b (job 6664241).** Live ML CICE
emulator (mid-month anchoring) + global budget closures (ramp-gated + EMA) + NO local
restoring completed 1980-01-01→1981-01-01, "Successful completion of POP run", zero
aborts. It passed the December winter overflow/TLT wall at the SAME cold subpolar state
that killed v6c (end-Nov 50–65°N N Atl: v7b 11.97 °C ≈ v6c 11.99 vs v4's warm-biased
14.12) — an honest pass; interactive ice at the winter ice edge was the difference,
exactly as hypothesized. Drift +0.23 °C/month (≈ v6c, 4–5× better than v4's +1); all
extremes sane. Remaining defects: residual drift accelerates late in the year
(Sep–Dec ~+0.3–0.4 °C/mo), east-Pacific cold tongue ~34 °C regional bias, late-year
aice ~−0.14 vs the GIAF clim reference (partly LE2-vs-GIAF world mismatch).

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

## Restoring-era runs (post-v4) — VERDICT: local SST restoring is the killer
| Run | Config delta | Result |
|---|---|---|
| v5 (6638084) | + SST restore q=30, UNCAPPED | died Jan-10 (+956 W/m² hammer on a cold wiggle) |
| v5b (6644187) | restore capped ±150 W/m² | died **Aug-14 Hudson Bay** |
| v5c (6648445) | + prect/qflx caps 2e-3 | died **Aug-14 Hudson Bay** — caps hit only 2×, irrelevant (deterministic v5b replay) |
| U-Cast (6648567) | v5c guards, native CRPS model | passed Jan-12 (seam fix CONFIRMED in coupling) but died **Aug-17 Hudson Bay** (61.8°N 272°E) |

Three restoring runs, two different flux models, all dead at Hudson Bay Aug 14–17;
v4 with NO restoring finished the year. Post-mortem of the crash cell (j=344 i=277):
SSS 25 → −54 psu at doubling rate with SST/SSS diverging = grid-scale vertical
computational mode, NOT a surface-flux event — offline replay shows the emulator's
fluxes there are sane and *cooling* (−110 W/m²); freshwater channels were capped and
silent. The restoring-maintained basin state is what destabilizes POP's vertical
tracer advection there. Conclusion: no more per-cell forcing toward climatology.

## Verified today (2026-07-06)
- **No flux-sign bug**: normalizer means prove targets stored +into-ocean; Fortran
  assigns directly; consistent. POP ignores lhflx (latent = qflx·Lv via EVAP_F).
- **Model inputs are Kelvin** (offline probes must add 273.15 — a °C probe gives
  convincing garbage: LHFLX +725 W/m², FLDS 0).
- **CICE emulator coupling bug found+fixed**: cice_coupler.apply() masked FLDS to ~0
  under ice while POP's internal LWUP kept radiating → the Feb-02 −35385 °C cold
  detonation. Fixed with the same σT⁴ blend as ice-mask v2. The ice model itself is
  good (offline aice R²=0.974). First fixed run still died Jan-11 (emu aice MAE 0.125,
  bias −0.06) — live-ice lineage needs its own debugging (ramp/strocn/melth suspects).
- **solar.py was a revert casualty** — restored from 8094288 (like train_unet.py).

## Global budget closures (CAMulator appendix B, v6 lineage)
- v6 (6652674): closures anchored to 0 → subtracted −82 W/m² UNIFORMLY → polar
  convection death Jan-27. Lesson: the Qnet proxy has formula bias (training clim is
  +20 W/m² in Jan, not 0, seasonal range −13..+27) and P+E over ocean is negative
  (net evaporative, −3.2..−5.6e-6 kg/m²/s) — anchoring P=E ADDS freshwater.
- v6b (6652964): closures anchor the ANOMALY vs the training-data monthly clim
  (budget_clim_CREDIT_1980-84.nc, same proxy formula); heat correction distributed
  over low-latitude open water only (tapered 55–65°, zero poleward). **Died Jan-23,
  TLT** — post-mortem found two (i5) implementation bugs, not a calibration problem:
  (1) the closure ran INSIDE the 10-day flux ramp: the proxy saw ramped-to-~0 fluxes,
  read a −414 W/m² "deficit", and injected up to +436 W/m² of full-strength heat into
  the tropics for 9 days (warm pool 32.2→33.4 °C by Jan-22; Irminger cells jumping
  1.3–1.7 K/day) while P−E pinned prect at its ×0.25 clamp; (2) post-ramp the
  correction chased the INSTANTANEOUS global anomaly, flapping ±60 W/m² step-to-step
  (the proxy has a diurnal cycle; the monthly clim target doesn't).
- Confirmed drift signal: post-ramp the live proxy ran +40..+82 W/m² vs clim +20 —
  the ~+62 W/m² secular excess matches v4's +1 °C/month warm drift over a ~50 m
  mixed layer. The closure TARGET is right; the delivery was wrong.
- **v6c (6658070)**: same flags, fixed (i5): closures skipped until the flux ramp
  completes; Qnet offset and P−E scale go through a 3-day EMA (α=1/12) so only the
  secular drift is removed, never the diurnal/weather flapping. **Result: reached
  1980-12-02 restart, died ~Dec-05 — 11 months with NO local restoring, past the
  Feb-10 wall AND the Aug-14 restoring graveyard; best closure run ever.**
  - **Drift verdict: closures work.** Global-mean SST 20.81→23.22 °C over 11 months
    = +0.22 °C/month vs v4's +1 °C/month (≈5× cut). No unphysical SST/SSS anywhere
    in the daily means through Dec-05 (extremes: Persian Gulf 35.3 °C / 42.5 psu —
    warm but sane). Caveats: drift accelerates late (Sep +0.18, Oct +0.29, Nov
    +0.40 °C/mo) and the east-Pacific cold tongue sits at ~34 °C (regional warm
    bias survives the global closure).
  - **Death = the WINTER overflow/TLT mode, not a flux event.** Sub-daily blowup:
    daily means clean through Dec-05, then all three overflows thrash product
    points simultaneously (steps 1155085–87) and hmix_gm:3393 aborts. Overflow #3
    had been swapping sporadically since late Oct. Seasonality confirmed: original
    wall Feb-10 (winter), v6c death ~Dec-05 (winter onset), whole summer clean.
  - **Why v4 passed December: its warm drift suppressed the mode.** End-of-Nov v6c
    minus v4: 50–65°N N Atlantic −2.1 °C, taper band 55–65° −1.8 °C, Nordic sills
    −0.8 °C (SSS ~equal). v4's subpolar warm bath capped winter deep convection at
    the sills; v6c, being closer to truth, re-exposed the underlying POP-side polar
    instability (LE2 restart density vs POP numerics at the sills).
- **v7 (6664113, killed mid-run)**: v6c closures + live CICE with the init-step
  fix. The fix fired correctly but the live bias marched to −0.074 by day 9 —
  WORSE than the frozen-IC run. Offline forensics found the real convention:
  ice_ic_1980-01.npz is byte-identical to the training cache's aice_prev for the
  sample whose TARGET is Feb-1980, i.e. the IC is the JANUARY MONTHLY MEAN and
  the emulator maps mean(m)→mean(m+1). So "predict at Jan-1 from the IC" applies
  the FEBRUARY state in early January — a one-month LEAD. The scary SH collapse
  (bias −0.56 vs Jan clim) was a *correct* Feb prediction (LE2 Antarctic summer
  minimum, truth 0.04) applied 4 weeks early. NH was near-perfect (+0.005). The
  model is fine (teacher-forced Jan bias −0.014); both dead schedulings were
  timing bugs: original = one-month LAG (under-iced NH growth season → Jan-11
  TLT), init-fix = one-month LEAD (Antarctic melted a month early).
  Note: the aice_clim reference file is from the MEMO_GIAF year while IC/training
  are LE2-1231.002 — part of the apparent SH "bias" is reference mismatch; LE2 is
  the consistent truth since POP starts from the LE2 restart.
- **v7b (6664241) = THE FULL PHYSICAL YEAR.** cice_coupler rewritten to MID-MONTH
  ANCHORING — monthly means are valid mid-month, so hold the IC (this month's
  mean) until the 15th, then predict next month's mean and glide until the NEXT
  mid-month (29–31 d spans; per-glide `_ramp_span_s` replaces the fixed 20-d
  ramp). Applied ice is the time-interpolated monthly state all year: kills both
  the lag and the lead. **Result: 1980-01-01→1981-01-01 complete, zero aborts,
  ~2 h wallclock.** All three success criteria met: bias flat (+0.017 in March;
  late-year −0.14 is largely the LE2-vs-GIAF reference offset), passed Jan-11 and
  Feb-10, and passed the December winter wall — at the same cold subpolar state
  that killed v6c (end-Nov 50–65°N N Atl 11.97 vs v6c 11.99), so the pass is
  honest, not drift-assisted. Drift matches v6c (+0.23 °C/mo, global SST
  20.81→23.55). The winter overflow/TLT mode is beaten by ice that responds to
  the actual SST at the ice edge — the "concrete fix instead of capping things".

## Next
1. **Multi-year run on the v7b stack** — the original goal is multi-year stability.
   Forcing file spans 1980–2014 (non-cyclic; crashes past 2014), so up to 35 yr;
   start with 5 yr to see whether the +0.23 °C/mo drift equilibrates, accelerates,
   or the year-2 winter wall returns at the drifted state.
2. Residual drift (+0.23 °C/mo, accelerating Sep–Dec): the closure removes the
   global-mean anomaly, so what remains is proxy-formula bias + the low-lat-only
   delivery footprint. Candidates: include the 55–65° taper band at reduced
   weight; recalibrate the proxy against POP's actual SHF diagnostics.
3. Regional bias: east-Pacific cold tongue ~34 °C under a correct global budget —
   only a training-side fix (aux-state retrain, the CAMulator lesson: predict
   Ubot/Vbot/Tbot/Qbot/PS as autoregressive aux channels, drop dSST_dt) or a
   spatially-aware closure can touch it without reintroducing local restoring.
4. U-Cast: seam fix verified in coupling; rerun under the winning v7b stack for
   the probabilistic-vs-MSE year.
5. Ice polish: build the aice/hi reference from LE2-1231.002 (not MEMO_GIAF) so
   the live skill diag reads true; consider re-deriving strocn/melth scaling.

## v7-ucast (6665177, 2026-07-07): U-Cast under the v7b stack — DIED Jan-13
Same winning stack as v7b, only the flux model swapped (native U-Cast CRPS,
output_unet_gx1v7_ucast, coupled via the single eval-mode pass = ensemble median).
Died 1980-01-13 12:00: single-step grid-scale detonation at the Northwest Corner
front (j=325 i=8, 49.4°N 329°E) — SST −3630..+1119 °C one step after the front
cells were pinned at freezing under strong cooling (shflx −114, lhflx −84 W/m²).
Escalation was fast: local max |dSST/dt| 2 → 6 → 13 → 14 K/day over the last ~36 h
(v7b stayed ~1.5 at the same date). NOT the old seam bug (native grid, no remap) —
it is the model's own flux response at that front.
Coupled biases far worse than the MSE model despite better offline test scores:
- PRECT ~3× too low globally (P−E closure pinned at ×2.9–3.1, drifting toward its
  ×4 clamp; v7b sat at ×1.13) — the CRPS median collapses precipitation.
- Qnet ~+30 W/m² hotter (smoothed correction −55 vs v7b's −25..−35).
Lesson: offline one-step skill (CRPS/R² better on 7/8 vars) does not transfer to
coupled stability; the eval-mode median of a CRPS/MC-dropout model is not a
feedback-consistent point forecast. Candidate fix if U-Cast coupling is pursued:
couple the MC-dropout ensemble MEAN (4–8 passes/step, ~8× inference cost) instead
of the median, or retrain S2 with a shorter CRPS fine-tune.
Logs: output/logs/memo_pop_g17v7u_6665177_*.

## Ablations a1/a2/a3 (6665240/41/42, 2026-07-07): every guardrail is still load-bearing
One variable removed per run, everything else = the v7b winning stack. All three
failed; the stack has no removable parts:
- **A1 — no dSST/dt input clamp** (`--dsst_cap 0`, job 6665240): died 1980-02-07.
  The unbounded warming feedback returned, this time at 70.6°N in the Norwegian
  Sea (SST +66795 °C, qnet −7.9e6 W/m²), hmix_gm:3393 abort. The clamp is not a
  legacy of the pre-closure era — the feedback loop is intrinsic to the flux model.
- **A2 — no 10-day flux ramp** (`--flux_ramp_days 0`, job 6665241): died
  1980-01-12, day-12 cold-start detonation at the Northwest Corner front
  (j=321,i=316, 47.8°N): FLDS blew to ~7200 W/m², global Qnet +235 W/m²,
  hmix_gm:3393. The restart-shock theory behind the ramp may have been wrong,
  but the ramp itself is doing real work at the fronts.
- **A3 — no P−E closure** (Qnet closure kept, job 6665242): the important one.
  Survived 11.5 months — past every previous wall — then went **all-NaN in a
  single 6-h window** (last clean exchange 1980-12-19 00Z). No TLT-FAIL, no
  solver warning — a silent death (the server then crashed on the all-NaN SST
  and POP hung to walltime, hence "walltime exceeded" in the PBS log; logs
  were NOT auto-copied, they live in the v6 rundir). Global SSS drift without
  the closure: +0.193 psu/yr vs v7b's +0.133.

### 2026-07-08 deep post-mortem of A3 — v7b hit the same wall; drift heat is the year-2 killer
- Escalation signature in A3's server log: max |dSST/dt| input sat at ~1.5
  K/day noise all December, then Dec-16→19 climbed 2.7 → 6.8 → 9.3 → 12.35
  K/day (clamp cell-count flat at ~32k — one accelerating site, not a global
  mode). **v7b's log shows the SAME escalation in its final exchanges**
  (Dec-31: 2.7 → 4.05 → 7.42 → 9.12 K/day) — the year ended hours before
  detonation.
- Site: Labrador Sea, (j=344–347, i=302–303) in A3, (j=352–354, i=303) in
  v7b's restart. **v7b's 1981-01-01 restart is mid-blowup**: |TEMP_CUR −
  TEMP_OLD| up to 16 K at the surface, top cell flip-flopping −2.000 ↔ +13.98
  °C — a 2Δt leapfrog oscillation confined to k=0–2 (calm below k=3). A3's
  day-18 daily means (warm +3 K AND fresh −5 psu in one day) are
  mid-oscillation garbage, not physics.
- **Exonerated**: MEMO fluxes at the site were ordinary winter values (shflx
  −80, lhflx −100 W/m², no precip, polar night); live ice negligible there
  (aice ≤ 0.03, melth ≤ 0.3 W/m² — checked from A3's memo_restart.pth coupler
  state); no remap seam (native grid). Note in passing: cice_coupler's
  melth = ρL·dhi/dt is POSITIVE (warms the ocean) during prescribed ice
  GROWTH, with no brine counterpart — harmless here (~0.3 W/m²) but worth a
  melt-only clip later.
- **Root cause: the year of drift heat is stored in the subpolar SUBSURFACE.**
  v7b's Dec-22 restart at (354,303): T = +8–9 °C from k=1 down through k=15,
  S 34.7→35.4 — real Labrador Sea is 3–4 °C. Salinity keeps the column
  marginally stable until winter cooling densifies the surface; convection
  onset then mixes +9 °C water into the surface layer under continued winter
  cooling, and POP's top-layer leapfrog goes unstable (the Qnet closure only
  removes the GLOBAL mean, delivered at low lat — the subpolar gyre keeps its
  local heat surplus all year).
- **Reinterpretation of A3**: the missing P−E closure did not cause a distinct
  death — it left the surface ~saltier, so convection onset (and the same
  blowup) came ~2 weeks earlier (Dec-16 vs Dec-31). The freshwater closure
  delays the wall; it does not remove it.
- **Consequences**: (1) multi-year CANNOT continue from the 1981-01-01
  restart (corrupted); last clean v7b restart ~Dec-22, but continuation dies
  at the same wall within days. (2) "Residual drift" is promoted from
  cosmetic defect to the primary blocker: the fix must stop the subpolar
  subsurface warming — spatially-aware closure (include the subpolar band in
  delivery) and/or proxy recalibration against POP SHF. (3) Death-signature
  catalog: TLT abort (hmix_gm:3393) vs silent NaN with top-layer 2Δt
  oscillation — grep both.

## v7c (6673562, 2026-07-08): Dec-22 continuation CONTROL — wall reproduced, state-driven
First continuation-restart run (mechanics validated: rpointer -> v7b's
r.1980-12-22 from the archive, ccsm_branch with namelist clock Dec-22 00Z,
stop nday 45, flux_ramp 0, server cold state). Result: **died 1981-01-17,
hmix_gm:3393**, after the same Labrador escalation — max |dSST/dt| quiet
(~2 K/day) through Jan-1, first flare Jan-2–3 (3.7→6.9), simmering ~4 K/day
through Jan-12, terminal runaway Jan-13→17 (10→16.5 K/day). New watch cells
caught the oscillation live: (j=354,i=303) SST flip-flopping 6.7→5.8→6.6→5.9
between 6-h exchanges under sane fluxes (shflx −80..−118, lhflx −83..−123,
flds ~283 W/m²) — flux innocence now confirmed per-step, not just post-hoc.
It ran ~3.5 weeks past v7b's ignition point (cold server EMA/memory, re-derived
overflow state — trajectory not bit-continuous, as expected), but the same mode
at the same site killed it: **the mid-winter Labrador wall is a property of the
drift-warmed STATE, not of the approach path.** Outputs: v7c_output/ in the ice
rundir. Next: v7d = identical + robert_nu 0.1→0.3 (2Δt damping ×3).
Ops notes: v7b's outputs were archived to
`memo_pop_standalone_gx1v7_ice/run/v7b_output/` (68 restarts incl. the
1981-01-01 final, 12 monthly + 12 nday1 history files — GIAF-named:
`g.e21.MEMO_GIAF_v01.pop.h.*`) before A1 reused that rundir; A3 reused the v6
rundir, overwriting v6c's nday1 files (v6c post-mortem numbers are already
recorded above). PBS scripts: run_pop_gx1v7_a{1,2,3}_ablation.pbs.

## v7d (6673640, 2026-07-08): robert_nu 0.1→0.3 — SURVIVED the wall. FIX VALIDATED
Identical to v7c (same Dec-22 restart, same stack, same 45-day window) except
one namelist value: `robert_nu = 0.3` (Robert–Asselin filter strength; damps
the 2Δt leapfrog computational mode ×3). Result: **completed all 45 days,
Dec-22 → 1981-02-05, 128× "Successful completion"**, restarts written through
r.1981-02-05. Max |dSST/dt| over the entire run: **3.33 K/day** (v7c hit 16.48
and aborted Jan-17). The Labrador watch cell (j=354,i=303) tells the physical
story: instead of v7c's stuck oscillation (~+6.2..6.8 °C flip-flopping to
death), v7d's column VENTS — SST descends +7.05 → +2.99 °C by mid-January and
sits at +0.1..0.3 °C by Feb-04 under −250..−450 W/m² net cooling (shflx −250..
−330, lhflx −120..−190, flds ~256 W/m², polar night). Neighbor (346,303)
reached −1.58 °C with 1% ice — winter deep convection proceeding physically
over the drift-warmed reservoir. Sanity vs control: January global-mean SST
v7d 20.348→20.479 vs v7c 20.394→20.514 (same trend; v7d slightly cooler
because Labrador actually releases its stored heat); extremes sane throughout
(SSTmax 35.24 → 34.85 by Feb-04, SSTmin −2.0 = freezing floor).

Why nu=0.3 is justified, not a band-aid: the GIAF coupled reference uses
robert_nu=0.1 at dt_count=24 (1-h step); standalone runs dt_count=48 (30-min),
halving the 2Δt-mode period, so the reference nu under-damps by construction.
robert_alpha=1.0 and time_mix_freq=17 unchanged. The fix removes the NUMERICAL
detonation only — the +8–9 °C subpolar subsurface drift reservoir still exists
and still gets convected up each winter; the spatial drift fix (subpolar-band
Qnet closure delivery / proxy recalibration vs POP SHF) remains the physical-
quality blocker for multi-year. Outputs: v7d_output/ in the ice rundir
(10 restarts, nday1 for Dec-22/Jan/Feb, both logs). PBS:
run_pop_gx1v7_v7d_dec22_robertfix.pbs. Logs: output/logs/memo_pop_g17v7d_6673640_*.

**Validation loop verdict: control (v7c) reproduced the blowup from the clean
Dec-22 state → state-driven; single-variable fix (v7d, robert_nu only) crossed
the wall with physical convection → robert_nu=0.3 is the new baseline for the
14-month cold-start rerun and multi-year work.**

## v8 (s1 6675299 + s2 6675300, 2026-07-09): 5-YEAR RUN COMPLETE — 1980-1984, no blowup

Full cold-start 5-year run with the v7d stack (v7b guardrails + robert_nu=0.3),
split into two chained jobs to dodge the >10 h queue starvation: s1 = 1980-01-01
cold start, 3 years (done 08:00, 128× "Successful completion", handed off
r.1983-01-01); s2 = ccsm_branch continuation, 2 years (done 14:11, 128×
"Successful completion", final restart **r.1985-01-01** written). ~2.1 h
wallclock per model year on 128 ranks. All five winters crossed, including four
crossings of the (former) year-2 wall — the Labrador detonation cells
(j=352-354, i=303) convect physically every winter under nu=0.3.

End-state health: max |dSST/dt| 1.62 K/day (clamp saturated on ~30k cells
throughout, never a runaway), SSTmax 43.3 °C (Persian Gulf warm pool — drifted
but stationary), live-ice CICE emulator bias −0.36 aice vs clim at year 5.
Outputs: 60 monthly pop.h + annual restarts + logs hard-linked into
`memo_pop_standalone_gx1v7_ice/run/v8_output/`; server logs archived as
`output/logs/memo_pop_g17v8s{1,2}_667529{9,300}_*`.

**Stability ≠ fidelity — the run drifts hard.** Global-mean SST: 17.97 °C
(1980-01) → 21.40 → 22.67 → 23.68 → ~23.7 °C (Dec-80/81/82 → 1984), vs the
coupled GIAF reference (g.e21.MEMO_GIAF_v02_nocap) 18.88 → 19.05 → 19.78 →
20.21 → 19.8. The drift decelerates (quasi-equilibrium ~+4 °C warm) but the
state is far from the training distribution.

## v8 drift diagnosis (2026-07-09): it is a MOMENTUM/VENTILATION problem, not a heat-input problem

Band/box diagnosis of all 47-60 v8 months against GIAF (scripts: scratchpad
diag_band_shf.py / diag_band_hc.py / diag_amoc.py; data band_shf_diag.npz).
Findings, in causal order:

1. **Global heat input is nearly right.** v8−GIAF global-mean SHF error:
   −4.0 (1980, ramp year), +2.3, +2.7, +3.8 W/m² (1981-83). The global Qnet
   closure works as designed.
2. **The ocean stopped mixing — heat is trapped in a surface film.** Tropical
   Dec SST errors reach +4..+5.4 °C while 0-500 m heat-content errors are only
   ±1.5 °C-equivalent. A +3-4 W/m² surplus into a ~30 m trapped layer
   reproduces the observed +1.2 °C/yr exactly. Subpolar-gyre winter mixed
   layer: **19-44 m vs GIAF 76-186 m, from the VERY FIRST winter** — a forcing
   property, not accumulated drift.
3. **Proximate cause: wind-stress collapse with a positive feedback.**
   |TAUX| 40-60N: applied Jan-80 = 0.057 N/m² (~73% of the 0.093 truth) but
   ALL later winters 0.013-0.024 vs GIAF 0.045-0.063 (~25%). Same model, same
   clamp ⇒ input-driven: the drifted SST pattern degrades the wind prediction,
   weak winds → shallow MLD → more trapping → more drift → weaker winds.
4. **dSST/dt input clamp (0.1 K/day) saturates from step 1** (25-30k cells at
   steady state, 76k in month 1) — not the month-1 differentiator, but it
   permanently amputates an input channel the model trained on unclamped.
5. **Salinity pattern errors supply the detonation mechanism.** Tropics fresh
   (0-10N −0.38 → −0.84 psu by yr 3; the global P−E scalar can't fix pattern),
   poles +1.5 psu (live-ice aice bias + crude ice-FW physics), and the exact
   detonation cells carry a FRESH CAP (dS −0.32 psu at 35 m) over saltier
   water — suppressing gradual convection until catastrophic overturn.
6. **AMOC-collapse hypothesis REJECTED** (v8 28-31 Sv vs GIAF 28-34).
   Persian Gulf has a real local Qnet bias (+34 W/m² yr 1) — secondary.

**Revised fix ranking** (demotes the previously planned band-resolved Qnet
closure to symptom treatment): (1) momentum-flux anchoring — spatially
resolved correction of TAUX/TAUY toward the training monthly climatology,
restoring storm-band stress, MLD ventilation, gyre forcing; (2) raise the
dSST/dt clamp to ~1.0-1.5 K/day now that robert_nu=0.3 handles overturns
(A1 proved removing it entirely is fatal); (3) freshwater/ice-FW fidelity
(melt-only clip + brine in cice_coupler.py:294, pattern-aware P−E);
(4) band-resolved Qnet delivery.

### Wind-collapse hypothesis VALIDATED-AND-REVISED (2026-07-09, offline controlled experiment)

`validate_tau_hypothesis.py` (archived in `v8_output/diagnosis/`) ran the
actual ql6lbazr flux model on Jan-1984 inputs, varying only the SST-related
channels. Mean |TAUX| over 40–60N ocean (N/m², truth = 0.186):

| case | inputs | |TAUX| |
|---|---|---|
| T1 | truth 6-h SST + real 24-h memory + unclamped dSST/dt | 0.174 |
| T2 | T1 with dSST/dt clamped at 0.1 K/day | 0.160 |
| P3 | truth SST, SST_prev=SST, dSST/dt=0 | 0.053 |
| G | GIAF monthly SST spliced, no memory | 0.039 |
| V | v8 drifted monthly SST spliced, no memory | 0.035 |
| — | applied in the real runs (monthly pop.h): v8 0.014, GIAF 0.049 | |

Verdict: the model's intrinsic winter wind skill is excellent (T1 = 93% of
truth) and the clamp costs only 8% (T2) because clamping preserves the
tendency's SIGN pattern. The collapse comes from ZEROING the temporal
channel (T2→P3, −67%): **the model reads winter storms out of the 24-h
SST-tendency pattern (storm cooling wakes), and the standalone system has
no atmosphere to imprint synoptic tendencies on SST** — its only tendencies
are ocean drift + diurnal, so the storm channel is structurally empty. The
drifted-SST-pattern feedback exists but is secondary (G→V, −11%). Month 1
looked better (73% of truth) only because the cold-start shock produced
large, spatially structured tendencies that mimicked weather. The coupled
GIAF reference (also MEMO-driven) sits in between because real CICE/DATM
inject some variability.

Consequence for the fix: τ-climatology anchoring restores the MEAN stress
(gyres, Ekman pumping) but not the synoptic variance that does much of the
mixing — pair it with stochastic τ perturbations carrying training-derived
winter variance. Raising the dSST/dt clamp will NOT recover the winds.
