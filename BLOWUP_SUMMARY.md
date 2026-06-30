# MEMO–POP Blowup Diagnosis

**Date:** 2026-06-23  
**Status:** Root cause identified; two distinct failure modes confirmed across MSE and CRPS models

---

## Bottom Line

Our coupled MEMO→POP simulation kept crashing at high-latitude Southern Ocean cells. Numerical patches (flux ramping, smaller timestep) only delayed the crash — day 7, then 133, then 225 — never cured it. After correcting two measurement bugs and building a valid analytic truth baseline, we found the real cause: **MEMO's air-sea heat flux feedback has the correct stabilizing sign but is 20–250× too weak.** SST anomalies that the real atmosphere would suppress instead grow slowly until POP's numerics blow up at steep-topography shelf cells. MEMO is the driver; POP's numerical fragility there is the amplifier.

---

## Coupling Experiments Run

All runs start from the CESM2-LE 1980-01-01 ocean restart. Crash location is always the Antarctic shelf (Ross/Amundsen Sea, ~60°S). Two distinct failure modes were identified across experiments:

- **TLT abort** — `hmix_gm.F90:3393` (`Incorrect TLT computations`): KPP boundary-layer depth (HBLT) goes NaN under abrupt or strong fluxes on the first ocean step, propagating into the GM transition-layer solver. Fires fast (within days) when fluxes are applied at full strength from the start.
- **Tracer-CFL blowup** — vertical advection Courant number exceeds 1 at steep shelf cells. SST spikes from ~33°C to >45°C in a single timestep. Fires slower (weeks to months) as SST anomalies accumulate where MEMO under-damps.

### MSE model (`output_unet_mem24h_dsst_temporal_radprecip`)

| Configuration | Crash day | Failure mode |
|---|---|---|
| No damping | 7 | TLT abort (`hmix_gm:3393`) |
| + flux ramp (0→1 over 10 d) | 133 | Tracer-CFL blowup |
| + flux ramp + smaller timestep | 225 | Tracer-CFL blowup |

Flux ramp delays the TLT abort by softening the initial flux shock, but then a slower thermal blowup takes over — because MEMO's heat flux damping is only ~0.4–5% of the physical truth, SST anomalies at shelf cells grow until POP's vertical numerics fail.

### CRPS model (`output_unet_mem24h_dsst_temporal_radprecip_ucast`, run p5ge0uvh)

| Configuration | Crash day | Failure mode |
|---|---|---|
| No damping | ~8 | TLT abort (`hmix_gm:3393`) |
| + flux ramp + tau cap (τ < 0.5 N/m²) | ~15 | Tracer-CFL blowup (SST 33→46°C at step 59) |

Same two failure modes as MSE. The CRPS model crashes *earlier* than the equivalently damped MSE run (day 15 vs 133), despite being trained with a probabilistic loss that should produce sharper, more physically realistic flux distributions. The wind-stress cap fired on 1600–3700 cells per step in the damped CRPS run; zero cells in the undamped run before the TLT abort, suggesting the CRPS model produces lower peak stress but still under-damps heat fluxes enough for the thermal blowup to develop.

### Summary

Neither model architecture nor loss function (MSE vs CRPS) prevents the crash. Both models share the same root deficiency — missing wind speed and boundary-layer air temperature as inputs — which causes ~20–250× under-damping of turbulent heat fluxes regardless of how the model is trained.

---

## False Leads (Why This Took Time)

We went through three wrong conclusions before arriving at the correct one. The PI should know the final answer survived each correction.

1. **First: thought MEMO was missing the feedback entirely.** Wrong — MEMO does respond, just weakly.

2. **Then: thought MEMO had the WRONG sign and was 150–1300× too strong (destabilizing).** This was an artifact of two simultaneous bugs:
   - *Perturbation bug:* When we perturbed SST by 1 K to measure the flux response, a derived input channel (dSST/dt) was being silently recomputed, injecting a spurious 16σ kick to that channel. We were measuring MEMO's response to the wrong input, inflating the apparent sensitivity ~160×.
   - *Sign convention confusion:* The zarr training data stores fluxes with positive = out of ocean (standard CESM output), but we were initially treating them as positive = into ocean, flipping the apparent sign of the feedback.

3. **The "truth" we were comparing against was also invalid.** We used natural SST–flux co-variation from consecutive time steps as a truth baseline. This is correlational, not causal — it's confounded by reverse causality (the atmosphere drives SST changes too) and synoptic noise. The resulting estimate had a p10–p90 range of −350 to +438 W/m²/K — pure noise, not a feedback signal.

---

## The Correct Measurement

After fixing the perturbation bug (hold dSST/dt fixed when perturbing SST), sourcing the sign convention from the CESM bulk formula, and replacing the correlational truth with an **analytic derivative of the bulk flux formula** (computed in exactly the same way as MEMO's derivative — perturb SST, hold wind and air temperature fixed), the comparison became trustworthy.

The bulk formula (CESM `shr_flux_mod.F90`, Large & Yeager 2009):

```
SHFLX = ρ · Cp · Ch · |U| · (T_air − SST)   →   ∂SHFLX/∂SST = −ρ · Cp · Ch · |U|
LHFLX = ρ · Lv · Ce · |U| · (q_air − q_sat(SST))   →   ∂LHFLX/∂SST = −ρ · Lv · Ce · |U| · dq_sat/dSST
```

Both analytic partials are negative (stabilizing) by construction. This is the causal truth.

---

## The Real Finding

| Flux | Analytic truth (bulk formula) | MEMO | MEMO as % of truth |
|---|---|---|---|
| ∂SHFLX/∂SST | −13.4 W/m²/K | −0.05 W/m²/K | **0.4%** |
| ∂LHFLX/∂SST | −10.8 W/m²/K | −0.59 W/m²/K | **5.4%** |

**Verdict: MEMO under-damps (correct sign, far too weak) — it does not destabilize.**

FLDS (downwelling longwave) was initially flagged as wrong-sign, but further analysis showed the sign is actually **correct**: both MEMO and physics give a positive ∂FLDS/∂SST, because warmer SST eventually warms the boundary-layer air, which emits more downwelling LW (a mild positive feedback). MEMO's +2.4 W/m²/K is about 5× too large for a 6-hour model (the instantaneous truth is +0.5 W/m²/K, a timescale mismatch), but the direction is not wrong, and POP's upwelling emission term FLUS (−4.6 W/m²/K, always present) dominates the net longwave regardless. The FLDS issue is secondary.

---

## Why This Explains Everything

Under-damping — not destabilization — explains the observed pattern perfectly. SST anomalies grow slowly because MEMO suppresses them at only ~5% the correct rate. The blowup develops over months (not days), patches that add damping delay it proportionally, and POP's shelf numerics eventually amplify what MEMO failed to suppress. True destabilization would have crashed immediately and been immune to numerical fixes.

---

## Next Steps

1. **Why is MEMO under-damping?** The feedback sharpness depends on wind speed and air temperature — variables MEMO does not currently receive as inputs. It regresses fluxes from SST, ice, and insolation alone and cannot capture the wind-driven term ρ·Ch·|U| that sets the magnitude of ∂SHFLX/∂SST. MEMO now predicts boundary-layer air state (Tbot, Qbot) as auxiliary outputs at R²>0.97, but has not yet used them as inputs.

2. **Candidate fixes (for discussion):**
   - Add boundary-layer air state (Tbot, Ubot) as MEMO inputs — directly targets the missing feedback magnitude.
   - Penalize under-damped SST sensitivity in training.
   - Fix the FLDS sign defect (likely also needs atmospheric temperature input to represent the LW feedback correctly).
   - POP-side GM taming (reduce eddy diffusivity at shelf slopes) — useful complementary stabilizer but not a cure on its own.

3. **FLDS (longwave) is a secondary concern.** The FLDS sign is correct and its net effect is dominated by POP's FLUS term. The one actual FLDS defect is a timescale mismatch (MEMO applies a quasi-equilibrium response at 6-hour steps), which would partly self-correct if Tbot is added as an input.
