# Fixing the warm drift in the ocean run

Plain-language guide to how we're fixing the flux UNet so the standalone ocean
run stops drifting too warm — using only our own trained network, no second ML
model.

> Base network training is in [README.md](README.md). The stability fixes that
> made the 5-year run possible are in [README_standalone_v8.md](README_standalone_v8.md).
> This file is about **making the network's physics correct**.

---

## What's going wrong

We run the ocean model (POP) with our trained network supplying the surface
heat and wind fluxes. The problem: after one simulated year the ocean ends up
**too warm** — about 1 to 3 °C above the real answer (the target is 18.7 °C).

**Why.** When the ocean surface warms, it should *lose* extra heat to the air,
which pulls the temperature back down. That's the natural brake. Our network had
the brake **backwards** — a warmer ocean got *more* heat, not less — so small
warm spots grew instead of fading. That's the drift.

There are two versions of this brake, and we have to fix both:

- **Slow brake** — how the network reacts to an ocean that *stays* warm.
- **Fast brake** — how it reacts to an ocean that is warming *right now*.

---

## What we tried, in order

**Step 1 — Add the weather as inputs.**
Gave the UNet 5 extra input channels — the real air state at the surface
(winds Ubot/Vbot, temp Tbot, humidity Qbot, pressure PS) — and retrained.
Inputs went 6 → 11 channels. **Drift halved: +2.7 → +1.3 °C** above truth.

*Why it worked.* With only SST/ICEFRAC/SOLIN as inputs the network had no
information about the wind — wind is weather, and on a 6-hour step it is nearly
independent of the ocean state. Asked to predict something unpredictable, MSE
training does the optimal thing and returns the conditional mean, which for a
field that changes sign is close to zero. So wind stress collapsed toward zero
everywhere. No stress → no wind-driven mixing → the heat the network delivered
stayed trapped in a thin surface layer instead of being stirred down, and the
surface warmed. Dec-1980 westerly-belt stress (dyne/cm²):

| | SH 40–60°S | NH 40–60°N |
|---|---:|---:|
| before | +0.06 | −0.19 |
| after | **+1.27** | **+0.87** |

A factor of ~20 in the Southern Ocean, back into the physically realistic range.
(The negative NH value before was a separate stress sign-flip bug, fixed earlier.)
Stress becomes predictable once you supply the winds — it is essentially
ρ·Cd·|U|·U. Tbot and Qbot matter for the same reason on the heat side: sensible
and latent flux depend on the air–sea *difference*, which the network had been
guessing from SST alone.

Note this is an **information** fix, not a feedback fix — which is why +1.3 °C
of drift survived it. Steps 3–5 are about the feedback.

**Step 2 — Penalize the wrong brake (failed).**
Added a loss term to training: measure the network's `d(heat)/dSST` by running it
at SST±δ, and penalize the gap to the correct bulk-physics value. Swept the
penalty weight, then tried gradient surgery (PCGrad) to stop it hurting accuracy.
Every setting either wrecked accuracy or left the brake unchanged — the labels
themselves encode the wrong correlation, so a penalty just fights the data.
Dropped it.

**Step 3 — Put the right brake into the training labels (slow brake).**
Instead of penalizing, we built corrected examples. For a fraction of samples:
shift the ocean temperature by `dS` (random ±2 °C), and add the true heat
response `bulk(SST+dS) − bulk(SST)` — from the CESM bulk formula at fixed air
state — onto the sensible/latent heat labels. Everything else (radiation, wind
stress) held fixed. Then train with plain MSE, no penalty. To make it the *slow*
brake, we shift both SST-now and SST-24h-ago together (so the "warming rate" input
doesn't change). Result: slow brake corrected (−23 → −30), accuracy unchanged —
but the coupled ocean run moved only ~10% (20.0 → 19.9 °C).

**Step 4 — Locate the real problem.**
Ran the physics check offline (`diag_dqdsst_atmin.py`): the *fast* brake — the
response to the warming-rate input — was **+107** (should be about −30), a
runaway. That's why a `dSST/dt` clamp was firing on ~1/3 of ocean cells to keep
the run alive. That clamp is the patch we want to delete.

**Step 5 — Same trick, fast brake (worked).**
One change from Step 3: shift only SST-now, not SST-24h-ago, so the warming-rate
input *does* jump — but keep the exact same heat-response label (a bulk flux
depends on temperature, not on how fast it's changing). Mode `both` mixes the two
so it fixes the fast brake while holding the slow one. Result:

| Brake | Before | After | Should be |
|---|---:|---:|---:|
| Fast (`dSST/dt` channel) | **+107** | **−22.6** | −10 … −40 |
| Slow (SST level) | −29.9 | −30.3 | ≈ −30 |
| Accuracy (val loss) | 0.0494 | 0.0499 | unchanged |

Negative in every latitude band, and latent heat dominates over sensible
(−22.5 vs −8.1) as textbook physics requires. Check it yourself with
`scratchpad/diag_dqdsst_atmin.py <model_dir>`.

**Step 6 — Delete the band-aid (crashed; under investigation).**
Coupled 1-year ocean run with the `dSST/dt` clamp **off** (`--dsst_cap 0`).
The clamp was genuinely gone — it used to fire on ~1/3 of ocean cells every step
and printed nothing here. But the run **died in early May 1980** after 4 months:

```
ERROR: Incorrect TLT computations   (hmix_gm.F90:3393, transition_layer)
```

That's the GM lateral-mixing failure, a known pre-existing crash in this stack
(it also killed the v8 extension in Dec-1995) — *not* the signature of a runaway
surface feedback. Fields through April showed no degradation: v16 tracked v15
within 0.1 °C globally with near-identical T/S extremes. So it's a sharp local
event, not a slow blowup.

Two things changed between v15 (survived 12 months) and v16 (died in month 5):
the weights, and the clamp. A control run — **v16 weights with the clamp back
on** — separates them. Survives the year → the clamp removal is the cause.
Dies in May again → the weights are steepening isopycnal slopes and the clamp
is beside the point.

Also worth noting: v16 ran ~0.11 °C *warmer* than v15 in every month it
completed. The instant fix did not reduce drift over those four months.

---

## Results so far

Ocean temperature after one year (target = 18.7 °C, lower is better):

| Version | Temperature | Too warm by |
|---|---:|---:|
| Target (truth) | 18.7 | — |
| Before this work | 21.4 | +2.7 |
| Step 1 (weather) | 20.0 | +1.3 |
| Step 3 (slow brake) | 19.9 | +1.2 |
| Step 5+6 (fast brake, no clamp) | *crashed in May — no Dec value* | — |

Every step kept the network just as accurate on real data as it started.

---

## What still props the run up

Two temporary supports were added to keep earlier runs alive. Neither can ship —
both are forms of flux adjustment, and each one hides a real defect:

| Support | What it hides | Status |
|---|---|---|
| `dSST/dt` clamp | the broken fast brake | **removed in Step 6** |
| `--qnet_balance` | the ocean's energy budget doesn't close on its own | still on |

The energy metric is the global mean of `SHF` in the `pop.h` output (W/m², total
surface heat flux). Zero means closed. The old run sat permanently at **+11**;
that heat had nowhere to go but into the ocean, which is the drift. Closure
should *emerge* from a correct brake — forcing it to zero by hand would just be
another adjustment. Turning `qnet_balance` off is the next test after Step 6,
and it's the one that could legitimately fail.

---

## Reproducing

```bash
# train (casper A100 — request 6h walltime, not 24h; 24h gets parked by the scheduler)
qsub scripts/submit_gx1v7_perturb2.pbs

# check the physics before running any ocean model
python scratchpad/diag_dqdsst_atmin.py output/output_unet_gx1v7_perturb2

# coupled 1-year ocean run, clamp off
qsub -q develop@desched1 camulator_ud/climate/run_pop_gx1v7_v16instant.pbs
```

| File | What it is |
|---|---|
| `train_unet.py` | `--perturb_mode sustained\|instant\|both` selects which brake gets taught |
| `scripts/submit_gx1v7_perturb2.pbs` | Step 5 training job |
| `camulator_ud/climate/memo_config_gx1v7_perturb2.yml` | points the ocean run at the Step 5 weights |
| `camulator_ud/climate/run_pop_gx1v7_v16instant.pbs` | Step 6 ocean run (`--dsst_cap 0`) |
| `scratchpad/diag_dqdsst_atmin.py` | the physics check — measures both brakes |

---

## Separate open problem

North Atlantic winter mixing is far too shallow — 28 m against ~198 m in the
reference. This is a sea-ice freshwater issue, not a heat-flux one, and it is
what ended the earlier 5-year attempt after 3.1 years. It is tracked on its own
and is the blocker for any multi-year run.
