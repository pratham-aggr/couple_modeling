# Why the ocean model keeps crashing

## Setup

We replaced the atmosphere in a climate model with **MEMO** — a neural network that
predicts air-sea heat and momentum fluxes from ocean surface conditions. The ocean
model (POP) runs driven by MEMO's predictions instead of a real atmosphere.

The goal is a 35-year simulation. We can't get past 8 months.

---

## The symptom

The ocean model crashes the same way every time: a single cell near a steep
underwater slope, at high latitude, during winter, explodes within hours.
No warning the day before — then suddenly temperatures of thousands of degrees.

Every fix we tried just pushed the crash later:

| What we tried | Crashed on day |
|---|---|
| Nothing | 7 (Jan) |
| Slow flux ramp-up at startup | 133 (May) |
| Halve the timestep | 225 (Aug) |

Same crash, different day, different cell. Something fundamental is wrong.

---

## What's actually happening

### Ocean side

The crash is a numerical instability: a cell near a steep underwater cliff develops
runaway upward water motion. The ocean model's math breaks when vertical velocity
grows too fast. At the crash: vertical tracer transport was **66× over the stability
limit** while horizontal motion was fine (0.61× the limit). It's a local, violent
event — not a gradual drift.

### Atmosphere side — the real problem

MEMO takes 6 inputs: sea surface temperature (SST), sea ice, and solar radiation —
all from the ocean. **It has no information about air temperature.**

In the real world, there's a natural brake on ocean cooling: when the ocean surface
gets cold, the air above it gets cold too, which reduces how much heat the ocean
loses. The ocean and atmosphere stabilize each other.

MEMO can't do this. It never sees the air temperature, so it can't reduce heat loss
when the ocean cools. We tested this directly:

**When SST drops by 1°C, what happens to ocean heat loss?**

| | Sensible heat (SHFLX) | Latent heat (LHFLX) |
|---|---|---|
| Real coupled atmosphere | −0.1 W/m²/°C *(less loss → stabilizing)* | −3 W/m²/°C *(less loss → stabilizing)* |
| MEMO | **+185 W/m²/°C** *(more loss → destabilizing)* | **+434 W/m²/°C** *(more loss → destabilizing)* |

MEMO gets the **sign wrong** and is **150–1300× too large**.

When the ocean cools, MEMO increases heat loss instead of reducing it — making the
ocean even colder, triggering deeper convection, driving faster vertical motion, until
the numerical limit is hit.

**How fast does MEMO amplify a cooling event?**

A 1°C SST drop causes extra cooling of:
- Real atmosphere: −0.005°C per 6-hour step
- MEMO: **−0.053°C per 6-hour step** (10× larger, wrong direction)

Over one day, MEMO adds ~0.2°C of spurious extra cooling on top of whatever the
ocean is already doing — enough to push fragile cells over the edge.

---

## Why the fixes didn't work

- **Timestep reduction**: delays the crash (less momentum per step) but the
  destabilizing MEMO feedback is still there, so another cell eventually tips over.
- **Wind stress cap**: wind wasn't the driver — capping it barely helped.
- **GM mixing reduction**: amplifies the instability once it starts, but not the cause.
- **Atm-aux model (kvjcofin)**: same 6 inputs, same problem, same wrong-sign sensitivity.

---

## The fix

The root cause is a **missing input**: MEMO needs near-surface air temperature
to compute the correct air-sea feedback.

| Option | How | Downside |
|---|---|---|
| SST restoring in POP | Add a weak pull toward observed SST in the ocean model | Slightly constrains SST |
| Retrain MEMO with `Tbot` as input | Give the network the missing variable | Requires retraining |
| Penalize sensitivity during training | Regularize so \|dF/dSST\| stays physical | Requires retraining |

The quickest fix is SST restoring. The proper fix is retraining with air temperature as an input.
