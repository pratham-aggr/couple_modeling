# Experiments in flight

**Goal:** run the ocean model for 500 years.
**Problem:** the flux model needs winds every step, but we only have 35 years of wind data (1980–2014).
**Two ways out:** (A) don't need winds, or (B) generate our own winds.

---

## 1. kvjcofin coupled test — *does a wind-free model already work?*
- **Job:** 6977135 (Derecho)
- **Idea:** run `kvjcofin` (a flux net that takes **no wind inputs**) coupled to POP, no training.
- **Watch:** global-mean SST over 1 year. Stable ≈ pathway A basically solved for free.
- **Baseline:** prescribed-wind v15 year drifted +0.002 K.

## 2. kvjcofin **+ sensreg** fine-tune — *fix the physics of the wind-free model*
- **Job:** 5470998 (Casper, GPU, wandb: `ffelx09o`) — **RUNNING**
- **Idea:** take kvjcofin (exp 1) and fine-tune it with the SST-sensitivity regularizer so
  it stops warming itself (dQ/dSST +7 → physical −10..−25 W/m²/K). Still **no wind inputs**.
- **Why this replaces the from-scratch job (5470345, dead):** warm-started from a model that
  already makes the right fields, so it's the better version of pathway A.
- **Watch:** offline dQ/dSST audit first, then a coupled drift test.
- **Caveat:** fixes the *feedback*, not the ~14× too-weak **wind stress** kvjcofin showed
  coupled. If stress collapse still blows it up, pathway A is blocked → exp 3 is the path.

## 3. Wind generator, rollout-trained — *make our own winds*
- **Job:** 5470183 (Casper, 2 GPU, wandb: `mem24h-atm-rollout`)
- **Idea:** train a model to predict winds and feed its own output back (autoregressive).
  A first version blew up (winds → 20 m/s in a month); this trains on the full rollout to stay stable.
- **Watch:** the free-run `|wind|` probe each epoch — should stay ~7 m/s, not climb.
- This is pathway B (winds past 2014).

---

**Bottom line:** exp 1 or 2 succeeding = don't need winds. Exp 3 succeeding = can make winds.
Any one of them unblocks the 500-year run.

*(As of 2026-08-01: exp 2 running (`ffelx09o`), exp 3 queued, exp 1 done. Production 34-yr run paused at 1998, resumable.)*
