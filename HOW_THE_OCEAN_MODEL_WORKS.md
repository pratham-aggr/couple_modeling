# How Everything Works

We take a **real ocean model** (POP2, the ocean component of CESM2) and, instead of coupling
it to a real atmosphere, we couple it to **a neural network we trained ourselves**. Every 6
simulated hours the ocean reports its sea-surface temperature (SST); the network replies with
the heat, freshwater, and wind-stress fluxes the ocean should feel; the ocean marches forward
and its SST changes; the loop repeats. A second small network stands in for sea ice. A few
things the ocean genuinely cannot know on its own — the incoming sunlight and the near-surface
winds/air — are **prescribed** from the historical record (the classic ocean-only "OMIP"
setup). The result is a cheap, self-contained ocean simulator whose **flux engine is our own
network**, built to stay physically honest for years with **no hand-tuned corrections**.

```
   +---------------------------------------------------------------+
   |  PRESCRIBED  --  the atmosphere that actually happened         |
   |    * sunlight (SOLIN)                                          |
   |    * near-surface winds & air:  Ubot Vbot Tbot Qbot PS         |
   +-------------------------------+-------------------------------+
                                   |  (read from files, remapped to ocean grid)
                                   v
      +----------------------------+                +-----------------+
  +-->|   FLUX UNet  (our model)   |--- fluxes ---> |                 |
  |   |   11 inputs  ->  8 outputs |                |   POP2  OCEAN   |
  |   +----------------------------+                |  (real physics, |
  |                 ^                               |    60 levels)   |
  |         ICEFRAC |                               |                 |
  |   +-------------+--+                            +--------+--------+
  |   | CICE emulator  | <-------- SST ---------------------+
  |   | (our ice net)  |                                    |
  |   +----------------+                                    |
  |                                                         |
  +---------------------- new SST <-------------------------+
                (this feedback loop is the whole game)
```

| Component | What it is | Whose |
|---|---|---|
| **POP2** | Real ocean GCM, 320×384 (≈1°) grid, 60 vertical levels. Owns the ocean state. | NCAR (unchanged) |
| **Flux UNet** | ~7.7 M-parameter U-Net mapping surface state → air-sea fluxes. **This is the product.** | Us |
| **CICE emulator** | Tiny network predicting sea-ice fraction/thickness/velocity. | Us |
| **model_server.py** | Long-running Python process: holds the UNet, remaps grids, talks to POP. | Us |
| **Forcing files** | Sunlight + near-surface atmosphere, 1980–2014, 6-hourly. | Data (read-only) |

### The 11 inputs

| # | Abbrev | What it is | Units | Source / kind |
|---|---|---|---|---|
| 0 | **SST** | Sea-Surface Temperature | K (from POP's °C) | POP — **live feedback** |
| 1 | **ICEFRAC** | Sea-Ice area Fraction | 0–1 | ice emulator — **generated** |
| 2 | **SOLIN** | Solar INsolation (top-of-atmosphere incoming SW) | W m⁻² | forcing file — **prescribed** |
| 3 | **SST_prev** | SST 24 h earlier | K | ring buffer — live (derived) |
| 4 | **ICEFRAC_prev** | ice fraction 24 h earlier | 0–1 | ring buffer — live (derived) |
| 5 | **dSST_dt** | SST time-tendency, $(\text{SST}-\text{SST}_{\text{prev}})/86400$ | K s⁻¹ | ring buffer — live (derived) |
| 6 | **Ubot** | zonal (eastward) wind, bottom/near-surface level | m s⁻¹ | forcing file — **prescribed** |
| 7 | **Vbot** | meridional (northward) near-surface wind | m s⁻¹ | forcing file — **prescribed** |
| 8 | **Tbot** | near-surface air Temperature | K | forcing file — **prescribed** |
| 9 | **Qbot** | near-surface specific humidity (Q) | kg kg⁻¹ | forcing file — **prescribed** |
| 10 | **PS** | surface Pressure | Pa | forcing file — **prescribed** |

### The 8 outputs

All eight are predicted by the network; the last column is the physics/convention conversion
the server applies before POP sees them

| # | Abbrev | What it is | Raw unit | Delivered to POP |
|---|---|---|---|---|
| 0 | **TAUX** | zonal wind stress $\tau_x$ | N m⁻² | ×(−1) → into-ocean |
| 1 | **TAUY** | meridional wind stress $\tau_y$ | N m⁻² | ×(−1) → into-ocean |
| 2 | **SHFLX** | Sensible Heat FLuX | J m⁻² / 6 h | ÷21600 → W m⁻² (+ = into ocean) |
| 3 | **LHFLX** | Latent Heat FLuX | J m⁻² / 6 h | ÷21600 → W m⁻² (+ = into ocean) |
| 4 | **QFLX** | evaporative freshwater (water-vapour) flux | m / 6 h | ×1000/21600 → kg m⁻² s⁻¹ |
| 5 | **FSDS_J** | surface Downwelling Shortwave (FSDS) | J m⁻² / 6 h | ÷21600 → W m⁻², then ×(1−α) |
| 6 | **FLDS_J** | surface Downwelling Longwave (FLDS) | J m⁻² / 6 h | ÷21600 → W m⁻² |
| 7 | **PRECT** | total PRECipitaTion | m / 6 h | ×1000/21600 → kg m⁻² s⁻¹ |


1. **Units.** The network was trained on 6-hour accumulations (energy in J m⁻², water in m);
   POP wants rates (W m⁻², kg m⁻² s⁻¹), so divide by $21600$ s (and ×1000 for water density).
2. **Wind-stress sign.** Training data stored stress in the *atmosphere's* convention; POP
   wants stress *into the ocean* — the opposite sign, so negate $\tau_x,\tau_y$. (Getting this
   wrong silently kills the winter overturning circulation.)
3. **Ocean albedo.** The net outputs *downwelling* sunlight; POP wants *absorbed* sunlight, so
   multiply FSDS by $(1-\alpha)$ with $\alpha$ from a physics formula (Spencer-1971 declination + Briegleb direct beam), not a fitted number.

### Autoregression: the feedback loops
```
  LOOP 1 -- THE OCEAN  (every 6 h, real physics)
     POP SST --> UNet --> fluxes --> POP integrates --> new SST --> (repeat)

  LOOP 2 -- THE MEMORY  (inside the server)
     SST(t) --> ring buffer --> returns as SST_prev and dSST_dt on the next call

  LOOP 3 -- THE ICE  (inside the emulator)
     aice(t) --> aice_prev --> predicts aice(t+1) --> (repeat)
```
## The CICE emulator
A *prescribed* ice fraction cannot react to POP's own state, so in enclosed/polar seas the
ocean could warm-drift unchecked. A **live** emulated ice supplies the missing damping: as POP
warms, the emulator melts ice ($dh_i<0$), which (a) draws latent heat *out* of the ocean and
(b) shrinks `aice`, re-exposing the surface. That is a genuine negative feedback the one-way
prescribed-ice forcing lacks.

### Inputs and outputs (on the ocean grid, so no remap)

| Inputs (6) | Outputs (6) |
|---|---|
| **SST** — sea-surface temperature (live, from POP) | **aice** — ice area fraction (0–1) |
| **SOLIN** — insolation (prescribed) | **hi** — ice thickness (m) |
| **aice_prev, hi_prev, uvel_prev, vvel_prev** — its own previous ice state (autoregressive) | **uvel, vvel** — ice velocity (m s⁻¹) |
| | **strocnx, strocny** — ice→ocean stress (N m⁻²) |

### It updates monthly, then glides smoothly in between

Sea ice changes slowly, so the emulator only makes a **new prediction once per model month** —
not every 6 hours like the flux network. In the middle of each month it looks at POP's current
SST and its own previous ice state, and predicts **next** month's average ice. The very first
prediction is seeded from a real CESM2-LE ice snapshot (`ice_ic_1980-01.npz`).

Between those monthly predictions, the ice fields POP actually feels **slide smoothly (linearly)
from one month's value to the next**. This matters: without the glide, the ice could jump in a
single step (e.g. `aice` dropping 0.57 → 0.27 at once), which shocks the ocean. Interpolating
keeps the ice POP sees changing gently the whole time.

### Three ways the ice acts on the ocean

1. **It covers part of the ocean.** `aice` is the fraction of the cell under ice, so only the
   open part, $(1-\text{aice})$, feels the air-sea fluxes. The flux network reads the same
   `aice` as its `ICEFRAC` input, so both sides agree on how much ocean is exposed.
2. **It drags the water underneath.** The stress the ocean feels is a blend of the wind stress
   on open water and the ice-to-ocean stress under the pack:
   $\tau_{\text{ocn}} = (1-\text{aice})\,\tau + \text{aice}\,\tau_{\text{strocn}}$.
3. **Melting ice cools the ocean.** When ice melts, it pulls heat out of the water. We compute
   that heat from how fast the ice is thinning,
   $Q = \rho_{\text{ice}} L_f \, \dfrac{dh_i}{dt}$
   ($\rho_{\text{ice}}=917\ \mathrm{kg\,m^{-3}}$, $L_f=3.34\times10^{5}\ \mathrm{J\,kg^{-1}}$;
   $+$ means into the ocean), each step from the *applied* thickness change, capped at
   $\pm200$ W m⁻². Melting ($dh_i<0$) removes heat — this is the negative feedback that keeps
   polar seas from warm-drifting.

### Training the ice net

**Data.** Unlike the flux network, the ice net does *not* learn from the CREDIT atmospheric
data (that record is atmosphere-only and has no ice variables). It learns from the **CESM2
Large Ensemble monthly archive**, taken **natively on POP/CICE's own gx1v7 grid** (384×320, so
no remapping is needed). Two collections are read and lined up on their shared monthly calendar:

- **ice** history → `aice, hi, uvel, vvel, strocnx, strocny`
- **ocean** history → `SST`, plus `SOLIN`

We use member **LE2-1231.002** — the *same* Large-Ensemble run we score everything else
against — over **1850–2100**, which gives **≈3,010 monthly snapshots**.

**Task.** It is a simple **one-month-ahead map**: from this month's ocean state (SST, SOLIN)
plus *last* month's ice state, predict *this* month's ice.

$$\underbrace{\text{SST},\ \text{SOLIN},\ \text{aice}_{-1},\ \text{hi}_{-1},\ \text{uvel}_{-1},\ \text{vvel}_{-1}}_{\text{6 inputs}}
\;\longrightarrow\;
\underbrace{\text{aice},\ \text{hi},\ \text{uvel},\ \text{vvel},\ \text{strocnx},\ \text{strocny}}_{\text{6 outputs}}$$

The "last month" ($_{-1}$) inputs are made simply by pairing each month with the one before it.
That one-month memory is exactly what lets the model **run autoregressively** at deployment: it
feeds its own previous prediction back in as the next step's `*_prev` (Loop 3 above).

**Recipe.** Same 5-level U-Net as the flux model (the code reuses `UNet`, the normalizer, and
the masked loss from the flux trainer). The split is **temporal** (train on early years, test on
held-out later years — no leakage), the loss is **masked to ocean points**, every channel is
standardized, and because the gx1v7 grid wraps around in longitude we augment with a random
circular shift. No hand-tuning — the same honest-by-construction philosophy as the flux net.

**Why "wind-free."** An earlier version also fed the near-surface winds (`uatm, vatm`). The
deployed version drops them and uses `SOLIN` instead, on purpose: in the coupled loop the ice
net should depend only on things it actually has — POP's live SST and the prescribed sunlight —
so it stays self-contained and never needs a wind field it isn't given.

### Training the UNET

**Data.** "CREDIT" reanalysis-driven runs, 1980–2014, 6-hourly, already on the ocean grid;
each timestep gives the 11 inputs and the 8 true fluxes. We split *temporally* and hold out
2013–2014 for testing. Every channel is standardized $(x-\mu)/\sigma$ with training-only
statistics; the server reuses the exact same normalizer. The loss is evaluated only over
ocean/ice points and is latitude-weighted so shrinking polar cells don't dominate.

Stage 1 is plain masked, latitude-weighted mean-squared error,

$$\mathcal{L}_{\text{MSE}} \;=\; \sum_{\text{ocean}} w(\varphi)\,\big(\hat{y}-y\big)^2 .$$

This fits the fluxes beautifully in the dataset but drives the coupled ocean into **runaway warming**. Across the historical record, warm SST *coincides* with more downwelling longwave and shortwave (humid, clear-then-cloudy days). So an MSE-fit network learns a **positive** net feedback,

$$\frac{dQ_{\text{net}}}{dT_s} \approx +7\ \mathrm{W\,m^{-2}\,K^{-1}},$$

whereas physics demands a **damping** one, $\dfrac{dQ_{\text{net}}}{dT_s} \approx -10$ to
$-25\ \mathrm{W\,m^{-2}\,K^{-1}}$ (Planck radiation + evaporative cooling). Put a positive
feedback into Loop 1 and the ocean cooks itself.

We add a term that penalizes the *wrong feedback directly*. During training we run each sample
twice — at its true SST and at $T_s+\Delta T$ — and measure the network's own sensitivity,

$$d_{\text{turb}} = \frac{Q_{\text{turb}}(T_s{+}\Delta T)-Q_{\text{turb}}(T_s)}{\Delta T},
\qquad
d_{\text{rad}}  = \frac{Q_{\text{rad}}(T_s{+}\Delta T)-Q_{\text{rad}}(T_s)}{\Delta T},$$

then push the turbulent part toward the physically correct target $d^{\star}$ (computed by a
verbatim Python port of CESM's own bulk-flux formula) and push the radiative part toward zero:

$$\mathcal{L} \;=\; \mathcal{L}_{\text{MSE}}
\;+\; \lambda_{\text{turb}}\,\big(d_{\text{turb}}-d^{\star}\big)^2
\;+\; \lambda_{\text{rad}}\,d_{\text{rad}}^{\,2}.$$

The MSE term *encodes* the $+7$ correlation, so it fights the physics term head-on. We resolve
this with **PCGrad gradient surgery**: compute the two gradients separately and, only when they
conflict ($g_{\text{sens}}\!\cdot g_{\text{MSE}}<0$), remove the projection of the physics
gradient onto the MSE gradient — physics bends the fit only in its null-space, and is kept in
full where the two agree. **This step is what turns a runaway model into one whose 6-year
global-mean SST drift is a bounded $\pm 0.4$ K.**

### probabilistic

Every conv block carries a dropout layer. Leaving dropout **on at inference** and running $N$
forward passes gives $N$ slightly different flux fields — an ensemble (MC-Dropout). Delivering
one random member per step turns the run into stochastic physics; several seeds give an
ensemble of ocean trajectories whose spread quantifies flux-model uncertainty. To make that
spread *calibrated*, we can fine-tune on the unbiased CRPS score,

$$\text{CRPS} = \text{Skill} - \tfrac{1}{2}\,\text{Spread},\qquad
\text{Skill}=\overline{|m-y|},\quad
\text{Spread}=\overline{|m_i-m_j|},$$

which rewards an ensemble that is accurate **and** honestly uncertain.