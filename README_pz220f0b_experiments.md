# pz220f0b coupled experiments

1) **Model trained on CREDIT dataset**  nonautoregressive unfinetuned flux+wind unet.

Inputs (6): `SST, ICEFRAC, SOLIN, SST_prev, ICEFRAC_prev, dSST_dt`
Outputs (13): `TAUX, TAUY, SHFLX, LHFLX, QFLX, FSDS_J, FLDS_J, PRECT` + winds `Ubot, Vbot, Tbot, Qbot, PS`


2) **POP2** (gx1v7, 384×320, 128 ranks) with an autoregressive ML CICE sea-ice emulator.
No prescribed atmosphere, no clamps/anchors.

3) **Autoregressive CICE ML Model trained on CESM2 LE2 dataset**

Inputs (6): `SST`, `SOLIN`, `aice_prev`, `hi_prev`, `uvel_prev`, `vvel_prev`
Outputs (6): `aice`, `hi`, `uvel`, `vvel`, `strocnx`, `strocny`

outputs are used to run POP2

- **Two trained models, two training distributions.** The flux+wind net (pz220f0b) is
  trained on **CREDIT** gx1v7 data. The **ML CICE emulator** is trained on **CESM2-LE
  (LE2-1231.002)** ice history, the *only* source carrying the full prognostic ice state
  (`aice, hi, uvel, vvel, strocnx, strocny`); CREDIT has none of these.

- **At run time, ICEFRAC = the CICE emulator's own live `aice`** (`--ice_input_from_emu`):
  the autoregressive emulator predicts `aice` from the ocean state + its previous ice,
  and that same `aice` feeds the flux net's `ICEFRAC`/`ICEFRAC_prev` inputs *and* POP
  (open-water flux weighting, ice→ocean stress, melt-heat feedback). 
- **The distribution mismatch.** The flux net trained on *CREDIT* `ICEFRAC` but is served
  *LE2*-flavored `aice` at run time — a train/serve gap. 


**1. 6yr Stability Test (1980-1981)**

| | 1980 | 1981 | 1982 | 1983 | 1984 | 1985 | drift |
|---|---|---|---|---|---|---|---|
| **pz220f0b** | 19.10 | 19.48 | 19.67 | 19.73 | 19.66 | 19.63 | **+0.52 K** |
| **LE2 truth** | 18.65 | 18.65 | 18.65 | 18.55 | 18.53 | 18.52 | −0.13 K (flat) |

Model drift concentrates in yr 1–2 then **plateaus** (1982–85 within ~0.1 K) — bounded,
not runaway. Against the near-flat LE2 truth it carries a **warm bias ~+0.45 K (yr 1)
growing to ~+1.1 K (yr 6)** with a larger seasonal amplitude.

![emu-aice control global-mean SST vs LE2 truth, 6 yr](output/pz220f0b_emuaice_drift.png)

*Grey = the climatology-ICEFRAC control (drift +0.52 K, essentially identical to emu-aice
in the global mean); blue = emu-aice baseline; **orange = the finetuned `ke9j7j75` (sensreg
dQ/dSST) run** (50-yr chain, 4 yr shown so far); green dashed = LE2 truth. The finetuned
curve drifts **+0.16 K over its first 4 yr — ~3× less than the base +0.52 K** and turns
over in yr 4 rather than accumulating, the dQ/dSST fix bounding the warm drift.*

**ENSO (Niño-3.4, 5°S–5°N, 170°W–120°W).** The closed diagnosed-wind loop generates
its own interannual variability — a warm-1980 → cool-1984 swing, phase moderately
tracking the LE2 truth after the yr-1 cold-start spin-up. Amplitude scoring vs truth:

| | mean Niño-3.4 SST | anomaly std |
|---|---|---|
| **pz220f0b emu-aice** (base) | 22.74 °C (−5.1 vs truth) | **1.57 K** |
| climatology ICEFRAC | 22.95 °C (−4.9) | 1.45 K |
| **finetuned `ke9j7j75`** (sensreg, 4 yr) | — | **1.12 K** |
| **LE2-1231.002 truth** | 27.88 °C | 0.84 K |

*(mean = raw box-mean SST over the record — the deseasonalized-anomaly mean is ≈0 by
construction, so the mean-state SST is the meaningful "mean.")*

Two separate biases: **(1) amplitude** — variability is **~1.9× over-amplified** vs LE2
truth (underdamped equatorial feedbacks; phase faithful, amplitude the open issue; model
range ≈ −2.4…+4.9 K, the +4.9 a yr-1 spin-up transient); **(2) mean state** — the cold
tongue runs **~5 °C too cold** (22.7 °C vs 27.9 °C), a regional cold bias in the
equatorial Pacific even though the *global* mean drifts warm. Series: `scratchpad/emuaice_nino34.npz`.

![emu-aice Niño-3.4 ENSO index, 6 yr](output/pz220f0b_emuaice_nino34.png)

*Grey = climatology-ICEFRAC control (std 1.45 K); blue = emu-aice base (std 1.57 K);
**orange = finetuned `ke9j7j75` (sensreg, std 1.12 K)**; green dashed = LE2 truth (std 0.84 K).
The emu-aice and climatology configs track closely — ENSO is tropical, so the (polar)
ICEFRAC choice barely affects it. The finetuned run's amplitude (1.12 K) sits between the
base (1.57 K) and truth (0.84 K) — the dQ/dSST damping pulls the over-amplified ENSO
partway back toward observed, on top of bounding the global drift.*

**2. Flux-decrease — climate forcing (−2 W/m² downwelling LW, 1980, 1 yr)**
Rundir `…_diagfw_iceemu_m2_1yr` · job 7145482 · `--flds_forcing_wm2 -2.0`; else identical
to the emu-aice control. 12 mo, no blowup.

**Physical, stable cooling** vs the emu-aice control (monotonic, still approaching equilibrium):

| Jan | Apr | Jul | Oct | Dec |
|---|---|---|---|---|
| −0.02 | −0.08 | −0.18 | −0.24 | **−0.21 K** |

A sustained −2 W/m² surface radiative forcing drives a smooth cooling reaching ~−0.21 K
by year-end (peak −0.25 K in Nov). Matches the climatology-config result (−0.19 K),
confirming the forced response is robust to the ICEFRAC choice.

**3. Temp-increase — (+2 K IC kick, 1980–1984, 5 yr)**

**The coupler damps the kick — anomaly decays, no runaway** (PERT − control, annual mean):

| 1980 | 1981 | 1982 | 1983 | 1984 |
|---|---|---|---|---|
| +1.92 | +1.65 | +1.48 | +1.18 | +0.97 K |

+2.0 K → +0.97 K over 5 yr — decays **faster** than the climatology-config run (which
reached +1.29 K), i.e. the emu-aice feedback restores stability more effectively.

QUESTIONS:

**1) should we finetune?**
$$\mathcal{L} = \underbrace{\text{MSE}(\hat F, F)}_{\text{match the fluxes}} + \;\lambda_s\,\big\lVert \partial_\text{SST}\hat Q_\text{turb} - \partial_\text{SST} Q_\text{bulk} \big\rVert^2.$$

$$\partial_\text{SST} Q_\text{bulk} = \frac{Q_\text{bulk}(\text{SST}+\delta) - Q_\text{bulk}(\text{SST}-\delta)}{2\delta}.$$
 
Symbols: $Q$ = net surface heat flux; $F,\hat F$ = true and predicted fluxes;
$Q_\text{turb}$ = sensible + latent heat; penalty weights $\lambda_s$ = **0.2**
(heat) and **0.1** (radiation); step $\delta = 0.5$ K. Result: coupled drift drops
to **+0.002 °C/yr** with real winds. Then **freeze** the network.


This looks complicated but essentially is some sort of ridge regression: 

$$\mathcal{L}_{\mathrm{ridge}}
=
\frac{1}{n}\|\mathbf{y}-\mathbf{X}\boldsymbol{\beta}\|_2^2
+
\lambda\|\boldsymbol{\beta}\|_2^2$$ 

But not exactly maybe worth a try though we don't observe any overfitting




**2) The ICEFRAC confusion:**
  - keep as is, for unet training use ICEFRAC from CREDIT & when coupling use the emulated ICEFRAC.
  - for unet training use ICEFRAC from CREDIT & when coupling use climatological ICEFRAC.
  - for unet training use emulated ICEFRAC & when coupling use the same emulated ICEFRAC. 

- either way we won't be able to eliminate 
CESM2 LE2 data, as hi, uvel, vvel, strocnx, strocny are not present in CREDIT runs
