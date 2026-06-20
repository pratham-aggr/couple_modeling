A U-Net that predicts ocean→atmosphere surface fluxes (**TAUX, TAUY, SHFLX, LHFLX, QFLX**)
directly from ocean state, replacing the CPL7 bulk formula. Trained on CESM2 output.

All numbers below are **held-out test R²** on the publication-grade **temporal split**:
train **1980–2010**, validate **2011–2012**, test **2013–2014** (training-only normalizer).
These are honest out-of-sample scores — the test years are never seen during training.

## Results (test R², 2013–2014)

| Model | TAUX | TAUY | SHFLX | LHFLX | QFLX |
|-------|-----:|-----:|------:|------:|-----:|
| Memory-free (lag 0h) | 0.377 | 0.183 | 0.720 | 0.679 | 0.702 |
| + 24h memory | 0.673 | 0.571 | 0.841 | 0.898 | 0.909 |
| + dSST/dt | 0.693 | 0.592 | 0.841 | 0.900 | 0.910 |
| + dropout (`drop`) | 0.716 | 0.622 | **0.848** | **0.905** | **0.914** |
| + augmentation (`aug`) | 0.698 | 0.596 | 0.843 | 0.905 | 0.914 |
| + CO2 (`co2`) | 0.690 | 0.589 | 0.842 | 0.901 | 0.911 |
| **dropout + aug (`combo`)** | **0.718** | **0.624** | 0.841 | 0.903 | 0.912 |

Bold = best per column.

## What each lever changes

Every experiment is the same `mem24h + dSST/dt` baseline with **one** change, to isolate
what actually helps generalization on the held-out years.

| Tag | Change | Effect |
|-----|--------|--------|
| `drop` | `--dropout 0.1` (spatial dropout in every U-Net block) | **biggest regularization win**: +0.023 TAUX, +0.030 TAUY |
| `aug`  | `--augment` (random circular shift along longitude, train set only — exact because the grid is lon-periodic) | small: +0.005 / +0.004 |
| `co2`  | `--with_co2` (atmospheric CO2 input channel) | slightly negative (−0.003 / −0.003) — no benefit once SST is known |
| `wd`   | `--weight_decay 1e-3` (10× default) | minor |
| `small`| `--base 32` (half width, fewer params) | minor |
| `combo`| `--dropout 0.1 --augment` (base 64) | best wind-stress model |

## Key findings

- **24h of ocean-state memory is the dominant design choice.** It lifts wind stress from
  R² = 0.377→0.673 (TAUX) and 0.183→0.571 (TAUY), and heat/moisture fluxes to high accuracy
  (LHFLX 0.898, QFLX 0.909). Memory closes ~87–88% of the gap to the best model.
- **Wind stress is the hardest target** because near-surface wind is not an input.
- **Dropout — not augmentation — recovers the remaining wind-stress skill.** Augmentation
  adds almost nothing on top of dropout (+0.002 / +0.001 in `combo`).
- **CO2 provides negligible benefit** once SST is known, even though the test years lie above
  the training CO2 range.

