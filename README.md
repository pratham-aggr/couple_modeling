# Coupled Ocean–Atmosphere Emulator

Trains a UNet to replace the atmosphere in CESM coupled simulations by predicting surface fluxes directly from ocean state, bypassing the CPL7 bulk formula.

## Task

| | |
|---|---|
| **Inputs** | SST, ICEFRAC, SOLIN (daily mean, t) |
| **Outputs** | TAUX, TAUY, SHFLX, LHFLX, QFLX (daily mean, t+1) |
| **Data** | 35 years CESM zarr, 12,740 daily samples |
| **Cache** | `/glade/work/praggarwal/couple_cache/` (25.4 GB) |

## Repository Layout

```
train_unet.py           Main training script (UNet emulator)
train_fake_atm.py       MLP fake atmosphere (radiation + TS + U10)
preprocess_data.py      Build numpy cache from zarr (run once, ~35 min)
compute_normalizer.py   Compute channel-wise mean/std for cache
add_co2_to_cache.py     Append co2vmr to cache as co2.npy
plot_predictions.py     Truth/pred/diff maps for 4 random days
plot_rmse_maps.py       Per-pixel RMSE maps over validation set
explore_fake_atm.py     Exploratory analysis for fake atm MLP
fake_atm_server.py      File-based MLP inference server for CESM coupling
test_fake_atm_server.py Standalone driver to test the inference server

scripts/
  submit_preprocess.pbs
  submit_train_unet.pbs     60 ep + 30 ep extension
  submit_train_co2.pbs      60 ep with CO2 channel
  submit_train_es.pbs       Early stopping, max 500 ep  ← active
  submit_plot_rmse.pbs
  submit_plot_predictions.pbs
  submit_fake_atm_server.pbs
  setup_fake_atm_case.sh
```

## Quick Start

```bash
# 1. One-time: build cache (submit to CPU queue, ~35 min)
qsub scripts/submit_preprocess.pbs

# 2. Train (A100, ~90 min for 60 epochs)
qsub scripts/submit_train_unet.pbs

# 3. Resume / extend
#    Edit submit_train_unet.pbs: add --resume --extra_epochs 30
qsub scripts/submit_train_unet.pbs

# 4. Early-stopping run (up to 500 epochs, patience=20)
qsub scripts/submit_train_es.pbs
```

## Experiments

All runs use `subsample=1.0`, `base=64`, `lr=1e-3`, daily means. Metric: R² on 10% val set, ocean/ice points only.

| Run | Epochs | Notes | TAUX | TAUY | SHFLX | LHFLX | QFLX |
|-----|--------|-------|-----:|-----:|------:|------:|-----:|
| Exp 1 | 60 | cosine LR | 0.560 | 0.373 | 0.714 | 0.782 | 0.768 |
| Exp 2 | 90 | +30 SGDR extension **[best]** | 0.584 | 0.385 | 0.724 | 0.796 | 0.782 |
| Exp 3 | 60 | +CO2 channel | 0.557 | 0.371 | 0.715 | 0.782 | 0.768 |
| Exp 4 | ES | patience=20, max 500 ep | — | — | — | — | — |

**CO2 result**: broadcasting co2vmr as a 4th input channel had no measurable effect. The model ignores it — day-to-day flux variability is dominated by SST/SOLIN, not the slow CO2 trend.

W&B project: `climate-analytics-lab/couple-unet`

## Output Directories

| Directory | Contents |
|-----------|----------|
| `output_unet/` | Best model (Exp 2, 90 ep) |
| `output_unet_co2/` | CO2 experiment (Exp 3) |
| `output_unet_es/` | Early stopping run (Exp 4, in progress) |
| `output_full/` | Fake atmosphere MLP (different task) |

Each output directory contains: `best_model.pt`, `checkpoint.pt`, `normalizer.npz`, `model_config.json`, `r2_scores.json`, `training_summary.png`.

## Fake Atmosphere MLP (separate model)

Predicts radiation fluxes + TS + U10 from ocean state for use as a CESM atmosphere replacement via the CAMulator file-based coupling protocol.

| Variable | R² | Variable | R² |
|----------|----|----------|----|
| FSDS_J | 0.893 | FLUS | 0.982 |
| FLDS_J | 0.912 | FSUTOA | 0.812 |
| FSUS | 0.954 | FLUT | 0.705 |
| TS | 0.979 | U10 | 0.493 |
| PRECT | 0.146 | | |

## Environment

```bash
module load conda
conda activate /glade/work/praggarwal/conda-envs/atm
```

Queue: `nvgpu` (A100), account: `ucsd0044`. Only 1 concurrent GPU job allowed under this allocation.
