# Coupled Ocean–Atmosphere Emulator

A UNet that predicts atmosphere-to-ocean surface fluxes directly from ocean state, bypassing the CPL7 bulk formula. Trained on 35 years of CESM2 output at 6-hourly resolution.

## Task

| | |
|---|---|
| **Inputs** | SST, ICEFRAC, SOLIN at time *t* (+ optional CO2, + optional 24h-prior state) |
| **Outputs** | TAUX, TAUY, SHFLX, LHFLX, QFLX at time *t* |
| **Data** | 35 years CESM zarr, ~200k 6-hourly samples |
| **Architecture** | 5-level UNet, circular longitude padding, masked MSE loss |
| **Hardware** | A100 GPU (`nvgpu` queue), account `ucsd0044` |

## Repository Layout

```
train_unet.py           UNet training (all experiments)
preprocess_data.py      Build numpy cache from zarr (run once per lag config)
add_co2_to_cache.py     Append co2vmr scalar to cache as co2.npy
plot_lag_results.py     All validation plots (R² summary + spatial maps + summary figs)
env.yaml                Conda environment

scripts/
  submit_preprocess_lag{0,12,24}.pbs    Preprocess for lag experiments
  submit_preprocess_mem24h.pbs          Preprocess for memory experiment
  submit_lag{0,12,24}.pbs               Training — lag experiments (no CO2)
  submit_lag{0,12,24}_co2.pbs           Training — lag experiments (+ CO2)
  submit_mem24h.pbs                     Training — memory, no solin_prev, no CO2
  submit_mem24h_solin.pbs               Training — memory, + solin_prev
  submit_mem24h_co2.pbs                 Training — memory, + CO2
  submit_mem24h_solin_co2.pbs           Training — memory, + solin_prev + CO2
  submit_plot_lag_results.pbs           Spatial validation maps (per experiment)
  submit_plot_summary.pbs               Cross-experiment summary figures (Figs 1–5)
```

## Experiments

All runs: `subsample=1.0`, `base=64`, AdamW `lr=1e-3 → 1e-6` (SGDR), `patience=20`, `max_epochs=500`. Metric: R² on 10% val set, ocean/ice points only. SHFLX and LHFLX converted from J m⁻² per 6h step to W m⁻².

### Lag experiments (single-timestep input)

| Run | Input | TAUX | TAUY | SHFLX | LHFLX | QFLX |
|-----|-------|-----:|-----:|------:|------:|-----:|
| lag=0h | SST[t], ICEFRAC[t], SOLIN[t] | 0.751 | 0.670 | 0.865 | 0.902 | 0.895 |
| lag=12h | SST[t−12h], … | 0.741 | 0.657 | 0.862 | 0.900 | 0.893 |
| lag=24h | SST[t−24h], … | 0.723 | 0.632 | 0.858 | 0.895 | 0.890 |
| lag=0h+CO2 | + co2vmr | 0.733 | 0.646 | 0.860 | 0.896 | 0.890 |
| lag=12h+CO2 | + co2vmr | 0.736 | 0.652 | 0.861 | 0.899 | 0.892 |
| lag=24h+CO2 | + co2vmr | 0.726 | 0.637 | 0.858 | 0.896 | 0.890 |

### Memory experiments (current + 24h-prior input)

| Run | Input | TAUX | TAUY | SHFLX | LHFLX | QFLX |
|-----|-------|-----:|-----:|------:|------:|-----:|
| mem24h | SST[t]+SST[t−24h], ICEFRAC[t]+ICEFRAC[t−24h], SOLIN[t] | **0.826** | **0.772** | **0.881** | **0.934** | **0.930** |
| mem24h+solin | + SOLIN[t−24h] | 0.819 | 0.765 | 0.878 | 0.933 | 0.928 |
| mem24h+CO2 | + co2vmr | 0.826 | 0.772 | 0.881 | 0.934 | 0.930 |
| mem24h+solin+CO2 | + SOLIN[t−24h] + co2vmr | 0.826 | 0.772 | 0.881 | 0.934 | 0.930 |

**Key findings:**
- Memory (24h prior state) is the most impactful change — wind stress improves by +7.5/+10pp over lag=0h
- CO2 as an input channel has no measurable effect (≤0.002 R² difference across all configs)
- SOLIN from 24h ago is redundant — current SOLIN captures all solar forcing information
- Performance degrades monotonically as input lag increases (using only past state)

## Caches

| Cache | Location | Contents |
|-------|----------|----------|
| `couple_cache_lag0h` | `/glade/work/praggarwal/` | X=(SST,ICEFRAC,SOLIN) at t, Y=fluxes at t |
| `couple_cache_lag12h` | `/glade/work/praggarwal/` | X at t−12h, Y at t |
| `couple_cache_lag24h` | `/glade/work/praggarwal/` | X at t−24h, Y at t |
| `couple_cache_mem24h` | `/glade/work/praggarwal/` | X=(6-ch: current+24h-prior), Y at t |

Each cache: `X.npy`, `Y.npy`, `mask.npy`, `co2.npy`, `meta.json`, `normalizer.npz`

## Quick Start

```bash
# Activate environment
module load conda
conda activate /glade/work/praggarwal/conda-envs/atm

# 1. Preprocess (once per cache, ~35 min on CPU)
qsub scripts/submit_preprocess_lag0.pbs

# 2. Add CO2 to cache (if running CO2 experiments)
python add_co2_to_cache.py --cache_dir /glade/work/praggarwal/couple_cache_lag0h

# 3. Train
qsub scripts/submit_lag0.pbs

# 4. Plot validation maps per experiment
qsub scripts/submit_plot_lag_results.pbs

# 5. Generate cross-experiment summary figures (Figs 1–5)
qsub scripts/submit_plot_summary.pbs
```

## Output Directories

Each `output_unet_<name>/` contains:

| File | Contents |
|------|----------|
| `best_model.pt` | Weights at lowest val loss |
| `checkpoint.pt` | Full checkpoint (model + optimizer + scheduler) |
| `normalizer.npz` | Per-channel mean/std |
| `model_config.json` | Architecture + input/output var names |
| `r2_scores.json` | Per-variable R² on val set |
| `training_summary.png` | Loss curve + R² bar chart |

## Plots

| Script flag | Output | Description |
|-------------|--------|-------------|
| `plot_lag_results.py` | `results/r2_summary.png` | R² bar chart across all experiments |
| `plot_lag_results.py --labels <exp>` | `results/val_maps_<exp>.png` | Truth/pred/bias maps per experiment |
| `plot_lag_results.py --summary` | `results/fig1_truth.png` … `fig5_best_overall_error.png` | 5 cross-experiment summary figures |

## Environment

```bash
module load conda
conda activate /glade/work/praggarwal/conda-envs/atm
```
