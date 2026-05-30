# UNet Training Guide

## Model
Inputs (t): SST, ICEFRAC, SOLIN  
Targets (t+1): TAUX, TAUY, SHFLX, LHFLX, QFLX  
Architecture: 5-level UNet, circular longitude padding, skip connections  
Loss: masked MSE over ocean/ice points only, in normalised space

---

## One-time setup

### 1. Preprocess data (run once, ~35 min)
Loads all 35 zarr years and saves to a fast numpy cache:
```bash
qsub scripts/submit_preprocess.pbs
```
Cache saved to `/glade/work/praggarwal/couple_cache/` (25 GB).  
Then compute normalisation stats on the full dataset:
```bash
python compute_normalizer.py --cache_dir /glade/work/praggarwal/couple_cache
```
Subsequent training runs load from cache in seconds.

---

## Training

### Fresh training run
```bash
qsub scripts/submit_train_unet.pbs
```
Key defaults: `--subsample 1.0 --epochs 60 --base 64 --lr 1e-3 --batch 8`

**LR schedule:** CosineAnnealingLR from `lr` → 0 over `epochs`.

### Resume interrupted run
If a job was killed mid-training, resume from the last checkpoint:
```bash
python -u train_unet.py \
    --cache_dir /glade/work/praggarwal/couple_cache \
    --out_dir   output_unet \
    --epochs    60 \
    --resume
```
Loads `output_unet/checkpoint.pt` (model + optimizer + scheduler + history).  
The wandb run continues on the same run ID (`output_unet/wandb_run_id.txt`).

### Extend beyond original epochs (SGDR warm restart)
After a completed run, extend training using a cosine warm restart from a lower peak LR.
Starting from `lr * 0.1` avoids destabilising converged weights while still allowing
the optimiser to escape its current basin (SGDR-style).
```bash
python -u train_unet.py \
    --cache_dir    /glade/work/praggarwal/couple_cache \
    --out_dir      output_unet \
    --epochs       60 \
    --extra_epochs 30 \
    --resume \
    --wandb_project couple-unet
```
Epochs 1–60: cosine from `1e-3` → 0 (original run)  
Epochs 61–90: cosine restart from `1e-4` → 0 (extension)

---

## Outputs

| File | Contents |
|---|---|
| `output_unet/best_model.pt` | Model weights at lowest val loss |
| `output_unet/checkpoint.pt` | Full checkpoint (model + optimizer + scheduler + history) |
| `output_unet/normalizer.npz` | Per-channel mean/std |
| `output_unet/model_config.json` | Architecture config |
| `output_unet/r2_scores.json` | Per-variable R² on val set |
| `output_unet/training_summary.png` | Loss curve + R² bar chart |
| `output_unet/wandb_run_id.txt` | W&B run ID for resuming |

## Evaluation plots
```bash
qsub scripts/submit_plot_rmse.pbs        # per-pixel RMSE maps
qsub scripts/submit_plot_predictions.pbs # truth / pred / diff sample maps
```

---

## W&B metrics logged per epoch

| Metric | Description |
|---|---|
| `train/loss` | Overall masked MSE (normalised, all vars) |
| `val/loss` | Overall val masked MSE |
| `val/loss_TAUX` … `val/loss_QFLX` | Per-variable val MSE (normalised) |
| `val/r2_TAUX` … `val/r2_QFLX` | Per-variable R² (logged at end of training) |
| `lr` | Learning rate |
| `maps/val/{var}` | Truth / predicted / error maps (every 10 epochs) |
| `rmse_maps/val/{var}` | Per-pixel RMSE maps (end of training) |

---

## W&B run naming convention
Auto-generated from hyperparameters:
```
unet-sub{subsample}-ep{epochs}-base{base}-lr{lr}-{daily|6h}
```
Example: `unet-sub1.0-ep60-base64-lr1e-03-daily`

Override with `--wandb_name my-run-name`.

---

## Hyperparameter guide

| Arg | Effect |
|---|---|
| `--base 64` | Model size. 64=7.7M params. Double to 128 for ~4× more capacity. |
| `--subsample 1.0` | Fraction of daily pairs used. 1.0=12,740 samples (all years). |
| `--epochs 60` | More epochs help but watch for overfitting (train << val loss). |
| `--lr 1e-3` | Peak learning rate. Default works well; lower for fine-tuning. |
| `--sixhour` | Use raw 6-hourly pairs instead of daily means (4× more data). |
