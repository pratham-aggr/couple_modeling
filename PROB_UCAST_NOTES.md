# Probabilistic MEMO (U-Cast recipe) — what was added & how to read it

_Added 2026-06-17. Companion to `papers/ucast.pdf`. All code changes are additive and
behind default-off flags — existing deterministic runs are byte-for-byte unchanged._

## Why
MEMO's deterministic (MSE) fluxes are over-smoothed conditional means. The U-Cast recipe
(MAE pre-train → short CRPS fine-tune with MC-Dropout) yields *calibrated probabilistic*
fluxes, most useful for the noisy variables (production radprecip **PRECT R²=0.41**).

A prior run "wasn't better than MSE" because it was judged on **deterministic R²** (which
MSE maximizes by construction). A CRPS model can only win on **CRPS** and **spread-skill
ratio (SSR)** — which we now measure.

## New in `train_unet.py` (all gated, default off)
| Flag | Effect |
|---|---|
| `--ema` / `--ema_decay` (0.9999) | EMA of weights (`ModelEMA`); eval + `best_model.pt` use EMA (raw kept as `best_model_raw.pt`); EMA stored in `checkpoint.pt`. |
| `--eval_prob` / `--eval_members` (8) | Compute CRPS, SSR, ensemble-mean R² via **BN-safe** MC-Dropout (`compute_crps_ssr`). Writes `crps_ssr_test.json`; logs val CRPS/SSR each epoch. **Selection metric unchanged.** |
| `--resume_weights_only` | Load only weights (+reset `best_val`) so Stage 2 can switch optimizer (Muon) without a param-group clash. |

Key helper `enable_mc_dropout(model)`: puts the model in eval mode but re-enables **only**
the dropout layers — BatchNorm stays in eval. (A plain `model.train()` would switch BN to
batch stats and corrupt predictions — the classic MC-Dropout gotcha.)

## The experiment (8-output radprecip, temporal split)
- out_dir: `output/output_unet_mem24h_dsst_temporal_radprecip_ucast` (new; baseline
  `output/output_unet_mem24h_dsst_temporal_radprecip` untouched).
- **Stage 1** `scripts/submit_mem24h_ucast_s1.pbs` — MAE pre-train + EMA (job 4679308).
- **Stage 2** `scripts/submit_mem24h_ucast_s2.pbs` — CRPS fine-tune + Muon + EMA +
  `--eval_prob`, `--resume_weights_only` from Stage 1 (job 4679309, after S1).
- **Compare** `scripts/eval_ucast_prob.pbs` — runs `--eval_test --eval_prob` on BOTH the
  U-Cast and baseline models → `crps_ssr_test.json` each → prints a side-by-side table
  (job 4679310, after S2).

Chain: `qsub s1` → `qsub -W depend=afterok:<s1> s2` → `qsub -W depend=afterok:<s2> eval`.

## How to read the result (`crps_ssr_test.json` / the eval table)
- **CRPS** (lower = better): the proper score for a probabilistic forecast. Expect the
  U-Cast model **lower** than the MSE baseline, biggest gain on **PRECT**.
- **SSR** = mean ensemble spread / ensemble-mean RMSE; **1.0 = well-calibrated**, <1 =
  under-dispersive (over-confident). The MSE baseline will be far below 1; U-Cast should
  be much closer to 1.
- **R² of the ensemble mean**: for reference — **not** expected to beat the MSE model.
  Judging probabilistic models on R² is the original mistake.

## Manual fallback (if a PBS dependency doesn't auto-fire)
```
# after Stage 1 finished, run Stage 2 by hand:
qsub scripts/submit_mem24h_ucast_s2.pbs
# after Stage 2, run the comparison:
qsub scripts/eval_ucast_prob.pbs
# or score any one model dir directly (CPU/GPU):
python train_unet.py --out_dir <DIR> --zarr_glob "<glob>" --memory --dsst_dt \
    --with_rad --with_precip --split_mode temporal --base 64 --dropout 0.1 \
    --eval_test --eval_prob --eval_members 8
```
