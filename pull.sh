#!/usr/bin/env bash
#
# pull.sh — rsync the MEMO "orography" training data from NCAR Casper to a local
# machine, so train_unet.py can run locally with --memory.
#
# Run this ON YOUR LOCAL MACHINE (it pulls FROM Casper).
#
# Usage:
#   ./pull.sh                       # uses defaults below
#   USER=jdoe DEST=~/memo_data ./pull.sh
#
set -euo pipefail

# ---- config (override via env vars) ----------------------------------------
USER="${USER:-praggarwal}"                       # your NCAR username
HOST="${HOST:-data-access.ucar.edu}"             # dedicated transfer node
DEST="${DEST:-$HOME/memo_data}"                  # local destination dir
# ----------------------------------------------------------------------------

CACHE_SRC="/glade/work/praggarwal/couple_cache_mem24h"
STATICS_SRC="/glade/derecho/scratch/wchapman/b_credit_runs/statics_b_credit_runs_f32_02.nc"
NORM_SRC="/glade/u/home/praggarwal/couple/output/output_unet_mem24h_dsst_temporal_radprecip_oro/normalizer.npz"

REMOTE="${USER}@${HOST}"
RSYNC="rsync -avP"   # archive, verbose, progress + resumable

echo "==> Destination: $DEST"
echo "==> Remote:      $REMOTE"
mkdir -p "$DEST/couple_cache_mem24h"

# 1) The cache (~159 GB) — X, Y, rad, precip, mask, meta. NOT Y_atm.npy.
# Unquoted brace expands to multiple source args -> ONE rsync/ssh (single 2FA).
echo "==> [1/3] Pulling cache (~159 GB)..."
$RSYNC \
  $REMOTE:$CACHE_SRC/X.npy \
  $REMOTE:$CACHE_SRC/Y.npy \
  $REMOTE:$CACHE_SRC/Y_rad.npy \
  $REMOTE:$CACHE_SRC/Y_precip.npy \
  $REMOTE:$CACHE_SRC/mask.npy \
  $REMOTE:$CACHE_SRC/meta.json \
  "$DEST/couple_cache_mem24h/"

# 2) Orography statics (~50 MB)
echo "==> [2/3] Pulling orography statics..."
$RSYNC "$REMOTE:$STATICS_SRC" "$DEST/"

# 3) Normalizer (~30 MB)
echo "==> [3/3] Pulling normalizer..."
$RSYNC "$REMOTE:$NORM_SRC" "$DEST/"

echo
echo "==> Done. Local layout:"
echo "    $DEST/couple_cache_mem24h/{X,Y,Y_rad,Y_precip,mask}.npy + meta.json"
echo "    $DEST/statics_b_credit_runs_f32_02.nc"
echo "    $DEST/normalizer.npz   (copy into your --out_dir before training)"
