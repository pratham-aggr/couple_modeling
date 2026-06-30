"""
diag_hpx_speed.py — separate GPU-compute cost from I/O cost for train_hpx.
Prints: torch/cuda info, ms/step for a synthetic GPU-only forward+backward,
and ms/step for real DataLoader steps (memmap I/O included).
"""
import os, sys, time, json
sys.path.insert(0, os.getcwd())  # project root (train_hpx.py lives here)
import numpy as np
import torch
from pathlib import Path
import train_hpx as T

cache = Path("/glade/work/praggarwal/couple_cache_hpx64")
out   = Path("output/output_hpx64_mem24h")

print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

meta = json.load(open(cache / "meta.json"))
nside, npix = meta["nside"], meta["npix"]
n_in, n_out, B = 6, 8, 8

# ---- 1) pure-GPU forward+backward (no I/O) ----
model = T.HEALPixUNet(nside=nside, n_in=n_in, n_out=n_out, base=64, dropout=0.0).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(B, n_in, npix, device=device)
y = torch.randn(B, n_out, npix, device=device)
m = torch.ones(B, npix, device=device)

for i in range(2):  # warmup
    opt.zero_grad(); loss = T.masked_mse(model(x), y, m); loss.backward(); opt.step()
if device == "cuda": torch.cuda.synchronize()
t0 = time.time(); NS = 10
for i in range(NS):
    opt.zero_grad(); loss = T.masked_mse(model(x), y, m); loss.backward(); opt.step()
if device == "cuda": torch.cuda.synchronize()
ms = (time.time() - t0) / NS * 1000
print(f"\n[GPU-only fp32]  {ms:.1f} ms/step  ->  {ms*5642/1000/60:.1f} min/epoch (5642 steps)", flush=True)

# ---- 1b) GPU-only with AMP (bf16 autocast) ----
if device == "cuda":
    scaler = None
    for i in range(2):  # warmup amp
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = T.masked_mse(model(x), y, m)
        loss.backward(); opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(NS):
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = T.masked_mse(model(x), y, m)
        loss.backward(); opt.step()
    torch.cuda.synchronize()
    ms_amp = (time.time() - t0) / NS * 1000
    print(f"[GPU-only bf16]  {ms_amp:.1f} ms/step  ->  {ms_amp*5642/1000/60:.1f} min/epoch  "
          f"(speedup {ms/ms_amp:.2f}x)", flush=True)

# ---- 2) real DataLoader steps (memmap I/O included, cold cache) ----
X = np.load(cache / "X.npy", mmap_mode="r")
Y = np.load(cache / "Y.npy", mmap_mode="r")
Yrad = np.load(cache / "Y_rad.npy", mmap_mode="r")
Yprecip = np.load(cache / "Y_precip.npy", mmap_mode="r")
mask = np.load(cache / "mask.npy", mmap_mode="r")
norm = T.Normalizer.load(out / "normalizer.npz")
idxs = np.random.RandomState(0).choice(45000, 8 * 30, replace=False)  # random -> cold reads
ds = T.HPXDataset(idxs, X, Y, Yrad, Yprecip, mask, norm)
from torch.utils.data import DataLoader
dl = DataLoader(ds, batch_size=B, shuffle=False, num_workers=8, pin_memory=True)

t0 = time.time(); nb = 0; tcompute = 0.0
for xb, yb, mb in dl:
    tc = time.time()
    xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
    opt.zero_grad(); loss = T.masked_mse(model(xb), yb, mb); loss.backward(); opt.step()
    if device == "cuda": torch.cuda.synchronize()
    tcompute += time.time() - tc
    nb += 1
total = time.time() - t0
print(f"[real I/O]  {total/nb*1000:.1f} ms/step total  ({tcompute/nb*1000:.1f} compute, "
      f"{(total-tcompute)/nb*1000:.1f} I/O+collate)  over {nb} steps", flush=True)
print(f"[real I/O]  -> {total/nb*5642/60:.1f} min/epoch cold (random-access reads)", flush=True)
