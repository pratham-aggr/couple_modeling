"""Measure batch-size scaling and torch.compile for HEALPixUNet (GPU-only)."""
import os, sys, time, json
sys.path.insert(0, os.getcwd())
import numpy as np, torch
from pathlib import Path
import train_hpx as T

cache = Path("/glade/work/praggarwal/couple_cache_hpx64")
meta = json.load(open(cache / "meta.json"))
nside, npix = meta["nside"], meta["npix"]
n_in, n_out = 6, 8
dev = "cuda"
print("torch", torch.__version__, torch.cuda.get_device_name(0), flush=True)

def bench(model, B, NS=8, tag=""):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(B, n_in, npix, device=dev)
    y = torch.randn(B, n_out, npix, device=dev)
    m = torch.ones(B, npix, device=dev)
    for _ in range(3):
        opt.zero_grad(); loss = T.masked_mse(model(x), y, m); loss.backward(); opt.step()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(NS):
        opt.zero_grad(); loss = T.masked_mse(model(x), y, m); loss.backward(); opt.step()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / NS * 1000
    steps = int(np.ceil(45136 / B))
    print(f"[{tag:14s}] B={B:3d}  {ms:7.1f} ms/step  {ms/B:6.2f} ms/sample  "
          f"-> {ms*steps/1000/60:5.1f} min/epoch ({steps} steps)", flush=True)
    del x, y, m, opt; torch.cuda.empty_cache()

model = T.HEALPixUNet(nside=nside, n_in=n_in, n_out=n_out, base=64, dropout=0.0).to(dev)
for B in [8, 16, 32, 64]:
    try:
        bench(model, B, tag="fp32")
    except RuntimeError as e:
        print(f"B={B} OOM/err: {str(e)[:80]}", flush=True); torch.cuda.empty_cache()

print("--- torch.compile ---", flush=True)
try:
    cmodel = torch.compile(model)
    for B in [16, 32]:
        bench(cmodel, B, tag="compile")
except Exception as e:
    print("compile failed:", str(e)[:200], flush=True)
