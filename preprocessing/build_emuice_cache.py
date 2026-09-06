"""build_emuice_cache.py — new flux-net training cache identical to
couple_cache_gx1v7_mem24h EXCEPT the ICEFRAC input channels (ch1 = ICEFRAC now,
ch4 = ICEFRAC_prev) are replaced by the CICE emulator's aice (from
gen_emulated_icefrac.py). All other arrays are symlinked (no copy). The normalizer
is NOT symlinked, so train_unet.py recomputes it from the emu-swapped X (temporal
split) -> the emu-aice channel is normalized by its OWN stats (no confound).

Sample->emu-timestep map (35 per-year blocks of 1456; each year drops its first 4
ramp steps):  g_now(i) = (i//1456)*1460 + (i%1456) + 4 ;  g_prev = g_now - 4.
"""
import os, json, numpy as np, shutil

SRC   = "/glade/work/praggarwal/couple_cache_gx1v7_mem24h"
DST   = "/glade/derecho/scratch/praggarwal/couple_cache_gx1v7_emuice"
EMU   = "/glade/derecho/scratch/praggarwal/emu_aice_global.npy"
SPY, STEPS, LAG = 1456, 1460, 4   # samples/yr, steps/yr, memory lag

def main():
    os.makedirs(DST, exist_ok=True)
    Xsrc = np.load(f"{SRC}/X.npy", mmap_mode="r")            # (N,6,H,W)
    emu  = np.load(EMU, mmap_mode="r")                        # (T,H,W) float16
    N, C, H, W = Xsrc.shape
    print(f"X {Xsrc.shape}  emu {emu.shape}  N/SPY={N/SPY}")
    assert C == 6 and N % SPY == 0, f"unexpected N={N}"
    nyr = N // SPY
    assert emu.shape[0] >= nyr * STEPS, "emu timeline too short"

    # sample -> global emu timestep
    i = np.arange(N)
    g_now  = (i // SPY) * STEPS + (i % SPY) + LAG
    g_prev = g_now - LAG
    assert g_now.max() < emu.shape[0]

    # sanity: real ICEFRAC(ch1) and emu at g_now should be spatially correlated
    for s in (0, N//2, N-1):
        a = np.asarray(Xsrc[s, 1]).ravel(); b = emu[g_now[s]].astype(np.float32).ravel()
        m = np.isfinite(a) & np.isfinite(b)
        r = np.corrcoef(a[m], b[m])[0, 1]
        print(f"  sample {s}: corr(realICEFRAC, emu@g_now)={r:.3f}  "
              f"real.mean={a[m].mean():.4f} emu.mean={b[m].mean():.4f}")

    # write new X.npy (chunked): copy all channels, overwrite 1 & 4 with emu
    Xdst = np.lib.format.open_memmap(f"{DST}/X.npy", mode="w+",
                                     dtype=Xsrc.dtype, shape=Xsrc.shape)
    B = 512
    for s in range(0, N, B):
        e = min(s + B, N)
        blk = np.asarray(Xsrc[s:e]).copy()
        blk[:, 1] = emu[g_now[s:e]].astype(blk.dtype)
        blk[:, 4] = emu[g_prev[s:e]].astype(blk.dtype)
        Xdst[s:e] = blk
        if s % (B * 20) == 0:
            print(f"  wrote {e}/{N}", flush=True)
    Xdst.flush(); del Xdst
    print("X.npy written.")

    # symlink everything else (big arrays untouched); DO NOT link normalizer.npz
    for f in os.listdir(SRC):
        if f in ("X.npy", "normalizer.npz"):
            continue
        src = os.path.join(SRC, f); dst = os.path.join(DST, f)
        if os.path.exists(dst):
            continue
        if f == "meta.json":
            meta = json.load(open(src)); meta["icefrac_source"] = "CICE emulator aice (emuice experiment)"
            json.dump(meta, open(dst, "w"), indent=2)
        else:
            os.symlink(os.path.realpath(src), dst)
    print(f"done. new cache at {DST}")

if __name__ == "__main__":
    main()
