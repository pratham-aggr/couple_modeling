"""build_gustiness_clim.py — production gustiness climatology for option 2.

g2(12,H,W) = per-month per-cell submonthly Var(Ubot)+Var(Vbot) of the TRUE winds,
pooled over ALL years in the cache (1980-2014). This is the second moment the
monthly-mean wind climatology discards; serving |U|_eff=sqrt(|U|^2+g2[month])
restores it (COARE gustiness, synoptic-generalized) -> recovers the rectified
latent cooling (diag_gustiness.py: 75% global / 89% SO of the dLH deficit).

Read-only reference derived from the same X_atm.npy the net trained on; on the
gx1v7 native grid, so it aligns with the served _atm in model_server.py.
"""
import numpy as np

CACHE = "/glade/work/praggarwal/couple_cache_gx1v7_mem24h"
OUT = "/glade/u/home/praggarwal/couple/output/gustiness_gx1v7.npy"
ROWS_PER_YEAR = 1456
MONTH_LEN = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
CUM = np.cumsum(MONTH_LEN)


def row_month_global(r):
    w = r % ROWS_PER_YEAR
    return int(np.searchsorted(CUM, (w + 4) // 4, side="right"))


def main():
    Xa = np.load(f"{CACHE}/X_atm.npy", mmap_mode="r")
    N, C, H, W = Xa.shape
    print(f"X_atm: {Xa.shape}  ({N} rows, {N/ROWS_PER_YEAR:.1f} years)")
    months = np.array([row_month_global(r) for r in range(N)])

    s1u = np.zeros((12, H, W)); s2u = np.zeros((12, H, W))
    s1v = np.zeros((12, H, W)); s2v = np.zeros((12, H, W))
    cnt = np.zeros(12, dtype=np.int64)

    CH = 400
    for a in range(0, N, CH):
        b = min(a + CH, N)
        u = Xa[a:b, 0].astype(np.float64)      # (chunk,H,W)
        v = Xa[a:b, 1].astype(np.float64)
        mo = months[a:b]
        for m in range(12):
            sel = (mo == m)
            if not sel.any():
                continue
            s1u[m] += u[sel].sum(0); s2u[m] += (u[sel]**2).sum(0)
            s1v[m] += v[sel].sum(0); s2v[m] += (v[sel]**2).sum(0)
            cnt[m] += int(sel.sum())
        print(f"  {b}/{N}", end="\r")
    print()

    g2 = np.zeros((12, H, W))
    for m in range(12):
        n = cnt[m]
        varu = s2u[m]/n - (s1u[m]/n)**2
        varv = s2v[m]/n - (s1v[m]/n)**2
        g2[m] = np.maximum(varu + varv, 0.0)
    print("per-month samples:", cnt.tolist())
    print(f"g (RMS submonthly wind std) global mean = {np.sqrt(g2.mean()):.2f} m/s  "
          f"max = {np.sqrt(g2.max()):.2f}  nan={np.isnan(g2).any()}")
    np.save(OUT, g2.astype(np.float32))
    print(f"wrote {OUT}  shape {g2.shape}")


if __name__ == "__main__":
    main()
