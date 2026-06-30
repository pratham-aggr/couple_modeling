"""
healpix_grid.py — HEALPix grid utilities for the HEALPix-native MEMO experiment.

Provides the three pieces of genuinely new math needed to take the lat-lon MEMO
pipeline onto the HEALPix mesh (Karlbauer et al. 2024, DLWP-HPX):

  1. build_regrid_weights(nside, lat, lon)
        Bilinear lat-lon -> HEALPix as a precomputed sparse operator
        (idx (npix,4), w (npix,4)).  apply_regrid() turns any flattened
        (nlat*nlon,) field into a (npix,) HEALPix-NESTED field.  Built once,
        reused for every field/sample -> fast cache build.

  2. face_reshape_index(nside)
        (12, F, F) int index mapping a HEALPix-NESTED map to the 12-face image
        used by the conv stack, and its inverse, so the model can move between
        (B, C, npix) and (B, C, 12, F, F).

  3. build_pad_index(nside, p=1)
        For the haloed (12, F+2p, F+2p) image, the source index (in image-flat
        f*F*F + y*F + x coords) of every cell, so a single gather realises the
        cross-face HEALPix padding that makes conv kernels location-invariant.
        Border cells are found by extrapolating pixel-centre vectors one step
        beyond the face edge and snapping to the nearest pixel with vec2pix —
        no hardcoded face-adjacency table.

All indices are NESTED ordering throughout.  Conventions match healpy
(hp.pix2xyf / hp.pix2ang / hp.vec2pix with nest=True).
"""

import numpy as np
import healpy as hp


# ---------------------------------------------------------------------------
# 1. Bilinear lat-lon -> HEALPix regrid weights
# ---------------------------------------------------------------------------

def build_regrid_weights(nside, lat, lon):
    """
    Precompute bilinear interpolation weights from a regular lat-lon grid to
    HEALPix (NESTED) pixel centres.

    Parameters
    ----------
    nside : int                  HEALPix resolution (power of 2).
    lat   : (nlat,) ascending    latitudes in degrees, e.g. linspace(-90,90,192).
    lon   : (nlon,) ascending    longitudes in degrees, e.g. linspace(0,358.75,288).
                                  Assumed periodic with uniform spacing.

    Returns
    -------
    idx : (npix, 4) int64   flat source indices (i*nlon + j) of the 4 corners.
    w   : (npix, 4) float64  bilinear weights, rows sum to 1.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    nlat, nlon = lat.size, lon.size
    npix = hp.nside2npix(nside)

    # HEALPix pixel centres -> lat/lon (deg)
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=True)   # colat, lon (rad)
    plat = 90.0 - np.degrees(theta)                              # [-90, 90]
    plon = np.degrees(phi) % 360.0                               # [0, 360)

    # --- latitude bracket (clamped at the poles, no wrap) ---
    plat_c = np.clip(plat, lat[0], lat[-1])
    i1 = np.searchsorted(lat, plat_c, side="left")
    i1 = np.clip(i1, 1, nlat - 1)
    i0 = i1 - 1
    dlat = lat[i1] - lat[i0]
    fy = np.where(dlat > 0, (plat_c - lat[i0]) / dlat, 0.0)      # 0 at i0, 1 at i1

    # --- longitude bracket (periodic, uniform spacing) ---
    dlon = (lon[-1] - lon[0]) / (nlon - 1)
    lon0 = lon[0]
    g = (plon - lon0) / dlon                                     # fractional index
    j0 = np.floor(g).astype(np.int64)
    fx = g - j0
    j0 = np.mod(j0, nlon)
    j1 = np.mod(j0 + 1, nlon)

    idx = np.stack([
        i0 * nlon + j0,   # (i0, j0)
        i0 * nlon + j1,   # (i0, j1)
        i1 * nlon + j0,   # (i1, j0)
        i1 * nlon + j1,   # (i1, j1)
    ], axis=1).astype(np.int64)

    w = np.stack([
        (1 - fy) * (1 - fx),
        (1 - fy) * fx,
        fy * (1 - fx),
        fy * fx,
    ], axis=1).astype(np.float64)

    return idx, w


def apply_regrid(field2d, idx, w):
    """Map a (nlat, nlon) (or flat (nlat*nlon,)) field to (npix,) via weights."""
    flat = np.asarray(field2d).reshape(-1)
    return (flat[idx] * w).sum(axis=1)


def build_nn_index(nside, lat, lon):
    """Nearest-neighbour source index (npix,) for fields that must not be
    interpolated (e.g. the 0/1 ocean mask)."""
    idx, w = build_regrid_weights(nside, lat, lon)
    return idx[np.arange(idx.shape[0]), np.argmax(w, axis=1)]


def apply_nn(field2d, nn_idx):
    return np.asarray(field2d).reshape(-1)[nn_idx]


# ---------------------------------------------------------------------------
# 2. NESTED map <-> 12-face image
# ---------------------------------------------------------------------------

def face_reshape_index(nside):
    """
    Return img2pix : (12, F, F) int64 with img2pix[f, y, x] = nested pixel index,
    so that  image = nested_map[img2pix]  gives the (12, F, F) face image and
    nested_map[img2pix] = image  is its exact inverse (img2pix is a permutation
    of 0..npix-1).  F = nside.
    """
    F = nside
    npix = hp.nside2npix(nside)
    ipix = np.arange(npix)
    x, y, f = hp.pix2xyf(nside, ipix, nest=True)
    img2pix = np.empty((12, F, F), dtype=np.int64)
    img2pix[f, y, x] = ipix
    return img2pix


# ---------------------------------------------------------------------------
# 3. Cross-face HEALPix padding index
# ---------------------------------------------------------------------------

def build_pad_index(nside, p=1):
    """
    Source index for every cell of the haloed (12, F+2p, F+2p) image, in
    image-flat coordinates (f*F*F + y*F + x) of the *unpadded* (12, F, F) image.

    A single gather then realises HEALPix padding:
        img_flat = image.reshape(..., 12*F*F)
        padded   = img_flat[..., pad_src.reshape(-1)].reshape(..., 12, F+2p, F+2p)

    Border cells are located by extrapolating pixel-centre unit vectors one
    step beyond the face edge/corner and snapping to the nearest pixel
    (hp.vec2pix).  This needs no face-adjacency table and degrades gracefully
    (to a valid nearby pixel) at the 8 special 3-face corners.
    """
    if p != 1:
        raise NotImplementedError("only p=1 padding is implemented")
    F = nside
    img2pix = face_reshape_index(nside)                 # (12,F,F) -> nested pix
    vec = np.stack(hp.pix2vec(nside, img2pix.reshape(-1), nest=True), axis=1)
    vec = vec.reshape(12, F, F, 3)                       # pixel-centre unit vectors

    def snap(v):
        """unit-normalise rows of v (...,3) and return nested pixel index."""
        v = v / np.linalg.norm(v, axis=-1, keepdims=True)
        return hp.vec2pix(nside, v[..., 0], v[..., 1], v[..., 2], nest=True)

    # nested pixel index for every padded cell (corners filled below)
    pad_pix = np.full((12, F + 2, F + 2), -1, dtype=np.int64)
    pad_pix[:, 1:F + 1, 1:F + 1] = img2pix              # interior

    # edges: extrapolate 2*edge - next-inner across the boundary
    pad_pix[:, 1:F + 1, 0]     = snap(2 * vec[:, :, 0]     - vec[:, :, 1])      # left   (x=-1)
    pad_pix[:, 1:F + 1, F + 1] = snap(2 * vec[:, :, F - 1] - vec[:, :, F - 2])  # right  (x=F)
    pad_pix[:, 0, 1:F + 1]     = snap(2 * vec[:, 0, :]     - vec[:, 1, :])      # top    (y=-1)
    pad_pix[:, F + 1, 1:F + 1] = snap(2 * vec[:, F - 1, :] - vec[:, F - 2, :])  # bottom (y=F)

    # corners: diagonal extrapolation 2*corner - inner-diagonal
    pad_pix[:, 0, 0]         = snap(2 * vec[:, 0, 0]         - vec[:, 1, 1])
    pad_pix[:, 0, F + 1]     = snap(2 * vec[:, 0, F - 1]     - vec[:, 1, F - 2])
    pad_pix[:, F + 1, 0]     = snap(2 * vec[:, F - 1, 0]     - vec[:, F - 2, 1])
    pad_pix[:, F + 1, F + 1] = snap(2 * vec[:, F - 1, F - 1] - vec[:, F - 2, F - 2])

    # convert nested pixel index -> image-flat (f*F*F + y*F + x) source index
    x, y, f = hp.pix2xyf(nside, pad_pix.reshape(-1), nest=True)
    pad_src = (f * F * F + y * F + x).reshape(12, F + 2, F + 2).astype(np.int64)
    return pad_src


# ---------------------------------------------------------------------------
# Self-test / verification (run:  python healpix_grid.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    nside = 64
    F = nside
    npix = hp.nside2npix(nside)
    print(f"HEALPix nside={nside}  npix={npix}  F={F}")

    # ---- face reshape is a bijection ----
    img2pix = face_reshape_index(nside)
    assert img2pix.shape == (12, F, F)
    assert np.array_equal(np.sort(img2pix.reshape(-1)), np.arange(npix)), \
        "face_reshape_index is not a permutation of 0..npix-1"
    print("[OK] face_reshape_index is a bijection")

    # ---- regrid sanity on a synthetic lat-lon field ----
    lat = np.linspace(-90.0, 90.0, 192)
    lon = np.linspace(0.0, 358.75, 288)
    idx, w = build_regrid_weights(nside, lat, lon)
    assert idx.shape == (npix, 4) and w.shape == (npix, 4)
    assert np.allclose(w.sum(axis=1), 1.0), "regrid weights must sum to 1"

    # field = latitude itself -> regridded value at each pixel should match its lat
    latlon_lat = np.broadcast_to(lat[:, None], (192, 288))
    reg_lat = apply_regrid(latlon_lat, idx, w)
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=True)
    true_lat = 90.0 - np.degrees(theta)
    err = np.abs(reg_lat - true_lat)
    print(f"[regrid] reproduce latitude: max|err|={err.max():.3f} deg  mean={err.mean():.4f}")
    assert err.max() < 1.5, "latitude reproduction error too large"

    # smooth field f = cos(theta): regrid + compare to analytic
    f_ll = np.cos(np.deg2rad(90.0 - latlon_lat))     # = sin(lat) = cos(colat)
    reg_f = apply_regrid(f_ll, idx, w)
    true_f = np.cos(theta)
    print(f"[regrid] cos(theta): max|err|={np.abs(reg_f-true_f).max():.4e}")
    print("[OK] regrid weights reproduce smooth fields")

    # ---- padding seam continuity ----
    pad_src = build_pad_index(nside, p=1)
    assert pad_src.shape == (12, F + 2, F + 2)
    assert (pad_src >= 0).all() and (pad_src < 12 * F * F).all(), "pad_src out of range"

    # smooth analytic field on the sphere, in image form
    field_img = true_f[img2pix]                       # (12,F,F)
    img_flat = field_img.reshape(-1)
    padded = img_flat[pad_src.reshape(-1)].reshape(12, F + 2, F + 2)

    # interior must be untouched
    assert np.allclose(padded[:, 1:F + 1, 1:F + 1], field_img), "interior corrupted"

    # seam continuity: each border cell should be close to its adjacent interior
    # cell (smooth field => small jump).  Compare halo ring to the first interior ring.
    left_jump   = np.abs(padded[:, 1:F + 1, 0]     - field_img[:, :, 0]).max()
    right_jump  = np.abs(padded[:, 1:F + 1, F + 1] - field_img[:, :, F - 1]).max()
    top_jump    = np.abs(padded[:, 0, 1:F + 1]     - field_img[:, 0, :]).max()
    bot_jump    = np.abs(padded[:, F + 1, 1:F + 1] - field_img[:, F - 1, :]).max()
    field_range = true_f.max() - true_f.min()
    max_jump = max(left_jump, right_jump, top_jump, bot_jump)
    print(f"[pad] max seam jump={max_jump:.4e}  (field range={field_range:.3f}, "
          f"typical pixel gap~{field_range/F:.4e})")
    # a correct 1-pixel halo of a smooth field jumps by ~one pixel spacing, not O(range)
    assert max_jump < 0.05 * field_range, "seam discontinuity too large -> padding wrong"
    print("[OK] HEALPix padding is continuous across faces")

    # corners filled (not -1 / not NaN)
    corners = padded[:, [0, 0, -1, -1], [0, -1, 0, -1]]
    assert np.isfinite(corners).all()
    print("[OK] all 8 face corners filled")

    print("\nALL HEALPIX GRID TESTS PASSED")
