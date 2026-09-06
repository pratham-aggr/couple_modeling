"""
learnable_remap.py
==================
Learnable remapping between the MEMO 192x288 regular lat/lon grid and the POP
gx1v7 tripole grid (384x320), replacing the FIXED interpolators used by
model_server.py on the no-coupler direct MEMO<->POP path:

    fixed today                          learnable here
    -----------                          --------------
    ScatterToRegular (IDW k-NN)    ->    EncoderRemap: same k-NN gather, but the
      gx1v7 SST -> 192x288               per-destination weights are trainable
                                         (softmax over k neighbours; initialised
                                         at the IDW weights, so step 0 == fixed).

    RegularToScatter (bilinear)    ->    DecoderRemap: same 4-point gather with
      192x288 fluxes -> gx1v7            PER-CHANNEL trainable weights
                                         (initialised at the bilinear weights),
                                         plus an optional zero-initialised
                                         residual CNN on the gx1v7 grid.

Both operators are exactly the fixed remap at initialisation, so a freshly
built LearnedCoupler reproduces the production pipeline bit-for-bit (up to
float32), and training can only move away from that baseline if it reduces the
flux error on the ocean grid.

Training (train_remap.py) wraps these around the FROZEN vfcg1m61 UNet:

    gx1v7 SST --EncoderRemap--> MEMO input --frozen UNet--> 192x288 fluxes
              --DecoderRemap--> gx1v7 fluxes  vs  truth fluxes on gx1v7

Deployment (model_server.py --learned_remap ckpt.pt) swaps the two fixed
remaps for the trained ones; everything else (units, sign handling, caps,
anchors, ice masking) is untouched.  IMPORTANT contracts:
  * EncoderRemap eats SST in KELVIN (server must convert degC -> K BEFORE
    encoding; the decoder weights are not constrained to sum to 1, so remap
    and the +273.15 offset do not commute).
  * DecoderRemap eats the 8 UNet outputs in STORED (zarr) units, i.e. BEFORE
    the J/m2-per-6h -> W/m2 conversions, because the residual CNN is nonlinear.
    The per-channel unit conversions are scalar multiplies and can be applied
    to the decoded gx1v7 fields afterwards.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Grid constants (must match model_server.py / train_unet.py)
CAM_NLAT, CAM_NLON = 192, 288
GX_NJ, GX_NI = 384, 320
GX1V7_DOMAIN = "/glade/campaign/cesm/cesmdata/inputdata/share/domains/domain.ocn.gx1v7.210716.nc"

OUTPUT_VARS = ["TAUX", "TAUY", "SHFLX", "LHFLX", "QFLX", "FSDS_J", "FLDS_J", "PRECT"]


def cam_latlon():
    """MEMO/CAM 1-deg grid: 192 lats S->N inclusive of poles, 288 lons 0..360."""
    lats = np.linspace(-90.0, 90.0, CAM_NLAT)
    lons = np.linspace(0.0, 360.0, CAM_NLON, endpoint=False)
    return lats, lons


def load_gx1v7_domain(path=GX1V7_DOMAIN):
    """gx1v7 cell-centre lon/lat (deg) and ocean mask, shapes (GX_NJ, GX_NI)."""
    import netCDF4 as ncf
    with ncf.Dataset(str(path), "r") as ds:
        xc = ds["xc"][:].data.astype(np.float64)
        yc = ds["yc"][:].data.astype(np.float64)
        mask = ds["mask"][:].data.astype(np.int32)
    if xc.shape != (GX_NJ, GX_NI):
        raise ValueError(f"gx1v7 domain shape {xc.shape} != ({GX_NJ},{GX_NI})")
    return xc, yc, mask


# =============================================================================
# Geometry builders — replicate the numerics of model_server.ScatterToRegular
# and RegularToScatter so that the initial weights ARE the production remap.
# =============================================================================

def build_encoder_geometry(gx_lon2d, gx_lat2d, gx_mask2d, k=8, k_base=4):
    """k-NN neighbour indices + IDW init weights for gx1v7(ocean) -> CAM.

    The PRODUCTION ScatterToRegular uses IDW over k=4 neighbours, so the init
    weights are IDW over the first k_base=4 only; neighbours beyond k_base get
    ~zero initial weight (they exist to give training room).  A freshly-built
    encoder therefore reproduces the fixed remap exactly for any k >= 4.

    Returns dict:
      src_valid : (nvalid,) int64  — flat gx indices of ocean cells (the encoder
                                     input is the compressed ocean-only vector)
      idx       : (ncam, k) int64  — neighbour positions INTO the compressed vector
      w_init    : (ncam, k) float32 — IDW over first k_base, ~0 beyond (sum=1)
    """
    from scipy.spatial import cKDTree
    m = gx_mask2d.astype(bool).ravel()
    src_valid = np.where(m)[0].astype(np.int64)
    slon = np.radians(gx_lon2d.ravel()[m])
    slat = np.radians(gx_lat2d.ravel()[m])
    src_xyz = np.stack([np.cos(slat) * np.cos(slon),
                        np.cos(slat) * np.sin(slon), np.sin(slat)], axis=1)
    tree = cKDTree(src_xyz)
    cam_lats, cam_lons = cam_latlon()
    latg, long_ = np.meshgrid(np.radians(cam_lats), np.radians(cam_lons), indexing="ij")
    dlat = latg.ravel(); dlon = long_.ravel()
    dst_xyz = np.stack([np.cos(dlat) * np.cos(dlon),
                        np.cos(dlat) * np.sin(dlon), np.sin(dlat)], axis=1)
    # First k_base neighbours from a SEPARATE k=k_base query — the identical call
    # ScatterToRegular makes, so tie-breaking (equidistant neighbours near the
    # poles) matches production exactly.  Extra neighbours come from the k query
    # (they may duplicate a tie partner at ~41 all-land polar points; harmless —
    # their init weight is ~0 and the weights are learnable anyway).
    dist, idx = tree.query(dst_xyz, k=k_base)
    w = 1.0 / (dist ** 2 + 1e-12)
    if k > k_base:
        _, idx_k = tree.query(dst_xyz, k=k)
        idx = np.concatenate([idx, idx_k[:, k_base:]], axis=1)
        w = np.concatenate([w, 1e-9 * w[:, :1].repeat(k - k_base, axis=1)], axis=1)
    w = w / w.sum(axis=1, keepdims=True)
    return dict(src_valid=src_valid,
                idx=idx.astype(np.int64),
                w_init=w.astype(np.float32))


def build_decoder_geometry(gx_lon2d, gx_lat2d):
    """Bilinear 4-point indices + weights for CAM -> gx1v7 (all gx cells).

    Returns dict:
      idx    : (ngx, 4) int64   — flat CAM indices (i*CAM_NLON + j), corner order
                                  (i0j0, i0j1, i1j0, i1j1) as in RegularToScatter
      w_init : (ngx, 4) float32 — bilinear weights (w00, w01, w10, w11)
    """
    cam_lats, cam_lons = cam_latlon()
    nlat, nlon = CAM_NLAT, CAM_NLON
    flat_lat = np.clip(gx_lat2d.ravel(), cam_lats[0], cam_lats[-1])
    flat_lon = gx_lon2d.ravel() % 360.0
    i0 = np.clip(np.searchsorted(cam_lats, flat_lat, side="right") - 1, 0, nlat - 2)
    i1 = i0 + 1
    j0 = np.clip(np.searchsorted(cam_lons, flat_lon, side="right") - 1, 0, nlon - 1)
    j1 = (j0 + 1) % nlon
    lon_right = np.where(j0 < nlon - 1, cam_lons[j1], cam_lons[0] + 360.0)
    dlat = cam_lats[i1] - cam_lats[i0]
    dlon = lon_right - cam_lons[j0]
    a = np.clip((flat_lat - cam_lats[i0]) / np.where(dlat == 0, 1.0, dlat), 0.0, 1.0)
    b = np.clip((flat_lon - cam_lons[j0]) / np.where(dlon == 0, 1.0, dlon), 0.0, 1.0)
    idx = np.stack([i0 * nlon + j0, i0 * nlon + j1,
                    i1 * nlon + j0, i1 * nlon + j1], axis=1)
    w = np.stack([(1 - a) * (1 - b), (1 - a) * b,
                  a * (1 - b), a * b], axis=1)
    return dict(idx=idx.astype(np.int64), w_init=w.astype(np.float32))


# =============================================================================
# Torch modules
# =============================================================================

class EncoderRemap(nn.Module):
    """gx1v7 ocean SST (K) -> CAM 192x288 SST (K), learnable k-NN weights.

    Weights are a softmax over the k neighbours (convex combination), so
    constants are preserved exactly — the degC/K offset commutes and the
    encoder cannot invent SST outside the local neighbour range.  Initialised
    at log(IDW weights) so step 0 reproduces ScatterToRegular.
    A single weight table is shared by SST and SST_prev (same physical field,
    same operator), keeping the derived dSST/dt channel consistent.
    """

    def __init__(self, geom):
        super().__init__()
        self.register_buffer("src_valid", torch.from_numpy(np.asarray(geom["src_valid"])))
        self.register_buffer("idx", torch.from_numpy(np.asarray(geom["idx"])))
        w0 = torch.from_numpy(np.asarray(geom["w_init"])).clamp_min(1e-12)
        self.theta = nn.Parameter(torch.log(w0))          # (ncam, k)

    def forward(self, x):
        """x: (B, C, GX_NJ, GX_NI) -> (B, C, CAM_NLAT, CAM_NLON).  Land gx cells ignored."""
        B, C = x.shape[0], x.shape[1]
        flat = x.reshape(B, C, -1)[:, :, self.src_valid]          # (B,C,nvalid)
        nb = flat[:, :, self.idx]                                 # (B,C,ncam,k)
        w = torch.softmax(self.theta, dim=-1)                     # (ncam,k)
        out = (nb * w).sum(-1)                                    # (B,C,ncam)
        return out.reshape(B, C, CAM_NLAT, CAM_NLON)


class GxResidualCNN(nn.Module):
    """Small residual corrector on the gx1v7 grid, applied in NORMALISED flux
    space.  Input = decoded fluxes (8ch, normalised) + 5 static geometry
    channels (ocean mask, sin/cos lat, sin/cos lon).  Circular padding along i
    (gx longitude index wraps); replicate along j.  Final conv is zero-init so
    the module starts as the identity."""

    def __init__(self, n_flux=8, n_static=5, hidden=48, depth=3):
        super().__init__()
        chans = [n_flux + n_static] + [hidden] * (depth - 1) + [n_flux]
        self.convs = nn.ModuleList(
            nn.Conv2d(chans[i], chans[i + 1], 3, padding=0) for i in range(depth))
        nn.init.zeros_(self.convs[-1].weight)
        nn.init.zeros_(self.convs[-1].bias)

    @staticmethod
    def _pad(x):
        x = F.pad(x, (1, 1, 0, 0), mode="circular")    # i (last dim) wraps
        return F.pad(x, (0, 0, 1, 1), mode="replicate")  # j: replicate

    def forward(self, y_norm, statics):
        h = torch.cat([y_norm, statics.expand(y_norm.shape[0], -1, -1, -1)], dim=1)
        for c, conv in enumerate(self.convs):
            h = conv(self._pad(h))
            if c < len(self.convs) - 1:
                h = F.gelu(h)
        return y_norm + h


class DecoderRemap(nn.Module):
    """CAM 192x288 fluxes (8ch, STORED zarr units) -> gx1v7 (384x320).

    Per-channel 4-point weights initialised at the production bilinear weights
    (raw, not softmax — sharpening beyond convex combinations is allowed; a
    sum-to-one penalty keeps constants approximately preserved), plus an
    optional zero-init residual CNN in normalised flux space.

    y_mean/y_std are the per-channel stats used ONLY inside the residual CNN
    (normalise -> correct -> unnormalise); pass the gx-grid target stats.
    """

    def __init__(self, geom, n_channels=8, residual=True, y_mean=None, y_std=None,
                 hidden=48, depth=3):
        super().__init__()
        self.register_buffer("idx", torch.from_numpy(np.asarray(geom["idx"])))
        w0 = torch.from_numpy(np.asarray(geom["w_init"]))          # (ngx,4)
        self.weight = nn.Parameter(w0[None].repeat(n_channels, 1, 1))  # (C,ngx,4)
        self.n_channels = n_channels
        self.residual = None
        if residual:
            self.residual = GxResidualCNN(n_flux=n_channels, hidden=hidden, depth=depth)
        ym = np.zeros(n_channels, np.float32) if y_mean is None else np.asarray(y_mean, np.float32)
        ys = np.ones(n_channels, np.float32) if y_std is None else np.asarray(y_std, np.float32)
        self.register_buffer("y_mean", torch.from_numpy(ym)[None, :, None, None])
        self.register_buffer("y_std", torch.from_numpy(ys)[None, :, None, None])
        self.register_buffer("statics", torch.zeros(1, 5, GX_NJ, GX_NI))

    def set_statics(self, gx_lon2d, gx_lat2d, gx_mask2d):
        lat = np.radians(gx_lat2d); lon = np.radians(gx_lon2d)
        s = np.stack([gx_mask2d.astype(np.float32),
                      np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)])
        self.statics.copy_(torch.from_numpy(s.astype(np.float32))[None])

    def forward(self, y):
        """y: (B, C, CAM_NLAT, CAM_NLON) stored units -> (B, C, GX_NJ, GX_NI)."""
        B, C = y.shape[0], y.shape[1]
        flat = y.reshape(B, C, -1)                                  # (B,C,ncam)
        nb = flat[:, :, self.idx]                                   # (B,C,ngx,4)
        out = (nb * self.weight[None]).sum(-1)                      # (B,C,ngx)
        out = out.reshape(B, C, GX_NJ, GX_NI)
        if self.residual is not None:
            out_n = (out - self.y_mean) / self.y_std
            out_n = self.residual(out_n, self.statics)
            out = out_n * self.y_std + self.y_mean
        return out

    def sum1_penalty(self):
        """Mean squared deviation of the 4-point weight sums from 1."""
        return ((self.weight.sum(-1) - 1.0) ** 2).mean()


class LearnedCoupler(nn.Module):
    """Encoder + decoder bundle with everything needed for deployment."""

    def __init__(self, enc_geom, dec_geom, dec_residual=True,
                 y_mean=None, y_std=None, hidden=48, depth=3, config=None):
        super().__init__()
        self.encoder = EncoderRemap(enc_geom)
        self.decoder = DecoderRemap(dec_geom, residual=dec_residual,
                                    y_mean=y_mean, y_std=y_std,
                                    hidden=hidden, depth=depth)
        self.config = dict(config or {})
        self.config.setdefault("dec_residual", dec_residual)
        self.config.setdefault("hidden", hidden)
        self.config.setdefault("depth", depth)
        self.config.setdefault("output_vars", OUTPUT_VARS)

    # ---- deployment API (numpy in / numpy out, no grad) --------------------
    @torch.no_grad()
    def encode_np(self, sst_gx_K):
        """(GX_NJ,GX_NI) SST in K -> (192,288) SST in K, float64."""
        dev = self.decoder.y_mean.device
        t = torch.from_numpy(np.ascontiguousarray(sst_gx_K, dtype=np.float32))[None, None].to(dev)
        return self.encoder(t)[0, 0].cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def decode_np(self, y_stack):
        """(8,192,288) stored units -> (8,GX_NJ,GX_NI) stored units, float64."""
        dev = self.decoder.y_mean.device
        t = torch.from_numpy(np.ascontiguousarray(y_stack, dtype=np.float32))[None].to(dev)
        return self.decoder(t)[0].cpu().numpy().astype(np.float64)

    # ---- persistence --------------------------------------------------------
    def save(self, path):
        torch.save({"state_dict": self.state_dict(),
                    "enc_shapes": {"src_valid": tuple(self.encoder.src_valid.shape),
                                   "idx": tuple(self.encoder.idx.shape)},
                    "dec_shapes": {"idx": tuple(self.decoder.idx.shape)},
                    "config": self.config}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        cfg = ck["config"]
        k = ck["enc_shapes"]["idx"][1]
        enc_geom = dict(src_valid=np.zeros(ck["enc_shapes"]["src_valid"], np.int64),
                        idx=np.zeros(ck["enc_shapes"]["idx"], np.int64),
                        w_init=np.full(ck["enc_shapes"]["idx"], 1.0 / k, np.float32))
        dec_geom = dict(idx=np.zeros(ck["dec_shapes"]["idx"], np.int64),
                        w_init=np.full(ck["dec_shapes"]["idx"], 0.25, np.float32))
        m = cls(enc_geom, dec_geom, dec_residual=cfg.get("dec_residual", True),
                hidden=cfg.get("hidden", 48), depth=cfg.get("depth", 3), config=cfg)
        m.load_state_dict(ck["state_dict"])
        return m.to(device).eval()


def build_learned_coupler(domain_path=GX1V7_DOMAIN, k_enc=8, dec_residual=True,
                          y_mean=None, y_std=None, hidden=48, depth=3):
    """Build a LearnedCoupler whose initial state IS the production fixed remap."""
    gx_xc, gx_yc, gx_mask = load_gx1v7_domain(domain_path)
    enc_geom = build_encoder_geometry(gx_xc, gx_yc, gx_mask, k=k_enc)
    dec_geom = build_decoder_geometry(gx_xc, gx_yc)
    lc = LearnedCoupler(enc_geom, dec_geom, dec_residual=dec_residual,
                        y_mean=y_mean, y_std=y_std, hidden=hidden, depth=depth,
                        config={"k_enc": k_enc, "domain": str(domain_path)})
    lc.decoder.set_statics(gx_xc, gx_yc, gx_mask)
    return lc, (gx_xc, gx_yc, gx_mask)
