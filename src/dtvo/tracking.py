"""DTVO tracking thread (Yan, Li, Zhou 2022, "Robust and efficient edge-based visual odometry", Sec. 4).

  * DT map of the current frame (Sec. 4.2): Euclidean distance to the nearest edge,
    built from ALL detected edges (every EDGE_* tag), one per pyramid level.
  * 3D-2D edge alignment (Eq. 5-7): reference-keyframe edges are back-projected with
    their depth, warped into the current frame and the residual is the DT value there.
    Huber-robust Gauss-Newton / Levenberg-Marquardt on SE(3).
  * Four initial motion hypotheses (Sec. 4.2): constant, none, half, double motion;
    the one with the lowest energy (Eq. 6) starts the optimisation.
  * Weighted edge alignment (Sec. 4.3, Eq. 8-9): every reference edge pixel gets one of
    5 types from its 3x3 neighbourhood (Fig. 5) and a per-level weight
        w_t^k = (log(N_sum / N_t))^((l - k) / l)
    so dominant edge directions drive coarse levels and rarer ones refine fine levels.
  * Coarse-to-fine over the 3-level pyramid.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..common.dataset import Camera
from ..common.edges import EdgeMap
from ..common.preprocess import FrameLevel
from .se3 import orthonormalize, se3_exp

N_TYPES = 5
TYPE_HORIZONTAL, TYPE_VERTICAL, TYPE_DIAG_MAIN, TYPE_DIAG_ANTI, TYPE_OTHER = range(N_TYPES)


@dataclass
class TrackConfig:
    huber_px: float = 1.5              # Huber threshold (pixels of DT distance, any level)
    outlier_px: tuple = (6.0, 8.0, 10.0)  # per level: residual above this = outlier (no gradient)
    iterations: tuple = (10, 15, 25)   # per level (0 = full resolution)
    weighted: bool = True              # weighted edge alignment (Sec. 4.3); False = ablation "w/o"
    weight_clip: tuple = (0.5, 4.0)    # "in practice the weight will be within a certain range"
    inlier_px: float = 2.0             # level-0 residual counted as a matched edge
    min_inlier_ratio: float = 0.25     # below this the frame counts as "tracking lost"
    retry_inlier_ratio: float = 0.7    # below this, also optimise from the other hypotheses
    max_ref_points: int = 12000        # level-0 reference points (random subset if more)


# ----------------------------------------------------------------------------- DT maps
@dataclass
class DTMap:
    dist: np.ndarray       # H x W float32, distance (px) to the nearest edge
    gx: np.ndarray         # d dist / du
    gy: np.ndarray         # d dist / dv
    inside: np.ndarray     # H x W bool, valid image content
    camera: Camera


def build_dt(edge_map: EdgeMap, level: FrameLevel) -> DTMap:
    """Distance transform of the edge image (Felzenszwalb & Huttenlocher exact L2 via OpenCV)."""
    dist = cv2.distanceTransform((~edge_map.mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # Pixels outside the valid image area must not attract points: their distance is computed
    # to edges inside anyway, which is what we want; we just reject projections landing there.
    gy, gx = np.gradient(dist)
    return DTMap(dist, gx.astype(np.float32), gy.astype(np.float32), level.inside, level.camera)


def build_dt_pyramid(edge_maps: list[EdgeMap], levels: list[FrameLevel]) -> list[DTMap]:
    return [build_dt(e, l) for e, l in zip(edge_maps, levels)]


def bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample img at float (u, v); caller guarantees 0 <= u < W-1, 0 <= v < H-1."""
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    a, b = u - u0, v - v0
    return ((1 - a) * (1 - b) * img[v0, u0] + a * (1 - b) * img[v0, u0 + 1]
            + (1 - a) * b * img[v0 + 1, u0] + a * b * img[v0 + 1, u0 + 1])


def project_and_sample(P: np.ndarray, T: np.ndarray, dt: DTMap, with_grad: bool):
    """Warp points P (N x 3, reference frame) by T (ref -> current) and read the DT there.

    Returns Pc (N x 3 in current frame), u, v, valid mask, residual (DT), and the
    DT gradient (gx, gy) if requested (only meaningful where valid).
    """
    Pc = P @ T[:3, :3].T + T[:3, 3]
    z = Pc[:, 2]
    cam = dt.camera
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.fx * Pc[:, 0] / z + cam.cx
        v = cam.fy * Pc[:, 1] / z + cam.cy
    h, w = dt.dist.shape
    valid = (z > 0.05) & (u >= 0) & (v >= 0) & (u < w - 1.001) & (v < h - 1.001)
    valid[valid] = dt.inside[np.round(v[valid]).astype(int), np.round(u[valid]).astype(int)]
    r = np.zeros(len(P), np.float64)
    r[valid] = bilinear(dt.dist, u[valid], v[valid])
    if not with_grad:
        return Pc, u, v, valid, r
    gx = np.zeros(len(P))
    gy = np.zeros(len(P))
    gx[valid] = bilinear(dt.gx, u[valid], v[valid])
    gy[valid] = bilinear(dt.gy, u[valid], v[valid])
    return Pc, u, v, valid, r, gx, gy


def dt_point_jacobian(Pc: np.ndarray, gx: np.ndarray, gy: np.ndarray, cam: Camera) -> np.ndarray:
    """d DT(pi(Pc)) / d Pc  (N x 3): DT gradient times the projection Jacobian."""
    x, y, z = Pc[:, 0], Pc[:, 1], Pc[:, 2]
    iz = 1.0 / z
    j = np.empty((len(Pc), 3))
    j[:, 0] = gx * cam.fx * iz
    j[:, 1] = gy * cam.fy * iz
    j[:, 2] = -(gx * cam.fx * x + gy * cam.fy * y) * iz * iz
    return j


def pose_jacobian(Pc: np.ndarray, j3: np.ndarray) -> np.ndarray:
    """Residual Jacobian w.r.t. a left twist (v, w) on the ref->current transform: [j3, Pc x j3]."""
    return np.hstack([j3, np.cross(Pc, j3)])


def huber_weight(r: np.ndarray, k: float) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= k, 1.0, k / np.maximum(a, 1e-12))


def huber_cost(r: np.ndarray, k: float) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= k, 0.5 * a * a, k * (a - 0.5 * k))


# ----------------------------------------------------------------------------- edge types
def edge_types(mask: np.ndarray) -> np.ndarray:
    """Type of every edge pixel from its 3x3 neighbourhood (paper Fig. 5), H x W int8 (-1 off-edge).

      0  left + right neighbours only          (horizontal run)
      1  up + down neighbours only             (vertical run)
      2  up-left + down-right only             (main diagonal)
      3  up-right + down-left only             (anti diagonal)
      4  anything else (ends, corners, junctions, thick edges)
    """
    m = np.pad(mask.astype(np.uint8), 1)
    h, w = mask.shape
    nb = {}
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                nb[(dy, dx)] = m[1 + dy:1 + dy + h, 1 + dx:1 + dx + w].astype(bool)
    count = sum(nb.values(), np.zeros((h, w), np.int32))
    t = np.full((h, w), TYPE_OTHER, np.int8)
    two = count == 2
    t[two & nb[(0, -1)] & nb[(0, 1)]] = TYPE_HORIZONTAL
    t[two & nb[(-1, 0)] & nb[(1, 0)]] = TYPE_VERTICAL
    t[two & nb[(-1, -1)] & nb[(1, 1)]] = TYPE_DIAG_MAIN
    t[two & nb[(-1, 1)] & nb[(1, -1)]] = TYPE_DIAG_ANTI
    t[~mask] = -1
    return t


def type_weights(types: np.ndarray, level: int, top_level: int, clip=(0.5, 4.0)) -> np.ndarray:
    """Eq. 8 per edge type: w_t^k = (log(N_sum / N_t))^((l - k) / l), returned per point."""
    counts = np.bincount(types, minlength=N_TYPES).astype(float)
    n_sum = counts.sum()
    if n_sum == 0:   # no reference points on this level
        return np.zeros(0)
    base = np.log(n_sum / np.maximum(counts, 1.0))
    base = np.clip(base, clip[0], clip[1])
    w_t = base ** ((top_level - level) / top_level) if top_level > 0 else np.ones(N_TYPES)
    return w_t[types]


# ----------------------------------------------------------------------------- reference
@dataclass
class RefLevel:
    P: np.ndarray          # N x 3 points in the keyframe camera frame
    types: np.ndarray      # N edge types
    weights: np.ndarray    # N weights (Eq. 8) for this level


def make_ref_level(uv: np.ndarray, depth: np.ndarray, types: np.ndarray, cam: Camera,
                   level: int, top_level: int, cfg: TrackConfig) -> RefLevel:
    ok = depth > 0
    uv, z, t = uv[ok].astype(np.float64), depth[ok].astype(np.float64), types[ok]
    P = np.column_stack([(uv[:, 0] - cam.cx) / cam.fx * z, (uv[:, 1] - cam.cy) / cam.fy * z, z])
    w = (type_weights(t, level, top_level, cfg.weight_clip) if cfg.weighted
         else np.ones(len(t)))
    return RefLevel(P, t, w)


# ----------------------------------------------------------------------------- tracker
@dataclass
class TrackResult:
    T_cr: np.ndarray            # reference keyframe -> current frame
    inlier_ratio: float
    type_inliers: np.ndarray    # matched reference edges per type (level 0)
    energy: float
    hypothesis: str
    lost: bool


def level_energy(ref: RefLevel, dt: DTMap, T: np.ndarray, cfg: TrackConfig, level: int) -> float:
    """Eq. 9 energy. Points leaving the image or beyond the outlier distance pay the
    outlier cost, so the optimiser cannot lower the energy by pushing points away."""
    _, _, _, valid, r = project_and_sample(ref.P, T, dt, with_grad=False)
    cap = cfg.outlier_px[level]
    r = np.where(valid, np.minimum(r, cap), cap)
    return float(np.sum(ref.weights * huber_cost(r, cfg.huber_px)) / max(len(r), 1))


def optimise_level(ref: RefLevel, dt: DTMap, T: np.ndarray, cfg: TrackConfig, level: int) -> np.ndarray:
    """Huber-weighted Levenberg-Marquardt on SE(3) at one pyramid level."""
    lam = 1e-4
    E = level_energy(ref, dt, T, cfg, level)
    cap = cfg.outlier_px[level]
    for _ in range(cfg.iterations[level]):
        Pc, _, _, valid, r, gx, gy = project_and_sample(ref.P, T, dt, with_grad=True)
        use = valid & (r < cap)
        if use.sum() < 12:
            break
        j3 = dt_point_jacobian(Pc[use], gx[use], gy[use], dt.camera)
        J = pose_jacobian(Pc[use], j3)
        w = ref.weights[use] * huber_weight(r[use], cfg.huber_px)
        H = J.T @ (J * w[:, None])
        b = J.T @ (w * r[use])
        improved = False
        for _ in range(5):
            try:
                delta = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), b)
            except np.linalg.LinAlgError:
                break
            T_new = se3_exp(delta) @ T
            E_new = level_energy(ref, dt, T_new, cfg, level)
            if E_new < E:
                T, E, lam = T_new, E_new, max(lam / 4, 1e-7)
                improved = True
                break
            lam *= 10
        if not improved or np.linalg.norm(delta) < 1e-6:
            break
    return T


def track(ref_levels: list[RefLevel], dts: list[DTMap], hypotheses: dict[str, np.ndarray],
          cfg: TrackConfig) -> TrackResult:
    """Coarse-to-fine alignment of the reference keyframe to the current frame."""
    top = len(ref_levels) - 1
    # Sec. 4.2: pick the initial motion with the lowest energy (evaluated at the coarsest level)
    energies = {k: level_energy(ref_levels[top], dts[top], T, cfg, top) for k, T in hypotheses.items()}
    order = sorted(energies, key=energies.get)
    best = None
    for name in order:
        res = _align(ref_levels, dts, hypotheses[name], name, cfg)
        if best is None or res.energy < best.energy:
            best = res
        # Safety net (not in the paper): a poor match usually means the coarse level fell
        # into a wrong basin, so try the remaining hypotheses and keep the lowest energy.
        if best.inlier_ratio >= cfg.retry_inlier_ratio:
            break
    return best


def _align(ref_levels, dts, T, name, cfg) -> TrackResult:
    top = len(ref_levels) - 1
    for lvl in range(top, -1, -1):
        T = optimise_level(ref_levels[lvl], dts[lvl], T, cfg, lvl)
    T = orthonormalize(T)
    ref = ref_levels[0]
    _, _, _, valid, r = project_and_sample(ref.P, T, dts[0], with_grad=False)
    inl = valid & (r < cfg.inlier_px)
    ratio = float(inl.mean()) if len(inl) else 0.0
    type_inl = np.bincount(ref.types[inl], minlength=N_TYPES)
    return TrackResult(T, ratio, type_inl, level_energy(ref, dts[0], T, cfg, 0), name,
                       ratio < cfg.min_inlier_ratio)
