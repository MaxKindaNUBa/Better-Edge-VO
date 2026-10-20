"""Box 2 - FIND EDGES.

Finds the edges that the VO tracks, on every pyramid level:
  * Canny whose thresholds adapt to each frame's gradient statistics
    (fixed thresholds break under auto-exposure and lighting changes)
  * tiny edge fragments removed (mostly sensor noise, cost time, add nothing)
  * every edge pixel tagged by how usable its depth is:

      EDGE_DEPTH      reliable depth -> can be lifted to a 3D reference point
      EDGE_NO_DEPTH   no depth reading -> only usable in the distance map
      EDGE_DEPTH_JUMP sits on a depth discontinuity (e.g. a person's outline):
                      its depth could belong to either side -> distance map only

All edges (any tag) go into the distance map; only EDGE_DEPTH edges become
reference points. Texture edges are kept on purpose: they are fixed on their
surface and are the best edges to track.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .preprocess import FrameLevel, PreparedFrame, depth_jump_threshold

EDGE_NONE = 0
EDGE_DEPTH = 1
EDGE_NO_DEPTH = 2
EDGE_DEPTH_JUMP = 3


@dataclass
class EdgeConfig:
    blur_ksize: int = 5
    blur_sigma: float = 1.0
    high_percentile: float = 88.0   # high threshold = this percentile of gradient magnitude
    low_ratio: float = 0.4          # low threshold = low_ratio * high
    min_high: float = 60.0          # clamps (Sobel-3x3 L2 units): stop noise edges in blank scenes...
    max_high: float = 300.0         # ...and stop losing real edges in busy ones
    min_chain_px: int = 12          # at full resolution; halved per pyramid level
    jump_k: float = 8.0             # depth spread > max(jump_floor, jump_k * sigma(z)) = discontinuity
    jump_floor: float = 0.04
    jump_radius: int = 4            # px at full resolution (halved per level); measured colour-vs-depth
                                    # outline offset on TUM/Bonn: 65-85 % of outline edges within 4 px
    min_jump_px: int = 8            # depth-jump lines shorter than this are noise, not outlines


@dataclass
class EdgeMap:
    level: int
    mask: np.ndarray        # H x W bool, all edges
    labels: np.ndarray      # H x W uint8, EDGE_* tag per pixel (EDGE_NONE off-edge)
    uv: np.ndarray          # N x 2 int32, (u = column, v = row) of each edge pixel
    depth: np.ndarray       # N float32, metres (0 unless EDGE_DEPTH)
    label: np.ndarray       # N uint8, EDGE_* tag per edge pixel
    thresholds: tuple[float, float]

    @property
    def reference_uv(self) -> np.ndarray:
        """Edge pixels usable as 3D reference points."""
        return self.uv[self.label == EDGE_DEPTH]

    def counts(self) -> dict[str, int]:
        return {"total": len(self.label),
                "depth": int((self.label == EDGE_DEPTH).sum()),
                "no_depth": int((self.label == EDGE_NO_DEPTH).sum()),
                "depth_jump": int((self.label == EDGE_DEPTH_JUMP).sum())}


def _adaptive_canny(gray: np.ndarray, inside: np.ndarray,
                    cfg: EdgeConfig) -> tuple[np.ndarray, tuple[float, float]]:
    blur = cv2.GaussianBlur(gray, (cfg.blur_ksize, cfg.blur_ksize), cfg.blur_sigma)
    gx = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3)
    mag = np.hypot(gx.astype(np.float32), gy.astype(np.float32))
    # Statistics from real image content only, so the black border can't skew them
    high = float(np.clip(np.percentile(mag[inside], cfg.high_percentile), cfg.min_high, cfg.max_high))
    low = cfg.low_ratio * high
    # Feed the same gradients to Canny so the thresholds mean exactly what was measured
    edges = cv2.Canny(gx, gy, low, high, L2gradient=True)
    return (edges > 0) & inside, (low, high)


def _remove_short_chains(mask: np.ndarray, min_px: int) -> np.ndarray:
    if min_px <= 1:
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_px
    keep[0] = False  # background
    return keep[lab]


def depth_discontinuities(z: np.ndarray, valid: np.ndarray, cfg: EdgeConfig = EdgeConfig(),
                          level: int = 0) -> np.ndarray:
    """Pixels on a real depth jump, from differences between *nearby* pixels only.

    Comparing pixels 1 and 2 steps apart (the 2-step bridges a one-pixel blur ramp) keeps
    steeply slanted surfaces out: they change a few mm per pixel, an outline jumps tens of cm.
    A max-min spread over a large window cannot tell those two apart. Holes don't count.
    """
    # 5x5 median first: the Kinect produces lone pixels 10+ cm off their neighbours, which
    # would otherwise show up as tiny fake jumps. Used for finding jumps only, never as depth.
    zm = cv2.medianBlur(np.where(valid, z, 0.0).astype(np.float32), 5)
    vm = valid & (zm > 0)
    h, w = z.shape
    disc = np.zeros((h, w), bool)
    for dy, dx in ((0, 1), (1, 0), (0, 2), (2, 0)):
        a, b = zm[:h - dy, :w - dx], zm[dy:, dx:]
        both = vm[:h - dy, :w - dx] & vm[dy:, dx:]
        jump = both & (np.abs(a - b) > depth_jump_threshold(np.minimum(a, b), cfg.jump_k, cfg.jump_floor))
        disc[:h - dy, :w - dx] |= jump
        disc[dy:, dx:] |= jump
    return _remove_short_chains(disc, max(3, cfg.min_jump_px >> level))


def _depth_labels(mask: np.ndarray, frame: FrameLevel, cfg: EdgeConfig) -> np.ndarray:
    z, valid = frame.depth, frame.valid
    # Widen the jump lines by r: the Kinect's foreground "fattening" puts depth outlines
    # 1-6 px outside the colour outline, so the two rarely coincide exactly.
    r = max(1, cfg.jump_radius >> frame.level)
    jump = cv2.dilate(depth_discontinuities(z, valid, cfg, frame.level).astype(np.uint8),
                      np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0

    labels = np.full(mask.shape, EDGE_NONE, np.uint8)
    labels[mask & ~valid] = EDGE_NO_DEPTH
    labels[mask & valid] = EDGE_DEPTH
    # An outline edge counts as a jump even if its own pixel lost its depth reading
    labels[mask & jump] = EDGE_DEPTH_JUMP
    return labels


def detect_edges(frame: FrameLevel, cfg: EdgeConfig = EdgeConfig()) -> EdgeMap:
    mask, thresholds = _adaptive_canny(frame.gray, frame.inside, cfg)
    mask = _remove_short_chains(mask, max(3, cfg.min_chain_px >> frame.level))
    labels = _depth_labels(mask, frame, cfg)

    v, u = np.nonzero(mask)
    label = labels[v, u]
    depth = np.where(label == EDGE_DEPTH, frame.depth[v, u], 0.0).astype(np.float32)
    return EdgeMap(frame.level, mask, labels, np.column_stack([u, v]).astype(np.int32),
                   depth, label, thresholds)


def detect_edges_pyramid(frame: PreparedFrame, cfg: EdgeConfig = EdgeConfig()) -> list[EdgeMap]:
    return [detect_edges(level, cfg) for level in frame.levels]


def remove_edges(em: EdgeMap, drop: np.ndarray) -> EdgeMap:
    """em without the edge pixels where drop (same size as em.mask) is True."""
    keep = ~drop[em.uv[:, 1], em.uv[:, 0]]
    labels = np.where(drop, EDGE_NONE, em.labels).astype(np.uint8)
    return EdgeMap(em.level, em.mask & ~drop, labels, em.uv[keep], em.depth[keep], em.label[keep],
                   em.thresholds)


def remove_edges_pyramid(maps: list[EdgeMap], drop: np.ndarray) -> list[EdgeMap]:
    """drop: full-resolution mask. A coarse pixel is dropped if any full-resolution pixel
    under it is (block average > 0), so no object edge survives on a coarse level."""
    out = []
    for em in maps:
        h, w = em.mask.shape
        d = drop if drop.shape == (h, w) else cv2.resize(
            drop.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0
        out.append(remove_edges(em, d))
    return out
