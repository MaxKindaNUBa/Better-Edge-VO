"""Box 3 - CUT OUT THE OBJECT USING DEPTH.

The network's mask is a few pixels off the true outline, and edges sit exactly on that
outline, so a sloppy mask leaks the edges that matter most. Depth fixes it: the object
is a connected surface at roughly one distance, separated from the wall behind it by a
depth jump.

    a. widen the box ~10 %                        (detector boxes are often too tight)
    b. enough valid depth in the object?          no  -> fallback
    c. object distance z0 = histogram peak of depth inside the network mask
    d. keep pixels near z0 that are not on a depth jump
    e. keep the connected pieces that overlap the network mask
       (and stay within a few px of it: stops leaks along the floor at the feet)
    f. sanity check: piece too small?             yes -> fallback
    g. add network-mask pixels that have no depth (hair, dark clothes)
    h. grow the outline by a margin (colour / depth images are a few px out of line)

    fallback: the network mask grown by the margin (or the widened box without a mask)

Only suspects (prior >= SUSPECT_THRESHOLD) are refined; likely-static objects keep the
network mask - their exact outline does not matter until the motion check (step 8).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..common.preprocess import FrameLevel, kinect_depth_sigma
from .detector import Detection

SRC_DEPTH = "depth"                 # refined with depth (normal case)
SRC_LOW_DEPTH = "fallback_low_depth"
SRC_SMALL = "fallback_small"
SRC_NETWORK = "network"             # not refined (likely-static object)
SRC_BOX = "box"                     # no network mask and no usable depth


@dataclass
class MaskConfig:
    widen: float = 0.10             # a. fraction of box size added on every side
    min_depth_coverage: float = 0.4  # b. share of mask pixels that need valid depth
    bin_m: float = 0.05             # c. histogram bin for the object distance
    extent_m: float = 0.6           # d. object may reach this far in front of / behind z0
    extent_sigma: float = 4.0       #    ...or this many sensor-noise sigmas, if larger
    core_erode_px: int = 3          # e. network-mask core used as the seed
    leash_px: int = 15              # e. refined mask may grow at most this far past the network mask
    min_keep_ratio: float = 0.3     # f. refined area must be >= this share of the network mask
    close_px: int = 5               # g. small morphological close of the result
    margin_px: int = 4              # h. final outline growth


@dataclass
class ObjectMask:
    mask: np.ndarray                # H x W bool, final
    source: str                     # SRC_*
    z0: float                       # object distance (m), 0 if unknown
    depth_coverage: float


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _grow(mask: np.ndarray, r: int) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), _disk(r)) > 0 if r > 0 else mask


def widen_box(box, w: int, h: int, frac: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    dx, dy = frac * (x2 - x1), frac * (y2 - y1)
    return (int(max(0, x1 - dx)), int(max(0, y1 - dy)),
            int(min(w, np.ceil(x2 + dx))), int(min(h, np.ceil(y2 + dy))))


def object_distance(z: np.ndarray, bin_m: float) -> float:
    """Most common distance (histogram peak), refined by the median around the peak.
    A median alone fails when background shows through the mask (between legs, arms)."""
    if len(z) == 0:
        return 0.0
    edges = np.arange(z.min(), z.max() + 2 * bin_m, bin_m)
    hist, edges = np.histogram(z, edges)
    peak = edges[np.argmax(hist)] + bin_m / 2
    near = z[np.abs(z - peak) <= 3 * bin_m]
    return float(np.median(near)) if len(near) else float(peak)


def cut_out(det: Detection, frame: FrameLevel, disc: np.ndarray,
            cfg: MaskConfig = MaskConfig(), refine: bool = True) -> ObjectMask:
    """frame: full-resolution level; disc: depth-discontinuity pixels of that frame."""
    h, w = frame.depth.shape
    x1, y1, x2, y2 = widen_box(det.box, w, h, cfg.widen)
    roi = (slice(y1, y2), slice(x1, x2))

    if det.mask is not None:
        net = det.mask
    else:   # box-only detector: treat the (unwidened) box as the rough mask
        net = np.zeros((h, w), bool)
        bx1, by1, bx2, by2 = det.box
        net[by1:by2, bx1:bx2] = True
    fallback = _grow(net, cfg.margin_px)
    n_net = int(net.sum())
    coverage = float(frame.valid[net].mean()) if n_net else 0.0

    if not refine:
        return ObjectMask(fallback, SRC_NETWORK if det.mask is not None else SRC_BOX, 0.0, coverage)
    if n_net == 0 or coverage < cfg.min_depth_coverage:
        return ObjectMask(fallback, SRC_LOW_DEPTH, 0.0, coverage)

    core = cv2.erode(net.astype(np.uint8), _disk(cfg.core_erode_px)) > 0
    core &= frame.valid
    if core.sum() < 20:
        core = net & frame.valid
    z0 = object_distance(frame.depth[core], cfg.bin_m)

    z, valid = frame.depth[roi], frame.valid[roi]
    extent = max(cfg.extent_m, cfg.extent_sigma * float(kinect_depth_sigma(z0)))
    band = valid & ~disc[roi] & (np.abs(z - z0) <= extent)

    n, lab = cv2.connectedComponents(band.astype(np.uint8), connectivity=8)
    seed = core[roi]
    hits = np.bincount(lab[seed], minlength=n)
    sizes = np.bincount(lab.ravel(), minlength=n)
    keep = (hits >= np.maximum(30, 0.05 * sizes))
    keep[0] = False
    grown_roi = keep[lab] & _grow(net, cfg.leash_px)[roi]

    if grown_roi.sum() < cfg.min_keep_ratio * n_net:
        return ObjectMask(fallback, SRC_SMALL, z0, coverage)

    grown_roi |= net[roi] & ~valid                       # g. depth holes inside the object
    grown_roi = cv2.morphologyEx(grown_roi.astype(np.uint8), cv2.MORPH_CLOSE,
                                 _disk(cfg.close_px)) > 0
    out = np.zeros((h, w), bool)
    out[roi] = grown_roi
    return ObjectMask(_grow(out, cfg.margin_px), SRC_DEPTH, z0, coverage)
