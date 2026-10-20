"""Box 1 - PREPARE INPUTS.

Turns a raw colour + depth pair into what the rest of the pipeline uses:
  * grayscale image for edge finding
  * depth in metres, with 0 wherever the reading cannot be trusted
  * a "has depth" mask, so missing depth is never mistaken for a real value
  * lens distortion removed (TUM fr1/fr2, Bonn) so edges sit where the pinhole model expects
  * a 3-level pyramid (640 -> 320 -> 160) for coarse-to-fine tracking
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from .dataset import Camera, RawFrame


@dataclass
class PrepConfig:
    min_depth: float = 0.3      # m; Kinect v1 / Xtion return nothing useful closer than ~0.4-0.5 m
    max_depth: float = 4.5      # m; noise grows ~z^2, beyond this depth is too coarse to trust
    undistort: bool = True
    border_px: int = 5          # ignore this margin: TUM fr3 has a dark last row/column, and
                                # blur/Sobel near any border or undistortion edge make fake edges
    pyramid_levels: int = 3


@dataclass
class FrameLevel:
    level: int
    gray: np.ndarray            # H x W uint8
    depth: np.ndarray           # H x W float32 metres, 0 where invalid
    valid: np.ndarray           # H x W bool, True where depth can be trusted
    inside: np.ndarray          # H x W bool, True on real image content (not border / undistortion fill)
    camera: Camera              # intrinsics for this resolution (no distortion)


@dataclass
class PreparedFrame:
    timestamp: float
    rgb: np.ndarray             # full-resolution undistorted BGR, for display only
    levels: list[FrameLevel] = field(default_factory=list)

    @property
    def full(self) -> FrameLevel:
        return self.levels[0]


def kinect_depth_sigma(z: np.ndarray | float) -> np.ndarray | float:
    """Axial depth noise (m) of Kinect v1 vs distance, Nguyen et al. 2012:
    sigma(z) = 0.0012 + 0.0019 (z - 0.4)^2."""
    return 0.0012 + 0.0019 * (np.asarray(z) - 0.4) ** 2


def depth_jump_threshold(z: np.ndarray | float, k: float = 8.0, floor: float = 0.04):
    """Depth difference (m) treated as a real discontinuity rather than sensor noise."""
    return np.maximum(floor, k * kinect_depth_sigma(z))


@lru_cache(maxsize=8)
def _undistort_maps(camera: Camera):
    K = camera.K
    return cv2.initUndistortRectifyMap(K, np.array(camera.dist), None, K,
                                       (camera.width, camera.height), cv2.CV_32FC1)


def _downsample_depth(depth: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2x2 block average over valid pixels only.

    Blocks that straddle a depth discontinuity are marked invalid instead of
    averaged: averaging foreground and background would invent 'flying'
    points floating between the two surfaces.
    """
    h, w = depth.shape[0] // 2 * 2, depth.shape[1] // 2 * 2
    d = depth[:h, :w].reshape(h // 2, 2, w // 2, 2)
    v = valid[:h, :w].reshape(h // 2, 2, w // 2, 2)
    n = v.sum(axis=(1, 3))
    mean = np.where(n > 0, d.sum(axis=(1, 3)) / np.maximum(n, 1), 0.0)
    dmax = np.where(v, d, -np.inf).max(axis=(1, 3))
    dmin = np.where(v, d, np.inf).min(axis=(1, 3))
    ok = (n > 0) & ((dmax - dmin) <= depth_jump_threshold(mean))
    return np.where(ok, mean, 0.0).astype(np.float32), ok


def prepare_frame(raw: RawFrame, camera: Camera, cfg: PrepConfig = PrepConfig()) -> PreparedFrame:
    rgb, depth_raw = raw.rgb, raw.depth_raw
    if (rgb.shape[1], rgb.shape[0]) != (camera.width, camera.height):
        raise ValueError(f"Image size {rgb.shape[1]}x{rgb.shape[0]} does not match camera "
                         f"{camera.width}x{camera.height}")

    inside = np.ones(rgb.shape[:2], np.uint8)
    if cfg.undistort and camera.has_distortion:
        mx, my = _undistort_maps(camera)
        rgb = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # Nearest neighbour: interpolating depth would blend surfaces across edges.
        # Pixels pulled in from outside the image become 0 (= no depth).
        depth_raw = cv2.remap(depth_raw, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        inside = cv2.remap(inside, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        camera = camera.undistorted()
    b = cfg.border_px
    if b > 0:
        inside[:b], inside[-b:], inside[:, :b], inside[:, -b:] = 0, 0, 0, 0
        inside = cv2.erode(inside, np.ones((2 * b + 1, 2 * b + 1), np.uint8))
    inside = inside > 0

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    depth = depth_raw.astype(np.float32) / camera.depth_scale
    valid = (depth >= cfg.min_depth) & (depth <= cfg.max_depth)
    depth[~valid] = 0.0

    levels = [FrameLevel(0, gray, depth, valid, inside, camera)]
    for lvl in range(1, cfg.pyramid_levels):
        prev = levels[-1]
        h, w = prev.gray.shape[0] // 2 * 2, prev.gray.shape[1] // 2 * 2
        # INTER_AREA at exactly 0.5 is a 2x2 mean -> same pixel grid as the depth pyramid
        g = cv2.resize(prev.gray[:h, :w], (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        d, v = _downsample_depth(prev.depth, prev.valid)
        ins = prev.inside[:h, :w].reshape(h // 2, 2, w // 2, 2).all(axis=(1, 3))
        levels.append(FrameLevel(lvl, g, d, v, ins, prev.camera.downscaled()))

    return PreparedFrame(raw.timestamp, rgb, levels)
