"""Boxes 3-4 as an edge filter for DTVO: leave out the edges of objects that may move.

    detect (YOLO-seg) -> cut out with depth (box 3) -> split edges (box 4)
    -> every detection with prior >= min_prior is masked: its edges never reach the DT
       map or a keyframe

There is no motion check yet (step 8), so "may move" means the class prior alone: a
parked chair is masked like a walking person. Pass an instance as `edge_filter` to
DTVO.process; it returns one (mask, prior) layer per masked object, so DTVO's edge budget
can give the least likely movers back when too few edges remain. After each call it holds
that frame's detections, masks and edge split for logging and the video
(layer_det[j] = index into dets of layer j, matching FrameDebug.kept_layers).
"""
from __future__ import annotations

import time

import numpy as np

from ..common.edges import EdgeConfig, EdgeMap, depth_discontinuities
from ..common.preprocess import PreparedFrame
from .associate import EdgeSplit, split_edges
from .detector import Detection, Detector
from .object_mask import MaskConfig, ObjectMask, cut_out
from .priors import SUSPECT_THRESHOLD, is_suspect


class MovingObjectMask:
    def __init__(self, detector: Detector, min_prior: float = SUSPECT_THRESHOLD,
                 edges: EdgeConfig = EdgeConfig(), mask: MaskConfig = MaskConfig()):
        self.detector, self.min_prior = detector, min_prior
        self.ecfg, self.mcfg = edges, mask
        # last frame
        self.dets: list[Detection] = []
        self.masks: list[ObjectMask] = []
        self.masked: list[bool] = []
        self.split: EdgeSplit | None = None
        self.layer_det: list[int] = []
        self.ms: dict[str, float] = {}

    def __call__(self, frame: PreparedFrame, edges: EdgeMap) -> list[tuple[np.ndarray, float]]:
        t1 = time.perf_counter()
        dets = self.detector(frame.rgb)
        t2 = time.perf_counter()
        self.masked = [d.prior >= self.min_prior for d in dets]
        disc = depth_discontinuities(frame.full.depth, frame.full.valid, self.ecfg, 0) if dets else None
        # only masked / suspect objects get the (slower) depth refinement, as in run_detection
        masks = [cut_out(d, frame.full, disc, self.mcfg, refine=m or is_suspect(d.prior))
                 for d, m in zip(dets, self.masked)]
        t3 = time.perf_counter()
        split = split_edges(edges, dets, masks)
        t4 = time.perf_counter()
        self.layer_det = [i for i, k in enumerate(self.masked) if k]
        self.dets, self.masks, self.split = dets, masks, split
        self.ms = {"detect": 1000 * (t2 - t1), "mask": 1000 * (t3 - t2), "associate": 1000 * (t4 - t3)}
        return [(masks[i].mask, dets[i].prior) for i in self.layer_det]
