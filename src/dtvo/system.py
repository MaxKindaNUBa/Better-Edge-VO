"""DTVO system (Yan, Li, Zhou 2022): tracking + local mapping, RGB-D mode (Sec. 4.5).

Per frame:
  box 1  prepare_frame            undistort, depth in metres, 3-level pyramid
  box 2  detect_edges_pyramid     adaptive Canny, every edge kept
         edge_filter (optional)   drop the edges of moving objects (run_dynamic.py); without
                                  it this is plain DTVO
         build_dt_pyramid         distance-transform map of the current frame
  track  4 motion hypotheses -> weighted coarse-to-fine DT alignment to the reference keyframe
  KF?    Sec. 5.1: > 20 frames, or translation / rotation threshold, or an edge type lost
         too many of its matches -> the frame becomes a keyframe, but only if it has enough
         edges with depth, spread over the image, to hold its position (not in the paper;
         otherwise the old keyframe is kept)
  map    depth propagation + sliding-window optimisation + marginalisation

The paper runs tracking and mapping in two threads; here they run one after the other
(the window optimisation finishes before the next frame is tracked), which makes runs
deterministic and gives the result the threaded version converges to.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..common.dataset import Camera, RawFrame
from ..common.edges import EDGE_DEPTH, EdgeConfig, EdgeMap, detect_edges_pyramid, remove_edges_pyramid
from ..common.preprocess import PrepConfig, PreparedFrame, prepare_frame
from .local_mapping import Keyframe, LocalMapper, MappingConfig
from .tracking import N_TYPES, TrackConfig, build_dt_pyramid, track
from .se3 import inv, orthonormalize, rot_angle_deg, scale_motion


@dataclass
class KeyframeConfig:
    max_frames: int = 20            # "more than 20 frames have passed"
    max_translation: float = 0.10   # m
    max_rotation_deg: float = 8.0
    type_keep_ratio: float = 0.55   # new KF if a main edge type keeps < this share of its matches
    type_min_count: int = 150       # ...only for types with enough reference pixels to judge
    max_lost_frames: int = 5        # after this many lost frames in a row, re-anchor a keyframe
    suspect_ratio: float = 0.6      # a frame matching fewer edges than this...
    suspect_jump_m: float = 0.10    # ...whose pose jumps this far from the motion model...
    suspect_jump_deg: float = 15.0  # ...(or turns this much) is treated as lost
    # Keyframe support (not in the paper): a keyframe with few edges is barely held in place by
    # the window optimisation and can slide by metres. Unmasked frames never go below ~1,600 /
    # ~150 depth edges (TUM structure_notexture, walking_xyz); some leave a quadrant empty.
    min_ref_points: int = 1000      # level-0 edges with depth
    min_top_points: int = 80        # coarsest-level edges with depth
    min_cells: int = 2              # image quadrants holding >= cell_share of those edges
    cell_share: float = 0.05
    unmask_below_prior: float = 0.7  # edge budget: masked objects below this prior may be given back


@dataclass
class DTVOConfig:
    prep: PrepConfig = field(default_factory=PrepConfig)
    # Lower Canny floor than the edge demo: in low-texture scenes (fr3 structure_notexture)
    # the faint wall folds are the only edges besides the floor line, and with that line
    # alone the pose is unobservable (248 m ATE at 60, 1.5 cm at 25; textured scenes unchanged).
    edges: EdgeConfig = field(default_factory=lambda: EdgeConfig(min_high=25.0))
    track: TrackConfig = field(default_factory=TrackConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    keyframe: KeyframeConfig = field(default_factory=KeyframeConfig)
    seed: int = 0


@dataclass
class FrameRecord:
    timestamp: float
    kf_id: int
    T_kc: np.ndarray          # current camera -> keyframe
    inlier_ratio: float
    hypothesis: str
    lost: bool
    is_keyframe: bool
    n_edges: int              # edges the tracker used (after the edge filter)
    n_masked: int = 0         # edges removed by the edge filter
    n_ref_points: int = 0     # level-0 edges with depth (keyframe support)
    n_unmasked: int = 0       # masked objects given back by the edge budget
    kf_refused: bool = False  # a keyframe was due, but the frame had too few edges to anchor it
    weak_kf: bool = False     # keyframe re-anchored after a lost run (pose from the motion model)
    ba_rejected: bool = False  # this keyframe's window optimisation moved a keyframe too far: undone


@dataclass
class FrameDebug:
    """What the last processed frame looked like to the tracker (for visualisation)."""
    rgb: np.ndarray               # undistorted colour image
    edge_mask: np.ndarray         # all detected edges, full resolution
    ref_P: np.ndarray | None      # reference keyframe edge points (keyframe frame), None on frame 0
    T_cr: np.ndarray | None       # tracked reference -> current transform
    camera: Camera
    inlier_ratio: float
    lost: bool
    is_keyframe: bool
    dropped: np.ndarray | None = None   # edges removed by the edge filter, full resolution
    kept_layers: list[int] = field(default_factory=list)   # filter masks applied (not given back)


# (prepared frame, full-resolution edges) -> [(H x W bool mask, prior)] of objects whose edges
# are left out; the prior orders which ones the edge budget may give back first
EdgeFilter = Callable[[PreparedFrame, EdgeMap], "list[tuple[np.ndarray, float]]"]


class DTVO:
    def __init__(self, camera: Camera, cfg: DTVOConfig = DTVOConfig()):
        self.camera, self.cfg = camera, cfg
        self.mapper = LocalMapper(cfg.mapping)
        self.keyframes: dict[int, Keyframe] = {}
        self.kf: Keyframe | None = None
        self.records: list[FrameRecord] = []
        self.T_prev = self.T_prev2 = None
        self.rng = np.random.default_rng(cfg.seed)
        self.times = {"prepare": 0.0, "edges+DT": 0.0, "tracking": 0.0, "mapping": 0.0}
        self.frame_index = 0
        self.lost_run = 0
        self.debug: FrameDebug | None = None
        self.T_good, self.motion_good = np.eye(4), np.eye(4)   # last frame not flagged lost

    # -------------------------------------------------------------------------
    def _new_keyframe(self, T_wc, frame, edge_maps, dts) -> Keyframe:
        kf = Keyframe(len(self.keyframes), self.frame_index, T_wc, frame, edge_maps, dts,
                      self.cfg.mapping, self.cfg.track, self.rng)
        self.keyframes[kf.id] = kf
        t0 = time.perf_counter()
        self.mapper.add_keyframe(kf)
        self.times["mapping"] += time.perf_counter() - t0
        self.kf = kf
        return kf

    def _need_keyframe(self, res, frames_since: int) -> bool:
        c = self.cfg.keyframe
        if frames_since > c.max_frames:
            return True
        T = res.T_cr
        if np.linalg.norm(T[:3, 3]) > c.max_translation or rot_angle_deg(T[:3, :3]) > c.max_rotation_deg:
            return True
        ref = self.kf.type_counts[:N_TYPES - 1]
        big = ref >= c.type_min_count
        return bool(np.any(res.type_inliers[:N_TYPES - 1][big] < c.type_keep_ratio * ref[big]))

    def _support(self, edge_maps: list[EdgeMap]) -> tuple[int, int, int]:
        """(level-0 edges with depth, coarsest-level edges with depth, image quadrants holding
        at least cell_share of the level-0 ones): what a keyframe built from these edges has."""
        e0 = edge_maps[0]
        uv = e0.uv[e0.label == EDGE_DEPTH]
        h, w = e0.mask.shape
        per = np.bincount((uv[:, 0] >= w // 2) + 2 * (uv[:, 1] >= h // 2), minlength=4)
        cells = int((per >= max(1, self.cfg.keyframe.cell_share * len(uv))).sum())
        return len(uv), int((edge_maps[-1].label == EDGE_DEPTH).sum()), cells

    def _can_anchor(self, s: tuple[int, int, int]) -> bool:
        c = self.cfg.keyframe
        return s[0] >= c.min_ref_points and s[1] >= c.min_top_points and s[2] >= c.min_cells

    def _filter_edges(self, frame, edge_maps, edge_filter):
        """Apply the edge filter, then the edge budget: while the frame has too few edges to
        anchor a keyframe, give back the least likely movers (lowest prior first; objects at or
        above unmask_below_prior, e.g. people, are never given back)."""
        layers = edge_filter(frame, edge_maps[0]) or []
        kept = list(range(len(layers)))

        def apply(kept):
            if not kept:
                return edge_maps, None
            drop = np.zeros(edge_maps[0].mask.shape, bool)
            for i in kept:
                drop |= layers[i][0]
            return remove_edges_pyramid(edge_maps, drop), drop

        filtered, drop = apply(kept)
        support = self._support(filtered)
        for i in sorted(kept, key=lambda i: layers[i][1]):
            if self._can_anchor(support) or layers[i][1] >= self.cfg.keyframe.unmask_below_prior:
                break
            kept.remove(i)
            filtered, drop = apply(kept)
            support = self._support(filtered)
        dropped = None if drop is None else edge_maps[0].mask & drop
        return filtered, dropped, kept, len(layers) - len(kept), support

    def process(self, raw: RawFrame, edge_filter: EdgeFilter | None = None) -> np.ndarray:
        """Track one frame; returns its current world pose estimate T_wc.

        edge_filter: edges under the masks it returns are removed on every pyramid level
        before the DT map is built, so they are neither tracked against nor become
        keyframe reference points (subject to the edge budget, see _filter_edges)."""
        t0 = time.perf_counter()
        frame = prepare_frame(raw, self.camera, self.cfg.prep)
        t1 = time.perf_counter()
        edge_maps = detect_edges_pyramid(frame, self.cfg.edges)
        dropped, kept, n_unmasked, t_filter = None, [], 0, 0.0
        if edge_filter is not None:
            tf = time.perf_counter()
            edge_maps, dropped, kept, n_unmasked, support = self._filter_edges(frame, edge_maps, edge_filter)
            t_filter = time.perf_counter() - tf
            self.times["edge filter"] = self.times.get("edge filter", 0.0) + t_filter
        else:
            support = self._support(edge_maps)
        n_masked = 0 if dropped is None else int(dropped.sum())
        can_anchor = self._can_anchor(support)
        dts = build_dt_pyramid(edge_maps, frame.levels)
        t2 = time.perf_counter()
        self.times["prepare"] += t1 - t0
        self.times["edges+DT"] += t2 - t1 - t_filter
        n_edges = len(edge_maps[0].uv)

        def record(kf_id, T_kc, ratio, hyp, lost, is_kf, refused=False, weak=False, ba_rejected=False):
            self.records.append(FrameRecord(frame.timestamp, kf_id, T_kc, ratio, hyp, lost, is_kf, n_edges,
                                            n_masked, support[0], n_unmasked, refused, weak, ba_rejected))

        if self.kf is None:  # RGB-D: no initialisation needed, first frame = first keyframe
            kf = self._new_keyframe(np.eye(4), frame, edge_maps, dts)
            record(kf.id, np.eye(4), 1.0, "init", False, True, weak=not can_anchor)
            self.debug = FrameDebug(frame.rgb, edge_maps[0].mask, None, None, frame.levels[0].camera,
                                    1.0, False, True, dropped, kept)
            self.T_prev = np.eye(4)
            self.frame_index += 1
            return self.T_prev

        # Sec. 4.2: constant / no / half / double motion hypotheses
        prev, prev2 = self.T_prev, self.T_prev2 if self.T_prev2 is not None else self.T_prev
        motion = inv(prev2) @ prev
        T_kw = self.kf.T_wk
        hyps = {name: inv(prev @ scale_motion(motion, s)) @ T_kw
                for name, s in (("constant", 1.0), ("none", 0.0), ("half", 0.5), ("double", 2.0))}
        t3 = time.perf_counter()
        res = track(self.kf.ref_levels, dts, hyps, self.cfg.track)
        self.times["tracking"] += time.perf_counter() - t3
        T_wc = orthonormalize(T_kw @ inv(res.T_cr))
        self.debug = FrameDebug(frame.rgb, edge_maps[0].mask, self.kf.ref0.P, res.T_cr,
                                frame.levels[0].camera, res.inlier_ratio, False, False, dropped, kept)

        # Lost-frame guard (the paper stops the system instead): a frame that matches too few
        # reference edges gets the constant-motion pose and is never turned into a keyframe,
        # so one bad alignment cannot become the reference for everything after it. "Too few"
        # = few matches AND a pose far from a constant-motion extrapolation of the last good
        # frame (fast but correct motion also lowers the match ratio). The tolerance grows
        # with every held frame, since that extrapolation gets less certain.
        c = self.cfg.keyframe
        k = self.lost_run + 1
        jump = inv(self.T_good @ scale_motion(self.motion_good, k)) @ T_wc
        lost = res.lost or (res.inlier_ratio < c.suspect_ratio and (
            np.linalg.norm(jump[:3, 3]) > k * c.suspect_jump_m or rot_angle_deg(jump[:3, :3]) > k * c.suspect_jump_deg))
        self.lost_run = self.lost_run + 1 if lost else 0
        self.debug.lost = lost
        coasted = orthonormalize(prev @ motion)
        # Keep coasting on the motion model while lost, and also after max_lost_frames for as
        # long as this frame has too few edges to anchor a keyframe (it would be as weak).
        if lost and (self.lost_run <= c.max_lost_frames or not can_anchor):
            record(self.kf.id, orthonormalize(inv(T_kw) @ coasted), res.inlier_ratio, res.hypothesis, True,
                   False, refused=self.lost_run > c.max_lost_frames)
            self.T_prev2, self.T_prev = self.T_prev, coasted
            self.frame_index += 1
            return coasted

        frames_since = self.frame_index - self.kf.frame_index
        want_kf = lost or self._need_keyframe(res, frames_since)
        if want_kf and can_anchor:
            # Re-anchoring after a lost run: start from the coasted pose (what the lost frames
            # got), not from the alignment that was just judged unreliable -> no jump.
            kf = self._new_keyframe(coasted if lost else T_wc, frame, edge_maps, dts)
            kf.weak = lost
            self.debug.is_keyframe = True
            T_wc = kf.T_wk  # refined by the window optimisation (unless the step was rejected)
            record(kf.id, np.eye(4), res.inlier_ratio, res.hypothesis, lost, True, weak=lost,
                   ba_rejected=self.mapper.last_rejected)
        else:
            # A keyframe was due but this frame cannot anchor one: keep tracking the old keyframe
            record(self.kf.id, orthonormalize(inv(res.T_cr)), res.inlier_ratio, res.hypothesis, res.lost, False,
                   refused=want_kf)
        self.motion_good = inv(self.T_prev) @ T_wc if self.lost_run == 0 else np.eye(4)
        self.T_good = T_wc
        self.T_prev2, self.T_prev = self.T_prev, T_wc
        self.frame_index += 1
        return T_wc

    def trajectory(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """Final poses: every frame expressed through its (window-optimised) keyframe."""
        stamps = np.array([r.timestamp for r in self.records])
        poses = [self.keyframes[r.kf_id].T_wk @ r.T_kc for r in self.records]
        return stamps, poses
