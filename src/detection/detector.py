"""Object detector front-end: one class, three interchangeable models.

    coco        YOLO11s-seg, COCO weights, 80 fixed classes
    yoloe_text  YOLOE-11s-seg, open vocabulary, told what to look for (TEXT_PROMPT)
    yoloe_pf    YOLOE-11s-seg-pf, prompt-free, names objects from a built-in
                4,585-name vocabulary

Every mode returns the same Detection records (box, class, score, rough mask, prior,
track ID), so everything downstream is detector-agnostic. Track IDs come from our
own IoU tracker rather than Ultralytics' trackers: ByteTrack returned no IDs for
YOLOE, and one tracker for all three modes keeps the comparison fair.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .priors import prior_for

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

MODES = ("coco", "yoloe_text", "yoloe_pf")
MODE_LABELS = {"coco": "COCO YOLO11s-seg", "yoloe_text": "YOLOE text prompt",
               "yoloe_pf": "YOLOE prompt-free"}

# Things that can move in indoor RGB-D sequences (people, what they carry or push).
TEXT_PROMPT = ["person", "chair", "office chair", "cardboard box", "box", "balloon", "bag",
               "backpack", "suitcase", "ball", "umbrella", "door", "cart", "dog", "cat"]


@dataclass
class Detection:
    box: tuple[int, int, int, int]      # x1, y1, x2, y2 (pixels, full-resolution undistorted image)
    class_name: str
    score: float
    mask: np.ndarray | None             # H x W bool, the network's own outline (rough)
    prior: float                        # P(moving) from priors.py
    track_id: int = -1


@dataclass
class DetectorConfig:
    conf: float = 0.25                  # Ultralytics' default score threshold
    iou: float = 0.7                    # NMS overlap
    imgsz: int = 640
    device: str | int = 0               # GPU 0; "cpu" also works


class Detector:
    def __init__(self, mode: str, cfg: DetectorConfig = DetectorConfig()):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode, self.cfg = mode, cfg
        import os
        from ultralytics import YOLO, YOLOE
        cwd = os.getcwd()
        os.chdir(MODELS_DIR)   # Ultralytics downloads weights / the CLIP text encoder into the cwd
        try:
            if mode == "coco":
                self.model = YOLO("yolo11s-seg.pt")
            elif mode == "yoloe_text":
                self.model = YOLOE("yoloe-11s-seg.pt")
                self.model.set_classes(TEXT_PROMPT)
            else:
                self.model = YOLOE("yoloe-11s-seg-pf.pt")
        finally:
            os.chdir(cwd)
        self.tracker = IoUTracker()

    def __call__(self, bgr: np.ndarray) -> list[Detection]:
        r = self.model.predict(bgr, conf=self.cfg.conf, iou=self.cfg.iou, imgsz=self.cfg.imgsz,
                               device=self.cfg.device, retina_masks=True, verbose=False)[0]
        dets = []
        if r.boxes is not None and len(r.boxes):
            boxes = r.boxes.xyxy.cpu().numpy().round().astype(int)
            cls = r.boxes.cls.cpu().numpy().astype(int)
            score = r.boxes.conf.cpu().numpy()
            masks = r.masks.data.cpu().numpy() > 0.5 if r.masks is not None else None
            for i in range(len(boxes)):
                name = r.names[int(cls[i])]
                dets.append(Detection(tuple(int(v) for v in boxes[i]), name, float(score[i]),
                                      None if masks is None else masks[i], prior_for(name)))
        self.tracker.update(dets)
        return dets


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between N x 4 and M x 4 boxes (x1, y1, x2, y2)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


@dataclass
class _Track:
    tid: int
    box: np.ndarray
    missed: int = 0
    hits: int = 1


@dataclass
class IoUTracker:
    """Frame-to-frame box matching (Hungarian on IoU). Class names are ignored on purpose:
    the prompt-free model calls the same person "manager" in one frame and "researcher"
    in the next. A track survives `max_missed` frames without a match."""
    min_iou: float = 0.3
    max_missed: int = 10
    tracks: list[_Track] = field(default_factory=list)
    next_id: int = 1
    created: int = 0

    def update(self, dets: list[Detection]):
        boxes = np.array([d.box for d in dets], float).reshape(-1, 4)
        old = np.array([t.box for t in self.tracks], float).reshape(-1, 4)
        iou = box_iou(old, boxes)
        matched_t, matched_d = set(), set()
        if iou.size:
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.min_iou:
                    t = self.tracks[r]
                    t.box, t.missed, t.hits = boxes[c], 0, t.hits + 1
                    dets[c].track_id = t.tid
                    matched_t.add(r)
                    matched_d.add(c)
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        for c, d in enumerate(dets):
            if c not in matched_d:
                self.tracks.append(_Track(self.next_id, boxes[c]))
                d.track_id = self.next_id
                self.next_id += 1
                self.created += 1
