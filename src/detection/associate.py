"""Box 4 - SPLIT EDGES between objects and background.

Every edge pixel gets the index of the object it lies on, or -1 for background.
Where masks overlap (a person sitting on a chair, a cup in a hand) the most suspicious
object wins (highest prior), then the smaller, more specific mask.

Scene tags: the prompt-free model also names whole scenes ("workplace", "mall") with
image-sized masks. A likely-static mask covering more than `scene_area` of the image
is logged but not painted, otherwise it would claim every background edge.

X (the project's dynamic edge ratio) is measured here in two versions:
    X_suspect = edges on suspect objects / all edges     (what step 7 will leave out)
    X_any     = edges on any object / all edges
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.edges import EdgeMap
from .detector import Detection
from .object_mask import ObjectMask
from .priors import is_suspect

BACKGROUND = -1


@dataclass
class EdgeSplit:
    object_map: np.ndarray       # H x W int16, object index per pixel, -1 background
    edge_object: np.ndarray      # N int16, object index per edge pixel (EdgeMap order)
    edges_per_object: np.ndarray  # n_objects int
    scene_tag: np.ndarray        # n_objects bool, logged but not painted
    n_edges: int
    n_suspect_edges: int
    n_object_edges: int

    @property
    def x_suspect(self) -> float:
        return self.n_suspect_edges / max(self.n_edges, 1)

    @property
    def x_any(self) -> float:
        return self.n_object_edges / max(self.n_edges, 1)


def split_edges(edges: EdgeMap, dets: list[Detection], masks: list[ObjectMask],
                scene_area: float = 0.5) -> EdgeSplit:
    h, w = edges.mask.shape
    n = len(dets)
    areas = np.array([m.mask.sum() for m in masks], float)
    priors = np.array([d.prior for d in dets], float)
    suspect = np.array([is_suspect(p) for p in priors], bool)
    scene = (~suspect) & (areas > scene_area * h * w)

    obj = np.full((h, w), BACKGROUND, np.int16)
    # paint the least important first so the winner is painted last:
    # ascending prior, then descending area
    for i in sorted(range(n), key=lambda i: (priors[i], -areas[i])):
        if not scene[i]:
            obj[masks[i].mask] = i

    u, v = edges.uv[:, 0], edges.uv[:, 1]
    eo = obj[v, u]
    counts = np.bincount(eo[eo >= 0], minlength=n)[:n] if n else np.zeros(0, int)
    on_obj = eo >= 0
    on_suspect = on_obj & suspect[np.clip(eo, 0, None)] if n else np.zeros(len(eo), bool)
    return EdgeSplit(obj, eo, counts, scene, len(eo), int(on_suspect.sum()), int(on_obj.sum()))
