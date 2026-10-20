"""Synthetic checks for boxes 3-4 (no network needed).  Run from the project root:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n 312ml python -m pytest src/tests -q"""
import numpy as np

from src.common.dataset import CAMERAS, RawFrame
from src.common.edges import EdgeConfig, depth_discontinuities, detect_edges
from src.common.preprocess import PrepConfig, prepare_frame
from src.detection.associate import split_edges
from src.detection.detector import Detection, IoUTracker, box_iou
from src.detection.object_mask import SRC_DEPTH, SRC_LOW_DEPTH, SRC_NETWORK, cut_out
from src.detection.priors import DEFAULT_PRIOR, is_suspect, prior_for

CAM = CAMERAS["tum_fr3"]
BOX = (200, 150, 360, 400)          # textured slab at 1 m in front of a wall at 3 m


def scene(slab_depth=1.0, hole=False):
    v, u = np.mgrid[0:480, 0:640]
    on = (u >= BOX[0]) & (u < BOX[2]) & (v >= BOX[1]) & (v < BOX[3])
    gray = np.where(on, 60 + 120 * ((u // 16 + v // 16) % 2), 140 + 60 * ((u // 40) % 2))
    depth = np.where(on, slab_depth, 3.0)
    if hole:
        depth[on] = 0.0
    raw = RawFrame(0.0, 0.0, np.dstack([gray.astype(np.uint8)] * 3), (depth * 5000).astype(np.uint16))
    return prepare_frame(raw, CAM, PrepConfig(pyramid_levels=1)).full, on


def network_mask(on, pad=12):
    """A sloppy network mask: the true slab grown by `pad` px (leaks onto the wall)."""
    import cv2
    return cv2.dilate(on.astype(np.uint8), np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)) > 0


def test_priors_never_zero_and_people_words():
    assert prior_for("person") >= 0.8 and prior_for("manager") >= 0.8 and prior_for("Polo Shirt") >= 0.7
    assert prior_for("crt screen") == DEFAULT_PRIOR > 0
    assert is_suspect(prior_for("chair")) and not is_suspect(prior_for("monitor"))


def test_cut_out_removes_wall_leak():
    f, on = scene()
    disc = depth_discontinuities(f.depth, f.valid, EdgeConfig(), 0)
    det = Detection(BOX, "person", 0.9, network_mask(on), prior_for("person"))
    m = cut_out(det, f, disc)
    assert m.source == SRC_DEPTH and abs(m.z0 - 1.0) < 0.05
    # covers the slab (plus the 4 px margin) but not the 12 px leak onto the wall
    assert m.mask[on].mean() > 0.99
    wall = ~network_mask(on, 6)
    assert m.mask[wall].sum() == 0


def test_cut_out_falls_back_without_depth_and_skips_static():
    f, on = scene(hole=True)
    disc = depth_discontinuities(f.depth, f.valid, EdgeConfig(), 0)
    det = Detection(BOX, "person", 0.9, network_mask(on, 2), 0.8)
    assert cut_out(det, f, disc).source == SRC_LOW_DEPTH
    assert cut_out(det, f, disc, refine=False).source == SRC_NETWORK


def test_split_edges_priority_and_scene_tags():
    f, on = scene()
    em = detect_edges(f, EdgeConfig(min_high=25.0))
    disc = depth_discontinuities(f.depth, f.valid, EdgeConfig(), 0)
    whole = np.ones_like(on)
    dets = [Detection((0, 0, 640, 480), "workplace", 0.5, whole, prior_for("workplace")),
            Detection(BOX, "chair", 0.6, network_mask(on, 2), prior_for("chair")),
            Detection(BOX, "person", 0.9, network_mask(on, 2), prior_for("person"))]
    masks = [cut_out(d, f, disc, refine=is_suspect(d.prior)) for d in dets]
    s = split_edges(em, dets, masks)
    assert s.scene_tag.tolist() == [True, False, False]       # image-sized static tag not painted
    slab_edges = on[em.uv[:, 1], em.uv[:, 0]]
    assert (s.edge_object[slab_edges] == 2).all()             # person beats chair on overlap
    assert (s.edge_object[~network_mask(on, 8)[em.uv[:, 1], em.uv[:, 0]]] == -1).all()
    assert 0 < s.x_suspect == s.x_any < 1


def test_tracker_keeps_id_across_label_change():
    tr = IoUTracker()
    a = [Detection((10, 10, 100, 200), "manager", 0.7, None, 0.8)]
    tr.update(a)
    b = [Detection((14, 12, 104, 204), "researcher", 0.6, None, 0.8)]
    tr.update(b)
    assert a[0].track_id == b[0].track_id == 1
    assert box_iou(np.array([[0, 0, 10, 10]]), np.array([[5, 0, 15, 10]]))[0, 0] == 1 / 3
