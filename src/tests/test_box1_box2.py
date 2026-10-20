"""Synthetic checks for boxes 1-2.  Run from the project root:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n 312ml python -m pytest src/tests -q
(autoload must be off: a system ROS 'launch_testing' plugin crashes pytest 9.)"""
import numpy as np

from src.common.dataset import CAMERAS, RawFrame, associate
from src.common.edges import EDGE_DEPTH, EDGE_DEPTH_JUMP, EDGE_NO_DEPTH, detect_edges
from src.common.preprocess import PrepConfig, prepare_frame

CAM = CAMERAS["tum_fr3"]


def make_frame(rgb_fn, depth_fn):
    v, u = np.mgrid[0:480, 0:640]
    gray = rgb_fn(u, v).astype(np.uint8)
    depth = (depth_fn(u, v) * 5000).astype(np.uint16)
    return RawFrame(0.0, 0.0, np.dstack([gray] * 3), depth)


def test_associate_one_to_one_and_max_dt():
    a = np.array([0.00, 0.033, 0.066, 0.50])
    b = np.array([0.005, 0.030, 0.031, 0.070])
    assert associate(a, b, 0.02) == [(0, 0), (1, 2), (2, 3)]  # 0.50 has no partner


def test_depth_metres_and_invalid():
    raw = make_frame(lambda u, v: u * 0 + 100, lambda u, v: np.where(u < 320, 2.0, 0.0))
    f = prepare_frame(raw, CAM).full
    assert np.allclose(f.depth[:, :320], 2.0)
    assert not f.valid[:, 320:].any() and (f.depth[:, 320:] == 0).all()
    raw.depth_raw[:] = 6 * 5000  # beyond max_depth -> untrusted
    assert not prepare_frame(raw, CAM).full.valid.any()


def test_pyramid_intrinsics_track_the_same_point():
    # bright dot at full-res pixel (401, 301) -> 2x2 block (200, 150) at level 1
    raw = make_frame(lambda u, v: np.where((abs(u - 401) <= 1) & (abs(v - 301) <= 1), 255, 0),
                     lambda u, v: u * 0 + 2.0)
    f = prepare_frame(raw, CAM, PrepConfig(pyramid_levels=3))
    X = np.array([(401 - CAM.cx) / CAM.fx, (301 - CAM.cy) / CAM.fy, 1.0])  # ray through the dot
    for lvl in f.levels:
        c = lvl.camera
        u, v = c.fx * X[0] + c.cx, c.fy * X[1] + c.cy
        w = lvl.gray.astype(float)
        vv, uu = np.mgrid[0:w.shape[0], 0:w.shape[1]]
        cu, cv = (w * uu).sum() / w.sum(), (w * vv).sum() / w.sum()  # brightness-weighted centre
        assert abs(cu - u) < 0.2 and abs(cv - v) < 0.2, (lvl.level, cu, cv, u, v)


def test_texture_edge_on_flat_surface_keeps_depth():
    # brightness step on a flat wall at 2 m: a texture edge, fully trackable
    raw = make_frame(lambda u, v: np.where(u < 320, 40, 200), lambda u, v: u * 0 + 2.0)
    em = detect_edges(prepare_frame(raw, CAM).full)
    assert em.counts()["total"] > 400
    assert (em.label == EDGE_DEPTH).all()
    assert np.allclose(em.depth, 2.0)


def test_occluding_outline_is_depth_jump_and_hole_is_no_depth():
    # object at 1 m in front of a wall at 3 m, outline at u = 320; no depth for v < 100
    raw = make_frame(lambda u, v: np.where(u < 320, 40, 200),
                     lambda u, v: np.where(v < 100, 0.0, np.where(u < 320, 1.0, 3.0)))
    em = detect_edges(prepare_frame(raw, CAM).full)
    rows = em.uv[:, 1]
    assert (em.label[rows < 95] == EDGE_NO_DEPTH).all()
    assert (em.label[rows > 105] == EDGE_DEPTH_JUMP).all()


def test_no_edges_at_image_border():
    # dark last row/column like real TUM fr3 frames
    raw = make_frame(lambda u, v: np.where((u == 639) | (v == 479), 30, 120), lambda u, v: u * 0 + 2.0)
    assert detect_edges(prepare_frame(raw, CAM).full).counts()["total"] == 0


def test_slanted_surface_is_not_a_depth_jump():
    # texture edge on a steeply slanted wall (1 cm of depth per pixel): no outline here
    raw = make_frame(lambda u, v: np.where(u < 320, 40, 200), lambda u, v: 1.0 + 0.01 * u)
    em = detect_edges(prepare_frame(raw, CAM).full)
    assert em.counts()["total"] > 400
    assert (em.label == EDGE_DEPTH).all()


def test_jump_found_despite_colour_depth_misalignment():
    # Kinect "fattening": depth outline 4 px outside the colour outline
    raw = make_frame(lambda u, v: np.where(u < 320, 40, 200),
                     lambda u, v: np.where(u < 324, 1.0, 3.0))
    em = detect_edges(prepare_frame(raw, CAM).full)
    assert (em.label == EDGE_DEPTH_JUMP).mean() > 0.95


def test_single_pixel_depth_spike_is_ignored():
    # lone depth outliers (sensor noise) next to a texture edge must not become a jump
    def depth(u, v):
        z = np.full(u.shape, 2.0)
        z[(u % 40 == 322 % 40) & (v % 40 == 0)] = 1.7
        return z
    raw = make_frame(lambda u, v: np.where(u < 320, 40, 200), depth)
    em = detect_edges(prepare_frame(raw, CAM).full)
    assert (em.label == EDGE_DEPTH).all()
