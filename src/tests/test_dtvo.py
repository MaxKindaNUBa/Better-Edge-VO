"""Synthetic checks for the DTVO modules. Run from the project root:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n 312ml python -m pytest src/tests -q"""
import cv2
import numpy as np

from src.common.dataset import CAMERAS, RawFrame
from src.dtvo.tracking import (TYPE_DIAG_ANTI, TYPE_DIAG_MAIN, TYPE_HORIZONTAL, TYPE_OTHER, TYPE_VERTICAL,
                             TrackConfig, build_dt_pyramid, edge_types, make_ref_level, track, type_weights)
from src.common.edges import detect_edges_pyramid
from src.dtvo.evaluation import evaluate, umeyama_rigid
from src.common.preprocess import prepare_frame
from src.dtvo.se3 import inv, rot_angle_deg, se3_exp, se3_log

CAM = CAMERAS["tum_fr3"]


def test_se3_roundtrip():
    xi = np.array([0.1, -0.2, 0.05, 0.3, -0.1, 0.2])
    assert np.allclose(se3_log(se3_exp(xi)), xi)
    T = se3_exp(xi)
    assert np.allclose(inv(T) @ T, np.eye(4))


def test_edge_types_fig5():
    m = np.zeros((9, 9), bool)
    m[2, 1:8] = True          # horizontal run
    t = edge_types(m)
    assert t[2, 4] == TYPE_HORIZONTAL and t[2, 1] == TYPE_OTHER  # line end -> type 4
    assert edge_types(m.T)[4, 2] == TYPE_VERTICAL
    d = np.eye(9, dtype=bool)
    assert edge_types(d)[4, 4] == TYPE_DIAG_MAIN
    assert edge_types(d[:, ::-1])[4, 4] == TYPE_DIAG_ANTI
    assert t[0, 0] == -1


def test_type_weights_eq8():
    types = np.array([0] * 80 + [1] * 15 + [4] * 5)
    assert np.allclose(type_weights(types, 2, 2), 1.0)        # k = l: all weights 1
    w0 = type_weights(types, 0, 2, clip=(0.0, 100.0))         # k = 0: log(N_sum / N_t)
    assert np.isclose(w0[0], np.log(100 / 80)) and np.isclose(w0[-1], np.log(100 / 5))
    assert w0[-1] > w0[0]                                      # rare types weigh more at fine levels


def _plane_frame(R, t, tex):
    """Render a textured fronto-parallel plane (z = 2 m in world) seen from pose (R, t) = T_cw."""
    K = CAM.K
    n, d = np.array([0, 0, 1.0]), 2.0
    # Homography from the world-view image (identity pose) to this view
    H = K @ (R + np.outer(t, n) / d) @ np.linalg.inv(K)
    img = cv2.warpPerspective(tex, H, (640, 480), flags=cv2.INTER_LINEAR)
    v, u = np.mgrid[0:480, 0:640]
    rays = np.linalg.inv(K) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    # depth of the plane in this camera: plane n.X = d in world, X = R^T (Xc - t)
    nc = R @ n
    z = (d + nc @ t) / (nc @ rays)
    depth = (z.reshape(480, 640) * 5000).astype(np.uint16)
    return RawFrame(0.0, 0.0, np.dstack([img] * 3), depth)


def test_tracking_recovers_known_motion():
    rng = np.random.default_rng(1)
    tex = np.full((480, 640), 128, np.uint8)
    for _ in range(60):
        x, y = rng.integers(0, 600), rng.integers(0, 440)
        cv2.rectangle(tex, (int(x), int(y)), (int(x + rng.integers(15, 80)), int(y + rng.integers(15, 80))),
                      int(rng.integers(0, 255)), -1)
    T_true = se3_exp(np.array([0.02, -0.01, 0.015, 0.01, -0.015, 0.008]))
    fa = prepare_frame(_plane_frame(np.eye(3), np.zeros(3), tex), CAM)
    fb = prepare_frame(_plane_frame(T_true[:3, :3], T_true[:3, 3], tex), CAM)
    ea, eb = detect_edges_pyramid(fa), detect_edges_pyramid(fb)
    cfg = TrackConfig()
    refs = []
    for k, (em, lvl) in enumerate(zip(ea, fa.levels)):
        t = edge_types(em.mask)[em.uv[:, 1], em.uv[:, 0]].astype(np.int64)
        z = np.where(lvl.valid[em.uv[:, 1], em.uv[:, 0]], lvl.depth[em.uv[:, 1], em.uv[:, 0]], 0)
        refs.append(make_ref_level(em.uv, z, t, lvl.camera, k, 2, cfg))
    res = track(refs, build_dt_pyramid(eb, fb.levels), {"none": np.eye(4)}, cfg)
    err = inv(T_true) @ res.T_cr
    assert np.linalg.norm(err[:3, 3]) < 0.005 and rot_angle_deg(err[:3, :3]) < 0.3
    assert res.inlier_ratio > 0.8


def test_evaluation_aligns_rigid_offset():
    rng = np.random.default_rng(0)
    stamps = np.arange(100) * 0.03
    gt = []
    for i in range(100):
        T = se3_exp(np.r_[np.sin(i / 10), np.cos(i / 15), 0.01 * i, 0.0, 0.01 * i, 0.0])
        gt.append(T)
    G = se3_exp(np.array([1.0, 2.0, -0.5, 0.3, 0.2, -0.4]))   # arbitrary world offset
    est = [inv(G) @ T for T in gt]
    ev = evaluate(stamps, est, stamps, gt)
    assert ev.ate_rmse < 1e-9 and ev.rot_rmse_deg < 1e-5 and ev.rpe_trans < 1e-9
    assert ev.start_rmse < 1e-9 and ev.start_rot_rmse_deg < 1e-5

    # A drifting estimate: start-aligned paths begin at exactly the same point, while the
    # best-fit alignment spreads the drift and does not pin the start.
    drift = [inv(G) @ T @ se3_exp(np.r_[0.002 * i, 0, 0, 0, 0, 0.001 * i]) for i, T in enumerate(gt)]
    ev = evaluate(stamps, drift, stamps, gt)
    assert np.allclose(ev.est_xyz[0], ev.gt_xyz[0]) and ev.start_rmse > ev.ate_rmse
    R, t = umeyama_rigid(rng.normal(size=(20, 3)) @ G[:3, :3].T, rng.normal(size=(20, 3)))
    assert np.isclose(np.linalg.det(R), 1.0)
