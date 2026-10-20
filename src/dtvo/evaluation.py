"""Trajectory evaluation against TUM / Bonn ground truth (Sturm et al. 2012 conventions).

Two ways of putting the estimate (which starts at the origin of its own frame) into the
ground-truth frame are reported:

  best fit     the TUM standard. One rigid transform (Horn / Umeyama, no scale for RGB-D)
               chosen to minimise the error over the WHOLE trajectory. Nothing pins the
               first frame, so drift is spread over the path and the two paths usually
               start a few cm apart.  -> ATE RMSE, comparable with published numbers.
  start-aligned  the estimate is placed exactly on the ground-truth pose of the first frame
               (position and orientation). Both paths start together and every gap is
               accumulated drift.  -> what the plots and the video show, with per-axis
               RMSE of x, y, z and roll, pitch, yaw.

  RPE  relative pose error over a fixed time step (default 1 s): drift per second; it does
       not depend on either alignment.

Bonn ground truth is the pose of the motion-capture marker, not the camera
(https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/, matrix T_m). Positions are
barely affected (marker ~1 cm from the camera), but orientations differ by a constant
rotation. For Bonn the constant camera->marker rotation is therefore estimated from the
data (rotation-only hand-eye fit) before comparing angles; for TUM it is identity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ..common.dataset import associate, read_file_list
from .se3 import R_to_quat, inv, pose_from_tum, rot_angle_deg


def load_groundtruth(path) -> tuple[np.ndarray, list[np.ndarray]]:
    stamps, rest = read_file_list(path)
    return stamps, [pose_from_tum(np.array([float(v) for v in r.split()])) for r in rest]


def write_tum(path, stamps, poses):
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for t, T in zip(stamps, poses):
            q = R_to_quat(T[:3, :3])
            f.write(f"{t:.6f} {T[0, 3]:.6f} {T[1, 3]:.6f} {T[2, 3]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def umeyama_rigid(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t minimising sum ||R src_i + t - dst_i||^2."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    C = (dst - mu_d).T @ (src - mu_s)
    U, _, Vt = np.linalg.svd(C)
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    return R, mu_d - R @ mu_s


def _polar(M: np.ndarray) -> np.ndarray:
    """Nearest rotation to M (maximises trace(R^T M))."""
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def fit_body_rotation(R_est: list[np.ndarray], R_gt: list[np.ndarray]) -> np.ndarray:
    """B minimising sum ||R_est_i B - R_gt_i||_F (orthogonal Procrustes)."""
    M = sum(Re.T @ Rg for Re, Rg in zip(R_est, R_gt))
    U, _, Vt = np.linalg.svd(M)
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ S @ Vt


def euler_deg(Rs: list[np.ndarray]) -> np.ndarray:
    """roll, pitch, yaw (deg; extrinsic x-y-z), unwrapped over time so plots are continuous."""
    e = Rotation.from_matrix(np.array(Rs)).as_euler("xyz")
    return np.degrees(np.unwrap(e, axis=0))


@dataclass
class Evaluation:
    t: np.ndarray                 # seconds from the start (matched frames only)
    est_xyz: np.ndarray           # N x 3, START-ALIGNED to the ground-truth frame
    gt_xyz: np.ndarray
    est_rpy: np.ndarray           # N x 3 deg, start-aligned
    gt_rpy: np.ndarray
    ate_rmse: float               # best-fit alignment (TUM standard)
    ate_mean: float
    ate_max: float
    rot_rmse_deg: float           # geodesic orientation error, best-fit alignment
    start_rmse: float             # position RMSE, start-aligned
    start_rot_rmse_deg: float     # geodesic orientation RMSE, start-aligned
    axis_rmse: np.ndarray         # x, y, z (m), start-aligned
    rpy_rmse: np.ndarray          # roll, pitch, yaw (deg), start-aligned
    rpe_trans: float              # m per rpe_delta seconds
    rpe_rot: float                # deg per rpe_delta seconds
    rpe_delta: float
    matched: int
    gt_length: float              # m travelled (ground truth)
    body_rotation_deg: float      # constant camera->GT-body rotation that was compensated
    anchor: np.ndarray            # 4x4: start-aligned position of est pose T = (anchor @ T)[:3, 3]
    est_index: np.ndarray         # index into the estimated poses of every matched frame


def evaluate(est_stamps, est_poses, gt_stamps, gt_poses, max_dt: float = 0.02,
             rpe_delta: float = 1.0, fit_body: bool = False) -> Evaluation:
    pairs = associate(np.asarray(est_stamps), np.asarray(gt_stamps), max_dt)
    if len(pairs) < 3:
        raise ValueError("Fewer than 3 estimated poses have a ground-truth pose within "
                         f"{max_dt}s; cannot evaluate.")
    E = [est_poses[i] for i, _ in pairs]
    G = [gt_poses[j] for _, j in pairs]
    t = np.array([est_stamps[i] for i, _ in pairs])

    pe = np.array([T[:3, 3] for T in E])
    pg = np.array([T[:3, 3] for T in G])

    # --- best fit (TUM ATE): positions by Umeyama, orientations by their own rotation fit
    # (the position fit leaves rotation undetermined when the camera barely translates,
    # e.g. fr3 sitting_static). Model R_gt ~ A R_est B (B = I unless fit_body).
    R_al, t_al = umeyama_rigid(pe, pg)
    err = np.linalg.norm(pe @ R_al.T + t_al - pg, axis=1)
    R_e0 = [T[:3, :3] for T in E]
    R_gt = [T[:3, :3] for T in G]
    A, B = R_al, np.eye(3)
    for _ in range(10 if fit_body else 1):
        if fit_body:
            B = fit_body_rotation([A @ R for R in R_e0], R_gt)
        A = _polar(sum(Rg @ (Re @ B).T for Re, Rg in zip(R_e0, R_gt)))
    rot_err_fit = np.array([rot_angle_deg((A @ Re @ B).T @ Rg) for Re, Rg in zip(R_e0, R_gt)])

    # --- start-aligned: first matched estimated pose put exactly on its ground-truth pose
    Bh = np.eye(4)
    Bh[:3, :3] = B
    anchor = G[0] @ inv(Bh) @ inv(E[0])
    pa = np.array([(anchor @ T)[:3, 3] for T in E])
    sa_err = np.linalg.norm(pa - pg, axis=1)
    R_est = [anchor[:3, :3] @ R @ B for R in R_e0]
    rot_err = np.array([rot_angle_deg(Re.T @ Rg) for Re, Rg in zip(R_est, R_gt)])
    rpy_e, rpy_g = euler_deg(R_est), euler_deg(R_gt)
    d = (rpy_e - rpy_g + 180) % 360 - 180

    # RPE over rpe_delta seconds (each pose paired with the first one >= delta later)
    rel_t, rel_r = [], []
    j = 0
    for i in range(len(t)):
        while j < len(t) and t[j] - t[i] < rpe_delta:
            j += 1
        if j >= len(t):
            break
        Ee = inv(E[i]) @ E[j]
        Eg = inv(G[i]) @ G[j]
        if fit_body:  # express the GT motion in the camera frame
            Bh = np.eye(4)
            Bh[:3, :3] = B
            Eg = Bh @ Eg @ inv(Bh)
        D = inv(Eg) @ Ee
        rel_t.append(np.linalg.norm(D[:3, 3]))
        rel_r.append(rot_angle_deg(D[:3, :3]))
    rel_t, rel_r = np.array(rel_t), np.array(rel_r)

    return Evaluation(
        # est angles drawn as GT + wrapped error, so both curves share one 360-degree branch
        t=t - t[0], est_xyz=pa, gt_xyz=pg, est_rpy=rpy_g + d, gt_rpy=rpy_g,
        ate_rmse=float(np.sqrt(np.mean(err ** 2))), ate_mean=float(err.mean()), ate_max=float(err.max()),
        rot_rmse_deg=float(np.sqrt(np.mean(rot_err_fit ** 2))),
        start_rmse=float(np.sqrt(np.mean(sa_err ** 2))),
        start_rot_rmse_deg=float(np.sqrt(np.mean(rot_err ** 2))),
        axis_rmse=np.sqrt(np.mean((pa - pg) ** 2, axis=0)),
        rpy_rmse=np.sqrt(np.mean(d ** 2, axis=0)),
        rpe_trans=float(np.sqrt(np.mean(rel_t ** 2))) if len(rel_t) else float("nan"),
        rpe_rot=float(np.sqrt(np.mean(rel_r ** 2))) if len(rel_r) else float("nan"),
        rpe_delta=rpe_delta, matched=len(pairs),
        gt_length=float(np.linalg.norm(np.diff(pg, axis=0), axis=1).sum()),
        body_rotation_deg=rot_angle_deg(B),
        anchor=anchor, est_index=np.array([i for i, _ in pairs]))
