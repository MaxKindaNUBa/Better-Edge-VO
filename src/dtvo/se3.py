"""Minimal SE(3) / SO(3) helpers (4x4 homogeneous matrices, twist = (v, w)).

Perturbations everywhere are applied on the LEFT:  T <- exp(delta) @ T.
"""
from __future__ import annotations

import numpy as np


def hat(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def so3_exp(w: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(w)
    W = hat(w)
    if th < 1e-10:
        return np.eye(3) + W
    return np.eye(3) + np.sin(th) / th * W + (1 - np.cos(th)) / th ** 2 * W @ W


def so3_log(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-8:
        return 0.5 * v
    if np.pi - th < 1e-5:  # near pi: axis from the symmetric part
        A = (R + np.eye(3)) / 2
        axis = A[:, np.argmax(np.diag(A))]
        return th * axis / np.linalg.norm(axis)
    return th / (2 * np.sin(th)) * v


def _V(w: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(w)
    W = hat(w)
    if th < 1e-10:
        return np.eye(3) + 0.5 * W
    return (np.eye(3) + (1 - np.cos(th)) / th ** 2 * W
            + (th - np.sin(th)) / th ** 3 * W @ W)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = so3_exp(xi[3:])
    T[:3, 3] = _V(xi[3:]) @ xi[:3]
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    w = so3_log(T[:3, :3])
    return np.concatenate([np.linalg.solve(_V(w), T[:3, 3]), w])


def inv(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def scale_motion(T: np.ndarray, s: float) -> np.ndarray:
    """T^s on the manifold (s = 0.5 half motion, 2 double motion)."""
    return se3_exp(s * se3_log(T))


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """(qx, qy, qz, qw) -> rotation matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R: np.ndarray) -> np.ndarray:
    """rotation matrix -> (qx, qy, qz, qw), qw >= 0."""
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        q = np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = np.zeros(4)
        q[i] = s / 4
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    return q if q[3] >= 0 else -q


def pose_from_tum(vals: np.ndarray) -> np.ndarray:
    """(tx ty tz qx qy qz qw) -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = quat_to_R(np.asarray(vals[3:7], float))
    T[:3, 3] = vals[:3]
    return T


def rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def orthonormalize(T: np.ndarray) -> np.ndarray:
    """Project the rotation block back onto SO(3); composing many poses slowly breaks it."""
    U, _, Vt = np.linalg.svd(T[:3, :3])
    out = T.copy()
    out[:3, :3] = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return out
