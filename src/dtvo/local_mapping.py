"""DTVO local-mapping thread (Yan, Li, Zhou 2022, Sec. 5).

  * Keyframes hold their edge pixels with an inverse depth each (RGB-D: from the sensor,
    Sec. 4.5) and the DT map of their own edges.
  * Depth propagation: when a keyframe is created, the edges of every keyframe in the
    sliding window are projected into it to give depth to edge pixels the sensor missed.
  * Sliding-window optimisation (Eq. 10-12): 5-7 active keyframes; the residual of an edge
    pixel of keyframe m is the DT value of keyframe n where it projects, for all n != m.
    All edge inverse depths and all keyframe poses are refined jointly with
    Levenberg-Marquardt. Each point has a single parameter, so the depth block of the
    Hessian is diagonal and is eliminated with the Schur complement (Eq. 13).
  * Marginalisation (Eq. 13): the oldest keyframe and the edges it hosts are removed
    with the Schur complement, leaving a Gaussian prior on the remaining poses.

RGB-D specifics (the paper leaves these open):
  * the sensor depth enters as a prior on each inverse depth (Kinect noise model), so the
    DT residual - which only constrains depth across an edge - cannot drift along it;
  * the camera is calibrated, so intrinsics are kept fixed (the paper optionally refines c);
  * the oldest keyframe in the window is held fixed to remove the 6-DoF gauge freedom.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..common.edges import EDGE_DEPTH, EDGE_DEPTH_JUMP, EdgeMap
from ..common.preprocess import FrameLevel, PreparedFrame, depth_jump_threshold, kinect_depth_sigma
from .tracking import (N_TYPES, DTMap, RefLevel, TrackConfig, dt_point_jacobian, edge_types,
                       huber_cost, huber_weight, make_ref_level, project_and_sample)
from .se3 import inv, orthonormalize, rot_angle_deg, se3_exp, se3_log


@dataclass
class MappingConfig:
    window_size: int = 6              # active keyframes (paper: 5-7)
    iterations: int = 6               # LM iterations per new keyframe
    points_per_kf: int = 3000         # edge pixels per keyframe in the window optimisation
    huber_px: float = 1.5
    outlier_px: float = 5.0
    depth_sigma_scale: float = 2.0    # inflate the Kinect noise model for the depth prior
    propagated_sigma_scale: float = 3.0  # weaker prior for depths propagated from other keyframes
    use_jump_edges: bool = False      # give depth-jump edges the foreground (nearest) depth
    jump_radius: int = 3
    smooth_radius: int = 0            # drop edge depths whose (2r+1)^2 neighbourhood is not smooth
    smooth_k: float = 3.0             # ... smooth = depth spread <= depth_jump_threshold(z, k)
    marginalise: bool = True
    # Not in the paper: a window optimisation that moves any keyframe further than a camera
    # can plausibly be corrected (the tracked poses are already within a few cm) is undone.
    max_kf_shift_m: float = 0.10
    max_kf_shift_deg: float = 10.0
    enabled: bool = True              # False = tracking only (no window optimisation)


def edge_depths(em: EdgeMap, lvl: FrameLevel, cfg: MappingConfig) -> np.ndarray:
    """Depth (m) for every edge pixel of em; 0 where unknown.

    EDGE_DEPTH pixels use their own reading. Edges on a depth discontinuity are object
    outlines, which move with the occluding (nearer) surface, so with use_jump_edges they
    take the nearest valid depth within jump_radius.
    """
    z = em.depth.astype(np.float32).copy()
    if cfg.smooth_radius > 0:
        # Colour and depth outlines are a few pixels apart on the Kinect, so an edge pixel
        # near a depth step often reads the other surface's depth: keep only smooth spots.
        r = max(1, cfg.smooth_radius >> lvl.level)
        ker = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
        zmax = cv2.dilate(np.where(lvl.valid, lvl.depth, -np.inf).astype(np.float32), ker)
        zmin = cv2.erode(np.where(lvl.valid, lvl.depth, np.inf).astype(np.float32), ker)
        u, v = em.uv[:, 0], em.uv[:, 1]
        spread = zmax[v, u] - zmin[v, u]
        z[~(spread <= depth_jump_threshold(z, cfg.smooth_k, 0.01))] = 0.0
    if cfg.use_jump_edges:
        r = max(1, cfg.jump_radius >> lvl.level)
        zinf = np.where(lvl.valid, lvl.depth, np.inf).astype(np.float32)
        zmin = cv2.erode(zinf, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
        jump = em.label == EDGE_DEPTH_JUMP
        u, v = em.uv[jump, 0], em.uv[jump, 1]
        zj = zmin[v, u]
        z[jump] = np.where(np.isfinite(zj), zj, 0.0)
    return z


def idepth_sigma(z: np.ndarray, scale: float) -> np.ndarray:
    """Std-dev of inverse depth from the Kinect axial noise: sigma_d = sigma_z / z^2."""
    return scale * kinect_depth_sigma(z) / np.maximum(z, 1e-3) ** 2


class Keyframe:
    def __init__(self, kf_id: int, frame_index: int, T_wk: np.ndarray, frame: PreparedFrame,
                 edge_maps: list[EdgeMap], dts: list[DTMap], mcfg: MappingConfig,
                 tcfg: TrackConfig, rng: np.random.Generator):
        self.id, self.frame_index, self.timestamp = kf_id, frame_index, frame.timestamp
        self.weak = False   # re-anchored after a lost run (pose from the motion model)
        self.T_wk = T_wk.copy()
        self.cam = frame.levels[0].camera
        self.dt0 = dts[0]
        self.tcfg = tcfg
        top = len(edge_maps) - 1

        em0 = edge_maps[0]
        self.uv = em0.uv
        self.types = edge_types(em0.mask)[em0.uv[:, 1], em0.uv[:, 0]].astype(np.int64)
        z = edge_depths(em0, frame.levels[0], mcfg)
        self.idepth = np.where(z > 0, 1.0 / np.maximum(z, 1e-6), 0.0)
        self.prior_idepth = self.idepth.copy()
        self.prior_sigma = np.where(z > 0, idepth_sigma(z, mcfg.depth_sigma_scale), np.inf)
        self.rays = np.column_stack([(self.uv[:, 0] - self.cam.cx) / self.cam.fx,
                                     (self.uv[:, 1] - self.cam.cy) / self.cam.fy,
                                     np.ones(len(self.uv))])
        self.rng = rng
        self.mcfg = mcfg

        # Coarser levels: their own edges with their own (downsampled) depth; fixed for the
        # keyframe's lifetime. Level 0 is rebuilt after the window optimisation refines depths.
        self._coarse = []
        for k in range(1, top + 1):
            em = edge_maps[k]
            t = edge_types(em.mask)[em.uv[:, 1], em.uv[:, 0]].astype(np.int64)
            zk = edge_depths(em, frame.levels[k], mcfg)
            self._coarse.append(make_ref_level(em.uv, zk, t, frame.levels[k].camera, k, top, tcfg))
        self.top = top
        self.select_ba_points()
        self.rebuild_reference()

    def select_ba_points(self):
        has = np.flatnonzero(self.idepth > 0)
        n = self.mcfg.points_per_kf
        self.ba_idx = np.sort(self.rng.choice(has, n, replace=False)) if len(has) > n else has

    def rebuild_reference(self):
        z = np.where(self.idepth > 0, 1.0 / np.maximum(self.idepth, 1e-9), 0.0)
        has = np.flatnonzero(z > 0)
        if len(has) > self.tcfg.max_ref_points:
            has = np.sort(self.rng.choice(has, self.tcfg.max_ref_points, replace=False))
        zz = np.zeros_like(z)
        zz[has] = z[has]
        self.ref0 = make_ref_level(self.uv, zz, self.types, self.cam, 0, self.top, self.tcfg)
        # Reference count per type for the keyframe-selection test (Sec. 5.1)
        self.type_counts = np.bincount(self.ref0.types, minlength=N_TYPES)

    @property
    def ref_levels(self) -> list[RefLevel]:
        return [self.ref0, *self._coarse]

    def points_world(self, idx=None) -> np.ndarray:
        idx = np.flatnonzero(self.idepth > 0) if idx is None else idx
        P = self.rays[idx] / self.idepth[idx, None]
        return P @ self.T_wk[:3, :3].T + self.T_wk[:3, 3]

    def release(self):
        """Drop image data once the keyframe has left the window (only its pose is kept)."""
        self.dt0 = self._coarse = self.ref0 = None
        self.uv = self.types = self.idepth = self.prior_idepth = self.prior_sigma = None
        self.rays = self.ba_idx = None


def propagate_depth(new: Keyframe, window: list[Keyframe]):
    """Give depth to new-keyframe edge pixels without a reading, from window keyframes (Sec. 5)."""
    missing = new.idepth <= 0
    if not window or not missing.any():
        return 0
    h, w = new.dt0.dist.shape
    zbuf = np.full(h * w, np.inf, np.float32)
    T_kw = inv(new.T_wk)
    for kf in window:
        Pw = kf.points_world()
        Pc = Pw @ T_kw[:3, :3].T + T_kw[:3, 3]
        z = Pc[:, 2]
        ok = z > 0.1
        u = np.round(new.cam.fx * Pc[ok, 0] / z[ok] + new.cam.cx).astype(np.int64)
        v = np.round(new.cam.fy * Pc[ok, 1] / z[ok] + new.cam.cy).astype(np.int64)
        inb = (u >= 0) & (v >= 0) & (u < w) & (v < h)
        np.minimum.at(zbuf, v[inb] * w + u[inb], z[ok][inb].astype(np.float32))
    zbuf = cv2.erode(zbuf.reshape(h, w), np.ones((3, 3), np.uint8))  # nearest within 1 px
    idx = np.flatnonzero(missing)
    zp = zbuf[new.uv[idx, 1], new.uv[idx, 0]]
    got = np.isfinite(zp)
    idx, zp = idx[got], zp[got].astype(np.float64)
    new.idepth[idx] = 1.0 / zp
    new.prior_idepth[idx] = 1.0 / zp
    new.prior_sigma[idx] = idepth_sigma(zp, new.mcfg.propagated_sigma_scale)
    new.select_ba_points()
    new.rebuild_reference()
    return len(idx)


class LocalMapper:
    """Sliding window of keyframes with joint pose / inverse-depth optimisation."""

    def __init__(self, cfg: MappingConfig):
        self.cfg = cfg
        self.window: list[Keyframe] = []
        self.prior = None   # {"ids": [kf ids], "H": 6n x 6n, "b": 6n, "T0": {id: 4x4}}
        self.stats = {"propagated": 0, "ba_runs": 0, "ba_energy_drop": [], "ba_rejected": 0,
                      "ba_rejected_shift_m": []}
        self.last_rejected = False   # was the last window optimisation undone?

    # ------------------------------------------------------------------ public
    def add_keyframe(self, kf: Keyframe):
        self.last_rejected = False
        if self.cfg.enabled:
            self.stats["propagated"] += propagate_depth(kf, self.window)
        self.window.append(kf)
        if self.cfg.enabled and len(self.window) >= 2:
            self.optimise()
            kf.rebuild_reference()
        while len(self.window) > self.cfg.window_size:
            old = self.window[0]
            if self.cfg.enabled and self.cfg.marginalise:
                self.marginalise_oldest()
            self.window.pop(0)
            old.release()

    # ------------------------------------------------------------------ residuals
    def _pair_terms(self, m: Keyframe, n: Keyframe, Tm, Tn, d, with_jac):
        """Residuals of m's window points observed in keyframe n's DT map (Eq. 10)."""
        idx = m.ba_idx
        x = m.rays[idx]
        Pm = x / d[:, None]
        T_nm = inv(Tn) @ Tm
        out = project_and_sample(Pm, T_nm, n.dt0, with_grad=with_jac)
        Pn, valid, r = out[0], out[3], out[4]
        cap = self.cfg.outlier_px
        if not with_jac:
            return np.where(valid, np.minimum(r, cap), cap)
        gx, gy = out[5], out[6]
        use = valid & (r < cap)
        j3 = dt_point_jacobian(Pn[use], gx[use], gy[use], n.cam)     # d r / d Pn
        R_wn, R_nm = Tn[:3, :3], T_nm[:3, :3]
        Pw = Pm[use] @ Tm[:3, :3].T + Tm[:3, 3]
        jw = j3 @ R_wn.T                                                # d r / d Pw
        J_host = np.hstack([jw, np.cross(Pw, jw)])                      # left twist on T_wm
        J_d = -np.einsum("ij,ij->i", j3, x[use] @ R_nm.T) / d[use] ** 2  # d r / d inverse depth
        return use, r[use], J_host, J_d

    def _prior_delta(self, poses: dict[int, np.ndarray]) -> np.ndarray:
        return np.concatenate([se3_log(poses[i] @ inv(self.prior["T0"][i])) for i in self.prior["ids"]])

    def _energy(self, kfs, poses, depths) -> float:
        hk = self.cfg.huber_px
        E = 0.0
        for a, m in enumerate(kfs):
            for n in kfs:
                if n is not m:
                    E += huber_cost(self._pair_terms(m, n, poses[m.id], poses[n.id], depths[a], False), hk).sum()
            s = m.prior_sigma[m.ba_idx]
            fin = np.isfinite(s)
            E += 0.5 * np.sum(((depths[a][fin] - m.prior_idepth[m.ba_idx][fin]) / s[fin]) ** 2)
        if self.prior is not None:
            dlt = self._prior_delta(poses)
            E += float(self.prior["b"] @ dlt + 0.5 * dlt @ self.prior["H"] @ dlt)
        return E

    def _build_system(self, kfs, free_ids, poses, depths, host_only=None):
        """Normal equations: pose block H_pp, coupling H_pd (6F x Npts), diagonal H_dd."""
        F = len(free_ids)
        col = {kid: i for i, kid in enumerate(free_ids)}
        offs = np.cumsum([0] + [len(d) for d in depths])
        Np = int(offs[-1])
        H_pp = np.zeros((6 * F, 6 * F))
        b_p = np.zeros(6 * F)
        H_pd = np.zeros((6 * F, Np))
        H_dd = np.zeros(Np)
        b_d = np.zeros(Np)
        hk = self.cfg.huber_px
        for a, m in enumerate(kfs):
            if host_only is not None and m is not host_only:
                continue
            pts = np.arange(offs[a], offs[a + 1])
            for n in kfs:
                if n is m:
                    continue
                use, r, Jh, Jd = self._pair_terms(m, n, poses[m.id], poses[n.id], depths[a], True)
                if not use.any():
                    continue
                w = huber_weight(r, hk)
                gidx = pts[use]
                H_dd[gidx] += w * Jd * Jd
                b_d[gidx] += w * Jd * r
                # Pose of the target n has Jacobian -Jh (relative transform)
                for kid, sgn in ((m.id, 1.0), (n.id, -1.0)):
                    if kid not in col:
                        continue
                    i = 6 * col[kid]
                    Ji = sgn * Jh
                    b_p[i:i + 6] += Ji.T @ (w * r)
                    H_pd[i:i + 6, gidx] += (Ji * (w * Jd)[:, None]).T
                    for kid2, sgn2 in ((m.id, 1.0), (n.id, -1.0)):
                        if kid2 in col:
                            j = 6 * col[kid2]
                            H_pp[i:i + 6, j:j + 6] += Ji.T @ (sgn2 * Jh * w[:, None])
            # Sensor / propagated depth prior
            s = m.prior_sigma[m.ba_idx]
            fin = np.isfinite(s)
            H_dd[pts[fin]] += 1.0 / s[fin] ** 2
            b_d[pts[fin]] += (depths[a][fin] - m.prior_idepth[m.ba_idx][fin]) / s[fin] ** 2
        return H_pp, b_p, H_pd, H_dd, b_d, offs

    def _add_prior(self, H_pp, b_p, free_ids, poses):
        if self.prior is None:
            return
        col = {kid: i for i, kid in enumerate(free_ids)}
        dlt = self._prior_delta(poses)
        g = self.prior["b"] + self.prior["H"] @ dlt
        sel = np.concatenate([np.arange(6 * col[i], 6 * col[i] + 6) for i in self.prior["ids"]])
        H_pp[np.ix_(sel, sel)] += self.prior["H"]
        b_p[sel] += g

    # ------------------------------------------------------------------ optimisation
    def optimise(self):
        kfs = self.window
        free_ids = [k.id for k in kfs[1:]]          # oldest keyframe fixed (gauge)
        poses = {k.id: k.T_wk.copy() for k in kfs}
        depths = [k.idepth[k.ba_idx].copy() for k in kfs]
        E = self._energy(kfs, poses, depths)
        E0, lam = E, 1e-3
        for _ in range(self.cfg.iterations):
            H_pp, b_p, H_pd, H_dd, b_d, offs = self._build_system(kfs, free_ids, poses, depths)
            self._add_prior(H_pp, b_p, free_ids, poses)
            for _ in range(4):
                Hdd = H_dd * (1 + lam) + 1e-9
                Hpp = H_pp + lam * np.diag(np.diag(H_pp)) + 1e-9 * np.eye(len(b_p))
                # Schur complement: eliminate the (diagonal) inverse-depth block
                S = Hpp - (H_pd / Hdd) @ H_pd.T
                rhs = b_p - H_pd @ (b_d / Hdd)
                try:
                    dp = -np.linalg.solve(S, rhs)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                dd = -(b_d + H_pd.T @ dp) / Hdd
                new_poses = dict(poses)
                for i, kid in enumerate(free_ids):
                    new_poses[kid] = se3_exp(dp[6 * i:6 * i + 6]) @ poses[kid]
                new_depths = [np.maximum(depths[a] + dd[offs[a]:offs[a + 1]], 1e-3)
                              for a in range(len(kfs))]
                E_new = self._energy(kfs, new_poses, new_depths)
                if E_new < E:
                    poses, depths, E = new_poses, new_depths, E_new
                    lam = max(lam / 3, 1e-6)
                    break
                lam *= 8
            else:
                break
        self.stats["ba_runs"] += 1
        shift_m = max(np.linalg.norm(poses[k.id][:3, 3] - k.T_wk[:3, 3]) for k in kfs)
        shift_deg = max(rot_angle_deg(poses[k.id][:3, :3].T @ k.T_wk[:3, :3]) for k in kfs)
        if shift_m > self.cfg.max_kf_shift_m or shift_deg > self.cfg.max_kf_shift_deg:
            self.last_rejected = True   # keep the tracked poses and the old depths
            self.stats["ba_rejected"] += 1
            self.stats["ba_rejected_shift_m"].append(float(shift_m))
            return
        for a, k in enumerate(kfs):
            k.T_wk = orthonormalize(poses[k.id])
            k.idepth[k.ba_idx] = depths[a]
        self.stats["ba_energy_drop"].append((E0 - E) / max(E0, 1e-9))

    def marginalise_oldest(self):
        """Schur-complement the oldest keyframe's points out into a prior on the other poses."""
        kfs = self.window
        old = kfs[0]
        free_ids = [k.id for k in kfs[1:]]
        poses = {k.id: k.T_wk.copy() for k in kfs}
        depths = [k.idepth[k.ba_idx].copy() for k in kfs]
        H_pp, b_p, H_pd, H_dd, b_d, offs = self._build_system(kfs, free_ids, poses, depths, host_only=old)
        sl = slice(offs[0], offs[1])               # only the old keyframe's points are coupled
        Hd = H_dd[sl] + 1e-9
        H_m = H_pp - (H_pd[:, sl] / Hd) @ H_pd[:, sl].T
        b_m = b_p - H_pd[:, sl] @ (b_d[sl] / Hd)
        self._add_prior(H_m, b_m, free_ids, poses)  # merge the existing prior (relinearised here)
        # The next-oldest keyframe becomes the fixed gauge keyframe: condition on it (drop its block)
        keep = np.arange(6, 6 * len(free_ids))
        H_m = H_m[np.ix_(keep, keep)]
        b_m = b_m[keep]
        H_m = 0.5 * (H_m + H_m.T)
        self.prior = {"ids": free_ids[1:], "H": H_m, "b": b_m,
                      "T0": {i: poses[i].copy() for i in free_ids[1:]}} if len(free_ids) > 1 else None
