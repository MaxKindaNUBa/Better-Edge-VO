"""Result video for a DTVO run: tracking overlay on the left, live x-y path on the right.

Left panel (drawn while DTVO runs, one image per frame):
    yellow  every edge detected in the current frame (these build the DT map)
    green   reference-keyframe edges re-projected with the tracked pose that land on a
            current edge (< 2 px)  -> matched, static structure
    red     re-projected reference edges that land nowhere -> occluded, moving objects,
            or a tracking error
  with an edge filter (run_dynamic.py) the masked objects are drawn too:
    red     tinted region + outline + label: object whose edges were left out, red edges
            are the removed ones; unmatched reference edges then turn blue
Right panel (drawn after the run, once the alignment to ground truth is known):
    matplotlib x-y plot growing frame by frame, estimate vs ground truth (and optionally
    a second estimate, e.g. plain DTVO), in the same aligned frame as dtvo_plot.png.

plot_evaluation: the static figure (x-y path, x / y / z and roll / pitch / yaw vs time,
summary table) for one or more estimates of the same sequence.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..common.video import VideoSink
from .system import FrameDebug

EDGE_BGR = (0, 210, 255)        # yellow
MATCH_BGR = (90, 220, 60)       # green
MISS_BGR = (70, 60, 235)        # red
MISS_MASKED_BGR = (235, 140, 40)  # blue, when red is taken by the masked objects
OBJ_BGR = (72, 73, 227)         # red, masked objects (same as the detection video)
GT_COLOUR, EST_COLOUR = "#52514e", "#2a78d6"
RUN_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]   # estimates, in order (ground truth stays neutral)
INK, MUTED, GRID, FAINT = "#0b0b0b", "#6b6a66", "#e4e3df", "#cfcdc6"


def _bar(img, text, y0, y1, ty, scale=0.5):
    cv2.rectangle(img, (0, y0), (img.shape[1], y1), (20, 20, 20), -1)
    cv2.putText(img, text, (8, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def tracking_overlay(dbg: FrameDebug, idx: int, n_keyframes: int, max_points: int = 6000,
                     match_px: float = 2.0,
                     objects: list[tuple[np.ndarray, str]] | None = None) -> np.ndarray:
    """objects: (mask, label) of every masked object, when an edge filter ran."""
    img = (dbg.rgb * 0.55).astype(np.uint8)
    masking = objects is not None
    if masking and objects:
        region = np.zeros(dbg.edge_mask.shape, bool)
        for m, _ in objects:
            region |= m
        img[region] = (0.6 * img[region] + 0.4 * np.array(OBJ_BGR)).astype(np.uint8)
    img[dbg.edge_mask] = EDGE_BGR
    if dbg.dropped is not None:
        img[dbg.dropped] = OBJ_BGR
    h, w = dbg.edge_mask.shape
    matched = 0.0
    if dbg.ref_P is not None and len(dbg.ref_P):
        P = dbg.ref_P[::max(1, int(np.ceil(len(dbg.ref_P) / max_points)))]
        Pc = P @ dbg.T_cr[:3, :3].T + dbg.T_cr[:3, 3]
        ok = Pc[:, 2] > 0.05
        cam = dbg.camera
        u = np.round(cam.fx * Pc[ok, 0] / Pc[ok, 2] + cam.cx).astype(int)
        v = np.round(cam.fy * Pc[ok, 1] / Pc[ok, 2] + cam.cy).astype(int)
        inb = (u >= 0) & (v >= 0) & (u < w - 1) & (v < h - 1)
        u, v = u[inb], v[inb]
        dist = cv2.distanceTransform((~dbg.edge_mask).astype(np.uint8), cv2.DIST_L2, 3)
        hit = dist[v, u] < match_px
        matched = hit.mean() if len(hit) else 0.0
        for sel, col in ((~hit, MISS_MASKED_BGR if masking else MISS_BGR), (hit, MATCH_BGR)):
            for dv in (0, 1):          # 2x2 dots so they stay visible after video compression
                for du in (0, 1):
                    img[v[sel] + dv, u[sel] + du] = col

    for m, label in objects or []:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, OBJ_BGR, 1, cv2.LINE_AA)
        ys, xs = np.nonzero(m)
        if len(xs):
            x, y = int(xs.min()), max(int(ys.min()), 44)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), OBJ_BGR, -1)
            cv2.putText(img, label, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1,
                        cv2.LINE_AA)

    status = "  LOST" if dbg.lost else ("  NEW KEYFRAME" if dbg.is_keyframe else "")
    masked = ""
    if masking:
        n_kept = int(dbg.edge_mask.sum())
        n_drop = 0 if dbg.dropped is None else int(dbg.dropped.sum())
        masked = f"   masked {len(objects)} obj, {100 * n_drop / max(n_kept + n_drop, 1):.0f}% of edges"
    _bar(img, f"frame {idx}   keyframes {n_keyframes}   matched {100 * matched:.0f}%{masked}{status}",
         0, 26, 18)
    _bar(img, "yellow: edges used  red: masked objects  green / blue: keyframe edges matched / not"
         if masking else
         "yellow: detected edges   green: keyframe edges matched   red: not matched",
         h - 24, h, h - 8, 0.4 if masking else 0.45)
    return img


class TrajectoryPanel:
    """matplotlib x-y plot rendered to an image, extended by one frame per call."""

    def __init__(self, est_xy: np.ndarray, gt_xy: np.ndarray | None, t: np.ndarray, title: str,
                 width: int = 640, height: int = 480, est_label: str = "DTVO",
                 other_xy: np.ndarray | None = None, other_label: str = ""):
        """other_xy: a second estimate drawn for comparison (same frames as est_xy)."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        self.est, self.gt, self.t, self.other = est_xy, gt_xy, t, other_xy
        self.fig = Figure(figsize=(width / 100, height / 100), dpi=100)
        self.canvas = FigureCanvasAgg(self.fig)
        ax = self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.13, right=0.97, top=0.9, bottom=0.12)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("x (m)", color=INK, fontsize=9)
        ax.set_ylabel("y (m)", color=INK, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK)

        pts = np.vstack([p for p in (est_xy, gt_xy, other_xy) if p is not None])
        pts = pts[np.isfinite(pts).all(1)]
        lo, hi = pts.min(0), pts.max(0)
        c, half = (lo + hi) / 2, max((hi - lo).max() / 2, 0.05) * 1.12
        ax.set_xlim(c[0] - half * width / height, c[0] + half * width / height)
        ax.set_ylim(c[1] - half, c[1] + half)
        ax.set_aspect("equal", adjustable="box")

        self.gt_line = None
        if gt_xy is not None:
            ax.plot(gt_xy[:, 0], gt_xy[:, 1], color=FAINT, lw=1, zorder=1)   # whole path, faint
            self.gt_line, = ax.plot([], [], color=GT_COLOUR, lw=2, label="Ground truth", zorder=2)
        self.other_line = None
        if other_xy is not None:
            self.other_line, = ax.plot([], [], color=RUN_COLOURS[1], lw=1.6, label=other_label, zorder=3)
        self.est_line, = ax.plot([], [], color=EST_COLOUR, lw=2, label=est_label, zorder=3)
        self.dot, = ax.plot([], [], "o", color=EST_COLOUR, ms=8, mec="white", mew=2, zorder=4)
        ax.legend(frameon=False, loc="upper right", fontsize=8)
        self.info = ax.text(0.01, 0.01, "", transform=ax.transAxes, fontsize=8, color=MUTED)

    def render(self, i: int) -> np.ndarray:
        self.est_line.set_data(self.est[:i + 1, 0], self.est[:i + 1, 1])
        self.dot.set_data(self.est[i:i + 1, 0], self.est[i:i + 1, 1])
        if self.other_line is not None:
            self.other_line.set_data(self.other[:i + 1, 0], self.other[:i + 1, 1])
        text = f"t = {self.t[i]:.1f} s"
        if self.gt_line is not None:
            self.gt_line.set_data(self.gt[:i + 1, 0], self.gt[:i + 1, 1])
            known = np.flatnonzero(np.isfinite(self.gt[:i + 1, 0]))
            if len(known):
                j = known[-1]
                text += f"   position error {100 * np.linalg.norm(self.est[j] - self.gt[j]):.1f} cm"
                if self.other is not None:
                    text += f" (vs {100 * np.linalg.norm(self.other[j] - self.gt[j]):.1f} cm)"
        self.info.set_text(text)
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[:, :, 2::-1].copy()   # RGBA -> BGR


def compose_video(overlay_video: Path, out: Path, panel: TrajectoryPanel, fps: float):
    """Put the recorded overlay frames and the growing plot side by side."""
    cap = cv2.VideoCapture(str(overlay_video))
    sink, i = None, 0
    while True:
        ok, left = cap.read()
        if not ok or i >= len(panel.est):
            break
        right = panel.render(i)
        if right.shape[0] != left.shape[0]:
            right = cv2.resize(right, (int(right.shape[1] * left.shape[0] / right.shape[0]), left.shape[0]))
        frame = np.hstack([left, right])
        if sink is None:
            sink = VideoSink(out, frame.shape[1], frame.shape[0], fps)
        sink.write(frame)
        i += 1
    cap.release()
    if sink is not None:
        sink.close()
    return i


def plot_evaluation(evs: dict, title: str, out: Path, show: bool):
    """evs: {label: Evaluation}, the first is the main estimate. Ground truth is drawn from it."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(15, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, width_ratios=[1.25, 1, 1])
    runs = list(evs.items())
    ev0 = runs[0][1]
    colour = {name: RUN_COLOURS[k] for k, (name, _) in enumerate(runs)}
    width = {name: 2 if k == 0 else 1.5 for k, (name, _) in enumerate(runs)}
    order = runs[::-1]   # main estimate drawn last, on top

    def style(ax):
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    # Top view of the path
    ax = fig.add_subplot(gs[0:2, 0])
    ax.plot(ev0.gt_xyz[:, 0], ev0.gt_xyz[:, 1], color=GT_COLOUR, lw=2, label="Ground truth")
    for name, e in order:
        ax.plot(e.est_xyz[:, 0], e.est_xyz[:, 1], color=colour[name], lw=width[name], label=name)
    ax.plot(*ev0.gt_xyz[0, :2], "o", color=GT_COLOUR, ms=8, mec="white", mew=2)
    ax.annotate("start", ev0.gt_xyz[0, :2], textcoords="offset points", xytext=(6, 6), color=MUTED)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Camera path, top view (x-y), start-aligned", loc="left", fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[0], *handles[1:][::-1]], [labels[0], *labels[1:][::-1]], frameon=False, loc="best")
    style(ax)

    # Summary: one value column per estimate
    ax = fig.add_subplot(gs[2, 0])
    ax.axis("off")
    rows = [("ATE RMSE, best-fit alignment (TUM)", lambda e: f"{e.ate_rmse:.4f} m"),
            ("Rotation RMSE, best-fit alignment", lambda e: f"{e.rot_rmse_deg:.2f}°"),
            ("Position RMSE, start-aligned (plotted)", lambda e: f"{e.start_rmse:.4f} m"),
            ("RMSE x / y / z, start-aligned", lambda e: " / ".join(f"{v:.4f}" for v in e.axis_rmse) + " m"
             if len(runs) == 1 else " / ".join(f"{100 * v:.1f}" for v in e.axis_rmse) + " cm"),
            ("Rotation RMSE, start-aligned", lambda e: f"{e.start_rot_rmse_deg:.2f}°"),
            ("RMSE roll / pitch / yaw, start-aligned", lambda e: " / ".join(
                f"{v:.2f}" if len(runs) == 1 else f"{v:.1f}" for v in e.rpy_rmse) + "°"),
            (f"RPE per {ev0.rpe_delta:g} s", lambda e: f"{e.rpe_trans:.4f} m, {e.rpe_rot:.2f}°"),
            ("Frames matched to GT", lambda e: f"{e.matched}   (path {e.gt_length:.2f} m)")]
    multi = len(runs) > 1
    xs = list(np.linspace(0.70, 1.0, len(runs))) if multi else [1.0]
    if multi:
        for x, (name, _) in zip(xs, runs):
            ax.text(x, 1.04, name, color=colour[name], transform=ax.transAxes, va="top", ha="right",
                    fontweight="bold")
    for i, (k, fn) in enumerate(rows):
        y = 0.97 - i * 0.12 - (0.06 if multi else 0)
        ax.text(0.0, y, k, color=MUTED, transform=ax.transAxes, va="top")
        for x, (name, e) in zip(xs, runs):
            v = f"{e.matched}" if multi and i == len(rows) - 1 else fn(e)   # no room for the path length
            ax.text(x, y, v, color=INK, transform=ax.transAxes, va="top", ha="right", fontweight="bold")

    labels_pos = ["x", "y", "z"]
    labels_rot = ["roll (about x)", "pitch (about y)", "yaw (about z)"]
    for i in range(3):
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(ev0.t, ev0.gt_xyz[:, i], color=GT_COLOUR, lw=2, label="Ground truth")
        for name, e in order:
            ax.plot(e.t, e.est_xyz[:, i], color=colour[name], lw=width[name], label=name)
        ax.set_ylabel(f"{labels_pos[i]} (m)")
        ax.set_title(f"Position {labels_pos[i]}   RMSE "
                     + " / ".join(f"{100 * e.axis_rmse[i]:.1f}" for _, e in runs) + " cm",
                     loc="left", fontweight="bold")
        style(ax)
        if i == 0:
            ax.legend(frameon=False, loc="best")
        if i == 2:
            ax.set_xlabel("time (s)")

        ax = fig.add_subplot(gs[i, 2])
        ax.plot(ev0.t, ev0.gt_rpy[:, i], color=GT_COLOUR, lw=2, label="Ground truth")
        for name, e in order:
            ax.plot(e.t, e.est_rpy[:, i], color=colour[name], lw=width[name], label=name)
        ax.set_ylabel("deg")
        ax.set_title(f"Rotation: {labels_rot[i]}   RMSE "
                     + " / ".join(f"{e.rpy_rmse[i]:.2f}" for _, e in runs) + "°",
                     loc="left", fontweight="bold")
        style(ax)
        if i == 2:
            ax.set_xlabel("time (s)")

    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    if not show:
        plt.close(fig)  # otherwise kept open for plt.show() at the very end
