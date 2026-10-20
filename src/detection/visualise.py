"""Video panels and figures for the detection runs."""
from __future__ import annotations

import cv2
import numpy as np

from .associate import EdgeSplit
from .detector import MODE_LABELS, Detection
from .object_mask import ObjectMask
from .priors import is_suspect

# BGR. Background edges neutral; object edges use the categorical red / aqua slots.
COL_BG_EDGE = (205, 205, 205)
COL_SUSPECT = (72, 73, 227)       # #e34948
COL_STATIC_OBJ = (122, 175, 27)   # #1baf7a
COL_TEXT = (245, 245, 245)
COL_BAR = (24, 24, 24)

# Matplotlib (light surface): one fixed categorical slot per detector mode
MODE_COLOURS = {"coco": "#2a78d6", "yoloe_text": "#eb6834", "yoloe_pf": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e4e3df"


def _put(img, text, org, scale=0.48, colour=COL_TEXT, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def render_panel(rgb: np.ndarray, edges_uv: np.ndarray, mode: str, dets: list[Detection],
                 masks: list[ObjectMask], split: EdgeSplit, ms: float) -> np.ndarray:
    img = (rgb * 0.45).astype(np.uint8)
    u, v = edges_uv[:, 0], edges_uv[:, 1]
    eo = split.edge_object
    img[v, u] = COL_BG_EDGE
    if len(dets):
        sus = np.array([is_suspect(d.prior) for d in dets])
        on = eo >= 0
        s = on & sus[np.clip(eo, 0, None)]
        img[v[on & ~s], u[on & ~s]] = COL_STATIC_OBJ
        img[v[s], u[s]] = COL_SUSPECT

    # outlines + labels for suspects only (prompt-free mode finds 20-40 objects per frame)
    for i, (d, m) in enumerate(zip(dets, masks)):
        if not is_suspect(d.prior):
            continue
        cnts, _ = cv2.findContours(m.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, COL_SUSPECT, 1, cv2.LINE_AA)
        tag = f"#{d.track_id} {d.class_name} P{d.prior:.1f}" + ("" if m.source == "depth" else " fb")
        x, y = d.box[0], max(d.box[1], 40)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), COL_SUSPECT, -1)
        _put(img, tag, (x + 3, y - 4), 0.42)

    n_sus = sum(is_suspect(d.prior) for d in dets)
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), COL_BAR, -1)
    _put(img, f"{MODE_LABELS[mode]}   det {len(dets)} (suspect {n_sus})   "
              f"X {100 * split.x_suspect:.0f}%   {ms:.0f} ms", (8, 17))
    return img


def compose(panels: list[np.ndarray], title: str) -> np.ndarray:
    row = np.hstack(panels)
    top = np.full((26, row.shape[1], 3), COL_BAR, np.uint8)
    _put(top, title, (8, 18), 0.5)
    bottom = np.full((26, row.shape[1], 3), COL_BAR, np.uint8)
    x = 8
    for colour, text in ((COL_SUSPECT, "edge on suspect object (P >= 0.3)  -  'fb' = depth cut-out fell back to the network mask"),
                         (COL_STATIC_OBJ, "edge on likely-static object"),
                         (COL_BG_EDGE, "background edge")):
        cv2.rectangle(bottom, (x, 8), (x + 12, 20), colour, -1)
        _put(bottom, text, (x + 18, 18), 0.45)
        x += 30 + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    return np.vstack([top, row, bottom])


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _rc():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def plot_sequence(series: dict[str, dict[str, np.ndarray]], metrics: dict[str, dict],
                  title: str, out):
    """series[mode] = {"t", "x_suspect", "n_suspect"}; metrics[mode] = the metrics json."""
    plt = _rc()
    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1])

    def smooth(y, k=15):
        if len(y) < k:
            return y
        return np.convolve(np.pad(y, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, "valid")

    for row, (key, ylabel, head) in enumerate((
            ("x_suspect", "% of edges", "X = share of edges on suspect objects"),
            ("n_suspect", "detections", "Suspect detections per frame (15-frame mean)"))):
        ax = fig.add_subplot(gs[row, 0])
        for mode, s in series.items():
            y = 100 * s[key] if key == "x_suspect" else smooth(s[key].astype(float))
            ax.plot(s["t"], y, color=MODE_COLOURS[mode], lw=1.6, label=MODE_LABELS[mode])
        ax.set_ylabel(ylabel)
        ax.set_title(head, loc="left", fontweight="bold")
        if row == 0:
            ax.legend(frameon=False, loc="upper right")
        else:
            ax.set_xlabel("time (s)")
        _style(ax)

    ax = fig.add_subplot(gs[:, 1])
    ax.axis("off")
    rows = [("", None),
            ("mean X (suspect edges)", lambda m: f"{100 * m['x_suspect']['mean']:.1f} %"),
            ("95th pct / max X", lambda m: f"{100 * m['x_suspect']['p95']:.0f} / {100 * m['x_suspect']['max']:.0f} %"),
            ("mean X (any object)", lambda m: f"{100 * m['x_any']['mean']:.1f} %"),
            ("frames with a suspect", lambda m: f"{m['frames_with_suspect_pct']:.0f} %"),
            ("detections / frame", lambda m: f"{m['detections_per_frame']:.1f}"),
            ("suspects / frame", lambda m: f"{m['suspects_per_frame']:.2f}"),
            ("suspect tracks started", lambda m: f"{m['suspect_tracks']}"),
            ("suspect masks from depth", lambda m: f"{m['suspect_mask_sources_pct'].get('depth', 0):.0f} %"),
            ("detect ms / frame", lambda m: f"{m['ms_per_frame']['detect']:.1f}"),
            ("mask + split ms / frame", lambda m: f"{m['ms_per_frame']['mask'] + m['ms_per_frame']['associate']:.1f}"),
            ("top suspect classes", None)]
    modes = list(metrics)
    xs = [0.52, 0.76, 1.0]
    for j, mode in enumerate(modes):
        ax.text(xs[j], 0.98, MODE_LABELS[mode].replace(" ", "\n", 1), transform=ax.transAxes,
                ha="right", va="top", color=INK, fontweight="bold")
    for i, (label, fn) in enumerate(rows[1:]):
        y = 0.90 - i * 0.055
        ax.text(0.0, y, label, transform=ax.transAxes, va="top", color=MUTED)
        if fn is None:
            continue
        for j, mode in enumerate(modes):
            ax.text(xs[j], y, fn(metrics[mode]), transform=ax.transAxes, va="top", ha="right",
                    color=INK, fontweight="bold")
    y = 0.90 - (len(rows) - 1) * 0.055
    for j, mode in enumerate(modes):
        top = list(metrics[mode]["suspect_class_counts"].items())[:6]
        txt = "\n".join(f"{k} {v}" for k, v in top) or "-"
        ax.text(0.0 + j * 0.34, y - 0.01, f"{MODE_LABELS[mode]}:\n{txt}", transform=ax.transAxes,
                va="top", color=INK, fontsize=8)

    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_summary(rows: list[dict], out):
    """rows: one dict per (sequence, mode) with sequence, mode, x_mean, frames_with_suspect_pct."""
    plt = _rc()
    seqs = list(dict.fromkeys(r["sequence"] for r in rows))
    modes = [m for m in MODE_COLOURS if any(r["mode"] == m for r in rows)]
    lookup = {(r["sequence"], r["mode"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.32 * len(seqs) * len(modes) / 2 + 2.5),
                             sharey=True, constrained_layout=True)
    bar_h = 0.8 / len(modes)
    yy = np.arange(len(seqs))
    for ax, key, head, scale in ((axes[0], "x_mean", "Mean X: share of edges on suspect objects (%)", 100),
                                 (axes[1], "frames_with_suspect_pct", "Frames with at least one suspect (%)", 1)):
        for j, mode in enumerate(modes):
            vals = [scale * lookup[(s, mode)][key] if (s, mode) in lookup else np.nan for s in seqs]
            ax.barh(yy + (j - (len(modes) - 1) / 2) * bar_h, vals, height=bar_h * 0.9,
                    color=MODE_COLOURS[mode], label=MODE_LABELS[mode], edgecolor="white", linewidth=1)
        ax.set_title(head, loc="left", fontweight="bold")
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(yy, [s.replace("rgbd_dataset_", "").replace("rgbd_", "") for s in seqs])
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Moving-object detection: all sequences, three detector modes", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.savefig(out, dpi=120)
    plt.close(fig)
