"""Presentation video: every detected edge, with moving objects and their edges marked.

    edges (as DTVO sees them) -> YOLO-seg detection -> depth cut-out -> edges split per object
    -> objects with prior >= --moving-prior (default 0.7: people) are labelled MOVING and
       their edges drawn red; all other edges white

There is no motion check yet, so "moving" means "a class that moves" (people), not measured
motion. Lower-prior suspects (chairs, P = 0.3) are not marked here, although masked DTVO
does leave their edges out.

Usage (from the project root):
    conda run -n 312ml python -m src.demo_moving_edges          -> dataset selector window
    conda run -n 312ml python -m src.demo_moving_edges datasets/tum/rgbd_dataset_freiburg3_walking_xyz
        -> output_videos/moving_edges_freiburg3_walking_xyz.mp4   (whole sequence)
    options: --start N --frames N   part of the sequence
             --mode coco            detector: coco / yoloe_text / yoloe_pf
             --moving-prior P       objects with prior >= P are marked moving (default 0.7)
             --scale S              output size = S x camera image (default 2)
             --out path.mp4         different output file
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # run as a plain file, not via `python -m src.demo_moving_edges`
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.dataset import RGBDSequence  # noqa: E402
from src.common.dataset_picker import sequence_from_args_or_gui  # noqa: E402
from src.common.edges import detect_edges_pyramid  # noqa: E402
from src.common.preprocess import prepare_frame  # noqa: E402
from src.common.video import VideoSink  # noqa: E402
from src.detection.detector import MODES, Detector  # noqa: E402
from src.detection.masking import MovingObjectMask  # noqa: E402
from src.dtvo.system import DTVOConfig  # noqa: E402

# BGR
COL_EDGE = (235, 235, 235)
COL_MOVING = (72, 73, 227)     # #e34948, as the detection videos
COL_TEXT = (245, 245, 245)
COL_BAR = (24, 24, 24)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put(img, text, org, scale=0.6, colour=COL_TEXT, thick=1):
    cv2.putText(img, text, org, FONT, scale, colour, thick, cv2.LINE_AA)


def render(rgb, edges_uv, filt: MovingObjectMask, s: int) -> tuple[np.ndarray, int]:
    """Dimmed frame + edges at s x size (edges drawn first, then upscaled: s px thick)."""
    img = (rgb * 0.4).astype(np.uint8)
    u, v = edges_uv[:, 0], edges_uv[:, 1]
    moving = np.array(filt.masked, bool)
    eo = filt.split.edge_object
    on = (eo >= 0) & moving[np.clip(eo, 0, None)] if len(moving) else np.zeros(len(u), bool)
    img[v, u] = COL_EDGE
    img[v[on], u[on]] = COL_MOVING
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)

    for d, m, k in zip(filt.dets, filt.masks, moving):
        if not k:
            continue
        cnts, _ = cv2.findContours(m.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, [c * s for c in cnts], -1, COL_MOVING, 2, cv2.LINE_AA)
        n = int(filt.split.edges_per_object[filt.dets.index(d)])
        tag = f"MOVING  {d.class_name} #{d.track_id}  ({n:,} edges)"
        (tw, th), _ = cv2.getTextSize(tag, FONT, 0.6, 2)
        x, y = d.box[0] * s, max(d.box[1] * s, 40 + th + 10)
        x = min(x, img.shape[1] - tw - 12)
        cv2.rectangle(img, (x, y - th - 12), (x + tw + 12, y), COL_MOVING, -1)
        _put(img, tag, (x + 6, y - 7), 0.6, thick=2)
    return img, int(on.sum())


def bars(img, title: str, n_edges: int, n_moving: int) -> np.ndarray:
    w = img.shape[1]
    top = np.full((40, w, 3), COL_BAR, np.uint8)
    _put(top, title, (12, 27), 0.65)
    stat = f"edges {n_edges:,}   on moving objects {n_moving:,} ({100 * n_moving / max(n_edges, 1):.0f}%)"
    (tw, _), _ = cv2.getTextSize(stat, FONT, 0.65, 1)
    _put(top, stat, (w - tw - 12, 27), 0.65)
    bottom = np.full((36, w, 3), COL_BAR, np.uint8)
    x = 12
    for colour, text in ((COL_EDGE, "static scene edge (used for tracking)"),
                         (COL_MOVING, "edge on a moving object (left out of tracking)")):
        cv2.rectangle(bottom, (x, 11), (x + 16, 25), colour, -1)
        _put(bottom, text, (x + 24, 24), 0.6)
        x += 60 + cv2.getTextSize(text, FONT, 0.6, 1)[0][0]
    return np.vstack([top, img, bottom])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", type=Path, nargs="?", default=None,
                    help="sequence folder (omit to choose in the selector window)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0, help="0 = whole sequence")
    ap.add_argument("--mode", default="coco", choices=MODES, help="detector")
    ap.add_argument("--moving-prior", type=float, default=0.7, help="mark objects with prior >= this")
    ap.add_argument("--scale", type=int, default=2, help="output size = scale x camera image")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    seq_dir = sequence_from_args_or_gui(args.sequence, "Moving-object edges: choose a sequence")
    seq = RGBDSequence(seq_dir)
    name = seq_dir.name.replace("rgbd_dataset_", "").replace("rgbd_", "")
    out = args.out or PROJECT_ROOT / "output_videos" / f"moving_edges_{name}.mp4"
    end = len(seq) if args.frames <= 0 else min(len(seq), args.start + args.frames)
    ecfg = DTVOConfig().edges   # the edges DTVO tracks with
    filt = MovingObjectMask(Detector(args.mode), min_prior=args.moving_prior, edges=ecfg)
    s = args.scale
    w, h = seq.camera.width * s, seq.camera.height * s + 40 + 36
    fps = (end - args.start - 1) / max(seq[end - 1].timestamp - seq[args.start].timestamp, 1e-6)
    sink = VideoSink(out, w, h, fps)
    print(f"{seq_dir.name}: frames {args.start}..{end - 1} -> {out}")

    t0, share = time.perf_counter(), []
    for k in range(args.start, end):
        frame = prepare_frame(seq[k], seq.camera)
        edges = detect_edges_pyramid(frame, ecfg)[0]
        filt(frame, edges)
        img, n_moving = render(frame.rgb, edges.uv, filt, s)
        share.append(n_moving / max(len(edges.uv), 1))
        sink.write(bars(img, f"{name}   frame {k + 1}/{len(seq)}", len(edges.uv), n_moving))
        if (k - args.start + 1) % 100 == 0:
            print(f"  {k - args.start + 1}/{end - args.start}  {(k - args.start + 1) / (time.perf_counter() - t0):.1f} fps")
    sink.close()
    share = np.array(share)
    print(f"wrote {out}  ({len(share)} frames, {fps:.0f} fps)\n"
          f"edges on moving objects: mean {100 * share.mean():.1f} %, max {100 * share.max():.1f} %, "
          f"frames with a moving object {100 * (share > 0).mean():.0f} %")


if __name__ == "__main__":
    main()
