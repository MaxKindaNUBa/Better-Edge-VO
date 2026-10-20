"""Run boxes 1-2 on a sequence: save an overlay video and print edge statistics.

Usage (from the project root):
    conda run -n 312ml python -m src.demo_edges          -> dataset selector window
    conda run -n 312ml python -m src.demo_edges datasets/tum/rgbd_dataset_freiburg3_walking_xyz
        -> output_videos/edges_freiburg3_walking_xyz.mp4   (whole sequence)
    options: --start N --frames N   part of the sequence
             --out path.mp4         different output file
             --no-video             statistics only

Can also be run directly as a file (e.g. the IDE Run button, with the 312ml interpreter
selected); with no sequence argument it opens the dataset selector.

Overlay colours:
    cyan    edge with reliable depth   -> becomes a 3D reference point
    yellow  edge with no depth reading -> distance map only
    magenta edge on a depth jump       -> distance map only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # run as a plain file, not via `python -m src.demo_edges`
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.dataset import RGBDSequence  # noqa: E402
from src.common.dataset_picker import sequence_from_args_or_gui  # noqa: E402
from src.common.edges import EDGE_DEPTH, EDGE_DEPTH_JUMP, EDGE_NO_DEPTH, EdgeConfig, detect_edges_pyramid  # noqa: E402
from src.common.preprocess import PrepConfig, prepare_frame  # noqa: E402
from src.common.video import VideoSink  # noqa: E402

COLOURS = {EDGE_DEPTH: (255, 255, 0), EDGE_NO_DEPTH: (0, 220, 255), EDGE_DEPTH_JUMP: (255, 0, 255)}


def depth_view(depth: np.ndarray, valid: np.ndarray, max_depth: float) -> np.ndarray:
    img = cv2.applyColorMap((np.clip(depth / max_depth, 0, 1) * 255).astype(np.uint8),
                            cv2.COLORMAP_TURBO)
    img[~valid] = 0
    return img


def render(frame, edge_maps, max_depth: float, idx: int) -> np.ndarray:
    full, em = frame.full, edge_maps[0]
    overlay = (frame.rgb * 0.6).astype(np.uint8)
    for lab, col in COLOURS.items():
        overlay[em.labels == lab] = col
    panel = np.hstack([overlay, depth_view(full.depth, full.valid, max_depth)])

    c = em.counts()
    pct = lambda k: 100.0 * c[k] / max(c["total"], 1)
    text = (f"#{idx}  edges {c['total']}  depth {pct('depth'):.0f}%  "
            f"no-depth {pct('no_depth'):.0f}%  jump {pct('depth_jump'):.0f}%  "
            f"canny {em.thresholds[0]:.0f}/{em.thresholds[1]:.0f}")
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(panel, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "cyan: depth  yellow: no depth  magenta: depth jump",
                (8, panel.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", type=Path, nargs="?", default=None,
                    help="sequence folder (omit to choose in the selector window)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0, help="0 = whole sequence")
    ap.add_argument("--out", type=Path, default=None,
                    help="output video (default: output_videos/edges_<sequence>.mp4)")
    ap.add_argument("--no-video", action="store_true", help="print statistics only")
    args = ap.parse_args()
    args.sequence = sequence_from_args_or_gui(args.sequence, "Edge demo - select a sequence")

    seq = RGBDSequence(args.sequence)
    pcfg, ecfg = PrepConfig(), EdgeConfig()
    end = len(seq) if args.frames <= 0 else min(len(seq), args.start + args.frames)
    print(f"{args.sequence.name}: {len(seq)} associated frames, processing {args.start}..{end - 1}")

    out = None
    if not args.no_video:
        short = args.sequence.name.replace("rgbd_dataset_", "").replace("rgbd_", "")
        out = args.out or PROJECT_ROOT / "output_videos" / f"edges_{short}.mp4"
    # Play back at the sequence's real frame rate (TUM / Bonn record at ~30 Hz)
    fps = (end - args.start - 1) / max(seq.frames[end - 1][0] - seq.frames[args.start][0], 1e-6)
    fps = float(np.clip(round(fps), 1, 60))

    writer = None
    t_prep = t_edge = 0.0
    stats = []
    for i in range(args.start, end):
        raw = seq[i]
        t0 = time.perf_counter()
        frame = prepare_frame(raw, seq.camera, pcfg)
        t1 = time.perf_counter()
        edge_maps = detect_edges_pyramid(frame, ecfg)
        t2 = time.perf_counter()
        t_prep += t1 - t0
        t_edge += t2 - t1

        c = edge_maps[0].counts()
        stats.append([c["total"], c["depth"], c["no_depth"], c["depth_jump"],
                      100.0 * (~frame.full.valid).mean(), *edge_maps[0].thresholds])

        if out:
            img = render(frame, edge_maps, pcfg.max_depth, i)
            if writer is None:
                writer = VideoSink(out, img.shape[1], img.shape[0], fps)
            writer.write(img)
        if (i - args.start + 1) % 100 == 0:
            print(f"  {i - args.start + 1}/{end - args.start} frames")

    if writer is not None:
        writer.close()
        print(f"wrote {out}  ({fps:.0f} fps)")

    s = np.array(stats)
    n = len(s)
    tot = s[:, 0].sum()
    print(f"frames {n} | edges/frame {s[:, 0].mean():.0f} | "
          f"depth {100 * s[:, 1].sum() / tot:.1f}%  no-depth {100 * s[:, 2].sum() / tot:.1f}%  "
          f"jump {100 * s[:, 3].sum() / tot:.1f}% | invalid depth px {s[:, 4].mean():.1f}%")
    print(f"canny high threshold: mean {s[:, 6].mean():.0f}, range {s[:, 6].min():.0f}-{s[:, 6].max():.0f}")
    print(f"time per frame: prepare {1000 * t_prep / n:.1f} ms, edges (3 levels) {1000 * t_edge / n:.1f} ms")


if __name__ == "__main__":
    main()
