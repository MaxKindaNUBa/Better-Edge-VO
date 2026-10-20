"""Run boxes 1-4 (prepare, edges, detect objects, cut them out with depth, split edges)
with three detector modes side by side on TUM / Bonn sequences.

Usage (from the project root):
    conda run -n 312ml python -m src.run_detection                -> dataset selector window (pick one or more)
    conda run -n 312ml python -m src.run_detection datasets/tum/rgbd_dataset_freiburg3_walking_xyz [more ...]
    conda run -n 312ml python -m src.run_detection --all          -> every sequence under datasets/,
                                                                     one after another
    options: --start N --frames N     part of each sequence
             --modes coco,yoloe_pf    subset of coco / yoloe_text / yoloe_pf
             --no-video               skip detection_video.mp4
             --out-dir DIR            where the run folder is made (default results/)
             --summary-only RUN_DIR   only rebuild summary.* of an earlier run folder

No VO runs here: this measures what the detectors find and which edges they claim.

Every run gets its own folder results/<YYYY-MM-DD_HH-MM-SS>_run_detection/ holding
    run_info.json         command, options, sequences, start / end time, per-sequence status
    log.txt               the console output
    prior_table.csv       every class each detector can output, with its prior
    summary.csv / summary_plot.png   across the sequences of this run
and per sequence, in <run folder>/<sequence>/  (per mode: coco_*, yoloe_text_*, yoloe_pf_*):
    <mode>_metrics.json   X statistics, detections, mask sources, classes, timings
    <mode>_frames.csv     one row per frame (edges, detections, X, timings)
    <mode>_objects.csv    one row per detection (track ID, class, prior, box, mask source, edges)
    detection_plot.png    X and suspect count over time for the three modes + summary table
    detection_video.mp4   the three modes side by side, edges coloured by owner
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.dataset import RGBDSequence  # noqa: E402
from src.common.dataset_picker import find_sequences, sequences_from_args_or_gui  # noqa: E402
from src.common.run_dir import RunDir  # noqa: E402
from src.common.edges import EdgeConfig, depth_discontinuities, detect_edges  # noqa: E402
from src.common.preprocess import PrepConfig, prepare_frame  # noqa: E402
from src.common.video import VideoSink  # noqa: E402
from src.detection.associate import split_edges  # noqa: E402
from src.detection.detector import MODES, TEXT_PROMPT, Detector, IoUTracker  # noqa: E402
from src.detection.object_mask import MaskConfig, cut_out  # noqa: E402
from src.detection.priors import DEFAULT_PRIOR, SUSPECT_THRESHOLD, group_of, is_suspect, prior_for  # noqa: E402
from src.detection.visualise import compose, plot_sequence, plot_summary, render_panel  # noqa: E402

FRAME_COLS = ["frame", "timestamp", "edges", "detections", "suspects", "scene_tags", "suspect_edges",
              "object_edges", "x_suspect", "x_any", "suspect_masks_depth", "suspect_masks_fallback",
              "ms_detect", "ms_mask", "ms_associate"]
OBJ_COLS = ["frame", "timestamp", "track_id", "class", "score", "prior", "suspect", "scene_tag",
            "x1", "y1", "x2", "y2", "mask_source", "z0_m", "depth_coverage", "mask_px", "edges"]


def stats(a: np.ndarray) -> dict:
    a = np.asarray(a, float)
    if not len(a):
        return {}
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max())}


def run_sequence(seq_dir: Path, out_dir: Path, detectors: dict[str, Detector], args) -> list[dict]:
    seq = RGBDSequence(seq_dir)
    pcfg = PrepConfig(pyramid_levels=1)
    ecfg = EdgeConfig(min_high=25.0)     # same edges DTVO tracks (see DTVOConfig)
    mcfg = MaskConfig()
    end = len(seq) if args.frames <= 0 else min(len(seq), args.start + args.frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {seq_dir.name}: frames {args.start}..{end - 1} of {len(seq)}", flush=True)

    for d in detectors.values():
        d.tracker = IoUTracker()
    fps_video = float(np.clip(round((end - args.start - 1) / max(
        seq.frames[end - 1][0] - seq.frames[args.start][0], 1e-6)), 1, 60))
    sink = None

    files, writers = {}, {}
    for mode in detectors:
        for kind, cols in (("frames", FRAME_COLS), ("objects", OBJ_COLS)):
            f = open(out_dir / f"{mode}_{kind}.csv", "w", newline="")
            files[(mode, kind)] = f
            writers[(mode, kind)] = csv.writer(f)
            writers[(mode, kind)].writerow(cols)

    acc = {m: {"t": [], "x_suspect": [], "x_any": [], "n_det": [], "n_suspect": [],
               "ms": Counter(), "src": Counter(), "cls_suspect": Counter(), "cls_all": Counter(),
               "suspect_ids": set()} for m in detectors}
    t_prep = t_edge = 0.0
    t_start = time.perf_counter()
    t0_seq = seq.frames[args.start][0]

    for i in range(args.start, end):
        raw = seq[i]
        ta = time.perf_counter()
        frame = prepare_frame(raw, seq.camera, pcfg)
        tb = time.perf_counter()
        em = detect_edges(frame.full, ecfg)
        tc = time.perf_counter()
        t_prep += tb - ta
        t_edge += tc - tb
        disc = None
        panels = []

        for mode, det in detectors.items():
            a = acc[mode]
            t1 = time.perf_counter()
            dets = det(frame.rgb)
            t2 = time.perf_counter()
            if disc is None:   # shared by all modes; timed as part of the first mode's mask step
                disc = depth_discontinuities(frame.full.depth, frame.full.valid, ecfg, 0)
            masks = [cut_out(d, frame.full, disc, mcfg, refine=is_suspect(d.prior)) for d in dets]
            t3 = time.perf_counter()
            split = split_edges(em, dets, masks)
            t4 = time.perf_counter()
            ms = {"detect": 1000 * (t2 - t1), "mask": 1000 * (t3 - t2), "associate": 1000 * (t4 - t3)}
            a["ms"].update(ms)

            sus = [is_suspect(d.prior) for d in dets]
            n_sus = sum(sus)
            src_sus = Counter(m.source for m, s in zip(masks, sus) if s)
            a["src"].update(src_sus)
            for d, s in zip(dets, sus):
                a["cls_all"][d.class_name] += 1
                if s:
                    a["cls_suspect"][d.class_name] += 1
                    a["suspect_ids"].add(d.track_id)
            a["t"].append(raw.timestamp - t0_seq)
            a["x_suspect"].append(split.x_suspect)
            a["x_any"].append(split.x_any)
            a["n_det"].append(len(dets))
            a["n_suspect"].append(n_sus)

            writers[(mode, "frames")].writerow(
                [i, f"{raw.timestamp:.6f}", split.n_edges, len(dets), n_sus, int(split.scene_tag.sum()),
                 split.n_suspect_edges, split.n_object_edges, f"{split.x_suspect:.4f}", f"{split.x_any:.4f}",
                 src_sus.get("depth", 0), n_sus - src_sus.get("depth", 0),
                 f"{ms['detect']:.2f}", f"{ms['mask']:.2f}", f"{ms['associate']:.2f}"])
            ow = writers[(mode, "objects")]
            for k, (d, m) in enumerate(zip(dets, masks)):
                ow.writerow([i, f"{raw.timestamp:.6f}", d.track_id, d.class_name, f"{d.score:.3f}",
                             d.prior, int(sus[k]), int(split.scene_tag[k]), *d.box, m.source,
                             f"{m.z0:.3f}", f"{m.depth_coverage:.3f}", int(m.mask.sum()),
                             int(split.edges_per_object[k])])

            if not args.no_video:
                panels.append(render_panel(frame.rgb, em.uv, mode, dets, masks, split, sum(ms.values())))

        if panels:
            img = compose(panels, f"{seq_dir.name}   frame {i}   t = {raw.timestamp - t0_seq:6.1f} s")
            if sink is None:
                sink = VideoSink(out_dir / "detection_video.mp4", img.shape[1], img.shape[0], fps_video, crf=23)
            sink.write(img)
        n = i - args.start + 1
        if n % 100 == 0 or i == end - 1:
            el = time.perf_counter() - t_start
            xs = "  ".join(f"{m} X {100 * acc[m]['x_suspect'][-1]:4.1f}%" for m in detectors)
            print(f"  {n}/{end - args.start}  {n / el:.1f} fps  {xs}", flush=True)

    if sink is not None:
        sink.close()
    for f in files.values():
        f.close()

    n = end - args.start
    summary_rows, series, metrics_all = [], {}, {}
    for mode, a in acc.items():
        x = np.array(a["x_suspect"])
        src_total = sum(a["src"].values())
        metrics = {
            "sequence": seq_dir.name, "mode": mode, "frames": n,
            "fps_detection_pipeline": n / max(sum(a["ms"].values()) / 1000, 1e-9),
            "ms_per_frame": {"prepare (shared)": 1000 * t_prep / n, "edges (shared)": 1000 * t_edge / n,
                             **{k: v / n for k, v in a["ms"].items()}},
            "x_suspect": stats(x), "x_any": stats(a["x_any"]),
            "frames_with_suspect_pct": 100 * float(np.mean(np.array(a["n_suspect"]) > 0)),
            "detections_per_frame": float(np.mean(a["n_det"])),
            "suspects_per_frame": float(np.mean(a["n_suspect"])),
            "suspect_tracks": len(a["suspect_ids"]),
            "tracks_started_all": detectors[mode].tracker.created,
            "suspect_mask_sources": dict(a["src"]),
            "suspect_mask_sources_pct": {k: 100 * v / max(src_total, 1) for k, v in a["src"].items()},
            "suspect_class_counts": dict(a["cls_suspect"].most_common(30)),
            "all_class_counts_top30": dict(a["cls_all"].most_common(30)),
            "settings": {"suspect_threshold": SUSPECT_THRESHOLD, "default_prior": DEFAULT_PRIOR,
                         "detector": vars(detectors[mode].cfg), "mask": vars(mcfg),
                         "canny_min_high": ecfg.min_high,
                         "text_prompt": TEXT_PROMPT if mode == "yoloe_text" else None},
        }
        (out_dir / f"{mode}_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
        metrics_all[mode] = metrics
        series[mode] = {"t": np.array(a["t"]), "x_suspect": x, "n_suspect": np.array(a["n_suspect"])}
        summary_rows.append(summary_row(metrics))
        print(f"  {mode:10s}  mean X {100 * metrics['x_suspect']['mean']:5.1f}%  "
              f"frames w/ suspect {metrics['frames_with_suspect_pct']:5.1f}%  "
              f"det {metrics['detections_per_frame']:5.1f}/frame  "
              f"detect {metrics['ms_per_frame']['detect']:5.1f} ms  mask {metrics['ms_per_frame']['mask']:5.1f} ms",
              flush=True)

    plot_sequence(series, metrics_all, f"Moving-object detection on {seq_dir.name} ({n} frames)",
                  out_dir / "detection_plot.png")
    print(f"  wrote {out_dir}", flush=True)
    return summary_rows


def summary_row(m: dict) -> dict:
    return {"sequence": m["sequence"], "mode": m["mode"], "frames": m["frames"],
            "x_mean": m["x_suspect"]["mean"], "x_p95": m["x_suspect"]["p95"], "x_max": m["x_suspect"]["max"],
            "x_any_mean": m["x_any"]["mean"], "frames_with_suspect_pct": m["frames_with_suspect_pct"],
            "detections_per_frame": m["detections_per_frame"], "suspects_per_frame": m["suspects_per_frame"],
            "suspect_tracks": m["suspect_tracks"],
            "suspect_masks_depth_pct": m["suspect_mask_sources_pct"].get("depth", 0.0),
            "ms_detect": m["ms_per_frame"]["detect"], "ms_mask": m["ms_per_frame"]["mask"],
            "ms_associate": m["ms_per_frame"]["associate"]}


def write_summary(run_dir: Path):
    rows = []
    for p in sorted(run_dir.glob("*/*_metrics.json")):
        rows.append(summary_row(json.loads(p.read_text())))
    if not rows:
        return
    order = {m: k for k, m in enumerate(MODES)}
    rows.sort(key=lambda r: (r["sequence"], order[r["mode"]]))
    with open(run_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot_summary(rows, run_dir / "summary_plot.png")
    print(f"wrote {run_dir / 'summary.csv'} and summary_plot.png ({len(rows)} rows)")


def write_prior_table(detectors: dict[str, Detector], run_dir: Path):
    """Every class name each detector can output, with the prior it gets (for review)."""
    with open(run_dir / "prior_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "class", "prior", "group", "suspect"])
        for mode, det in detectors.items():
            for name in det.model.names.values():
                p = prior_for(name)
                w.writerow([mode, name, p, group_of(name), int(is_suspect(p))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", type=Path, nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0, help="0 = whole sequence")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None, help="parent of the run folder (default results/)")
    ap.add_argument("--summary-only", type=Path, default=None, metavar="RUN_DIR")
    args = ap.parse_args()

    if args.summary_only:
        write_summary(args.summary_only)
        return
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    if args.all:
        # an archive next to a folder means the download script is still extracting it
        seqs = [s.path for s in find_sequences()
                if not any(s.path.with_suffix(ext).exists() for ext in (".zip", ".tgz"))]
    else:
        seqs = sequences_from_args_or_gui(args.sequence, "Detection - select sequences")

    with RunDir("run_detection", args, seqs, args.out_dir) as run:
        print("loading detectors: " + ", ".join(modes), flush=True)
        detectors = {m: Detector(m) for m in modes}
        write_prior_table(detectors, run.dir)
        for s in seqs:
            with run.sequence(s) as out_dir:
                run_sequence(s, out_dir, detectors, args)
            write_summary(run.dir)
    if run.failed:
        sys.exit(f"{len(run.failed)} of {len(seqs)} sequences failed: {', '.join(run.failed)}")


if __name__ == "__main__":
    main()
