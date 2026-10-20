"""DTVO with the edges of moving objects masked out (boxes 1-4 + DTVO), compared with
plain DTVO on the same frames.

    frame -> edges -> YOLO-seg (COCO by default) -> depth cut-out (box 3)
          -> objects with prior >= --min-prior are masked: their edges are removed on
             every pyramid level before the DT map / keyframes (no motion check yet)
          -> DTVO tracking + window optimisation on the remaining edges
    plain DTVO on all edges runs alongside as the baseline (--no-baseline to skip)

Keyframe guards (both runs, --no-guard switches them off for the ablation):
    edge budget     if masking leaves too few edges with depth to anchor a keyframe, masked
                    objects are given back, least likely movers first (never people)
    keyframe check  a frame with too few / too clustered depth edges never becomes a
                    keyframe: the old one is kept (while lost: keep coasting)
    shift check     a window optimisation that moves any keyframe > 10 cm / 10 deg is undone

Usage (from the project root):
    conda run -n 312ml python -m src.run_masked_dtvo                 -> dataset selector window
    conda run -n 312ml python -m src.run_masked_dtvo datasets/bonn/rgbd_bonn_balloon [more ...]
    conda run -n 312ml python -m src.run_masked_dtvo --all           -> every sequence under datasets/
    options: --start N --frames N    part of each sequence
             --mode coco             detector: coco / yoloe_text / yoloe_pf
             --min-prior P           mask objects with prior >= P (default 0.3 = "suspect")
             --no-baseline           don't run plain DTVO for comparison (half the VO time)
             --no-guard              keyframe guards off (edge budget, keyframe check, shift check)
             --no-mapping            tracking only (both runs)
             --no-weights            plain alignment (both runs)
             --no-show               only save the figures, don't open a window
                                     (always the case when more than one sequence is run)
             --no-video              skip masked_dtvo_video.mp4
             --out-dir DIR           where the run folder is made (default results/)
             --summary-only RUN_DIR  only rebuild summary.* of an earlier run folder

Every run gets its own folder results/<YYYY-MM-DD_HH-MM-SS>_run_masked_dtvo/ holding
    run_info.json         command, options, sequences, start / end time, per-sequence status
    log.txt               the console output
    prior_table.csv       every class the detector can output, with its prior
    summary.csv / summary_plot.png   ATE masked vs plain DTVO and masked edge share, per sequence
and per sequence, in <run folder>/<sequence>/:
    masked_dtvo_trajectory.txt   estimated poses, TUM format
    baseline_trajectory.txt      plain DTVO poses, TUM format
    masked_dtvo_plot.png  x-y path, x / y / z and roll / pitch / yaw vs ground truth for both
                          runs, with RMSE (as dtvo_plot.png)
    masking_plot.png      position error over time for both runs, share of edges masked,
                          matched edges, detection summary table
    masked_dtvo_metrics.json   ATE / RPE / per-axis RMSE, timings, keyframes for both runs;
                          detection statistics (as <mode>_metrics.json of run_detection)
    frames.csv            one row per frame: detections, edges masked, X, tracking state and
                          position error of both runs, timings
    objects.csv           one row per detection (track ID, class, prior, masked, box, mask, edges)
    masked_dtvo_video.mp4 edges + masked objects + tracked keyframe edges (left) next to the
                          x-y path drawn live against ground truth and plain DTVO (right)
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
if str(PROJECT_ROOT) not in sys.path:  # run as a plain file, not via `python -m src.run_masked_dtvo`
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.dataset import RGBDSequence  # noqa: E402
from src.common.dataset_picker import find_sequences, sequences_from_args_or_gui  # noqa: E402
from src.common.run_dir import RunDir  # noqa: E402
from src.common.video import VideoSink  # noqa: E402
from src.detection.detector import MODE_LABELS, MODES, Detector, IoUTracker  # noqa: E402
from src.detection.masking import MovingObjectMask  # noqa: E402
from src.detection.priors import DEFAULT_PRIOR, SUSPECT_THRESHOLD, is_suspect  # noqa: E402
from src.dtvo.evaluation import evaluate, load_groundtruth, write_tum  # noqa: E402
from src.dtvo.system import DTVO, DTVOConfig  # noqa: E402
from src.dtvo.visualise import (GRID, INK, MUTED, RUN_COLOURS, TrajectoryPanel, compose_video,  # noqa: E402
                                plot_evaluation, tracking_overlay)
from src.run_detection import OBJ_COLS, stats, write_prior_table  # noqa: E402

MASKED, BASELINE = "Masked DTVO", "DTVO, all edges"

FRAME_COLS = ["frame", "timestamp", "edges", "edges_used", "masked_edges", "x_masked",
              "detections", "masked_objects", "suspects", "scene_tags", "suspect_edges", "object_edges",
              "x_suspect", "x_any", "suspect_masks_depth", "suspect_masks_fallback",
              "depth_edges", "objects_unmasked", "keyframe_id", "is_keyframe", "lost", "kf_refused", "weak_kf",
              "ba_rejected", "hypothesis", "matched_edges", "pos_err_m",
              "baseline_keyframe_id", "baseline_lost", "baseline_kf_refused", "baseline_ba_rejected",
              "baseline_matched_edges", "baseline_pos_err_m",
              "ms_detect", "ms_mask", "ms_associate", "ms_vo", "ms_baseline"]


def guard_off(cfg: DTVOConfig):
    """Keyframe guards off: any frame may anchor a keyframe, nothing is given back, any
    window-optimisation step is kept (the re-anchor after a lost run still starts from the
    motion-model pose)."""
    c = cfg.keyframe
    c.min_ref_points = c.min_top_points = c.min_cells = 0
    c.unmask_below_prior = 0.0
    cfg.mapping.max_kf_shift_m = cfg.mapping.max_kf_shift_deg = float("inf")


def vo_metrics(vo: DTVO, wall: float, cfg: DTVOConfig) -> dict:
    n = len(vo.records)
    hyp = Counter(r.hypothesis for r in vo.records[1:])
    st = vo.mapper.stats
    return {"frames": n, "keyframes": len(vo.keyframes),
            "frames_lost": int(sum(bool(r.lost) for r in vo.records)), "fps": n / wall,
            "keyframes_refused": int(sum(r.kf_refused for r in vo.records)),
            "weak_keyframes": int(sum(r.weak_kf for r in vo.records)),
            "window_optimisations": st["ba_runs"], "window_optimisations_rejected": st["ba_rejected"],
            "rejected_max_shift_m": st["ba_rejected_shift_m"],
            "frames_with_unmasking": int(sum(r.n_unmasked > 0 for r in vo.records)),
            "objects_unmasked": int(sum(r.n_unmasked for r in vo.records)),
            "min_depth_edges": int(min(r.n_ref_points for r in vo.records)),
            "ms_per_frame": {k: 1000 * v / n for k, v in vo.times.items()},
            "initial_hypothesis_chosen": dict(hyp),
            "mean_matched_edges": float(np.mean([r.inlier_ratio for r in vo.records])),
            "mean_edges_used": float(np.mean([r.n_edges for r in vo.records])),
            "weighted_alignment": cfg.track.weighted, "window_optimisation": cfg.mapping.enabled,
            "keyframe_guards": {**{k: getattr(cfg.keyframe, k) for k in (
                "min_ref_points", "min_top_points", "min_cells", "cell_share", "unmask_below_prior")},
                "max_kf_shift_m": cfg.mapping.max_kf_shift_m, "max_kf_shift_deg": cfg.mapping.max_kf_shift_deg}}


def ev_metrics(ev) -> dict:
    return {"ate_rmse_m": ev.ate_rmse, "ate_mean_m": ev.ate_mean, "ate_max_m": ev.ate_max,
            "rot_rmse_deg": ev.rot_rmse_deg,
            "start_aligned_rmse_m": ev.start_rmse, "start_aligned_rot_rmse_deg": ev.start_rot_rmse_deg,
            "start_aligned_rmse_xyz_m": ev.axis_rmse.tolist(),
            "start_aligned_rmse_roll_pitch_yaw_deg": ev.rpy_rmse.tolist(),
            "rpe_trans_m_per_s": ev.rpe_trans, "rpe_rot_deg_per_s": ev.rpe_rot,
            "gt_path_length_m": ev.gt_length, "matched_frames": ev.matched,
            "gt_body_rotation_compensated_deg": ev.body_rotation_deg}


def print_ev(name: str, ev):
    print(f"{name}:")
    print(f"  best fit (TUM):  ATE RMSE {ev.ate_rmse:.4f} m, rotation RMSE {ev.rot_rmse_deg:.2f} deg")
    print(f"  start-aligned:   position RMSE {ev.start_rmse:.4f} m (x {ev.axis_rmse[0]:.4f}, "
          f"y {ev.axis_rmse[1]:.4f}, z {ev.axis_rmse[2]:.4f}), rotation RMSE {ev.start_rot_rmse_deg:.2f} deg "
          f"(roll {ev.rpy_rmse[0]:.2f}, pitch {ev.rpy_rmse[1]:.2f}, yaw {ev.rpy_rmse[2]:.2f})")
    print(f"  RPE per {ev.rpe_delta:g} s: {ev.rpe_trans:.4f} m, {ev.rpe_rot:.2f} deg")


def per_frame(ev, poses, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Start-aligned positions of every frame, and their error (NaN where no ground truth)."""
    xyz = np.array([(ev.anchor @ T)[:3, 3] for T in poses])
    gt = np.full((n, 3), np.nan)
    gt[ev.est_index] = ev.gt_xyz
    return xyz, np.linalg.norm(xyz - gt, axis=1)


def run_sequence(seq_dir: Path, out_dir: Path, detector: Detector, args, show: bool) -> None:
    seq = RGBDSequence(seq_dir)
    cfg = DTVOConfig()
    cfg.mapping.enabled = not args.no_mapping
    cfg.track.weighted = not args.no_weights
    if args.no_guard:
        guard_off(cfg)
    end = len(seq) if args.frames <= 0 else min(len(seq), args.start + args.frames)
    print(f"{seq_dir.name}: {len(seq)} frames, running masked DTVO"
          f"{'' if args.no_baseline else ' and plain DTVO'} on {args.start}..{end - 1}")

    out_dir.mkdir(parents=True, exist_ok=True)
    fps_video = float(np.clip(round((end - args.start - 1) / max(
        seq.frames[end - 1][0] - seq.frames[args.start][0], 1e-6)), 1, 60))
    overlay_tmp = out_dir / ".overlay_tmp.mp4"
    overlay_sink = None

    detector.tracker = IoUTracker()
    masker = MovingObjectMask(detector, args.min_prior, cfg.edges)
    vo = DTVO(seq.camera, cfg)
    base = None if args.no_baseline else DTVO(seq.camera, cfg)

    obj_file = open(out_dir / "objects.csv", "w", newline="")
    obj_writer = csv.writer(obj_file)
    obj_writer.writerow(OBJ_COLS + ["masked", "unmasked"])
    rows = []
    acc = {"x_suspect": [], "x_any": [], "x_masked": [], "n_det": [], "n_suspect": [], "n_masked": [],
           "ms": Counter(), "src": Counter(), "cls_masked": Counter(), "cls_all": Counter(),
           "masked_ids": set()}
    t_vo = t_base = t_video = 0.0
    t_start = time.perf_counter()
    t0_seq = seq.frames[args.start][0]

    for i in range(args.start, end):
        raw = seq[i]
        ta = time.perf_counter()
        vo.process(raw, masker)
        tb = time.perf_counter()
        if base is not None:
            base.process(raw)
        tc = time.perf_counter()
        t_vo += tb - ta
        t_base += tc - tb

        dets, masks, split = masker.dets, masker.masks, masker.split
        kept = {masker.layer_det[j] for j in vo.debug.kept_layers}
        mk = [i in kept for i in range(len(dets))]                    # edges actually left out
        given_back = [m and not k for m, k in zip(masker.masked, mk)]  # masked, then given back
        r = vo.records[-1]
        rb = base.records[-1] if base is not None else None
        x_masked = r.n_masked / max(split.n_edges, 1)
        sus = [is_suspect(d.prior) for d in dets]
        src_sus = Counter(m.source for m, s in zip(masks, sus) if s)
        acc["src"].update(Counter(m.source for m, k in zip(masks, mk) if k))
        acc["ms"].update(masker.ms)
        for d, k in zip(dets, mk):
            acc["cls_all"][d.class_name] += 1
            if k:
                acc["cls_masked"][d.class_name] += 1
                acc["masked_ids"].add(d.track_id)
        for key, v in (("x_suspect", split.x_suspect), ("x_any", split.x_any), ("x_masked", x_masked),
                       ("n_det", len(dets)), ("n_suspect", sum(sus)), ("n_masked", sum(mk))):
            acc[key].append(v)

        rows.append([i, f"{raw.timestamp:.6f}", split.n_edges, r.n_edges, r.n_masked, f"{x_masked:.4f}",
                     len(dets), sum(mk), sum(sus), int(split.scene_tag.sum()), split.n_suspect_edges,
                     split.n_object_edges, f"{split.x_suspect:.4f}", f"{split.x_any:.4f}",
                     src_sus.get("depth", 0), sum(sus) - src_sus.get("depth", 0),
                     r.n_ref_points, r.n_unmasked, r.kf_id, int(r.is_keyframe), int(r.lost), int(r.kf_refused),
                     int(r.weak_kf), int(r.ba_rejected), r.hypothesis, f"{r.inlier_ratio:.4f}", "",
                     *(["", "", "", ""] if rb is None else [rb.kf_id, int(rb.lost), int(rb.kf_refused),
                                                            int(rb.ba_rejected)]),
                     "" if rb is None else f"{rb.inlier_ratio:.4f}", "",
                     f"{masker.ms['detect']:.2f}", f"{masker.ms['mask']:.2f}", f"{masker.ms['associate']:.2f}",
                     f"{1000 * (tb - ta):.2f}", f"{1000 * (tc - tb):.2f}" if base is not None else ""])
        for k, (d, m) in enumerate(zip(dets, masks)):
            obj_writer.writerow([i, f"{raw.timestamp:.6f}", d.track_id, d.class_name, f"{d.score:.3f}",
                                 d.prior, int(sus[k]), int(split.scene_tag[k]), *d.box, m.source,
                                 f"{m.z0:.3f}", f"{m.depth_coverage:.3f}", int(m.mask.sum()),
                                 int(split.edges_per_object[k]), int(mk[k]), int(given_back[k])])

        if not args.no_video:
            tv = time.perf_counter()
            objects = [(m.mask, f"#{d.track_id} {d.class_name} P{d.prior:.1f}"
                        + ("" if m.source == "depth" else " fb"))
                       for d, m, k in zip(dets, masks, mk) if k]
            img = tracking_overlay(vo.debug, i, len(vo.keyframes), objects=objects)
            if overlay_sink is None:   # near-lossless: it is re-encoded once more when composed
                overlay_sink = VideoSink(overlay_tmp, img.shape[1], img.shape[0], fps_video, crf=12)
            overlay_sink.write(img)
            t_video += time.perf_counter() - tv
        n = i - args.start + 1
        if n % 50 == 0 or i == end - 1:
            fps = n / (time.perf_counter() - t_start)
            line = (f"  {n}/{end - args.start}  keyframes {len(vo.keyframes)}  matched edges "
                    f"{100 * r.inlier_ratio:.0f}%  masked {100 * x_masked:.0f}% of edges")
            if rb is not None:
                line += f"  | plain DTVO matched {100 * rb.inlier_ratio:.0f}%"
            print(f"{line}  {fps:.1f} fps", flush=True)
    obj_file.close()
    if overlay_sink is not None:
        overlay_sink.close()

    n = end - args.start
    stamps, poses = vo.trajectory()
    write_tum(out_dir / "masked_dtvo_trajectory.txt", stamps, poses)
    if base is not None:
        b_stamps, b_poses = base.trajectory()
        write_tum(out_dir / "baseline_trajectory.txt", b_stamps, b_poses)

    src_total = sum(acc["src"].values())
    metrics = {
        "sequence": seq_dir.name, "frames": n, "detector": args.mode, "min_prior": args.min_prior,
        "masked_dtvo": vo_metrics(vo, t_vo, cfg),
        "baseline_dtvo": vo_metrics(base, t_base, cfg) if base is not None else None,
        "detection": {
            "x_masked": stats(acc["x_masked"]), "x_suspect": stats(acc["x_suspect"]), "x_any": stats(acc["x_any"]),
            "frames_with_masked_pct": 100 * float(np.mean(np.array(acc["n_masked"]) > 0)),
            "detections_per_frame": float(np.mean(acc["n_det"])),
            "suspects_per_frame": float(np.mean(acc["n_suspect"])),
            "masked_per_frame": float(np.mean(acc["n_masked"])),
            "masked_tracks": len(acc["masked_ids"]),
            "tracks_started_all": detector.tracker.created,
            "ms_per_frame": {k: v / n for k, v in acc["ms"].items()},
            "masked_mask_sources": dict(acc["src"]),
            "masked_mask_sources_pct": {k: 100 * v / max(src_total, 1) for k, v in acc["src"].items()},
            "masked_class_counts": dict(acc["cls_masked"].most_common(30)),
            "all_class_counts_top30": dict(acc["cls_all"].most_common(30)),
            "settings": {"suspect_threshold": SUSPECT_THRESHOLD, "default_prior": DEFAULT_PRIOR,
                         "detector": vars(detector.cfg), "mask": vars(masker.mcfg),
                         "canny_min_high": cfg.edges.min_high},
        },
    }
    m = metrics["masked_dtvo"]
    print(f"\nmasked DTVO: {n} frames in {t_vo:.1f} s ({n / t_vo:.1f} fps) | keyframes {m['keyframes']} | "
          f"frames flagged lost {m['frames_lost']}")
    print("  time per frame: " + ", ".join(f"{k} {v:.1f} ms" for k, v in m["ms_per_frame"].items()))
    print(f"  guards: keyframes refused {m['keyframes_refused']}, weak keyframes {m['weak_keyframes']}, "
          f"window optimisations undone {m['window_optimisations_rejected']} of {m['window_optimisations']}, "
          f"objects given back {m['objects_unmasked']} (in {m['frames_with_unmasking']} frames)")
    d = metrics["detection"]
    print(f"  masked {100 * d['x_masked']['mean']:.1f}% of edges on average (max {100 * d['x_masked']['max']:.0f}%), "
          f"{d['masked_per_frame']:.2f} objects / frame, in {d['frames_with_masked_pct']:.0f}% of frames; "
          f"top classes: " + ", ".join(f"{k} {v}" for k, v in list(d["masked_class_counts"].items())[:5]))
    if base is not None:
        mb = metrics["baseline_dtvo"]
        print(f"plain DTVO: {n / t_base:.1f} fps | keyframes {mb['keyframes']} | frames flagged lost {mb['frames_lost']}")
        print(f"  guards: keyframes refused {mb['keyframes_refused']}, weak keyframes {mb['weak_keyframes']}, "
              f"window optimisations undone {mb['window_optimisations_rejected']} of {mb['window_optimisations']}")

    est_xyz, gt_xyz, base_xyz = np.array([T[:3, 3] for T in poses]), None, None
    err = base_err = None
    ev = ev_b = None
    gt_file = seq_dir / "groundtruth.txt"
    if not gt_file.exists():
        print("no groundtruth.txt: skipping evaluation")
    else:
        gt_t, gt_T = load_groundtruth(gt_file)
        bonn = "bonn" in seq_dir.name.lower()
        ev = evaluate(stamps, poses, gt_t, gt_T, fit_body=bonn)
        metrics["masked_dtvo"].update(ev_metrics(ev))
        print_ev("masked DTVO", ev)
        if ev.body_rotation_deg > 0:
            print(f"(Bonn: compensated constant camera-to-marker rotation of {ev.body_rotation_deg:.1f} deg)")
        est_xyz, err = per_frame(ev, poses, n)
        gt_xyz = np.full_like(est_xyz, np.nan)
        gt_xyz[ev.est_index] = ev.gt_xyz
        evs = {MASKED: ev}
        if base is not None:
            ev_b = evaluate(b_stamps, b_poses, gt_t, gt_T, fit_body=bonn)
            metrics["baseline_dtvo"].update(ev_metrics(ev_b))
            print_ev("plain DTVO (all edges)", ev_b)
            base_xyz, base_err = per_frame(ev_b, b_poses, n)
            evs[BASELINE] = ev_b
            metrics["change_pct"] = {
                k: 100 * (metrics["masked_dtvo"][k] / metrics["baseline_dtvo"][k] - 1)
                for k in ("ate_rmse_m", "rot_rmse_deg", "start_aligned_rmse_m", "start_aligned_rot_rmse_deg",
                          "rpe_trans_m_per_s", "rpe_rot_deg_per_s") if metrics["baseline_dtvo"][k] > 0}
            c = metrics["change_pct"]
            print(f"masked vs plain: ATE {c['ate_rmse_m']:+.0f}%, start-aligned position "
                  f"{c['start_aligned_rmse_m']:+.0f}%, RPE {c['rpe_trans_m_per_s']:+.0f}% (negative = better)")
        variant = [v for v, on in (("no weighting", args.no_weights), ("no window optimisation", args.no_mapping),
                                   ("keyframe guards off", args.no_guard)) if on]
        title = (f"Masked DTVO ({MODE_LABELS[args.mode]}, prior >= {args.min_prior:g}) on {seq_dir.name}"
                 + (f" ({', '.join(variant)})" if variant else "")
                 + f"  -  ATE {ev.ate_rmse * 100:.1f} cm"
                 + (f" vs {ev_b.ate_rmse * 100:.1f} cm plain DTVO" if ev_b is not None else ""))
        plot_evaluation(evs, title, out_dir / "masked_dtvo_plot.png", show=show)

    for row, k in zip(rows, range(n)):
        if err is not None and np.isfinite(err[k]):
            row[FRAME_COLS.index("pos_err_m")] = f"{err[k]:.4f}"
        if base_err is not None and np.isfinite(base_err[k]):
            row[FRAME_COLS.index("baseline_pos_err_m")] = f"{base_err[k]:.4f}"
    with open(out_dir / "frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FRAME_COLS)
        w.writerows(rows)

    t_rel = stamps - stamps[0]
    plot_masking(t_rel, acc, vo, base, err, base_err, metrics,
                 f"Masking on {seq_dir.name} ({n} frames, {MODE_LABELS[args.mode]})", out_dir / "masking_plot.png")

    if overlay_sink is not None:
        print("composing video ...", flush=True)
        panel = TrajectoryPanel(est_xyz[:, :2], None if gt_xyz is None else gt_xyz[:, :2], t_rel,
                                "Camera path (x-y), live", est_label=MASKED,
                                other_xy=None if base_xyz is None else base_xyz[:, :2], other_label=BASELINE)
        n_written = compose_video(overlay_tmp, out_dir / "masked_dtvo_video.mp4", panel, fps_video)
        overlay_tmp.unlink(missing_ok=True)
        print(f"wrote {out_dir / 'masked_dtvo_video.mp4'}  ({n_written} frames, {fps_video:.0f} fps)")

    (out_dir / "masked_dtvo_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"wrote {out_dir}: trajectories, frames.csv, objects.csv, masked_dtvo_metrics.json")


def _plt():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_masking(t, acc, vo, base, err, base_err, metrics, title, out):
    """Position error, masked edge share and matched edges over time + a summary table."""
    plt = _plt()
    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.8, 1])
    panels = [("Position error vs ground truth, start-aligned", "cm",
               [(MASKED, err), (BASELINE, base_err)], 100),
              ("Share of edges masked (left out of tracking)", "% of edges",
               [(MASKED, np.array(acc["x_masked"]))], 100),
              ("Matched keyframe edges (tracking inlier ratio)", "%",
               [(MASKED, np.array([r.inlier_ratio for r in vo.records])),
                (BASELINE, None if base is None else np.array([r.inlier_ratio for r in base.records]))], 100)]
    for row, (head, ylabel, series, scale) in enumerate(panels):
        ax = fig.add_subplot(gs[row, 0], sharex=None if row == 0 else fig.axes[0])
        for k, (name, y) in list(enumerate(series))[::-1]:   # masked run on top
            if y is not None:
                ax.plot(t, scale * y, color=RUN_COLOURS[k], lw=1.8 if k == 0 else 1.4, label=name)
        ax.set_ylabel(ylabel)
        ax.set_title(head + ("" if err is not None or row else "  (no ground truth)"), loc="left",
                     fontweight="bold")
        if row != 1 and sum(y is not None for _, y in series) > 1:
            h, lab = ax.get_legend_handles_labels()
            ax.legend(h[::-1], lab[::-1], frameon=False, loc="upper left")
        if row == 2:
            ax.set_xlabel("time (s)")
        _style(ax)

    ax = fig.add_subplot(gs[:, 1])
    ax.axis("off")
    m, b, d = metrics["masked_dtvo"], metrics["baseline_dtvo"], metrics["detection"]
    cm = lambda x, k: f"{100 * x[k]:.1f} cm" if x and k in x else "-"
    rows = [("ATE RMSE (best fit)", cm(m, "ate_rmse_m"), cm(b, "ate_rmse_m")),
            ("Position RMSE, start-aligned", cm(m, "start_aligned_rmse_m"), cm(b, "start_aligned_rmse_m")),
            ("RPE per second", cm(m, "rpe_trans_m_per_s"), cm(b, "rpe_trans_m_per_s")),
            ("Rotation RMSE (best fit)", f"{m['rot_rmse_deg']:.2f}°" if "rot_rmse_deg" in m else "-",
             f"{b['rot_rmse_deg']:.2f}°" if b and "rot_rmse_deg" in b else "-"),
            ("Mean matched edges", f"{100 * m['mean_matched_edges']:.0f} %",
             f"{100 * b['mean_matched_edges']:.0f} %" if b else "-"),
            ("Mean edges used / frame", f"{m['mean_edges_used']:.0f}", f"{b['mean_edges_used']:.0f}" if b else "-"),
            ("Keyframes / frames lost", f"{m['keyframes']} / {m['frames_lost']}",
             f"{b['keyframes']} / {b['frames_lost']}" if b else "-"),
            ("Keyframes refused / weak", f"{m['keyframes_refused']} / {m['weak_keyframes']}",
             f"{b['keyframes_refused']} / {b['weak_keyframes']}" if b else "-"),
            ("Window optimisations undone", f"{m['window_optimisations_rejected']} / {m['window_optimisations']}",
             f"{b['window_optimisations_rejected']} / {b['window_optimisations']}" if b else "-"),
            ("VO speed", f"{m['fps']:.1f} fps", f"{b['fps']:.1f} fps" if b else "-")]
    ax.text(0.70, 0.99, MASKED, color=RUN_COLOURS[0], transform=ax.transAxes, ha="right", va="top",
            fontweight="bold")
    ax.text(1.0, 0.99, BASELINE, color=RUN_COLOURS[1], transform=ax.transAxes, ha="right", va="top",
            fontweight="bold")
    y = 0.94
    for label, v1, v2 in rows:
        ax.text(0.0, y, label, color=MUTED, transform=ax.transAxes, va="top")
        ax.text(0.70, y, v1, color=INK, transform=ax.transAxes, va="top", ha="right", fontweight="bold")
        ax.text(1.0, y, v2, color=INK, transform=ax.transAxes, va="top", ha="right", fontweight="bold")
        y -= 0.045
    y -= 0.025
    ax.text(0.0, y, "Masking", color=INK, transform=ax.transAxes, va="top", fontweight="bold")
    y -= 0.05
    srcs = d["masked_mask_sources_pct"]
    for label, v in (("mean / max share of edges masked",
                      f"{100 * d['x_masked'].get('mean', 0):.1f} / {100 * d['x_masked'].get('max', 0):.0f} %"),
                     ("frames with a masked object", f"{d['frames_with_masked_pct']:.0f} %"),
                     ("detections / masked objects per frame",
                      f"{d['detections_per_frame']:.1f} / {d['masked_per_frame']:.2f}"),
                     ("masked objects tracked", f"{d['masked_tracks']}"),
                     ("masks cut out with depth", f"{srcs.get('depth', 0):.0f} %"),
                     ("objects given back (edge budget)", f"{m['objects_unmasked']} in {m['frames_with_unmasking']} frames"),
                     ("detect + mask ms / frame", f"{sum(d['ms_per_frame'].values()):.1f}")):
        ax.text(0.0, y, label, color=MUTED, transform=ax.transAxes, va="top")
        ax.text(1.0, y, v, color=INK, transform=ax.transAxes, va="top", ha="right", fontweight="bold")
        y -= 0.048
    top = list(d["masked_class_counts"].items())[:8]
    ax.text(0.0, y - 0.01, "masked classes (detections):\n" + ("\n".join(f"  {k} {v}" for k, v in top) or "  -"),
            color=INK, transform=ax.transAxes, va="top", fontsize=8)

    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


SUMMARY_COLS = ["sequence", "frames", "ate_masked_m", "ate_plain_m", "ate_change_pct",
                "start_rmse_masked_m", "start_rmse_plain_m", "rpe_masked_m_per_s", "rpe_plain_m_per_s",
                "rot_rmse_masked_deg", "rot_rmse_plain_deg", "x_masked_mean", "x_masked_max",
                "frames_with_masked_pct", "masked_per_frame", "fps_masked", "fps_plain",
                "kf_refused_masked", "kf_refused_plain", "ba_undone_masked", "ba_undone_plain",
                "objects_unmasked"]


def summary_row(m: dict) -> dict:
    a, b, d = m["masked_dtvo"], m["baseline_dtvo"] or {}, m["detection"]
    return {"sequence": m["sequence"], "frames": m["frames"],
            "ate_masked_m": a.get("ate_rmse_m"), "ate_plain_m": b.get("ate_rmse_m"),
            "ate_change_pct": m.get("change_pct", {}).get("ate_rmse_m"),
            "start_rmse_masked_m": a.get("start_aligned_rmse_m"), "start_rmse_plain_m": b.get("start_aligned_rmse_m"),
            "rpe_masked_m_per_s": a.get("rpe_trans_m_per_s"), "rpe_plain_m_per_s": b.get("rpe_trans_m_per_s"),
            "rot_rmse_masked_deg": a.get("rot_rmse_deg"), "rot_rmse_plain_deg": b.get("rot_rmse_deg"),
            "x_masked_mean": d["x_masked"].get("mean"), "x_masked_max": d["x_masked"].get("max"),
            "frames_with_masked_pct": d["frames_with_masked_pct"], "masked_per_frame": d["masked_per_frame"],
            "fps_masked": a["fps"], "fps_plain": b.get("fps"),
            "kf_refused_masked": a.get("keyframes_refused"), "kf_refused_plain": b.get("keyframes_refused"),
            "ba_undone_masked": a.get("window_optimisations_rejected"),
            "ba_undone_plain": b.get("window_optimisations_rejected"), "objects_unmasked": a.get("objects_unmasked")}


def write_summary(run_dir: Path):
    rows = [summary_row(json.loads(p.read_text())) for p in sorted(run_dir.glob("*/masked_dtvo_metrics.json"))]
    if not rows:
        return
    with open(run_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        w.writerows(rows)
    rows = [r for r in rows if r["ate_masked_m"] is not None]
    if rows:
        plt = _plt()
        names = [r["sequence"].replace("rgbd_dataset_", "").replace("rgbd_", "") for r in rows]
        yy = np.arange(len(rows))
        fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(rows) + 2.2), sharey=True,
                                 constrained_layout=True)
        ax = axes[0]
        for k, (key, name) in enumerate((("ate_masked_m", MASKED), ("ate_plain_m", BASELINE))):
            vals = [100 * r[key] if r[key] is not None else np.nan for r in rows]
            ax.barh(yy + (k - 0.5) * 0.4, vals, height=0.36, color=RUN_COLOURS[k], label=name,
                    edgecolor="white", linewidth=1)
        ax.set_title("ATE RMSE, best fit (cm)", loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="lower right")
        ax = axes[1]
        ax.barh(yy, [100 * r["x_masked_mean"] for r in rows], height=0.5, color=RUN_COLOURS[0],
                edgecolor="white", linewidth=1)
        ax.set_title("Mean share of edges masked (%)", loc="left", fontweight="bold")
        for ax in axes:
            ax.grid(True, axis="x", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
        axes[0].set_yticks(yy, names)
        axes[0].invert_yaxis()
        fig.suptitle("Masked DTVO vs plain DTVO, all sequences of this run", x=0.01, ha="left",
                     fontsize=12, fontweight="bold", color=INK)
        fig.savefig(run_dir / "summary_plot.png", dpi=120)
        plt.close(fig)
    print(f"wrote {run_dir / 'summary.csv'} and summary_plot.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", type=Path, nargs="*", default=[],
                    help="sequence folder(s) (omit to choose in the selector window)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0, help="0 = whole sequence")
    ap.add_argument("--mode", default="coco", choices=MODES, help="detector")
    ap.add_argument("--min-prior", type=float, default=SUSPECT_THRESHOLD,
                    help="mask objects whose class prior is at least this")
    ap.add_argument("--no-baseline", action="store_true", help="don't run plain DTVO alongside")
    ap.add_argument("--no-guard", action="store_true", help="keyframe guards off (ablation)")
    ap.add_argument("--no-mapping", action="store_true", help="disable sliding-window optimisation")
    ap.add_argument("--no-weights", action="store_true", help="disable weighted edge alignment")
    ap.add_argument("--no-show", action="store_true", help="don't open the plot window")
    ap.add_argument("--no-video", action="store_true", help="don't write masked_dtvo_video.mp4")
    ap.add_argument("--out-dir", type=Path, default=None, help="parent of the run folder (default results/)")
    ap.add_argument("--summary-only", type=Path, default=None, metavar="RUN_DIR")
    args = ap.parse_args()

    if args.summary_only:
        write_summary(args.summary_only)
        return
    if args.all:
        # an archive next to a folder means the download script is still extracting it
        seqs = [s.path for s in find_sequences()
                if not any(s.path.with_suffix(ext).exists() for ext in (".zip", ".tgz"))]
    else:
        seqs = sequences_from_args_or_gui(args.sequence, "Masked DTVO - select sequences")
    show = not args.no_show and len(seqs) == 1   # one plot window per sequence would pile up
    if not show:
        import matplotlib
        matplotlib.use("Agg")

    with RunDir("run_masked_dtvo", args, seqs, args.out_dir) as run:
        print(f"loading detector: {MODE_LABELS[args.mode]}", flush=True)
        detector = Detector(args.mode)
        write_prior_table({args.mode: detector}, run.dir)
        for k, seq_dir in enumerate(seqs, 1):
            print(f"\n=== [{k}/{len(seqs)}] {seq_dir.name}", flush=True)
            with run.sequence(seq_dir) as out_dir:
                run_sequence(seq_dir, out_dir, detector, args, show)
            write_summary(run.dir)
    if show and not run.failed:
        import matplotlib.pyplot as plt
        if plt.get_fignums():   # no plot without ground truth
            plt.show()
    if run.failed:
        sys.exit(f"{len(run.failed)} of {len(seqs)} sequences failed: {', '.join(run.failed)}")


if __name__ == "__main__":
    main()
