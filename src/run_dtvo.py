"""Run DTVO (Yan, Li, Zhou 2022) on TUM / Bonn sequences and compare with ground truth.

Usage (from the project root):
    conda run -n 312ml python -m src.run_dtvo                 -> dataset selector window (pick one or more)
    conda run -n 312ml python -m src.run_dtvo datasets/tum/rgbd_dataset_freiburg1_desk [more sequences ...]
    options: --start N --frames N    part of each sequence
             --no-mapping            tracking only (no sliding-window optimisation)
             --no-weights            plain alignment (ablation of Sec. 4.3 weighting)
             --no-show               only save the figure, don't open a window
                                     (always the case when more than one sequence is run)
             --no-video              skip the result video (saves ~25 ms per frame)
             --out-dir DIR           where the run folder is made (default results/)

Can also be run directly as a file (IDE Run button with the 312ml interpreter).

Every run gets its own folder results/<YYYY-MM-DD_HH-MM-SS>_run_dtvo/ holding
    run_info.json         command, options, sequences, start / end time, per-sequence status
    log.txt               the console output
and per sequence, in <run folder>/<sequence>/:
    dtvo_trajectory.txt   estimated poses, TUM format (usable with TUM's evaluate_ate.py)
    dtvo_plot.png         x / y / z and roll / pitch / yaw vs. ground truth, with RMSE
    dtvo_metrics.json     ATE / RPE / per-axis RMSE, timings, keyframe counts
    dtvo_video.mp4        edges + tracked keyframe edges (left) next to the x-y path drawn
                          live against ground truth (right)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # run as a plain file, not via `python -m src.run_dtvo`
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.dataset import RGBDSequence  # noqa: E402
from src.common.dataset_picker import sequences_from_args_or_gui  # noqa: E402
from src.common.run_dir import RunDir  # noqa: E402
from src.dtvo.system import DTVO, DTVOConfig  # noqa: E402
from src.common.video import VideoSink  # noqa: E402
from src.dtvo.evaluation import Evaluation, evaluate, load_groundtruth, write_tum  # noqa: E402
from src.dtvo.visualise import TrajectoryPanel, compose_video, plot_evaluation, tracking_overlay  # noqa: E402

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", type=Path, nargs="*", default=[],
                    help="sequence folder(s) (omit to choose in the selector window)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0, help="0 = whole sequence")
    ap.add_argument("--no-mapping", action="store_true", help="disable sliding-window optimisation")
    ap.add_argument("--no-weights", action="store_true", help="disable weighted edge alignment")
    ap.add_argument("--no-show", action="store_true", help="don't open the plot window")
    ap.add_argument("--no-video", action="store_true", help="don't write dtvo_video.mp4")
    ap.add_argument("--out-dir", type=Path, default=None, help="parent of the run folder (default results/)")
    args = ap.parse_args()

    seqs = sequences_from_args_or_gui(args.sequence, "DTVO - select sequences")
    show = not args.no_show and len(seqs) == 1   # one plot window per sequence would pile up
    with RunDir("run_dtvo", args, seqs, args.out_dir) as run:
        for k, seq_dir in enumerate(seqs, 1):
            print(f"\n=== [{k}/{len(seqs)}] {seq_dir.name}", flush=True)
            with run.sequence(seq_dir) as out_dir:
                run_sequence(seq_dir, out_dir, args, show)
    if show and not run.failed:
        import matplotlib.pyplot as plt
        if plt.get_fignums():   # no plot without ground truth
            plt.show()
    if run.failed:
        sys.exit(f"{len(run.failed)} of {len(seqs)} sequences failed: {', '.join(run.failed)}")


def run_sequence(seq_dir: Path, out_dir: Path, args, show: bool):
    seq = RGBDSequence(seq_dir)
    cfg = DTVOConfig()
    cfg.mapping.enabled = not args.no_mapping
    cfg.track.weighted = not args.no_weights
    end = len(seq) if args.frames <= 0 else min(len(seq), args.start + args.frames)
    print(f"{seq_dir.name}: {len(seq)} frames, running DTVO on {args.start}..{end - 1}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # Play the video at the sequence's real frame rate (TUM / Bonn record at ~30 Hz)
    fps_video = float(np.clip(round((end - args.start - 1) / max(
        seq.frames[end - 1][0] - seq.frames[args.start][0], 1e-6)), 1, 60))
    overlay_tmp = out_dir / ".overlay_tmp.mp4"
    overlay_sink = None

    vo = DTVO(seq.camera, cfg)
    t0 = time.perf_counter()
    t_video = 0.0
    for i in range(args.start, end):
        vo.process(seq[i])
        if not args.no_video:
            tv = time.perf_counter()
            img = tracking_overlay(vo.debug, i, len(vo.keyframes))
            if overlay_sink is None:   # near-lossless: it is re-encoded once more when composed
                overlay_sink = VideoSink(overlay_tmp, img.shape[1], img.shape[0], fps_video, crf=12)
            overlay_sink.write(img)
            t_video += time.perf_counter() - tv
        n = i - args.start + 1
        if n % 50 == 0 or i == end - 1:
            r = vo.records[-1]
            fps = n / (time.perf_counter() - t0)
            print(f"  {n}/{end - args.start}  keyframes {len(vo.keyframes)}  "
                  f"matched edges {100 * r.inlier_ratio:.0f}%  {fps:.1f} fps", flush=True)
    wall = time.perf_counter() - t0 - t_video   # VO time only
    if overlay_sink is not None:
        overlay_sink.close()

    stamps, poses = vo.trajectory()
    write_tum(out_dir / "dtvo_trajectory.txt", stamps, poses)

    n = len(vo.records)
    lost = int(sum(bool(r.lost) for r in vo.records))
    hyp = {}
    for r in vo.records[1:]:
        hyp[r.hypothesis] = hyp.get(r.hypothesis, 0) + 1
    metrics = {"sequence": seq_dir.name, "frames": n, "keyframes": len(vo.keyframes),
               "frames_lost": lost, "fps": n / wall,
               "ms_per_frame": {k: 1000 * v / n for k, v in vo.times.items()},
               "initial_hypothesis_chosen": hyp,
               "mean_matched_edges": float(np.mean([r.inlier_ratio for r in vo.records])),
               "weighted_alignment": cfg.track.weighted, "window_optimisation": cfg.mapping.enabled}
    print(f"\n{n} frames in {wall:.1f} s ({n / wall:.1f} fps) | keyframes {len(vo.keyframes)} | "
          f"frames flagged lost {lost}")
    print("time per frame: " + ", ".join(f"{k} {v:.1f} ms" for k, v in metrics["ms_per_frame"].items()))

    est_xyz = np.array([T[:3, 3] for T in poses])
    gt_xyz = None   # per frame, NaN where no ground-truth pose is close enough in time
    gt_file = seq_dir / "groundtruth.txt"
    if not gt_file.exists():
        print("no groundtruth.txt: skipping evaluation")
    else:
        gt_t, gt_T = load_groundtruth(gt_file)
        ev = evaluate(stamps, poses, gt_t, gt_T, fit_body="bonn" in seq_dir.name.lower())
        metrics.update({"ate_rmse_m": ev.ate_rmse, "ate_mean_m": ev.ate_mean, "ate_max_m": ev.ate_max,
                        "rot_rmse_deg": ev.rot_rmse_deg,
                        "start_aligned_rmse_m": ev.start_rmse, "start_aligned_rot_rmse_deg": ev.start_rot_rmse_deg,
                        "start_aligned_rmse_xyz_m": ev.axis_rmse.tolist(),
                        "start_aligned_rmse_roll_pitch_yaw_deg": ev.rpy_rmse.tolist(),
                        "rpe_trans_m_per_s": ev.rpe_trans, "rpe_rot_deg_per_s": ev.rpe_rot,
                        "gt_path_length_m": ev.gt_length, "matched_frames": ev.matched,
                        "gt_body_rotation_compensated_deg": ev.body_rotation_deg})
        print(f"best fit (TUM):  ATE RMSE {ev.ate_rmse:.4f} m, rotation RMSE {ev.rot_rmse_deg:.2f} deg")
        print(f"start-aligned:   position RMSE {ev.start_rmse:.4f} m (x {ev.axis_rmse[0]:.4f}, "
              f"y {ev.axis_rmse[1]:.4f}, z {ev.axis_rmse[2]:.4f}), rotation RMSE {ev.start_rot_rmse_deg:.2f} deg "
              f"(roll {ev.rpy_rmse[0]:.2f}, pitch {ev.rpy_rmse[1]:.2f}, yaw {ev.rpy_rmse[2]:.2f})")
        print(f"RPE per {ev.rpe_delta:g} s: {ev.rpe_trans:.4f} m, {ev.rpe_rot:.2f} deg")
        if ev.body_rotation_deg > 0:
            print(f"(Bonn: compensated constant camera-to-marker rotation of {ev.body_rotation_deg:.1f} deg)")
        variant = []
        if args.no_weights:
            variant.append("no weighting")
        if args.no_mapping:
            variant.append("no window optimisation")
        title = (f"DTVO on {seq_dir.name}" + (f" ({', '.join(variant)})" if variant else "")
                 + f"  -  ATE RMSE {ev.ate_rmse * 100:.1f} cm (best fit), "
                 f"{ev.start_rmse * 100:.1f} cm start-aligned")
        plot_evaluation({"DTVO estimate": ev}, title, out_dir / "dtvo_plot.png", show=show)
        est_xyz = np.array([(ev.anchor @ T)[:3, 3] for T in poses])   # start-aligned, as in dtvo_plot.png
        gt_xyz = np.full_like(est_xyz, np.nan)
        gt_xyz[ev.est_index] = ev.gt_xyz

    if overlay_sink is not None:
        print("composing video ...", flush=True)
        panel = TrajectoryPanel(est_xyz[:, :2], None if gt_xyz is None else gt_xyz[:, :2],
                                stamps - stamps[0], "Camera path (x-y), live")
        n_written = compose_video(overlay_tmp, out_dir / "dtvo_video.mp4", panel, fps_video)
        overlay_tmp.unlink(missing_ok=True)
        print(f"wrote {out_dir / 'dtvo_video.mp4'}  ({n_written} frames, {fps_video:.0f} fps)")

    (out_dir / "dtvo_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"wrote {out_dir / 'dtvo_trajectory.txt'} and dtvo_metrics.json")


if __name__ == "__main__":
    main()
