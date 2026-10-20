# Better Edge VO

RGB-D edge-based visual odometry that stays accurate when moving objects are in view.
Built on a reimplementation of DTVO (Yan et al. 2022, *Robust and efficient edge-based
visual odometry*), with YOLO-seg moving-object detection whose edges are masked out
before tracking. Course project for AM5011 Virtual Reality Engineering.

## Layout

```
src/
  common/        dataset loading, Tk dataset picker, preprocessing, edge extraction, run folders
  dtvo/          DTVO: distance-transform tracking, local mapping (window BA), evaluation
  detection/     YOLO-seg detector (coco / yoloe_text / yoloe_pf), priors, depth cut-out, edge association
  tests/         pytest suite
  demo_edges.py        edge-extraction demo video
  run_dtvo.py          plain DTVO on one or more sequences
  run_detection.py     moving-object detection on one or more sequences
  run_masked_dtvo.py   DTVO with moving-object edges masked, vs plain DTVO
scripts/
  download_datasets.sh TUM RGB-D + Bonn RGB-D Dynamic sequences -> datasets/
  process_video.py     early RGB / depth edge video experiment
```

Not tracked (see `.gitignore`): `datasets/`, `models/`, `results/`, `output_videos/`, `papers/`.

## Setup

```bash
pip install -r requirements.txt
bash scripts/download_datasets.sh          # ~27 GB into datasets/tum and datasets/bonn
```

Put the YOLO weights in `models/` (`yolo11s-seg.pt`, `yoloe-11s-seg.pt`, `yoloe-11s-seg-pf.pt`,
plus `mobileclip_blt.ts` for YOLOE text prompts); Ultralytics can download them.

## Usage (from the project root)

```bash
python -m src.run_dtvo                         # dataset selector window
python -m src.run_masked_dtvo datasets/bonn/rgbd_bonn_balloon
python -m src.run_detection --all
python -m pytest src/tests -p no:cacheprovider
```

Each run writes to `results/<date>_<time>_<script>/` (run_info.json, log.txt, per-sequence
trajectories, plots and videos). Run any script with `--help` for its options.
