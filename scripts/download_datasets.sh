#!/usr/bin/env bash
# Downloads every dataset sequence used by the project into datasets/{tum,bonn}.
# Resumable: re-running skips sequences that are already extracted and
# continues partial downloads (wget -c). Archives are deleted after a
# successful extraction to save disk space.
set -u
mkdir -p "$(dirname "$0")/../datasets"
cd "$(dirname "$0")/../datasets"

TUM=https://cvg.cit.tum.de/rgbd/dataset
BONN=https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019

TUM_SEQS=(
  # dynamic (main real-world evaluation)
  freiburg3/rgbd_dataset_freiburg3_walking_xyz
  freiburg3/rgbd_dataset_freiburg3_walking_rpy
  freiburg3/rgbd_dataset_freiburg3_walking_halfsphere
  freiburg3/rgbd_dataset_freiburg3_walking_static
  freiburg3/rgbd_dataset_freiburg3_sitting_xyz
  freiburg3/rgbd_dataset_freiburg3_sitting_static
  # static sanity / base sequences for the semi-synthetic sweep
  freiburg1/rgbd_dataset_freiburg1_desk
  freiburg2/rgbd_dataset_freiburg2_desk
  freiburg3/rgbd_dataset_freiburg3_long_office_household
  # low-texture (secondary objective)
  freiburg3/rgbd_dataset_freiburg3_structure_notexture_near
  freiburg3/rgbd_dataset_freiburg3_structure_notexture_far
  # extra dynamic sequences for the moving-object detector evaluation
  freiburg3/rgbd_dataset_freiburg3_sitting_rpy
  freiburg3/rgbd_dataset_freiburg3_sitting_halfsphere
  freiburg2/rgbd_dataset_freiburg2_desk_with_person
)

BONN_SEQS=(
  rgbd_bonn_crowd
  rgbd_bonn_crowd2
  rgbd_bonn_crowd3
  rgbd_bonn_person_tracking
  rgbd_bonn_person_tracking2
  rgbd_bonn_moving_obstructing_box
  rgbd_bonn_moving_obstructing_box2
  rgbd_bonn_moving_nonobstructing_box
  rgbd_bonn_synchronous
  rgbd_bonn_synchronous2
  rgbd_bonn_static
  # extra sequences for the moving-object detector evaluation: balloons and boxes
  # (objects a COCO detector has no class for), plus people placing/removing things
  rgbd_bonn_balloon
  rgbd_bonn_balloon2
  rgbd_bonn_balloon_tracking
  rgbd_bonn_balloon_tracking2
  rgbd_bonn_kidnapping_box
  rgbd_bonn_kidnapping_box2
  rgbd_bonn_moving_nonobstructing_box2
  rgbd_bonn_placing_nonobstructing_box
  rgbd_bonn_placing_nonobstructing_box2
  rgbd_bonn_placing_nonobstructing_box3
  rgbd_bonn_placing_obstructing_box
  rgbd_bonn_removing_nonobstructing_box
  rgbd_bonn_removing_nonobstructing_box2
  rgbd_bonn_removing_obstructing_box
)

fetch() {  # fetch <url> <dest_dir> <seq_name> <archive_ext>
  local url=$1 dest=$2 name=$3 ext=$4
  mkdir -p "$dest"
  if [ -f "$dest/$name/groundtruth.txt" ]; then
    echo "[skip] $name"; return 0
  fi
  echo "[get ] $name"
  wget -q -c -O "$dest/$name.$ext" "$url" || { echo "[FAIL] download $name"; return 1; }
  if [ "$ext" = tgz ]; then
    tar -xzf "$dest/$name.$ext" -C "$dest"
  else
    unzip -q -o "$dest/$name.$ext" -d "$dest"
  fi
  if [ -f "$dest/$name/groundtruth.txt" ]; then
    rm -f "$dest/$name.$ext"; echo "[ ok ] $name"
  else
    echo "[FAIL] extract $name (archive kept)"; return 1
  fi
}

for s in "${TUM_SEQS[@]}"; do
  fetch "$TUM/$s.tgz" tum "$(basename "$s")" tgz
done
for s in "${BONN_SEQS[@]}"; do
  fetch "$BONN/$s.zip" bonn "$s" zip
done
echo "ALL DONE"
