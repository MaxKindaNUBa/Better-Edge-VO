"""Loading TUM RGB-D / Bonn RGB-D Dynamic sequences.

Both datasets share the TUM layout:
    <seq>/rgb.txt, depth.txt, groundtruth.txt   ("timestamp ..." lines, '#' comments)
    <seq>/rgb/*.png      8-bit colour
    <seq>/depth/*.png    16-bit depth, 5000 units = 1 m, 0 = no reading

Colour and depth are captured at different instants, so frames are paired by
timestamp (as TUM's associate.py does), never by index.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Camera:
    """Pinhole intrinsics + OpenCV distortion (k1, k2, p1, p2, k3)."""
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    width: int = 640
    height: int = 480
    depth_scale: float = 5000.0  # raw depth units per metre

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    @property
    def has_distortion(self) -> bool:
        return any(abs(d) > 0 for d in self.dist)

    def undistorted(self) -> "Camera":
        return replace(self, dist=(0.0, 0.0, 0.0, 0.0, 0.0))

    def downscaled(self) -> "Camera":
        """Intrinsics after 2x2 block averaging (pixel centres at 2i + 0.5)."""
        return replace(self,
                       fx=self.fx / 2, fy=self.fy / 2,
                       cx=(self.cx + 0.5) / 2 - 0.5, cy=(self.cy + 0.5) / 2 - 0.5,
                       width=self.width // 2, height=self.height // 2)


# Official calibrations:
#   TUM:  https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
#   Bonn: https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/
CAMERAS = {
    "tum_fr1": Camera(517.3, 516.5, 318.6, 255.3, (0.2624, -0.9531, -0.0054, 0.0026, 1.1633)),
    "tum_fr2": Camera(520.9, 521.0, 325.1, 249.7, (0.2312, -0.7849, -0.0033, -0.0001, 0.9172)),
    "tum_fr3": Camera(535.4, 539.2, 320.1, 247.6),  # fr3 images are already undistorted
    "bonn": Camera(542.822841, 542.576870, 315.593520, 237.756098,
                   (0.039903, -0.099343, -0.000730, -0.000144, 0.0)),
    "ros_default": Camera(525.0, 525.0, 319.5, 239.5),
}


def camera_for_sequence(seq_dir: str | Path) -> Camera:
    name = Path(seq_dir).name.lower()
    for key, tag in (("tum_fr1", "freiburg1"), ("tum_fr2", "freiburg2"),
                     ("tum_fr3", "freiburg3"), ("bonn", "bonn")):
        if tag in name:
            return CAMERAS[key]
    raise ValueError(f"Cannot infer camera from sequence name '{name}'; pass one explicitly.")


def read_file_list(path: Path) -> tuple[np.ndarray, list[str]]:
    """Read a TUM 'timestamp value...' list -> (timestamps, rest-of-line strings)."""
    stamps, rest = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        t, *vals = line.replace(",", " ").split()
        stamps.append(float(t))
        rest.append(" ".join(vals))
    return np.asarray(stamps), rest


def associate(t_a: np.ndarray, t_b: np.ndarray, max_dt: float = 0.02) -> list[tuple[int, int]]:
    """One-to-one nearest-timestamp matching, closest pairs first (TUM associate.py rule)."""
    order_b = np.argsort(t_b)
    tb = t_b[order_b]
    pos = np.clip(np.searchsorted(tb, t_a), 1, len(tb) - 1)
    left, right = pos - 1, pos
    nearest = np.where(np.abs(t_a - tb[left]) <= np.abs(t_a - tb[right]), left, right)
    diff = np.abs(t_a - tb[nearest])

    pairs, used_b = [], set()
    for ia in np.argsort(diff):  # each a appears once; resolve clashes on b by closeness
        if diff[ia] > max_dt:
            break
        ib = int(order_b[nearest[ia]])
        if ib in used_b:
            continue
        used_b.add(ib)
        pairs.append((int(ia), ib))
    pairs.sort()
    return pairs


@dataclass
class RawFrame:
    timestamp: float          # colour timestamp
    depth_timestamp: float
    rgb: np.ndarray           # H x W x 3 uint8, BGR
    depth_raw: np.ndarray     # H x W uint16, dataset units


class RGBDSequence:
    """Timestamp-associated colour/depth frames of one TUM or Bonn sequence."""

    def __init__(self, seq_dir: str | Path, camera: Camera | None = None, max_dt: float = 0.02):
        self.root = Path(seq_dir)
        self.camera = camera or camera_for_sequence(self.root)
        t_rgb, f_rgb = read_file_list(self.root / "rgb.txt")
        t_dep, f_dep = read_file_list(self.root / "depth.txt")
        frames = [(t_rgb[i], f_rgb[i], t_dep[j], f_dep[j])
                  for i, j in associate(t_rgb, t_dep, max_dt)]
        # Bonn 'static' lists 10,908 depth images but its official zip ships only the first
        # 10,000 - keep only pairs whose files exist
        self.frames = [f for f in frames
                       if (self.root / f[1]).is_file() and (self.root / f[3]).is_file()]
        if len(self.frames) < len(frames):
            print(f"[dataset] {self.root.name}: {len(frames) - len(self.frames)} listed frames "
                  f"have missing files, using {len(self.frames)}")
        if not self.frames:
            raise RuntimeError(f"No colour/depth pairs within {max_dt}s in {self.root}")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> RawFrame:
        t_rgb, f_rgb, t_dep, f_dep = self.frames[idx]
        rgb = cv2.imread(str(self.root / f_rgb), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(self.root / f_dep), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise IOError(f"Failed to read frame {idx} ({f_rgb}, {f_dep})")
        if depth.dtype != np.uint16:
            raise ValueError(f"Depth must be 16-bit, got {depth.dtype} in {f_dep}")
        return RawFrame(t_rgb, t_dep, rgb, depth)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def groundtruth(self) -> np.ndarray:
        """N x 8 array: timestamp tx ty tz qx qy qz qw."""
        stamps, rest = read_file_list(self.root / "groundtruth.txt")
        vals = np.array([[float(v) for v in r.split()] for r in rest])
        return np.column_stack([stamps, vals])
