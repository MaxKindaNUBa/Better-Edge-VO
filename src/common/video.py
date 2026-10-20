"""Writing .mp4 videos (shared by the edge demo and the DTVO runner)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


class VideoSink:
    """H.264 .mp4 via ffmpeg (plays in browsers, VS Code, PowerPoint); falls back to
    OpenCV's MPEG-4 'mp4v' writer if ffmpeg is not installed."""

    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int = 20):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.proc, self.cv_writer = path, None, None
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}",
                 "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                stdin=subprocess.PIPE)
        else:
            print("ffmpeg not found: writing MPEG-4 (mp4v), which some players can't open")
            self.cv_writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (width, height))

    def write(self, img: np.ndarray):
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(img).tobytes())
        else:
            self.cv_writer.write(img)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed while writing {self.path}")
        else:
            self.cv_writer.release()
