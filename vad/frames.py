"""Frame decoding. Grab-and-skip so we only pay JPEG decode on kept frames."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .config import FPS, FRAME_SIZE


def video_meta(path: Path) -> tuple[float, float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    fps = fps if 1.0 < fps < 240.0 else 25.0
    return fps, (n / fps if n > 0 else 0.0)


def iter_frames(path: Path, target_fps: float = FPS,
                size: int = FRAME_SIZE) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp_sec, RGB uint8 HxWx3) at roughly target_fps."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = src_fps if 1.0 < src_fps < 240.0 else 25.0
    step = max(1, int(round(src_fps / target_fps)))
    i = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % step == 0:
                ok, bgr = cap.retrieve()
                if not ok:
                    break
                if size:
                    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
                yield i / src_fps, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()


def sample_frames(path: Path, start: float, end: float, k: int = 6,
                  size: int = 336) -> list[np.ndarray]:
    """k frames evenly spread over [start, end] — used to feed the VLM."""
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = src_fps if 1.0 < src_fps < 240.0 else 25.0
    out = []
    span = max(end - start, 0.2)
    for j in range(k):
        t = start + span * (j + 0.5) / k
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, bgr = cap.read()
        if not ok:
            continue
        h, w = bgr.shape[:2]
        s = size / max(h, w)
        if s < 1.0:
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out
