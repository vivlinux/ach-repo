"""Turn per-frame features into labelled temporal windows."""
from __future__ import annotations

import numpy as np

from .config import L2I, LABELS, NORMAL, STRIDE, WIN
from .io import Video


def window_bounds(times: np.ndarray, win: int = WIN, stride: int = STRIDE):
    """Yield (i0, i1, t_start, t_end) index/time spans."""
    n = len(times)
    if n == 0:
        return []
    if n < win:
        return [(0, n, float(times[0]), float(times[-1]) + 0.5)]
    out = []
    for i in range(0, n - win + 1, stride):
        out.append((i, i + win, float(times[i]), float(times[i + win - 1])))
    if out[-1][1] < n:                      # tail window
        out.append((n - win, n, float(times[n - win]), float(times[-1])))
    return out


def window_labels(v: Video, spans, overlap: float = 0.3) -> np.ndarray:
    """Multi-hot [W, C]. Only meaningful for videos with timestamps."""
    W = len(spans)
    y = np.zeros((W, len(LABELS)), np.float32)
    for w, (_, _, t0, t1) in enumerate(spans):
        dur = max(t1 - t0, 1e-6)
        hit = False
        for e in v.events:
            if e.cls == NORMAL or e.start is None or e.end is None:
                continue
            inter = max(0.0, min(t1, e.end) - max(t0, e.start))
            if inter <= 0:
                continue
            ev_dur = max(e.end - e.start, 1e-6)
            if inter / dur >= overlap or inter / ev_dur >= 0.5:
                y[w, L2I[e.cls]] = 1.0
                hit = True
        if not hit:
            y[w, L2I[NORMAL]] = 1.0
    return y


def pack(feat: np.ndarray, spans) -> np.ndarray:
    """[W, WIN, D] float32, tail-padded by repetition when the clip is short."""
    if len(feat) == 0:
        return np.zeros((0, WIN, 1), np.float32)
    out = np.zeros((len(spans), WIN, feat.shape[1]), np.float32)
    for w, (i0, i1, _, _) in enumerate(spans):
        chunk = feat[i0:i1]
        if len(chunk) < WIN:
            reps = np.repeat(chunk[-1:], WIN - len(chunk), 0) if len(chunk) else \
                np.zeros((WIN, feat.shape[1]), np.float32)
            chunk = np.concatenate([chunk, reps], 0)
        out[w] = chunk[:WIN]
    return out
