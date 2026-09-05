"""Window scores -> event intervals.

Per-class hysteresis plus per-class minimum duration is what makes the
different event dynamics work: an accident can fire on one window, a stalled
vehicle needs eight seconds of persistence before it counts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ANOMALY_LABELS, DEFAULT_PRIOR, L2I, PRIORS


@dataclass
class Interval:
    cls: str
    start: float
    end: float
    score: float


def smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    if len(x) < k or k < 2:
        return x
    pad = k // 2
    p = np.pad(x, (pad, pad), mode="edge")
    return np.array([np.median(p[i:i + k]) for i in range(len(x))], np.float32)


def _runs(score: np.ndarray, hi: float, lo: float):
    """Hysteresis: a run opens at hi and stays open until it drops below lo."""
    runs, on, s = [], False, 0
    for i, v in enumerate(score):
        if not on and v >= hi:
            on, s = True, i
        elif on and v < lo:
            runs.append((s, i - 1))
            on = False
    if on:
        runs.append((s, len(score) - 1))
    return runs


# Length-aware merging. PRIORS.merge_gap (2-10 s) was set for the 5 s events
# that dominate train, where no event exceeds 30 s. Test Level 3 has 125 s /
# 75 s / 45 s events on 5-10 minute videos, and the arena scores at IoU 0.5
# with fragments counting *against* you. So on long videos, slow classes
# (min_dur >= LONG_MERGE_MIN_DUR) get merge_gap widened to a fraction of the
# video length, capped. Off by default: it is untested on anything but the
# 34 public videos, and tuning on those is tuning on noise.
LONG_MERGE_VIDEO_S = 120.0     # only videos longer than this
LONG_MERGE_FRAC = 0.10         # merge_gap -> max(merge_gap, frac * duration)
LONG_MERGE_CAP_S = 60.0
LONG_MERGE_MIN_DUR = 4.0       # "slow" classes: congestion, stalled, blocking, flood, loitering


def extract(scores: np.ndarray, spans, scale: float = 1.0,
            classes=None, smooth_k: int | None = None,
            long_merge: bool = False) -> list[Interval]:
    """scores [W,C] -> intervals. `scale` shifts every threshold at once."""
    out: list[Interval] = []
    if len(spans) == 0:
        return out
    starts = np.array([s[2] for s in spans])
    ends = np.array([s[3] for s in spans])
    duration = float(ends[-1] - starts[0])
    for cls in (classes or ANOMALY_LABELS):
        p = PRIORS.get(cls, DEFAULT_PRIOR)
        k = smooth_k if smooth_k is not None else p.get("smooth_k", 3)
        col = smooth(scores[:, L2I[cls]].astype(np.float32), k)
        hi, lo = p["hi"] * scale, p["lo"] * scale
        gap = p["merge_gap"]
        if long_merge and duration > LONG_MERGE_VIDEO_S and p["min_dur"] >= LONG_MERGE_MIN_DUR:
            gap = max(gap, min(LONG_MERGE_CAP_S, LONG_MERGE_FRAC * duration))
        cand = []
        for a, b in _runs(col, hi, lo):
            cand.append(Interval(cls, float(starts[a]), float(ends[b]),
                                 float(col[a:b + 1].max())))
        # merge, then enforce minimum duration
        merged: list[Interval] = []
        for iv in sorted(cand, key=lambda x: x.start):
            if merged and iv.start - merged[-1].end <= gap:
                merged[-1].end = max(merged[-1].end, iv.end)
                merged[-1].score = max(merged[-1].score, iv.score)
            else:
                merged.append(iv)
        out += [iv for iv in merged if iv.end - iv.start >= p["min_dur"] - 1e-6]
    return sorted(out, key=lambda x: x.start)


def video_decision(scores: np.ndarray, intervals: list[Interval]):
    """Level-1 answer: (is_anomaly, class_name, confidence)."""
    if intervals:
        best = max(intervals, key=lambda i: (i.score, i.end - i.start))
        return 1, best.cls, best.score
    if len(scores) == 0:
        return 0, "normal", 0.0
    peak = scores[:, [L2I[c] for c in ANOMALY_LABELS]].max(0)
    j = int(peak.argmax())
    if peak[j] >= 0.5:
        return 1, ANOMALY_LABELS[j], float(peak[j])
    return 0, "normal", float(1.0 - peak[j])
