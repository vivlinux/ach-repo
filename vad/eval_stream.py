"""Causal evaluation — scores alerts, not clips.

Clip-level metrics let a model see the whole video before deciding. This
matches each emitted alert against ground truth using only information
available at the alert timestamp, and reports the two numbers an operator
cares about: how long after onset the alert arrived, and how many false alerts
they get per drone-hour.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .alerts import Alert
from .io import Video


def match_alerts(alerts: list[Alert], videos: dict[str, Video],
                 require_class: bool = True, min_overlap: float = 0.1):
    """Greedy onset-ordered matching. Returns (tp, fp, fn, latencies)."""
    gt = defaultdict(list)
    for vid, v in videos.items():
        for e in v.events:
            if e.start is None or e.end is None or e.end <= e.start:
                continue
            gt[vid].append([e.cls, e.start, e.end, False])   # last = matched

    tp, fp, lat = [], [], []
    for a in sorted(alerts, key=lambda x: x.detected_at):
        vid = a.drone_id.split("+")[0]
        hit = None
        for g in gt.get(vid, []):
            if g[3]:
                continue
            if require_class and g[0] != a.cls:
                continue
            inter = max(0.0, min(a.t_end, g[2]) - max(a.t_start, g[1]))
            if inter / max(min(a.t_end - a.t_start, g[2] - g[1]), 1e-6) >= min_overlap:
                hit = g
                break
        if hit is not None:
            hit[3] = True
            tp.append(a)
            lat.append(max(0.0, a.detected_at - hit[1]))    # onset -> alert
        else:
            fp.append(a)
    fn = [g for gs in gt.values() for g in gs if not g[3]]
    return tp, fp, fn, lat


def report(alerts: list[Alert], videos: dict[str, Video], drone_hours: float,
           suppressed: int = 0) -> str:
    tp, fp, fn, lat = match_alerts(alerts, videos)
    n = len(tp) + len(fp)
    prec = len(tp) / max(n, 1)
    rec = len(tp) / max(len(tp) + len(fn), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    lat = sorted(lat)

    def q(p):
        return round(lat[min(int(p * len(lat)), len(lat) - 1)], 2) if lat else 0.0

    lines = [
        "== causal / streaming evaluation ==",
        f"  drone-hours evaluated        {drone_hours:.3f}",
        f"  alerts emitted               {n}   (deduped away: {suppressed})",
        f"  true / false                 {len(tp)} / {len(fp)}",
        f"  missed events                {len(fn)}",
        f"  precision / recall / f1      {prec:.3f} / {rec:.3f} / {f1:.3f}",
        "",
        "  -- the numbers an operator feels --",
        f"  FALSE ALERTS PER DRONE-HOUR  {len(fp)/max(drone_hours,1e-6):.2f}   (budget: 1-2)",
        f"  latency to detect  p50       {q(0.50)} s",
        f"                     p95       {q(0.95)} s",
        f"                     worst     {round(lat[-1],2) if lat else 0.0} s",
    ]
    by = defaultdict(lambda: [0, 0])
    for a in tp:
        by[a.cls][0] += 1
    for a in fp:
        by[a.cls][1] += 1
    if by:
        lines += ["", "  false alerts by class (where the noise comes from):"]
        for c, (t, f) in sorted(by.items(), key=lambda x: -x[1][1]):
            lines.append(f"    {c:34s} tp {t:3d}  fp {f:3d}")
    return "\n".join(lines)
