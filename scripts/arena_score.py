"""Emulate the arena's scoring so we can compare strategies before spending a run.

Rules implemented from the submission-format PDF:
  Level 1 (25 pts): pooled over Level-1 videos, half anomaly-vs-normal
      accuracy + half class accuracy.
  Levels 2 (35) / 3 (40): scored per video, then averaged.
      - ground truth normal -> predict nothing = 1, predict anything = 0
      - ground truth has events -> weighted mix of (did you alert),
        (matched events), (how well timings line up). Timing weighs more at L3.
      - a match needs the right class AND IoU >= 0.5; at most one prediction
        can match each ground-truth event, the rest count against you.

Calibration: two leaderboard entrants who submit nothing for L2/L3 score
exactly 11.7 and 0.0, and this emulator reproduces both -- 35 * (2/6 normal)
= 11.67. So the L1/L2/L3 caps and the normal-video rule are confirmed
against real arena behaviour. The alert/match/timing split inside an
event video is NOT published; ALERT_W below is our assumption, and
--weights lets you check that a conclusion survives changing it.

    python scripts/arena_score.py --pred out/pred_safety.csv --gt $VAD_DATA/test/ground_truth.csv
    python scripts/arena_score.py --pred out/pred_safety.csv --gt ... --compare
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vad.config import NORMAL  # noqa: E402

CAPS = {1: 25.0, 2: 35.0, 3: 40.0}
ALERT_W = {2: (0.2, 0.5, 0.3), 3: (0.2, 0.4, 0.4)}   # (alert, match, timing)
IOU_GATE = 0.5


def _iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def load_gt(path):
    gt = pd.read_csv(path)
    gt.columns = [c.strip().lower() for c in gt.columns]
    lvl, events, cls = {}, defaultdict(list), {}
    for _, r in gt.iterrows():
        v = str(r["video_id"])
        lvl[v] = int(r["level"])
        c = str(r.get("class_name", NORMAL))
        if c == NORMAL:
            cls.setdefault(v, NORMAL)
            continue
        cls[v] = c
        try:
            s, e = float(r["start_time_sec"]), float(r["end_time_sec"])
        except (TypeError, ValueError):
            continue
        if e > s:
            events[v].append((c, s, e))
    return lvl, events, cls


def load_pred(path):
    p = pd.read_csv(path)
    p.columns = [c.strip().lower() for c in p.columns]
    p["video_id"] = p["video_id"].astype(str)
    l1, ivs = {}, defaultdict(list)
    for _, r in p.iterrows():
        v, c = r["video_id"], str(r.get("class_name", NORMAL))
        if int(r["level"]) == 1:
            l1[v] = c
        else:
            try:
                s, e = float(r["start_time_sec"]), float(r["end_time_sec"])
            except (TypeError, ValueError):
                continue
            if e > s and c != NORMAL:
                ivs[v].append((c, s, e))
    return l1, ivs


def score(lvl, gt_ev, gt_cls, p_l1, p_iv, emit=lambda v, L: True, weights=None):
    weights = weights or ALERT_W
    L1 = [v for v in lvl if lvl[v] == 1]
    b = sum((gt_cls[v] != NORMAL) == (p_l1.get(v, NORMAL) != NORMAL) for v in L1)
    c = sum(p_l1.get(v, NORMAL) == gt_cls[v] for v in L1)
    out = {1: CAPS[1] * (0.5 * b / max(1, len(L1)) + 0.5 * c / max(1, len(L1)))}
    for L in (2, 3):
        vids = [v for v in lvl if lvl[v] == L]
        if not vids:
            out[L] = 0.0
            continue
        aw, mw, tw = weights[L]
        tot = 0.0
        for v in vids:
            ivs = p_iv.get(v, []) if emit(v, L) else []
            g = gt_ev.get(v, [])
            if not g:
                tot += 1.0 if not ivs else 0.0
                continue
            if not ivs:
                continue
            used, ious = set(), []
            for (pc, ps, pe) in ivs:
                best, bi = 0.0, -1
                for i, (gc, gs, ge) in enumerate(g):
                    if i in used or gc != pc:
                        continue
                    x = _iou(ps, pe, gs, ge)
                    if x > best:
                        best, bi = x, i
                if best >= IOU_GATE:
                    used.add(bi)
                    ious.append(best)
            m = len(used)
            pr, rc = m / len(ivs), m / len(g)
            f1 = 2 * pr * rc / max(1e-9, pr + rc)
            tot += aw + mw * f1 + tw * (float(np.mean(ious)) if ious else 0.0)
        out[L] = CAPS[L] * tot / len(vids)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--compare", action="store_true",
                    help="also score silent-on-L2 / silent-on-L3 / silent-everywhere")
    a = ap.parse_args()
    lvl, gt_ev, gt_cls = load_gt(a.gt)
    p_l1, p_iv = load_pred(a.pred)

    strategies = [("as predicted", lambda v, L: True)]
    if a.compare:
        strategies += [
            ("silent on L2, predict L3", lambda v, L: L != 2),
            ("predict L2, silent on L3", lambda v, L: L != 3),
            ("silent on L2 and L3", lambda v, L: False),
        ]
    print(f"{'strategy':32s} {'L1/25':>7s} {'L2/35':>7s} {'L3/40':>7s} {'TOTAL':>7s}")
    best = None
    for name, emit in strategies:
        s = score(lvl, gt_ev, gt_cls, p_l1, p_iv, emit)
        t = s[1] + s[2] + s[3]
        print(f"{name:32s} {s[1]:7.1f} {s[2]:7.1f} {s[3]:7.1f} {t:7.1f}")
        if best is None or t > best[1]:
            best = (name, t)
    if a.compare:
        print(f"\nbest: {best[0]} ({best[1]:.1f}/100)")
    print("\nnote: L1/L2/L3 caps and the normal-video rule are confirmed against the "
          "leaderboard (silence = 11.7 on L2). The alert/match/timing split inside an "
          "event video is assumed.")


if __name__ == "__main__":
    main()
