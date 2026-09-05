"""Does the VLM gate earn its cost?  (Scenario 2 from the handoff.)

Takes a Stage-1-only prediction CSV, the same run with --verify, and the
verify log, and audits every KEEP/DROP the VLM made against ground truth:

    true_keep   a real event the gate correctly let through
    true_drop   a real event the gate wrongly killed      (recall cost)
    false_drop  a false alarm the gate correctly killed    (the gate's value)
    false_keep  a false alarm the gate let through         (missed value)

"True" for a timestamped video means the interval reaches the IoU threshold
against a same-class ground-truth event -- the arena's own definition, so a
partial overlap below threshold is a false alarm here too. For a Level-1
video (no timestamps) it means the video's ground-truth class matches.

Then prints Level-1 and temporal metrics before/after so the four numbers
above can be read next to their effect on the score.

    python scripts/gate_report.py --before out/pred_zs.csv \
        --after out/pred_zs_verified.csv --log out/verify_test.log \
        --gt $VAD_DATA/test/ground_truth.csv --iou 0.5
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vad.config import NORMAL          # noqa: E402
from vad.score import _iou, level1, temporal  # noqa: E402

LINE = re.compile(r"verify (\S+) (\S+) \[([\d.]+)-([\d.]+)\] -> (KEEP|DROP) :: (.*)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="prediction CSV without --verify")
    ap.add_argument("--after", required=True, help="same run with --verify")
    ap.add_argument("--log", required=True, help="stdout of the --verify run")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()

    gt = pd.read_csv(a.gt)
    gt.columns = [c.strip().lower() for c in gt.columns]
    events = defaultdict(list)          # (vid, cls) -> [(s, e)]
    vid_cls: dict[str, str] = {}        # vid -> anomaly class (Level 1 truth)
    timestamped: set[str] = set()
    for _, r in gt.iterrows():
        vid, cls = str(r["video_id"]), str(r.get("class_name", NORMAL))
        if cls == NORMAL:
            vid_cls.setdefault(vid, NORMAL)
            continue
        vid_cls[vid] = cls
        try:
            s, e = float(r["start_time_sec"]), float(r["end_time_sec"])
        except (TypeError, ValueError):
            continue
        if e > s:
            events[(vid, cls)].append((s, e))
            timestamped.add(vid)

    tally = defaultdict(int)
    per_cls = defaultdict(lambda: defaultdict(int))
    n_lines = 0
    for line in open(a.log, errors="replace"):
        m = LINE.search(line)
        if not m:
            continue
        n_lines += 1
        vid, cls, s, e, dec = m.group(1), m.group(2), float(m.group(3)), float(m.group(4)), m.group(5)
        if vid in timestamped:
            best = max((_iou(s, e, g0, g1) for g0, g1 in events.get((vid, cls), [])), default=0.0)
            truth = "true" if best >= a.iou else "false"
        else:
            truth = "true" if vid_cls.get(vid) == cls else "false"
        key = f"{truth}_{dec.lower()}"
        tally[key] += 1
        per_cls[cls][key] += 1

    if n_lines == 0:
        sys.exit("no 'verify <vid> <cls> [...] -> KEEP|DROP' lines found in the log -- "
                 "was it produced with the video-id logging in verify_intervals?")

    tk, td = tally["true_keep"], tally["true_drop"]
    fd, fk = tally["false_drop"], tally["false_keep"]
    print(f"== VLM gate audit ({n_lines} decisions, IoU>={a.iou}) ==")
    print(f"  real events     : kept {tk:3d}   dropped {td:3d}   -> recall kept {tk/max(1,tk+td):.0%}")
    print(f"  false alarms    : dropped {fd:3d}   kept {fk:3d}   -> false alarms removed {fd/max(1,fd+fk):.0%}")
    print(f"  net: {fd} false alarms removed at the cost of {td} real events")
    print()
    print(f"  {'class':34s} {'true_keep':>9s} {'true_drop':>9s} {'false_drop':>10s} {'false_keep':>10s}")
    for cls in sorted(per_cls, key=lambda c: -sum(per_cls[c].values())):
        d = per_cls[cls]
        print(f"  {cls:34s} {d['true_keep']:9d} {d['true_drop']:9d} {d['false_drop']:10d} {d['false_keep']:10d}")

    print()
    print(f"== score before / after (temporal at IoU>={a.iou}) ==")
    for name, path in [("Stage 1 only", a.before), ("+ VLM gate", a.after)]:
        p = pd.read_csv(path)
        p.columns = [c.strip().lower() for c in p.columns]
        l1, t = level1(p, gt), temporal(p, gt, a.iou)
        print(f"  {name:13s}  L1 f1 {l1['f1']:.3f}  FAR {l1['false_alarm_rate']:.3f}  "
              f"class_acc {l1['class_acc_on_detected']:.3f}  |  temporal macroF1 {t['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
