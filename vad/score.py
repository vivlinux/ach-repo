"""Local scoring against ground_truth.csv, so you can tune before submitting."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .config import ANOMALY_LABELS, NORMAL


def _iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def level1(pred: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """Video-level binary detection + class accuracy on true positives."""
    g = {}
    for _, r in gt.iterrows():
        vid = str(r["video_id"])
        cls = str(r.get("class_name", NORMAL))
        if vid not in g or cls != NORMAL:
            g[vid] = cls
    p = {}
    for _, r in pred.iterrows():
        vid = str(r["video_id"])
        cls = str(r.get("class_name", NORMAL))
        if vid not in p or cls != NORMAL:
            p[vid] = cls
    tp = fp = fn = tn = 0
    cls_ok = cls_n = 0
    for vid, gc in g.items():
        pc = p.get(vid, NORMAL)
        ga, pa = gc != NORMAL, pc != NORMAL
        if ga and pa:
            tp += 1
            cls_n += 1
            cls_ok += int(pc == gc)
        elif ga and not pa:
            fn += 1
            cls_n += 1
        elif pa:
            fp += 1
        else:
            tn += 1
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {
        "videos": len(g), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(2 * prec * rec / max(1e-9, prec + rec), 4),
        "false_alarm_rate": round(fp / max(1, fp + tn), 4),
        "class_acc_on_detected": round(cls_ok / max(1, cls_n), 4),
    }


def temporal(pred: pd.DataFrame, gt: pd.DataFrame, iou_thr: float = 0.2) -> dict:
    """Greedy interval matching, same class required, per-class F1."""
    def collect(df):
        d = defaultdict(list)
        for _, r in df.iterrows():
            cls = str(r.get("class_name", NORMAL))
            if cls == NORMAL:
                continue
            s, e = r.get("start_time_sec"), r.get("end_time_sec")
            try:
                s, e = float(s), float(e)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(s) or not np.isfinite(e) or e <= s:
                continue
            d[(str(r["video_id"]), cls)].append((s, e))
        return d

    G, P = collect(gt), collect(pred)
    per = defaultdict(lambda: [0, 0, 0])            # tp, fp, fn
    for key in set(G) | set(P):
        cls = key[1]
        gs, ps = list(G.get(key, [])), list(P.get(key, []))
        used = set()
        for pi, (a0, a1) in enumerate(ps):
            best, bi = 0.0, -1
            for gi, (b0, b1) in enumerate(gs):
                if gi in used:
                    continue
                v = _iou(a0, a1, b0, b1)
                if v > best:
                    best, bi = v, gi
            if best >= iou_thr:
                used.add(bi)
                per[cls][0] += 1
            else:
                per[cls][1] += 1
        per[cls][2] += len(gs) - len(used)

    rows, f1s = {}, []
    for cls in ANOMALY_LABELS:
        tp, fp, fn = per.get(cls, [0, 0, 0])
        if tp + fp + fn == 0:
            continue
        pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
        f1 = 2 * pr * rc / max(1e-9, pr + rc)
        rows[cls] = dict(tp=tp, fp=fp, fn=fn, precision=round(pr, 3),
                         recall=round(rc, 3), f1=round(f1, 3))
        f1s.append(f1)
    return {"iou_thr": iou_thr, "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4),
            "per_class": rows}


def report(pred_csv, gt_csv, iou_thr: float = 0.2) -> str:
    pred, gt = pd.read_csv(pred_csv), pd.read_csv(gt_csv)
    gt.columns = [c.strip().lower() for c in gt.columns]
    pred.columns = [c.strip().lower() for c in pred.columns]
    l1 = level1(pred, gt)
    t = temporal(pred, gt, iou_thr)
    lines = ["== Level 1 (video-level) =="]
    lines += [f"  {k:26s} {v}" for k, v in l1.items()]
    lines += ["", f"== Level 2/3 (temporal, IoU>={iou_thr}) ==",
              f"  macro_f1                   {t['macro_f1']}"]
    for cls, r in sorted(t["per_class"].items(), key=lambda x: -x[1]["f1"]):
        lines.append(f"  {cls:34s} P{r['precision']:.2f} R{r['recall']:.2f} "
                     f"F1 {r['f1']:.2f}  (tp{r['tp']} fp{r['fp']} fn{r['fn']})")
    return "\n".join(lines)
