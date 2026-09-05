"""One command: is this head checkpoint better, or just louder?

The keep/discard oracle for the hidden-set robustness plan
(~/.claude/plans/ok-we-have-come-bright-creek.md). Every retrain/wiring
change in that plan is judged by running this before and after.

Reports, for one checkpoint (or a comma-list averaged into an ensemble):
  - emulator score (scripts/arena_score.py, corrected L1 formula)
  - SATURATION: fraction of test windows whose max class score is <0.01 or
    >0.99. The current head (out/head.pt) is badly saturated -- loitering
    sits at 1.000 on >10% of ALL test windows -- which is why per-class
    threshold tuning has nothing to grip. A de-saturated head should show
    this number drop even before the arena score moves.
  - per-class FIRE RATE ON THE 6 GT-NORMAL test videos (T001-T004, T029,
    T030): does this class hallucinate on footage with nothing in it.
  - per-class max score: confirms whether the self-suppressed classes
    (stalled/wrong_way/road_spill, each one camera or no data) stay
    suppressed after a change, or start firing blind.
  - train/val gap, parsed from a training log if given.

    python scripts/head_report.py --ckpt out/head.pt
    python scripts/head_report.py --ckpt out/head.pt --pred out/pred_head_timed.csv
    python scripts/head_report.py --ckpt a.pt,b.pt,c.pt --pred out/pred_ens.csv
    python scripts/head_report.py --ckpt out/head_reg.pt --train-log out/train_head_reg.log
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vad import head as H                                     # noqa: E402
from vad import io as VIO                                      # noqa: E402
from vad.config import ANOMALY_LABELS, DATA, L2I, LABELS, WIN, STRIDE  # noqa: E402
from vad.embed import Embedder, embed_video, pick_device        # noqa: E402
from vad.windows import pack, window_bounds                     # noqa: E402
from scripts.arena_score import load_gt, load_pred, score        # noqa: E402

NORMAL_TEST_VIDEOS = ["T001", "T002", "T003", "T004", "T029", "T030"]


def train_val_gap(log_path: str | None):
    if not log_path or not Path(log_path).exists():
        return None
    lines = Path(log_path).read_text().splitlines()
    rows = []
    for ln in lines:
        m = re.search(r"epoch\s+(\d+)/\d+\s+train\s+([\d.]+)\s+val\s+([\d.]+)", ln)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    if not rows:
        return None
    best = min(rows, key=lambda r: r[2])
    final = rows[-1]
    return dict(best_epoch=best[0], best_train=best[1], best_val=best[2],
                final_epoch=final[0], final_train=final[1], final_val=final[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="comma-list of head .pt files; averaged if >1")
    ap.add_argument("--pred", default=None, help="a predictions CSV already built from this "
                    "checkpoint -- if given, also prints the emulator score")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--train-log", dest="train_log", default=None)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()

    gt_path = a.gt or str(Path(DATA) / "test" / "ground_truth.csv")

    emb = Embedder(a.model)
    dev = pick_device()
    ckpts = [c.strip() for c in a.ckpt.split(",") if c.strip()]
    models = [H.load(Path(c), dev)[0] for c in ckpts]
    print(f"[head_report] {len(models)} checkpoint(s): {ckpts}")

    vids = VIO.load_split("test")
    all_scores = []                      # (video, spans, scores[W,C])
    for v in vids:
        try:
            f, t = embed_video(emb, v.path, VIO.cache_key(v), fps=a.fps)
        except Exception as e:                              # noqa: BLE001
            print(f"  !! {v.vid}: {e}")
            continue
        if len(f) < 4:
            continue
        spans = window_bounds(t, WIN, STRIDE)
        x = pack(f, spans)
        s = np.mean([H.predict(m, x, dev) for m in models], axis=0)
        all_scores.append((v, spans, s))

    # ---- saturation: fraction of windows where the max class score is
    # extreme (near 0 or near 1) -- the sign that thresholds have nothing
    # left to grip
    all_max = np.concatenate([s.max(1) for _, _, s in all_scores])
    saturation = float(np.mean((all_max < 0.01) | (all_max > 0.99)))
    print(f"\nSATURATION (windows with max score <0.01 or >0.99): {saturation:.1%}")

    # ---- per-class max score across the whole test set
    print(f"\n{'class':34s} {'max score':>10s}  {'fire-rate on 6 normals':>24s}")
    normal_vids = {v.vid: (spans, s) for v, spans, s in all_scores if v.vid in NORMAL_TEST_VIDEOS}
    for c in ANOMALY_LABELS:
        col = L2I[c]
        mx = max((s[:, col].max() for _, _, s in all_scores), default=0.0)
        # "fires" on a normal video if any window's score for this class
        # crosses the same hi threshold PP.extract would use by default (0.5
        # as a generic cutoff here, independent of PRIORS, to keep this a
        # pure model-behaviour diagnostic rather than a priors diagnostic)
        fires = sum(1 for _, (spans, s) in normal_vids.items() if s[:, col].max() > 0.5)
        print(f"  {c:32s} {mx:10.3f}  {fires:6d} / {len(normal_vids)} normal videos")

    if a.pred:
        lvl, gt_ev, gt_cls = load_gt(gt_path)
        p_l1, p_iv = load_pred(a.pred)
        sc = score(lvl, gt_ev, gt_cls, p_l1, p_iv)
        total = sc[1] + sc[2] + sc[3]
        print(f"\nEMULATOR SCORE   L1 {sc[1]:.1f}/25   L2 {sc[2]:.1f}/35   "
              f"L3 {sc[3]:.1f}/40   TOTAL {total:.1f}/100")
        normal_intervals = sum(len(p_iv.get(v, [])) for v in NORMAL_TEST_VIDEOS)
        print(f"intervals predicted on the 6 GT-normal videos: {normal_intervals}  "
              f"(each one costs that video its full L2/L3 credit)")
        silenced = sum(1 for v, evs in gt_ev.items() if evs and not p_iv.get(v))
        print(f"GT-event videos with zero predicted intervals (lost alert credit): {silenced}")

    gap = train_val_gap(a.train_log)
    if gap:
        print(f"\nTRAIN/VAL GAP (from {a.train_log}):")
        print(f"  best-val checkpoint: epoch {gap['best_epoch']}  train {gap['best_train']:.4f}  "
              f"val {gap['best_val']:.4f}")
        print(f"  final epoch:         epoch {gap['final_epoch']}  train {gap['final_train']:.4f}  "
              f"val {gap['final_val']:.4f}")
        print(f"  gap at final epoch: {gap['final_val'] - gap['final_train']:.4f} "
              f"(bigger = more overfit)")


if __name__ == "__main__":
    main()
