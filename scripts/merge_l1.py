"""Merge Level-1 answers: head OR VLM says anomaly, VLM's class on disagreement.

Why this policy, not "agree-only" (measured P75/R30 where both models agree):
Level 1 marks = accuracy over all L1 videos (arena_score.py, corrected
2026-09-05), and 20 of the public set's 24 L1 videos are real anomalies.
Under that metric, answering "normal" more often (which is what agree-only
does on every video the two models call differently) trades a few normal
videos' worth of upside for many anomalies' worth of downside -- recall
matters more than precision here. (Agree-only WOULD be the right call for
an operator false-alert budget; that's a different objective, noted in the
deck as a deliberate choice, not the one this metric rewards.)

So: union for the binary answer (whichever model says anomaly wins), and
the VLM's class when it says anomaly (measured class accuracy .458 vs the
head's .417) -- otherwise the head's class stands.

L2/L3 rows are untouched; they come from --head-pred as-is.

    python scripts/merge_l1.py --head-pred out/pred_head_timed.csv \
        --vlm-pred out/pred_l1_q3.csv --out out/pred_merged_l1.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vad.config import NORMAL  # noqa: E402

COLS = ["video_id", "level", "is_anomaly", "class_name",
        "start_time_sec", "end_time_sec", "description_summary"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-pred", dest="head_pred", required=True)
    ap.add_argument("--vlm-pred", dest="vlm_pred", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    head = pd.read_csv(a.head_pred)
    vlm = pd.read_csv(a.vlm_pred)
    head["video_id"] = head["video_id"].astype(str)
    vlm["video_id"] = vlm["video_id"].astype(str)

    head_l1 = {r.video_id: r for r in head[head.level == 1].itertuples()}
    vlm_l1 = {r.video_id: r for r in vlm[vlm.level == 1].itertuples()}

    rows = []
    n_flip_bin, n_flip_cls = 0, 0
    for vid, hr in head_l1.items():
        vr = vlm_l1.get(vid)
        h_anom, h_cls = hr.class_name != NORMAL, hr.class_name
        v_anom, v_cls = (vr.class_name != NORMAL, vr.class_name) if vr is not None else (False, NORMAL)

        anom = h_anom or v_anom
        if not anom:
            cls = NORMAL
        elif v_anom:
            cls = v_cls               # VLM's class wins when it says anomaly
        else:
            cls = h_cls               # only the head said anomaly

        if anom != h_anom:
            n_flip_bin += 1
        if cls != h_cls:
            n_flip_cls += 1

        rows.append(dict(zip(COLS, [vid, 1, int(anom), cls, "", "",
                                    f"merged(head={h_cls},vlm={v_cls})"])))

    # L2/L3 rows pass through from the head file untouched
    rows += head[head.level == 2].to_dict("records")

    out = pd.DataFrame(rows, columns=COLS)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({len(head_l1)} L1 videos, "
          f"{n_flip_bin} binary flips, {n_flip_cls} class changes vs head-only)")


if __name__ == "__main__":
    main()
