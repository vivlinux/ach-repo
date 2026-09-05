"""Level-1 answers by asking the VLM to CLASSIFY, not to gate.

Why this exists: the verification gate can only drop an interval, never
relabel one. Measured on the public test set, it moved L1 class accuracy
0.286 -> 0.321 -- because when Stage 1 picks the wrong class, the gate's only
options are keep-the-wrong-label or kill it (which answers "normal", also
wrong). Half of Level 1's 25 points is class accuracy, and the 24 Level-1
clips are 4.6 minutes of video in total, so a direct VLM classification on
every one is affordable and attacks the actual bottleneck.

Writes a predictions CSV in the same shape as `predict` (level-1 rows only)
plus the matching .runtime.json, so `vad.cli submit` consumes it unchanged.
Level-2/3 videos are left alone -- merge with a separate run for those.

    python scripts/l1_classify.py --out out/pred_l1.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vad.config import ANOMALY_LABELS, DATA, LABELS, NORMAL  # noqa: E402
from vad.frames import sample_frames, video_meta  # noqa: E402
from vad.io import load_split  # noqa: E402

COLS = ["video_id", "level", "is_anomaly", "class_name",
        "start_time_sec", "end_time_sec", "description_summary"]

# Plain-language gloss per class: the raw enum strings are jargon, and the
# model answers noticeably better when the option list reads like English.
GLOSS = {
    "traffic_accident": "a crash or collision between vehicles",
    "traffic_congestion": "a traffic jam, queued or slow-moving vehicles",
    "stalled_or_broken_down_vehicle": "a vehicle stopped or broken down where it should not be",
    "vehicle_blocking_traffic": "a vehicle obstructing the flow of other traffic",
    "wrong_way_driving": "a vehicle driving against the direction of surrounding traffic",
    "road_spill_or_debris": "spilled material or debris lying on the road",
    "waterlogging_or_flood": "flooding or standing water",
    "fire": "visible flames or an active fire",
    "smoke": "a visible plume of smoke",
    "fighting_or_violence": "people physically fighting",
    "loitering_or_suspicious_presence": "a person loitering or lingering suspiciously",
}


def build_prompt(style: str = "cautious") -> str:
    """Two wordings, both measured. `cautious` won on Qwen2.5-VL-3B (14.1 vs
    10.4/8.9). Whether it also wins on a stronger instruction-follower is a
    separate question: Qwen3-VL-8B scores the same 14.1 overall but with
    HIGHER class accuracy (.458 vs .417) and LOWER binary accuracy (.667 vs
    .708) -- i.e. it obeys the "only if clearly visible" hedge harder and
    sends real anomalies to `normal`. Hence `balanced`."""
    opts = "\n".join(f"- {c}: {GLOSS[c]}" for c in ANOMALY_LABELS)
    head = ("These frames are sampled in time order from one short surveillance, "
            "dashcam or drone video.\n\n"
            "Which ONE of these best describes the video?\n"
            f"{opts}\n"
            f"- {NORMAL}: nothing unusual is happening\n\n")
    if style == "cautious":
        tail = ("Most videos showing ordinary traffic or an ordinary scene are "
                f"'{NORMAL}'. Only choose an anomaly if it is clearly visible.\n")
    elif style == "balanced":
        # no thumb on the scale either way; just widen what counts as visible
        tail = ("Judge only what the frames show. Look at people as well as "
                "vehicles -- a scuffle, or someone lingering where they have no "
                "reason to be, both count.\n")
    else:
        raise SystemExit(f"unknown --prompt-style {style!r}")
    return head + tail + "Answer with the single category name and nothing else."


def parse(ans: str) -> str:
    """Map a free-text answer onto exactly one label."""
    a = ans.strip().lower().replace("-", "_").replace(" ", "_")
    for c in LABELS:                       # exact / substring on the enum
        if c in a:
            return c
    keys = {                               # fall back to distinctive words
        "fire": "fire", "flame": "fire", "smoke": "smoke",
        "flood": "waterlogging_or_flood", "water": "waterlogging_or_flood",
        "crash": "traffic_accident", "collision": "traffic_accident",
        "accident": "traffic_accident", "jam": "traffic_congestion",
        "congest": "traffic_congestion", "fight": "fighting_or_violence",
        "violen": "fighting_or_violence", "loiter": "loitering_or_suspicious_presence",
        "debris": "road_spill_or_debris", "spill": "road_spill_or_debris",
        "stall": "stalled_or_broken_down_vehicle", "broken": "stalled_or_broken_down_vehicle",
        "block": "vehicle_blocking_traffic", "obstruct": "vehicle_blocking_traffic",
        "wrong": "wrong_way_driving",
    }
    for k, v in keys.items():
        if k in a:
            return v
    return NORMAL                          # unparseable -> silence, the cheap error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/pred_l1.csv")
    ap.add_argument("--vlm", default=None, help="MLX id; default config.VLM_MODEL")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--prompt-style", dest="prompt_style", default="cautious",
                    choices=["cautious", "balanced"])
    ap.add_argument("--gt", default=None, help="to pick which videos are Level 1")
    ap.add_argument("--only", default="", help="comma-separated video ids")
    a = ap.parse_args()

    gt_path = Path(a.gt or Path(DATA) / "test" / "ground_truth.csv")
    levels = {str(r.video_id): int(r.level) for r in pd.read_csv(gt_path).itertuples()}
    want = {x.strip() for x in a.only.split(",") if x.strip()}

    vids = [v for v in load_split("test") if levels.get(v.vid) == 1]
    if want:
        vids = [v for v in vids if v.vid in want]
    print(f"{len(vids)} Level-1 videos")

    from vad.verify import Verifier
    vlm = Verifier(a.vlm, max_tokens=16)

    prompt = build_prompt(a.prompt_style)
    rows, runtime = [], {}
    for i, v in enumerate(vids, 1):
        t0 = time.time()
        before = len(vlm.call_times_ms)
        # sample_frames seeks by timestamp, so it needs the real duration --
        # a sentinel like 1e9 seeks past EOF and silently returns nothing
        _, dur = video_meta(v.path)
        frames = sample_frames(v.path, 0.0, dur if dur > 0 else 10.0, k=a.frames)
        cls = NORMAL
        raw = ""
        if not frames:
            print(f"  !! {v.vid}: no frames decoded (duration {dur:.1f}s)")
        if frames:
            raw = vlm.ask([Image.fromarray(f) for f in frames], prompt)
            cls = parse(raw)
        rows.append(dict(zip(COLS, [v.vid, 1, int(cls != NORMAL), cls, "", "",
                                    f"vlm={raw[:60]}"])))
        runtime[v.vid] = dict(
            frames_processed=len(frames), chunks_processed=1,
            end_to_end_internal_time_ms=round((time.time() - t0) * 1000, 1),
            vlm_call_times_ms=[round(x, 1) for x in vlm.call_times_ms[before:]])
        print(f"  {i:2d}/{len(vids)} {v.vid} -> {cls:34s} ({time.time()-t0:.1f}s) {raw[:40]!r}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLS).to_csv(out, index=False)
    out.with_suffix(".runtime.json").write_text(json.dumps(
        dict(videos=runtime, vlm_model=vlm.model_id, encoder=None,
             total_wall_time_ms=round(sum(r["end_to_end_internal_time_ms"]
                                          for r in runtime.values()), 1)), indent=2))
    print(f"wrote {out} and {out.with_suffix('.runtime.json')}")


if __name__ == "__main__":
    main()
