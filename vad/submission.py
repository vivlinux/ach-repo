"""Stage 3: arena submission JSON.

Nothing upstream of this writes the format the arena actually scores —
`predict` writes a ground-truth-shaped CSV for local iteration. This module
is the translation layer, and it encodes every rule from the submission-format
spec so a bad file gets caught here, not as a burned run on the leaderboard:

  - Level 1 rows must carry null timestamps; Levels 2/3 must carry real ones.
  - "normal" is never a class_name -- it is `"events": []`.
  - A video absent from your file scores as normal (per spec), not an error.
  - explanation is optional and only checked if present (20-500 chars).
  - runtime_metadata is required on every video; model_runtimes' average must
    match total/calls within 2%, and call_times_ms must have call_count entries.

`predict`'s CSV already contains both a whole-video decision (its internal
level==1 row) and any detected intervals (level==2 rows), independent of what
the arena's manifest actually asks for on that video. build() picks the right
one per video based on the manifest's level.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import DATA, LABELS, NORMAL

REQUIRED_VIDEO_KEYS = ("video_id", "events", "runtime_metadata")


def load_manifest(path: str | Path | None, data_root: Path | None = None) -> dict[str, int]:
    """video_id -> level.

    Pass the real manifest.json downloaded from the arena's Benchmark tab.
    Without one (e.g. while iterating locally before that's available), falls
    back to this dataset's own test/ground_truth.csv level column -- useful
    for local dry runs, but NOT a substitute for the arena's actual manifest,
    which is the one that determines what you're scored against.
    """
    if path:
        raw = json.loads(Path(path).read_text())
        items = raw.get("videos", raw) if isinstance(raw, dict) else raw
        out: dict[str, int] = {}
        if isinstance(items, dict):
            for vid, v in items.items():
                out[str(vid)] = int(v["level"] if isinstance(v, dict) else v)
        else:
            for it in items:
                out[str(it["video_id"])] = int(it["level"])
        return out
    gt = Path(data_root or DATA) / "test" / "ground_truth.csv"
    if not gt.exists():
        raise FileNotFoundError(
            f"no --manifest given and {gt} not found. Pass the manifest.json "
            "from the arena's Benchmark tab -> Submission format; this dataset's "
            "own ground_truth.csv is only a stand-in for local dry runs.")
    df = pd.read_csv(gt)
    return {str(r.video_id): int(r.level) for r in df.itertuples()}


def _event_from_row(r) -> dict | None:
    cls = str(getattr(r, "class_name", "")).strip()
    if not cls or cls == "nan" or cls == NORMAL:
        return None
    return {"class_name": cls}


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return round(s[i], 1)


def _runtime_block(rt: dict | None, vlm_model: str | None) -> dict:
    """runtime_metadata for one video. Required on every video, and the
    latency bonus (reported time / video duration) is computed from it.
    average_time_ms must equal total/calls within 2%, and call_times_ms must
    have exactly call_count entries -- so derive all of it from one list."""
    if not rt:
        return {"frames_processed": 0, "chunks_processed": 0,
                "end_to_end_internal_time_ms": 0.0, "model_runtimes": []}
    calls = list(rt.get("vlm_call_times_ms") or [])
    models = []
    if calls and vlm_model:
        total = round(sum(calls), 1)
        models.append({
            "model_name": vlm_model,
            "call_count": len(calls),
            "total_time_ms": total,
            "average_time_ms": round(total / len(calls), 1),
            "p50_time_ms": _pct(calls, 0.50),
            "p95_time_ms": _pct(calls, 0.95),
            "max_time_ms": round(max(calls), 1),
        })
    return {
        "frames_processed": int(rt.get("frames_processed", 0)),
        "chunks_processed": int(rt.get("chunks_processed", 0)),
        "end_to_end_internal_time_ms": float(rt.get("end_to_end_internal_time_ms", 0.0)),
        "model_runtimes": models,
    }


def build(pred_csv: str | Path, manifest: dict[str, int], *,
          submission_id: str = "local-run",
          model_name: str = "vad-cascade",
          hardware: str = "1x Apple M2 Max",
          runtime_json: str | Path | None = None) -> dict:
    df = pd.read_csv(pred_csv)
    df["video_id"] = df["video_id"].astype(str)
    # sidecar written by `predict`: real per-video timings
    rt_path = Path(runtime_json) if runtime_json else Path(pred_csv).with_suffix(".runtime.json")
    rt_all, vlm_model = {}, None
    if rt_path.exists():
        blob = json.loads(rt_path.read_text())
        rt_all = blob.get("videos", {})
        vlm_model = blob.get("vlm_model")
    else:
        print(f"[submission] no {rt_path.name} -- runtime_metadata will be zeros. "
              f"Required field, but the latency bonus needs real numbers; "
              f"re-run `predict` to generate it.")

    preds, missing = [], []
    for vid, level in manifest.items():
        rows = df[df.video_id == vid]
        if rows.empty:
            missing.append(vid)
            events: list[dict] = []
        elif level == 1:
            r1 = rows[rows.level == 1]
            r1 = r1.iloc[0] if len(r1) else rows.iloc[0]
            ev = _event_from_row(r1)
            events = [] if ev is None else [
                {**ev, "start_time_sec": None, "end_time_sec": None}
            ]
        else:
            events = []
            for r in rows[rows.level == 2].itertuples():
                ev = _event_from_row(r)
                if ev is None:
                    continue
                s, e = float(r.start_time_sec), float(r.end_time_sec)
                if e <= s or s < 0:
                    continue
                ev["start_time_sec"] = round(s, 2)
                ev["end_time_sec"] = round(e, 2)
                desc = str(getattr(r, "description_summary", "") or "")
                if desc.startswith("confidence="):
                    desc = ""
                if 20 <= len(desc) <= 500:
                    ev["explanation"] = desc
                events.append(ev)

        preds.append({
            "video_id": vid,
            "events": events,
            "runtime_metadata": _runtime_block(rt_all.get(vid), vlm_model),
        })

    if missing:
        print(f"[submission] {len(missing)} manifest video(s) had no row in "
              f"{pred_csv} -> written as normal (events=[]), scored as normal "
              f"either way per spec: {missing[:10]}{' ...' if len(missing) > 10 else ''}")

    return {
        "schema_version": "1.0",
        "submission_id": submission_id,
        "model_name": model_name,
        "run_metadata": {
            "total_wall_time_ms": round(sum(
                p["runtime_metadata"]["end_to_end_internal_time_ms"] for p in preds), 1),
            "max_parallel_videos": 1,
            "hardware": hardware,
        },
        "predictions": preds,
    }


def validate(sub: dict, manifest: dict[str, int]) -> tuple[list[str], list[str]]:
    """(errors, notes). Errors are spec violations that would get the file
    rejected or the video zero-scored; notes are informational (e.g. a video
    you never answered -- allowed, scored as normal, but worth a glance).
    Empty errors means clean against the spec as written -- not a guarantee
    the arena's own validator agrees, since that's the one ground truth we
    can't see."""
    problems: list[str] = []
    notes: list[str] = []
    if "predictions" not in sub:
        return (["top-level 'predictions' key is missing -- the only strictly required field"], [])

    seen: set[str] = set()
    by_id: dict[str, dict] = {}
    for p in sub["predictions"]:
        vid = p.get("video_id")
        for k in REQUIRED_VIDEO_KEYS:
            if k not in p:
                problems.append(f"{vid or '?'}: missing required field '{k}'")
        if vid in seen:
            problems.append(f"{vid}: video_id appears more than once in predictions")
        seen.add(vid)
        if vid is not None:
            by_id[vid] = p
        if vid is not None and vid not in manifest:
            problems.append(f"{vid}: not in the manifest -- won't match anything to score")

    for vid, level in manifest.items():
        p = by_id.get(vid)
        if p is None:
            continue  # absent -> scored as normal per spec, not itself an error
        for ev in p.get("events", []):
            cls = ev.get("class_name")
            if cls == NORMAL:
                problems.append(f"{vid}: class_name='normal' is rejected -- use events: []")
            elif cls not in LABELS:
                problems.append(f"{vid}: class_name '{cls}' is not one of the 11 classes")
            s, e = ev.get("start_time_sec"), ev.get("end_time_sec")
            if level == 1:
                if s is not None or e is not None:
                    problems.append(f"{vid}: level-1 event has non-null timestamps -- must be null")
            else:
                if s is None or e is None:
                    problems.append(f"{vid}: level-{level} event missing start/end timestamps")
                elif s < 0:
                    problems.append(f"{vid}: start_time_sec {s} < 0")
                elif e <= s:
                    problems.append(f"{vid}: end_time_sec {e} must be greater than start_time_sec {s}")
            expl = ev.get("explanation")
            if expl is not None and not (20 <= len(expl) <= 500):
                problems.append(f"{vid}: explanation length {len(expl)} outside 20-500 chars "
                                 f"(safe to omit; invalid only if present and out of range)")
        rt = p.get("runtime_metadata") or {}
        for mr in (rt.get("model_runtimes") or []):
            tt, cc, av = mr.get("total_time_ms"), mr.get("call_count"), mr.get("average_time_ms")
            if tt is not None and cc not in (None, 0) and av is not None:
                if abs(av - tt / cc) > 0.02 * max(abs(av), 1e-9):
                    problems.append(f"{vid}: model_runtimes '{mr.get('model_name')}' "
                                     f"average_time_ms doesn't match total/calls within 2%")
            ct = mr.get("call_times_ms")
            if ct is not None and cc is not None and len(ct) != cc:
                problems.append(f"{vid}: model_runtimes '{mr.get('model_name')}' call_times_ms "
                                 f"has {len(ct)} entries, call_count says {cc}")

    for vid in manifest:
        if vid not in by_id:
            notes.append(f"{vid}: no prediction supplied -- will score as normal")
    return problems, notes
