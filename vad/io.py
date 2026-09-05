"""Dataset IO.

The pack ships videos.csv + ground_truth.csv per class folder. Exact column
names are not guaranteed, so everything here resolves columns by candidate
name and falls back to globbing the videos/ directory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DATA, LABELS, NORMAL

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _pick(cols: Iterable[str], *cands: str) -> str | None:
    low = {str(c).strip().lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    for c in cands:                      # substring fallback
        for k, v in low.items():
            if c in k:
                return v
    return None


def _num(x):
    try:
        v = float(x)
        return None if v != v else v     # NaN -> None
    except (TypeError, ValueError):
        return None


@dataclass
class Event:
    cls: str
    start: float | None
    end: float | None
    level: int = 1
    desc: str = ""


@dataclass
class Video:
    vid: str
    path: Path
    split: str
    folder_class: str | None = None
    events: list[Event] = field(default_factory=list)

    @property
    def is_anomaly(self) -> bool:
        return any(e.cls != NORMAL for e in self.events)

    @property
    def has_timestamps(self) -> bool:
        return any(e.start is not None and e.end is not None and e.end > e.start
                   for e in self.events if e.cls != NORMAL)

    @property
    def classes(self) -> list[str]:
        return sorted({e.cls for e in self.events if e.cls != NORMAL})


def _index_videos(folder: Path) -> dict[str, Path]:
    """video_id -> path, from videos.csv when usable, else from the files."""
    vdir = folder / "videos"
    files = {}
    if vdir.is_dir():
        for p in sorted(vdir.iterdir()):
            if p.suffix.lower() in VIDEO_EXT:
                files[p.name] = p
                files[p.stem] = p
    csv = folder / "videos.csv"
    out: dict[str, Path] = {}
    if csv.exists():
        try:
            df = pd.read_csv(csv)
            idc = _pick(df.columns, "video_id", "id", "video")
            fc = _pick(df.columns, "file_name", "filename", "file", "path", "video_path", "video")
            if idc is not None:
                for _, r in df.iterrows():
                    vid = str(r[idc]).strip()
                    p = None
                    if fc is not None and str(r[fc]) != "nan":
                        name = Path(str(r[fc]).strip()).name
                        p = files.get(name) or files.get(Path(name).stem)
                        if p is None:
                            cand = folder / str(r[fc]).strip()
                            p = cand if cand.exists() else None
                    if p is None:
                        p = files.get(vid) or files.get(f"{vid}.mp4")
                    if p is not None:
                        out[vid] = p
        except Exception as e:                       # noqa: BLE001
            print(f"[io] videos.csv unreadable in {folder}: {e}")
    if not out:
        for p in sorted(set(files.values())):
            out[p.stem] = p
    return out


def _load_gt(folder: Path) -> dict[str, list[Event]]:
    # scripts/relabel_train.py fixes folder-level label noise (a description
    # that reads as a collision but sits in the road_spill_or_debris folder,
    # etc) and writes its output alongside the original, never overwriting
    # it. Prefer that corrected file when it exists.
    csv = folder / "ground_truth_relabeled.csv"
    if not csv.exists():
        csv = folder / "ground_truth.csv"
    ev: dict[str, list[Event]] = {}
    if not csv.exists():
        return ev
    df = pd.read_csv(csv)
    idc = _pick(df.columns, "video_id", "id", "video")
    cc = _pick(df.columns, "class_name", "class", "label")
    sc = _pick(df.columns, "start_time_sec", "start_time", "start")
    ec = _pick(df.columns, "end_time_sec", "end_time", "end")
    lc = _pick(df.columns, "level")
    dc = _pick(df.columns, "description_summary", "description", "caption")
    ac = _pick(df.columns, "is_anomaly", "anomaly")
    if idc is None:
        return ev
    for _, r in df.iterrows():
        vid = str(r[idc]).strip()
        cls = str(r[cc]).strip() if cc is not None and str(r[cc]) != "nan" else NORMAL
        if cls not in LABELS:
            if ac is not None and _num(r[ac]) in (0, 0.0):
                cls = NORMAL
            else:
                continue
        ev.setdefault(vid, []).append(Event(
            cls=cls,
            start=_num(r[sc]) if sc is not None else None,
            end=_num(r[ec]) if ec is not None else None,
            level=int(_num(r[lc]) or 1) if lc is not None else 1,
            desc=str(r[dc]) if dc is not None and str(r[dc]) != "nan" else "",
        ))
    return ev


def load_split(split: str, data_root: Path | None = None) -> list[Video]:
    """split is 'train' or 'test'. Train is scanned per class folder."""
    root = Path(data_root or DATA) / split
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found — point VAD_DATA at the folder holding train/ and test/")
    folders = []
    if (root / "videos").is_dir() or (root / "videos.csv").exists():
        folders.append((root, None))
    for d in sorted(root.iterdir()):
        if d.is_dir() and ((d / "videos").is_dir() or (d / "ground_truth.csv").exists()):
            folders.append((d, d.name if d.name in LABELS else None))

    vids: list[Video] = []
    seen: set[str] = set()
    for folder, fcls in folders:
        idx = _index_videos(folder)
        gt = _load_gt(folder)
        for vid, path in idx.items():
            key = f"{folder.name}:{vid}"
            if key in seen:
                continue
            seen.add(key)
            events = gt.get(vid, [])
            if not events:
                events = [Event(cls=fcls or NORMAL, start=None, end=None)]
            vids.append(Video(vid=vid, path=path, split=split, folder_class=fcls, events=events))
    return vids


def cache_key(v: Video) -> str:
    parent = v.path.parent.parent.name
    return f"{v.split}__{parent}__{v.vid}".replace("/", "_")


def summarize(vids: list[Video]) -> str:
    from collections import Counter
    c = Counter()
    for v in vids:
        for cls in (v.classes or [NORMAL]):
            c[cls] += 1
    strong = sum(1 for v in vids if v.has_timestamps)
    lines = [f"{len(vids)} videos | {strong} with timestamps | {len(vids)-strong} video-level only"]
    for k, n in c.most_common():
        lines.append(f"  {k:36s} {n}")
    return "\n".join(lines)
