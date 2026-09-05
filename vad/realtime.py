"""Streaming demo: one pass over a video, alerts printed as they fire.

This is the thing to show at 6pm. It reports the achieved throughput so the
real-time claim is a measurement, not an assertion.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np

from . import head as H
from . import postprocess as PP
from . import zeroshot as ZS
from .config import ANOMALY_LABELS, DEFAULT_PRIOR, L2I, OUT, PRIORS, STRIDE, WIN
from .embed import Embedder, pick_device
from .frames import iter_frames


def run(a):
    emb = Embedder(getattr(a, "model", None), dtype=getattr(a, "dtype", "float32"))
    dev = pick_device() if a.device == "auto" else a.device
    model, T = None, None
    if a.mode == "head":
        model, _ = H.load(Path(a.ckpt or OUT / "head.pt"), dev)
    else:
        T = ZS.class_matrix(emb)

    buf = deque(maxlen=WIN)
    times = deque(maxlen=WIN)
    open_ev: dict[str, list] = {}
    n_frames = 0
    n_win = 0
    enc_time = 0.0
    wall = time.time()
    print(f"streaming {a.video} at {a.fps} fps ...")

    for t, img in iter_frames(Path(a.video), a.fps):
        n_frames += 1
        t0 = time.time()
        f = emb.encode_images([img])[0]
        enc_time += time.time() - t0
        buf.append(f)
        times.append(t)
        if len(buf) < WIN or n_frames % STRIDE:
            continue
        n_win += 1
        x = np.stack(buf)[None]
        if T is not None:
            v = x[0].mean(0)
            v /= np.linalg.norm(v) + 1e-8
            lg = (T @ v) / 0.01
            lg -= lg.max()
            e = np.exp(lg)
            s = (e / e.sum())
        else:
            s = H.predict(model, x, dev)[0]

        t_start, t_end = times[0], times[-1]
        for cls in ANOMALY_LABELS:
            p = PRIORS.get(cls, DEFAULT_PRIOR)
            v = float(s[L2I[cls]])
            if cls not in open_ev and v >= p["hi"] * a.scale:
                open_ev[cls] = [t_start, t_end, v]
            elif cls in open_ev:
                if v >= p["lo"] * a.scale:
                    open_ev[cls][1] = t_end
                    open_ev[cls][2] = max(open_ev[cls][2], v)
                else:
                    s0, s1, pk = open_ev.pop(cls)
                    if s1 - s0 >= p["min_dur"]:
                        print(f"  ALERT {cls:34s} {s0:6.1f}-{s1:6.1f}s  peak {pk:.2f}")

    for cls, (s0, s1, pk) in open_ev.items():
        p = PRIORS.get(cls, DEFAULT_PRIOR)
        if s1 - s0 >= p["min_dur"]:
            print(f"  ALERT {cls:34s} {s0:6.1f}-{s1:6.1f}s  peak {pk:.2f}")

    elapsed = time.time() - wall
    vid_sec = n_frames / a.fps
    print(f"\n{n_frames} sampled frames, {n_win} windows in {elapsed:.1f}s")
    print(f"encoder: {n_frames/max(enc_time,1e-6):.1f} frames/s  "
          f"({1000*enc_time/max(n_frames,1):.1f} ms/frame)")
    print(f"realtime factor: {vid_sec/max(elapsed,1e-6):.1f}x  "
          f"=> ~{int(vid_sec/max(elapsed,1e-6))} concurrent {a.fps}fps streams on this machine")
