"""Causal streaming engine — the deployment-shaped path.

Everything here is online: no frame is ever read ahead of the alert timestamp,
which is the property clip-level evaluation quietly breaks. It also models the
things that go wrong on a real uplink — dropped frames, link outages — because
a detector that desynchronizes under packet loss is not deployable.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import head as H
from . import postprocess as PP
from .adaptive import AdaptiveSampler
from .alerts import Alert, AlertManager
from .config import ANOMALY_LABELS, DEFAULT_PRIOR, L2I, PRIORS, WIN
from .frames import iter_frames, video_meta
from .geo import Telemetry, interpolate, pixel_to_latlon
from .motion import EgoMotion, WorldDwell, objectness, stopped_object_score
from .openset import UNKNOWN, NoveltyBank

# classes whose definition depends on persistence at a fixed ground location
DWELL_CLASSES = {"stalled_or_broken_down_vehicle", "vehicle_blocking_traffic",
                 "loitering_or_suspicious_presence", "road_spill_or_debris"}


@dataclass
class LinkModel:
    """Simulates an LTE/5G uplink: random frame loss plus occasional outages."""
    drop_rate: float = 0.0
    outage_every_s: float = 0.0
    outage_len_s: float = 0.0
    seed: int = 0
    _rng: np.random.Generator = field(default=None, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def state(self, t: float) -> str:
        if self.outage_every_s > 0 and (t % self.outage_every_s) < self.outage_len_s:
            return "outage"
        if self.drop_rate > 0 and self._rng.random() < self.drop_rate:
            return "dropped"
        return "ok"


@dataclass
class StreamConfig:
    drone_id: str = "drone-1"
    scale: float = 1.0
    adaptive: bool = True
    ego_motion: bool = True          # motion features into the head — validated, keep on
    dwell_gate: bool = False         # gate dwell classes on the detector-free cue —
                                     # OFF by default: it does not discriminate without
                                     # an objectness source. See scripts/validate_motion.py
    openset: bool = True
    device: str = "cpu"
    store_and_forward: bool = True
    max_buffer: int = 500


class StreamProcessor:
    """One drone feed. Instantiate one per stream; they share nothing."""

    def __init__(self, model, embedder, cfg: StreamConfig,
                 manager: AlertManager, bank: NoveltyBank | None = None,
                 telemetry: list[Telemetry] | None = None,
                 link: LinkModel | None = None):
        self.model, self.emb, self.cfg = model, embedder, cfg
        self.mgr, self.bank, self.tel = manager, bank, telemetry
        self.link = link or LinkModel()
        self.sampler = AdaptiveSampler()
        self.ego = EgoMotion() if cfg.ego_motion else None
        self.dwell = WorldDwell() if (cfg.ego_motion and cfg.dwell_gate) else None
        self.buf: deque = deque(maxlen=WIN)
        self.times: deque = deque(maxlen=WIN)
        self.open: dict[str, list] = {}
        self.forward_queue: list[Alert] = []
        self.stats = dict(frames=0, encoded=0, dropped=0, outage_s=0.0,
                          windows=0, enc_ms=0.0, head_ms=0.0, motion_ms=0.0)

    # ------------------------------------------------------------------
    def _geo(self, t: float):
        if not self.tel:
            return None, None
        tel = interpolate(self.tel, t)
        if tel is None:
            return None, None
        if self.dwell:
            k, _ = self.dwell.hottest()
            if k is not None:
                return pixel_to_latlon(112, 112, 224, 224, tel)   # centre fallback
        return tel.lat, tel.lon

    def _fire(self, cls: str, t0: float, t1: float, score: float,
              now: float, novelty: float = 0.0):
        lat, lon = self._geo(t1)
        a = Alert(self.cfg.drone_id, cls, t0, t1, score, detected_at=now,
                  lat=lat, lon=lon, novelty=novelty)
        if self.link.state(now) == "outage" and self.cfg.store_and_forward:
            if len(self.forward_queue) < self.cfg.max_buffer:
                self.forward_queue.append(a)      # held locally, replayed on reconnect
            return None
        self._flush(now)
        return self.mgr.submit(a)

    def _flush(self, now: float):
        if not self.forward_queue or self.link.state(now) == "outage":
            return
        for a in self.forward_queue:
            self.mgr.submit(a)
        self.forward_queue.clear()

    # ------------------------------------------------------------------
    def run(self, video: Path, log=print) -> dict:
        _, dur = video_meta(Path(video))
        decode_fps = 10.0                       # what the link delivers
        wall0 = time.time()

        for t, img in iter_frames(Path(video), decode_fps):
            self.stats["frames"] += 1
            st = self.link.state(t)
            if st != "ok":
                self.stats["dropped"] += 1
                if st == "outage":
                    self.stats["outage_s"] += 1.0 / decode_fps
                continue                        # a lost frame must not desync anything

            ms = None
            if self.ego is not None:
                m0 = time.time()
                ms = self.ego.update(img)
                self.stats["motion_ms"] += (time.time() - m0) * 1000

            if self.cfg.adaptive and not self.sampler.should_process(t):
                continue

            e0 = time.time()
            f = self.emb.encode_images([img])[0]
            self.stats["enc_ms"] += (time.time() - e0) * 1000
            self.stats["encoded"] += 1

            if ms is not None:
                f = np.concatenate([f, ms.feature()])
            self.buf.append(f)
            self.times.append(t)
            if len(self.buf) < WIN:
                continue

            h0 = time.time()
            s = H.predict(self.model, np.stack(self.buf)[None], self.cfg.device)[0]
            self.stats["head_ms"] += (time.time() - h0) * 1000
            self.stats["windows"] += 1

            # ego-motion-compensated dwell, in ground coordinates
            dwell_s = 0.0
            if ms is not None and self.dwell is not None:
                dwell_s = self.dwell.update(ms, stopped_object_score(ms, objectness(img)), t)

            nov = self.bank.novelty(np.stack(self.buf)[:, :self.bank.bank.shape[1]], s) \
                if (self.cfg.openset and self.bank is not None) else 0.0

            an = s[[L2I[c] for c in ANOMALY_LABELS]]
            self.sampler.observe(float(an.max()), t)

            t0, t1 = self.times[0], self.times[-1]
            for cls in ANOMALY_LABELS:
                p = PRIORS.get(cls, DEFAULT_PRIOR)
                v = float(s[L2I[cls]])
                # a dwell-defined class must also be persistent on the ground
                gate = True
                if self.cfg.dwell_gate and cls in DWELL_CLASSES and self.dwell is not None:
                    gate = dwell_s >= p["min_dur"]
                if cls not in self.open and v >= p["hi"] * self.cfg.scale and gate:
                    self.open[cls] = [t0, t1, v]
                elif cls in self.open:
                    if v >= p["lo"] * self.cfg.scale:
                        self.open[cls][1] = t1
                        self.open[cls][2] = max(self.open[cls][2], v)
                    else:
                        s0, s1, pk = self.open.pop(cls)
                        if s1 - s0 >= p["min_dur"]:
                            a = self._fire(cls, s0, s1, pk, now=t1)
                            if a:
                                log(f"  [{t1:7.1f}s] ALERT {a.id} {cls:32s} "
                                    f"{s0:.1f}-{s1:.1f}s peak {pk:.2f} "
                                    f"lat {a.latency:.1f}s prio {a.priority:.2f}")
            if self.cfg.openset and self.bank is not None and \
                    self.bank.is_unknown(nov, s) and UNKNOWN not in self.open:
                a = self._fire(UNKNOWN, t0, t1, float(an.max()), now=t1, novelty=nov)
                if a:
                    log(f"  [{t1:7.1f}s] ALERT {a.id} {UNKNOWN:32s} novelty {nov:.2f} "
                        f"— outside the label set, needs a human")
            self._flush(t)

        for cls, (s0, s1, pk) in list(self.open.items()):
            p = PRIORS.get(cls, DEFAULT_PRIOR)
            if s1 - s0 >= p["min_dur"]:
                self._fire(cls, s0, s1, pk, now=s1)
        self._flush(1e9)

        wall = time.time() - wall0
        n = max(self.stats["encoded"], 1)
        return {
            **self.stats,
            "video_seconds": round(dur, 1),
            "wall_seconds": round(wall, 1),
            "realtime_factor": round(dur / max(wall, 1e-6), 2),
            "ms_per_encoded_frame": round(self.stats["enc_ms"] / n, 2),
            "ms_per_window_head": round(self.stats["head_ms"] / max(self.stats["windows"], 1), 2),
            "ms_per_frame_motion": round(self.stats["motion_ms"] / max(self.stats["frames"], 1), 2),
            "adaptive": self.sampler.report(dur) if self.cfg.adaptive else {},
        }
