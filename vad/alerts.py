"""Operator-facing alerting: deduplication, triage ranking, alert economics.

Detection quality on a curated clip set overstates real precision badly,
because the base rate of anomalies in live footage is tiny. The number that
matters to an operator is false alerts per drone-hour, and the tolerable budget
is roughly 1-2 per hour across the whole fleet.

Deduplication is the other half. One congestion event seen by an orbiting drone
for four minutes is one alert, not forty; the same junction seen by two drones
is still one alert.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import PRIORS, DEFAULT_PRIOR
from .geo import haversine

# rough operator cost of getting each class wrong, used for triage ranking only
SEVERITY = {
    "traffic_accident": 1.0, "fire": 1.0, "fighting_or_violence": 0.95,
    "smoke": 0.8, "waterlogging_or_flood": 0.7, "wrong_way_driving": 0.7,
    "road_spill_or_debris": 0.6, "stalled_or_broken_down_vehicle": 0.55,
    "vehicle_blocking_traffic": 0.5, "traffic_congestion": 0.35,
    "loitering_or_suspicious_presence": 0.4, "unknown_event": 0.5,
}


@dataclass
class Alert:
    drone_id: str
    cls: str
    t_start: float
    t_end: float
    score: float
    detected_at: float                  # stream time the alert actually fired
    lat: float | None = None
    lon: float | None = None
    novelty: float = 0.0
    verified: bool | None = None
    evidence: str | None = None
    dedup_count: int = 1
    id: str = ""

    @property
    def latency(self) -> float:
        """Seconds from event onset to alert. The operator-facing latency."""
        return max(0.0, self.detected_at - self.t_start)

    @property
    def priority(self) -> float:
        sev = SEVERITY.get(self.cls, 0.5)
        conf = self.score if self.verified is not False else self.score * 0.4
        return round(sev * conf * (1.0 + 0.15 * min(self.dedup_count, 5)), 4)


class AlertManager:
    """Merges alerts across time and space before they reach a human."""

    def __init__(self, geo_radius_m: float = 120.0, time_window_s: float = 180.0,
                 min_priority: float = 0.0):
        self.geo_radius = geo_radius_m
        self.window = time_window_s
        self.min_priority = min_priority
        self.active: list[Alert] = []
        self.emitted: list[Alert] = []
        self.suppressed = 0
        self._n = 0

    def _same(self, a: Alert, b: Alert) -> bool:
        if a.cls != b.cls:
            return False
        if abs(b.t_start - a.t_end) > self.window and abs(b.detected_at - a.detected_at) > self.window:
            return False
        if a.lat is not None and b.lat is not None:
            return haversine(a.lat, a.lon, b.lat, b.lon) <= self.geo_radius
        return a.drone_id == b.drone_id      # no geo: only merge within one feed

    def submit(self, alert: Alert) -> Alert | None:
        """Returns the alert if it is new to the operator, else None (merged)."""
        for ex in self.active:
            if self._same(ex, alert):
                ex.t_end = max(ex.t_end, alert.t_end)
                ex.score = max(ex.score, alert.score)
                ex.dedup_count += 1
                if alert.drone_id != ex.drone_id:
                    ex.drone_id = f"{ex.drone_id}+{alert.drone_id}"
                self.suppressed += 1
                return None
        self._n += 1
        alert.id = f"A{self._n:05d}"
        if alert.priority < self.min_priority:
            self.suppressed += 1
            return None
        self.active.append(alert)
        self.emitted.append(alert)
        # retire alerts that can no longer absorb anything
        self.active = [a for a in self.active
                       if alert.detected_at - a.t_end <= self.window * 2]
        return alert

    def queue(self) -> list[Alert]:
        return sorted(self.emitted, key=lambda a: -a.priority)

    def metrics(self, drone_hours: float, true_ids: set[str] | None = None) -> dict:
        n = len(self.emitted)
        m = {
            "alerts_emitted": n,
            "alerts_suppressed_by_dedup": self.suppressed,
            "dedup_ratio": round(self.suppressed / max(n + self.suppressed, 1), 3),
            "drone_hours": round(drone_hours, 3),
            "alerts_per_drone_hour": round(n / max(drone_hours, 1e-6), 2),
            "mean_latency_s": round(sum(a.latency for a in self.emitted) / max(n, 1), 2),
            "p95_latency_s": round(sorted(a.latency for a in self.emitted)[int(0.95 * n)]
                                   if n else 0.0, 2),
        }
        if true_ids is not None:
            fp = [a for a in self.emitted if a.id not in true_ids]
            m["false_alerts_per_drone_hour"] = round(len(fp) / max(drone_hours, 1e-6), 2)
            m["precision"] = round(1 - len(fp) / max(n, 1), 3)
        return m

    def save(self, path: Path):
        path = Path(path)
        path.write_text(json.dumps([asdict(a) | {"priority": a.priority,
                                                 "latency_s": round(a.latency, 2)}
                                    for a in self.queue()], indent=2))
        return path
