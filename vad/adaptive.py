"""Adaptive compute governor.

The largest cost lever in the whole system and the one nobody implements.
Ordinary footage is the overwhelming majority of what a drone records, so
running the encoder at a fixed high frame rate spends almost all of its budget
confirming that nothing is happening.

Three tiers:
  IDLE   1 fps  — cheap encoder only, watching for any deviation
  WATCH  4 fps  — something crossed the low threshold
  BURST 10 fps  — an alert is forming; sample densely for localization

Every transition is hysteretic with a cooldown, so a noisy score cannot
oscillate the sampler and blow the budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Tier:
    name: str
    fps: float
    enter: float          # suspicion score that promotes into this tier
    hold: float           # score below which we demote (after cooldown)


DEFAULT_TIERS = [
    Tier("idle", 1.0, 0.00, 0.00),
    Tier("watch", 4.0, 0.30, 0.20),
    Tier("burst", 10.0, 0.55, 0.40),
]


@dataclass
class AdaptiveSampler:
    tiers: list[Tier] = field(default_factory=lambda: list(DEFAULT_TIERS))
    cooldown_s: float = 4.0        # minimum time in a tier before demotion
    max_fps: float = 10.0
    level: int = 0
    _since: float = 0.0
    _next_t: float = -1.0
    # accounting
    frames_seen: int = 0
    frames_processed: int = 0
    time_in_tier: dict = field(default_factory=dict)
    _last_t: float | None = None

    @property
    def fps(self) -> float:
        return self.tiers[self.level].fps

    @property
    def tier(self) -> str:
        return self.tiers[self.level].name

    def should_process(self, t: float) -> bool:
        """Call for every decoded frame; True means run the encoder on it."""
        self.frames_seen += 1
        if self._last_t is not None:
            self.time_in_tier[self.tier] = self.time_in_tier.get(self.tier, 0.0) + (t - self._last_t)
        self._last_t = t
        if self._next_t < 0 or t >= self._next_t:
            self._next_t = t + 1.0 / self.fps
            self.frames_processed += 1
            return True
        return False

    def observe(self, suspicion: float, t: float):
        """Feed the max anomaly score of the latest window."""
        cur = self.level
        # promote immediately
        for i in range(len(self.tiers) - 1, cur, -1):
            if suspicion >= self.tiers[i].enter:
                self.level = i
                self._since = t
                self._next_t = t          # resample right away at the new rate
                return
        # demote only after the cooldown and below the hold threshold
        if cur > 0 and suspicion < self.tiers[cur].hold and (t - self._since) >= self.cooldown_s:
            self.level = cur - 1
            self._since = t

    def report(self, video_seconds: float) -> dict:
        fixed = video_seconds * self.max_fps
        used = self.frames_processed
        return {
            "frames_decoded": self.frames_seen,
            "frames_encoded": used,
            "encoder_calls_vs_fixed_10fps": round(used / max(fixed, 1e-6), 4),
            "compute_saved_pct": round(100 * (1 - used / max(fixed, 1e-6)), 1),
            "time_in_tier_s": {k: round(v, 1) for k, v in self.time_in_tier.items()},
        }
