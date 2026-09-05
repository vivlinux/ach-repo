"""Where an alert's latitude and longitude come from.

Flat-earth pinhole projection onto the ground plane using drone telemetry. Not
survey-grade — it ignores terrain relief and lens distortion — but it turns an
alert from "somewhere in this video" into a map pin, which is the difference
between a demo and something an operator can dispatch against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

R_EARTH = 6_378_137.0


@dataclass
class Telemetry:
    t: float                # seconds into the flight
    lat: float
    lon: float
    alt_agl: float          # metres above ground
    yaw_deg: float = 0.0    # 0 = north, clockwise
    pitch_deg: float = -90.0  # -90 = straight down (nadir)
    hfov_deg: float = 80.0

    @staticmethod
    def nadir(lat: float, lon: float, alt: float, t: float = 0.0) -> "Telemetry":
        return Telemetry(t=t, lat=lat, lon=lon, alt_agl=alt)


def load_telemetry(path: Path) -> list[Telemetry]:
    """CSV with t/lat/lon/alt columns; missing optional columns get defaults."""
    import pandas as pd
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}

    def g(row, *names, default=None):
        for n in names:
            if n in cols:
                try:
                    return float(row[cols[n]])
                except (TypeError, ValueError):
                    pass
        return default

    out = []
    for _, r in df.iterrows():
        out.append(Telemetry(
            t=g(r, "t", "time", "timestamp", "time_sec", default=0.0),
            lat=g(r, "lat", "latitude", default=0.0),
            lon=g(r, "lon", "lng", "longitude", default=0.0),
            alt_agl=g(r, "alt_agl", "alt", "altitude", "height", default=100.0),
            yaw_deg=g(r, "yaw", "heading", "yaw_deg", default=0.0),
            pitch_deg=g(r, "pitch", "gimbal_pitch", default=-90.0),
            hfov_deg=g(r, "hfov", "fov", default=80.0)))
    return sorted(out, key=lambda x: x.t)


def interpolate(track: list[Telemetry], t: float) -> Telemetry | None:
    if not track:
        return None
    if t <= track[0].t:
        return track[0]
    if t >= track[-1].t:
        return track[-1]
    for a, b in zip(track, track[1:]):
        if a.t <= t <= b.t:
            f = (t - a.t) / max(b.t - a.t, 1e-6)
            return Telemetry(t, a.lat + f * (b.lat - a.lat), a.lon + f * (b.lon - a.lon),
                             a.alt_agl + f * (b.alt_agl - a.alt_agl),
                             a.yaw_deg, a.pitch_deg, a.hfov_deg)
    return track[-1]


def ground_sample_distance(tel: Telemetry, width_px: int) -> float:
    """Metres per pixel at the image centre. Also the scale-invariance fix."""
    footprint = 2.0 * tel.alt_agl * math.tan(math.radians(tel.hfov_deg / 2.0))
    return footprint / max(width_px, 1)


def pixel_to_latlon(u: float, v: float, w: int, h: int, tel: Telemetry):
    """Image point -> ground lat/lon. Nadir-accurate, degrades with oblique pitch."""
    gsd = ground_sample_distance(tel, w)
    dx = (u - w / 2.0) * gsd                       # right of centre, metres
    dy = (v - h / 2.0) * gsd                       # below centre in image = forward
    tilt = math.radians(90.0 + tel.pitch_deg)      # 0 at nadir
    dy = dy / max(math.cos(tilt), 0.2)             # crude oblique stretch

    yaw = math.radians(tel.yaw_deg)
    north = dy * math.cos(yaw) - dx * math.sin(yaw)
    east = dy * math.sin(yaw) + dx * math.cos(yaw)

    dlat = north / R_EARTH * 180.0 / math.pi
    dlon = east / (R_EARTH * math.cos(math.radians(tel.lat))) * 180.0 / math.pi
    return tel.lat + dlat, tel.lon + dlon


def haversine(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(a))


def expected_image_translation(prev: Telemetry, cur: Telemetry, width_px: int) -> float:
    """Frame-to-frame image translation in pixels predicted from telemetry.

    Ground speed divided by ground sample distance. Pass this to
    EgoMotion.update(expected_px=...) so a visual fit that has locked onto
    moving traffic is rejected rather than propagated into world coordinates.
    """
    dt = max(cur.t - prev.t, 1e-6)
    metres = haversine(prev.lat, prev.lon, cur.lat, cur.lon)
    gsd = ground_sample_distance(cur, width_px)
    return (metres / dt) * dt / max(gsd, 1e-9)
