"""Ego-motion compensation and stabilized dwell tracking.

The gap this closes: on a moving drone nothing is stationary in pixel space, so
"stopped for 30 seconds" is meaningless until camera motion is removed. This
module estimates the frame-to-frame camera transform, subtracts it, and
accumulates evidence in a stabilized reference frame instead of image
coordinates.

The stopped-vehicle cue falls out of that directly: a cell with near-zero
residual motion surrounded by cells with high residual motion is an object that
is not moving while its neighbours are. No object detector required.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

GRID = 12          # residual-flow grid, GRID x GRID cells
MIN_PTS = 30


@dataclass
class MotionState:
    H: np.ndarray                 # 3x3 frame->previous-frame transform
    H_cum: np.ndarray             # 3x3 frame->reference-frame transform
    inlier_ratio: float
    ego_translation: float        # px/frame, magnitude
    ego_rotation: float           # radians/frame
    ego_scale: float              # >1 = descending / zooming in
    residual: np.ndarray          # [GRID, GRID] mean unexplained flow magnitude
    coverage: np.ndarray | None = None   # [GRID, GRID] tracked points per cell
    valid: bool = True

    def feature(self) -> np.ndarray:
        """Compact 8-dim descriptor to concatenate to the visual embedding."""
        r = self.residual
        return np.array([
            self.ego_translation / 20.0,
            self.ego_rotation * 10.0,
            (self.ego_scale - 1.0) * 10.0,
            self.inlier_ratio,
            float(r.mean()) / 5.0,
            float(r.std()) / 5.0,
            float(np.percentile(r, 90)) / 5.0,
            float((r < 0.3 * max(r.mean(), 1e-3)).mean()),   # fraction of static cells
        ], np.float32)


class EgoMotion:
    """Sparse LK flow + robust partial-affine fit. ~2-4 ms/frame at 224px."""

    def __init__(self, grid: int = GRID, max_pts: int = 250):
        self.grid = grid
        self.max_pts = max_pts
        self.prev: np.ndarray | None = None
        self.H_cum = np.eye(3, dtype=np.float32)
        self._hist: list[float] = []

    def reset(self):
        self.prev = None
        self.H_cum = np.eye(3, dtype=np.float32)

    def update(self, rgb: np.ndarray, expected_px: float | None = None,
               tol: float = 2.5) -> MotionState:
        """`expected_px` is the frame-to-frame image translation predicted from
        telemetry (see geo.expected_image_translation). Supply it whenever you
        have it.

        Purely visual ego-motion is not robust on low-texture ground: when the
        background carries little trackable structure, the vehicles do, and the
        robust fit converges on traffic motion rather than camera motion. The
        bundled validation script reproduces this — the estimate pins to the
        vehicle velocity regardless of the true pan, and neither the inlier
        ratio nor a spatial-spread test rejects it, because the traffic is
        itself spread across the frame.

        A drone always knows its own velocity and altitude. Feeding that in
        turns an ill-posed estimation into a verification, which is both more
        robust and cheaper than any purely visual fix.
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.prev is None:
            self.prev = gray
            return MotionState(np.eye(3, dtype=np.float32), self.H_cum.copy(), 0.0,
                               0.0, 0.0, 1.0, np.zeros((self.grid, self.grid), np.float32),
                               np.zeros((self.grid, self.grid), np.float32), valid=False)

        p0 = self._seed(self.prev)
        if p0 is None or len(p0) < MIN_PTS:
            self.prev = gray
            return MotionState(np.eye(3, dtype=np.float32), self.H_cum.copy(), 0.0,
                               0.0, 0.0, 1.0, np.zeros((self.grid, self.grid), np.float32),
                               np.zeros((self.grid, self.grid), np.float32), valid=False)

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev, gray, p0, None,
                                               winSize=(15, 15), maxLevel=2)
        st = st.reshape(-1).astype(bool)
        if err is not None and st.sum() > MIN_PTS * 2:
            # Relative, not absolute: an absolute cut removes seeds on smooth
            # tarmac, leaving only points on the vehicles — RANSAC then fits the
            # traffic instead of the camera. Drop the worst tail only.
            e = err.reshape(-1)
            st &= e <= np.percentile(e[st], 85)
        a, b = p0.reshape(-1, 2)[st], p1.reshape(-1, 2)[st]
        if len(a) < MIN_PTS:
            self.prev = gray
            return MotionState(np.eye(3, dtype=np.float32), self.H_cum.copy(), 0.0,
                               0.0, 0.0, 1.0, np.zeros((self.grid, self.grid), np.float32),
                               np.zeros((self.grid, self.grid), np.float32), valid=False)

        M, inl = cv2.estimateAffinePartial2D(b, a, method=cv2.RANSAC,
                                             ransacReprojThreshold=2.0,
                                             maxIters=2000, confidence=0.995)
        if M is None:
            self.prev = gray
            return MotionState(np.eye(3, dtype=np.float32), self.H_cum.copy(), 0.0,
                               0.0, 0.0, 1.0, np.zeros((self.grid, self.grid), np.float32),
                               np.zeros((self.grid, self.grid), np.float32), valid=False)

        H = np.vstack([M, [0, 0, 1]]).astype(np.float32)
        inlier_ratio = float(inl.mean()) if inl is not None else 0.0
        scale = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
        rot = float(np.arctan2(M[1, 0], M[0, 0]))
        trans = float(np.hypot(M[0, 2], M[1, 2]))

        # residual = observed flow minus the flow the camera model predicts
        pred = (np.hstack([b, np.ones((len(b), 1), np.float32)]) @ H.T)[:, :2]
        res = np.linalg.norm(pred - a, axis=1)
        grid, cover = self._bin(b, res, gray.shape)

        # Temporal sanity: camera motion is continuous. An estimate that jumps
        # far from the recent median usually means the fit locked onto a moving
        # object, so report it invalid instead of corrupting the world frame.
        # Spatial spread test. The camera transform is the motion of the whole
        # ground plane, so its inliers should cover the frame. A fit whose
        # inliers cluster in a few cells has locked onto the traffic instead —
        # the dominant failure mode on smooth surfaces where vehicles carry
        # most of the trackable texture.
        spread = 1.0
        if inl is not None:
            m = inl.reshape(-1).astype(bool)
            if m.sum() >= MIN_PTS:
                pin = b[m]
                cy = np.clip((pin[:, 1] / gray.shape[0] * self.grid).astype(int), 0, self.grid - 1)
                cx = np.clip((pin[:, 0] / gray.shape[1] * self.grid).astype(int), 0, self.grid - 1)
                spread = len(set(zip(cy.tolist(), cx.tolist()))) / float(self.grid ** 2)
        ok = inlier_ratio >= 0.35 and spread >= 0.25
        if expected_px is not None:
            ok = ok and abs(trans - expected_px) <= tol * max(expected_px, 0.5)
        if len(self._hist) >= 5:
            med = float(np.median(self._hist[-10:]))
            if med > 0.5 and (trans > 4 * med or trans < 0.2 * med):
                ok = False
        if ok:
            self._hist.append(trans)
            self.H_cum = self.H_cum @ H
        self.prev = gray
        return MotionState(H, self.H_cum.copy(), inlier_ratio, trans, rot, scale,
                           grid, cover, valid=ok)

    def _seed(self, gray: np.ndarray) -> np.ndarray | None:
        """Uniform grid + strong corners.

        Pure goodFeaturesToTrack concentrates on the most textured regions,
        which on aerial video is background. Independent motion then goes
        unsampled and the residual grid stays empty exactly where it matters,
        so the grid seeding is not cosmetic — without it the stopped-object cue
        never fires.
        """
        h, w = gray.shape
        step = max(6, int(min(h, w) / (self.grid * 2)))
        ys, xs = np.mgrid[step:h - step:step, step:w - step:step]
        grid_pts = np.stack([xs.ravel(), ys.ravel()], -1).astype(np.float32)
        corners = cv2.goodFeaturesToTrack(gray, self.max_pts // 2, 0.01, 7)
        pts = grid_pts if corners is None else np.vstack([grid_pts, corners.reshape(-1, 2)])
        if len(pts) > self.max_pts * 2:
            sel = np.linspace(0, len(pts) - 1, self.max_pts * 2).astype(int)
            pts = pts[sel]
        return pts.reshape(-1, 1, 2).astype(np.float32)

    def _bin(self, pts: np.ndarray, vals: np.ndarray, shape) -> np.ndarray:
        h, w = shape
        g = np.zeros((self.grid, self.grid), np.float32)
        n = np.zeros((self.grid, self.grid), np.float32)
        gy = np.clip((pts[:, 1] / h * self.grid).astype(int), 0, self.grid - 1)
        gx = np.clip((pts[:, 0] / w * self.grid).astype(int), 0, self.grid - 1)
        np.add.at(g, (gy, gx), vals)
        np.add.at(n, (gy, gx), 1.0)
        return g / np.maximum(n, 1.0), n


def objectness(rgb: np.ndarray, grid: int = GRID) -> np.ndarray:
    """[grid,grid] cheap "is there a thing here" score, ~0.5 ms.

    Local gradient energy relative to the frame's own background level. Tarmac,
    water and grass are smooth; vehicles, debris and people are not. This is a
    weak prior, not a detector — see the note on stopped_object_score.
    """
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    e = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))
    h, w = e.shape
    cell = cv2.resize(e, (grid, grid), interpolation=cv2.INTER_AREA)
    base = float(np.percentile(cell, 40)) + 1e-3        # the smooth-surface level
    return np.clip(cell / base - 1.0, 0.0, 3.0) / 3.0


def stopped_object_score(ms: MotionState, obj: np.ndarray | None = None,
                         min_scene_motion: float = 0.4, min_points: int = 2) -> np.ndarray:
    """[GRID,GRID]: something is *there*, it is still, and its neighbours are not.

    Three conjuncts, and all three are load-bearing:

      quiet      residual flow at background level once ego-motion is removed
      neighbours independent motion in the surrounding cells — a stalled vehicle
                 sits inside a traffic stream, empty shoulder usually does not
      objectness texture above the smooth-surface level — this is what separates
                 a stopped car from bare tarmac, and without it the cue fires
                 identically on both (verified: it does)

    Even so, treat this as a prior rather than an answer. The correct version
    feeds `WorldDwell` from a real detector — a nano-scale model runs in ~3 ms
    on this hardware. `WorldDwell.update_from_detections` is the hook for that.
    """
    r = ms.residual
    cov = ms.coverage if ms.coverage is not None else np.ones_like(r)
    seen = cov >= min_points
    if not ms.valid or ms.inlier_ratio < 0.5 or seen.sum() < 8:
        return np.zeros_like(r)

    obs = r[seen]
    med = float(np.median(obs))
    spread = float(np.median(np.abs(obs - med))) * 1.4826
    thr = max(med + 3.0 * max(spread, 1e-3), min_scene_motion)
    moving = (r > thr) & seen
    if moving.sum() < 2:
        return np.zeros_like(r)

    ref = float(r[moving].mean())
    quiet = 1.0 - np.clip((r - med) / max(ref - med, 1e-6), 0.0, 1.0)

    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0.0
    neigh = cv2.filter2D(moving.astype(np.float32), -1, k,
                         borderType=cv2.BORDER_CONSTANT) / 8.0

    score = quiet * np.clip(neigh / 0.25, 0.0, 1.0) * seen * (~moving)
    if obj is not None:
        score = score * np.clip(obj / 0.35, 0.0, 1.0)     # nothing there -> no alert
    return score.astype(np.float32)


class WorldDwell:
    """Accumulates per-cell evidence in the stabilized reference frame.

    Cells are projected through the cumulative camera transform and quantized,
    so the same patch of ground keeps the same key as the drone moves. Dwell
    time is then measured on ground, not on pixels.
    """

    def __init__(self, frame_size: int = 224, grid: int = GRID,
                 cell_px: float = 24.0, decay: float = 0.85):
        self.fs, self.grid, self.cell, self.decay = frame_size, grid, cell_px, decay
        self.first_seen: dict[tuple[int, int], float] = {}
        self.last_seen: dict[tuple[int, int], float] = {}
        self.energy: dict[tuple[int, int], float] = {}

    def _keys(self, H_cum: np.ndarray) -> list[tuple[int, int]]:
        step = self.fs / self.grid
        c = np.stack(np.meshgrid(
            (np.arange(self.grid) + 0.5) * step,
            (np.arange(self.grid) + 0.5) * step, indexing="xy"), -1).reshape(-1, 2)
        w = (np.hstack([c, np.ones((len(c), 1))]) @ H_cum.T)[:, :2]
        return [(int(round(x / self.cell)), int(round(y / self.cell))) for x, y in w]

    def update(self, ms: MotionState, cell_scores: np.ndarray, t: float,
               thr: float = 0.5) -> float:
        """Returns the longest dwell (seconds) currently active. O(GRID^2)."""
        keys = self._keys(ms.H_cum)
        flat = cell_scores.reshape(-1)
        for k, s in zip(keys, flat):
            e = self.energy.get(k, 0.0) * self.decay + float(s)
            self.energy[k] = e
            if s >= thr:
                self.first_seen.setdefault(k, t)
                self.last_seen[k] = t
        best = 0.0
        stale = []
        for k, t0 in self.first_seen.items():
            if t - self.last_seen.get(k, t0) > 3.0:       # lost for 3s -> drop
                stale.append(k)
                continue
            best = max(best, self.last_seen[k] - t0)
        for k in stale:
            self.first_seen.pop(k, None)
            self.last_seen.pop(k, None)
        return best

    def update_from_detections(self, ms: MotionState, boxes, t: float,
                               moving_thr: float = 0.5) -> float:
        """Preferred path: feed real detections (x,y,w,h in image px).

        A box whose residual flow is at background level is a stationary
        object; its world key is stable under camera motion, so dwell time
        accumulates correctly however the drone moves.
        """
        cells = np.zeros((self.grid, self.grid), np.float32)
        step = self.fs / self.grid
        for (x, y, bw, bh) in boxes:
            cx, cy = x + bw / 2.0, y + bh / 2.0
            gx = int(np.clip(cx / step, 0, self.grid - 1))
            gy = int(np.clip(cy / step, 0, self.grid - 1))
            if ms.residual[gy, gx] < moving_thr:
                cells[gy, gx] = 1.0
        return self.update(ms, cells, t)

    def hottest(self) -> tuple[tuple[int, int] | None, float]:
        if not self.energy:
            return None, 0.0
        k = max(self.energy, key=self.energy.get)
        return k, self.energy[k]
