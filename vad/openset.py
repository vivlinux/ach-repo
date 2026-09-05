"""Open-set handling: what happens on the thirteenth event type.

The brief says the list is not fixed but scores against twelve labels. A closed
softmax cannot express "this is wrong but I have no name for it", so it will
either force the event into the nearest label or call it normal. Both are bad
answers for an operator.

Two independent signals, either of which can raise `unknown_event`:
  1. distance to the training manifold (kNN in embedding space)
  2. classifier disagreement — high anomaly energy with no confident class
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import ANOMALY_LABELS, L2I, LABELS, NORMAL

UNKNOWN = "unknown_event"


class NoveltyBank:
    """Sampled training embeddings + per-class prototypes. Tiny and fast."""

    def __init__(self, protos: np.ndarray | None = None,
                 bank: np.ndarray | None = None, calib: dict | None = None):
        self.protos = protos          # [C, D]
        self.bank = bank              # [N, D] subsample of normal training frames
        self.calib = calib or {}

    # ------------------------------------------------------------ build
    @classmethod
    def fit(cls, feats_by_class: dict[str, list[np.ndarray]],
            bank_size: int = 4000, seed: int = 0):
        rng = np.random.default_rng(seed)
        D = next(iter(f[0].shape[-1] for f in feats_by_class.values() if len(f)))
        protos = np.zeros((len(LABELS), D), np.float32)
        for c, arrs in feats_by_class.items():
            if c not in L2I or not len(arrs):
                continue
            v = np.concatenate(arrs, 0).mean(0)
            protos[L2I[c]] = v / (np.linalg.norm(v) + 1e-8)
        normal = feats_by_class.get(NORMAL, [])
        allf = np.concatenate(normal, 0) if normal else np.concatenate(
            [a for v in feats_by_class.values() for a in v], 0)
        idx = rng.choice(len(allf), min(bank_size, len(allf)), replace=False)
        bank = allf[idx].astype(np.float32)
        bank /= np.linalg.norm(bank, axis=1, keepdims=True) + 1e-8
        return cls(protos, bank)

    def calibrate(self, novelty_scores: np.ndarray, q: float = 0.98):
        """Set the unknown threshold from held-out normal footage."""
        self.calib["novelty_thr"] = float(np.quantile(novelty_scores, q))
        return self.calib["novelty_thr"]

    def calibrate_dist(self, dist_scores: np.ndarray, q: float = 0.98):
        """Fixed 2026-09-05: this used to be the only calibration, and it was
        fed raw distance() values while is_unknown() compared against
        novelty()'s different [0,1] scale -- a straight units mismatch.
        is_ood() below is the one that actually uses this threshold."""
        self.calib["dist_thr"] = float(np.quantile(dist_scores, q))
        return self.calib["dist_thr"]

    # ------------------------------------------------------------ score
    def distance(self, feat: np.ndarray, k: int = 8) -> float:
        """Mean cosine distance to the k nearest training frames. 0 = familiar."""
        if self.bank is None or len(feat) == 0:
            return 0.0
        v = feat.mean(0)
        v = v / (np.linalg.norm(v) + 1e-8)
        sims = self.bank @ v
        k = min(k, len(sims))
        return float(1.0 - np.sort(sims)[-k:].mean())

    def novelty(self, feat: np.ndarray, scores: np.ndarray) -> float:
        """Combine manifold distance with classifier indecision. [0,1]."""
        d = self.distance(feat)
        an = scores[[L2I[c] for c in ANOMALY_LABELS]]
        energy = float(an.max())                       # something is off
        top2 = np.sort(an)[-2:]
        indecision = 1.0 - float(top2[-1] - top2[0]) if len(top2) == 2 else 0.0
        conf = float(scores[L2I[NORMAL]])
        # novel = far from training data, or clearly not normal yet no clear class
        return float(np.clip(0.6 * min(d / 0.35, 1.0)
                             + 0.4 * (energy * indecision * (1 - conf)), 0, 1))

    def is_unknown(self, nov: float, scores: np.ndarray,
                   class_thr: float = 0.55) -> bool:
        thr = self.calib.get("novelty_thr", 0.55)
        an = scores[[L2I[c] for c in ANOMALY_LABELS]]
        return nov >= thr and float(an.max()) < class_thr

    def is_ood(self, feat: np.ndarray) -> bool:
        """Pure manifold-distance abstention signal for cmd_predict, added
        2026-09-05: 'is this scene type in training at all', independent of
        what any classifier says about it. Uses calibrate_dist()'s threshold
        (raw distance() units), not novelty()'s mixed scale."""
        thr = self.calib.get("dist_thr", 0.35)
        return self.distance(feat) >= thr

    # ------------------------------------------------------------ io
    def save(self, path: Path):
        np.savez_compressed(path, protos=self.protos, bank=self.bank,
                            calib=json.dumps(self.calib))

    @classmethod
    def load(cls, path: Path):
        d = np.load(path, allow_pickle=False)
        return cls(d["protos"], d["bank"], json.loads(str(d["calib"])))
