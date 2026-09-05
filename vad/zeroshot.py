"""Zero-shot baseline: prompt-ensembled text embeddings vs frame embeddings.

No training. Gives a working end-to-end submission within ~20 minutes so the
rest of the day is spent improving a number rather than chasing a first one.
"""
from __future__ import annotations

import numpy as np

from .config import LABELS, PROMPTS


def class_matrix(embedder) -> np.ndarray:
    """[C, D] mean-of-prompts text embedding per class, L2 normalised."""
    rows = []
    for c in LABELS:
        t = embedder.encode_texts(PROMPTS[c])
        v = t.mean(0)
        rows.append(v / (np.linalg.norm(v) + 1e-8))
    return np.stack(rows).astype(np.float32)


def score_windows(feat: np.ndarray, spans, T: np.ndarray,
                  temp: float = 0.01) -> np.ndarray:
    """Mean-pool frames inside each window, then softmax over classes."""
    if len(feat) == 0:
        return np.zeros((0, T.shape[0]), np.float32)
    out = np.zeros((len(spans), T.shape[0]), np.float32)
    for w, (i0, i1, _, _) in enumerate(spans):
        v = feat[i0:i1].mean(0)
        v = v / (np.linalg.norm(v) + 1e-8)
        logits = (T @ v) / temp
        logits -= logits.max()
        e = np.exp(logits)
        out[w] = e / e.sum()
    return out
