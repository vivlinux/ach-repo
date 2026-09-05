"""Stage 1b: a small temporal head over frozen embeddings.

~1M params. Trains in minutes on an M2 because the encoder is frozen and its
outputs are cached. Videos that carry timestamps get per-window supervision;
videos with only a video-level label are trained with top-k MIL pooling, which
is the standard weakly-supervised VAD trick and stops noisy labels from
poisoning the normal class.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import LABELS, NORMAL, L2I, WIN


class TemporalHead(nn.Module):
    def __init__(self, dim: int, d_model: int = 256, n_layers: int = 2,
                 n_classes: int = len(LABELS), dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, d_model), nn.GELU())
        self.pos = nn.Parameter(torch.zeros(1, WIN, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead=4, dim_feedforward=d_model * 2, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model * 2, n_classes)

    def forward(self, x):                     # x: [B, WIN, D]
        h = self.enc(self.proj(x) + self.pos[:, :x.shape[1]])
        h = torch.cat([h.mean(1), h.max(1).values], -1)
        return self.out(self.drop(h))         # logits [B, C]


def _bce(logits, target, pos_weight=None, label_smoothing: float = 0.0):
    """label_smoothing pulls targets off the 0/1 rails (0->e/2, 1->1-e/2).
    Added 2026-09-05: the trained head's output was found saturated at
    0/1 on >2/3 of test windows (loitering sat at 1.000 on >10% of ALL
    test windows), which left per-class thresholds nothing to grip.
    Smoothing makes 0/1 targets structurally unreachable during training."""
    if label_smoothing > 0:
        target = target * (1 - label_smoothing) + label_smoothing / 2
    return nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight)


def train_head(samples, dim: int, epochs: int = 25, lr: float = 3e-4,
               device: str = "cpu", topk: int = 3, seed: int = 0,
               val_frac: float = 0.12, log=print,
               d_model: int = 256, n_layers: int = 2, dropout: float = 0.15,
               weight_decay: float = 0.02, label_smoothing: float = 0.0,
               feat_noise: float = 0.0, pos_weight_max: float = 8.0):
    """samples: list of dicts {x:[W,WIN,D], y:[W,C] or None, weak:[C] or None}.

    d_model/n_layers/dropout/weight_decay: capacity/regularisation knobs,
    all hardcoded before 2026-09-05. label_smoothing/feat_noise/
    pos_weight_max added the same day to fight the saturation found in the
    first trained checkpoint (out/head.pt): loitering (79 videos, ALL
    weakly-labelled MIL, no timestamps) fired at 1.000 on >10% of every
    test window, and pos_weight_max=8.0 (the old hardcoded clip) is a
    likely cause -- MIL's top-k windows get pushed hard toward 1 under a
    large positive weight. feat_noise perturbs L2-normalised embeddings by
    Gaussian noise at train time only, as a cheap stand-in for "this
    camera wasn't in training" since no camera/source id exists to hold
    out properly (see PROGRESS.md).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    idx = list(range(len(samples)))
    random.shuffle(idx)
    n_val = max(1, int(len(idx) * val_frac)) if len(idx) > 12 else 0
    val, tr = idx[:n_val], idx[n_val:]

    model = TemporalHead(dim, d_model=d_model, n_layers=n_layers, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * len(tr)), pct_start=0.2)

    # counter the heavy 'normal' skew
    pos = np.zeros(len(LABELS), np.float32) + 1.0
    for s in samples:
        if s["y"] is not None:
            pos += s["y"].sum(0)
        elif s["weak"] is not None:
            pos += s["weak"]
    pw = torch.tensor(np.clip(pos.sum() / (len(LABELS) * pos), 0.3, pos_weight_max),
                      dtype=torch.float32, device=device)

    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        random.shuffle(tr)
        tot = 0.0
        for i in tr:
            s = samples[i]
            x = torch.from_numpy(s["x"]).to(device)
            if x.shape[0] == 0:
                continue
            if x.shape[0] > 96:                       # cap very long videos
                sel = np.random.choice(x.shape[0], 96, replace=False)
                x = x[sel]
            if feat_noise > 0:
                x = x + torch.randn_like(x) * feat_noise
            logits = model(x)
            if s["y"] is not None:
                y = torch.from_numpy(s["y"]).to(device)
                if x.shape[0] != y.shape[0]:
                    y = y[:x.shape[0]] if y.shape[0] > x.shape[0] else y
                    logits = logits[:y.shape[0]]
                loss = _bce(logits, y, pw, label_smoothing)
            else:                                     # MIL: top-k windows only
                w = torch.from_numpy(s["weak"]).to(device)
                k = min(topk, logits.shape[0])
                bag = logits.topk(k, dim=0).values.mean(0, keepdim=True)
                loss = _bce(bag, w.unsqueeze(0), pw, label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
        msg = f"epoch {ep+1:>3}/{epochs}  train {tot/max(1,len(tr)):.4f}"
        if val:
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for i in val:
                    s = samples[i]
                    x = torch.from_numpy(s["x"]).to(device)
                    if x.shape[0] == 0:
                        continue
                    lg = model(x)
                    if s["y"] is not None:
                        y = torch.from_numpy(s["y"]).to(device)[:lg.shape[0]]
                        vl += float(_bce(lg[:y.shape[0]], y, pw))
                    else:
                        w = torch.from_numpy(s["weak"]).to(device)
                        k = min(topk, lg.shape[0])
                        vl += float(_bce(lg.topk(k, 0).values.mean(0, keepdim=True),
                                         w.unsqueeze(0), pw))
            vl /= max(1, len(val))
            msg += f"  val {vl:.4f}"
            if vl < best:
                best, best_state = vl, {k: v.detach().cpu().clone()
                                        for k, v in model.state_dict().items()}
        log(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def save(model: TemporalHead, dim: int, path: Path, meta: dict | None = None,
         d_model: int = 256, n_layers: int = 2, dropout: float = 0.15):
    torch.save({"state": model.state_dict(), "dim": dim,
                "d_model": d_model, "n_layers": n_layers, "dropout": dropout,
                "labels": LABELS, "meta": meta or {}}, path)


def load(path: Path, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    # older checkpoints (e.g. out/head.pt, saved before 2026-09-05) have no
    # architecture keys -- default to what they were actually trained with
    m = TemporalHead(ck["dim"], d_model=ck.get("d_model", 256),
                      n_layers=ck.get("n_layers", 2),
                      dropout=ck.get("dropout", 0.15)).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


@torch.no_grad()
def predict(model, x: np.ndarray, device: str = "cpu") -> np.ndarray:
    """[W,WIN,D] -> per-window sigmoid scores [W,C]."""
    if len(x) == 0:
        return np.zeros((0, len(LABELS)), np.float32)
    out = []
    for i in range(0, len(x), 64):
        t = torch.from_numpy(x[i:i + 64]).to(device)
        out.append(torch.sigmoid(model(t)).cpu().numpy())
    return np.concatenate(out, 0)
