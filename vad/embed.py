"""Stage 1: frozen image encoder.

Runs on Apple MPS. Frame embeddings are cached to .npz per video so every
later step (training, threshold sweeps, re-running inference) is seconds,
not minutes. This is the single biggest time-saver on a laptop.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import CACHE, EMBED_MODELS, FPS
from .frames import iter_frames


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    def __init__(self, model_id: str | None = None, device: str | None = None,
                 dtype: str = "float32"):
        from transformers import AutoModel, AutoProcessor
        self.device = device or pick_device()
        self.dtype = torch.float16 if dtype == "float16" else torch.float32
        last = None
        for mid in ([model_id] if model_id else EMBED_MODELS):
            try:
                self.processor = AutoProcessor.from_pretrained(mid)
                self.model = AutoModel.from_pretrained(mid, torch_dtype=self.dtype)
                self.model_id = mid
                break
            except Exception as e:                      # noqa: BLE001
                last = e
                print(f"[embed] could not load {mid}: {type(e).__name__}")
        else:
            raise RuntimeError(f"no image encoder could be loaded: {last}")
        self.model.eval().to(self.device)
        self.is_siglip = "siglip" in self.model_id.lower()
        print(f"[embed] {self.model_id} on {self.device} ({self.dtype})")

    @torch.no_grad()
    def encode_images(self, frames: list[np.ndarray], batch: int = 32) -> np.ndarray:
        feats = []
        for i in range(0, len(frames), batch):
            chunk = [Image.fromarray(f) for f in frames[i:i + batch]]
            inp = self.processor(images=chunk, return_tensors="pt")
            inp = {k: v.to(self.device, self.dtype if v.is_floating_point() else v.dtype)
                   for k, v in inp.items()}
            f = self.model.get_image_features(**inp)
            f = torch.nn.functional.normalize(f.float(), dim=-1)
            feats.append(f.cpu().numpy())
        return np.concatenate(feats, 0) if feats else np.zeros((0, self.dim), np.float32)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        kw = dict(padding="max_length", max_length=64) if self.is_siglip else dict(padding=True)
        inp = self.processor(text=texts, return_tensors="pt", truncation=True, **kw)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        f = self.model.get_text_features(**inp)
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()

    @property
    def dim(self) -> int:
        return int(getattr(self.model.config, "projection_dim", 0)
                   or getattr(self.model.config, "hidden_size", 768))


def embed_video(emb: Embedder, path: Path, key: str, fps: float = FPS,
                force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Returns (features [N,D], timestamps [N]). Cached under CACHE/key.npz."""
    f = CACHE / f"{key}.npz"
    if f.exists() and not force:
        d = np.load(f)
        return d["feat"].astype(np.float32), d["time"].astype(np.float32)
    times, frames = [], []
    for t, img in iter_frames(path, fps):
        times.append(t)
        frames.append(img)
    feat = emb.encode_images(frames) if frames else np.zeros((0, emb.dim), np.float32)
    time = np.asarray(times, np.float32)
    np.savez_compressed(f, feat=feat.astype(np.float16), time=time)
    return feat.astype(np.float32), time
