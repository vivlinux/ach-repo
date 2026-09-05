"""Stage 2: VLM verification of candidate intervals only.

The cheap stage runs on every frame; this runs on the handful of windows that
crossed threshold. That is the whole point of the cascade — the false-alarm
rate is cut by a model that never has to keep up with the video stream.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .config import SYSTEM_HINT, VERIFY_Q, VLM_FALLBACKS, VLM_MODEL
from .frames import sample_frames
from .postprocess import Interval


class Verifier:
    def __init__(self, model_id: str | None = None, max_tokens: int = 24):
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
        self._generate, self._tmpl = generate, apply_chat_template
        last = None
        for mid in ([model_id] if model_id else [VLM_MODEL, *VLM_FALLBACKS]):
            try:
                self.model, self.processor = load(mid)
                self.config = load_config(mid)
                self.model_id = mid
                break
            except Exception as e:                       # noqa: BLE001
                last = e
                print(f"[verify] could not load {mid}: {type(e).__name__}: {e}")
        else:
            raise RuntimeError(f"no VLM could be loaded: {last}")
        self.max_tokens = max_tokens
        # per-call latencies feed runtime_metadata.model_runtimes in the
        # submission -- that block is where the latency bonus is computed from
        self.call_times_ms: list[float] = []
        print(f"[verify] {self.model_id}")

    def ask(self, images: list, question: str) -> str:
        import time
        t0 = time.time()
        prompt = self._tmpl(self.processor, self.config,
                            f"{SYSTEM_HINT}\n\n{question}", num_images=len(images))
        out = self._generate(self.model, self.processor, prompt, images,
                             max_tokens=self.max_tokens, verbose=False)
        text = getattr(out, "text", out)
        self.call_times_ms.append((time.time() - t0) * 1000.0)
        return str(text).strip()

    def check(self, video_path: Path, iv: Interval, k: int = 6):
        """Returns (keep: bool, answer: str)."""
        pad = 0.5
        frames = sample_frames(video_path, max(0.0, iv.start - pad), iv.end + pad, k=k)
        if not frames:
            return True, "no frames"
        images = [Image.fromarray(f) for f in frames]
        q = VERIFY_Q.get(iv.cls, f"Do these frames show {iv.cls.replace('_', ' ')}?")
        ans = self.ask(images, q)
        low = ans.lower()
        keep = low.startswith("yes") or (" yes" in low[:40] and "no," not in low[:10])
        return keep, ans


def verify_intervals(verifier: "Verifier", video_path: Path,
                     intervals: list[Interval], keep_score: float = 0.85,
                     k: int = 6, log=print) -> list[Interval]:
    """Drop intervals the VLM rejects. Very high-confidence ones skip the check."""
    kept = []
    for iv in intervals:
        if iv.score >= keep_score:
            kept.append(iv)
            continue
        ok, ans = verifier.check(video_path, iv, k=k)
        # video id is in the line so scripts/gate_report.py can audit each
        # decision against ground truth
        log(f"    verify {Path(video_path).stem} {iv.cls} [{iv.start:.1f}-{iv.end:.1f}] -> "
            f"{'KEEP' if ok else 'DROP'} :: {ans[:70]}")
        if ok:
            kept.append(iv)
    return kept


def describe(verifier: "Verifier", video_path: Path, iv: Interval, k: int = 4) -> str:
    """One-line description_summary for the submission CSV."""
    frames = sample_frames(video_path, iv.start, iv.end, k=k)
    if not frames:
        return iv.cls.replace("_", " ")
    images = [Image.fromarray(f) for f in frames]
    return verifier.ask(images, "Describe what is happening in one short sentence.")
