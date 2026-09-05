#!/usr/bin/env bash
# One-time setup on an M-series Mac.
set -e
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
pip install -r requirements.txt
python - <<'PY'
import torch
print("torch", torch.__version__, "| MPS available:", torch.backends.mps.is_available())
try:
    import mlx.core as mx
    print("mlx ok, default device:", mx.default_device())
except Exception as e:
    print("mlx not available (stage 2 will be skipped):", e)
PY
echo
echo "Now: export VAD_DATA=/path/to/dataset   # the folder holding train/ and test/"
echo "Then: python -m vad.cli inspect"
