#!/bin/bash
# Start the ONNX model service on its own (the repo's ../run.sh starts it together with everything else).
# Creates .venv on first run (GPU/ROCm build on a box with /dev/kfd via setup.sh, else CPU onnxruntime).
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  if [ -e /dev/kfd ]; then ./setup.sh && touch .gpu
  else python3 -m venv .venv && .venv/bin/pip install -q numpy==1.26.4 onnxruntime; fi
fi
export MODEL_PATH="${MODEL_PATH:-$(cd .. && pwd)/model/transformer_model_known_v16.onnx}"
[ -f "$MODEL_PATH" ] || { echo "model file missing: $MODEL_PATH (set MODEL_PATH=...)"; exit 1; }
export HOST="${HOST:-0.0.0.0}" PORT="${PORT:-8000}"
if [ -z "${DEVICE:-}" ]; then [ -f .gpu ] && DEVICE=hybrid || DEVICE=cpu; fi   # hybrid = GPU + CPU workers
export DEVICE
export GPU_PROVIDER="${GPU_PROVIDER:-migraphx}"  # migraphx | rocm
export GPU_WORKERS="${GPU_WORKERS:-1}"
export CPU_WORKERS="${CPU_WORKERS:-2}"
export CPU_THREADS="${CPU_THREADS:-6}"
# Radeon 890M is gfx1150; ROCm 6.4 kernels run on it as gfx1100.
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
exec .venv/bin/python server.py
