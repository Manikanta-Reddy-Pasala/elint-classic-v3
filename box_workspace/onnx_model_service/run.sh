#!/bin/bash
# Start the model service. Every setting can be overridden from the environment.
cd "$(dirname "$0")"
export MODEL_PATH="${MODEL_PATH:-$PWD/model/transformer_model_known_v16.onnx}"
export HOST="${HOST:-0.0.0.0}" PORT="${PORT:-8000}"
export DEVICE="${DEVICE:-hybrid}"            # hybrid = GPU + CPU workers together
export GPU_PROVIDER="${GPU_PROVIDER:-migraphx}"  # migraphx | rocm
export GPU_WORKERS="${GPU_WORKERS:-1}"
export CPU_WORKERS="${CPU_WORKERS:-1}"
export CPU_THREADS="${CPU_THREADS:-12}"
# Radeon 890M is gfx1150; ROCm 6.4 kernels run on it as gfx1100.
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
exec .venv/bin/python server.py
