#!/bin/bash
# One-time setup: Python venv with the ROCm build of onnxruntime.
# Needs ROCm 6.4.2 runtime + libs on the host (apt: rocm-libs migraphx).
set -euo pipefail
cd "$(dirname "$0")"
WHL=onnxruntime_rocm-1.21.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q numpy==1.26.4
[ -f "$WHL" ] || curl -sSLO "https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/$WHL"
.venv/bin/pip install -q "./$WHL"
.venv/bin/python -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
