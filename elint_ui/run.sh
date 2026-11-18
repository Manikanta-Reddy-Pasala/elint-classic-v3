#!/bin/bash
# Start the ELINT UI on its own (the repo's ../run.sh starts it together with everything else).
# Creates .venv (psycopg2) on first run. Override any setting via environment.
set -e
cd "$(dirname "$0")"
R="$(cd .. && pwd)"
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv && .venv/bin/pip install -q psycopg2-binary; fi
export AI_PROCESSOR="${AI_PROCESSOR:-$R/target/release/ai_processor}"
export DATA_DIR="${DATA_DIR:-$R/data}"
export EXPECTED_JSON="${EXPECTED_JSON:-$PWD/expected_results.json}"
export PG_URL="${PG_URL:-postgresql://elint:elint@127.0.0.1:15432/sigint}"
export CH_URL="${CH_URL:-http://127.0.0.1:18123}"
export ONNX_URL="${ONNX_URL:-http://127.0.0.1:8000}"
export UI_PORT="${UI_PORT:-8800}"
[ -x "$AI_PROCESSOR" ] || echo "NOTE: $AI_PROCESSOR not built yet (run ../run.sh build)"
exec .venv/bin/python server.py
