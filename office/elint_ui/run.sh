#!/bin/bash
# Start the ELINT runner UI on :8800 (override any setting via environment).
cd "$(dirname "$0")"
export AI_PROCESSOR="${AI_PROCESSOR:-$HOME/coderepo/ai_elint/target/release/ai_processor}"
export DATA_DIR="${DATA_DIR:-$HOME/coderepo/ai_elint/data}"
export EXPECTED_JSON="${EXPECTED_JSON:-$HOME/coderepo/ai_elint_tests/tests/test_data/expected_results.json}"
export PG_URL="${PG_URL:-postgresql://elint:elint@127.0.0.1:15432/sigint}"
export CH_URL="${CH_URL:-http://127.0.0.1:18123}"
export ONNX_URL="${ONNX_URL:-http://127.0.0.1:8000}"
export UI_PORT="${UI_PORT:-8800}"
exec python3 server.py
