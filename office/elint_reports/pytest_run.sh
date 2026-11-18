#!/bin/bash
# pytest_run.sh <classic|onnx> <outprefix> [pytest args...]
# Runs the ai_elint_tests harness (AI_ENGINE=rust) in the elint_pytest2 container.
# REPORT_DIR set -> engine runs through engine_wrapper.py and per-file reports are saved.
ENGINE="$1"; OUT="$2"; shift 2
HERE=/home/sysadm/coderepo/elint_reports
BIN=/home/sysadm/coderepo/ai_elint/target/release/ai_processor
[ -n "$REPORT_DIR" ] && PROC=$HERE/engine_wrapper.py || PROC=$BIN
docker exec -u 1000 -w /home/sysadm/coderepo/ai_elint_tests \
  -e HOME=/tmp -e AI_ENGINE=rust -e RUST_ENGINE="$ENGINE" -e RUST_AI_TIMEOUT=1800 \
  -e RUST_AI_PROCESSOR=$PROC -e REAL_AI_PROCESSOR=$BIN -e RUST_REPORT_DIR="$REPORT_DIR" \
  -e ONNX_URL=http://127.0.0.1:8000 -e ELINT_LOG_DIR=/tmp/elint_logs \
  -e CH_HOST=127.0.0.1 -e CH_PORT=19000 -e CH_USER=elint -e CH_PASSWORD=elint -e CH_SECURE=0 \
  -e PG_HOST=127.0.0.1 -e PG_PORT=15432 -e PG_USER=elint -e PG_PASSWORD=elint \
  elint_pytest2 python -m pytest tests/test_pdw_generator.py -p no:cacheprovider -q --no-header -rA \
  -o junit_family=xunit2 --junitxml="$OUT.xml" "$@" > "$OUT.log" 2>&1
echo "exit=$? $(tail -1 "$OUT.log")"
