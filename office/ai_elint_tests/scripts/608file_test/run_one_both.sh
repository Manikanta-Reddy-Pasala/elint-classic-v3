#!/bin/bash
# Run one known-emitter file through classic and ONNX; write RESULT lines.
BIN=/home/sysadm/ai_elint_rust/target/release/ai_processor
CSV="$1"
OUT="$2"
mkdir -p "$OUT"
BASE=$(basename "$CSV" .csv)
/home/sysadm/run_one_binfixed.sh "$BIN" classic "$CSV" "$OUT/${BASE}_classic" >> "$OUT/all.log" 2>&1
grep "^RESULT|" "$OUT/${BASE}_classic"/*.log >> "$OUT/RESULTS.txt"
/home/sysadm/run_one_binfixed.sh "$BIN" onnx "$CSV" "$OUT/${BASE}_onnx" >> "$OUT/all.log" 2>&1
grep "^RESULT|" "$OUT/${BASE}_onnx"/*.log >> "$OUT/RESULTS.txt"
