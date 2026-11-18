#!/bin/bash
# run_5_both.sh — run the 5 smoke CSVs through the MASTER binary in both modes
# (classic = no ONNX_PATH, onnx = with ONNX_PATH), one file at a time, DBs wiped each time.
pkill -9 -f "timeout 900 ./target/release/ai_processor" 2>/dev/null
pkill -9 -f run_batch.sh 2>/dev/null
sleep 3
BIN=/home/sysadm/ai_elint_master_run/target/release/ai_processor
export PG_URL="postgresql://elint:elint@127.0.0.1:15432/sigint"
export CH_URL="http://127.0.0.1:18123"
LIST=/home/sysadm/smoke3_list.txt
OUT5=/home/sysadm/5files_both_results.txt
: > "$OUT5"
while IFS= read -r csv; do
  [ -z "$csv" ] && continue
  for MODE in classic onnx; do
    bash /home/sysadm/run_one.sh "$BIN" "$MODE" "$csv" /home/sysadm/5files_logs >> "$OUT5" 2>&1
    grep "^RESULT" /home/sysadm/5files_logs/run_${MODE}.log >> "$OUT5" 2>/dev/null
    echo "  $MODE $(basename "$csv" .csv) done"
  done
done < "$LIST"
echo "=== 5-FILE BOTH RESULTS ==="
cat "$OUT5"
echo "=== SESSION CHECK (onnx loaded?) ==="
grep -E "session ready|failed to load real|engine:" /home/sysadm/5files_logs/run_onnx.log | tail -3
grep -c "^RESULT" "$OUT5"
