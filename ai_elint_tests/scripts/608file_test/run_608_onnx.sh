#!/bin/bash
# run_608_onnx.sh — Resume the ONNX phase for all 608 files.
# Skips files already in the output. Appends new results.
# Usage: bash run_608_onnx.sh <BIN> <LIST> <ONNX_OUT>

BIN="$1"
LIST="$2"
ONNX_OUT="$3"

mkdir -p "$(dirname "$ONNX_OUT")"
echo "=== ONNX resume: $(date) ==="
echo "BIN=$BIN  LIST=$LIST  OUT=$ONNX_OUT"

while IFS= read -r csv; do
  [ -z "$csv" ] && continue
  base=$(basename "$csv" .csv)
  if grep -q "RESULT|${base}|engine=onnx|" "$ONNX_OUT" 2>/dev/null; then
    echo "  SKIP (done): $base"
    continue
  fi
  echo "  RUN (onnx): $base"
  bash /home/sysadm/run_one.sh "$BIN" "onnx" "$csv" "$(dirname "$ONNX_OUT")" 2>&1 \
    | grep '^RESULT|' | sed "s/RESULT|/RESULT|/; s/\$|/|engine=onnx|/" \
    >> "$ONNX_OUT" || echo "RESULT|${base}|engine=onnx|rc=-1|FAIL" >> "$ONNX_OUT"
done < "$LIST"
echo "BATCH-DONE onnx $(grep -c '^RESULT|' "$ONNX_OUT" 2>/dev/null || echo 0) files" >> "$ONNX_OUT"
echo "=== ONNX batch complete: $(date) ==="
