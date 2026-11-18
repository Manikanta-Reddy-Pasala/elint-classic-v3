#!/bin/bash
# run_608_both.sh — Run all 608 files: classic first, then ONNX.
# Resume-aware: skips files already in the output.
# Usage: bash run_608_both.sh <BIN> <LIST> <CLASSIC_OUT> <ONNX_OUT>

BIN="$1"
LIST="$2"
CLASSIC_OUT="$3"
ONNX_OUT="$4"

mkdir -p "$(dirname "$CLASSIC_OUT")"
mkdir -p "$(dirname "$ONNX_OUT")"

echo "=== 608-file batch: $(date) ==="
echo "BIN=$BIN"
echo "LIST=$LIST"
echo "CLASSIC_OUT=$CLASSIC_OUT"
echo "ONNX_OUT=$ONNX_OUT"
echo ""

# Phase 1: classic
echo ">>> PHASE 1: classic"
while IFS= read -r csv; do
  [ -z "$csv" ] && continue
  base=$(basename "$csv" .csv)
  # Skip if already done (PASS or FAIL line exists for this file+engine)
  if grep -q "RESULT|${base}|engine=classic|" "$CLASSIC_OUT" 2>/dev/null; then
    echo "  SKIP (done): $base"
    continue
  fi
  echo "  RUN (classic): $base"
  bash /home/sysadm/run_one.sh "$BIN" "classic" "$csv" "$(dirname "$CLASSIC_OUT")" 2>&1 \
    | grep '^RESULT|' | sed "s/RESULT|/RESULT|/; s/\$|/|engine=classic|/" \
    >> "$CLASSIC_OUT" || echo "RESULT|${base}|engine=classic|rc=-1|FAIL" >> "$CLASSIC_OUT"
done < "$LIST"
echo "BATCH-DONE classic $(grep -c '^RESULT|' "$CLASSIC_OUT" 2>/dev/null || echo 0) files" >> "$CLASSIC_OUT"

# Phase 2: onnx
echo ""
echo ">>> PHASE 2: onnx"
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

echo ""
echo "=== 608-file batch complete: $(date) ==="
