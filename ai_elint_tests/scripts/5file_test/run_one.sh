#!/bin/bash
# run_one.sh — Run ONE CSV file through ai_processor with correct env
# Usage: bash run_one.sh <BIN> <MODE> <CSV_FILE> <OUT_DIR>
set -e
BIN="$1"
MODE="$2"
CSV="$3"
OUT="$4"
mkdir -p "$OUT"
LOG="${OUT}/run_${MODE}.log"

export PG_URL="postgresql://elint:elint@127.0.0.1:15432/sigint"
export CH_URL="http://127.0.0.1:18123"
if [ "$MODE" = "onnx" ]; then
  export ONNX_PATH="/home/sysadm/ai_elint_run/integration/model/transformer_model_known_v16.onnx"
  echo "MODE=onnx  ONNX_PATH=$ONNX_PATH" >> "$LOG"
else
  unset ONNX_PATH
  echo "MODE=classic" >> "$LOG"
fi
echo "BIN=$BIN  CSV=$CSV" >> "$LOG"
echo "" >> "$LOG"

# Wipe PG
sudo PGPASSWORD=elint psql -h 127.0.0.1 -p 15432 -U elint -d sigint \
  -c "TRUNCATE clusters, signals, emitters CASCADE;" 2>>"$LOG" || true
# Wipe CH
curl -s "http://127.0.0.1:18123/?query=TRUNCATE%20pdw" >> /dev/null 2>&1 || true
curl -s "http://127.0.0.1:18123/?query=TRUNCATE%20clusters" >> /dev/null 2>&1 || true
curl -s "http://127.0.0.1:18123/?query=TRUNCATE%20csv_input" >> /dev/null 2>&1 || true

mkdir -p /home/sysadm/5files_logs
t0=$(date +%s.%N)
"$BIN" --csv "$CSV" >> "$LOG" 2>&1
rc=$?
t1=$(date +%s.%N)
dt=$(python3 -c "print(f'{$t1 - $t0:.2f}')")

pg_em=$(sudo PGPASSWORD=elint psql -h 127.0.0.1 -p 15432 -U elint -d sigint -t -A -c "SELECT COUNT(*) FROM emitters" 2>/dev/null || echo "?")
pg_sig=$(sudo PGPASSWORD=elint psql -h 127.0.0.1 -p 15432 -U elint -d sigint -t -A -c "SELECT COUNT(*) FROM signals" 2>/dev/null || echo "?")
pg_clu=$(sudo PGPASSWORD=elint psql -h 127.0.0.1 -p 15432 -U elint -d sigint -t -A -c "SELECT COUNT(*) FROM clusters" 2>/dev/null || echo "?")
ch_pdw=$(curl -s "http://127.0.0.1:18123/?query=SELECT%20count()%20FROM%20pdw" 2>/dev/null || echo "?")
ch_clu=$(curl -s "http://127.0.0.1:18123/?query=SELECT%20count()%20FROM%20clusters" 2>/dev/null || echo "?")

status="PASS"
if [ $rc -ne 0 ] || [ "$pg_em" = "0" ] || [ "$pg_em" = "?" ]; then
  status="FAIL"
fi
base=$(basename "$CSV" .csv)
echo "RESULT|${base}|${status}|rc=${rc}|t=${dt}s|pg_em=${pg_em}|pg_sig=${pg_sig}|pg_clu=${pg_clu}|ch_pdw=${ch_pdw}|ch_clu=${ch_clu}" >> "$LOG"
echo "  ${status}  rc=$rc  t=${dt}s  emitters=$pg_em signals=$pg_sig clusters=$pg_clu ch_pdw=$ch_pdw ch_clu=$ch_clu"
