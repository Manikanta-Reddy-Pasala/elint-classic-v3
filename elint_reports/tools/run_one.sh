#!/bin/bash
# usage: run_one.sh <classic|onnx> <csv> [logdir]
# Wipes DBs, runs ai_processor once, prints a RESULT line with counts + names.
MODE="$1"; CSV="$2"; OUT="${3:-$HOME/coderepo/runs/adhoc}"
BIN="${BIN:-$HOME/coderepo/ai_elint/target/release/ai_processor}"
mkdir -p "$OUT"; base=$(basename "$CSV" .csv); LOG="$OUT/${base}_${MODE}.log"
export PG_URL=postgresql://elint:elint@127.0.0.1:15432/sigint CH_URL=http://127.0.0.1:18123
export CLICKHOUSE_DB=signals CLICKHOUSE_USER=elint CLICKHOUSE_PASSWORD=elint
export ONNX_URL="${ONNX_URL:-http://127.0.0.1:8000}"
if [ "$MODE" = onnx ]; then export CLASSIC=0; unset EMBED; else export CLASSIC=1 EMBED=classic; fi
psql_q(){ PGPASSWORD=elint psql -h 127.0.0.1 -p 15432 -U elint -d sigint -t -A -c "$1" 2>/dev/null; }
t0=$(date +%s.%N); timeout 1800 "$BIN" --csv "$CSV" > "$LOG" 2>&1; rc=$?; t1=$(date +%s.%N)
dt=$(python3 -c "print(f'{$t1-$t0:.2f}')")
em=$(psql_q "select count(*) from emitters"); sg=$(psql_q "select count(*) from signals")
cl=$(psql_q "select count(*) from clusters"); orphan=$(psql_q "select count(*) from clusters where signal_id is null")
names=$(psql_q "select string_agg(name||':'||c, ',' order by name) from (select name, count(*) c from emitters group by name) x")
echo "RESULT|$base|engine=$MODE|rc=$rc|t=${dt}s|pg_em=$em|pg_sig=$sg|pg_clu=$cl|orphan_clu=$orphan|names=$names"
