#!/bin/bash
# elint.sh — build, start and test the ELINT engines (Classic v3 / Classic v2 / ONNX) from this repo.
#
#   ./elint.sh setup                 build Rust + model-service venv + UI venv + test-judge image
#   ./elint.sh start                 databases + ONNX model service + UI   (prints the UI URL)
#   ./elint.sh run <file.csv> [classic|v2|onnx]   one file through ai_processor, prints + saves emitters
#   ./elint.sh selftest [quick]      classic accuracy on the 608 files, no DB / model needed
#   ./elint.sh judge [classic|v2|onnx] [pytest -k expr]   608-file test judge (ai_elint_tests)
#   ./elint.sh status | stop [db] | logs [model|ui]
#   ./elint.sh clean --yes           ONLY way anything is deleted: removes the docker DB containers + their data
#
# Databases (default DB_MODE=docker: create or reuse local containers, never deleted):
#   DB_MODE=external PG_HOST=.. PG_PORT=.. PG_USER=.. PG_PASS=.. CH_HOST=.. CH_PORT=.. CH_NPORT=..   use your own DBs
#   PG_DB=<name>   use another Postgres database (created + schema loaded if missing; default sigint)
# Note: ai_processor --csv itself replaces the previous run's results in the chosen database
# (v2 behaviour, the judge relies on it). Use another PG_DB to keep a database untouched;
# every ./elint.sh run also keeps a copy of its emitters in .run/results/.
set -euo pipefail
R="$(cd "$(dirname "$0")" && pwd)"

DATA="${DATA:-$R/data}"                                  # folder holding pdw_generator/{known,unknown}_emitters/*.csv
MODEL_FILE="${MODEL_FILE:-$R/model/transformer_model_known_v16.onnx}"   # FP32 v16 (not int8)
EXPECTED_DIR="${EXPECTED_DIR:-$R/ai_elint_tests/tests/test_data}"        # judge expected-result CSVs (copy from the box)
DB_MODE="${DB_MODE:-docker}"                             # docker | external
PG_HOST="${PG_HOST:-127.0.0.1}"; PG_PORT="${PG_PORT:-15432}"; PG_USER="${PG_USER:-elint}"; PG_PASS="${PG_PASS:-elint}"; PG_DB="${PG_DB:-sigint}"
CH_HOST="${CH_HOST:-127.0.0.1}"; CH_PORT="${CH_PORT:-18123}"; CH_NPORT="${CH_NPORT:-19000}"; CH_USER="${CH_USER:-elint}"; CH_PASS="${CH_PASS:-elint}"
PG_NAME="${PG_NAME:-elint_pg}"; CH_NAME="${CH_NAME:-elint_ch}"           # docker container names (DB_MODE=docker)
MODEL_PORT="${MODEL_PORT:-8000}"; UI_PORT="${UI_PORT:-8800}"
JUDGE_IMAGE="${JUDGE_IMAGE:-elint-judge:latest}"
# Model device: ROCm box (/dev/kfd + GPU setup done) -> hybrid (GPU + CPU workers); otherwise cpu.
if [ -z "${DEVICE:-}" ]; then [ -e /dev/kfd ] && [ -f "$R/onnx_model_service/.gpu" ] && DEVICE=hybrid || DEVICE=cpu; fi
RUN="$R/.run"; mkdir -p "$RUN/logs" "$RUN/results"

export PG_URL="postgresql://$PG_USER:$PG_PASS@$PG_HOST:$PG_PORT/$PG_DB"
export CH_URL="http://$CH_HOST:$CH_PORT" CLICKHOUSE_DB=signals CLICKHOUSE_USER="$CH_USER" CLICKHOUSE_PASSWORD="$CH_PASS"
export ONNX_URL="http://127.0.0.1:$MODEL_PORT"
BIN="$R/target/release/ai_processor"

say() { printf '\n== %s\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null || die "'$1' not found ($2)"; }
alive() { [ -f "$RUN/$1.pid" ] && kill -0 "$(cat "$RUN/$1.pid")" 2>/dev/null; }
bg() { local name=$1; shift; setsid nohup "$@" > "$RUN/logs/$name.log" 2>&1 < /dev/null & echo $! > "$RUN/$name.pid"; }

# psql against the chosen database: local container, host psql, or a throw-away client container
pgq() {
  local db="${PGQ_DB:-$PG_DB}"
  if [ "$DB_MODE" = docker ]; then docker exec -i -e PGPASSWORD="$PG_PASS" "$PG_NAME" psql -U "$PG_USER" -d "$db" "$@"
  elif command -v psql >/dev/null; then PGPASSWORD="$PG_PASS" psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$db" "$@"
  else docker run --rm -i --network host -e PGPASSWORD="$PG_PASS" postgres:16 psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$db" "$@"; fi
}

build() {
  need cargo "install Rust: https://rustup.rs"
  say "build ai_processor + classic_eval (release)"
  (cd "$R" && RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}" cargo build --release -q -p ai_processor --bins 2>&1 | grep -E "^error" || true)
  [ -x "$BIN" ] || die "build failed (run: cargo build --release -p ai_processor)"
}

setup() {
  build
  need python3 "python3 + venv"
  say "model service venv"
  if [ ! -x "$R/onnx_model_service/.venv/bin/python" ]; then
    if [ -e /dev/kfd ] && [ "${GPU_SETUP:-1}" = 1 ]; then
      (cd "$R/onnx_model_service" && ./setup.sh) && touch "$R/onnx_model_service/.gpu"   # ROCm onnxruntime (box GPU)
    else
      python3 -m venv "$R/onnx_model_service/.venv"
      "$R/onnx_model_service/.venv/bin/pip" install -q numpy==1.26.4 onnxruntime
    fi
  fi
  echo ok
  say "UI venv"
  [ -x "$R/elint_ui/.venv/bin/python" ] || { python3 -m venv "$R/elint_ui/.venv"; "$R/elint_ui/.venv/bin/pip" install -q psycopg2-binary; }
  echo ok
  if command -v docker >/dev/null; then
    say "test-judge image $JUDGE_IMAGE"; docker build -q -t "$JUDGE_IMAGE" "$R/ai_elint_tests/docker" >/dev/null && echo ok
  fi
  [ -f "$MODEL_FILE" ] || echo "NOTE: copy the model to $MODEL_FILE (or set MODEL_FILE=...) before ./elint.sh start"
  ls "$DATA"/pdw_generator/*/*.csv >/dev/null 2>&1 || echo "NOTE: put the CSVs in $DATA/pdw_generator/{known,unknown}_emitters/ (or set DATA=...)"
}

db() {
  if [ "$DB_MODE" = docker ]; then
    need docker "docker is needed for local Postgres + ClickHouse (or use DB_MODE=external)"
    say "databases: docker $PG_NAME :$PG_PORT, $CH_NAME :$CH_PORT/:$CH_NPORT (reused if they exist, never deleted)"
    if docker inspect "$PG_NAME" >/dev/null 2>&1; then docker start "$PG_NAME" >/dev/null; else
      docker run -d --name "$PG_NAME" --restart unless-stopped -p 127.0.0.1:$PG_PORT:5432 \
        -e POSTGRES_USER="$PG_USER" -e POSTGRES_PASSWORD="$PG_PASS" postgres:16 >/dev/null; fi
    if docker inspect "$CH_NAME" >/dev/null 2>&1; then docker start "$CH_NAME" >/dev/null; else
      docker run -d --name "$CH_NAME" --restart unless-stopped -p 127.0.0.1:$CH_PORT:8123 -p 127.0.0.1:$CH_NPORT:9000 \
        -e CLICKHOUSE_USER="$CH_USER" -e CLICKHOUSE_PASSWORD="$CH_PASS" -e CLICKHOUSE_DB=signals -e CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1 \
        -v "$R/clickhouse/users.d/00_perf.xml":/etc/clickhouse-server/config.d/00_perf.xml \
        -v "$R/clickhouse/users_profiles/old-analyzer.xml":/etc/clickhouse-server/users.d/old-analyzer.xml \
        clickhouse/clickhouse-server:24.8 >/dev/null; fi
  else
    say "databases: external Postgres $PG_HOST:$PG_PORT/$PG_DB, ClickHouse $CH_HOST:$CH_PORT (nothing started)"
  fi
  local ok=0
  for i in $(seq 90); do
    PGQ_DB=postgres pgq -tAc "select 1" >/dev/null 2>&1 && curl -sf "$CH_URL/ping" >/dev/null && { ok=1; break; }; sleep 1
  done
  [ $ok = 1 ] || die "databases not reachable (Postgres $PG_HOST:$PG_PORT, ClickHouse $CH_URL)"
  # ClickHouse tables (CREATE ... IF NOT EXISTS: existing tables and data are kept)
  local stmt=""
  while IFS= read -r line; do
    case "$line" in --*) continue ;; ";") [ -n "$stmt" ] && { curl -sf -u "$CH_USER:$CH_PASS" "$CH_URL/" --data-binary "$stmt" >/dev/null || die "ClickHouse schema failed: ${stmt:0:60}"; }; stmt="" ;; *) stmt+="$line"$'\n' ;; esac
  done < "$R/clickhouse/engine_schema.sql"
  if [ "$(PGQ_DB=postgres pgq -tAc "select 1 from pg_database where datname='$PG_DB'")" != 1 ]; then
    say "create Postgres database $PG_DB + schema (postgres/engine_schema.sql)"
    PGQ_DB=postgres pgq -qc "create database \"$PG_DB\"" >/dev/null
    pgq -q -v ON_ERROR_STOP=1 < "$R/postgres/engine_schema.sql" >/dev/null
  elif [ "$(pgq -tAc "select data_type from information_schema.columns where table_name='emitters' and column_name='id'")" = integer ]; then
    echo "WARNING: database $PG_DB has the old schema (integer ids) and the engine will fail on it."
    echo "         Use a new one (PG_DB=<new name>) - created with the right schema automatically."
  fi
  echo "ready: $PG_URL , $CH_URL"
}

model() {
  [ -f "$MODEL_FILE" ] || die "model file missing: $MODEL_FILE (set MODEL_FILE=...)"
  [ -x "$R/onnx_model_service/.venv/bin/python" ] || die "run ./elint.sh setup first"
  if alive model; then echo "model service already running (pid $(cat "$RUN/model.pid"))"; return; fi
  curl -sf "$ONNX_URL/health" >/dev/null && die "port $MODEL_PORT already has a model service not started by this script (stop it, or set MODEL_PORT=...)"
  say "ONNX model service :$MODEL_PORT ($DEVICE)"
  (cd "$R/onnx_model_service" && export MODEL_PATH="$MODEL_FILE" HOST=127.0.0.1 PORT=$MODEL_PORT DEVICE=$DEVICE \
     CPU_WORKERS=${CPU_WORKERS:-2} CPU_THREADS=${CPU_THREADS:-6} HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-11.0.0} \
     && bg model .venv/bin/python server.py)
  for i in $(seq 180); do curl -sf "$ONNX_URL/health" >/dev/null && { curl -s "$ONNX_URL/health"; echo; return; }; sleep 1; done
  die "model service not ready (./elint.sh logs model)"
}

ui() {
  [ -x "$R/elint_ui/.venv/bin/python" ] || die "run ./elint.sh setup first"
  [ -x "$BIN" ] || die "run ./elint.sh setup first (no $BIN)"
  if alive ui; then echo "UI already running (pid $(cat "$RUN/ui.pid"))"; return; fi
  curl -sf "http://127.0.0.1:$UI_PORT/" >/dev/null && die "port $UI_PORT is already in use (stop that process, or set UI_PORT=...)"
  say "UI :$UI_PORT"
  (cd "$R/elint_ui" && export AI_PROCESSOR="$BIN" DATA_DIR="$DATA" UI_PORT=$UI_PORT \
     EXPECTED_JSON="$EXPECTED_DIR/expected_results.json" && bg ui .venv/bin/python server.py)
  for i in $(seq 20); do curl -sf "http://127.0.0.1:$UI_PORT/" >/dev/null && break; sleep 1; done
  local ip; ip=$(hostname -I 2>/dev/null | awk '{print $1}'); echo "Open: http://${ip:-127.0.0.1}:$UI_PORT"
}

engine_env() {   # $1 = classic | v2 | onnx
  case "$1" in
    classic) echo "CLASSIC=1 EMBED=classic CLASSIC_V3=1" ;;
    v2)      echo "CLASSIC=1 EMBED=classic CLASSIC_V3=0" ;;
    onnx)    echo "CLASSIC=0" ;;
    *) die "engine must be classic, v2 or onnx" ;;
  esac
}

run_one() {
  local f="${1:-}" e="${2:-classic}"; [ -f "$f" ] || die "usage: ./elint.sh run <file.csv> [classic|v2|onnx]"
  [ -x "$BIN" ] || die "run ./elint.sh setup first"
  [ "$e" = onnx ] && ! curl -sf "$ONNX_URL/health" >/dev/null && die "model service not running (./elint.sh start)"
  say "ai_processor $e on $(basename "$f")  (database $PG_DB)"
  local log="$RUN/logs/run_last.log"
  env $(engine_env "$e") ELINT_LOG_DIR="$RUN/logs" "$BIN" --csv "$f" > "$log" 2>&1 || { tail -20 "$log"; die "ai_processor failed (full log $log)"; }
  grep -oE "LAYER\|(engine_total|load_csv|wipe_tables)\|[0-9.]+" "$log" | sed 's/LAYER|/  ms /' || true
  local res; res="$RUN/results/$(date +%Y%m%d_%H%M%S)_${e}_$(basename "$f" .csv).txt"
  pgq -c "select name, count(*) as emitters from emitters group by name order by name" | tee "$res"
  echo "saved: $res"
}

judge() {
  local e="${1:-classic}"; shift || true; local k="${1:-source_omni or main_lobe}"
  need docker "the judge runs in a container"
  [ -x "$BIN" ] || die "run ./elint.sh setup first"
  ls "$EXPECTED_DIR"/known_emitters_db_emitter_level.csv >/dev/null 2>&1 || \
    die "expected-result CSVs missing in $EXPECTED_DIR (copy ai_elint_tests/tests/test_data/*.csv from the box, or set EXPECTED_DIR)"
  [ "$e" = onnx ] && ! curl -sf "$ONNX_URL/health" >/dev/null && die "model service not running (./elint.sh start)"
  docker image inspect "$JUDGE_IMAGE" >/dev/null 2>&1 || docker build -q -t "$JUDGE_IMAGE" "$R/ai_elint_tests/docker" >/dev/null
  local out="$RUN/judge_${e}_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$out"
  say "test judge: $e  (-k \"$k\")  database $PG_DB  -> $out"
  local v3=1; [ "$e" = v2 ] && v3=0
  mkdir -p "$R/ai_elint_tests/data/input_csv_files/pdw_generator"   # mount point (CSVs are mounted, not copied)
  docker run --rm --network host -u "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$R":/repo:ro -v "$R/ai_elint_tests":/repo/ai_elint_tests -v "$DATA/pdw_generator":/repo/ai_elint_tests/data/input_csv_files/pdw_generator:ro \
    -v "$EXPECTED_DIR":/repo/ai_elint_tests/tests/test_data:ro -v "$out":/out \
    -w /repo/ai_elint_tests \
    -e AI_ENGINE=rust -e RUST_ENGINE="$([ "$e" = onnx ] && echo onnx || echo classic)" -e CLASSIC_V3=$v3 \
    -e RUST_AI_PROCESSOR=/repo/target/release/ai_processor -e RUST_AI_TIMEOUT=1800 -e ONNX_URL="$ONNX_URL" \
    -e PG_URL="$PG_URL" -e CH_URL="$CH_URL" -e CLICKHOUSE_USER="$CH_USER" -e CLICKHOUSE_PASSWORD="$CH_PASS" -e CLICKHOUSE_DB=signals \
    -e ELINT_LOG_DIR=/tmp/elint_logs -e CH_HOST="$CH_HOST" -e CH_PORT=$CH_NPORT -e CH_USER="$CH_USER" -e CH_PASSWORD="$CH_PASS" -e CH_SECURE=0 \
    -e PG_HOST="$PG_HOST" -e PG_PORT=$PG_PORT -e PG_USER="$PG_USER" -e PG_PASSWORD="$PG_PASS" -e PG_DATABASE="$PG_DB" -e PG_DB="$PG_DB" \
    "$JUDGE_IMAGE" python -m pytest tests/test_pdw_generator.py -p no:cacheprovider -q --no-header -rA \
      -o junit_family=xunit2 --junitxml=/out/pytest.xml -k "$k" > "$out/pytest.log" 2>&1 || true
  tail -1 "$out/pytest.log"; echo "full log: $out/pytest.log"
}

status() {
  for s in model ui; do alive $s && echo "$s: running (pid $(cat "$RUN/$s.pid"))" || echo "$s: stopped"; done
  if [ "$DB_MODE" = docker ] && command -v docker >/dev/null; then
    docker ps -a --filter "name=^($PG_NAME|$CH_NAME)\$" --format '{{.Names}}: {{.Status}}'; fi
  curl -sf "$ONNX_URL/health" >/dev/null && echo "model health: ok" || echo "model health: down"
  echo "database: $PG_URL , $CH_URL"
}

stop() {   # stops the model service + UI only; databases and their data are kept
  for s in ui model; do
    if alive $s; then kill -- -"$(cat "$RUN/$s.pid")" 2>/dev/null || kill "$(cat "$RUN/$s.pid")"; echo "$s stopped"; fi
    rm -f "$RUN/$s.pid"
  done
  if [ "${1:-}" = db ] && [ "$DB_MODE" = docker ]; then docker stop "$PG_NAME" "$CH_NAME" >/dev/null && echo "databases stopped (data kept)"; fi
}

clean() {
  [ "${1:-}" = --yes ] || die "this deletes the docker DB containers $PG_NAME/$CH_NAME and all their data; run: ./elint.sh clean --yes"
  [ "$DB_MODE" = docker ] || die "DB_MODE=external: nothing to clean here (your databases are never touched by clean)"
  stop; docker rm -f "$PG_NAME" "$CH_NAME" >/dev/null 2>&1 || true; echo "removed $PG_NAME $CH_NAME"
}

case "${1:-help}" in
  setup) setup ;;
  build) build ;;
  db) db ;;
  model) model ;;
  ui) ui ;;
  start) db; model; ui ;;
  run) shift; run_one "$@" ;;
  selftest) shift; DATA="$DATA" "$R/tools/classic/selftest.sh" "$@" ;;
  judge) shift; judge "$@" ;;
  status) status ;;
  stop) shift; stop "$@" ;;
  clean) shift; clean "$@" ;;
  logs) tail -f "$RUN/logs/${2:-model}.log" ;;
  *) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
