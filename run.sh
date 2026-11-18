#!/bin/bash
# run.sh — one script for the ELINT engine: build, databases, ONNX model service, UI, API, tests.
#
#   ./run.sh                          ONE SHOT: build (if needed) + databases + model service + UI + API
#   ./run.sh --pg-url URL --ch-url URL      same, using your own databases (nothing started for them)
#   ./run.sh run <file.csv> [classic|v2|onnx]   one file through ai_processor, prints + saves emitters
#   ./run.sh selftest [quick]         classic accuracy on the 608 files (no database / model needed)
#   ./run.sh status | stop [db] | logs [model|ui|api] | build | clean --yes
#
# Options (or the same names as environment variables):
#   --pg-url  postgresql://user:pass@host:port/db   your own Postgres   (default: local docker elint_pg :15432/sigint)
#   --ch-url  http://user:pass@host:port            your own ClickHouse (default: local docker elint_ch :18123)
#   --pg-db   NAME     another Postgres database (created with the schema if missing)
#   --data    DIR      folder holding pdw_generator/{known,unknown}_emitters/*.csv   (default ./data)
#   --model   FILE     ONNX model (default ./model/transformer_model_known_v16.onnx, FP32 not int8)
#   --device  cpu|gpu|hybrid   model service device (default: hybrid on a ROCm GPU box, else cpu)
#   --ui-port N  --model-port N  --api-port N
#   --no-model  --no-ui  --no-api      leave that part out of the one shot
#   --rebuild                          rebuild the Rust binaries even if they exist
#
# Nothing is deleted unless you run: ./run.sh clean --yes (removes the local docker DB containers + data).
# Each engine run replaces the previous results in the database it writes to (v2 behaviour);
# use --pg-db NAME to keep another database untouched. ./run.sh run keeps a copy in .run/results/.
set -euo pipefail
R="$(cd "$(dirname "$0")" && pwd)"

# ---------- options ----------
CMD=all; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --pg-url) PG_URL_IN="$2"; shift 2 ;;
    --ch-url) CH_URL_IN="$2"; shift 2 ;;
    --pg-db) PG_DB="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --model) MODEL_FILE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --ui-port) UI_PORT="$2"; shift 2 ;;
    --model-port) MODEL_PORT="$2"; shift 2 ;;
    --api-port) API_PORT="$2"; shift 2 ;;
    --no-model) NO_MODEL=1; shift ;;
    --no-ui) NO_UI=1; shift ;;
    --no-api) NO_API=1; shift ;;
    --rebuild) REBUILD=1; shift ;;
    -h|--help|help) CMD=help; shift ;;
    *) if [ "$CMD" = all ] && [ ${#ARGS[@]} -eq 0 ] && [[ "$1" != -* ]]; then CMD="$1"; else ARGS+=("$1"); fi; shift ;;
  esac
done

DATA="${DATA:-$R/data}"
MODEL_FILE="${MODEL_FILE:-$R/model/transformer_model_known_v16.onnx}"
PG_NAME="${PG_NAME:-elint_pg}"; CH_NAME="${CH_NAME:-elint_ch}"
MODEL_PORT="${MODEL_PORT:-8000}"; UI_PORT="${UI_PORT:-8800}"; API_PORT="${API_PORT:-8788}"
RUN="$R/.run"; mkdir -p "$RUN/logs" "$RUN/results"

# databases: --pg-url / --ch-url (or PG_URL_IN / CH_URL_IN) = external, else local docker
url_part() {  # url_part <url> user|pass|host|port|db
  local u="${1#*://}" auth="" hp db=""
  [[ "$u" == *@* ]] && { auth="${u%%@*}"; u="${u#*@}"; }
  hp="${u%%/*}"; [[ "$u" == */* ]] && db="${u#*/}"; db="${db%%\?*}"
  case "$2" in
    user) echo "${auth%%:*}" ;; pass) [[ "$auth" == *:* ]] && echo "${auth#*:}" || true ;;
    host) echo "${hp%%:*}" ;; port) [[ "$hp" == *:* ]] && echo "${hp##*:}" || true ;; db) echo "$db" ;;
  esac
}
if [ -n "${PG_URL_IN:-}" ]; then
  PG_EXT=1; PG_USER=$(url_part "$PG_URL_IN" user); PG_PASS=$(url_part "$PG_URL_IN" pass)
  PG_HOST=$(url_part "$PG_URL_IN" host); PG_PORT=$(url_part "$PG_URL_IN" port); PG_PORT=${PG_PORT:-5432}
  PG_DB="${PG_DB:-$(url_part "$PG_URL_IN" db)}"
fi
if [ -n "${CH_URL_IN:-}" ]; then
  CH_EXT=1; CH_USER=$(url_part "$CH_URL_IN" user); CH_PASS=$(url_part "$CH_URL_IN" pass)
  CH_HOST=$(url_part "$CH_URL_IN" host); CH_PORT=$(url_part "$CH_URL_IN" port); CH_PORT=${CH_PORT:-8123}
fi
PG_HOST="${PG_HOST:-127.0.0.1}"; PG_PORT="${PG_PORT:-15432}"; PG_USER="${PG_USER:-elint}"; PG_PASS="${PG_PASS:-elint}"; PG_DB="${PG_DB:-sigint}"
CH_HOST="${CH_HOST:-127.0.0.1}"; CH_PORT="${CH_PORT:-18123}"; CH_NPORT="${CH_NPORT:-19000}"; CH_USER="${CH_USER:-elint}"; CH_PASS="${CH_PASS:-elint}"
PG_EXT="${PG_EXT:-0}"; CH_EXT="${CH_EXT:-0}"

export PG_URL="postgresql://$PG_USER:$PG_PASS@$PG_HOST:$PG_PORT/$PG_DB"
export CH_URL="http://$CH_HOST:$CH_PORT" CLICKHOUSE_DB=signals CLICKHOUSE_USER="$CH_USER" CLICKHOUSE_PASSWORD="$CH_PASS"
export ONNX_URL="http://127.0.0.1:$MODEL_PORT"
BIN="$R/target/release/ai_processor"; API_BIN="$R/target/release/refresh_api"

if [ -z "${DEVICE:-}" ]; then [ -e /dev/kfd ] && [ -f "$R/onnx_model_service/.gpu" ] && DEVICE=hybrid || DEVICE=cpu; fi

say() { printf '\n== %s\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null || die "'$1' not found ($2)"; }
alive() { [ -f "$RUN/$1.pid" ] && kill -0 "$(cat "$RUN/$1.pid")" 2>/dev/null; }
bg() {  # bg <name> <cmd...>: start detached (Linux + macOS), log to .run/logs/<name>.log
  local name=$1; shift
  if command -v setsid >/dev/null; then setsid nohup "$@" > "$RUN/logs/$name.log" 2>&1 < /dev/null &
  else nohup "$@" > "$RUN/logs/$name.log" 2>&1 < /dev/null & fi
  echo $! > "$RUN/$name.pid"
}
my_ip() { hostname -I 2>/dev/null | awk '{print $1}' || true; }
ip_or_local() { local ip; ip=$(my_ip); [ -n "$ip" ] || ip=$(ipconfig getifaddr en0 2>/dev/null || true); echo "${ip:-127.0.0.1}"; }

pgq() {  # psql against the chosen database: local container, host psql, or a client container
  local db="${PGQ_DB:-$PG_DB}"
  if [ "$PG_EXT" = 0 ]; then docker exec -i -e PGPASSWORD="$PG_PASS" "$PG_NAME" psql -U "$PG_USER" -d "$db" "$@"
  elif command -v psql >/dev/null; then PGPASSWORD="$PG_PASS" psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$db" "$@"
  else docker run --rm -i --network host -e PGPASSWORD="$PG_PASS" postgres:16 psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$db" "$@"; fi
}

# ---------- parts ----------
build() {
  if [ -x "$BIN" ] && [ -x "$API_BIN" ] && [ -z "${REBUILD:-}" ] && [ "${1:-}" != force ]; then echo "binaries present (use --rebuild to rebuild)"; return; fi
  need cargo "install Rust: https://rustup.rs"
  say "build ai_processor + classic_eval + refresh_api (release)"
  local flags="${RUSTFLAGS:-}"; [ -z "$flags" ] && [ "$(uname -m)" = x86_64 ] && flags="-C target-cpu=native"
  (cd "$R" && RUSTFLAGS="$flags" cargo build --release -q -p ai_processor --bins 2>&1 | grep -E "^error" || true)
  [ -x "$BIN" ] || die "build failed (run: cargo build --release -p ai_processor)"
}

venvs() {
  mkdir -p "$(dirname "$MODEL_FILE")" "$DATA/pdw_generator/known_emitters" "$DATA/pdw_generator/unknown_emitters"
  need python3 "python3 with venv"
  if [ -z "${NO_MODEL:-}" ] && [ ! -x "$R/onnx_model_service/.venv/bin/python" ]; then
    say "model service venv"
    if [ -e /dev/kfd ] && [ "${GPU_SETUP:-1}" = 1 ]; then
      (cd "$R/onnx_model_service" && ./setup.sh) && touch "$R/onnx_model_service/.gpu" && DEVICE=hybrid   # ROCm (box GPU)
    else
      python3 -m venv "$R/onnx_model_service/.venv" && "$R/onnx_model_service/.venv/bin/pip" install -q numpy==1.26.4 onnxruntime
    fi
  fi
  if [ -z "${NO_UI:-}" ] && [ ! -x "$R/elint_ui/.venv/bin/python" ]; then
    say "UI venv"; python3 -m venv "$R/elint_ui/.venv" && "$R/elint_ui/.venv/bin/pip" install -q psycopg2-binary
  fi
}

db() {
  if [ "$PG_EXT" = 0 ] || [ "$CH_EXT" = 0 ]; then need docker "docker runs the local databases (or give --pg-url and --ch-url)"; fi
  if [ "$PG_EXT" = 0 ]; then
    say "Postgres: local docker $PG_NAME :$PG_PORT (reused if it exists, never deleted)"
    if docker inspect "$PG_NAME" >/dev/null 2>&1; then docker start "$PG_NAME" >/dev/null; else
      docker run -d --name "$PG_NAME" --restart unless-stopped -p 127.0.0.1:$PG_PORT:5432 \
        -e POSTGRES_USER="$PG_USER" -e POSTGRES_PASSWORD="$PG_PASS" postgres:16 >/dev/null; fi
  else say "Postgres: $PG_HOST:$PG_PORT/$PG_DB (yours, nothing started)"; fi
  if [ "$CH_EXT" = 0 ]; then
    say "ClickHouse: local docker $CH_NAME :$CH_PORT (reused if it exists, never deleted)"
    if docker inspect "$CH_NAME" >/dev/null 2>&1; then docker start "$CH_NAME" >/dev/null; else
      docker run -d --name "$CH_NAME" --restart unless-stopped -p 127.0.0.1:$CH_PORT:8123 -p 127.0.0.1:$CH_NPORT:9000 \
        -e CLICKHOUSE_USER="$CH_USER" -e CLICKHOUSE_PASSWORD="$CH_PASS" -e CLICKHOUSE_DB=signals -e CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1 \
        -v "$R/clickhouse/users.d/00_perf.xml":/etc/clickhouse-server/config.d/00_perf.xml \
        -v "$R/clickhouse/users_profiles/old-analyzer.xml":/etc/clickhouse-server/users.d/old-analyzer.xml \
        clickhouse/clickhouse-server:24.8 >/dev/null; fi
  else say "ClickHouse: $CH_URL (yours, nothing started)"; fi
  local ok=0
  for i in $(seq 90); do
    PGQ_DB=postgres pgq -tAc "select 1" >/dev/null 2>&1 && curl -sf "$CH_URL/ping" >/dev/null && { ok=1; break; }; sleep 1
  done
  [ $ok = 1 ] || die "databases not reachable (Postgres $PG_HOST:$PG_PORT, ClickHouse $CH_URL)"
  # ClickHouse tables: CREATE ... IF NOT EXISTS (existing tables and data kept)
  local stmt=""
  while IFS= read -r line; do
    case "$line" in --*) continue ;; ";") [ -n "$stmt" ] && { curl -sf -u "$CH_USER:$CH_PASS" "$CH_URL/" --data-binary "$stmt" >/dev/null || die "ClickHouse schema failed: ${stmt:0:60}"; }; stmt="" ;; *) stmt+="$line"$'\n' ;; esac
  done < "$R/clickhouse/engine_schema.sql"
  # Postgres database: created with the engine schema if missing
  if [ "$(PGQ_DB=postgres pgq -tAc "select 1 from pg_database where datname='$PG_DB'")" != 1 ]; then
    say "create Postgres database $PG_DB + schema"
    PGQ_DB=postgres pgq -qc "create database \"$PG_DB\"" >/dev/null
    pgq -q -v ON_ERROR_STOP=1 < "$R/postgres/engine_schema.sql" >/dev/null
  elif [ "$(pgq -tAc "select data_type from information_schema.columns where table_name='emitters' and column_name='id'")" = integer ]; then
    echo "WARNING: database $PG_DB has the old schema (integer ids); the engine fails on it. Use --pg-db <new name>."
  fi
  echo "ready: $PG_URL , $CH_URL"
}

model() {
  [ -f "$MODEL_FILE" ] || die "model file missing: $MODEL_FILE (copy it there, or --model FILE)"
  [ -x "$R/onnx_model_service/.venv/bin/python" ] || venvs
  if alive model; then MODEL_PORT=$(cat "$RUN/model.port" 2>/dev/null || echo "$MODEL_PORT"); export ONNX_URL="http://127.0.0.1:$MODEL_PORT"; echo "model service already running (pid $(cat "$RUN/model.pid"), port $MODEL_PORT)"; return; fi
  curl -sf "$ONNX_URL/health" >/dev/null && die "port $MODEL_PORT already has a model service not started by run.sh (stop it, or --model-port N)"
  say "ONNX model service :$MODEL_PORT ($DEVICE)"
  (cd "$R/onnx_model_service" && export MODEL_PATH="$MODEL_FILE" HOST=127.0.0.1 PORT=$MODEL_PORT DEVICE=$DEVICE \
     CPU_WORKERS=${CPU_WORKERS:-2} CPU_THREADS=${CPU_THREADS:-6} HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-11.0.0} \
     && bg model .venv/bin/python server.py); echo "$MODEL_PORT" > "$RUN/model.port"
  for i in $(seq 180); do curl -sf "$ONNX_URL/health" >/dev/null && { echo "model service ready"; return; }; alive model || break; sleep 1; done
  tail -5 "$RUN/logs/model.log"; die "model service did not start (./run.sh logs model)"
}

ui() {
  [ -x "$BIN" ] || build
  [ -x "$R/elint_ui/.venv/bin/python" ] || venvs
  if alive ui; then UI_PORT=$(cat "$RUN/ui.port" 2>/dev/null || echo "$UI_PORT"); echo "UI already running (pid $(cat "$RUN/ui.pid"), port $UI_PORT)"; return; fi
  curl -sf "http://127.0.0.1:$UI_PORT/" >/dev/null && die "port $UI_PORT is already in use (stop that process, or --ui-port N)"
  say "UI :$UI_PORT"
  (cd "$R/elint_ui" && export AI_PROCESSOR="$BIN" DATA_DIR="$DATA" UI_PORT=$UI_PORT && bg ui .venv/bin/python server.py); echo "$UI_PORT" > "$RUN/ui.port"
  for i in $(seq 30); do curl -sf "http://127.0.0.1:$UI_PORT/" >/dev/null && { echo "UI ready"; return; }; alive ui || break; sleep 1; done
  tail -5 "$RUN/logs/ui.log"; die "UI did not start (./run.sh logs ui)"
}

api() {   # refresh_api: POST /api/refresh-data {"csv_filename": "..."}  (engine: CLASSIC=1 classic v3, else ONNX)
  [ -x "$API_BIN" ] || build force
  if alive api; then API_PORT=$(cat "$RUN/api.port" 2>/dev/null || echo "$API_PORT"); echo "API already running (pid $(cat "$RUN/api.pid"), port $API_PORT)"; return; fi
  say "refresh API :$API_PORT (engine: $([ "${CLASSIC:-1}" = 1 ] && echo classic v3 || echo onnx))"
  (cd "$R" && export REFRESH_API_PORT=$API_PORT CLASSIC=${CLASSIC:-1} EMBED=$([ "${CLASSIC:-1}" = 1 ] && echo classic || echo onnx) \
     ELINT_LOG_DIR="$RUN/logs/api" && bg api "$API_BIN"); echo "$API_PORT" > "$RUN/api.port"
  for i in $(seq 30); do curl -sf "http://127.0.0.1:$API_PORT/health" >/dev/null && { echo "API ready"; return; }; alive api || break; sleep 1; done
  tail -5 "$RUN/logs/api.log"; die "API did not start (./run.sh logs api)"
}

all() {
  build; venvs; db
  [ -n "${NO_MODEL:-}" ] || model
  [ -n "${NO_UI:-}" ] || ui
  [ -n "${NO_API:-}" ] || api
  local ip; ip=$(ip_or_local)
  say "running"
  [ -n "${NO_UI:-}" ] || echo "UI:            http://$ip:$UI_PORT"
  [ -n "${NO_MODEL:-}" ] || echo "model service: $ONNX_URL/health"
  [ -n "${NO_API:-}" ] || echo "refresh API:   http://127.0.0.1:$API_PORT  (POST /api/refresh-data {\"csv_filename\": \"...\"})"
  echo "database:      $PG_URL , $CH_URL"
  echo "one file:      ./run.sh run <file.csv> classic|v2|onnx      stop: ./run.sh stop"
}

engine_env() {
  case "$1" in
    classic) echo "CLASSIC=1 EMBED=classic CLASSIC_V3=1" ;;
    v2)      echo "CLASSIC=1 EMBED=classic CLASSIC_V3=0" ;;
    onnx)    echo "CLASSIC=0" ;;
    *) die "engine must be classic, v2 or onnx" ;;
  esac
}

run_one() {
  local f="${1:-}" e="${2:-classic}"; [ -f "$f" ] || die "usage: ./run.sh run <file.csv> [classic|v2|onnx]"
  [ -x "$BIN" ] || build
  [ "$e" = onnx ] && ! curl -sf "$ONNX_URL/health" >/dev/null && die "model service not running (./run.sh)"
  say "ai_processor $e on $(basename "$f")  (database $PG_DB)"
  local log="$RUN/logs/run_last.log"
  env $(engine_env "$e") ELINT_LOG_DIR="$RUN/logs" "$BIN" --csv "$f" > "$log" 2>&1 || { tail -20 "$log"; die "ai_processor failed (full log $log)"; }
  grep -oE "LAYER\|(engine_total|load_csv|wipe_tables)\|[0-9.]+" "$log" | sed 's/LAYER|/  ms /' || true
  local res; res="$RUN/results/$(date +%Y%m%d_%H%M%S)_${e}_$(basename "$f" .csv).txt"
  pgq -c "select name, count(*) as emitters from emitters group by name order by name" | tee "$res"
  echo "saved: $res"
}

status() {
  for s in model ui api; do alive $s && echo "$s: running (pid $(cat "$RUN/$s.pid"))" || echo "$s: stopped"; done
  [ "$PG_EXT$CH_EXT" != 11 ] && command -v docker >/dev/null && docker ps -a --filter "name=^($PG_NAME|$CH_NAME)\$" --format '{{.Names}}: {{.Status}}' || true
  curl -sf "$ONNX_URL/health" >/dev/null && echo "model health: ok" || echo "model health: down"
  echo "database: $PG_URL , $CH_URL"
}

stop() {  # model service, UI, API; databases keep running unless 'stop db' (data always kept)
  for s in api ui model; do
    if alive $s; then local p; p=$(cat "$RUN/$s.pid"); kill -- -"$p" 2>/dev/null || kill "$p"; echo "$s stopped"; fi
    rm -f "$RUN/$s.pid" "$RUN/$s.port"
  done
  if [ "${1:-}" = db ]; then
    [ "$PG_EXT" = 0 ] && docker stop "$PG_NAME" >/dev/null && echo "Postgres stopped (data kept)"
    [ "$CH_EXT" = 0 ] && docker stop "$CH_NAME" >/dev/null && echo "ClickHouse stopped (data kept)"
  fi
  return 0
}

clean() {
  [ "${1:-}" = --yes ] || die "this deletes the local docker DB containers $PG_NAME/$CH_NAME and all their data; run: ./run.sh clean --yes"
  stop; [ "$PG_EXT" = 0 ] && docker rm -f "$PG_NAME" >/dev/null 2>&1 && echo "removed $PG_NAME"
  [ "$CH_EXT" = 0 ] && docker rm -f "$CH_NAME" >/dev/null 2>&1 && echo "removed $CH_NAME"; return 0
}

case "$CMD" in
  all|start) all ;;
  build) REBUILD=1; build ;;
  setup) build; venvs ;;
  db) db ;;
  model) model ;;
  ui) ui ;;
  api) api ;;
  run) run_one "${ARGS[@]:-}" ;;
  selftest) DATA="$DATA" "$R/tools/classic/selftest.sh" "${ARGS[@]:-}" ;;
  status) status ;;
  stop) stop "${ARGS[@]:-}" ;;
  clean) clean "${ARGS[@]:-}" ;;
  logs) tail -f "$RUN/logs/${ARGS[0]:-model}.log" ;;
  help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) die "unknown command '$CMD' (./run.sh help)" ;;
esac
