#!/usr/bin/env bash
# run_rust_tests.sh — Run 10 test CSVs through the Rust ai_processor engine
# Usage:
#   bash run_rust_tests.sh            # run all 10 files sequentially
#   bash run_rust_tests.sh --parallel # run in parallel (xargs -P 4)
#
# Prerequisites:
#   1. Build the Rust engine:
#      cd ai_elint && RUSTFLAGS="-C target-feature=+avx512f" cargo +nightly-2025-09-01 build --release
#   2. Set environment variables (or load .env):
#      export AI_ENGINE_BINARY=/path/to/ai_processor
#      export PG_URL="postgresql://elint:elint@127.0.0.1:15432/sigint"
#      export CH_URL="http://127.0.0.1:18123"
#      export ONNX_PATH=/path/to/transformer_model_known_v16.onnx
#   3. Ensure PostgreSQL and ClickHouse containers are running.
#
# Each run is fully independent: wipes DBs, loads CSV, runs pipeline, exits.

set -euo pipefail

# --- Load .env if present ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    echo "Loading environment from .env"
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# --- Defaults ---
AI_ENGINE_BINARY="${AI_ENGINE_BINARY:-./target/release/ai_processor}"
PG_URL="${PG_URL:-postgresql://elint:elint@127.0.0.1:15432/sigint}"
CH_URL="${CH_URL:-http://127.0.0.1:18123}"
ONNX_PATH="${ONNX_PATH:-./models/transformer_model_known_v16.onnx}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-tests/data}"

# --- 10 test files (2 known + 8 unknown emitters) ---
FILES=(
    "${TEST_DATA_ROOT}/input_csv_files/pdw_generator/known_emitters/mf04_psr5m1_best-0_plus2.csv"
    "${TEST_DATA_ROOT}/input_csv_files/pdw_generator/known_emitters/mf04_psr1_best-3_plus3.csv"
    "${TEST_DATA_ROOT}/input_csv_files/pdw_generator/unknown_emitters/cta1_cta2_best-0_plus0.csv"
    "${TEST_DATA_ROOT}/input_csv_files/pdw_generator/unknown_emitters/cta1_cta2_best-0_plus2.csv"
    "${TEST_DATA_ROOT}/input_csv_files/pdw_generator/unknown_emitters/cta1_cta2_best-0_plus3.csv"
    "${TEST_DATA_ROOT}/input_csv_files/pdw_generator/unknown_emitters/cta1_cta2_best-0_plus5.csv"
    "${TEST_DATA_ROOT}/input_csv_files/pdw_generator/unknown_emitters/cta1_cta2_best-10_plus0.csv"
    "${TEST_DATA_ROOT}/input_csv_files/pdw_generator/unknown_emitters/cta1_cta2_best-10_plus2.csv"
    "${TEST_DATA_ROOT}/input_csv_files/pdw_generator/unknown_emitters/cta1_cta2_best-10_plus3.csv"
    "${TEST_DATA_ROOT}/input_csv_files/pdw_generator/unknown_emitters/cta1_cta2_best-10_plus5.csv"
)

# --- Pre-flight checks ---
echo "=== AI ELINT Rust Test Runner ==="
echo "Engine:   $AI_ENGINE_BINARY"
echo "PG:       $PG_URL"
echo "CH:       $CH_URL"
echo "ONNX:     $ONNX_PATH"
echo "Files:    ${#FILES[@]}"
echo ""

if [[ ! -x "$AI_ENGINE_BINARY" ]]; then
    echo "ERROR: Engine binary not found or not executable: $AI_ENGINE_BINARY"
    echo "Build it first: cd ai_elint && cargo +nightly-2025-09-01 build --release"
    exit 1
fi

# --- Wipe DBs helper (called before each file) ---
wipe_dbs() {
    local pg_host="${PG_URL#postgresql://}"
    local pg_host="${pg_host%%/*}"
    local pg_db="${PG_URL##*/}"

    # PG: truncate all tables in sigint
    psql "$PG_URL" -c "TRUNCATE emitters, signals, clusters, signals_clusters, signatures CASCADE" 2>/dev/null || \
        docker exec elint_pg psql -U elint -d "$pg_db" -c "TRUNCATE emitters, signals, clusters, signals_clusters, signatures CASCADE"

    # CH: truncate tables in signals database
    curl -s -X POST "${CH_URL}/" -d "TRUNCATE TABLE IF EXISTS signals.pdw" 2>/dev/null || \
        docker exec elint_ch clickhouse-client --database signals --query "TRUNCATE TABLE IF EXISTS pdw"
    curl -s -X POST "${CH_URL}/" -d "TRUNCATE TABLE IF EXISTS signals.clusters" 2>/dev/null || \
        docker exec elint_ch clickhouse-client --database signals --query "TRUNCATE TABLE IF EXISTS clusters"
    curl -s -X POST "${CH_URL}/" -d "TRUNCATE TABLE IF EXISTS signals.csv_input" 2>/dev/null || \
        docker exec elint_ch clickhouse-client --database signals --query "TRUNCATE TABLE IF EXISTS csv_input"
}

# --- Query counts helper ---
get_counts() {
    local pg_host="${PG_URL#postgresql://}"
    local pg_host="${pg_host%%/*}"
    local pg_db="${PG_URL##*/}"

    local emitters=$(docker exec elint_pg psql -U elint -d "$pg_db" -t -A -c "SELECT count(*) FROM emitters" 2>/dev/null || echo "?")
    local signals=$(docker exec elint_pg psql -U elint -d "$pg_db" -t -A -c "SELECT count(*) FROM signals" 2>/dev/null || echo "?")
    local clusters=$(docker exec elint_pg psql -U elint -d "$pg_db" -t -A -c "SELECT count(*) FROM clusters" 2>/dev/null || echo "?")
    local ch_pdw=$(docker exec elint_ch clickhouse-client --database signals --query "SELECT count(*) FROM pdw" 2>/dev/null || echo "?")
    local ch_clusters=$(docker exec elint_ch clickhouse-client --database signals --query "SELECT count(*) FROM clusters" 2>/dev/null || echo "?")

    echo "emitters=$emitters signals=$signals clusters=$clusters ch_pdw=$ch_pdw ch_clusters=$ch_clusters"
}

# --- Run one file ---
run_one() {
    local csv_file="$1"
    local name=$(basename "$csv_file")
    echo ""
    echo "--- Running: $name ---"

    if [[ ! -f "$csv_file" ]]; then
        echo "  SKIP: file not found"
        return 1
    fi

    wipe_dbs
    local start=$(date +%s)
    "$AI_ENGINE_BINARY" --csv "$csv_file" 2>&1 | tail -5
    local rc=${PIPESTATUS[0]}
    local end=$(date +%s)
    local elapsed=$((end - start))

    local counts=$(get_counts)
    local rc_str="PASS"
    [[ $rc -ne 0 ]] && rc_str="FAIL(rc=$rc)"

    printf "  %s  %ds  %s\n" "$rc_str" "$elapsed" "$counts"
    return $rc
}

# --- Main ---
PARALLEL=false
[[ "${1:-}" == "--parallel" ]] && PARALLEL=true

declare -a RESULTS=()
PASS_COUNT=0
FAIL_COUNT=0

if $PARALLEL; then
    echo "Running in parallel (4 at a time)..."
    printf '%s\n' "${FILES[@]}" | xargs -P 4 -I {} bash -c \
        'wipe_dbs() { :; }; source <(sed -n "/^wipe_dbs/,/^}/p" "$0") 2>/dev/null; run_one "$@"' \
        _ run_one {} 2>&1 | tee /tmp/rust_test_results.txt
    # Fallback: just run sequentially if parallel setup is complex
    echo "NOTE: Parallel mode requires DB isolation. Running sequentially for safety."
fi

# Sequential run (default and fallback)
for f in "${FILES[@]}"; do
    if run_one "$f"; then
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo ""
echo "=== RESULTS SUMMARY ==="
printf "%-50s %6s %10s %10s %10s %10s %10s\n" "File" "RC" "Emitters" "Signals" "Clusters" "CH_pdw" "CH_clust"
echo "-----------------------------------------------------------------------------------------------------------------"
for r in "${RESULTS[@]}"; do
    echo "$r"
done
echo "-----------------------------------------------------------------------------------------------------------------"
echo "Total: ${#FILES[@]}  Passed: $PASS_COUNT  Failed: $FAIL_COUNT"
