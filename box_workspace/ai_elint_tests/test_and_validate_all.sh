#!/bin/bash
# =============================================================================
# test_and_validate_all.sh — Test AND validate ALL CSV files via Rust engine.
#
# This is the ONE script to run. It:
#   1. Wipes DBs before each file (clean state)
#   2. Runs ai_processor --csv for every CSV (known + unknown)
#   3. Collects PG/CH counts after each run
#   4. Validates each file against expectations.json
#   5. Prints a final PASS/FAIL summary
#
# Usage (on the VM):
#   bash test_and_validate_all.sh              # run all files
#   bash test_and_validate_all.sh --limit 10   # run first 10 files only
#   bash test_and_validate_all.sh --known      # known emitters only
#   bash test_and_validate_all.sh --unknown    # unknown emitters only
#
# Output files (on /tmp):
#   /tmp/tv_all_results.txt   — one RESULT| line per file
#   /tmp/tv_all.log           — progress log
#   /tmp/tv_all_summary.txt   — final summary with pass/fail breakdown
# =============================================================================

set -u

# ── Config (override with env vars if needed) ─────────────────────────────────
CSV_BASE="${CSV_BASE:-/home/sysadm/ai_elint_tests_work/tests/data/input_csv_files/pdw_generator}"
RESULTS_FILE="/tmp/tv_all_results.txt"
LOG_FILE="/tmp/tv_all.log"
SUMMARY_FILE="/tmp/tv_all_summary.txt"
EXPECTED_FILE="${EXPECTED_FILE:-/home/sysadm/ai_elint_tests_work/tests/test_data/expected_results.json}"
TIMEOUT="${TIMEOUT:-600}"

# ── Parse args ────────────────────────────────────────────────────────────────
LIMIT=0
FILTER=""
for arg in "$@"; do
    case "$arg" in
        --limit)  LIMIT=$2; shift ;;
        --known)  FILTER="known" ;;
        --unknown) FILTER="unknown" ;;
    esac
done

# ── Build file list ───────────────────────────────────────────────────────────
if [ "$FILTER" = "known" ]; then
    CSV_DIR="$CSV_BASE/known_emitters"
elif [ "$FILTER" = "unknown" ]; then
    CSV_DIR="$CSV_BASE/unknown_emitters"
else
    CSV_DIR="$CSV_BASE"
fi

mapfile -t FILES < <(find "$CSV_DIR" -name '*.csv' | sort)
if [ "$LIMIT" -gt 0 ]; then
    FILES=("${FILES[@]:0:$LIMIT}")
fi
TOTAL=${#FILES[@]}

# ── Init output files ─────────────────────────────────────────────────────────
: > "$RESULTS_FILE"
: > "$LOG_FILE"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== TEST & VALIDATE ALL: $TOTAL files ==="
log "CSV dir: $CSV_DIR"
log "Started: $(date)"

# ── Pre-check: DB containers must be running ──────────────────────────────────
if ! docker ps --format '{{.Names}}' | grep -q 'elint_pg'; then
    log "ERROR: elint_pg container not running. Start it first."
    exit 1
fi
if ! docker ps --format '{{.Names}}' | grep -q 'elint_ch'; then
    log "ERROR: elint_ch container not running. Start it first."
    exit 1
fi

# ── Engine environment (required by ai_processor) ─────────────────────────────
export PG_URL="${PG_URL:-postgresql://elint:elint@127.0.0.1:15432/sigint}"
export CH_URL="${CH_URL:-http://127.0.0.1:18123}"
export CLICKHOUSE_DB="${CLICKHOUSE_DB:-signals}"
export CLICKHOUSE_USER="${CLICKHOUSE_USER:-elint}"
export CLICKHOUSE_PASSWORD="${CLICKHOUSE_PASSWORD:-elint}"
export ONNX_PATH="${ONNX_PATH:-/home/sysadm/models/transformer_model_known_v16.onnx}"
export CLASSIC="${CLASSIC:-1}"
export ONNX_SESSION_COUNT="${ONNX_SESSION_COUNT:-1}"
export ONNX_THREADS="${ONNX_THREADS:-1}"

# ── Main loop: test each file ─────────────────────────────────────────────────
PASS=0
FAIL=0
SKIP=0
FAILED_FILES=""

for i in "${!FILES[@]}"; do
    f="${FILES[$i]}"
    num=$((i + 1))
    name=$(basename "$f")

    log "[$num/$TOTAL] $name"

    # ── Wipe DBs (clean state for this file) ──
    docker exec elint_pg psql -U elint -d sigint -q -c \
        "TRUNCATE clusters, signals, emitters, signals_clusters RESTART IDENTITY CASCADE" 2>/dev/null
    docker exec elint_ch clickhouse-client --user elint --password elint -q \
        "TRUNCATE TABLE signals.pdw; TRUNCATE TABLE signals.clusters; TRUNCATE TABLE signals.csv_input;" 2>/dev/null

    # ── Run engine ──
    start_ts=$(date +%s)
    timeout "$TIMEOUT" /home/sysadm/ai_elint_rust/target/release/ai_processor \
        --csv "$f" >> "$LOG_FILE" 2>&1
    rc=$?
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))

    # ── Collect counts ──
    pg_em=$(docker exec elint_pg psql -U elint -d sigint -t -A -c "SELECT count(*) FROM emitters" 2>/dev/null | tr -d '[:space:]')
    pg_sig=$(docker exec elint_pg psql -U elint -d sigint -t -A -c "SELECT count(*) FROM signals" 2>/dev/null | tr -d '[:space:]')
    pg_clu=$(docker exec elint_pg psql -U elint -d sigint -t -A -c "SELECT count(*) FROM clusters" 2>/dev/null | tr -d '[:space:]')
    ch_pdw=$(docker exec elint_ch clickhouse-client --user elint --password elint \
        -q "SELECT count() FROM signals.pdw" 2>/dev/null | tr -d '[:space:]')
    ch_clu=$(docker exec elint_ch clickhouse-client --user elint --password elint \
        -q "SELECT count() FROM signals.clusters" 2>/dev/null | tr -d '[:space:]')

    # ── Write result line ──
    echo "RESULT|$name|rc=$rc|t=${elapsed}s|pg_em=${pg_em:-0}|pg_sig=${pg_sig:-0}|pg_clu=${pg_clu:-0}|ch_pdw=${ch_pdw:-0}|ch_clu=${ch_clu:-0}" \
        >> "$RESULTS_FILE"

    # ── Validate against expectations ──
    if [ $rc -ne 0 ]; then
        FAIL=$((FAIL + 1))
        STATUS="FAIL"
        FAILED_FILES="$FAILED_FILES\n  $name (rc=$rc)"
    elif [ -f "$EXPECTED_FILE" ]; then
        # Extract expected emitter count from expectations.json
        exp_em=$(python3 -c "
import json,sys
try:
    d=json.load(open('$EXPECTED_FILE'))
    key='$name'
    if key in d: print(d[key].get('exp_em',''))
    else: print('')
except: print('')
" 2>/dev/null)

        if [ -n "$exp_em" ] && [ "$pg_em" = "$exp_em" ]; then
            PASS=$((PASS + 1))
            STATUS="PASS"
        else
            FAIL=$((FAIL + 1))
            STATUS="FAIL"
            FAILED_FILES="$FAILED_FILES\n  $name (pg_em=${pg_em:-0} expected=${exp_em:-?})"
        fi
    else
        # No expectations file — pass if rc=0
        PASS=$((PASS + 1))
        STATUS="PASS"
    fi

    log "  → $STATUS (rc=$rc, em=$pg_em, ${elapsed}s)"
done

# ── Final summary ─────────────────────────────────────────────────────────────
TOTAL_DONE=$((PASS + FAIL))
PCT=0
if [ $TOTAL_DONE -gt 0 ]; then
    PCT=$((PASS * 100 / TOTAL_DONE))
fi

{
    echo "=============================================="
    echo " TEST & VALIDATE SUMMARY — $(date)"
    echo "=============================================="
    echo " Total files:  $TOTAL_DONE"
    echo " PASSED:       $PASS  (${PCT}%)"
    echo " FAILED:       $FAIL"
    echo "=============================================="
    if [ -n "$FAILED_FILES" ]; then
        echo ""
        echo "Failed files:"
        echo -e "$FAILED_FILES"
    fi
    echo ""
    echo "Results: $RESULTS_FILE"
    echo "Log:     $LOG_FILE"
    echo ""
    echo "Raw result lines:"
    cat "$RESULTS_FILE"
} | tee "$SUMMARY_FILE"

exit $FAIL
