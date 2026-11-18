#!/usr/bin/env bash
# Start the refresh_api server.
# Usage: bash run.sh [path-to-binary]
#   defaults to ./bin/refresh_api, then ./target/release/refresh_api
# All settings below have defaults for the lab VM — override any of them
# by exporting the same variable before calling this script, e.g.:
#   ONNX_SESSION_COUNT=4 REFRESH_API_PORT=9000 bash run.sh
set -euo pipefail
cd "$(dirname "$0")"

# --- settings (defaults for the lab VM; edit or export to change) ---
export PG_URL="${PG_URL:-postgres://elint:elint@127.0.0.1:5432/sigint}"
export CH_URL="${CH_URL:-http://127.0.0.1:8123}"
export CLICKHOUSE_USER="${CLICKHOUSE_USER:-elint}"            # required for CH writes (keep-alive socket)
export CLICKHOUSE_PASSWORD="${CLICKHOUSE_PASSWORD:-elint}"
export ONNX_PATH="${ONNX_PATH:-$PWD/model/transformer_model_known_v16_int8.onnx}"
export ORT_LIB_PATH="${ORT_LIB_PATH:-/opt/ai_elint/onnxruntime/libonnxruntime.so}"
# Measured sweet spot on the lab APU (526,873-pulse run, INT8 v16 model):
# 2 sessions x 6 threads = 38.5s onnx vs 43.8s for 1 session x 12 threads
# (-12% onnx, -11% total: 42.5s vs 47.8s). 1x24 and 4x6 are both slower.
export ONNX_SESSION_COUNT="${ONNX_SESSION_COUNT:-2}"
export ONNX_THREADS="${ONNX_THREADS:-6}"
export REFRESH_API_PORT="${REFRESH_API_PORT:-8788}"
export ELINT_LOG_DIR="${ELINT_LOG_DIR:-$PWD/logs}"
export RUST_LOG="${RUST_LOG:-info}"
# ---------------------------------------------------------

# --- auto build (see below) ------------------------------------------
BIN="${1:-}"
if [ -z "$BIN" ]; then
  for candidate in ./bin/refresh_api ./target/release/refresh_api; do
    if [ -x "$candidate" ]; then BIN="$candidate"; break; fi
  done
fi

# --- auto build -------------------------------------------------------
# If no usable binary was found, build it in place before starting.
# Skipped entirely when a binary is present (so re-runs stay fast).
#
# Override CARGO_TOOLCHAIN to change the toolchain (e.g. "stable", or
# a full rustup toolchain name).  The README default is
# nightly-2025-09-01 (AVX-512 intrinsics are unstable on stable).
if [ -z "${BIN:-}" ] || [ ! -x "$BIN" ]; then
  echo "refresh_api binary not found — building from source (this can take a while) ..." >&2
  export CARGO_REGISTRIES_CRATES_IO_PROTOCOL="${CARGO_REGISTRIES_CRATES_IO_PROTOCOL:-sparse}"
  CARGO_TOOLCHAIN="${CARGO_TOOLCHAIN:-nightly-2025-09-01}"
  if command -v cargo >/dev/null 2>&1; then
    # Use the toolchain override only when it is already installed
    # (avoid auto-downloading a toolchain the user may not want).
    if [ "$CARGO_TOOLCHAIN" = "stable" ]; then
      RUSTFLAGS="${RUSTFLAGS:--C target-feature=+avx512f}" \
        cargo build --release -p ai_processor --bin refresh_api
    else
      RUSTFLAGS="${RUSTFLAGS:--C target-feature=+avx512f}" \
        cargo "+$CARGO_TOOLCHAIN" build --release -p ai_processor --bin refresh_api
    fi
  else
    echo "cargo not found in PATH — cannot auto-build." >&2
    echo "Install the Rust toolchain, or build in Docker (README, section 3)," >&2
    echo "or extract the binary from the builder image (README, section 2)," >&2
    echo "then re-run: bash run.sh [path-to-binary]" >&2
    exit 1
  fi
fi

# After a build (or if one was already present), re-locate the binary.
if [ -z "${BIN:-}" ] || [ ! -x "$BIN" ]; then
  for candidate in ./target/release/refresh_api ./bin/refresh_api; do
    if [ -x "$candidate" ]; then BIN="$candidate"; break; fi
  done
fi

if [ -z "${BIN:-}" ] || [ ! -x "$BIN" ]; then
  echo "refresh_api binary not found after build attempt." >&2
  echo "Build it (README, 'Build & run from source') or extract it from the" >&2
  echo "builder image (README, 'Run using a pre-built binary'), then re-run:" >&2
  echo "  bash run.sh [path-to-binary]" >&2
  exit 1
fi

exec "$BIN"
