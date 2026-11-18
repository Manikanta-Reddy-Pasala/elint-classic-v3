#!/usr/bin/env bash
# clippy-gate.sh — Rust clippy gate for CI / local runs.
#
# cargo clippy requires the clippy component, which is only installable via
# rustup. This box (and the sandbox image) ships a plain apt/source rustc
# 1.75 with no rustup, so `cargo clippy` reports "no such command".
#
# Policy: if clippy is present, run it and fail on lints. If it is absent,
# skip with a clear notice (warning only) instead of failing the build —
# the toolchain simply cannot run it. This keeps the gate meaningful
# everywhere clippy IS available (CI, dev machines with rustup) while
# staying green here.
set -euo pipefail

cd "$(dirname "$0")/.."

# apt's rust-1.76-clippy ships cargo-clippy under /usr/lib/rust-1.76/bin, not on PATH.
if ! cargo clippy --version >/dev/null 2>&1; then
  CLIPPY_BIN_DIR=$(ls -d /usr/lib/rust-*/bin 2>/dev/null | while read -r d; do [ -x "$d/cargo-clippy" ] && echo "$d" && break; done)
  [ -n "$CLIPPY_BIN_DIR" ] && PATH="$CLIPPY_BIN_DIR:$PATH"
fi

if ! cargo clippy --version >/dev/null 2>&1; then
  echo "clippy: not installed on this toolchain (needs rustup component) — SKIPPED"
  echo "clippy: install with 'rustup component add clippy' to enable the gate"
  exit 0
fi

cargo clippy --all-targets -- -D warnings
echo "clippy: clean"
