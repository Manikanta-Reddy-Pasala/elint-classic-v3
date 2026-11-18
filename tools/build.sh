#!/bin/bash
# build classic_eval (or run cargo args) in docker; usage: tools/build.sh [cargo args]
cd "$(dirname "$0")/.."
args=("$@"); [ ${#args[@]} -eq 0 ] && args=(build --release -p ai_processor --bin classic_eval)
docker run --rm -u $(id -u):$(id -g) -e CARGO_HOME=/w/.cargo-cache -v $PWD:/w -w /w rust:1-bookworm cargo "${args[@]}"
