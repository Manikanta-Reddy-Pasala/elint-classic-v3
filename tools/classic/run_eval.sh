#!/bin/bash
# usage: tools/classic/run_eval.sh <tag> [ENV=val ...]
# DATA=dir with pdw_generator/{known,unknown}_emitters/*.csv, SYNTH=dir from gen_freqmodes.py,
# EXP=dir holding tests/test_data/{known_emitters_db_emitter_level,unknown_emitters_db}.csv (ai_elint_tests)
set -e; cd "$(dirname "$0")/../.."; tag=$1; shift; mkdir -p results
BIN=./target/release/classic_eval; PY=${PY:-python3}
DATA=${DATA:-data}; SYNTH=${SYNTH:-synth}
env "$@" $BIN $DATA/pdw_generator/*/*.csv 2>/dev/null | grep "^{" > results/${tag}_corpus.jsonl
env "$@" $BIN $SYNTH/*.csv 2>/dev/null | grep "^{" > results/${tag}_synth.jsonl
echo "== $tag corpus"; $PY tools/classic/score.py results/${tag}_corpus.jsonl
echo "== $tag synth"; $PY tools/classic/score.py results/${tag}_synth.jsonl $SYNTH/manifest.json
