#!/bin/bash
# Self-test using ONLY this repo (no ai_elint_tests repo, no databases, no model service).
# usage: tools/classic/selftest.sh [quick]      quick = corpus only (skip synthetic sets)
# needs: cargo, python3 (+ numpy for the synthetic sets), data/pdw_generator/ (608 CSVs, not in git)
# env:   DATA=dir holding pdw_generator/ (default ./data); any CLASSIC_* setting is passed through
set -e; cd "$(dirname "$0")/../.."
PY=${PY:-python3}; DATA=${DATA:-data}; T=results/selftest; mkdir -p $T
echo "== build"; RUSTFLAGS=${RUSTFLAGS:-"-C target-cpu=native"} cargo build --release -p ai_processor --bin classic_eval 2>&1 | tail -1
BIN=./target/release/classic_eval
ls $DATA/pdw_generator/*/*.csv >/dev/null 2>&1 || { echo "no CSVs in $DATA/pdw_generator (copy data/ from the box)"; exit 1; }
for mode in v3 v2; do
  v=1; [ $mode = v2 ] && v=0
  CLASSIC_V3=$v $BIN $DATA/pdw_generator/*/*.csv 2>/dev/null | grep "^{" > $T/corpus_$mode.jsonl
  echo "== corpus 608, classic $mode   (expect TOTAL 608 608)"; $PY tools/classic/score.py $T/corpus_$mode.jsonl | grep -E "^corpus|TOTAL"
done
[ "$1" = quick ] && exit 0
for s in 7 99; do
  SEED=$s $PY tools/classic/gen_freqmodes.py $T/synth_$s >/dev/null
  CLASSIC_LIBRARY=tools/classic/library_exp.json $BIN $T/synth_$s/*.csv 2>/dev/null | grep "^{" > $T/synth_$s.jsonl
  echo "== frequency-mode synthetic, seed $s   (expect 80 / 79 of 86)"; $PY tools/classic/score.py $T/synth_$s.jsonl $T/synth_$s/manifest.json | grep TOTAL
  SEED=$s $PY tools/classic/gen_primix.py $T/pri_$s >/dev/null
  $BIN $T/pri_$s/*.csv 2>/dev/null | grep "^{" > $T/pri_$s.jsonl
  echo "== PRI pattern, seed $s   (expect 59-60 / 60)"; $PY tools/classic/score_pri.py $T/pri_$s.jsonl $T/pri_$s/manifest.json 2>&1 | grep -iE "total|type acc" | head -3
done
