#!/bin/bash
# 10 known-emitter files (each holds 2 ground-truth emitters) through both
# pipelines, one file at a time. Classic and ONNX both truncate the tables
# at the start of their run, so the RESULT lines are per pipeline.
set -u
DIR=/home/sysadm/ai_elint_tests_work/tests/data/input_csv_files/pdw_generator/known_emitters
OUT=${1:-/home/sysadm/cmp10}
mkdir -p "$OUT"
: > "$OUT/RESULTS.txt"
for f in $(ls "$DIR"/*.csv | head -10); do
  bash /home/sysadm/run_one_both.sh "$f" "$OUT"
done
cat "$OUT/RESULTS.txt"
