#!/bin/bash
# parity.sh <csv>...  — run Python master and Rust ONNX on each CSV, diff DB fingerprints.
OUT=$HOME/coderepo/runs/parity; mkdir -p $OUT
for f in "$@"; do
  b=$(basename $f .csv)
  (cd ~/coderepo/ai_elint_python_master/integration && timeout 3600 venv/bin/python ~/coderepo/elint_reports/tools/py_ref_run.py $f > $OUT/${b}_py.log 2>&1)
  bash ~/coderepo/elint_reports/tools/db_snapshot.sh > $OUT/${b}_py.snap
  pt=$(grep -oE "MASTER E2E done in [0-9.]+s" $OUT/${b}_py.log | grep -oE "[0-9.]+s")
  r=$(bash ~/coderepo/elint_reports/tools/run_one.sh onnx $f $OUT)
  bash ~/coderepo/elint_reports/tools/db_snapshot.sh > $OUT/${b}_rs.snap
  rt=$(echo "$r" | grep -oE "t=[0-9.]+s")
  if diff -q $OUT/${b}_py.snap $OUT/${b}_rs.snap >/dev/null; then st=SAME; else st=DIFF; fi
  echo "PARITY|$b|$st|py=$pt|rs_$rt|py: $(head -2 $OUT/${b}_py.snap | tr '\n' ' ')|rs: $(head -2 $OUT/${b}_rs.snap | tr '\n' ' ')"
done
