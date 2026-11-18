#!/bin/bash
# compare.sh <listfile> <outdir>
# Per CSV: Python master (ONNX reference), Rust ONNX, Rust classic -> one CMP line each.
LIST="$1"; OUT="$2"; mkdir -p "$OUT"
snapline(){ head -2 "$1" | tr '\n' ' ' | sed -E 's/ +/ /g'; }
while read -r f; do
  b=$(basename "$f" .csv)
  (cd ~/coderepo/ai_elint_python_master/integration && /usr/bin/time -f "%e" -o "$OUT/${b}_py.time" timeout 3600 venv/bin/python ~/coderepo/elint_reports/tools/py_ref_run.py "$f" > "$OUT/${b}_py.log" 2>&1)
  bash ~/coderepo/elint_reports/tools/db_snapshot.sh > "$OUT/${b}_py.snap"
  ro=$(bash ~/coderepo/elint_reports/tools/run_one.sh onnx "$f" "$OUT"); bash ~/coderepo/elint_reports/tools/db_snapshot.sh > "$OUT/${b}_onnx.snap"
  rc=$(bash ~/coderepo/elint_reports/tools/run_one.sh classic "$f" "$OUT"); bash ~/coderepo/elint_reports/tools/db_snapshot.sh > "$OUT/${b}_classic.snap"
  par=$(diff -q "$OUT/${b}_py.snap" "$OUT/${b}_onnx.snap" >/dev/null && echo SAME || echo DIFF)
  echo "CMP|$b|py_t=$(cat $OUT/${b}_py.time)s|onnx_$(echo "$ro" | grep -oE 't=[0-9.]+s')|classic_$(echo "$rc" | grep -oE 't=[0-9.]+s')|parity=$par|py: $(snapline $OUT/${b}_py.snap)|onnx: $(snapline $OUT/${b}_onnx.snap)|classic: $(snapline $OUT/${b}_classic.snap)"
done < "$LIST"
