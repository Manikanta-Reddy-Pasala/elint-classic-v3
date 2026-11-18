#!/bin/bash
# full_chain.sh — sequential (fair timing): Rust classic 608 -> Rust ONNX 608 -> Python master 30 -> time report
R=/home/sysadm/coderepo/runs; H=/home/sysadm/coderepo/elint_reports
D=/home/sysadm/coderepo/ai_elint/data/pdw_generator
log(){ echo "$(date '+%F %T') $*" >> $R/full_chain.log; }
until grep -q ALL_DONE $R/cmp20.txt.out 2>/dev/null; do sleep 30; done
log "start classic_608"; $H/run_engine.sh classic classic_608; log "done classic_608"
log "start onnx_608";    $H/run_engine.sh onnx onnx_608;       log "done onnx_608"
# python: the 20 compared files + 10 more (5 known, 5 unknown)
cp $R/cmp20.txt $R/py30.txt
(ls $D/known_emitters/*.csv | shuf --random-source=<(yes 31) -n 5; ls $D/unknown_emitters/*.csv | shuf --random-source=<(yes 31) -n 5) | grep -vxFf $R/cmp20.txt >> $R/py30.txt
mkdir -p $R/python_30
log "start python_30 ($(wc -l < $R/py30.txt) files)"
while read -r f; do
  b=$(basename "$f" .csv)
  (cd /home/sysadm/coderepo/ai_elint_python_master/integration && timeout 3600 venv/bin/python $H/tools/py_time_run.py "$f" $R/python_30/$b.python.json > $R/python_30/$b.log 2>&1)
done < $R/py30.txt
log "done python_30"
docker exec -u 1000 elint_pytest2 python $H/time_report.py --classic $R/pytest/classic_608/report.csv --onnx $R/pytest/onnx_608/report.csv \
  --python-dir $R/python_30 --expected /home/sysadm/coderepo/ai_elint_tests/tests/test_data/expected_results.json --out $R/time_report >> $R/full_chain.log 2>&1
log "ALL_DONE"
