#!/bin/bash
# run_engine.sh <classic|onnx> <tag> [pytest -k expr]
# Full harness run for one engine with per-file reports in ~/coderepo/runs/pytest/<tag>/.
E=$1; TAG=$2; K=${3:-"source_omni or main_lobe"}
HERE=/home/sysadm/coderepo/elint_reports
D=/home/sysadm/coderepo/runs/pytest/$TAG; mkdir -p $D/files
s=$(date +%s)
REPORT_DIR=$D/files $HERE/pytest_run.sh $E $D/pytest -k "$K" > $D/summary.txt 2>&1
echo "elapsed=$(( $(date +%s)-s ))s" >> $D/summary.txt
docker exec -u 1000 elint_pytest2 python $HERE/build_report.py \
  --report-dir $D/files --junit $D/pytest.xml \
  --expected /home/sysadm/coderepo/ai_elint_tests/tests/test_data/expected_results.json \
  --engine $E --out $D/report >> $D/summary.txt 2>&1
