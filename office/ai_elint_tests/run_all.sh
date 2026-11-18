#!/bin/bash
# run_all.sh — Run ALL test CSVs (known + unknown) through the Rust ai_processor.
# Loops over every .csv under the pdw_generator folder, wipes DBs per file,
# and writes one RESULT|... line per file to the results file.
#
# Usage (on the VM):
#   nohup bash run_all.sh > /tmp/run_all_stdout.log 2>&1 &
#   tail -f /tmp/run_all.log            # progress
#   cat /tmp/run_all_results.txt        # one RESULT line per file
#
# Expected files on the VM:
#   /home/sysadm/ai_elint_tests_work/tests/data/input_csv_files/pdw_generator/
#   ├── known_emitters/    448 CSVs
#   └── unknown_emitters/  160 CSVs
#
# Pass criteria per file: rc=0 and pg_em == exp_em (from expectations.json).

set -u

B="/home/sysadm/ai_elint_tests_work/tests/data/input_csv_files/pdw_generator"
RESULTS=/tmp/run_all_results.txt
LOG=/tmp/run_all.log
: > "$RESULTS"
: > "$LOG"

TOTAL=$(find "$B" -name '*.csv' | wc -l)
DONE=0
echo "Starting batch of $TOTAL files" | tee -a "$LOG"
echo "Started $(date)" | tee -a "$LOG"

# Find all CSVs (known + unknown), sort for deterministic order
mapfile -t FILES < <(find "$B" -name '*.csv' | sort)

for f in "${FILES[@]}"; do
    DONE=$((DONE + 1))
    echo "[$DONE/$TOTAL] $(basename "$f")" | tee -a "$LOG"
    bash /home/sysadm/vm_oneshot_one.sh "$f" >> "$RESULTS" 2>&1
    if [ $((DONE % 25)) -eq 0 ]; then
        echo "progress: $DONE/$TOTAL $(date)" | tee -a "$LOG"
    fi
done

echo "DONE $(date)" | tee -a "$LOG"
echo "=== BATCH COMPLETE: $DONE files ===" | tee -a "$LOG"
echo "Results: $RESULTS"
echo "Log:     $LOG"
