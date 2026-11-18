#!/bin/bash
# Full gate: tools/classic/gate.sh <tag> [ENV=val ...]  (uses ./target/release/classic_eval)
# Sets: corpus 608, synth (seed 7), synth_holdout (seed 99), holdout40, labeled ext files,
# unlabeled real captures (mf6, pdw.csv), extra synth dirs in $EXTRA_SYNTH (dir:manifest pairs by name).
set -e; cd "$(dirname "$0")/../.."; tag=$1; shift; mkdir -p results/$tag
BIN=${BIN:-./target/release/classic_eval}; PY=${PY:-python3}
R=${R:-/mnt/t9/elint_box_coderepo}; D=${DATA:-data}
export EXP=$R/ai_elint_tests
run() { local name=$1; shift; env CLASSIC_LIBRARY=tools/classic/library_exp.json "${ENVS[@]}" $BIN "$@" 2>/dev/null | grep "^{" > results/$tag/$name.jsonl; }
ENVS=("$@")
run corpus $D/pdw_generator/*/*.csv
run synth synth/*.csv
run synth_holdout synth_holdout/*.csv
run holdout40 $R/elint_improvements/holdout/*.csv
run ext $D/2_unk_emitters_PDWTable.csv $D/2unk_freq-agile-only_plus10_PDWTable.csv $D/2unk_freq-agile-only_plus5_PDWTable.csv $D/PSR-1vsPSR-2_PDWTable.csv $D/anurag_PDWTable.csv $D/anurag_agility_PDWTable.csv $D/anurag_agility_plus10emitters_PDWTable.csv
run real $D/mf6_08_09_4_15_pm.csv $D/pdw.csv
for s in $EXTRA_SYNTH; do run $s $s/*.csv; done
for s in corpus holdout40 ext; do echo "== $tag $s"; $PY tools/classic/score.py results/$tag/$s.jsonl | grep -E "TOTAL|^corpus|^holdout|^data"; done
for s in synth synth_holdout $EXTRA_SYNTH; do echo "== $tag $s"; $PY tools/classic/score.py results/$tag/$s.jsonl $s/manifest.json | grep -E "TOTAL"; done
$PY tools/classic/doa_report.py results/$tag/real.jsonl
