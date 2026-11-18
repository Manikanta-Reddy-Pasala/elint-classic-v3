#!/bin/bash
# quick tune runs on TUNE sets only (seed 7 + corpus + ext + mf6); usage tune.sh tag ENV...
cd ~/work/elint-proper-2026-11-18; tag=$1; shift
R=${R:-/mnt/t9/elint_box_coderepo}; D=${DATA:-data}; mkdir -p results/$tag
r() { n=$1; shift; env CLASSIC_LIBRARY=tools/classic/library_exp.json "$@" 2>/dev/null | grep "^{" > results/$tag/$n.jsonl; }
r corpus env "$@" ./target/release/classic_eval $D/pdw_generator/*/*.csv
r synth env "$@" ./target/release/classic_eval synth/*.csv
r dfmix env "$@" ./target/release/classic_eval dfmix/*.csv
r ext env "$@" ./target/release/classic_eval $D/2_unk_emitters_PDWTable.csv $D/2unk_freq-agile-only_plus10_PDWTable.csv $D/2unk_freq-agile-only_plus5_PDWTable.csv $D/PSR-1vsPSR-2_PDWTable.csv $D/anurag_PDWTable.csv $D/anurag_agility_PDWTable.csv $D/anurag_agility_plus10emitters_PDWTable.csv
r real env "$@" ./target/release/classic_eval $D/mf6_08_09_4_15_pm.csv
c=$(python3 tools/classic/score.py results/$tag/corpus.jsonl | grep TOTAL | awk '{print $3}')
s=$(python3 tools/classic/score.py results/$tag/synth.jsonl synth/manifest.json | grep TOTAL | awk '{print $3}')
m=$(python3 tools/classic/score.py results/$tag/dfmix.jsonl dfmix/manifest.json | grep TOTAL | awk '{print $3}')
e=$(python3 tools/classic/score.py results/$tag/ext.jsonl | grep TOTAL | awk '{print $3}')
x=$(python3 tools/classic/doa_report.py results/$tag/real.jsonl | head -1 | cut -c25-)
echo "$tag [$*] corpus $c/608 synth $s/86 dfmix $m/24 ext $e/7 | mf6$x"
