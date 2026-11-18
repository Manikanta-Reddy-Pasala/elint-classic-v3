# PROGRESS (nuc2 ~/work/elint-proper-2026-11-18)

## 2026-11-18 fix 1: DOA gate (committed)
- Base 30437f2. Tooling: tools/gate.sh (all sets), tools/tune.sh (tune sets only),
  tools/build.sh (docker rust build), tools/doa_report.py, classic_eval `doa_arc80`,
  classic_eval blanks DOA for `source=omni` rows.
- Path of the work (what failed):
  1. same-moment gate + bearing drift fallback: corpus fine, mf6 unchanged.
  2. track-centre ("look") gate: corpus broke -- two receivers report the same pulse at
     80 vs 194 deg -> added "same pulse twice" check; ext broke -- spinning DF shows one
     radar at many bearings (side lobes).
  3. Final: same-moment evidence only; whole-train bearings only for a fixed DF
     (drift <= 15 deg/s from track slopes); splits only for fixed DF.
- Results: see README "Update 5". Fixes 2 and 3 are done by other workers (not here).
