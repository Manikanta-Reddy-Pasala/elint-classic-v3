# elint-classic-v3 — frequency-agile support for the classic engine

Code only. No CSV / PDW data in this repo (see `.gitignore`).

## What it is
`pipeline/` is the `ai_processor` crate (from `ai_elint`) with classic v3 added to
`pipeline/src/classic_v2.rs`. Everything new is behind `CLASSIC_V3=1`; without it
classic runs exactly as v2.

Handles every frequency mode on the "Frequency mode" slide:

| mode | v2 | v3 |
|---|---|---|
| fixed | ok | ok |
| agile discrete (random hops) | split into many | joined |
| agile continuous (sparse) | pulses dropped | rescued + joined |
| burst | split per burst | joined |
| sliding | pulses dropped | joined |
| n-modal | split when PRI jittered | joined |
| unknown | - | labelled by `freq_mode()` |

## v3 steps (all env-tunable)
1. **Rescue**: pulses of sparse cells (< `CLASSIC_MIN_CELL`) regrouped on modulation/PW/BW, carrier ignored; kept if >= `CLASSIC_RESCUE_MIN` (200) pulses and >= `CLASSIC_RESCUE_SPAN_MHZ` (20) wide.
2. **Join**: pieces with same modulation, PW within 2 %, BW overlap are one radar if together they make one clean pulse train:
   - gap histogram top-8 share >= `CLASSIC_REG_MIN` (0.25), each piece fits >= `CLASSIC_REG_FIT` (0.5) x that;
   - collisions <= `CLASSIC_COLL_MAX` (2 %): one transmitter never sends two pulses closer than its own shortest gap;
   - time-disjoint pieces must share a PRI-scale gap (or whole multiple), tol `CLASSIC_GAP_RATIO_TOL` (5 %).
3. **DOA split**: only when angles show 2+ tight peaks (direction-finder data); omni / random DOA never splits.
4. **Naming**: rule needs >= `CLASSIC_NAME_SHARE` (80 %) of pulses inside `f`. New rule keys: `"match":"span"` (name by range, agile radars), `"pw":[lo,hi]` (median PW, us), `"modulation"`.
5. `freq_mode()` labels each group: fixed / sliding / burst / n_modal / agile_discrete / agile_continuous.

## Results (offline runner, labels only for scoring)
| set | v2 | v3 |
|---|---|---|
| corpus 608 files | 608/608 | 608/608 |
| synthetic freq-mode set, seed 7 (86 files, used to tune) | 0/86 | 76/86 |
| synthetic holdout, seed 99 (86 files, never tuned) | 1/86 | 75/86 |

Not solvable: two radars of the same type, both agile, no DOA (identical pulses).

## Use in the office repo
- Apply only the classic change: `cd ai_elint && patch -p1 < patches/classic_v2_to_v3.diff` (made against the 11-17 box copy of `classic_v2.rs`).
- `patches/hdbscan_arm_build_shim.diff` is ONLY for building on Apple Silicon. Do not apply on the box (x86).
- Library with test agile radars: `tools/library_exp.json` (`CLASSIC_LIBRARY=...`).

## Tools
- `pipeline/src/bin/classic_eval.rs` — CSV -> classic detect -> one JSON line per file (no DB).
- `tools/gen_freqmodes.py <outdir>` (`SEED=99` for holdout) — synthetic PDW CSVs, same columns as the corpus.
- `tools/score.py`, `tools/run_eval.sh <tag> [CLASSIC_V3=1 ...]` — env `DATA`, `SYNTH`, `EXP` (ai_elint_tests dir), `PY`.

```
cargo build --release -p ai_processor --bin classic_eval
cargo test --release -p ai_processor --lib classic_v2
```

## Update 2: tested on data from other sources (no tuning on it)
New v3 rules (physical, not tuned):
- **Overlap veto** in the beat (coherence) step: never join two groups when a pulse of one starts while a pulse of the other is still on air (one transmitter cannot). All 6,775 correct joins in the 608 corpus have 0 overlap. `CLASSIC_COH_MAX_OVERLAP` (0.002).
- **No modulation reported** (receiver without intra-pulse analysis, < 10 % pulses with a modulation): every pulse is one "none" class instead of being skipped (v2 found 0 radars in such a file).
- Optional `CLASSIC_COH_MAX_FGAP_MHZ` (default 0 = off): widest agile band; set it from radar specs.
- `classic_eval` reads real-capture format too (date-time TOA, `modulation` column, quoted fields) and prints DOA / carrier spread per group.

| set | v2 | v3 |
|---|---|---|
| corpus 608 | 608/608 | 608/608 |
| holdout 40 mixed files | 38/40 | 40/40 |
| labeled files from other generators (2unk agile, anurag, MF-05, PSR-1vs2) | 5/10 | 7/10 (9/10 with FGAP cap 450) |
| real pdw.csv (no modulation) | 0 radars | 8 radars (unlabeled) |

Still open:
- Two radars, same carrier, same modulation, PRI exactly 5:1, no overlap (anurag_PDWTable): not separable by timing/carrier/modulation.
- **Real capture mf6 (DF receiver, unlabeled):** v3 makes some groups with DOA spread 125-280 deg = pulses from different directions mixed by rescue/join (they ignore angle). v2 groups keep tighter angles. Needs a DOA gate for real DF data and a labeled real capture to verify. Do not use v3 on real DF data until then.

## Update 3: parallelism
- v3 join: candidate sets are decided in parallel (groups read-only, output assembled in the original order) -> identical results.
- `classic_eval`: files run in parallel (`EVAL_JOBS`, default all cores; `EVAL_JOBS=1` = one file at a time with classic using all cores inside the file).
- Checked: all 608 files, every group identical to the serial version, v2 and v3 modes; 608/608 both.

Timing, Mac Studio (28 cores), 608 files / 59.3 M pulses:
| | v2 | v3 |
|---|---|---|
| batch, files in parallel (wall) | 4.3 s | 4.0 s |
| batch, one file at a time (wall) | 59 s | 66 s |
| grouping per file, median | 9.3 ms | 12.8 ms (was 13.5) |
Most of one-at-a-time wall time is CSV reading in the runner. Largest classic stage = cells (~5-6 ms, shared v2 code).
