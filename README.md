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
