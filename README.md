# elint-classic-v3 — frequency-agile support for the classic engine

Code only. No CSV / PDW data in this repo (see `.gitignore`).

**Full office setup** (engines ONNX + Classic v3, model service, UI, reports, test judge): see `office/OFFICE_SETUP.md`.

## What it is
`pipeline/` is the `ai_processor` crate (from `ai_elint`) with classic v3 added to
`pipeline/src/classic_v2.rs`. **v3 is ON by default** (since update 4). `CLASSIC_V3=0` gives exact v2 behaviour.

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
- **Latest (all fixes, 2026-11-18):** `cd ai_elint && patch -p1 < patches/office_ai_elint_2026-11-18.diff`.
  Made against the box copy of `ai_elint` (synced 11-17). Covers `classic_v2.rs`, `lib.rs`, new `pri_pattern.rs` + `signature.rs`,
  `bin/classic_eval.rs`, `pipeline/Cargo.toml` (classic_eval bin) and `config/classic_library.json` (`pri` keys).
  Checked on a copy: applies clean, release build ok, tests lib 69/69, unit 80/80, parallel_clustering 3/3.
  On real direction-finder data keep `CLASSIC_V3=0` until the bearing question (Update 5) is answered.
- Older (v3 only, 11-17):
- Apply only the classic change: `cd ai_elint && patch -p1 < patches/classic_v2_to_v3.diff` (made against the 11-17 box copy of `classic_v2.rs`).
- `patches/hdbscan_arm_build_shim.diff` is ONLY for building on Apple Silicon. Do not apply on the box (x86).
- Library with test agile radars: `tools/library_exp.json` (`CLASSIC_LIBRARY=...`).

## Tools
- `tools/gate.sh <tag> [ENV=..]` = every test set + mf6 DOA report; `EXTRA_SYNTH="dfmix dfmix_holdout"` adds the DF sets. `tools/tune.sh` = tune sets only. `tools/build.sh` = cargo in docker.
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

## Update 4: v3 on by default
- `CLASSIC_V3` default = 1. Set `CLASSIC_V3=0` for exact v2 behaviour (checked: identical groups on all 608 files).
- ONNX engine not affected: ONNX uses only `ClassicCfg` beat settings + `same_clock()` (optional ONNX_CLOCK_MERGE); those functions and defaults are byte-identical to the box version. Only `classic_v2.rs` changed (+ new `bin/classic_eval.rs`).
- Tests, original box code vs this code (Apple Silicon, ARM shim on both): lib 53/53 -> 56/56 (+3 v3 tests), tests/unit.rs (ONNX, inference, global match, hdbscan, db helpers) 80/80 = 80/80, parallel_clustering 3/3 = 3/3.
- 608 files: default (v3) 608/608, CLASSIC_V3=0 608/608.
- Reminder: on the real DF capture (mf6) v3 mixed directions in some groups (see update 2). If real DF data looks wrong, set CLASSIC_V3=0.

## Update 5: DOA gate for direction-finder data (fix 1)
Goal: v3 must not put pulses from different directions into one radar, while
omni / unusable bearings change nothing. Everything is measured from the data;
`CLASSIC_DOA_GATE=0` turns it off, `CLASSIC_V3=0` is still exact v2.

What the data showed first:
- **Corpus (608) and the other-generator files** are a *spinning* direction finder
  plus an omni receiver. The omni rows carry no bearing (now read as "no DOA" when
  `source=omni`). The spinning DF reports its own pointing angle and hears one radar
  on several bearings (side lobes): a radar's looks land at many angles.
- **mf6** is the same kind of receiver: inside one look the bearing sweeps ~270 deg/s.
- So **only bearings measured at the same moment prove two radars**. Bearings from
  different looks are evidence only for a *fixed* direction finder (bearing steady).

Rules (`classic_v2.rs`, v3 only):
1. `doa_ctx` per file: bearing noise (closest bearing among the next 6 pulses of a
   group), "same pulse twice" check (two reports of one pulse that disagree = bearings
   unusable), drift = least-squares slope of each bearing track. DF if noise <= 10 deg
   and duplicates agree; **fixed DF** if drift <= 15 deg/s, else spinning.
2. `doa_differs`: compare two trains at the same moments (other train's bearing
   interpolated between neighbours <= 20 ms away); median difference > 6 deg = different.
   Fixed DF only: if never on air together, compare whole-train bearings.
3. Veto (DF): split repair, beat/coherence join, v3 candidate sets and v3 join never
   join two pieces that differ.
4. Split (fixed DF only): bearing tracks -> parts by same-moment / whole-train
   evidence; done on every group before the v3 join (pieces must be direction-pure)
   and on the final groups (replaces the v2-era histogram rule there). Leftover pulses
   go to the part with the nearest bearing. A spinning DF never splits (side lobes).
5. Generator: `DFMIX=1 tools/gen_freqmodes.py <dir>` = 2-4 radars of the SAME type at
   bearings >= 20 deg apart, `dfmixstatic` (fixed DF, 1.5 deg noise) and `dfmixrot`
   (antenna spinning 0.8-2.5 s, hears +-16 deg, reports its pointing angle like mf6).
   Rows now carry `source` = df / omni. Tune = seed 7, holdout = seed 99 (never tuned).

Results (`tools/gate.sh`, library `tools/library_exp.json`):
| set | v2 | v3 before | v3 + DOA gate |
|---|---|---|---|
| corpus 608 | 608 | 608 | 608 |
| holdout 40 mixed | 38 | 40 | 40 |
| other-generator labeled (7 files here) | 2 | 4 | 4 |
| synth seed 7 (tune) | 0 | 76 | 80 |
| synth seed 99 (holdout) | 1 | 75 | 79 |
| DFMIX seed 7 (tune), fixed DF / spinning | 0 / 0 of 12 | 4 / 0 | 8 / 0 (all 12 fixed-DF radars recovered) |
| DFMIX seed 99 (holdout), fixed DF / spinning | 0 / 0 of 12 | 3 / 0 | 8 / 0 |

`CLASSIC_V3=0`: groups identical to the old binary on 720 files (corpus, synth, DFMIX,
mf6, anurag). Tests: lib 56 -> 59 (+3 DOA tests), unit 80/80, parallel 3/3.
Speed: corpus one file at a time, median 26.3 -> 29.8 ms per file.

mf6 (unlabeled, spinning DF), DOA arc holding 80 % of a group's pulses:
| | groups | pulses grouped | median arc | groups > 30 deg (share of pulses) |
|---|---|---|---|---|
| v2 | 21 | 13,603 | 17.3 | 4 (19 %) |
| v3 before | 17 | 20,857 | 35.0 | 10 (73 %) |
| v3 + DOA gate (default) | 17 | 20,857 | 35.0 | 10 (73 %) |
| v3 + DOA gate, `CLASSIC_DOA_FIXED_RATE=100000` (treat as fixed DF) | 53 | 20,439 | 12.8 | 6 (6 %) |

Honest reading of mf6: the wide groups are made of *different looks* (never on air at
the same time). On a spinning DF that is not proof of different radars: the generator
data shows the same radar on several bearings, and forcing the split there breaks the
corpus (608 -> 109) and ext (4 -> 1). So by default mf6 is unchanged. If the real
receiver reports true bearings (no side-lobe looks), `CLASSIC_DOA_FIXED_RATE=100000`
gives tight groups; that needs a labeled real capture to confirm.

Still open:
- Spinning DF, same-type radars (DFMIX `dfmixrot`): 0/12 -- bearing cannot be trusted
  across looks without knowing the receiver. Needs receiver knowledge (true-bearing DF
  or not) as a setting, or a labeled real capture.
- Fixed DF: 4/12 DFMIX files still have extra small groups (all radars recovered).
- The DB path (`ai_processor`) has no `source` column in `FlowPulse`: omni rows from the
  DB still carry their DOA. Only `classic_eval` blanks them.

## Update 6: PRI pattern (fix 3, report only)
New module `pipeline/src/pri_pattern.rs`: for every group, the PRI type is worked out from TOA only.
`classic_eval` adds `"pri_pattern"` to each group. Grouping is not changed.

How it works:
1. Sort TOAs. Drop duplicate pulses (< 1 us after the previous one = same pulse seen by two receivers).
2. Gaps between pulses are split into levels (sorted gaps, cut where neighbours differ by > 0.3 %; wider for short trains). Levels with < 1 % of gaps are ignored.
3. From the smallest level up: a level is a **harmonic** (missing pulse) when it is a sum of 2-4 main levels, rarely follows itself and holds fewer gaps than the main levels. Otherwise it is a **main level**.
4. Type:
   - smallest main level wide (spread > 0.5 %): consecutive gaps close to each other = `sliding`, else `jitter` (jitter % = half-width of the spread);
   - one sharp main level = `constant`;
   - several sharp levels: a gap usually followed by the same level = `dwell_switch` (dwell PRIs, pulses per dwell, dwell time); else the shortest repeat of the level sequence (2-16) = `stagger` (period + elements in pulse order, starting at the smallest);
   - many sharp levels in smooth order (a fast ramp whose steps repeat) = `sliding`;
   - else `complex`; fewer than 16 gaps = `too_few`.
5. `missing_pct`: gaps outside the main levels counted as whole multiples of the mean PRI.

Output per group: `kind, pri_us, jitter_pct, stagger{period_us, elements_us}, dwell{pris_us, dwell_pulses, dwell_us}, sliding{min_us, max_us, step_us, ramp_pulses}, levels_us, harmonics_us, missing_pct, dup_skipped, n_gaps`.

Test data: `tools/gen_primix.py <dir>` (new; uses gen_freqmodes helpers). Each file = one labelled radar + 1-2 other radars.
Types: constant, jitter (+-1..20 %), stagger (2-8 elements, some with a repeated element), dwell-and-switch (2-5 PRIs, 16-128 pulses per dwell), sliding (saw or triangle, 50-400 pulse ramp, 1.1-1.6x range).
Missing pulses 0 / 5 / 10 / 20 %; 30 % of files have duplicate pulses from a second receiver. 12 files per type.
SEED=7 = tune set (rules fixed on it), SEED=99 = holdout (run once, never tuned). Score: `tools/score_pri.py results.jsonl manifest.json`.

Type found vs true type (labelled radar):
| set | constant | jitter | stagger | dwell-switch | sliding | total |
|---|---|---|---|---|---|---|
| tune (seed 7) | 12/12 | 11/12 (1 = grouping, not PRI) | 12/12 | 12/12 | 12/12 | 59/60 |
| holdout (seed 99) | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 60/60 |

Values (only where the type is right), holdout:
| check | result |
|---|---|
| PRI value, median error | 0.00-0.06 % |
| missing pulses within 5 points | 60/60 |
| jitter % within 1.5 points | 12/12 |
| stagger period within 1 % | 12/12 |
| stagger elements, same cycle order, each within 1 % | 12/12 |
| dwell PRIs all within 1 % | 12/12 |
| pulses per dwell within 10 % | 11/12 |
| sliding min and max within 3 % | 12/12 |

The other (unlabelled-type) radars in the same files: holdout constant 36/36, stagger 24/24, jitter (gaussian 5 %) 26/26; tune 27/28, 31/33, 35/36 (all misses = grouping, not PRI).

Grouping unchanged (`tools/same_groups.py`, groups compared as a set, PRI key and timing ignored):
| set | result |
|---|---|
| corpus 608 | 608/608 files identical (2,736 groups) |
| synth seed 7 | 86/86 files identical |
| synth seed 99 | 86/86 |
| PRI set seed 7 | 60/60 |
| PRI set seed 99 | 60/60 |
Note: unknown groups come out in a run-dependent order (already so before this change), so the check compares groups as a set.

PRI types found on the 608 corpus (sanity check, no PRI-type labels there): `tools/pri_summary.py`.
| emitter | groups | type found (all groups) | found levels vs generator PRI column (one file) |
|---|---|---|---|
| 1104 | 112 | dwell_switch 112 | 1105 / 1165 / 1250 / 1540 = generator's 4 values |
| 1105 | 112 | dwell_switch 112 | 116 / 122 / 1400 / 1605 = generator |
| 1106 | 112 | dwell_switch 112 | 75 / 86 / 1200 / 1480 = generator (75 found twice: 74.8 + 75.0) |
| 1107 | 112 | dwell_switch 112 | 1065 / 1125 / 1200 / 1485 = generator |
| 1108 | 112 | stagger 112 | generator has 16 values (1321-1331 jitter steps); found 147 / 1127 / 1326 |
| 1109 | 112 | stagger 112 | 307 / 2894 / 3094 / 3321 / 3564 of generator's 8 |
| 1113 | 112 | dwell_switch 112 | 294-451 levels = generator values |
| 1116 | 112 | sliding 112 | 130-158 us ramp + longer gaps |
| unknown_ai | 1,840 | constant 1,461, dwell_switch 314, stagger 64, complex 1 | - |

Every library emitter gets the same type in all 112 of its groups.

Limits:
- Jittered stagger / jittered dwell-switch (sharp levels plus large jitter) come out `complex`.
- Sliding range above ~1.75x looks like missing pulses at the top of the ramp; not tested.
- Staggers longer than 16 pulses are not looked for.
- Labels are synthetic; no real capture with known PRI types yet.

## Update 7: unseen radars (fix 2)
Goal: a radar type that is not in the naming library comes out as ONE clean `unknown_ai` group, is never given a library name, and is described by what was measured.

**Problem found.** Naming looked mainly at the carrier. A new radar sitting on a library radar's carrier (other pulse width, modulation or PRI) got that radar's name, and because the engine keeps one emitter per library id, its pulses were then merged into the real radar's group. On the new test set below, 27 of 50 new radars got a false name and 5 were merged into a library radar.

**Change (v3 only, `CLASSIC_V3=0` = exact v2):**
- Library rules can carry `"pri"` = the radar's PRI fundamental in us, measured on the corpus. Each of the 8 corpus radars has one stable fundamental in every corpus file (MF03 294, PSR3 75, PSR4 1065, PSR1 1105, PSR5m2 130, MF04 307, PSR2 116, PSR5m1 147). A group is named only when its own PRI fundamental is within `CLASSIC_NAME_PRI_TOL` (default 0.03 = 3 %) of it. `CLASSIC_NAME_PRI_TOL=0` switches the check off. Rules without `pri` (for example the synthetic agile radars in `tools/library_exp.json`) are not checked.
- PRI fundamental = the shortest pulse gap whose +-1 us neighbourhood holds >= 25 % of the busiest gap, gaps < 1 us skipped (same pulse seen twice). A staggered train gives its base interval, not a harmonic.
- New `pipeline/src/signature.rs`: per-group `signature` in `classic_eval` JSON: `n`, `fmode`, `f_q5_mhz`/`f_q95_mhz`, `pw_med_us`, `bw_med_mhz`, `modulation` + `mod_share`, `pri_mode_us`, `pri_fund_us`, `doa_spread_deg` (narrowest arc holding 80 % of the angles). This is report output only. The grouping steps are unchanged.

**Test set** `tools/gen_unseen.py`: real corpus files (2 library radars each) plus 1-2 new radars (`emitter_id` 101/102, never read by the engine). Duty cycle is capped at 10 %. Classes:
- `far`: carrier away from every library band, any frequency mode.
- `carrier`: fixed carrier inside a library radar's band, with other PW, BW, modulation and PRI (15-45 % away).
- `pri`: same band, modulation and BW as a library radar; only the PRI differs (15-45 %).
- `two`: one far and one carrier radar in the same file.

Tune on `SEED=7`; check on `SEED=99`, which was never tuned on. Scored by `tools/score_unseen.py`:
- **rec**: one group holds >= 80 % of the new radar and is >= 80 % that radar.
- **named**: a group that is mostly the new radar got a library name.
- **split**: more than one group is mostly the new radar.
- **known_ok**: each library radar in the base file keeps the same name it gets on the base file alone.

```
SEED=7  tools/gen_unseen.py <corpus known_emitters dir> unseen
SEED=99 tools/gen_unseen.py <corpus known_emitters dir> unseen_holdout
tools/score_unseen.py results.jsonl unseen/manifest.json ref.jsonl   # ref = classic_eval on the base files
```

| set (40 files, 50 new radars) | files ok | new radars recovered | false library names | split | library radars ok |
|---|---|---|---|---|---|
| tune seed 7, before | 12 | 45 | 27 | 1 | 177/177 |
| tune seed 7, after | 38 | 50 | 1 | 1 | 177/177 |
| holdout seed 99, before | 11 | 47 | 29 | 0 | 198/198 |
| holdout seed 99, after | 39 | 50 | 1 | 0 | 198/198 |

Gate (all unchanged):

| set | before | after |
|---|---|---|
| corpus 608 | 608/608 | 608/608 |
| holdout 40 mixed files | 40/40 | 40/40 |
| synthetic freq-mode seed 7 (with `library_exp.json`) | 76/86 | 76/86 |
| synthetic holdout seed 99 | 75/86 | 75/86 |
| 9 files from other sources (names per group) | - | identical |
| `CLASSIC_V3=0` on corpus, synth, synth holdout, unseen | - | byte-identical to v2 |
| unit tests: lib / unit.rs / parallel_clustering | 56 / 80 / 3 | 59 / 80 / 3 |

Still open:
- Jittered look-alike: 5 % jitter around 157 us on PSR5m1's band (147 us). The jitter's short gaps pull the measured fundamental to about 145 us, inside 3 %, so it is still named PSR5m1 (1 of 50, tune).
- A look-alike whose PRI is within 1 % of the library radar (146 vs 147 us, PSR5m1 band) is named. PRI cannot tell them apart; it would need PW or modulation envelopes in the library, which the corpus radars vary too much to set safely (1 of 50, holdout).
- One sliding new radar (3679-3740 MHz, jittered PRI) comes out as 3 groups (tune `far_07`). This is a grouping issue, not naming, and it was the same before the change.
- The `pri` values describe the generator corpus radars. For real radars, take them from the radar's spec sheet or a clean recording.
