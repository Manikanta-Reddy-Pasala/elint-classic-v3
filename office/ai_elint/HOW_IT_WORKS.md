# How the two engines work

Both engines get the same input: the pulses (PDWs) in `signals.pdw`, in
arrival order, each with TOA, carrier frequency, pulse width (PW), bandwidth
(BW), power, DOA, SNR and intra-pulse modulation. Both write the same output:
`emitters` → `signals` → `clusters` in Postgres, plus `signals.clusters`
(cluster → pulse ids) in ClickHouse. A known radar gets its library id as
name (`1113` = MF03, `1109` = MF04, …); anything else is `unknown_ai`.
Select with `CLASSIC=1` (classic) or `CLASSIC=0` (ONNX).

## Classic — rules on measured pulse parameters (no model)

1. **Cells.** Put every pulse in a cell by modulation, PW (1 % steps), BW
   (1 MHz) and carrier (1 MHz). Pulses with failed modulation measurement
   (`unk…`) and cells with < 20 pulses are noise.
2. **Components.** Join neighbouring cells (±1 step in PW and BW, ±2 MHz in
   carrier) — one emitter's jitter spreads over a few cells.
3. **Split repair.** Join components with overlapping carrier and BW ranges
   and PW within 2 %.
4. **Pulse-train coherence.** Two groups are the same emitter if their pulses
   keep a fixed time relation: for each pulse of A take the gap to the
   previous pulse of B; if a few 1 µs gap values hold ≥ 13 % of the pulses
   (and ≥ 4 % the other way round), join them. This joins the different
   modes of one radar (e.g. PSR4's long lfm_down pulse + short unmodulated
   pulse) and the frequency hops of agile radars (CTA1), while two
   unrelated radars give a flat gap histogram.
5. **Naming.** A small decision tree on each group (main carrier, BW, PRI
   measured from TOA, longest PW, share of lfm_up pulses) returns the
   library id or nothing → `unknown_ai`. Groups with the same library id
   become one emitter.
6. **Write.** One emitter + signal per group; one cluster row per 1 s of its
   pulses, with PRI measured on the real pulse train (duplicate pulses from
   the second receiver skipped).

Fast (~0.1 s per file, all CPU cores) and exact on the generator corpus
(608/608 pytest, exact emitter count on 606/608).

## ONNX — the trained transformer, same logic as the Python reference

1. **Windows.** Cut the pulses into windows of 512 (a trailing partial window
   is dropped, as in Python).
2. **Features.** Per pulse 458 numbers: scaled frequency, scaled power, 200
   PW kernels and 256 TOA-lag kernels over its 40 neighbours.
3. **Model.** The window goes to `onnx_model_service` as one sequence; the
   transformer returns per pulse a 1024-d embedding, the nearest learned
   centre id and its similarity. (GPU + CPU, ~33 ms per window.)
4. **Known emitters.** Pulses whose centre similarity ≥ 0.6 are grouped by
   centre id; the group's medoid is refreshed when it fits better; pulses
   ≥ 0.7 (0.5 for MF06) with a library id are that known emitter →
   emitter/signal (first time) and one cluster row per window.
5. **Unknown pulses.** The rest are buffered (≥ 512 pulses, last 200 s),
   then: greedy clustering (pulses with embedding cosine ≥ 0.8 to a leader,
   ≥ 20 members, angle spread OK) → merge clusters whose mean vectors and
   bearings agree → score gate → match against earlier unknown tracks (same
   track = same signal) or create a new `unknown_ai` emitter.
6. **Write.** Per window: cluster rows with statistics; PRI per cluster
   (shortest well-populated interval, de-harmonized against its signal).

Steps 1–3 run ahead in parallel; steps 4–6 carry state (medoids, buffer,
tracks) and run in window order. Results equal Python master's on the
compared files; it is ~8× faster than Python but ~55× slower than classic,
and like Python it splits unknown emitters into more signals (602/608
pytest, exact count on 127/608).

## Which to use

* **Classic** — generator-style data, need for speed or no GPU; transparent
  rules, tuned on the 608-file corpus.
* **ONNX** — the trained model, i.e. what the Python product does; use when
  results must match the Python reference or for emitters the rules were
  not tuned on.

## Same Rust layers, different results — why

Both engines run the **same** Rust code for everything around detection:

| Layer | Shared? |
| --- | --- |
| CSV load → ClickHouse `signals.pdw` | same code (`refresh.rs`) |
| read pulses back from ClickHouse | same code (`ch_read::read_flow_pulses`) |
| write emitters / signals / clusters, cluster statistics (freq, PW, PRI, DOA…) | same writer (`onnx_flow::Db`) |
| schema, timing lines, reports | same |
| **detection — which pulses belong together** | **different** (`classic_v2.rs` vs `onnx_flow.rs`) |
| **what one cluster row means** | **different** |

So the input and the way results are stored are identical; only the decision
"these pulses are one emitter" and the size of a cluster row differ.

### Why the cluster counts differ (293,616 ONNX vs 51,518 classic)

| | ONNX | Classic |
| --- | --- | --- |
| one cluster row = | the pulses of one signal **inside one 512-pulse window** (Python's design: it works window by window) | the pulses of one emitter **inside one second** of time |
| rows per emitter in a 10 s file | ~200–400 (one per window it appears in) | ~10 |
| emitters per file | more: unknown emitters are often split into several signals, so each split piece gets its own rows | exact count on 606/608 files |

Example `mf03_mf04_best-0_plus0` (2 emitters, 10 s): ONNX 404 clusters
(2 signals × ~200 windows), classic 61 clusters (2 emitters × ~30 one-second
slices of their pulse trains).

### Why the pulse totals differ (54.1 M ONNX vs 56.2 M classic, of 59.3 M)

| Pulses left out | ONNX | Classic |
| --- | --- | --- |
| last partial window (< 512 pulses) | dropped (as Python) | kept |
| pulses with failed modulation measurement (`unk…`) | used | skipped |
| pulses that fit no group | below similarity thresholds, groups ≤ 20, low-score clusters | cells / groups with < 20 / 50 pulses |

### In one line

Classic groups by **measured parameters + pulse timing** and writes per
second; ONNX groups by the **model's learned similarity** and writes per
window. Same pipes in and out, different grouping and a different cluster
granularity.
