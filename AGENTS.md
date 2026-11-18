# AGENTS.md — ai_elint

Guide for anyone (engineer or coding agent) changing this repo.

## What matters

* Two engines behind `CLASSIC=1|0`: `pipeline/src/classic_v2.rs` and
  `pipeline/src/onnx_flow.rs`. Everything else listed as *legacy* in
  `FILES.md` is not on the default path.
* **Judge = `ai_elint_tests` pytest harness** (608 files). ONNX must also stay
  identical to Python master (`ai_elint_python_master`) — check with
  `~/coderepo/elint_reports/tools/parity.sh` / `compare.sh`.
* **Never use the CSV's `emitter_id` or `pulse_repetition_interval_ns`
  columns** in an engine — they are ground truth, absent in real data.
* The model runs in `~/coderepo/onnx_model_service` (HTTP). Do not load
  onnxruntime in this repo.

## Build / test / run (on the box, never on a laptop)

```bash
cd ~/coderepo/ai_elint
RUSTFLAGS="-C target-cpu=native" cargo build --release
cargo test --release --workspace            # 1 known pre-existing failure
~/coderepo/elint_reports/tools/run_one.sh classic <csv>     # one file, prints RESULT line
~/coderepo/elint_reports/run_engine.sh onnx <tag> [-k expr] # pytest + per-file report
```

DBs: `docker start elint_pg elint_ch` (auto-start is on). ClickHouse user
profile `users.d/old-analyzer.xml` in the container sets
`allow_experimental_analyzer=0` (test queries need it) and
`precise_float_parsing=1` (bit-exact pulse read-back).

## Invariants / traps (each one cost a debugging session)

1. The model is a **sequence model**: one request = whole 512-pulse windows
   (`X-Window-Size: 512`), never pulse-by-pulse.
2. Features: TOA lag is `abs(toa_i − toa_j)`, window centre clamped to
   `[20, 491]` (41 rows) — `onnx_flow::window_features`.
3. `signals.pdw.id` = file row number = `signals.csv_input.id`; the CSV's own
   `id` repeats per emitter block. `ORDER BY id` = arrival order.
4. `pdw` INSERT value order must match its column list (DOA/SNR were swapped
   once). Never format strings with `{:0.6}` (truncates to 6 chars).
5. Float32 CH columns (power, DOA, SNR) are parsed as `f32` then widened.
6. Each pulse appears twice (two receivers, identical TOA) — PRI statistics
   skip intervals < 1 µs (`*_PRI_MIN_GAP_US`).
7. ONNX post-processing is stateful (medoids, unknown buffer, track cache) —
   keep it in window order; only features + model calls run ahead.
8. Keep DB writes batched: one PG transaction per run (classic) / per 32
   windows (ONNX); CH cluster rows buffered into large inserts.
9. Classic coherence must hold **both ways** (max ≥ 0.13, min ≥ 0.04): a
   one-way peak is a PRI beat between unrelated emitters.
10. ONNX model calls retry 5× with backoff; the model service listen
   backlog is 256 (the default 5 reset connections under load).
11. ONNX cluster PRI: fundamental bin + median + de-harmonization are
   statistics only — never let them change cluster membership.

## Conventions

Conventional commits (`COMMIT_RULE.md`); work on a branch, never push to
master without the owner's OK. Every new engine stage emits a `LAYER` line.
