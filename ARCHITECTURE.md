# ai_elint — Architecture

```
            receiver / CSV
                  │  (--csv: load; production: receiver writes)
                  ▼
     ClickHouse signals.pdw  (id = arrival order)
                  │  db_read  (ch_read::read_flow_pulses, parallel TSV parse)
                  ▼
        ┌──────────────── engine (CLASSIC=1 / 0) ─────────────────┐
        │                                                         │
   classic_v2.rs                                           onnx_flow.rs
   cells (mod, PW 1 %, BW 1 MHz, f 1 MHz)          512-pulse windows (CSV order)
     → components (neighbour cells)                  features (458-d, per window)
     → split repair (carrier+BW overlap, PW 2 %)     → onnx_model_service (HTTP,
     → pulse-train coherence (both directions)          GPU MIGraphX + CPU workers)
     → library decision tree (known ids)              → known path: id remap,
     → emitter / signal / clusters per 1 s slice         medoid refinement, thresholds
                                                       → unknown buffer (≥512 pulses,
                                                          200 s) → greedy cosine
                                                          clusters → merge → score
                                                          gate → track cache
        └─────────────────────────────┬───────────────────────────┘
                                      ▼
            Postgres: emitters, signals, clusters (one transaction per run /
                      per 32 windows)
            ClickHouse: signals.clusters (cluster id → pulse ids, few large inserts)
```

## ONNX engine threading

* `ONNX_CONCURRENCY` worker threads (default 8) each take the next window,
  compute its features (only the TOA kernels within reach of each lag are
  evaluated — identical to the full 256-kernel sum at f32), and call the
  model service. They may run up to `ONNX_MAX_AHEAD` windows ahead.
* The model service spreads windows over a MIGraphX GPU session and a CPU
  session (one shared queue). The GPU takes ~78 % of windows.
* Post-processing runs in window order on the async runtime (it carries
  state: medoids, track cache, unknown buffer), overlapped with the model
  calls of later windows.

## Classic engine threading

Cell building (parallel fold), split-repair summaries, coherence pairs and
naming run on rayon across all cores; result writes are one transaction plus
one batched cluster insert and one ClickHouse insert.

## Why the model is called per window

The transformer attends across the 512 pulses of a window; the Python
reference feeds `(1, 512, 458)`. Feeding pulses one at a time (`(n, 1, 458)`,
the old Rust behaviour) produces different embeddings and centre ids.
