# Implementation — every level, both engines

Companion to `HOW_IT_WORKS.md` (short version) and `FILES.md` (file list).
Numbers in brackets are the defaults; every one is an environment variable
(`FlowCfg` in `onnx_flow.rs`, `ClassicCfg` in `classic_v2.rs`).

---

## 0. Shared layers (both engines)

```mermaid
flowchart LR
    CSV["CSV file<br/>(test / replay)"] -->|"parse_pdw_csv<br/>refresh.rs"| LOAD["load<br/>signals.pdw + signals.csv_input<br/>(id = row number)"]
    RX["receiver<br/>(production)"] -.->|writes| PDW
    LOAD --> PDW[("ClickHouse<br/>signals.pdw")]
    PDW -->|"read_flow_pulses<br/>ch_read.rs (parallel TSV parse)"| FP["Vec&lt;FlowPulse&gt;<br/>arrival order"]
    FP -->|CLASSIC=1| CL["classic_v2::run"]
    FP -->|CLASSIC=0| ON["onnx_flow::run"]
    CL --> DB["Db writer (onnx_flow.rs)<br/>pinned PG connection, transactions,<br/>batched INSERTs, buffered CH rows"]
    ON --> DB
    DB --> PG[("Postgres<br/>emitters / signals / clusters")]
    DB --> CHC[("ClickHouse<br/>signals.clusters<br/>cluster_id → pulse ids")]
```

| Level | Code | Detail |
| --- | --- | --- |
| Input record | `FlowPulse` | `input_id` (row number, CH join key), `freq_hz`, `power_dbm`, `bandwidth_hz`, `duration_ns`, `doa_deg`, `snr_db`, `toa_ns`, `modulation` — no ground-truth columns |
| Read | `ch_read::read_flow_pulses` | `SELECT … FROM signals.pdw ORDER BY id`, TSV, rayon parse; Float32 columns parsed as `f32` |
| Emitter row | `Db::insert_emitter` | name = library id (`"1113"`, with `signature_matched_id`) or `"unknown_ai"` |
| Signal row | `Db::insert_signal` | first member's freq/BW/power/SNR (truncated), bearing, modulation id, duration = TOA span |
| Cluster row | `Db::write_clusters_gap` → `stats()` | count, first TOA, mean/min/max/std of freq, power, PW, BW, DOA; PRI statistics (§3.6); one multi-row INSERT … RETURNING id |
| Cluster → pulses | same | `(cluster_id, ['<input_id>', …])` rows buffered, sent in ≤ 8 MB INSERTs |
| Timing | `layer_timing` | `LAYER|<stage>|<ms>` per stage |

```mermaid
erDiagram
    EMITTERS ||--o{ SIGNALS : "emitter_id"
    SIGNALS ||--o{ CLUSTERS : "signal_id"
    CLUSTERS ||--|| CH_CLUSTERS : "id = cluster_id"
    CH_CLUSTERS }o--o{ PDW : "activity_ids = pdw.id"
    EMITTERS { bigint id  text name  int signature_matched_id }
    SIGNALS { bigint id  bigint emitter_id  float frequency_hz  float doa_deg  int modulation_type_id  float duration_ms }
    CLUSTERS { int id  int signal_id  int pdw_count  bigint frequency_hz  float pulse_width_us  float pulse_repetition_interval_us  float doa_deg }
    CH_CLUSTERS { uint64 cluster_id  array activity_ids }
    PDW { uint64 id  string time_of_arrival_ns  float center_frequency_hz  float duration_ns  float bandwidth_hz  float32 power_dbm  float32 direction_of_arrival_deg }
```

---

## 1. Classic engine (`classic_v2.rs`)

```mermaid
flowchart TD
    P["pulses (arrival order)"] --> F{"modulation measured?<br/>(not empty, not unk…)"}
    F -->|no| X1["skip"]
    F -->|yes| C1["1 · CELLS<br/>key = (modulation id, PW log-step 1 %,<br/>BW 1 MHz, carrier 1 MHz)<br/>parallel fold per thread"]
    C1 --> C2{"cell ≥ 20 pulses?"}
    C2 -->|no| X2["noise"]
    C2 -->|yes| C3["2 · COMPONENTS<br/>union-find over neighbour cells<br/>±1 PW step, ±1 BW, ±2 MHz"]
    C3 --> C4["3 · SPLIT REPAIR<br/>join if carrier ranges overlap (±3 MHz),<br/>BW ranges overlap (±1 MHz),<br/>PW medians within 2 %"]
    C4 --> C5["4 · PULSE-TRAIN COHERENCE<br/>all group pairs (rayon)<br/>join if max(cAB,cBA) ≥ 0.13<br/>and min(cAB,cBA) ≥ 0.04"]
    C5 --> C6{"group ≥ 50 pulses?"}
    C6 -->|no| X3["drop"]
    C6 -->|yes| C7["5 · NAMING<br/>decision tree → library id or none"]
    C7 --> C8["same library id → one emitter<br/>no id → one unknown_ai emitter per group"]
    C8 --> C9["6 · WRITE<br/>emitter + signal per emitter<br/>cluster rows per 1 s slice<br/>(slices < 50 pulses folded in)<br/>one PG transaction, one CH insert"]
```

### 1.1 Cells
`wc = round(ln(max(PW µs, 0.05)) / ln(1.01))`, `bc = round(BW / 1 MHz)`,
`fc = round(freq / 1 MHz)`; modulation interned to an integer. Cells under
`CLASSIC_MIN_CELL` [20] pulses are noise (spurious measurements).

### 1.2 Components
Union-find: a cell joins every existing cell at `(m, wc±1, bc±1, fc±2)`.
Handles measurement jitter that spreads one emitter over adjacent cells.

### 1.3 Split repair
For each component: 2nd/98th percentile of carrier and BW, median PW
(selection, O(n)). Join two components if both ranges overlap
(carrier ± `CLASSIC_MERGE_F_HZ` [3 MHz], BW ± `CLASSIC_MERGE_BW_HZ` [1 MHz])
and PW medians differ ≤ `CLASSIC_MERGE_PW_REL` [2 %].

### 1.4 Pulse-train coherence
```
coherence(A → B):
  sample ≤ 3000 pulses of A evenly
  for each: gap = toa_A − (latest toa_B before it), rounded to 1 µs
  share = (pulses in the 4 most common gap values) / sampled pulses
same emitter  ⇔  max(c(A→B), c(B→A)) ≥ 0.13  and  min(…) ≥ 0.04
```
Two modes of one radar (or two hops of an agile radar) are emitted on one
clock, so the gaps repeat; unrelated radars give near-uniform gaps
(measured on the corpus: same emitter max ≥ 0.19 / min ≥ 0.06, different
emitters ≤ 0.14 / ≤ 0.09). The two-way rule rejects a one-way "beat" between
PRIs (e.g. 2116 µs vs 220 µs).

### 1.5 Naming (library decision tree)
Group features: carrier mode (0.1 MHz bins), BW mode, PRI = most common
TOA interval (> 0.5 µs, 1 µs bins), PW 99th percentile, share of `lfm_up`.

| Rule (in order) | Library id |
| --- | --- |
| BW ≥ 6 MHz | none (unknown) |
| carrier 2898–2905 MHz | 1113 MF03 |
| carrier 2848–2852 | 1106 PSR3 |
| carrier 2748–2752 | PW ≥ 86 µs → 1107 PSR4, else 1104 PSR1 |
| carrier 2798–2822 | 1116 PSR5m2 |
| carrier 2858–2892 | lfm_up ≥ 80 % → 1109 MF04; carrier 2874–2876.5 and PRI < 140 µs → 1105 PSR2; else 1108 PSR5m1 |
| otherwise | none (unknown) |

### 1.6 Write
Members sorted by TOA; slices of `CLASSIC_CLUSTER_SEC` [1 s];
slices < `CLASSIC_MIN_SLICE` [50] pulses merge into the neighbour. PRI per
slice skips gaps < 1 µs (each pulse is reported by two receivers).

---

## 2. ONNX engine (`onnx_flow.rs` + `onnx_model_service`)

```mermaid
flowchart TD
    P["pulses (arrival order)"] --> W["1 · WINDOWS of 512<br/>(trailing partial window dropped)"]
    W --> FE["2 · FEATURES per window<br/>512 × 458 f32"]
    FE --> MS["3 · MODEL SERVICE (HTTP /predict)<br/>transformer, sequence (1, 512, 458)<br/>→ embedding 1024, centre id, similarity"]
    MS --> K0{"4 · KNOWN PATH<br/>similarity ≥ 0.6 ?"}
    K0 -->|no| U0
    K0 -->|yes| K1["group by centre id<br/>(ids 1117–1137 above 9 GHz → 1117)"]
    K1 --> K2["medoid refresh per id<br/>(keep the better of old medoid / new medoid)"]
    K2 --> K3{"refined similarity ≥ 0.7<br/>(0.5 for 1117)<br/>and id in library list?"}
    K3 -->|no| U0
    K3 -->|yes| K4["known emitter (name = id)<br/>first time: emitter + signal if > 20 pulses<br/>later: extend signal, update bearing"]
    K4 --> WR
    U0["5 · UNKNOWN PATH<br/>buffer pulses"] --> U1{"buffer ≥ 512 pulses?"}
    U1 -->|no| NEXT["next window"]
    U1 -->|yes| U2["keep last 200 s of the buffer"]
    U2 --> U3["greedy clustering<br/>leader + all free pulses with cosine ≥ 0.8<br/>≥ 20 members, beam score > 0.3"]
    U3 --> U4["merge clusters<br/>0.8·cos(mean vectors) + 0.2·angle score > 0.7<br/>connected components (BFS)"]
    U4 --> U5{"score = (medoid sim + beam score)/3 > 0.3 ?"}
    U5 -->|no| X["discard"]
    U5 -->|yes| U6["match unknown track cache<br/>0.8·cos(medoid, track) + 0.2·cos² bearing > 0.5"]
    U6 -->|match| U7["extend that signal"]
    U6 -->|no match, > 20 pulses| U8["new unknown_ai emitter + signal"]
    U7 --> WR
    U8 --> WR
    WR["6 · WRITE per window<br/>cluster rows + stats, CH pulse ids<br/>(PG commit every 32 windows)"] --> NEXT
    NEXT --> END["end of file: flush CH rows,<br/>de-harmonize cluster PRI"]
```

### 2.1 Windows
`n_win = ⌊pulses / 512⌋`; pulses of the last partial window are not processed
(the Python reference does the same).

### 2.2 Features (`window_features`) — 458 per pulse, f32

| Index | Feature | Formula |
| --- | --- | --- |
| 0 | frequency | `(f32(freq_hz) / 1e6 − 50) / 10950` |
| 1 | power | `(f32(power_dbm) + 100) / 100` |
| 2–201 | PW kernels | `exp(−(pw − c_k)² / (2 σ_k²))`, `pw = clamp(PW µs, 0.1, 300)`, 200 centres `c_k` log-spaced 0.1…300 µs |
| 202–457 | TOA-lag kernels | `Σ_j exp(−(|t_i − t_j| − c_k)² / (2 σ_k²))` over the 41 pulses `j` centred on `i` (centre clamped to [20, 491] inside the window), 256 centres log-spaced 10…5000 µs |

`σ_k = ⅔ · min(gap to previous centre, gap to next centre)`. Only kernels
whose exponent is < 30 are evaluated (walk outwards from the nearest centre);
the result equals the full 256-kernel sum to f32 precision (unit-tested).

### 2.3 Model call and service

```mermaid
sequenceDiagram
    autonumber
    participant M as onnx_flow::run (async, in order)
    participant W as worker threads ×8 (ONNX_CONCURRENCY)
    participant S as onnx_model_service (HTTP :8000)
    participant G as GPU session (MIGraphX)
    participant C as CPU session (12 threads)
    M->>W: spawn, each takes the next window index
    loop per window (up to 64 ahead of M)
        W->>W: features 512×458
        W->>S: POST /predict  X-Window-Size: 512  (n×458 f32)
        S->>S: queue window
        alt GPU free
            S->>G: run (1,512,458)
            G-->>S: emb, ids, sims
        else CPU free
            S->>C: run (1,512,458)
            C-->>S: emb, ids, sims
        end
        S-->>W: [u64 sizes] + 512×1024 f32 + 512 i64 + 512 f32
        W-->>M: (window idx, result)
    end
    M->>M: reorder buffer → process windows 0,1,2… in order (known + unknown paths, DB writes)
```

* Request: `n × 458` little-endian f32; header `X-Window-Size: 512`;
  each window runs as one sequence (the transformer attends across the 512
  pulses — never send pulses one by one).
* Reply: 24-byte header (byte sizes), embeddings `n × 1024` f32, centre ids
  `n` i64, similarities `n` f32.
* Client retries 5× (0.2 → 3.2 s backoff); service listen backlog 256.
* GPU MIGraphX ~36 ms/window, CPU ~132 ms/window; hybrid ≈ 33 ms/window.

### 2.4 Known-emitter path (Python `handle_known_emitters_vectorized`)
1. `id ∈ [1117, 1137]` and freq > 9 GHz → `1117`.
2. Pre-mask: similarity ≥ `ONNX_KNOWN_DOT` [0.6].
3. Per id (sorted): rows `E`; `sim_old = E·medoid_old` (or the model's
   similarities if no medoid yet); candidate medoid = `argmax_i Σ_j e_i·e_j`
   (computed as `argmax_i e_i · Σ_j e_j`); if mean(new) > mean(old) and
   rows > 30 → store the new medoid and use its similarities.
4. Known ⇔ refined similarity ≥ `ONNX_SIM_THR` [0.7] (`ONNX_SIM_THR_1117`
   [0.5] for 1117) **and** id ∈ {1104, 1105, 1106, 1107, 1108, 1109, 1113,
   1115, 1116, 1117}.
5. Per id in order of appearance: no signal yet and > 20 pulses → insert
   emitter (`name = id`, signature id) + signal; signal exists → extend its
   power/DOA history, update bearing and duration. One cluster row per id per
   window.

### 2.5 Unknown path (Python `handle_unknown_pulses_buffered` → `simple_clustering`)
1. Append non-known pulses (with embeddings) to the buffer; when it holds
   ≥ 512 pulses take those within 200 s of the newest and clear it.
2. **Greedy clustering**: for each free pulse `i`: members = free pulses
   with `e_i·e_j ≥ 0.8`; if ≥ 20 → label them; keep the cluster if its beam
   score > 0.3. Beam: peak = most powerful pulse, bearing = power-weighted
   mean of DOA of peak ±2 pulses; score = share of pulses within ±40° of it.
3. **Merge**: cluster vector = mean embedding; for every pair
   `0.8·cos + 0.2·angle_score(Δbearing) > 0.7`
   (angle score 1 up to 20°, cos² fall-off to 0 at 40°); connected
   components (BFS) become merged clusters; medoid =
   `argmax_i e_i·Σe`; score = (mean medoid similarity + beam score) / 3.
4. Score > 0.3 → **track match** against earlier unknown medoids:
   `0.8·cos(medoid, track) + 0.2·cos²(Δbearing/40°·π/2)`; > 0.5 → same
   track (extend its signal, or create one when the track reaches > 20
   pulses); otherwise new track → new `unknown_ai` emitter + signal when
   > 20 pulses. Bearing of the signal re-estimated from all its pulses.

### 2.6 Cluster statistics (both engines, `stats()`)
PRI from TOA intervals of the cluster's members (in member order):
skip gaps < 1 µs → IQR filter (≥ 4 values) → histogram with 3 µs bins.

| Estimator | Use |
| --- | --- |
| Python: tallest bin, mean of its points | classic; ONNX with `ONNX_PRI_ESTIMATOR=python` |
| fundamental: shortest bin with ≥ 25 % of the tallest, median of its points | ONNX default |

ONNX end of file: a cluster whose PRI is k × (its signal's median cluster
PRI), k ≥ 2 within 2 %, is divided by k (de-harmonization; statistics only).

### 2.7 Writes and transactions
PG `BEGIN` every 32 windows (`ONNX_COMMIT_EVERY`), all cluster rows of a
window in one INSERT; CH pulse-id rows buffered to 8 MB; final flush and
commit at the end.

---

## 3. Side by side

```mermaid
flowchart LR
    subgraph Shared["shared Rust layers"]
        R["CH read → FlowPulse"]
        Wr["Db writer + stats"]
    end
    subgraph Classic["classic_v2"]
        a1["cells"] --> a2["components"] --> a3["split repair"] --> a4["coherence"] --> a5["decision tree"]
    end
    subgraph ONNX["onnx_flow"]
        b1["512 windows"] --> b2["features 458"] --> b3["transformer 1024-d"] --> b4["known path"] --> b5["unknown buffer → greedy → merge → tracks"]
    end
    R --> a1
    R --> b1
    a5 -->|"1 cluster / emitter / second"| Wr
    b5 -->|"1 cluster / signal / window"| Wr
    b4 --> Wr
```

| Level | Classic | ONNX |
| --- | --- | --- |
| unit of work | whole file at once | 512-pulse windows, stateful in order |
| representation | measured parameters (mod, PW, BW, carrier) | 458 hand features → 1024-d learned embedding |
| grouping | parameter cells + timing coherence | cosine ≥ 0.8 greedy + merge + tracks |
| known radar | decision tree on carrier/PW/BW/PRI/modulation | model centre id + similarity thresholds |
| cluster row | emitter × 1 s | signal × window |
| hardware | all CPU cores (rayon) | GPU + CPU (model), CPU (features, post-processing) |
| time per file (median) | 108 ms | 6.0 s |
