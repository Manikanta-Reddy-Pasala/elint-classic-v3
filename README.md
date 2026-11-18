# ELINT — how to use and test (Classic v3 / Classic v2 / ONNX)

One script does everything: `./elint.sh` (run it with no arguments for help).

## What you need
| Item | Where it goes |
|---|---|
| Rust (`cargo`), `python3` (+ venv), `docker` | installed on the machine |
| PDW CSV files | `data/pdw_generator/known_emitters/` and `unknown_emitters/` (or `DATA=/path`) |
| ONNX model `transformer_model_known_v16.onnx` (FP32, not int8) | `model/` (or `MODEL_FILE=/path`) |
| Test-judge expected results (`ai_elint_tests/tests/test_data/*.csv`, from the box) | `ai_elint_tests/tests/test_data/` (or `EXPECTED_DIR=/path`) — judge only |

## Steps
```bash
./elint.sh setup        # build Rust, model-service venv (GPU on the box, else CPU), UI venv, judge image
./elint.sh start        # Postgres + ClickHouse (docker) + ONNX model service + UI -> prints http://<ip>:8800
./elint.sh run data/pdw_generator/known_emitters/mf03_mf04_best-0_plus2.csv classic   # or v2 / onnx
./elint.sh stop         # stops model service + UI (databases keep running, data kept)
```

## Test
| What | Command | Needs | Expect |
|---|---|---|---|
| Classic accuracy, 608 files, v3 + v2 | `./elint.sh selftest quick` | CSVs only | `TOTAL 608 608` twice |
| + synthetic freq-mode and PRI sets | `./elint.sh selftest` | + numpy | 80 / 79 of 86, PRI 60/60 |
| One file, any engine | `./elint.sh run <csv> classic\|v2\|onnx` | `start` | emitters table (saved in `.run/results/`) |
| Test judge (pytest), all 608 | `./elint.sh judge classic` (or `v2`, `onnx`) | `start` + expected CSVs | box: classic 608/608, ONNX 602/608 |
| Test judge, some files | `./elint.sh judge classic "mf03_mf04_best-0_plus2 or psr2"` | same | |
| UI | open the URL from `start`; Classic / ONNX buttons | `start` | |

## Engines
| Setting | Engine |
|---|---|
| `CLASSIC` unset or `0` | ONNX (needs the model service, `ONNX_URL`, default `http://127.0.0.1:8000`) |
| `CLASSIC=1` | **Classic v3 (standard)** |
| `CLASSIC=1 CLASSIC_V3=0` | Classic v2 (fallback for real direction-finder data) |

## Databases
- Default: local docker containers `elint_pg` (:15432) and `elint_ch` (:18123 / :19000), created once and reused.
- Your own databases: `DB_MODE=external PG_HOST=.. PG_PORT=.. PG_USER=.. PG_PASS=.. CH_HOST=.. CH_PORT=.. CH_NPORT=.. ./elint.sh start`
- Another Postgres database: `PG_DB=mytest ./elint.sh start` (created with the right schema if missing; others untouched).
- Nothing is deleted unless you ask: `stop` keeps data, `stop db` stops the containers (data kept),
  only `./elint.sh clean --yes` removes the docker containers and their data.
- Each `ai_processor --csv` run replaces the previous results in the database it uses (same as v2; the judge needs it).
  Use another `PG_DB` to keep results; `./elint.sh run` also keeps a copy in `.run/results/`.
- Schemas: `postgres/engine_schema.sql` and `clickhouse/engine_schema.sql` (the old `postgres/init`, `clickhouse/init`
  have integer ids / old columns and make the engine fail on a fresh database).
- Ports, names and paths: `PG_PORT CH_PORT CH_NPORT MODEL_PORT UI_PORT PG_NAME CH_NAME DATA MODEL_FILE EXPECTED_DIR`.

## Folders
| Folder | What |
|---|---|
| `pipeline/`, `engine/`, `config/` ... | the Rust engine (same layout as the v2 repo) |
| `onnx_model_service/` | ONNX model server (port 8000) used by the ONNX engine |
| `elint_ui/` | web UI (port 8800) |
| `ai_elint_tests/` | test judge (pytest), run by `./elint.sh judge` |
| `elint_reports/` | optional report tools (per-file reports, engine compare, timing) |
| `tools/classic/` | classic self-test, test-set generators, scorers |
| `CHANGES_2026-11-17.md`, `CHANGES_2026-11-18.md` | what changed vs v2 and the test results |

---

# ai_elint

Rust ELINT pipeline: reads pulses (PDWs) from **ClickHouse**, finds the
emitters, and writes emitters / signals / clusters to **Postgres** and the
pulse-to-cluster mapping to **ClickHouse** (`signals.clusters`).

Two engines, picked with one environment variable:

| Engine | Select | What it does |
| --- | --- | --- |
| **classic** | `CLASSIC=1` | Model-free. Groups pulses on measured PDW fields (carrier, PW, BW, intra-pulse modulation), joins the modes of multi-mode radars and the hops of frequency-agile ones by pulse-train timing coherence, names known radars with a library decision tree. `pipeline/src/classic_v2.rs` |
| **onnx** | `CLASSIC=0` (default) | Window-by-window port of the Python reference (`ai_elint_python_master/integration/app`): features → transformer model → known-emitter path (centre ids, medoid refinement) → unknown buffer → greedy cosine clustering → merge → track cache. The model runs in the separate **`onnx_model_service`** (GPU + CPU), called over HTTP. `pipeline/src/onnx_flow.rs` |

Neither engine reads the CSV's ground-truth `emitter_id` column or the
generator's `pulse_repetition_interval_ns` column — only what a receiver
measures.

ONNX parity: emitters, signals, clusters and pulse membership are identical
to Python master (20/20 files). Only the cluster PRI *statistic* differs by
default (fundamental estimator + de-harmonization, see settings), because
Python's per-cluster PRI reads a multiple of the true PRI when a cluster holds
every 2nd/3rd pulse; `ONNX_PRI_ESTIMATOR=python ONNX_PRI_HARMONIC=0
ONNX_PRI_MIN_GAP_US=0` reproduces Python's numbers exactly.

Related folders in `~/coderepo`: `onnx_model_service/` (model server),
`ai_elint_tests/` (pytest harness + reports), `elint_ui/` (runner UI),
`ai_elint_python_master/` (Python reference).

## Layout

| Path | What |
| --- | --- |
| `pipeline/src/onnx_flow.rs` | ONNX engine (features, model client, known/unknown logic, DB writers) |
| `pipeline/src/classic_v2.rs` | classic engine |
| `pipeline/src/refresh.rs` | one-shot flow: wipe → load CSV into CH → read back → engine |
| `pipeline/src/ch_read.rs` | ClickHouse readers (`read_flow_pulses`: parallel TSV parse) |
| `pipeline/src/layer_timing.rs` | `LAYER|<stage>|<ms>` timing lines |
| `pipeline/src/main.rs` | `ai_processor` binary |
| `engine/` | `elint_engine` continuous poller (older global-HDBSCAN path) |
| `data/` | sample CSVs; `data/pdw_generator/` = the 608-file test corpus (8.9 GB, not in git) |
| `model/` | model files (also used by `onnx_model_service`) |

Older modules (`global_match.rs`, `classic_classify.rs`, `hdbscan.rs`, …)
are kept for `ONNX_FLOW=legacy` / `CLASSIC_IMPL=v1` comparisons only.

## Build

```bash
cd ~/coderepo/ai_elint
RUSTFLAGS="-C target-cpu=native" cargo build --release
# → target/release/ai_processor, target/release/elint_engine
```

Stable Rust ≥ 1.88 is enough (`target-cpu=native` gives AVX-512 on the
Ryzen AI 9 HX 370).

## Run

Databases (docker on the box, auto-start): Postgres `elint_pg` :15432
(`elint/elint`, db `sigint`), ClickHouse `elint_ch` :18123 / :19000
(`elint/elint`, db `signals`).

```bash
export PG_URL=postgresql://elint:elint@127.0.0.1:15432/sigint
export CH_URL=http://127.0.0.1:18123 CLICKHOUSE_USER=elint CLICKHOUSE_PASSWORD=elint

# classic
CLASSIC=1 ./target/release/ai_processor --csv data/pdw_generator/known_emitters/mf03_mf04_best-0_plus0.csv

# onnx (start ../onnx_model_service/run.sh first)
CLASSIC=0 ONNX_URL=http://127.0.0.1:8000 ./target/release/ai_processor --csv <file.csv>
```

| Mode | Command | Does |
| --- | --- | --- |
| one-shot | `ai_processor --csv f.csv` | wipe all tables → load CSV into `signals.pdw` + `signals.csv_input` → read pulses back from CH → engine → writes |
| load only | `ai_processor --csv f.csv --load-only` | wipe + load, no processing |
| from DB | `ai_processor --from-db` | clear result tables, process whatever is in `signals.pdw` |
| poller | `ai_processor` (no args) / `elint_engine` | continuous mode (older path) |

`signals.pdw.id` is the file row number (arrival order) and equals
`signals.csv_input.id`, the key the test harness joins cluster pulses on.

## Settings

| Var | Default | Meaning |
| --- | --- | --- |
| `CLASSIC` | 0 | 1 = classic engine |
| `ONNX_URL` | `http://127.0.0.1:8000` | model service |
| `ONNX_CONCURRENCY` | 8 | windows in flight to the model service (features are computed in the same workers) |
| `ONNX_MAX_AHEAD` | 64 | how far model calls may run ahead of in-order post-processing |
| `ONNX_COMMIT_EVERY` | 32 | windows per Postgres transaction |
| `ONNX_PRI_MIN_GAP_US` | 1.0 | cluster PRI ignores intervals below this (same pulse from two receivers); 0 = Python's exact formula |
| `ONNX_PRI_ESTIMATOR` | `fundamental` | cluster PRI = median of the shortest well-populated 3 µs interval bin; `python` = Python's mean of the tallest bin |
| `ONNX_PRI_HARMONIC` | 1 | end of run: clusters whose PRI is k × (signal's median cluster PRI), k ≥ 2 within 2 %, are divided by k |
| `ONNX_KNOWN_DOT`, `ONNX_SIM_THR`, `ONNX_SIM_THR_1117`, `ONNX_CLUSTER_THR`, `ONNX_MERGE_THR`, … | Python `config.py` values | ONNX thresholds (see `FlowCfg` in `onnx_flow.rs`) |
| `CLASSIC_COH_THR` / `CLASSIC_COH_MIN` | 0.13 / 0.04 | pulse-train coherence: max / min of the two directions to join two groups |
| `CLASSIC_CLUSTER_SEC` | 1.0 | classic writes one cluster row per emitter per time slice |
| `CLASSIC_MIN_SLICE` | 50 | thinner slices fold into their neighbour |
| `CLASSIC_*` (others) | see `ClassicCfg` | cell sizes, merge gates |
| `FLOW_READ` | (CH) | `memory` = skip the ClickHouse read-back and use the parsed CSV rows |
| `ONNX_FLOW=legacy`, `CLASSIC_IMPL=v1` | – | old engines, for comparison only |

## Timing per layer

Every run prints one line per layer on stdout, `LAYER|<stage>|<ms>`:

| Stage | Engine | Meaning |
| --- | --- | --- |
| `wipe_tables`, `load_csv`, `csv_rows` | both | test setup (not processing time) |
| `db_read`, `pulses_read` | both | read the pulses back from ClickHouse |
| `classic.cells`, `.components`, `.split_repair`, `.coherence`, `.naming`, `.detect_total` | classic | detection steps |
| `onnx.features_cpu_sum`, `onnx.model_service_sum`, `onnx.model_per_window`, `onnx.windows` | onnx | feature + model calls (summed over the parallel workers) |
| `onnx.postprocess_inorder` | onnx | known/unknown logic + writes, in window order |
| `<engine>.db_writes`, `.db_pg_clusters`, `.db_ch_clusters` | both | result writes (PG cluster insert / CH insert split) |
| `<engine>.total_wall`, `engine_total` | both | engine wall time / whole run |

**Pipeline time** = `engine_total − wipe_tables − load_csv` (DB read →
results written). The test harness report and the UI show it per file.

## Validation

- **Test repo (the judge):** `ai_elint_tests` pytest harness, both engines,
  all 608 files — run via `~/coderepo/elint_reports/run_engine.sh`; see the
  results section below.
- **ONNX parity with Python master:** `~/coderepo/elint_reports/tools/compare.sh` /
  `parity.sh` run Python master and Rust ONNX on the same file and diff
  emitters, signals, clusters and pulses per signal.
- Unit tests: `cargo test --release --workspace`.

## Results

See `RESULTS.md` (filled from the latest full runs).
