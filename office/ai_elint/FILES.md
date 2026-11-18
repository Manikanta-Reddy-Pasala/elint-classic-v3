# ai_elint — every file

**Active path** = used by `ai_processor` today (classic v2 / ONNX flow).
**Legacy** = kept for `ONNX_FLOW=legacy`, `CLASSIC_IMPL=v1` or the continuous
`elint_engine`; not used by the default one-shot path.

## pipeline/ (library `ai_processor` + binaries)

| File | Status | What it does |
| --- | --- | --- |
| `src/main.rs` | active | `ai_processor` binary: `--csv <f>` one-shot, `--load-only`, `--from-db`, no args = poller |
| `src/lib.rs` | active | crate root, module list, `AiState` (config, PG pool, CH writer, engine handle); also the legacy per-batch functions (`run_batch*`) |
| `src/refresh.rs` | active | one-shot flow: wipe tables → parse CSV → load `signals.pdw` + `signals.csv_input` → read pulses back from CH → dispatch to `classic_v2` or `onnx_flow`; `run_from_db`; CSV parser and CH DDL |
| `src/onnx_flow.rs` | active | **ONNX engine**: per-window features, model-service client (`/predict`), known-emitter path (id remap, medoid refinement, thresholds, signature map), unknown buffer + greedy clustering + merge + track cache (Python `custom_clustering.py`), cluster statistics (IQR PRI), DB writer `Db` (pinned connection, transactions, batched PG/CH inserts) |
| `src/classic_v2.rs` | active | **classic engine**: cells → components → split repair → two-way pulse-train coherence → library decision tree → writes (1 s cluster slices) |
| `src/ch_read.rs` | active | ClickHouse HTTP reads; `read_flow_pulses` (parallel TSV parse of `signals.pdw`, Float32 columns parsed as f32) and the poller's batch reader |
| `src/db.rs` | active | PG connection pool, harness-schema detection, CH keep-alive write sockets (`ChState::execute`), legacy insert helpers |
| `src/layer_timing.rs` | active | `LAYER|<stage>|<ms>` timing lines, DB-write timers |
| `src/config.rs` | active | env-driven `Config` (DB URLs, batch sizes, legacy ONNX settings) |
| `src/refresh_logger.rs` | active | console + rotating log file sink |
| `src/engine.rs` | legacy | embedding-engine trait (old remote-ONNX per-pulse client, classic bag-of-windows) |
| `src/model_fetch.rs` | legacy | old model-service HTTP client (had the `/predict` framing bug) |
| `src/onnx_predict.rs` | legacy | old per-batch prediction grouping (named emitters from CSV labels) |
| `src/global_match.rs` | legacy | old global-HDBSCAN finalize |
| `src/hdbscan.rs` | legacy | Rust HDBSCAN (cosine, rayon k-NN) |
| `src/cluster_merge.rs` | legacy | agglomerative merge + score gate for the old path |
| `src/known_match.rs` | legacy | EMA known-track matching for the poller path |
| `src/cache.rs` | legacy | in-memory known-emitter cache |
| `src/features.rs` | legacy | old feature builder (TOA diff without `abs`, wrong window clamp — not used by `onnx_flow`) |
| `src/inference.rs` | legacy | in-process ONNX runtime session pool (not used: the model runs in `onnx_model_service`) |
| `src/classic_classify.rs` | legacy | classic v1 (used CSV ground-truth labels — do not use for results) |
| `src/classic_embed.rs` | legacy | bag-of-windows embedding for the old classic embedding engine |
| `src/refresh_api_core.rs`, `src/bin/refresh_api.rs` | legacy | HTTP refresh API (`POST /api/refresh-data`) |
| `src/tests/*.rs`, `tests/unit.rs`, `tests/parallel_clustering.rs` | tests | unit tests (`cargo test --release --workspace`); `refresh_test::parse_csv_missing_id_column_fails` fails since before this work |

## engine/ (`elint_engine`, continuous poller — legacy path)

| File | What it does |
| --- | --- |
| `src/main.rs` | polls `signals.pdw`, band split, per-batch AI stage, global finalize |
| `src/band.rs` | S / L / X band routing |
| `src/ch_read.rs`, `src/config.rs`, `src/lib.rs` | CH reads, env config, test exports |
| `src/tests/*` | band + CH-read tests |
| `Dockerfile` | builder/runtime images |

## Other folders

| Path | What |
| --- | --- |
| `clickhouse/init/`, `clickhouse/users.d/` | canonical CH DDL / users |
| `postgres/init/` | canonical PG DDL |
| `data/` | sample PDW CSVs; `data/pdw_generator/` = 608-file test corpus (not in git) |
| `model/` | `transformer_model_known_v16.onnx` (FP32) + INT8 variant |
| `poller/` | separate small poller crate (older) |
| `debian/` | systemd unit + package scripts |
| `scripts/` | clippy gate, maintenance helpers |
| `docker-compose.yml`, `Dockerfile`, `run.sh` | container / refresh-API launch (legacy) |
| `README.md` | engines, run modes, settings, timing layers |
| `ARCHITECTURE.md`, `OPTIMIZATION.md` | data flow / threading; measured speed-ups |
| `RESULTS.md` | latest full-run results |
| `HOW_IT_WORKS.md` | short explanation of both engines, why their clusters differ |
| `IMPLEMENTATION.md` | every level of both engines (features, model, clustering, merging, naming, statistics, writes, threading) with diagrams |
| `docs/diagrams/*.svg` | the IMPLEMENTATION.md diagrams rendered (shared layers, data model, classic flow, ONNX flow, ONNX threading, side by side) |
| `AGENTS.md` | working guide for engineers and coding agents |
