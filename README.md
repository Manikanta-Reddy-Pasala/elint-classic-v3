# ELINT AI engine — Classic v3 / Classic v2 / ONNX

One script does everything: `./run.sh` (one shot: build, databases, model service, UI, API).

## What you need
| Item | Where it goes |
|---|---|
| Rust (`cargo`), `python3` (+ venv), `docker` | installed on the machine |
| PDW CSV files | `data/pdw_generator/known_emitters/` and `unknown_emitters/` (or `DATA=/path`) |
| ONNX model `transformer_model_known_v16.onnx` (FP32, not int8) | `model/` (or `MODEL_FILE=/path`) |

## Start (one shot)
```bash
./run.sh                 # build (first time) + Postgres + ClickHouse (docker) + ONNX model service + UI + refresh API
                         # prints the UI URL (http://<ip>:8800)
./run.sh --pg-url postgresql://user:pass@host:5432/sigint --ch-url http://user:pass@host:8123   # your own databases
./run.sh --pg-db mytest  # another Postgres database (created if missing)
./run.sh --data /path/csvs --model /path/transformer_model_known_v16.onnx --ui-port 9000        # other paths/ports
./run.sh --no-model      # leave out a part: --no-model / --no-ui / --no-api
./run.sh status
./run.sh stop            # stops model service + UI + API (databases keep running, data kept)
./run.sh help            # all commands and options
```
Parts on their own: `./run.sh build | db | model | ui | api`, or `elint_ui/run.sh`, `onnx_model_service/run.sh`.

## Test
| What | Command | Expect |
|---|---|---|
| Classic accuracy, 608 files, v3 and v2 (no DB, no model) | `./run.sh selftest quick` | `TOTAL 608 608` twice |
| + synthetic frequency-mode and PRI sets | `./run.sh selftest` | 80 / 79 of 86, PRI 60/60 |
| One file through the engine | `./run.sh run <file.csv> classic` (or `v2`, `onnx`) | emitters table, copy in `.run/results/` |
| UI | open the URL printed by `./run.sh`, Classic / ONNX | |
| API | `curl -X POST localhost:8788/api/refresh-data -d '{"csv_filename":"<file.csv>"}'` | `{"status":"success"}` |

## Engines
| Setting | Engine |
|---|---|
| `CLASSIC` unset or `0` | ONNX (needs the model service, `ONNX_URL`, default `http://127.0.0.1:8000`) |
| `CLASSIC=1` | **Classic v3 (standard)** |
| `CLASSIC=1 CLASSIC_V3=0` | Classic v2 (fallback for real direction-finder data) |

Direct use: `CLASSIC=1 ./target/release/ai_processor --csv <file.csv>` (needs `PG_URL`, `CH_URL`, see `./run.sh`).
`./target/release/classic_eval <file.csv>` = classic only, no database, JSON per group (frequency mode, PRI pattern, signature).

## Databases
- Default: local docker containers `elint_pg` (:15432) and `elint_ch` (:18123 / :19000), created once and reused.
- Your own databases: `DB_MODE=external PG_HOST=.. PG_PORT=.. PG_USER=.. PG_PASS=.. CH_HOST=.. CH_PORT=.. CH_NPORT=.. ./run.sh start`
- Another Postgres database: `PG_DB=mytest ./run.sh start` (created with the schema if missing; others untouched).
- Nothing is deleted unless you ask: `stop` keeps data, `stop db` stops the containers (data kept),
  only `./run.sh clean --yes` removes the docker containers and their data.
- Each `ai_processor --csv` run replaces the previous results in its database (same as v2).
  Use another `PG_DB` to keep results; `./run.sh run` keeps a copy in `.run/results/`.
- Schemas: `postgres/engine_schema.sql`, `clickhouse/engine_schema.sql`.
- Settings: `PG_PORT CH_PORT CH_NPORT MODEL_PORT UI_PORT PG_NAME CH_NAME DATA MODEL_FILE DEVICE`.

## Folders
| Folder | What |
|---|---|
| `pipeline/` | the engine (`ai_processor`, `classic_eval`): classic v3/v2 + ONNX flow |
| `config/` | classic naming rules (`classic_library.json`) |
| `onnx_model_service/` | ONNX model server (port 8000) used by the ONNX engine |
| `elint_ui/` | web UI (port 8800) |
| `pipeline/src/bin/refresh_api.rs` | refresh API (port 8788): `POST /api/refresh-data` runs one CSV |
| `tools/classic/` | classic self-test, test-set generators, scorers |
| `postgres/`, `clickhouse/` | database schemas and settings |
