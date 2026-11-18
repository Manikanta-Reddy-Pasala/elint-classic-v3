# ELINT AI engine — Classic v3 / Classic v2 / ONNX

One script does everything: `./elint.sh` (no arguments = help).

## What you need
| Item | Where it goes |
|---|---|
| Rust (`cargo`), `python3` (+ venv), `docker` | installed on the machine |
| PDW CSV files | `data/pdw_generator/known_emitters/` and `unknown_emitters/` (or `DATA=/path`) |
| ONNX model `transformer_model_known_v16.onnx` (FP32, not int8) | `model/` (or `MODEL_FILE=/path`) |

## Start
```bash
./elint.sh setup        # build Rust, model-service venv (GPU on the ROCm box, else CPU), UI venv
./elint.sh start        # Postgres + ClickHouse (docker) + ONNX model service + UI -> prints http://<ip>:8800
./elint.sh status
./elint.sh stop         # stops model service + UI (databases keep running, data kept)
```

## Test
| What | Command | Expect |
|---|---|---|
| Classic accuracy, 608 files, v3 and v2 (no DB, no model) | `./elint.sh selftest quick` | `TOTAL 608 608` twice |
| + synthetic frequency-mode and PRI sets | `./elint.sh selftest` | 80 / 79 of 86, PRI 60/60 |
| One file through the engine | `./elint.sh run <file.csv> classic` (or `v2`, `onnx`) | emitters table, copy in `.run/results/` |
| UI | open the URL printed by `start`, Classic / ONNX | |

## Engines
| Setting | Engine |
|---|---|
| `CLASSIC` unset or `0` | ONNX (needs the model service, `ONNX_URL`, default `http://127.0.0.1:8000`) |
| `CLASSIC=1` | **Classic v3 (standard)** |
| `CLASSIC=1 CLASSIC_V3=0` | Classic v2 (fallback for real direction-finder data) |

Direct use: `CLASSIC=1 ./target/release/ai_processor --csv <file.csv>` (needs `PG_URL`, `CH_URL`, see `./elint.sh`).
`./target/release/classic_eval <file.csv>` = classic only, no database, JSON per group (frequency mode, PRI pattern, signature).

## Databases
- Default: local docker containers `elint_pg` (:15432) and `elint_ch` (:18123 / :19000), created once and reused.
- Your own databases: `DB_MODE=external PG_HOST=.. PG_PORT=.. PG_USER=.. PG_PASS=.. CH_HOST=.. CH_PORT=.. CH_NPORT=.. ./elint.sh start`
- Another Postgres database: `PG_DB=mytest ./elint.sh start` (created with the schema if missing; others untouched).
- Nothing is deleted unless you ask: `stop` keeps data, `stop db` stops the containers (data kept),
  only `./elint.sh clean --yes` removes the docker containers and their data.
- Each `ai_processor --csv` run replaces the previous results in its database (same as v2).
  Use another `PG_DB` to keep results; `./elint.sh run` keeps a copy in `.run/results/`.
- Schemas: `postgres/engine_schema.sql`, `clickhouse/engine_schema.sql`.
- Settings: `PG_PORT CH_PORT CH_NPORT MODEL_PORT UI_PORT PG_NAME CH_NAME DATA MODEL_FILE DEVICE`.

## Folders
| Folder | What |
|---|---|
| `pipeline/` | the engine (`ai_processor`, `classic_eval`): classic v3/v2 + ONNX flow |
| `config/` | classic naming rules (`classic_library.json`) |
| `onnx_model_service/` | ONNX model server (port 8000) used by the ONNX engine |
| `elint_ui/` | web UI (port 8800) |
| `tools/classic/` | classic self-test, test-set generators, scorers |
| `postgres/`, `clickhouse/` | database schemas and settings |
