# elint_ui — ELINT runner

Small web UI to run the `ai_elint` engines and look at the results. Kept
outside the engine and test repos; it only calls the `ai_processor` binary
and reads the result tables.


| Source | What happens |
| --- | --- |
| CSV file | `ai_processor --csv <file>`: load into ClickHouse, then process |
| folder | every `*.csv` under it (up to *max files*), one after another |
| database | `ai_processor --from-db`: process the pulses already in `signals.pdw` (e.g. loaded by the receiver or by *Load to DB only*) |
| *Load to DB only* | `ai_processor --csv <file> --load-only`: wipe + load, no processing |

Engines: **Python** (the master reference, `elint_reports/tools/py_time_run.py` in
`ai_elint_python_master/integration/venv`, writes `sigint_py`/`signals_py`; CSV only,
~1 min per file), **classic** (model-free) and/or **onnx** (needs `onnx_model_service`
running). Choosing both runs each file twice so the results can be compared
row by row.

Per run the table shows the expected result (when the file is in the test
repo's `expected_results.json`), the emitters found, whether names and count
match, signals, clusters, pulses in clusters, and time: **pipeline ms** is DB
read → engine → result writes (CSV loading is setup and not counted). Click a
row for the per-emitter table (pulses, clusters, PRI, PW, frequency, DOA) and
every layer time the engine reported. *History* lists the last 200 runs
(kept in `results.jsonl`).

## Run

```bash
cd ~/coderepo/elint_ui
./run.sh                     # http://<box>:8800
```

Needs Python 3 with `psycopg2` (present on the box), the Postgres/ClickHouse
containers (`elint_pg`, `elint_ch`) and the built engine
(`~/coderepo/ai_elint/target/release/ai_processor`). Settings (env):
`AI_PROCESSOR`, `DATA_DIR`, `EXPECTED_JSON`, `PG_URL`, `CH_URL`, `ONNX_URL`,
`UI_PORT`.

Runs execute one at a time — they share the databases, and each run clears
the result tables first.

## Files

| File | What it does |
| --- | --- |
| `server.py` | stdlib HTTP server + one worker thread; API: `GET /api/config`, `GET /api/browse?path=`, `POST /api/run`, `GET /api/jobs/<id>`, `GET /api/history`; runs `ai_processor`, parses `LAYER` lines, snapshots result tables with psycopg2 |
| `index.html` | the page: source picker (file / folder / database), engines, Run / Load-only, progress, results table with per-emitter details and stage times, history |
| `run.sh` | start with settings |
| `results.jsonl` | every finished run (history), generated |
| `logs/` | engine rotating logs, generated |

## Hardware panel

Live CPU / GPU / RAM / VRAM card at the top (GET /api/hw, sampled every 0.1 s from /proc/stat and /sys/class/drm/card0/device). Each run also stores CPU and GPU avg/peak while it ran (columns CPU peak / GPU peak; row details show averages). GPU sysfs path: env GPU_SYSFS.

## Compare demo

Top card. Buttons Compare 1/2/3 = preset file + engines (PRESETS in server.py). Click = show the file pulses (every k-th, ~6000 dots, coloured by generator truth) and run the engines (classic first). Each engine panel: pipeline time, bar, radars found vs expected, CPU/GPU peak, same pulses coloured by the radar that engine assigned (read from ClickHouse clusters.activity_ids right after the run; saved as logs/plot_<id>.json). Page load shows the last saved results without running. APIs: /api/presets, /api/sample?path=, /api/plot/<id>. Plain style on purpose.
