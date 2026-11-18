# ELINT UI (port 8800)

Web page to run the engines on a CSV file, a folder, or the pulses already in the database,
and to compare **Rust classic (v3)** with **Rust ONNX** side by side.

Start: `../run.sh` (starts everything) or `./run.sh` here (UI only; creates `.venv` with psycopg2 on first run).

Defaults point inside this repo: binary `../target/release/ai_processor`, CSVs `../data/`,
expected results `expected_results.json`. Settings (env): `AI_PROCESSOR`, `DATA_DIR`, `EXPECTED_JSON`,
`PG_URL`, `CH_URL`, `ONNX_URL` (model service, needed for ONNX), `UI_PORT`.

- **Compare** buttons: ready-made files (from `data/pdw_generator/known_emitters/`), ONNX vs classic.
- **Manual run**: pick a file/folder or "database", choose engines, see emitters, names, timings per layer.
