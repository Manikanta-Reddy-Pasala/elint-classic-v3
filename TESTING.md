# Testing this repo (Classic v3 and ONNX)

You need: this repo, the PDW CSV files, and (ONNX only) `transformer_model_known_v16.onnx` (FP32, not int8).
Commands run from the repo root. Put the CSVs at `data/pdw_generator/{known_emitters,unknown_emitters}/`
(or set `DATA=/path`, the folder that holds `pdw_generator/`).

## 1. Classic (latest code) - no database, no model
```bash
cargo build --release -p ai_processor --bin classic_eval
tools/classic/selftest.sh quick      # 608 files, classic v3 then v2   -> expect "TOTAL 608 608" twice
tools/classic/selftest.sh            # + synthetic freq-mode (80 / 79 of 86) and PRI sets (60/60)
./target/release/classic_eval data/pdw_generator/known_emitters/mf03_mf04_best-0_plus2.csv   # one file -> JSON
```
Apple Silicon only: `patch -p1 < tools/classic/hdbscan_arm_build_shim.diff` first (never commit it).

## 2. Database stack (needed for ai_processor: ONNX, and Classic the way the box runs it)
```bash
docker run -d --name elint_pg -p 127.0.0.1:15432:5432 \
  -e POSTGRES_USER=elint -e POSTGRES_PASSWORD=elint -e POSTGRES_DB=sigint \
  -v $PWD/postgres/init:/docker-entrypoint-initdb.d postgres:16

docker run -d --name elint_ch -p 127.0.0.1:18123:8123 -p 127.0.0.1:19000:9000 \
  -e CLICKHOUSE_USER=elint -e CLICKHOUSE_PASSWORD=elint -e CLICKHOUSE_DB=signals \
  -e CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1 \
  -v $PWD/clickhouse/init:/docker-entrypoint-initdb.d \
  -v $PWD/clickhouse/users.d/00_perf.xml:/etc/clickhouse-server/config.d/00_perf.xml \
  -v $PWD/clickhouse/users_profiles/old-analyzer.xml:/etc/clickhouse-server/users.d/old-analyzer.xml \
  clickhouse/clickhouse-server:24.8

export PG_URL=postgresql://elint:elint@127.0.0.1:15432/sigint
export CH_URL=http://127.0.0.1:18123 CLICKHOUSE_DB=signals CLICKHOUSE_USER=elint CLICKHOUSE_PASSWORD=elint
cargo build --release -p ai_processor --bin ai_processor
```
(`old-analyzer.xml` keeps ClickHouse float parsing identical to the box; without it ~1 % of values differ.)

## 3. ONNX
Start the model service (`box_workspace/onnx_model_service`, port 8000):
```bash
cd box_workspace/onnx_model_service
mkdir -p model && cp /path/to/transformer_model_known_v16.onnx model/
python3 -m venv .venv && .venv/bin/pip install numpy==1.26.4 onnxruntime   # CPU build
DEVICE=cpu CPU_WORKERS=2 CPU_THREADS=6 .venv/bin/python server.py &        # box GPU: ./setup.sh once, then ./run.sh
curl -s http://127.0.0.1:8000/health                                       # workers ready
cd ../..
```
Run one file:
```bash
CLASSIC=0 ONNX_URL=http://127.0.0.1:8000 ./target/release/ai_processor --csv data/pdw_generator/known_emitters/mf03_mf04_best-10_plus2.csv
docker exec elint_pg psql -U elint -d sigint -c "select name, count(*) from emitters group by name"
```
CPU is ~4x slower than the box GPU (box: ~6 s median per file). Box reference: ONNX 602/608 on the test judge.

## 4. Classic through ai_processor (same as the box/UI)
```bash
CLASSIC=1 ./target/release/ai_processor --csv <file.csv>                 # classic v3 (standard)
CLASSIC=1 CLASSIC_V3=0 ./target/release/ai_processor --csv <file.csv>    # classic v2 fallback
```
Each run wipes and reloads the tables. Engine switch: `CLASSIC` unset/0 = ONNX, 1 = classic.
`LAYER|<stage>|<ms>` lines on stdout give the timing per stage.

## 5. Stop
```bash
docker rm -f elint_pg elint_ch; pkill -f "server.py"     # note: on the box this also stops the UI (same file name)
```
