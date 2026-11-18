# ai_elint_tests

Test harness for the ELINT AI pipeline. The engine has migrated from Python
to Rust: this repo provides the **pytest harness** and the test data (CSV
files); the engine itself lives in the companion `ai_elint` repo
(branch `aiforge/rust-ai-integration`).

The legacy Python `helpers/app*` folders are **deprecated** — the Rust
engine replaces them.

---

## How the test harness works

```
pytest (this repo)
   │
   │  AI_ENGINE=rust
   ▼
ai_processor --csv <file>        (Rust binary, from ai_elint repo)
   │
   ├─ 1. Wipe DBs (PG: clusters, signals, emitters; CH: pdw, clusters, csv_input)
   ├─ 2. Load CSV → ClickHouse signals.pdw + signals.csv_input
   ├─ 3. Run AI pipeline: features → ONNX → known/unknown match → HDBSCAN → merge
   └─ 4. Write results → PG (emitters, signals, clusters) + CH (signals.clusters)
   │
   ▼
pytest asserts: PG counts match expectations.json
```

Data loading is done **entirely in Rust** (`pipeline/src/refresh.rs` →
`stage_load_csv_input`). No Python is involved in the engine run.

---

## 1. Build the Rust engine

The engine binary is in the `ai_elint` repo, not here.

**Prerequisites:** Rust nightly `2025-09-01` + AVX-512 CPU, or Docker.

```bash
cd ai_elint
RUSTFLAGS="-C target-feature=+avx512f" \
cargo +nightly-2025-09-01 build --release
# → target/release/ai_processor
```

Or build via Docker (no local toolchain needed):

```bash
cd ai_elint
cp /opt/ai_elint/onnxruntime/libonnxruntime.so engine/ort.so
docker build --target builder -f engine/Dockerfile -t elint-builder .
mkdir -p bin
docker create --name elint-tmp elint-builder true
docker cp elint-tmp:/build/target/release/ai_processor ./bin/ai_processor
docker rm elint-tmp
```

---

## 2. Run one CSV (one-shot mode)

```bash
export PG_URL="postgresql://elint:elint@<host>:5432/sigint"
export CH_URL="http://<host>:18123"
export CLICKHOUSE_DB=signals
export CLICKHOUSE_USER=elint
export CLICKHOUSE_PASSWORD=elint
export ONNX_PATH=/path/to/transformer_model_known_v16.onnx
export CLASSIC=1

./target/release/ai_processor --csv path/to/pdw.csv
```

Exit code 0 = success. The binary wipes the DBs, loads the CSV, runs the
full AI pipeline, and writes results before exiting.

---

## 2a. Testing on the VM (10.10.120.13)

All test runs happen on the lab VM. Do **not** run tests locally.

### VM layout

| Path | What |
|---|---|
| `/home/sysadm/ai_elint_rust/target/release/ai_processor` | Rust engine binary (release build) |
| `/home/sysadm/ai_elint_tests_work/` | Test harness checkout |
| `/home/sysadm/ai_elint_tests_work/tests/data/input_csv_files/pdw_generator/known_emitters/` | 448 known-emitter CSVs |
| `/home/sysadm/ai_elint_tests_work/tests/data/input_csv_files/pdw_generator/unknown_emitters/` | 160 unknown-emitter CSVs |
| `/home/sysadm/models/transformer_model_known_v16.onnx` | ONNX model |
| `elint_pg` (docker) | Postgres on port 15432 |
| `elint_ch` (docker) | ClickHouse on port 18123 |

### Prerequisites on the VM

```bash
# Confirm the DB containers are running
docker ps | grep -E 'elint_pg|elint_ch'

# Confirm the binary exists
ls -la /home/sysadm/ai_elint_rust/target/release/ai_processor
```

### Copy test CSVs to the VM (if not already there)

All 608 CSVs are already deployed on the VM at
`/home/sysadm/ai_elint_tests_work/tests/data/input_csv_files/pdw_generator/`
(448 known + 160 unknown). To (re)copy from the local machine:

```bash
# All 608 files in one shot (from the repo root)
scp tests/data/input_csv_files/pdw_generator/known_emitters/*.csv \
  sysadm@10.10.120.13:/home/sysadm/ai_elint_tests_work/tests/data/input_csv_files/pdw_generator/known_emitters/

scp tests/data/input_csv_files/pdw_generator/unknown_emitters/*.csv \
  sysadm@10.10.120.13:/home/sysadm/ai_elint_tests_work/tests/data/input_csv_files/pdw_generator/unknown_emitters/
```

Verify the copy:

```bash
ssh sysadm@10.10.120.13 \
  'ls /home/sysadm/ai_elint_tests_work/tests/data/input_csv_files/pdw_generator/known_emitters/*.csv | wc -l'
# expect: 448

ssh sysadm@10.10.120.13 \
  'ls /home/sysadm/ai_elint_tests_work/tests/data/input_csv_files/pdw_generator/unknown_emitters/*.csv | wc -l'
# expect: 160
```

### Run one CSV on the VM

SSH into the VM and use the `vm_oneshot_one.sh` helper (or run inline):

```bash
ssh sysadm@10.10.120.13

export PG_URL="postgresql://elint:elint@127.0.0.1:15432/sigint"
export CH_URL="http://127.0.0.1:18123"
export CLICKHOUSE_DB=signals CLICKHOUSE_USER=elint CLICKHOUSE_PASSWORD=elint
export ONNX_PATH=/home/sysadm/models/transformer_model_known_v16.onnx
export CLASSIC=1
export ONNX_SESSION_COUNT=1 ONNX_THREADS=1

cd /home/sysadm/ai_elint_rust

# Wipe DBs (same as the pytest fixtures)
docker exec elint_pg psql -U elint -d sigint -q -c \
  "truncate clusters, signals, emitters, signals_clusters restart identity cascade"
docker exec elint_ch clickhouse-client --user elint --password elint -q \
  "TRUNCATE TABLE signals.pdw; TRUNCATE TABLE signals.clusters; TRUNCATE TABLE signals.csv_input;"

# Run
timeout 600 ./target/release/ai_processor \
  --csv /home/sysadm/ai_elint_tests_work/tests/data/input_csv_files/pdw_generator/known_emitters/mf03_mf04_best-0_plus0.csv

# Check results
docker exec elint_pg psql -U elint -d sigint -t -A -c "SELECT count(*) FROM emitters"
docker exec elint_pg psql -U elint -d sigint -t -A -c "SELECT name FROM emitters"
docker exec elint_pg psql -U elint -d sigint -t -A -c "SELECT count(*) FROM signals"
docker exec elint_pg psql -U elint -d sigint -t -A -c "SELECT count(*) FROM clusters"
docker exec elint_ch clickhouse-client --user elint --password elint \
  -q "SELECT count() FROM signals.pdw"
docker exec elint_ch clickhouse-client --user elint --password elint \
  -q "SELECT count() FROM signals.clusters"
```

### How to test in the VM (quick start)

1. **Prereqs** — DB containers up + binary present:
   ```bash
   ssh sysadm@10.10.120.13
   docker ps | grep -E 'elint_pg|elint_ch'
   ls -la /home/sysadm/ai_elint_rust/target/release/ai_processor
   ```
2. **Run one file** (wipe → run → print counts):
   ```bash
   bash /home/sysadm/vm_oneshot_one.sh \
     /home/sysadm/ai_elint_tests_work/tests/data/input_csv_files/pdw_generator/known_emitters/mf03_mf04_best-0_plus0.csv
   # → RESULT|mf03_mf04_best-0_plus0.csv|rc=0|t=0.5s|pg_em=2|pg_sig=2|pg_clu=2|ch_pdw=105181|ch_clu=2
   ```
3. **Run all 608 files** (serial, ~1.5 s/file ≈ 17 min):
   ```bash
   nohup bash /home/sysadm/run608_local.sh > /tmp/run608_stdout.log 2>&1 &
   tail -f /tmp/run608.log            # progress
   cat /tmp/run608_results.txt        # one RESULT line per file
   ```
4. **Verify** against `/home/sysadm/expectations.json`
   (pass = `rc=0` and `pg_em == exp_em`):
   ```bash
   python3 verify608.py   # from the local scratch dir, reads both files over SSH
   ```

Latest full-batch result (aiforge/rust-ai-integration binary, 2026-10-29):
**608/608 rc=0; 605/608 exact emitter count match** — 3 files give 6 emitters
instead of 7 (`mf03_psr5m2_best-10_plus5`, `cta1_cta2_best-6_plus5`,
`cta1_nav1_best-3_plus5`), reproduced on fresh rerun.

### Run a batch of files on the VM (serial, with Artifactory download)

The 608 CSVs are hosted in Artifactory and are **not** committed to this
repo. The batch helper downloads each file on demand (cached in
`/tmp/dl_<name>`).

Helper scripts (deployed to the VM):

| Script | What it does |
|---|---|
| `/home/sysadm/vm_oneshot_one.sh <csv>` | Wipe DBs → run `ai_processor --csv` → print `RESULT\|…\|pg_em=\|pg_sig=\|pg_clu=\|ch_pdw=\|ch_clu=` |
| `/home/sysadm/dl_one.sh <name> <category>` | Download from Artifactory (if not cached) → call `vm_oneshot_one.sh` |
| `/home/sysadm/run10_serial.sh` | Run 5 known + 5 unknown files serially, compare vs `expectations.json` |
| `/home/sysadm/run608.sh` | Run all 608 files serially, results in `/tmp/run608_results.txt` |

Start the full batch:

```bash
ssh sysadm@10.10.120.13
nohup bash /home/sysadm/run608.sh > /tmp/run608_stdout.log 2>&1 &
# monitor: tail -f /tmp/run608.log
# results:  cat /tmp/run608_results.txt
```

Each line in the results file is pipe-delimited:

```
RESULT|mf03_mf04_best-0_plus0.csv|rc=0|t=0.54s|pg_em=2|pg_sig=2|pg_clu=2|ch_pdw=105181|ch_clu=2
```

Pass criteria per file: `rc=0` and `pg_em == exp_em` from
`/home/sysadm/expectations.json`.

### Run a batch of files on the VM (serial, local CSVs)

If the CSVs are already on the VM (e.g. copied via `scp`):

```bash
ssh sysadm@10.10.120.13
cd /home/sysadm/ai_elint_rust

export PG_URL="postgresql://elint:elint@127.0.0.1:15432/sigint"
export CH_URL="http://127.0.0.1:18123"
export CLICKHOUSE_DB=signals CLICKHOUSE_USER=elint CLICKHOUSE_PASSWORD=elint
export ONNX_PATH=/home/sysadm/models/transformer_model_known_v16.onnx
export CLASSIC=1 ONNX_SESSION_COUNT=1 ONNX_THREADS=1

CSV_DIR=/home/sysadm/ai_elint_tests_work/tests/data/input_csv_files/pdw_generator/known_emitters

for f in "$CSV_DIR"/*.csv; do
  # wipe
  docker exec elint_pg psql -U elint -d sigint -q -c \
    "truncate clusters, signals, emitters, signals_clusters restart identity cascade" 2>/dev/null
  docker exec elint_ch clickhouse-client --user elint --password elint -q \
    "TRUNCATE TABLE signals.pdw; TRUNCATE TABLE signals.clusters; TRUNCATE TABLE signals.csv_input;" 2>/dev/null

  # run
  timeout 600 ./target/release/ai_processor --csv "$f" > /tmp/eng.log 2>&1
  rc=$?

  # collect
  em=$(docker exec elint_pg psql -U elint -d sigint -t -A -c "select count(*) from emitters" 2>/dev/null | tr -d '[:space:]')
  sg=$(docker exec elint_pg psql -U elint -d sigint -t -A -c "select count(*) from signals" 2>/dev/null | tr -d '[:space:]')
  cl=$(docker exec elint_pg psql -U elint -d sigint -t -A -c "select count(*) from clusters" 2>/dev/null | tr -d '[:space:]')

  echo "RESULT|$(basename $f)|rc=$rc|em=$em|sig=$sg|clu=$cl"
done
```

### Run the pytest harness on the VM

The pytest harness runs inside the `elint_pytest` Docker container (host
networking mode, `/work` → `/home/sysadm/ai_elint_tests_work`):

```bash
ssh sysadm@10.10.120.13
cd /home/sysadm/ai_elint_tests_work

# Run one specific test case with the Rust engine
docker exec -w /work \
  -e AI_ENGINE=rust \
  -e RUST_AI_TIMEOUT=600 \
  -e CH_HOST=127.0.0.1 \
  -e CH_PORT=19000 \
  -e ARTIFACTORY_USER=manikanta.pasala \
  -e "ARTIFACTORY_PASSWORD=<password>" \
  elint_pytest python3 -m pytest \
  "tests/test_pdw_generator.py::TestPDWGeneratorFiles::test_known_emitters_detection_source_omni[data/input_csv_files/pdw_generator/known_emitters/mf03_mf04_best-0_plus0.csv-1113-1109-1-2-MF03-MF04-2-0]" \
  -q --no-header
```

**Required env vars:**

| Var | Value | Why |
|---|---|---|
| `AI_ENGINE` | `rust` | Tells `conftest.py` to call `ai_processor --csv` instead of Python `app.py` |
| `CH_HOST` | `127.0.0.1` | Container uses host networking; reach CH on the VM host |
| `CH_PORT` | `19000` | ClickHouse native port published on the VM host (9000 inside `elint_ch`) |
| `ARTIFACTORY_USER` | `manikanta.pasala` | Download test CSVs from Artifactory (fixture deletes them after each run) |
| `ARTIFACTORY_PASSWORD` | `<password>` | Same |
| `RUST_AI_TIMEOUT` | `600` | Per-file engine timeout in seconds |

The test node ID is the full parametrised form — find it with:

```bash
docker exec -w /work elint_pytest python3 -m pytest \
  tests/test_pdw_generator.py::TestPDWGeneratorFiles::test_known_emitters_detection_source_omni \
  --collect-only -q 2>/dev/null | head -5
```

Run multiple specific tests in one command (they run serially):

```bash
docker exec -w /work \
  -e AI_ENGINE=rust -e RUST_AI_TIMEOUT=600 \
  -e CH_HOST=127.0.0.1 -e CH_PORT=19000 \
  -e ARTIFACTORY_USER=manikanta.pasala -e "ARTIFACTORY_PASSWORD=<password>" \
  elint_pytest python3 -m pytest \
  "tests/.../test_...[...known...0]" \
  "tests/.../test_...[...known...2]" \
  "tests/.../test_...[...unknown...0]" \
  -q --no-header
# → 3 passed
```

### Updating the Rust binary on the VM

After making changes to the `ai_elint` repo and pushing to
`aiforge/rust-ai-integration`:

```bash
# On the VM — pull the latest source and rebuild
cd /home/sysadm/ai_elint_rust
git pull origin aiforge/rust-ai-integration
RUSTFLAGS="-C target-feature=+avx512f" \
cargo +nightly-2025-09-01 build --release
# → target/release/ai_processor (updated)
```

Or copy a pre-built binary from a builder machine:

```bash
scp target/release/ai_processor sysadm@10.10.120.13:/home/sysadm/ai_elint_rust/target/release/
```

---

## 3. Run the pytest harness with the Rust engine

```bash
cd ai_elint_tests
uv sync          # or: pip install -e .

# Run a single test with the Rust engine
AI_ENGINE=rust pytest tests/test_pdw_generator.py \
  -k "test_known_emitters_detection_source_omni" \
  --csv_file_path=tests/data/input_csv_files/pdw_generator/known_emitters/<file>.csv
```

The `AI_ENGINE=rust` env var tells `conftest.py` to call `ai_processor --csv`
instead of the legacy Python `app.py`.

### DB fixtures (what pytest does before each test)

| Fixture | What it does |
|---|---|
| `create_csv_input_table` | Creates `signals.csv_input` in ClickHouse |
| `truncate_activities_db_tables` | Truncates CH `clusters` + `csv_input` |
| `truncate_intelligence_db_tables` | Deletes PG `clusters`, `signals`, `emitters` |
| `populate_signatures_table` | Loads manufacturer/radar-type/signature CSVs into PG |
| `populate_csv_input_table` | Inserts the test CSV rows into CH `csv_input` |

When `AI_ENGINE=rust`, the Rust binary re-loads the CSV itself (into
`signals.pdw` and `signals.csv_input`) during its own run, so the Python
`populate_csv_input_table` fixture is redundant but harmless.

---

## 4. Parallel execution of multiple files

Each `ai_processor --csv` run is **fully independent** (own wipe, load,
process, write). To run N files in parallel:

```bash
# 4 files at a time
ls tests/data/input_csv_files/pdw_generator/known_emitters/*.csv | \
  xargs -P 4 -I{} ./target/release/ai_processor --csv {}
```

**Caveat:** the wipe is global (all 5 tables), so true parallel runs need
separate DB instances or schemas per file. For the full 608-file batch,
run **serially** (one at a time) to keep DB state clean:

```bash
for f in tests/data/input_csv_files/pdw_generator/known_emitters/*.csv; do
  ./target/release/ai_processor --csv "$f" || echo "FAILED: $f"
done
```

---

## 5. Verifying results

After each run, check the counts in Postgres and ClickHouse:

```bash
# Postgres (via docker on the VM)
docker exec elint_pg psql -U elint -d sigint -t -A \
  -c "SELECT count(*) FROM emitters"
docker exec elint_pg psql -U elint -d sigint -t -A \
  -c "SELECT count(*) FROM signals"
docker exec elint_pg psql -U elint -d sigint -t -A \
  -c "SELECT count(*) FROM clusters"
docker exec elint_pg psql -U elint -d sigint -t -A \
  -c "SELECT name FROM emitters"

# ClickHouse
docker exec elint_ch clickhouse-client --user elint --password elint \
  -q "SELECT count() FROM signals.pdw"
docker exec elint_ch clickhouse-client --user elint --password elint \
  -q "SELECT count() FROM signals.clusters"
```

Compare against `expectations.json` (608 keys, one per test CSV):

```json
{
  "mf03_mf04_best-0_plus0.csv": {
    "kind": "known",
    "expected_names": ["1113", "1109"],
    "expected_ids": ["1113", "1109"],
    "exp_em": 2,
    "std": 0
  }
}
```

A file **passes** when:
- `ai_processor` exits with rc=0
- PG `emitters` count == `exp_em` from `expectations.json`
- Emitter names in PG match `expected_names`
- CH `pdw` row count == CSV row count
- CH `clusters` row count == PG `clusters` count

---

## 6. Test data

```
tests/data/input_csv_files/pdw_generator/
├── known_emitters/      # 448 CSVs
└── unknown_emitters/    # 160 CSVs
```

Test parameters (emitter IDs, names, expected counts) are read from
CSVs in `tests/data/test_data/` via `helpers/misc.read_test_parameters`.

---

## 7. Repo layout

```
ai_elint_tests/
├── tests/
│   ├── conftest.py              # fixtures + AI_ENGINE=rust hook
│   ├── test_pdw_generator.py    # main test class
│   └── data/
│       ├── input_csv_files/     # 608 test CSVs
│       └── test_data/           # test parameter CSVs
├── helpers/
│   ├── db/                      # PG + CH connection managers
│   ├── config.py                # DB connection config
│   ├── types.py                 # RadarMapper
│   ├── misc.py                  # test parameter reader
│   ├── emitter_specifications/  # expected emitter specs
│   └── app*                     # LEGACY Python engine (deprecated, do not modify)
├── model/
│   └── transformer_model_known_v16.onnx   # ONNX model (also deployed on VM)
├── pyproject.toml
└── README.md
```

---

## 8. Companion repo

| Repo | Branch | What it provides |
|---|---|---|
| `ai_elint` | `aiforge/rust-ai-integration` | Rust engine: `ai_processor` binary, ONNX inference, HDBSCAN, PG/CH writes |
| `ai_elint_tests` (this repo) | `aiforge/rust-ai-integration` | pytest harness, test data, expectations |

See `ai_elint/ARCHITECTURE.md` for the full pipeline architecture and
ONNX threading config.
