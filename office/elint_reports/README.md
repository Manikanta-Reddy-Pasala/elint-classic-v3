# elint_reports — engine run reports (outside the test repo)

Run the pytest harness for one engine over any set of files and get a
per-file report: expected vs found emitters, names, signals, clusters,
pulses, pytest result + reason, and time per engine layer.

## How it works

1. The harness (`tests/conftest_rust.py`) runs `$RUST_AI_PROCESSOR --csv <file>`; `pytest_run.sh` points that at `engine_wrapper.py`, which runs the real binary and
   saves, right after the engine finishes,
   `<file>.<engine>.log` (engine output incl. `LAYER|<stage>|<ms>` lines) and
   `<file>.<engine>.json` (emitter names, signal/cluster counts, pulses in
   clusters, per-emitter PRI/PW/frequency).
2. pytest writes its junit XML.
3. `build_report.py` joins those with `tests/test_data/expected_results.json`
   → `report.csv`, `report.html` (sortable), `report.summary.json`.

## Run (on the box)

```bash
# all 608 files, one engine (tag = output folder name)
~/coderepo/elint_reports/run_engine.sh classic classic_full
~/coderepo/elint_reports/run_engine.sh onnx onnx_full          # needs onnx_model_service

# a subset: third argument is a pytest -k expression
~/coderepo/elint_reports/run_engine.sh onnx onnx_10 "(source_omni or main_lobe) and (mf03_mf04_best-0_plus0 or cta1_nav1_best-0_plus0)"
```

Output: `~/coderepo/runs/pytest/<tag>/{report.html,report.csv,report.summary.json,pytest.log,files/}`.

`run_engine.sh` → `pytest_run.sh` runs pytest in the `elint_pytest2`
container (Python 3.12) with `AI_ENGINE=rust RUST_ENGINE=<engine>` and the DB
settings for the local `elint_pg` / `elint_ch` containers.

## Columns

| Column | Meaning |
| --- | --- |
| `expected_names`, `expected_ids`, `expected_emitters` | from the param CSVs (via expected_results.json) |
| `found_emitters`, `found_known`, `found_unknown_ai` | rows in `emitters` after the run |
| `known_names_ok`, `count_exact` | both expected library ids present / count equals expected |
| `signals`, `clusters`, `clusters_without_signal`, `pulses_in_clusters` | result tables |
| `csv_rows`, `pulses_read`, `cluster_coverage_pct` | input size and share of pulses that ended in a cluster |
| `pytest`, `pytest_reason` | harness verdict and first assertion message |
| `ms.<layer>` | engine layer times; `ms.pipeline` = DB read → results written (CSV load excluded) |
| `pulses_per_s` | `pulses_read / pipeline` |
| `per_emitter` (unknown files) | PRI / PW / frequency / pulses per detected emitter |

## Files

| File | What it does |
| --- | --- |
| `run_engine.sh <classic\|onnx> <tag> [-k expr]` | full harness run for one engine + `build_report.py` → `~/coderepo/runs/pytest/<tag>/` |
| `pytest_run.sh <engine> <outprefix> [pytest args]` | runs pytest in the `elint_pytest2` container with engine/DB env; uses `engine_wrapper.py` when `REPORT_DIR` is set |
| `engine_wrapper.py` | stands in for `ai_processor` (via `RUST_AI_PROCESSOR`): runs the real binary, passes output/exit code through, saves per-file log + result JSON |
| `build_report.py` | joins per-file JSON + junit + expected results → `report.csv/.html/.summary.json` |
| `time_report.py` | Python vs Rust ONNX vs Rust classic, per file and per stage → `time_report.csv/.html/.summary.json` |
| `full_chain.sh` | sequential overnight chain: classic 608 → ONNX 608 → Python 30 → time report (log: `~/coderepo/runs/full_chain.log`) |
| `tools/run_one.sh <classic\|onnx> <csv> [dir]` | one file, no pytest: wipe → engine → one `RESULT|…` line (counts, names) |
| `tools/db_snapshot.sh` | fingerprint of the result tables (emitters by name; per signal: clusters, pulses) |
| `tools/parity.sh <csv>…` | Python master vs Rust ONNX on the same files → `PARITY|file|SAME/DIFF|…` |
| `tools/compare.sh <list> <outdir>` | Python master vs Rust ONNX vs Rust classic per file → `CMP|…` lines |
| `tools/py_ref_run.py <csv>` | runs Python master's ONNX flow on one CSV (DB `sigint`) |
| `tools/py_time_run.py <csv> <out.json>` | same flow, timed per stage (features / model / post-processing), DBs `sigint_py` / `signals_py` |
| `tools/dev/classic_proto.py` | pandas prototype used to design/calibrate classic v2 (labels used only to score) |
| `tools/dev/inspect_csv.py` | per-ground-truth-emitter profile of a CSV (freq, PW, BW, PRI, modulation) |

## Where results land

`~/coderepo/runs/` — `pytest/<tag>/` (harness runs), `cmp20/` (20-file
comparison), `python_30/` (Python timing), `time_report.*`,
`full_chain.log`, `adhoc/` (`run_one.sh` logs).
