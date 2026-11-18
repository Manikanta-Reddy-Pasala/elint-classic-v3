# AGENTS.md — elint_reports

Test/benchmark tooling kept **outside** `ai_elint_tests` so the test repo
stays almost unchanged. Rules:

* The harness is the judge. Reports never change pass/fail; they only record
  what the engine produced and how long each stage took.
* All runs share one Postgres + ClickHouse — run them **one at a time**
  (`full_chain.sh` is sequential on purpose; parallel runs also distort
  timings).
* Timing: CSV parse/load is setup. "Pipeline" = DB read → results written
  (`engine_total − wipe_tables − load_csv`). Python reads the CSV directly, so
  its pipeline = features + model + post-processing.
* Python runs use their own DBs (`sigint_py`, `signals_py`) and delete
  Python's cache files at start (`py_ref_run.py`), so runs don't leak state.
* Engine stages are discovered from `LAYER|<stage>|<ms>` lines — add a line in
  the Rust code and it appears in the reports automatically
  (`build_report.py` picks known keys per engine; extend `LAYERS` there).
