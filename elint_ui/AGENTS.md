# AGENTS.md — elint_ui

* Thin front end: never reimplement engine logic here — call `ai_processor`
  (`--csv`, `--csv --load-only`, `--from-db`) and read the result tables.
* Jobs run strictly one at a time (shared DBs; every run clears the result
  tables). Do not add parallel workers.
* No third-party web framework; Python stdlib + psycopg2 only, single HTML
  file with inline JS/CSS (light/dark via CSS variables).
* Expected results come from `ai_elint_tests/tests/test_data/expected_results.json`
  (matched by file name); unknown files show "–".
* "Pipeline ms" = `engine_total − wipe_tables − load_csv`, same definition as
  the reports.
