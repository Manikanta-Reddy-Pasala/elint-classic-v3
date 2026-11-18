# Results — 2026-11-13 (box 192.168.70.103, Ryzen AI 9 HX 370 + Radeon 890M)

Judge: `ai_elint_tests` pytest harness, all 608 files (448 known-emitter,
160 unknown-emitter), run with `~/coderepo/elint_reports/run_engine.sh`.
Per-file reports (expected vs found, signals, clusters, pulses, pytest
reason, time per stage):
`~/coderepo/runs/pytest/classic_final2/report.html`,
`~/coderepo/runs/pytest/onnx_final/report.html`; combined time report
`~/coderepo/runs/time_report_final.html`.

## Pass rates

| | Rust classic | Rust ONNX |
| --- | --- | --- |
| pytest pass, all | **608 / 608** | **602 / 608** |
| known files (both library names found, no orphan clusters) | 448 / 448 | 448 / 448 |
| unknown files (≥ 2 emitters, each named emitter's PRI/PW/freq in spec) | 160 / 160 | 154 / 160 |
| exact emitter count, known (warning-only in pytest) | 448 / 448 | 75 / 448 |
| exact emitter count, unknown | 158 / 160 | 52 / 160 |

ONNX reproduces Python master's grouping (identical emitters, signals,
clusters and pulses on 20/20 compared files), so it inherits Python's
over-splitting of unknown emitters (count) and its 6 misses: one emitter split
across partial signals or merged with stray pulses (NAV2 / NAV3 / CTA2 in
`cta2_nav2_best-0_plus3`, `cta2_nav2_best-10_plus3`, `nav1_nav2_best-10_plus0`,
`nav1_nav2_best-10_plus2`, `nav1_nav3_best-3_plus5`, `nav2_nav3_best-0_plus5`).
With the model service in hybrid mode (GPU + CPU share windows) 1e-6
embedding differences can move boundary cases between runs; `DEVICE=gpu` on
the service makes runs repeatable.

## Time (ms per file; pipeline = DB read → results written, CSV load excluded)

| Stage | Python master (29 files) | Rust ONNX (608) | Rust classic (608) |
| --- | --- | --- | --- |
| read pulses | (reads CSV) | 34.6 | 36.9 |
| features | 10,651 | 1,047 (sum over 8 workers) | – |
| model | 33,031 | 45,103 (sum over 8 workers; 33 ms/window throughput) | – |
| post-processing / detect | 20,790 | 1,207 (in order, overlapped) | 19.0 |
| result writes | (inside post-processing) | 635 | 24.3 |
| **pipeline (median)** | **64,080** | **6,040** | **108.5** |
| pulses / s (median) | ~2.6k | ~15.4k | ~847k |
| speed-up vs Python (median, same files) | 1× | **8.2×** | **462×** |
| all 608 files, pipeline total | – | 3,887 s | 69 s |

Classic detect stages (median): cells 6.5, components 0.1, split repair 0.9,
coherence 2.9, naming 7.6 ms.

## Known limits

* Pulses read back from ClickHouse differ from the CSV in the last bit for
  ~1 % of float values (ClickHouse VALUES parsing); no effect on results.
* `refresh_test::parse_csv_missing_id_column_fails` unit test fails (also
  before this work).
* Test-repo fixes used for these numbers (uncommitted, minimal): NAV1 PW
  spec typo, NAV1 frequency upper bound 3051 → 3052 MHz (ground truth sits at
  3051.000 ± 0.002 MHz), missing `request` fixture in the unknown test, empty
  `IN ()` guard, env-driven DB config, engine switch in `conftest_rust.py`,
  local-CSV-safe download fixture. ClickHouse profile: old analyzer +
  precise float parsing.
