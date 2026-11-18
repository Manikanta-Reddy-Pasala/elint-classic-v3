# Office ELINT setup (code snapshot 2026-11-18)

Code-only copy of the box workspace (`~/coderepo` on 192.168.70.103) with all
2026-11-18 classic fixes applied. No PDW/CSV data, no model weights, no secrets.

## Engines
| Engine | Switch | Status |
|---|---|---|
| Classic v3 (frequency-agile, DOA gate, PRI pattern, unseen-radar naming check) | `CLASSIC=1` (default `CLASSIC_V3=1`) | **standard** |
| ONNX (transformer model over HTTP, port of the Python master) | `CLASSIC=0` | supported, unchanged |
| Classic v2 | `CLASSIC=1 CLASSIC_V3=0` | hidden fallback only (use on real direction-finder data until the bearing question is answered) |

ONNX model: `transformer_model_known_v16.onnx` (FP32, **not** the int8 file), as used by
Vidya's Python master. Weights are not in git: copy from the box
`~/coderepo/ai_elint/model/` to `ai_elint/model/` and `onnx_model_service/model/`.

## Folders
| Folder | What |
|---|---|
| repo root (`ai_elint`) | Rust engines, `ai_processor` binary, `classic_eval` offline runner. Docs: README, AGENTS, HOW_IT_WORKS, ARCHITECTURE |
| `onnx_model_service/` | model server (:8000), MIGraphX GPU + CPU |
| `ai_elint_tests/` | pytest judge (608 files) with the harness fixes |
| `elint_reports/` | run tooling, per-file reports, engine compare, timing |
| `elint_ui/` | web UI (:8800): Python / ONNX / Classic runs, compare demo |
| `elint_improvements/` | change notes (CHANGES.md) |
| `start_demo.sh` | restart DBs + services, print URL |

Merge into the office `ai_elint` repo: `the repo root` = the box copy (v2 version) plus only these changes:
`pipeline/src/classic_v2.rs`, `pipeline/src/lib.rs`, new `pipeline/src/pri_pattern.rs` + `pipeline/src/signature.rs`,
new `pipeline/src/bin/classic_eval.rs`, `pipeline/Cargo.toml` (classic_eval bin), `config/classic_library.json`.
Every other file is byte-identical to the box. Same change as one patch: `../CHANGES_2026-11-18.diff`.

Data needed (not in git, no CSV files in this repo): the judge's `ai_elint_tests/tests/test_data/*.csv` (expected results) and `ai_elint/data/pdw_generator/{known,unknown}_emitters/*.csv` (608 files)
and the real captures, from the box.

Credentials: none in git. Test-repo VM scripts read `VM_PASS` from the environment;
`ai_elint_tests/.env` (Artifactory) must be created locally from `.env.example`.

Same classic change as a patch for the office `ai_elint` git repo:
`../CHANGES_2026-11-18.diff`.
