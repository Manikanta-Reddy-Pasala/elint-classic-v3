# ~/coderepo — ELINT workspace (box 192.168.70.103)

| Folder | What | Git |
| --- | --- | --- |
| `ai_elint/` | **Rust engines** (classic + ONNX), `ai_processor` binary. Start with `README.md`, `FILES.md`, `AGENTS.md` | gitlab `architecture/ai_elint` |
| `onnx_model_service/` | model server (GPU MIGraphX + CPU) the ONNX engine calls over HTTP | local |
| `ai_elint_tests/` | pytest harness = the judge (608 files); branch `aiforge/rust-ai-integration`, minimal local changes | gitlab `elint/ai_elint_tests` |
| `ai_elint_python_master/` | Python reference implementation (ONNX only), read-only | gitlab `AI/ai_elint` |
| `elint_reports/` | run tooling: harness runs with per-file reports, parity/compare, Python timing, time report | local |
| `elint_ui/` | web UI (:8800) to run classic/ONNX on a file, folder or the DB and view results | local |
| `runs/` | outputs of all runs (reports, logs) | – |

## How the pieces connect

```
CSV / receiver ──▶ ClickHouse signals.pdw ──▶ ai_processor (ai_elint)
                                               ├─ CLASSIC=1 → classic_v2 (CPU, rayon)
                                               └─ CLASSIC=0 → onnx_flow ──HTTP──▶ onnx_model_service (GPU+CPU)
                                                     │
                                Postgres emitters/signals/clusters + ClickHouse signals.clusters
                                                     │
       ai_elint_tests (pytest judge) ◀── elint_reports (runs + reports) ── elint_ui (interactive)
```

## Daily use

```bash
docker ps | grep -E "elint_pg|elint_ch"                 # DBs (auto-start)
~/coderepo/onnx_model_service/run.sh &                  # model service (ONNX engine only)
cd ~/coderepo/ai_elint && RUSTFLAGS="-C target-cpu=native" cargo build --release
~/coderepo/elint_reports/tools/run_one.sh classic <csv> # quick single run
~/coderepo/elint_reports/run_engine.sh onnx onnx_608    # full judged run + report
~/coderepo/elint_ui/run.sh &                            # UI on :8800
```

DB access: Postgres `127.0.0.1:15432` elint/elint `sigint` (Python runs:
`sigint_py`); ClickHouse `127.0.0.1:18123` / `:19000` elint/elint `signals`
(`signals_py`). Test data: `ai_elint/data/pdw_generator/` (608 CSVs).

See `AGENTS.md` for the rules that keep results valid.

## Demo (auto-start)

Model service (:8000) and demo UI (:8800) are systemd services `elint-model` and `elint-ui`; they start at boot after the docker databases (restart policy unless-stopped). Open http://192.168.70.103:8800. Restart everything: `~/coderepo/start_demo.sh`. Logs: `journalctl -u elint-model` / `journalctl -u elint-ui`.
