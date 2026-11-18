# AGENTS.md — ~/coderepo (read first)

1. **Run everything on this box**, one DB-using job at a time (shared
   Postgres/ClickHouse; every engine run clears the result tables).
2. **Judge = `ai_elint_tests` pytest.** Keep changes there minimal
   (configuration + clear bug fixes only); tooling goes in `elint_reports/`.
3. **No ground truth in engines**: never read the CSV `emitter_id` or
   `pulse_repetition_interval_ns` columns in `ai_elint`.
4. **ONNX engine must stay identical to Python master** — after any ONNX
   change run `elint_reports/tools/compare.sh` (or `parity.sh`) on a sample
   and expect `SAME`.
5. Timing = DB read → results written; CSV loading is setup. Every engine
   stage prints `LAYER|<stage>|<ms>`.
6. Commits: conventional, on a branch; never push to master/gitlab without
   the owner's explicit OK.
7. Per-folder detail: `ai_elint/AGENTS.md`, `onnx_model_service/AGENTS.md`,
   `elint_reports/AGENTS.md`, `elint_ui/AGENTS.md`.
