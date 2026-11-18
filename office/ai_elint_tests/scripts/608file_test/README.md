# 608-file Classic + ONNX Test

Runs all 608 CSV files through both the **classic** and **ONNX** pipelines on the VM
(10.10.120.13), recording per-file results.

## Files in this folder

| File | Purpose |
|------|---------|
| `run_608_both.sh` | Runs classic (all 608) then ONNX (all 608). Resume-aware: skips files already in the output. |
| `run_608_onnx.sh` | Runs only the ONNX phase. Resume-aware: skips files already in the output. |
| `build_expected_results.py` | Builds `tests/test_data/expected_results.json` (flat, used by `test_and_validate_all.sh`) and `expected_results_full.json` (rich review report) from the param CSVs + measured result files. |
| `build_expected_json.py` | (Legacy) Merges the classic + ONNX result files into `expected_results_608.json` — superseded by `build_expected_results.py`. |
| `expected_results_full.json` | Rich per-file report: expected emitter names/counts + measured classic/onnx results + pass flag. |

---

## Expected results — how it is built

The expected-results file is the single source of truth for validating engine runs.
It is **not** hand-written. It is generated from the same parameter CSVs that the
pytest harness itself reads, so the two always agree.

### Sources

| File | What it provides |
|------|-----------------|
| `tests/test_data/known_emitters_db_emitter_level.csv` | 448 known-emitter rows: emitter names (MF03, PSR1, …), PG emitter IDs, expected count (`no_emitters`), allowed variance (`std_no_emitters`) |
| `tests/test_data/unknown_emitters_db.csv` | 160 unknown-emitter rows: emitter names (CTA1, CTA2, NAV1, …), expected count (`number_emitters`), allowed variance |
| `helpers/emitter_specifications/unknown_emitters.py` | Spec ranges (frequency, PRI, PW) for each unknown emitter — used by the pytest harness to verify detected emitter characteristics |
| `master_608_classic_results.txt` (VM) | Measured classic results: one `RESULT\|` line per file |
| `master_608_onnx_results.txt` (VM) | Measured ONNX results: one `RESULT\|` line per file |

### What the pytest harness actually validates

The harness (`tests/test_pdw_generator.py`) does **not** just count emitters. For each
file it asserts, via live DB queries:

**Known emitters** (`test_known_emitters_detection`):
1. `SELECT count(*) FROM emitters WHERE name = '<emitter_id_1>'` **> 0**
2. `SELECT count(*) FROM emitters WHERE name = '<emitter_id_2>'` **> 0**
3. `SELECT count(*) FROM clusters WHERE signal_id IS NULL` **== 0** (every cluster has a signal)
4. Per-emitter PDW accuracy: how many PDWs labelled emitter-1 were clustered under
   emitter-1 vs. mislabelled as emitter-2 or `unknown_ai`

**Unknown emitters** (`test_unknown_emitter_detection_main_lobe`):
1. Total emitters detected **≥ `number_emitters`**
2. For each emitter name: the **average PRI, PW and frequency** of its detected
   signals fall **inside the spec range** from `unknown_emitters.py`

### How `build_expected_results.py` maps this to JSON

For each of the 608 files it reads the param CSV row and produces:

```json
{
  "mf03_mf04_best-0_plus0.csv": {
    "kind": "known",
    "expected_names": ["MF03", "MF04"],
    "expected_ids": ["1113", "1109"],
    "exp_em": 2,
    "std": 0
  }
}
```

- `exp_em` — the expected emitter count (`no_emitters` for known, `number_emitters` for unknown).
  This is the value `test_and_validate_all.sh` compares `pg_em` against.
- `expected_names` / `expected_ids` — the emitter names and PG IDs the harness asserts on.
- `std` — allowed variance (informational; the harness uses it for accuracy thresholds).

For unknown-emitter files the rich report additionally embeds `spec_ranges`
(frequency, PRI, PW bounds) so a reviewer can see what the harness checks.

### Pass logic in the rich report

`expected_results_full.json` also records the measured results from the batch runs:

```
pass = (rc == 0) AND (pg_em == exp_em)
```

This is a **simplified** pass — it checks the emitter count only. The full pytest
harness additionally checks emitter names and per-emitter accuracy, which requires
a live DB run and cannot be captured in the batch result lines.

### How to regenerate

```bash
# From the repo root, after the batch runs have produced result files:
python3 scripts/608file_test/build_expected_results.py \
  --repo . \
  --classic /path/to/master_608_classic_results.txt \
  --onnx    /path/to/master_608_onnx_results.txt
```

This writes:
- `tests/test_data/expected_results.json` — flat, consumed by `test_and_validate_all.sh`
- `scripts/608file_test/expected_results_full.json` — rich review report

Both files must be committed after regeneration.

---

## File list (CSVs)

The 608-file list lives on the VM at `/home/sysadm/all608_list.txt` (608 lines).
Source CSVs are under:
- `/home/sysadm/ai_elint_tests_work/tests/data/input_csv_files/pdw_generator/known_emitters/` (448 files)
- `/home/sysadm/ai_elint_tests_work/tests/data/input_csv_files/pdw_generator/unknown_emitters/` (160 files)

## Result files (on the VM)

| File | Contents |
|------|----------|
| `/home/sysadm/master_608_classic_results.txt` | 608 `RESULT\|` lines, engine=classic. **Complete.** |
| `/home/sysadm/master_608_onnx_results.txt` | `RESULT\|` lines, engine=onnx. Resumes from where the last run stopped. |

### Result line format

```
RESULT|<filename.csv>|engine=<classic|onnx>|rc=<int>|t=<float>s|pg_em=<int>|pg_sig=<int>|pg_clu=<int>|ch_pdw=<int>|ch_clu=<int>|
```

`pg_em` = Postgres `emitters` count (0 = no emitters found = FAIL).
`ch_pdw` = ClickHouse `pdw` count.

## How to run

All commands run **on the VM** (10.10.120.13, user sysadm).

### Prerequisites

- `ai_processor` binary: `/home/sysadm/ai_elint_rust_work2/target/release/ai_processor`
- ONNX model service running: `python3 /home/sysadm/scratch280/model_service.py` (port 8000)
- PG: `postgresql://elint:elint@127.0.0.1:15432/sigint`
- ClickHouse: `http://127.0.0.1:18123`

### Run the full batch (classic + ONNX)

```bash
nohup bash /home/sysadm/run_608_both.sh \
  /home/sysadm/ai_elint_rust_work2/target/release/ai_processor \
  /home/sysadm/all608_list.txt \
  /home/sysadm/master_608_classic_results.txt \
  /home/sysadm/master_608_onnx_results.txt \
  > /home/sysadm/batch_608.log 2>&1 &
```

### Run only ONNX (resume)

```bash
nohup bash /home/sysadm/run_608_onnx.sh \
  /home/sysadm/ai_elint_rust_work2/target/release/ai_processor \
  /home/sysadm/all608_list.txt \
  /home/sysadm/master_608_onnx_results.txt \
  > /home/sysadm/onnx_resume_608.log 2>&1 &
```

### Check progress

```bash
grep -c '^RESULT|' /home/sysadm/master_608_classic_results.txt   # classic count / 608
grep -c '^RESULT|' /home/sysadm/master_608_onnx_results.txt      # onnx count / 608
tail -5 /home/sysadm/onnx_resume_608.log                         # live log
```

### Run test + validate all (uses expected_results.json)

```bash
# Validate every CSV against the expected emitter counts
bash /home/sysadm/ai_elint_tests_work/test_and_validate_all.sh
# or a subset:
bash /home/sysadm/ai_elint_tests_work/test_and_validate_all.sh --limit 10 --known
```

### Rebuild the expected-results JSON after a new batch run

```bash
# Pull the latest result files from the VM, then rebuild locally:
scp sysadm@10.10.120.13:/home/sysadm/master_608_classic_results.txt /tmp/classic.txt
scp sysadm@10.10.120.13:/home/sysadm/master_608_onnx_results.txt    /tmp/onnx.txt

python3 scripts/608file_test/build_expected_results.py \
  --repo . \
  --classic /tmp/classic.txt \
  --onnx    /tmp/onnx.txt
```

Then commit both output files:
- `tests/test_data/expected_results.json`
- `scripts/608file_test/expected_results_full.json`

## Timing

- Classic: ~0.5–5 s per file → ~15 min total
- ONNX: ~10–90 s per file → ~3–5 hours total
- Both phases run serially to keep DB state clean.
