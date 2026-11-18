# 5-file test — classic + ONNX (ai_elint master, Rust engine)

Runs 4 CSVs through the `ai_processor` binary from `architecture/ai_elint` **master**
in both modes on the VM `10.10.120.13` (user `sysadm`, password from env `VM_PASS`):

- **classic**: no `ONNX_PATH` (deterministic pipeline only)
- **onnx**: `ONNX_PATH=/home/sysadm/ai_elint_run/integration/model/transformer_model_known_v16.onnx`

Each run is **fresh**: PG tables (`clusters, signals, emitters`) and CH tables
(`pdw, clusters, csv_input`) are truncated before every file, so results are
independent.

## Files

- `run_one.sh` — run a single CSV: wipe DBs, run `ai_processor --csv`, capture
  rc + wall time + PG/CH counts, append a `RESULT|...` line to the log.
- `run_5_both.sh` — VM-side driver: loop over `smoke3_list.txt` (4 CSVs) in both
  modes, one file at a time.
- `upload_5.py` — sandbox-side launcher: kills stale engine jobs, SFTP-uploads
  the two scripts, starts `run_5_both.sh` under `nohup`.
- `ssh_vm.py` — thin paramiko wrapper (credentials from env) for ad-hoc VM commands.
- `results_5files.txt` — captured per-file results (see below).

## 4 selected files

| # | File | Type |
|---|---|---|
| 1 | `mf03_mf04_best-0_plus0.csv` | known emitters |
| 2 | `mf03_psr5m2_best-0_plus5.csv` | known emitters |
| 3 | `psr2_psr3_best-3_plus5.csv` | known emitters |
| 4 | `cta1_cta2_best-0_plus0.csv` | unknown emitters |

## How to run

From this sandbox:

```bash
python3 scripts/5file_test/upload_5.py   # kills stale jobs, uploads, launches
```

Then poll:

```bash
python3 scripts/5file_test/ssh_vm.py "tail -5 /home/sysadm/5files_both_run.log"
python3 scripts/5file_test/ssh_vm.py "grep '^RESULT' /home/sysadm/5files_both_results.txt"
```

## Results (10.10.120.13, master binary, 2026-11-10)

**Classic — 4/4 PASS** (rc=0, PG `emitters` populated, ~143 s per file):

| File | Status | emitters |
|---|---|---|
| mf03_mf04_best-0_plus0 | PASS | 3 |
| mf03_psr5m2_best-0_plus5 | PASS | 2 |
| psr2_psr3_best-3_plus5 | PASS | 3 |
| cta1_cta2_best-0_plus0 | PASS | 1 |

**ONNX — 4/4 PASS** (rc=0, session ready `real=true`, PG `emitters` populated,
~143–147 s per file):

| File | Status | emitters |
|---|---|---|
| mf03_mf04_best-0_plus0 | PASS | 2 |
| mf03_psr5m2_best-0_plus5 | PASS | 2 |
| psr2_psr3_best-3_plus5 | PASS | 1 |
| cta1_cta2_best-0_plus0 | PASS | 2 |

Note: `pg_sig=0` and `pg_clu=0` are expected for these smoke files (few
emitters, no signal/cluster fan-out). CH `pdw`/`clusters` table lookups return
`UNKNOWN_TABLE` — the one-shot `--csv` path writes to PG only, not CH.
