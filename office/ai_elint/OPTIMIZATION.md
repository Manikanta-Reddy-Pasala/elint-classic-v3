# ai_elint — Optimisation notes

Measured on the box (Ryzen AI 9 HX 370, 24 threads, Radeon 890M), 177,256-pulse
file `mf03_psr2_best-3_plus5.csv` unless noted. Pipeline time excludes CSV
load (test setup).

## ONNX engine

| Change | Effect |
| --- | --- |
| Model on GPU (MIGraphX) + CPU worker in `onnx_model_service` | 132 ms/window (CPU only) → **33 ms/window** |
| MIGraphX compiled-program cache | service start 26 s → 3 s |
| TOA RBF features: only kernels within reach (exp(-30) cut-off) | features 90 ms → 6.5 ms per window per thread; identical output (unit test) |
| One model call per window (no duplicate `embed` + `predict` call) | half the model work of the old path |
| ClickHouse cluster rows buffered, sent in ≤ 8 MB inserts | CH insert time 2,885 ms → 19 ms |
| Postgres writes in one transaction per 32 windows | DB writes 6.4 s → 1.3 s |
| Whole file | 89.6 s (Python master) → **~11 s** (model-bound) |

Tried and rejected: FP16 MIGraphX (16 ms/window but NaN outputs — the model
overflows in half precision); batching 2–4 windows per GPU call (1.4–1.9 s per
window with MIGraphX); 2 GPU sessions (no gain, GPU saturated); 2–3 CPU
sessions or 16 threads (slower than 1 × 12 — they compete with feature
threads).

## Classic engine

| Change | Effect |
| --- | --- |
| Parallel cell building, integer-interned modulation | cells ~6 ms |
| Coherence pairs / naming on rayon | detect total median 25 ms |
| One transaction + one batched cluster insert + one CH insert | DB writes ~194 ms → ~30 ms |
| Whole file | engine ~50 ms for 177k pulses; pipeline (incl. CH read) median ~0.1–0.2 s per file |

## Setup (not counted as processing)

CSV parse + load into ClickHouse ~0.25–0.5 s per file. `signals.pdw` load
previously truncated TOA and modulation strings to 6 characters
(`'{:0.6}'` on a string) — fixed, so the engines now read real pulses back
from ClickHouse like the production path.

## Robustness (ONNX)

* Model service listen backlog 5 → 256 and client retries (5×, 0.2–3.2 s
  backoff): one full run had failed on a connection reset.
