# onnx_model_service

Standalone HTTP service that runs the ELINT transformer model
(`transformer_model_known_v16.onnx`, FP32) on every device of the box — the
Radeon iGPU (ROCm or MIGraphX execution provider) **and** the CPU at the same
time. Engines (the Rust `ai_elint` pipeline, tools, tests) call it over HTTP,
so they never load onnxruntime themselves and the model can live on another
machine.

## Why a window protocol

The model is a sequence model: it reads one **window of 512 consecutive
pulses** and attends across the window (the Python reference feeds
`(1, 512, 458)`). Sending pulses one by one (`(n, 1, 458)`) gives different,
wrong embeddings and predictions. A request therefore carries one or more
whole windows and every window runs as one sequence.

## API

All binary payloads are little-endian, row-major.

| Call | Body | Reply |
| --- | --- | --- |
| `POST /predict` | `n × 458` f32 features; header `X-Window-Size: 512` (omit = whole body is one window) | 24-byte header `[u64 emb_bytes, u64 id_bytes, u64 sim_bytes]`, then `n × 1024` f32 embeddings, `n` i64 centre ids, `n` f32 similarities |
| `POST /embed` | same | `n × 1024` f32 embeddings |
| `GET /health` | – | JSON: model, workers, provider per worker, windows served |

## Setup (once)

Needs the ROCm 6.4.2 runtime + libraries on the host
(`apt install rocm-libs migraphx`; on this box the Ubuntu mirror
`artifactory.internal` is unreachable, so `/etc/apt/sources.list.d/ubuntu-archive-tmp.list`
points apt at archive.ubuntu.com for the few Ubuntu dependencies).

```bash
cd onnx_model_service
mkdir -p model && cp ../ai_elint/model/transformer_model_known_v16.onnx model/
./setup.sh            # .venv with onnxruntime_rocm 1.21.0 (repo.radeon.com, rocm-rel-6.4.2)
```

## Run

```bash
./run.sh                                   # GPU + CPU workers on :8000
curl -s localhost:8000/health
```

Settings (environment, defaults in `run.sh`):

| Var | Default | Meaning |
| --- | --- | --- |
| `DEVICE` | `hybrid` | `hybrid` (GPU + CPU workers share one queue), `gpu`, `cpu` |
| `GPU_PROVIDER` | `migraphx` | `migraphx` (faster; compiles the window shape once, ~25 s, then loads from `migraphx_cache/`) or `rocm` |
| `GPU_WORKERS` / `CPU_WORKERS` | 1 / 1 | sessions per device |
| `CPU_THREADS` | 12 | intra-op threads per CPU session |
| `HSA_OVERRIDE_GFX_VERSION` | 11.0.0 | runs ROCm gfx1100 kernels on the gfx1150 (Radeon 890M) |
| `MODEL_PATH` | `model/transformer_model_known_v16.onnx` | model file |
| `PORT` | 8000 | listen port |

## Measured (Ryzen AI 9 HX 370 + Radeon 890M, FP32 model, 512-pulse windows)

| Setup | ms / window | pulses / s |
| --- | --- | --- |
| CPU only, 24 threads | 132 | ~3.9k |
| GPU (ROCm EP) only | 45 | ~11.4k |
| hybrid ROCm GPU + CPU | 40 | ~12.6k |
| GPU (MIGraphX EP) only | 36 | ~14.2k |
| **hybrid MIGraphX GPU + CPU (default)** | **33** | **~15.5k** |

GPU and CPU outputs agree to 1e-6 (max abs diff on the 1024-d embeddings).

Load test: `python bench_client.py http://127.0.0.1:8000 128 8 4`
(128 windows, 8 windows per request, 4 concurrent requests).

FP16 (MIGraphX `migraphx_fp16_enable`) runs at 16 ms/window but the model
overflows in half precision (NaN embeddings, every centre id wrong), so the
service stays FP32. A second GPU worker gives no gain — the iGPU is saturated
by one.

## Files

| File | What it does |
| --- | --- |
| `server.py` | HTTP server: `/predict`, `/embed`, `/health`; one shared window queue drained by GPU and CPU worker threads (one ORT session each); results returned in request order |
| `run.sh` | start with all settings (env-overridable) |
| `setup.sh` | create `.venv` with `onnxruntime_rocm 1.21.0` (ROCm 6.4.2 build) |
| `bench_client.py` | load test: windows/s, ms/window, per-worker split |
| `model/` | model file(s) (not in git; copy from `ai_elint/model/`) |
| `migraphx_cache/` | compiled MIGraphX programs (generated) |
| `service.log` | runtime log when started via `run.sh` in the background |

## Repeatability

`DEVICE=hybrid` (default) is fastest, but which device runs a window varies
between runs and GPU/CPU outputs differ by ~1e-6, which can move rare
boundary decisions downstream. For bit-repeatable runs use `DEVICE=gpu`
(~36 ms/window) or `DEVICE=cpu`.
