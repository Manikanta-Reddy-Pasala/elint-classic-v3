# AGENTS.md — onnx_model_service

* The model is a **sequence model over 512-pulse windows**. Every request
  carries whole windows; each window runs as `(1, 512, 458)`. Never reshape to
  `(n, 1, 458)`.
* Keep FP32. FP16 overflows (NaN). Batching several windows per GPU call is
  slower with MIGraphX. One GPU session saturates the iGPU.
* Provider options that work in this ORT build: MIGraphX
  `{"device_id":0,"migraphx_model_cache_dir": <dir>}`; ROCm `{"device_id":0}`.
  `HSA_OVERRIDE_GFX_VERSION=11.0.0` is required on the gfx1150 iGPU.
* If the GPU provider fails to load, the service logs an ERROR and keeps
  serving on CPU only — check `/health` (provider per worker) after changes.
* Wire format is shared with `ai_elint/pipeline/src/onnx_flow.rs::predict`;
  change both together.
* GPU vs CPU outputs agree to ~1e-6; decisions downstream are unchanged
  (ONNX parity with Python master holds with this service).
