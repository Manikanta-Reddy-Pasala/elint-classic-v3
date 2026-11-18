"""ELINT ONNX model service.

Runs the transformer model (transformer_model_known_v16.onnx) behind a small
HTTP API so any engine (the Rust ai_elint pipeline, Python tools, tests) can
call it without loading onnxruntime itself.

The model is a sequence model: it reads a WINDOW of consecutive pulses
(512 in the Python reference) and attends across the window. A request
therefore carries one or more whole windows, and the service runs each
window as one sequence of shape (1, window, 458).

Work is spread over every device on the box: GPU worker(s) (ROCm or
MIGraphX execution provider) and CPU worker(s) pull windows from one shared
queue, so a request with many windows keeps the GPU and the CPU busy at the
same time. Results always come back in request order.

Wire protocol (little-endian, row-major):

  POST /predict          body: n * 458 f32 (n pulses, n = k * window)
  POST /embed            same body
  header X-Window-Size:  pulses per window (default: the whole body is one window)

  /embed   200 -> n * 1024 f32 embeddings
  /predict 200 -> 24-byte header [u64 emb_bytes, u64 label_bytes, u64 sim_bytes]
                  then n*1024 f32 embeddings, n i64 centre ids, n f32 similarities
  GET /health -> JSON with workers and providers
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import onnxruntime as ort

IN_DIM = 458
OUT_DIM = 1024

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "model", "transformer_model_known_v16.onnx"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
DEVICE = os.environ.get("DEVICE", "hybrid").lower()          # hybrid | gpu | cpu
GPU_PROVIDER = os.environ.get("GPU_PROVIDER", "migraphx").lower()  # migraphx | rocm
GPU_WORKERS = int(os.environ.get("GPU_WORKERS", "1"))
CPU_WORKERS = int(os.environ.get("CPU_WORKERS", "1"))
CPU_THREADS = int(os.environ.get("CPU_THREADS", str(max(1, (os.cpu_count() or 2) // 2))))
WARMUP_WINDOW = int(os.environ.get("WARMUP_WINDOW", "512"))
MIGRAPHX_CACHE = os.environ.get("MIGRAPHX_CACHE", os.path.join(os.path.dirname(__file__), "migraphx_cache"))

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("onnx_model_service")


def _gpu_providers():
    if GPU_PROVIDER == "migraphx":
        # MIGraphX compiles the graph for the (1, 512, 458) window shape
        # (~25 s); the compiled program is cached on disk (hashed by model
        # and shape) and loaded on the next start.
        os.makedirs(MIGRAPHX_CACHE, exist_ok=True)
        opts = {"device_id": 0, "migraphx_model_cache_dir": MIGRAPHX_CACHE}
        return [("MIGraphXExecutionProvider", opts)]
    return [("ROCMExecutionProvider", {"device_id": 0})]


def _make_session(kind: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if kind == "cpu":
        so.intra_op_num_threads = CPU_THREADS
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return ort.InferenceSession(MODEL_PATH, so, providers=["CPUExecutionProvider"])
    sess = ort.InferenceSession(MODEL_PATH, so, providers=_gpu_providers() + ["CPUExecutionProvider"])
    if sess.get_providers()[0] == "CPUExecutionProvider":
        raise RuntimeError(f"GPU provider {GPU_PROVIDER} did not load (got {sess.get_providers()})")
    return sess


class Worker(threading.Thread):
    """Owns one session and runs windows from the shared queue."""

    def __init__(self, name: str, kind: str, jobs: "queue.Queue"):
        super().__init__(name=name, daemon=True)
        self.kind = kind
        self.jobs = jobs
        self.session = _make_session(kind)
        self.providers = self.session.get_providers()
        self.windows_done = 0
        self.busy_s = 0.0
        x = np.zeros((1, WARMUP_WINDOW, IN_DIM), dtype=np.float32)
        t = time.time()
        self.session.run(None, {"input_features": x})
        log.info("worker %s ready on %s (warm-up %.1fs)", name, self.providers[0], time.time() - t)

    def run(self):
        while True:
            x, fut = self.jobs.get()
            t = time.time()
            try:
                emb, ids, sim = self.session.run(None, {"input_features": x})
                fut.set_result((emb, ids, sim))
            except Exception as e:  # report to the caller, keep serving
                fut.set_exception(e)
            self.busy_s += time.time() - t
            self.windows_done += 1


JOBS: "queue.Queue" = queue.Queue()
WORKERS: list[Worker] = []


def start_workers():
    if DEVICE in ("hybrid", "gpu"):
        for i in range(GPU_WORKERS):
            try:
                WORKERS.append(Worker(f"gpu{i}", "gpu", JOBS))
            except Exception as e:
                log.error("GPU worker %d unavailable, continuing without it: %s", i, e)
    if DEVICE in ("hybrid", "cpu") or not WORKERS:
        for i in range(CPU_WORKERS):
            WORKERS.append(Worker(f"cpu{i}", "cpu", JOBS))
    for w in WORKERS:
        w.start()


def infer(rows: np.ndarray, window: int):
    """rows: (n, 458). Runs each window as one sequence; returns (emb, ids, sim) in order."""
    n = rows.shape[0]
    futs = []
    for s in range(0, n, window):
        fut: Future = Future()
        JOBS.put((np.ascontiguousarray(rows[s:s + window][None, ...]), fut))
        futs.append(fut)
    embs, ids, sims = [], [], []
    for f in futs:
        e, i, s = f.result()
        embs.append(e.reshape(-1, OUT_DIM))
        ids.append(i.reshape(-1))
        sims.append(s.reshape(-1))
    return np.concatenate(embs), np.concatenate(ids), np.concatenate(sims)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, ctype: str = "application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            info = {
                "status": "ok",
                "model": os.path.basename(MODEL_PATH),
                "workers": [
                    {"name": w.name, "provider": w.providers[0], "windows": w.windows_done, "busy_s": round(w.busy_s, 2)}
                    for w in WORKERS
                ],
            }
            self._send(200, json.dumps(info).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        size = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(size) if size else b""
        if path not in ("/predict", "/embed"):
            return self._send(404, b"not found", "text/plain")
        if size == 0 or size % (IN_DIM * 4):
            return self._send(400, f"body must be n*{IN_DIM} f32 (got {size} bytes)".encode(), "text/plain")
        rows = np.frombuffer(body, dtype="<f4").reshape(-1, IN_DIM)
        window = int(self.headers.get("X-Window-Size", "0") or 0) or rows.shape[0]
        try:
            emb, ids, sim = infer(rows, window)
        except Exception as e:
            log.exception("inference failed")
            return self._send(500, str(e).encode(), "text/plain")
        e = emb.astype("<f4").tobytes()
        if path == "/embed":
            return self._send(200, e)
        i = ids.astype("<i8").tobytes()
        s = sim.astype("<f4").tobytes()
        head = len(e).to_bytes(8, "little") + len(i).to_bytes(8, "little") + len(s).to_bytes(8, "little")
        self._send(200, head + e + i + s)

    def log_message(self, fmt, *args):
        log.debug("%s %s", self.address_string(), fmt % args)


def main():
    log.info("model=%s device=%s gpu_provider=%s gpu_workers=%d cpu_workers=%d cpu_threads=%d",
             MODEL_PATH, DEVICE, GPU_PROVIDER, GPU_WORKERS, CPU_WORKERS, CPU_THREADS)
    start_workers()
    # listen backlog: the default (5) resets connections when an engine opens
    # many windows at once
    ThreadingHTTPServer.request_queue_size = int(os.environ.get("LISTEN_BACKLOG", "256"))
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    srv.daemon_threads = True
    log.info("listening on %s:%d with %d workers", HOST, PORT, len(WORKERS))
    srv.serve_forever()


if __name__ == "__main__":
    main()
