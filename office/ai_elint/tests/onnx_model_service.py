"""VM ONNX model service: embeddings, centre predictions and similarities.

This test service is read-only Python tooling. It does not replace the Rust
engine or the production Python application.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "/home/sysadm/models/transformer_model_known_v16.onnx",
    providers=["CPUExecutionProvider"],
)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        size = int(self.headers.get("Content-Length", 0))
        x = np.frombuffer(self.rfile.read(size), dtype="<f4").reshape(-1, 1, 458)
        embeddings, ids, similarities = session.run(None, {"input_features": x})
        emb = embeddings.reshape(-1).astype("<f4").tobytes()
        ids = ids.reshape(-1).astype("<i8").tobytes()
        score = similarities.reshape(-1).astype("<f4").tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        predictor_payload = self.path.rstrip("/") == "/predict"
        content_length = 24 + len(emb) + len(ids) + len(score) if predictor_payload else len(emb)
        self.send_header("Content-Length", str(content_length))
        self.end_headers()
        if self.path.rstrip("/") == "/predict":
            self.wfile.write(
                len(emb).to_bytes(8, "little")
                + len(ids).to_bytes(8, "little")
                + len(score).to_bytes(8, "little")
            )
            self.wfile.write(emb + ids + score)
        else:
            self.wfile.write(emb)

    def log_message(self, fmt, *args):
        print("ONNX-SVC", fmt % args, flush=True)


ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
