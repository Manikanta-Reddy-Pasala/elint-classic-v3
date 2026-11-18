# Build the single ELINT pipeline binary (engine = Rust, no Go, no Kafka).
# Workspace layout: repo-root Cargo.toml holds [workspace] with engine + pipeline.
# Build the image from the engine/ directory (context = engine/), so the
# workspace root is the parent dir of the build context:
#   docker build --target builder -f engine/Dockerfile -t elint-engine-builder:latest .
# (run from the repo root, so Cargo.toml/Cargo.lock/engine/pipeline/
#  model/ are all visible in the context).
FROM busybox AS model
ARG MODEL_FILE=model/transformer_model_known_v16_int8.onnx
COPY ${MODEL_FILE} /models/transformer_model_known_v16.onnx

# AVX-512 build: the hand-written 512-bit intrinsics in hdbscan.rs need a
# nightly toolchain (stdarch AVX-512 is unstable on stable Rust) and
# target-feature=+avx512f so they emit vfmaddps on %zmm registers.
# The resulting binary runs only on AVX-512 CPUs (production host = Ryzen AI 9 HX 370).
FROM rust:1.88-slim AS builder
RUN rustup toolchain install nightly-2025-09-01 2>&1 | tail -1
WORKDIR /build
COPY Cargo.toml ./Cargo.toml
COPY Cargo.lock ./Cargo.lock
COPY engine ./engine
COPY pipeline ./pipeline
# build with the locked versions — pinned to 1.75-compatible deps
# (tokio-postgres 0.7.10, postgres-protocol 0.6.7, url 2.3.1) so the
# crypto stack stays on 0.10/0.12 and no edition2024 crate is pulled.
RUN CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse \
    RUSTFLAGS="-C target-feature=+avx512f" \
    cargo +nightly-2025-09-01 build --release

FROM debian:bookworm-slim
# libonnxruntime is NOT in the debian repos — pass the .so as a build arg
# (default: /opt/ai_elint/onnxruntime/libonnxruntime.so on the lab build box)
# or drop one at build/onnxruntime/libonnxruntime.so.
# (the bare .so name on the lab box is a symlink — COPY needs a real file;
#  the path is relative to the build context, so stage it at engine/ort.so)
ARG ORT_LIB=engine/ort.so
COPY $ORT_LIB /usr/lib/libonnxruntime.so
# ONNX model (same file the Python engine used)
COPY --from=model /models/transformer_model_known_v16.onnx /models/transformer_model_known_v16.onnx
COPY --from=builder /build/target/release/elint_engine /usr/local/bin/elint_engine
ENV MODEL_PATH=/models/transformer_model_known_v16.onnx
ENV ORT_LIB_PATH=/usr/lib/libonnxruntime.so
ENTRYPOINT ["elint_engine"]
