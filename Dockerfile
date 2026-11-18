# Build the refresh_api binary (Rust, AVX-512, ONNX runtime).
# Usage: docker build -f Dockerfile -t elint-refresh-api:latest .
# (run from the repo root)
FROM busybox AS model
ARG MODEL_FILE=model/transformer_model_known_v16_int8.onnx
COPY ${MODEL_FILE} /models/transformer_model_known_v16_int8.onnx

FROM rust:1.88-slim AS builder
RUN rustup toolchain install nightly-2025-09-01 2>&1 | tail -1
WORKDIR /build
COPY Cargo.toml ./Cargo.toml
COPY Cargo.lock ./Cargo.lock
COPY pipeline ./pipeline
COPY engine ./engine
RUN CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse \
    RUSTFLAGS="-C target-feature=+avx512f,+avx512cd,+avx512er,+avx512pf,+f16c" \
    cargo +nightly-2025-09-01 build --release -p ai_processor --bin refresh_api

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates curl libnuma1 && rm -rf /var/lib/apt/lists/*
ARG ORT_LIB=engine/ort.so
COPY $ORT_LIB /usr/lib/libonnxruntime.so
COPY --from=model /models/transformer_model_known_v16_int8.onnx /models/transformer_model_known_v16_int8.onnx
COPY --from=builder /build/target/release/refresh_api /usr/local/bin/refresh_api
ENV MODEL_PATH=/models/transformer_model_known_v16_int8.onnx
ENV ORT_LIB_PATH=/usr/lib/libonnxruntime.so
EXPOSE 8788
ENTRYPOINT ["refresh_api"]
