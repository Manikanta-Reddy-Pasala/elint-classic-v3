FROM rust:1.88-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ make cmake libclang-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY poller/Cargo.toml .
COPY poller/src/ src/
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/cdc_poller /usr/local/bin/
ENV RUST_LOG=info
ENTRYPOINT ["cdc_poller"]
