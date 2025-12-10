#!/bin/bash
set -euo pipefail

# Fix timestamps for deterministic builds
export SOURCE_DATE_EPOCH=0

# Set rustflags for deterministic builds
# Note: This overrides .cargo/config.toml, so we include those flags here too
export RUSTFLAGS="-C debuginfo=0 -C target-cpu=generic --remap-path-prefix=$(realpath $(pwd))=/build"

# Prevent network access during build
export CARGO_NET_OFFLINE=true

# Build with explicit target and single-threaded compilation
cargo build --release --locked --target aarch64-apple-darwin -j 1
