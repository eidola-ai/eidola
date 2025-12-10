#!/bin/bash
set -euo pipefail

# Fix timestamps for deterministic builds
export SOURCE_DATE_EPOCH=0

# Set rustflags for deterministic builds
# Note: This overrides .cargo/config.toml, so we include those flags here too
export RUSTFLAGS="-C debuginfo=0 -C target-cpu=generic --remap-path-prefix=$(realpath $(pwd))=/build"

# Prevent network access during build
export CARGO_NET_OFFLINE=true

# Define targets from rust-toolchain.toml
# Update this list when you uncomment targets in rust-toolchain.toml
TARGETS=(
    "x86_64-unknown-linux-musl"
    "aarch64-unknown-linux-musl"
)

# Build for each target
for TARGET in "${TARGETS[@]}"; do
    echo "Building for target: $TARGET"
    cargo build --release --locked --target "$TARGET" -j 1
    echo "✓ Completed: $TARGET"
    echo ""
done

echo "All builds complete!"
echo "Artifacts are in target/<target-triple>/release/"
