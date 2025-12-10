# Deterministic build environment for Eidolons
FROM rust:1.91.1@sha256:867f1d1162913c401378a8504fb17fe2032c760dc316448766f150a130204aad

# Set working directory
WORKDIR /build

# Copy the entire project
# .dockerignore will exclude unnecessary files
COPY . .

# Set environment variables for deterministic builds
ENV SOURCE_DATE_EPOCH=0
ENV CARGO_NET_OFFLINE=true
ENV RUSTFLAGS="-C debuginfo=0 -C target-cpu=generic --remap-path-prefix=/build=/build"

# Build for x86_64 Linux musl target explicitly
# This ensures deterministic builds regardless of host architecture
# Use docker buildx --platform linux/amd64 to ensure this builds correctly on ARM64 hosts
RUN cargo build --release --locked --target x86_64-unknown-linux-musl -j 1

# Artifacts will be in /build/target/x86_64-unknown-linux-musl/release/
