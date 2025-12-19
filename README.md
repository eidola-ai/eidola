# Eidolons

Eidolons consists of two components:
- **Server** (`server/`) - An OpenAI-compatible proxy that routes requests to AI providers (currently Anthropic Claude)
- **Core library** (`core/`) - A Rust library with Swift bindings for iOS/macOS apps

All builds are deterministic and reproducible via Nix.

## Server

The server exposes an OpenAI-compatible `/v1/chat/completions` endpoint that proxies to Anthropic's Claude API, handling format translation and streaming.

### Running with Docker

```bash
# Build the Linux container image
nix build '.#server-oci--aarch64-unknown-linux-musl'  # ARM64
nix build '.#server-oci--x86_64-unknown-linux-musl'   # x86_64

# Load and run
docker load < result
docker run --rm -d -p 8080:8080 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY eidolons-server:latest

# Test
curl http://localhost:8080/health
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"Hello!"}]}'
```

### Running locally

```bash
nix develop
ANTHROPIC_API_KEY=sk-ant-... cargo run -p eidolons-server
```

## Building

This project uses Nix for reproducible builds. [Install Nix](https://nixos.org/download.html) with flakes enabled.

```bash
# Build targets
nix build '.#server'                 # Server binary (native)
nix build '.#server-oci'             # Server OCI image (native, for macOS won't run in Docker)
nix build '.#core-swift-xcframework' # XCFramework for iOS/macOS apps

# Cross-compile for Linux deployment
nix build '.#server--aarch64-unknown-linux-musl'       # Linux ARM64 binary
nix build '.#server-oci--aarch64-unknown-linux-musl'   # Linux ARM64 container

# Run all checks
nix flake check
```

**Development shell:**
```bash
nix develop  # Provides Rust toolchain, cargo-watch, rust-analyzer
```

**Swift bindings:**
```bash
nix run '.#update-core-swift-bindings'    # Update generated Swift bindings (committed)
nix run '.#update-core-swift-xcframework' # Update XCFramework (not committed)
```

Generated Swift bindings are committed to `core/swift/Sources/EidolonsCore/` and verified by CI.
